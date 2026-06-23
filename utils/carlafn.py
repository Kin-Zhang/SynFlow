"""
CARLA utility functions for sensor handling, state monitoring, and geometry operations.
"""
import collections
import math
import weakref

import carla
import numpy as np
import glog as log


# ==============================================================================
# -- Sensors -------------------------------------------------------------------
# ==============================================================================

class CollisionSensor:
    """Collision sensor that tracks collision events on an actor."""
    
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Returns collision history as a dict of {frame: total_intensity}."""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- State Monitors ------------------------------------------------------------
# ==============================================================================

class EgoStateMonitor:
    """Monitors ego vehicle state and handles traffic light forcing."""
    
    def __init__(self, map_):
        self.stop_timer = 0.0
        self.max_stop_time = 15.0
        self.map_ = map_
        self.collision = False

    def check_ego_stop(self, ego_vehicle, settings, world):
        """Monitor ego vehicle stopping and force green light if stuck."""
        v_ego = ego_vehicle.get_velocity()
        speed = math.sqrt(v_ego.x**2 + v_ego.y**2 + v_ego.z**2)
        
        if speed <= 0.1:
            self.stop_timer += settings.fixed_delta_seconds
        else:
            self.stop_timer = 0.0

        if self.stop_timer > 2.0:
            self._try_force_green_light(ego_vehicle, world)

    def _try_force_green_light(self, ego_vehicle, world):
        """Find and force green light if ego is stuck."""
        target_light = None
        
        # Method 1: Check if ego is directly at traffic light
        if ego_vehicle.is_at_traffic_light():
            target_light = ego_vehicle.get_traffic_light()
        
        # Method 2: Search ahead along the lane topology
        if target_light is None:
            target_light = self._search_traffic_light_ahead(ego_vehicle, world)

        if target_light and target_light.get_state() != carla.TrafficLightState.Green:
            log.debug(f"Force Green Light (ID: {target_light.id}, stuck {self.stop_timer:.1f}s)")
            target_light.set_state(carla.TrafficLightState.Green)
            target_light.set_green_time(5.0)
            self.stop_timer = 0.0

    def _search_traffic_light_ahead(self, ego_vehicle, world, check_dist=50.0, step_dist=5.0):
        """Search for traffic lights ahead along the lane."""
        try:
            current_wp = self.map_.get_waypoint(ego_vehicle.get_location())
            dist_traveled = 0.0
            temp_wp = current_wp
            
            while dist_traveled < check_dist:
                landmarks = temp_wp.get_landmarks(distance=step_dist, stop_at_junction=False)
                for lm in landmarks:
                    light_actor = world.get_traffic_light(lm)
                    if light_actor:
                        return light_actor
                
                next_wps = temp_wp.next(step_dist)
                if not next_wps:
                    break
                temp_wp = next_wps[0]
                dist_traveled += step_dist
        except Exception:
            pass
        return None

    def check_collision(self, collision_sensor):
        """Check if a collision has occurred."""
        collision_events = collision_sensor.get_collision_history()
        self.collision = len(collision_events) > 1
        return self.collision


# ==============================================================================
# -- Geometry Functions --------------------------------------------------------
# ==============================================================================

def carla_vec_to_np(vec):
    """Convert carla.Vector3D to numpy array."""
    return np.array([vec.x, vec.y, vec.z])


def carla_transform_to_mat4(transform):
    """Convert carla.Transform to 4x4 transformation matrix."""
    return np.array(transform.get_matrix())


def get_points_in_box_mask(points, bbox_matrix, extent, margin=1.1):
    """
    Check which points are inside a 3D bounding box.
    
    Args:
        points: (N, 3) points in world frame
        bbox_matrix: (4, 4) bounding box transform matrix
        extent: carla.Vector3D (half-size of the box)
        margin: expansion factor for box size
    """
    inv_box_mat = np.linalg.inv(bbox_matrix)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_local = (inv_box_mat @ points_h.T).T[:, :3]
    
    in_x = np.abs(points_local[:, 0]) <= (extent.x * margin)
    in_y = np.abs(points_local[:, 1]) <= (extent.y * margin)
    in_z = np.abs(points_local[:, 2]) <= (extent.z * margin)
    
    return in_x & in_y & in_z


def refine_bbx_w_ins_tags(geo_mask, curr_ins_tags):
    """
    Refine bounding box mask using instance segmentation tags.
    Uses the most frequent instance ID within the geometric mask.
    """
    final_mask = np.zeros_like(geo_mask, dtype=bool)

    if np.any(geo_mask):
        candidate_ids = curr_ins_tags[geo_mask]
        if len(candidate_ids) > 0:
            counts = np.bincount(candidate_ids)
            target_ins_id = np.argmax(counts)
            final_mask = (curr_ins_tags == target_ins_id)
    
    return final_mask