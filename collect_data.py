"""
# Created: 2025-12-27 21:32
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Google AI Studios
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: collect data from CARLA simulator for scene flow estimation.
#
"""
import faulthandler
import gc
import glob
import os
import queue
import random
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import carla
import glog as log
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

try:
    from agents.navigation.behavior_agent import BehaviorAgent
    from agents.navigation.local_planner import RoadOption
except ImportError as e:
    print(f"Error: {e}. Please ensure CARLA PythonAPI is in your PYTHONPATH.")

from utils import CARLA_SEMANTIC_TAG_TO_CATEGORY, CATEGORY_TO_INDEX
from utils.carlafn import (
    CollisionSensor, EgoStateMonitor, carla_transform_to_mat4,
    get_points_in_box_mask, refine_bbx_w_ins_tags
)
from utils.spawn_npc import spawn_npcs, spawn_walkers

faulthandler.enable()


# ==============================================================================
# -- Data Saver ----------------------------------------------------------------
# ==============================================================================

class H5DataSaver:
    """HDF5 data saver for scene flow data."""
    
    # Semantic tags considered as ground
    GROUND_TAGS = {1, 2, 10, 24, 25}
    MIN_FRAMES_THRESHOLD = 120
    
    def __init__(self, output_dir, file_name, verbose=False):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.output_path / file_name
        self.h5_file = h5py.File(self.file_path, 'w')
        
        self.frame_count = 0
        self.total_frames = 0
        self.scene_count = 1
        self.verbose = verbose
        self.prev_data = None  # Buffer for Frame T data

    def reset(self):
        """Reset for next scene."""
        self.total_frames += self.frame_count
        self.close()

        file_name = self.file_path.stem[:-2] + f"{self.scene_count:02d}.h5"
        self.scene_count += 1
        self.file_path = self.output_path / file_name
        self.h5_file = h5py.File(self.file_path, 'w')
        self.frame_count = 0
        self.prev_data = None
    
    def delete_last_scene(self, is_long_stop, is_collision, img_dir=None):
        """Delete incomplete scene due to errors."""
        self.close()
        reason = []
        if is_long_stop:
            reason.append('long stop')
        if is_collision:
            reason.append('2x collision')
        reason_str = ' and '.join(reason)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            log.warn(f"Due to {reason_str}, deleted incomplete scene: {self.file_path}.")
        if img_dir and os.path.isdir(img_dir):
            shutil.rmtree(img_dir)
            log.warn(f"Due to {reason_str}, deleted image dir: {img_dir}.")

    def close(self):
        """Close HDF5 file and cleanup."""
        if not self.h5_file:
            return
        
        if self.frame_count >= self.MIN_FRAMES_THRESHOLD:
            self.h5_file.close()
            if self.total_frames == 0:
                self.total_frames += self.frame_count
            log.debug(f"HDF5 closed. Total frames: {self.total_frames} in {self.file_path}.")
        else:
            self.h5_file.close()
            os.remove(self.file_path)
            log.warn(f"Incomplete HDF5 deleted: {self.file_path} ({self.frame_count} frames).")

    def _convert_to_rhs(self, xyz, flow, pose):
        """Convert from CARLA LHS to standard RHS by flipping Y axis.
        xyz and flow are mutated in-place (callers pass freshly-sliced arrays).
        pose is copied since ego_mat is reused by the caller."""
        xyz[:, 1] *= -1
        flow[:, 1] *= -1
        pose_rhs = pose.copy()
        pose_rhs[0, 1] *= -1
        pose_rhs[1, 0] *= -1
        pose_rhs[1, 2] *= -1
        pose_rhs[2, 1] *= -1
        pose_rhs[1, 3] *= -1
        return xyz, flow, pose_rhs

    def _save_frame(self, timestamp, ego_mat, xyz_sensor, ground_mask, 
                    flow_vectors, valid_mask, category_indices, instance_ids):
        """Save a single frame to HDF5."""
        # Skip the first frame of the very first scene
        if self.frame_count < 1 and self.total_frames == 0:
            self.frame_count += 1
            return
            
        ts_str = str(int(timestamp * 1e6))
        grp = self.h5_file.create_group(ts_str)
        
        # LHS (CARLA) to RHS (Standard) conversion
        xyz_rhs, flow_rhs, pose_rhs = self._convert_to_rhs(xyz_sensor, flow_vectors, ego_mat)

        grp.create_dataset('lidar', data=xyz_rhs.astype(np.float32))
        grp.create_dataset('pose', data=pose_rhs.astype(np.float64))
        grp.create_dataset('ground_mask', data=ground_mask)
        grp.create_dataset('flow', data=flow_rhs.astype(np.float32))
        grp.create_dataset('flow_is_valid', data=valid_mask)
        grp.create_dataset('flow_category_indices', data=category_indices)
        grp.create_dataset('flow_instance_id', data=instance_ids)
        
        self.frame_count += 1
        if self.frame_count % 10 == 0 and self.verbose:
            log.debug(f"Frame {self.frame_count} | Ground: {np.sum(ground_mask)} | Dynamic: {np.sum(instance_ids != -1)}")

    def _compute_and_save(self, prev, ego_mat_next, npc_transforms_next):
        """Compute geometric flow from T to T+1 and save frame T."""
        xyz_sensor_t = prev['xyz_sensor']
        ego_mat_t = prev['ego_mat']
        lidar_mat = prev['lidar_mat']
        xyz_world_t = prev['xyz_world']
        
        # Background flow: Flow = P_sensor_next - P_sensor_curr
        inv_lidar_mat = np.linalg.inv(lidar_mat)
        inv_ego_next = np.linalg.inv(ego_mat_next)
        # P1: precompute combined world→sensor(T+1) transform; reused for NPC flow below
        M_bg = inv_lidar_mat @ inv_ego_next

        N = len(xyz_world_t)
        xyz_world_h = np.hstack((xyz_world_t, np.ones((N, 1))))

        p_sensor_next = (M_bg @ xyz_world_h.T).T[:, :3]
        flow_vectors = p_sensor_next - xyz_sensor_t
        
        # Initialize labels: -1 = background (training code uses 0-indexed instances)
        instance_ids = np.full(N, -1, dtype=np.int16)
        category_indices = np.zeros(N, dtype=np.uint8)
        
        # Check ego self-hit
        ego_bbox = prev['ego_bbox']
        ego_bbox_local_mat = carla_transform_to_mat4(
            carla.Transform(ego_bbox.location, ego_bbox.rotation)
        )
        ego_bbox_world_mat = ego_mat_t @ ego_bbox_local_mat
        mask_ego = get_points_in_box_mask(xyz_world_t, ego_bbox_world_mat, ego_bbox.extent)

        # Process NPC flow
        for npc_data in prev['npc_list_snapshot']:
            nid = npc_data['id']
            if nid not in npc_transforms_next:
                continue 
            
            npc_mat_t = npc_data['transform']
            npc_mat_next = npc_transforms_next[nid]
            bbox_world_mat = npc_mat_t @ npc_data['bbox_local_pose']
   
            geo_mask = get_points_in_box_mask(xyz_world_t, bbox_world_mat, npc_data['bbox_extent'])
            mask = refine_bbx_w_ins_tags(geo_mask, prev['ins_tags'])

            if not np.any(mask):
                continue
                
            instance_ids[mask] = nid % 32000 + 1  # Avoid 0 id
            
            # Determine category from semantic tags
            obj_tags = prev['sem_tags'][mask]
            if len(obj_tags) > 0:
                mode_tag = np.bincount(obj_tags).argmax()
                category_name = CARLA_SEMANTIC_TAG_TO_CATEGORY.get(mode_tag, "NONE")
                if category_name == "NONE":
                    instance_ids[mask] = -1
                    continue
                category_indices[mask] = CATEGORY_TO_INDEX.get(category_name, 0)

            # P2: merge full chain world_t→npc_local→world_next→sensor_next into one matmul
            inv_npc_t = np.linalg.inv(npc_mat_t)
            M_npc = M_bg @ npc_mat_next @ inv_npc_t
            points_sensor_next = (M_npc @ xyz_world_h[mask].T).T[:, :3]
            flow_vectors[mask] = points_sensor_next - xyz_sensor_t[mask]
        
        # Filter out ego self-hit points
        keep_mask = ~mask_ego
        self._save_frame(
            prev['timestamp'], ego_mat_t, xyz_sensor_t[keep_mask], 
            prev['ground_mask'][keep_mask], flow_vectors[keep_mask], 
            np.ones(np.sum(keep_mask), dtype=bool), 
            category_indices[keep_mask], instance_ids[keep_mask]
        )

    def process_frame(self, timestamp, ego_vehicle, sem_lidar_raw,
                      lidar_to_ego_transform, npc_list):
        """Process a single frame of sensor data."""
        # Parse LiDAR data
        dtype_lidar = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('cos', np.float32), ('idx', np.uint32), ('tag', np.uint32)
        ])
        data_np = np.frombuffer(sem_lidar_raw.raw_data, dtype=dtype_lidar)
        xyz_sensor = np.stack((data_np['x'], data_np['y'], data_np['z']), axis=1)
        ins_tags = data_np['idx']
        sem_tags = data_np['tag']
        ground_mask = np.isin(sem_tags, list(self.GROUND_TAGS))

        # Get transforms
        ego_mat = carla_transform_to_mat4(ego_vehicle.get_transform())
        lidar_mat = carla_transform_to_mat4(lidar_to_ego_transform)

        npc_transforms_curr = {
            npc.id: carla_transform_to_mat4(npc.get_transform())
            for npc in npc_list if npc.is_alive
        }

        # Compute flow for previous frame if exists
        if self.prev_data is not None:
            self._compute_and_save(self.prev_data, ego_mat, npc_transforms_curr)

        # Transform points to world frame
        xyz_ego = (lidar_mat[:3, :3] @ xyz_sensor.T).T + lidar_mat[:3, 3]
        xyz_world = (ego_mat[:3, :3] @ xyz_ego.T).T + ego_mat[:3, 3]

        # Snapshot NPC states
        npc_snapshots = [
            {
                'id': npc.id,
                'transform': carla_transform_to_mat4(npc.get_transform()),
                'bbox_extent': npc.bounding_box.extent,
                'bbox_local_pose': carla_transform_to_mat4(
                    carla.Transform(npc.bounding_box.location, npc.bounding_box.rotation)
                )
            }
            for npc in npc_list if npc.is_alive
        ]

        # Update buffer
        self.prev_data = {
            'timestamp': timestamp,
            'ego_mat': ego_mat,
            'lidar_mat': lidar_mat,
            'ego_bbox': ego_vehicle.bounding_box,
            'xyz_sensor': xyz_sensor,
            'xyz_world': xyz_world,
            'ins_tags': ins_tags,
            'sem_tags': sem_tags,
            'ground_mask': ground_mask,
            'npc_list_snapshot': npc_snapshots
        }


# ==============================================================================
# -- Route & Sensor Setup ------------------------------------------------------
# ==============================================================================

class RouteParser:
    """XML route file parser."""
    
    @staticmethod
    def parse_route(xml_path, route_id):
        """Parse route waypoints from XML file."""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Route file not found: {xml_path}")
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        route_elem = None
        for r in root.findall('route'):
            if int(r.get('id')) == int(route_id):
                route_elem = r
                break
                
        if route_elem is None:
            return None, None
            
        town_name = route_elem.get('town')
        waypoints = [
            carla.Location(
                x=float(pos.get('x')), 
                y=float(pos.get('y')), 
                z=float(pos.get('z')) + 0.5
            )
            for pos in route_elem.find('waypoints').findall('position')
        ]
        return town_name, waypoints

    @staticmethod
    def get_num_routes(xml_path):
        """Get total number of routes in XML file."""
        tree = ET.parse(xml_path)
        return len(tree.getroot().findall('route'))


class SensorManager:
    """Manages sensor spawning and data synchronization."""
    
    def __init__(self, world, vehicle, cfg, save_images=False):
        self.world = world
        self.vehicle = vehicle
        self.queue = queue.Queue()
        self.sensor_actors = []
        self.save_images = save_images
        self.img_dir = None
        self.img_count = 0
        
        lidar_cfg = cfg.sensors.lidar_semantic
        self.lidar_transform = carla.Transform(carla.Location(z=lidar_cfg.z_offset))
        self._spawn_sensor(lidar_cfg, 'lidar_semantic', self.lidar_transform)
        self.collision_sensor = CollisionSensor(self.vehicle)

        # Spawn third-person RGB camera for cover images
        if self.save_images:
            self._spawn_rgb_camera()

    def _spawn_sensor(self, sensor_cfg, tag, transform):
        """Spawn a sensor and attach to vehicle."""
        bp = self.world.get_blueprint_library().find(sensor_cfg.type)
        for key, value in sensor_cfg.items():
            if key not in ['type', 'z_offset']:
                bp.set_attribute(str(key), str(value))
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        sensor.listen(lambda data: self.queue.put((tag, data)))
        self.sensor_actors.append(sensor)

    def _spawn_rgb_camera(self):
        """Spawn a third-person RGB camera attached to the ego vehicle."""
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', '1920')
        bp.set_attribute('image_size_y', '1080')
        bp.set_attribute('fov', '100')

        # Third-person view: behind and above the vehicle (like manual_control.py)
        bound_x = 0.5 + self.vehicle.bounding_box.extent.x
        bound_z = 0.5 + self.vehicle.bounding_box.extent.z
        camera_transform = carla.Transform(
            carla.Location(x=-2.0 * bound_x, z=3.0 * bound_z),
            carla.Rotation(pitch=10.0)
        )
        camera = self.world.spawn_actor(
            bp, camera_transform,
            attach_to=self.vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost
        )
        camera.listen(lambda data: self.queue.put(('camera_rgb', data)))
        self.sensor_actors.append(camera)

    def setup_image_dir(self, output_dir, scene_name):
        """Setup directory for saving camera images."""
        self.img_dir = Path(output_dir) / 'scene-img' / scene_name
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.img_count = 0

    def save_image(self, image_data):
        """Save a camera RGB image to disk."""
        if not self.save_images or self.img_dir is None:
            return
        image_data.save_to_disk(str(self.img_dir / f'{self.img_count:06d}'))
        self.img_count += 1

    def get_synced_data(self, frame_id, timeout=2.0):
        """Get synchronized sensor data for a specific frame."""
        result = {}
        expected = len(self.sensor_actors)
        start_time = time.time()
        while time.time() - start_time < timeout and len(result) < expected:
            try:
                tag, data = self.queue.get(timeout=0.1)
                if data.frame == frame_id:
                    result[tag] = data
            except queue.Empty:
                continue
        # Drain any stale items left in the queue so they don't bleed into the next frame.
        # In synchronous mode sensors produce exactly one frame per tick, so anything
        # remaining here is either a duplicate or a leftover from a previous timeout.
        while True:
            try:
                tag, data = self.queue.get_nowait()
                if data.frame == frame_id and tag not in result:
                    result[tag] = data
            except queue.Empty:
                break
        return result

    def destroy(self):
        """Cleanup all sensors."""
        for s in self.sensor_actors:
            if s and s.is_alive:
                s.stop()
                s.destroy()
        self.sensor_actors.clear()

        if self.collision_sensor:
            self.collision_sensor.sensor.stop()
            self.collision_sensor.sensor.destroy()
            self.collision_sensor = None


# ==============================================================================
# -- Main Functions ------------------------------------------------------------
# ==============================================================================

def _adjust_npc_counts(town_id, num_vehicles, num_walkers):
    """Adjust NPC counts based on town characteristics."""
    if town_id in ['04', '06']:
        return int(num_vehicles * 2.0), num_walkers
    elif town_id in ['01', '02']:
        return int(num_vehicles * 0.7), int(num_walkers * 0.5)
    return num_vehicles, num_walkers


def _setup_world(client, cfg, town_name, town_id, run_cnt):
    """Setup CARLA world with proper settings."""
    world = client.get_world()
    current_map = world.get_map().name.split('/')[-1]
    
    # Determine if we need to load/reload the world
    need_load = False
    
    # Always load if map doesn't match
    if current_map != town_name:
        log.info(f"Loading {town_name} (current: {current_map})...")
        need_load = True
    # Periodically reload to clear memory (every 5 routes)
    elif run_cnt > 0 and run_cnt % 5 == 0:
        log.debug(f"Reloading world to clear memory... (run count: {run_cnt})")
        need_load = True
    
    if need_load:
        world = client.load_world(town_name)
        # Town12 is large and needs extra time to load
        if town_id in ['12', '13']:
            time.sleep(10)
        else:
            time.sleep(3)
    else:
        # Even without reload, give a small delay for stability
        time.sleep(1)

    # Configure synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.world_settings.fixed_delta_seconds
    world.apply_settings(settings)
    
    return world, settings


def _setup_traffic_manager(client, cfg):
    """Setup traffic manager with proper settings."""
    tm = client.get_trafficmanager(cfg.simulation.port + 8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(random.randint(0, 10000))
    tm.global_percentage_speed_difference(cfg.npcs.speed_factor)
    return tm


def main_per_route(cfg, town_name, waypoints, route_id, run_cnt):
    """Run data collection for a single route."""
    town_id = ''.join(filter(str.isdigit, town_name))
    client = carla.Client(cfg.simulation.host, cfg.simulation.port)
    
    # Town12 needs longer timeout to avoid segfaults
    if town_id == '12':
        cfg.simulation.timeout = 30.0
    client.set_timeout(cfg.simulation.timeout)
            
    actor_list, sensor_manager, saver = [], None, None
    route_fail = False
    
    try:
        world, settings = _setup_world(client, cfg, town_name, town_id, run_cnt)
        tm = _setup_traffic_manager(client, cfg)
        map_ = world.get_map()

        # Create data saver
        saver = H5DataSaver(
            cfg.simulation.data_output,
            f"scene-{town_id}{cfg.sensors.lidar_semantic.channels}{route_id:04d}00.h5"
        )

        # Optional CARLA recording
        if cfg.simulation.get('record_carla_log', False):
            log_name = f"{town_name.lower()}_scene{route_id:04d}_{cfg.sensors.lidar_semantic.channels}.log"
            log_path = str(Path(cfg.simulation.data_output) / 'carla_log' / log_name)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Starting CARLA recorder: {log_path}")
            client.start_recorder(log_path, True)

        # Spawn ego vehicle
        start_tf = map_.get_waypoint(waypoints[0]).transform
        start_tf.location.z += 0.5
        
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego_vehicle = world.spawn_actor(ego_bp, start_tf)
        actor_list.append(ego_vehicle)
        world.get_spectator().set_transform(start_tf)

        # Setup sensors
        save_images = cfg.simulation.get('save_images', False)
        sensor_manager = SensorManager(world, ego_vehicle, cfg, save_images=save_images)
        if save_images:
            sensor_manager.setup_image_dir(
                cfg.simulation.data_output,
                f"scene-{town_id}{cfg.sensors.lidar_semantic.channels}{route_id:04d}"
            )
        
        # Adjust NPC counts per town
        num_vehicles, num_walkers = _adjust_npc_counts(
            town_id, cfg.npcs.num_vehicles, cfg.npcs.num_walkers
        )
        
        # Extend frames for small LiDAR channels
        if cfg.sensors.lidar_semantic.channels < 64:
            cfg.simulation.max_frames_per_scene = 302

        # Spawn NPCs
        npc_vehicles = spawn_npcs(client, world, map_, waypoints, int(num_vehicles), tm.get_port())
        actor_list.extend(npc_vehicles)

        npc_walkers, walker_controllers = spawn_walkers(client, world, int(num_walkers), waypoints)
        actor_list.extend(npc_walkers)
        actor_list.extend(walker_controllers)
        
        all_npcs = npc_vehicles + npc_walkers
        
        # Configure traffic behavior
        for npc in all_npcs:
            tm.ignore_signs_percentage(npc, 80.0) 
        tm.ignore_signs_percentage(ego_vehicle, 100.0)

        # Setup navigation agent
        agent = BehaviorAgent(ego_vehicle)
        
        # Find start index (skip waypoints too close to start)
        start_loc = waypoints[0]
        start_index = next(
            (i for i, wp in enumerate(waypoints[1:], 1) if start_loc.distance(wp) > 5.0),
            1
        )
        
        # Convert waypoints to plan, filtering out any invalid ones
        clean_plan = []
        for w in waypoints[start_index:]:
            wp = map_.get_waypoint(w)
            if wp is not None:
                clean_plan.append((wp, RoadOption.LANEFOLLOW))
        
        if len(clean_plan) == 0:
            log.error(f"No valid waypoints found for route {route_id}! Map: {map_.name}")
            route_fail = True
            raise RuntimeError("No valid waypoints")
        
        log.debug(f"Route plan: {len(clean_plan)} waypoints (from {len(waypoints)} total)")
        agent.set_global_plan(clean_plan)
        
        ego_monitor = EgoStateMonitor(map_=map_)
        
        # Main simulation loop
        while True:
            world.tick()
            
            control = agent.run_step()
            control.max_throttle = 0.8
            ego_vehicle.apply_control(control)

            if agent.done():
                log.info(f"Route {route_id} completed successfully (Total frames: {saver.total_frames})")
                break
            
            if saver.frame_count > cfg.simulation.max_frames_per_scene:
                saver.reset()

            # Check for failures
            is_long_stop = ego_monitor.stop_timer > ego_monitor.max_stop_time
            is_collision = ego_monitor.check_collision(sensor_manager.collision_sensor)
            if is_long_stop or is_collision:
                saver.delete_last_scene(is_long_stop, is_collision, img_dir=sensor_manager.img_dir)
                route_fail = True
                break

            ego_monitor.check_ego_stop(ego_vehicle, settings, world)

            # Process sensor data
            snapshot = world.get_snapshot()
            data = sensor_manager.get_synced_data(snapshot.frame)
            
            if 'lidar_semantic' in data:
                ts = snapshot.timestamp.elapsed_seconds + 1.6e6
                alive_npcs = [n for n in all_npcs if n.is_alive]
                saver.process_frame(
                    ts, ego_vehicle, data['lidar_semantic'],
                    sensor_manager.lidar_transform, alive_npcs
                )
                
                # Update spectator view
                tf = ego_vehicle.get_transform()
                world.get_spectator().set_transform(
                    carla.Transform(tf.location + carla.Location(z=35), carla.Rotation(pitch=-90))
                )

            # Save third-person camera image
            if 'camera_rgb' in data:
                sensor_manager.save_image(data['camera_rgb'])

    except KeyboardInterrupt: 
        log.info("\nStopped by user.")
    except Exception:
        traceback.print_exc()
    finally:
        # Use timeout for all cleanup calls to prevent blocking on unresponsive CARLA
        cleanup_timeout = 15.0  # seconds
        
        def _run_with_timeout(func, timeout, name):
            """Run a function with timeout, return True if completed, False if timed out."""
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func)
                    future.result(timeout=timeout)
                return True
            except FuturesTimeoutError:
                log.warn(f"Cleanup '{name}' timed out after {timeout}s - skipping")
                return False
            except Exception as e:
                log.warn(f"Cleanup '{name}' failed: {e}")
                return False
        
        if saver: 
            saver.close()
        
        if sensor_manager:
            _run_with_timeout(sensor_manager.destroy, cleanup_timeout, "sensor_manager.destroy")
        
        if actor_list:
            def destroy_actors():
                client.apply_batch_sync([carla.command.DestroyActor(x) for x in actor_list if x.is_alive])
            _run_with_timeout(destroy_actors, cleanup_timeout, "destroy_actors")
        
        if cfg.simulation.get('record_carla_log', False):
            log.debug("Stopping CARLA recorder.")
            _run_with_timeout(client.stop_recorder, cleanup_timeout, "stop_recorder")
        
        if saver and saver.total_frames <= 200:
            route_fail = True
            log.warn(f"Only {saver.total_frames} frames collected for route {route_id}. Marking as failed.")

        # Ensure world ticks complete (with timeout protection)
        if 'world' in locals():
            def finalize_world():
                for _ in range(3):
                    world.tick()
                s = world.get_settings()
                s.synchronous_mode = False
                world.apply_settings(s)
            _run_with_timeout(finalize_world, cleanup_timeout * 2, "finalize_world")
        
        gc.collect()
        return route_fail


def save_load_route_fail(town_id, route_id, save=False, max_fails=5):
    """Track route failures to skip problematic routes."""
    fail_file = f"assets/logs/fail-{town_id}.txt"
    route_failure_count = {}
    
    # Load existing failure counts
    if os.path.exists(fail_file):
        with open(fail_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    rid, count = line.split(':')
                    route_failure_count[int(rid)] = int(count)
    else:
        os.makedirs(os.path.dirname(fail_file), exist_ok=True)
    
    if save:
        route_failure_count[route_id] = route_failure_count.get(route_id, 0) + 1
        with open(fail_file, 'w') as f:
            for rid, count in sorted(route_failure_count.items()):
                f.write(f"{rid}:{count}\n")
    else:
        return route_failure_count.get(route_id, 0) >= max_fails
    
    return False


@hydra.main(version_base=None, config_path="conf", config_name="collect")
def main(cfg: DictConfig):
    """Main entry point for data collection."""
    log.setLevel(cfg.world_settings.logging_level)
    
    # Normalize route_id to list
    if isinstance(cfg.simulation.route_id, ListConfig):
        route_ids = list(cfg.simulation.route_id)
    elif cfg.simulation.route_id > 0:
        route_ids = [cfg.simulation.route_id]
    else:
        num_routes = RouteParser.get_num_routes(cfg.simulation.route_file)
        route_ids = list(range(num_routes))
    if 'start_id' in cfg.simulation and cfg.simulation.start_id > 0:
        route_ids = route_ids[(cfg.simulation.start_id-1):]
    # as we saved the same town inside the route file, we can just parse once
    town_name, _ = RouteParser.parse_route(cfg.simulation.route_file, 0)
    town_id = ''.join(filter(str.isdigit, town_name))
    for cnt, route_id in enumerate(route_ids):
        route_id = int(route_id)
        
        # Skip routes that have failed too many times
        if save_load_route_fail(town_id, route_id, save=False, max_fails=5):
            log.error(f"Route ID {route_id} has failed more than 5 times. Skipping...")
            continue

        # Skip already completed routes
        file_pattern = f"scene-{town_id}{cfg.sensors.lidar_semantic.channels}{route_id:04d}00.h5"
        if glob.glob(os.path.join(cfg.simulation.data_output, file_pattern)):
            log.debug(f"Route ID {route_id} already finished. Skipping...")
            continue

        town_name, waypoints = RouteParser.parse_route(cfg.simulation.route_file, route_id)
        if town_name is None or waypoints is None:
            log.error(f"Route ID {route_id} not found in {cfg.simulation.route_file}. Skipping...")
            continue
            
        log.info(f"Processing Route ID: {route_id} / Total: {len(route_ids)} in {town_name}.")

        if main_per_route(cfg.copy(), town_name, waypoints, route_id, cnt):
            save_load_route_fail(town_id, route_id, save=True)

    log.info("Data collection completed for all routes.")


if __name__ == '__main__':
    main()