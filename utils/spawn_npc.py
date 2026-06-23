"""
NPC spawning utilities for CARLA simulation.
Provides functions to spawn vehicles and walkers near a given route.
"""
import random

import carla
import glog as log


# ==============================================================================
# -- Helper Functions ----------------------------------------------------------
# ==============================================================================

def get_closest_distance_to_route(location, route_waypoints, sampling_stride=10):
    """
    Calculate minimum distance from a location to the route.
    Uses sampling stride to speed up calculation.
    """
    min_dist = float('inf')
    for i in range(0, len(route_waypoints), sampling_stride):
        dist = location.distance(route_waypoints[i])
        if dist < min_dist:
            min_dist = dist
    return min_dist


def get_curvature(wp, check_dist=10.0):
    """Calculate yaw angle difference (in degrees) over check_dist meters."""
    next_wps = wp.next(check_dist)
    if not next_wps:
        return 0.0
    
    yaw1 = wp.transform.rotation.yaw
    yaw2 = next_wps[0].transform.rotation.yaw
    
    # Handle 0/360 degree wraparound
    diff = abs(yaw1 - yaw2)
    if diff > 180:
        diff = 360 - diff
    return diff


# ==============================================================================
# -- Vehicle Spawning ----------------------------------------------------------
# ==============================================================================

def spawn_npcs(client, world, map_, route_waypoints, num_npcs, tm_port):
    """Spawn NPC vehicles near the route with density control."""
    start_loc = route_waypoints[0]
    start_wp = map_.get_waypoint(start_loc)
    forbidden_road_id = start_wp.road_id
    forbidden_lane_id = start_wp.lane_id

    # P4: pre-filter by bounding box before generating all waypoints.
    # On large maps (Town12) generate_waypoints returns tens of thousands of points;
    # the bbox check is O(1) per point vs the O(route_len) distance check below.
    SPAWN_RADIUS = 80.0
    xs = [wp.x for wp in route_waypoints]
    ys = [wp.y for wp in route_waypoints]
    bbox_min_x = min(xs) - SPAWN_RADIUS
    bbox_max_x = max(xs) + SPAWN_RADIUS
    bbox_min_y = min(ys) - SPAWN_RADIUS
    bbox_max_y = max(ys) + SPAWN_RADIUS

    all_waypoints = [
        wp for wp in map_.generate_waypoints(distance=10.0)
        if bbox_min_x <= wp.transform.location.x <= bbox_max_x
        and bbox_min_y <= wp.transform.location.y <= bbox_max_y
    ]
    random.shuffle(all_waypoints)
    
    valid_spawn_points = []
    lane_density = {}
    MAX_CARS_PER_LANE = 5
    MAX_CURVATURE_DEGREE = 10.0
    same_lane_count = 0

    for wp in all_waypoints:
        # Skip junctions and curved roads
        next_wps = wp.next(10.0)
        if wp.is_junction:
            continue
        if next_wps and (get_curvature(wp) > MAX_CURVATURE_DEGREE or next_wps[0].is_junction):
            continue

        # Limit spawning on ego's starting lane
        if wp.road_id == forbidden_road_id and wp.lane_id == forbidden_lane_id:
            dist = wp.transform.location.distance(start_loc)
            if dist < 15.0 or same_lane_count > MAX_CARS_PER_LANE // 2:
                same_lane_count += 1
                continue

        # Filter by distance to route
        dist = get_closest_distance_to_route(wp.transform.location, route_waypoints, sampling_stride=5)
        if dist > 80.0:
            continue
            
        # Enforce lane density limit
        lane_key = (wp.road_id, wp.lane_id)
        current_count = lane_density.get(lane_key, 0)
        if current_count >= MAX_CARS_PER_LANE:
            continue

        sp_transform = wp.transform
        sp_transform.location.z += 0.5
        valid_spawn_points.append(sp_transform)
        lane_density[lane_key] = current_count + 1
        
        if len(valid_spawn_points) >= num_npcs:
            break

    spawn_points = valid_spawn_points[:num_npcs]
    log.debug(f"Spawning {len(spawn_points)} NPCs within 80m of route...")
    
    # Batch spawn vehicles with autopilot
    batch = []
    bp_lib = world.get_blueprint_library().filter('vehicle.*')
    
    for sp in spawn_points:
        bp = random.choice(bp_lib)
        if bp.has_attribute('driver_id'): 
            bp.set_attribute('driver_id', 'autopilot')
        
        sp.location.z += 0.5
        cmd = carla.command.SpawnActor(bp, sp).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
        )
        batch.append(cmd)
    
    results = client.apply_batch_sync(batch, True)
    return [world.get_actor(r.actor_id) for r in results if not r.error]


# ==============================================================================
# -- Walker Spawning -----------------------------------------------------------
# ==============================================================================

def _spawn_walker_actors(client, world, spawn_points):
    """Spawn walker actors at given spawn points."""
    batch = []
    bp_lib = world.get_blueprint_library()
    walker_bps = bp_lib.filter('walker.pedestrian.*')
    
    for sp in spawn_points:
        bp = random.choice(walker_bps)
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')
        batch.append(carla.command.SpawnActor(bp, sp))

    results = client.apply_batch_sync(batch, True)
    return [{'id': r.actor_id} for r in results if not r.error]


def _spawn_walker_controllers(client, world, walker_ids):
    """Spawn AI controllers for walkers."""
    bp_lib = world.get_blueprint_library()
    controller_bp = bp_lib.find('controller.ai.walker')
    
    batch = []
    for w in walker_ids:
        batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), w['id']))
    
    results = client.apply_batch_sync(batch, True)
    for i, r in enumerate(results):
        if not r.error:
            walker_ids[i]['con'] = r.actor_id
    
    return walker_ids


def _get_walker_and_controller_actors(world, walker_ids):
    """Retrieve walker and controller actors from IDs."""
    all_ids = []
    for w in walker_ids:
        all_ids.append(w['con'])
        all_ids.append(w['id'])
    
    all_actors = world.get_actors(all_ids)
    controllers = [a for a in all_actors if a.type_id.startswith('controller.ai.walker')]
    walkers = [a for a in all_actors if a.type_id.startswith('walker.pedestrian')]
    
    return walkers, controllers


def _initialize_walker_physics(walkers):
    """Enable physics and collisions for walkers. Fix from: https://github.com/carla-simulator/carla/issues/7092"""
    for actor in walkers:
        actor.set_collisions(True)
        actor.set_simulate_physics(True)

def _get_sidewalks_topology_only(map_, route_waypoints, max_sidewalks=200):
    """
    Safe sidewalk search using ONLY topology traversal.
    Avoids geometric projection which can crash CARLA on large maps.
    """
    sidewalk_waypoints = []
    seen_keys = set()
    
    def add_if_new(wp):
        if wp is None:
            return False
        try:
            if wp.lane_type != carla.LaneType.Sidewalk:
                return False
            key = (wp.road_id, wp.lane_id, int(wp.s))
            if key not in seen_keys:
                sidewalk_waypoints.append(wp)
                seen_keys.add(key)
                return True
        except Exception:
            pass
        return False

    # Sample route waypoints (every 10th point)
    sample_indices = range(0, len(route_waypoints), 10)
    
    for i in sample_indices:
        if len(sidewalk_waypoints) >= max_sidewalks:
            break
            
        center_loc = route_waypoints[i]
        
        try:
            driving_wp = map_.get_waypoint(center_loc)
            if driving_wp is None:
                continue
        except Exception:
            continue
        
        # Topology search: traverse lanes left and right
        for get_lane_fn in [lambda w: w.get_right_lane(), lambda w: w.get_left_lane()]:
            curr = driving_wp
            for _ in range(8):  # Search up to 8 lanes deep
                try:
                    curr = get_lane_fn(curr)
                    if curr is None:
                        break
                    if add_if_new(curr):
                        break  # Found sidewalk on this side
                except Exception:
                    break

    return sidewalk_waypoints


def _get_spawn_points_from_navmesh(world, route_waypoints, num_points, max_dis=40):
    """
    Fallback: Get spawn points from navigation mesh near route.
    Safe but may return fewer points on large maps.
    """
    spawn_points = []
    max_attempts = num_points * 30
    attempts = 0
    
    while len(spawn_points) < num_points and attempts < max_attempts:
        attempts += 1
        try:
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            
            dist = get_closest_distance_to_route(loc, route_waypoints, sampling_stride=10)
            if dist <= max_dis:
                spawn_points.append(carla.Transform(loc))
        except Exception:
            continue
    
    return spawn_points


def spawn_walkers(client, world, num_walkers, route_waypoints, max_dis=40):
    """
    Spawn walkers using a hybrid approach with crash protection.
    
    Strategy:
    1. Try safe topology-based sidewalk search first
    2. Fall back to navmesh sampling if needed
    3. Combine both methods if neither provides enough points
    """

    map_ = world.get_map()
    spawn_points = []
    sidewalk_wps = []
    
    # Method 1: Safe topology-based sidewalk search
    try:
        sidewalk_wps = _get_sidewalks_topology_only(map_, route_waypoints, max_sidewalks=num_walkers * 2)
        log.debug(f"Found {len(sidewalk_wps)} sidewalk waypoints via topology")
        
        if sidewalk_wps:
            random.shuffle(sidewalk_wps)
            for wp in sidewalk_wps[:num_walkers]:
                try:
                    trans = wp.transform
                    trans.location.z += 1.0
                    spawn_points.append(trans)
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"Topology sidewalk search failed: {e}")
    
    # Method 2: If not enough points, supplement with navmesh sampling
    remaining = num_walkers - len(spawn_points)
    if remaining > 0:
        log.debug(f"Need {remaining} more spawn points, trying navmesh sampling...")
        navmesh_points = _get_spawn_points_from_navmesh(
            world, route_waypoints, remaining, max_dis
        )
        spawn_points.extend(navmesh_points)
        log.debug(f"Got {len(navmesh_points)} points from navmesh")
    
    if not spawn_points:
        log.warning("No walker spawn points found!")
        return [], []
    
    log.debug(f"Spawning {len(spawn_points)} walkers...")
    
    # Spawn walkers
    walker_ids = _spawn_walker_actors(client, world, spawn_points)
    if not walker_ids:
        log.warning("Failed to spawn any walkers")
        return [], []
    
    walker_ids = _spawn_walker_controllers(client, world, walker_ids)
    walkers, controllers = _get_walker_and_controller_actors(world, walker_ids)

    # Start walking with safe target selection
    for con in controllers:
        try:
            con.start()
            
            # Choose target location
            target_loc = None
            
            # Try sidewalk-based target first
            if sidewalk_wps:
                walker_actor = con.parent
                if walker_actor:
                    start_loc = walker_actor.get_location()
                    for _ in range(10):
                        candidate_wp = random.choice(sidewalk_wps)
                        try:
                            cand_loc = candidate_wp.transform.location
                            dist = start_loc.distance(cand_loc)
                            if 20.0 < dist < 80.0:
                                target_loc = cand_loc
                                break
                        except Exception:
                            continue
            
            # Fallback to random navmesh location
            if target_loc is None:
                target_loc = world.get_random_location_from_navigation()
            
            if target_loc:
                con.go_to_location(target_loc)
                con.set_max_speed(1.4 + random.random())
        except Exception:
            continue
    
    _initialize_walker_physics(walkers)
    return walkers, controllers