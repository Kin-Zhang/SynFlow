#!/usr/bin/env python3
"""
# Created: 2025-12-26 18:42
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Google AI Studios
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: generate route for different towns using spatial grid logic (no overlap).
# 
"""

import sys, glob, os
import random
import pickle
import xml.etree.ElementTree as ET
import xml.dom.minidom
from pathlib import Path

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ )))
sys.path.append(BASE_DIR)

try:
    import carla
except ImportError as e:
    print("CARLA module not found. Make sure pip install carla.")

import fire
import matplotlib.pyplot as plt
import numpy as np

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as e:
    print("WARNING: 'agents' module not found. Details: ", e)

class TopologyCoverage:
    """
    Manages coverage using Road ID + Lane ID + S-coordinate logic.
    This strictly distinguishes between parallel lanes and driving directions.
    """
    def __init__(self, resolution=2.0):
        self.resolution = resolution # 每隔2米切分一个段
        # Stores tuples: (road_id, section_id, lane_id, s_bucket_index)
        self.covered_segments = set() 

    def _get_key(self, wp):
        # 核心逻辑：组合 ID + 离散化的距离(s)
        # road_id: 道路唯一ID
        # section_id: 同一条路不同段（车道数变化时section会变）
        # lane_id: 区分左右车道及方向
        # s: 沿道路的纵向距离
        s_idx = int(wp.s // self.resolution)
        return (wp.road_id, wp.section_id, wp.lane_id, s_idx)

    def is_covered(self, wp):
        return self._get_key(wp) in self.covered_segments

    def mark_route(self, waypoints):
        """Marks all segments along the route as covered."""
        for wp in waypoints:
            self.covered_segments.add(self._get_key(wp))

    def calculate_new_coverage(self, waypoints):
        """
        Returns the approx distance (in meters) of the route that traverses 
        segments NOT yet covered.
        """
        if len(waypoints) < 2:
            return 0.0
            
        new_meters = 0.0
        # 遍历路径上的每一个点
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i+1]
            
            # 如果这两个点所在的段（Segment）没有被覆盖过，则计入长度
            # 为了更严格，只要起点或终点任意一个没覆盖，就算这段是新的
            if not self.is_covered(wp1) or not self.is_covered(wp2):
                # 计算两点间欧氏距离
                dist = wp1.transform.location.distance(wp2.transform.location)
                new_meters += dist
                
        return new_meters


class AutoRouteTools:
    def __init__(self, filename="routes.xml"):
        self.filename = Path(BASE_DIR) / 'assets' / 'data' / filename

    def _lateral_shift(self, transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    def _extract_map_data(self, carla_map):
        """Extracts background map lines."""
        print("Extracting map geometry...")
        precision = 1.0 
        topology = carla_map.get_topology()
        topology = [x[0] for x in topology]
        
        lines = []
        for waypoint in topology:
            waypoints = [waypoint]
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break

            road_left = [self._lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
            road_right = [self._lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

            if len(road_left) > 1:
                lines.append(([p.x for p in road_left], [-p.y for p in road_left]))
            if len(road_right) > 1:
                lines.append(([p.x for p in road_right], [-p.y for p in road_right]))
        return lines

    def _find_endpoint_topology(self, start_wp, target_dist):
        """
        Walk along road topology to find endpoint, returning the path taken.
        This avoids GlobalRoutePlanner's shortest-path which can create weird detours.
        """
        current_wp = start_wp
        traveled = 0.0
        step_size = 4.0
        path_wps = [start_wp]  # Record the path we actually walk
        
        while traveled < target_dist:
            next_wps = current_wp.next(step_size)
            if not next_wps:
                break
            # Prefer staying on the same road/lane if possible
            same_road = [wp for wp in next_wps if wp.road_id == current_wp.road_id]
            if same_road:
                current_wp = same_road[0]
            else:
                current_wp = random.choice(next_wps)
            path_wps.append(current_wp)
            traveled += step_size
        
        if traveled < target_dist * 0.5:  # Path too short
            return None, []
        return current_wp, path_wps

    def generate(self, 
                 map="Town03", 
                 min_len=150.0, 
                 max_len=250.0,
                 
                 sampling_dist=10.0,   # Very dense sampling (every 10m)
                 resolution=2.0,        # 10m grid resolution
                 min_new_meters=20.0,  # A route must add at least 20m of NEW coverage

                 # default carla connection params
                 host="localhost", 
                 port=2000):
        """
        Generates routes using Spatial Grid Logic.
        Guarantees no gaps by forcing routes where the grid is empty.
        """
        target_map = map 
        print(f"Connecting to CARLA at {host}:{port}...")
        client = carla.Client(host, port)
        client.set_timeout(60.0)

        world = client.get_world()
        if world.get_map().name.split('/')[-1] != target_map:
            print(f"Loading map: {target_map}")
            world = client.load_world(target_map)

        carla_map = world.get_map()
        # Note: We no longer use GlobalRoutePlanner as it can create weird detours
        # Instead we walk along the road topology directly

        # 1. Initialize Spatial Grid
        spatial_grid = TopologyCoverage(resolution=resolution) # 2m resolution

        # 2. Dense Sampling (Candidates)
        print(f"Generating dense candidates every {sampling_dist}m...")
        candidate_wps = carla_map.generate_waypoints(distance=sampling_dist)
        random.shuffle(candidate_wps) # Shuffle to spread initial routes randomly
        
        generated_routes = []
        stats = {'checked': 0, 'accepted': 0, 'rejected_coverage': 0, 'rejected_invalid': 0}
        
        print(f"Processing {len(candidate_wps)} candidates. Strategy: Must add >{min_new_meters}m unique path.")

        # Multi-pass approach usually isn't needed if sampling is dense enough,
        # but the logic ensures we check everyone.
        for start_wp in candidate_wps:
            if start_wp.is_junction:
                continue
            stats['checked'] += 1
            sys.stdout.write(f"\rChecked: {stats['checked']} | Routes: {len(generated_routes)} | Rej(Cov): {stats['rejected_coverage']}")
            sys.stdout.flush()

            # Optimization: If the start point itself is deep in covered territory,
            # we might skip. But to be safe (for gaps), we proceed to trace.
            
            # A. Plan Route - walk along topology directly instead of using trace_route
            target_distance = random.uniform(min_len, max_len)
            end_wp, path_wps = self._find_endpoint_topology(start_wp, target_distance)
            
            if not end_wp or len(path_wps) < 5:
                stats['rejected_invalid'] += 1
                continue

            # Extract Location objects for saving to xml
            path_locs = [wp.transform.location for wp in path_wps]
            
            # B. CHECK COVERAGE
            unique_length = spatial_grid.calculate_new_coverage(path_wps)

            # C. DECISION
            # Even if the route overlaps 80%, if it provides 25m of new road (e.g. a gap), we take it.
            if unique_length < min_new_meters:
                stats['rejected_coverage'] += 1
                continue
            
            # Accept Route
            generated_routes.append({
                'id': len(generated_routes),
                'waypoints': path_locs
            })
            
            # Mark cells as covered
            spatial_grid.mark_route(path_wps)

        print(f"\n\nGeneration Complete.")
        print(f"Total Generated: {len(generated_routes)}")
        print(f"Rejected (No new coverage): {stats['rejected_coverage']}")

        # Save PKL
        map_data = self._extract_map_data(carla_map)
        # change self.filename to map_file with pkl suffix
        map_file = self.filename.with_suffix('.pkl')
        with open(map_file, 'wb') as f:
            pickle.dump({'town': target_map, 'lines': map_data}, f)
        print(f"Saved map geometry to {map_file}")

        # Save XML
        self._save_xml(generated_routes, target_map, self.filename)

    def _save_xml(self, routes, town_name, filename):
        root = ET.Element('routes')
        for r in routes:
            route_elem = ET.SubElement(root, 'route', id=str(r['id']), town=town_name)
            
            weathers = ET.SubElement(route_elem, 'weathers')
            ET.SubElement(weathers, 'weather', route_percentage="0", cloudiness="5.0", precipitation="0.0", precipitation_deposits="0.0", wetness="0.0", wind_intensity="10.0", sun_azimuth_angle="-1.0", sun_altitude_angle="90.0", fog_density="2.0")
            ET.SubElement(weathers, 'weather', route_percentage="100", cloudiness="5.0", precipitation="0.0", precipitation_deposits="0.0", wetness="0.0", wind_intensity="10.0", sun_azimuth_angle="-1.0", sun_altitude_angle="15.0", fog_density="2.0")
            
            waypoints_elem = ET.SubElement(route_elem, 'waypoints')
            for wp_loc in r['waypoints']: 
                ET.SubElement(waypoints_elem, 'position', x=f"{wp_loc.x:.2f}", y=f"{wp_loc.y:.2f}", z=f"{wp_loc.z:.2f}")

        xmlstr = xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(filename, "w") as f:
            f.write(xmlstr)
        print(f"Saved routes to {filename}")

    def visualize(self, show_num=None):
        """High-quality 'Ribbon' visualization."""
        filename = str(self.filename)
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return

        map_file = self.filename.with_suffix('.pkl')
        map_lines = []
        map_name = "Unknown"
        
        if os.path.exists(map_file):
            print(f"Loading map background from {map_file}...")
            with open(map_file, 'rb') as f:
                data = pickle.load(f)
                map_lines = data.get('lines', [])
                map_name = data.get('town', 'Unknown')
        
        print("Parsing XML routes...")
        tree = ET.parse(filename)
        root = tree.getroot()
        routes = []
        for route in root.findall('route'):
            pts = []
            for pos in route.find('waypoints').findall('position'):
                pts.append((float(pos.get('x')), float(pos.get('y'))))
            routes.append(pts)

        full_route_count = len(routes)
        if show_num is not None:
            if int(show_num) < len(routes):
                print(f"Randomly selecting {show_num} routes from {len(routes)} to visualize...")
                routes = random.sample(routes, int(show_num))

        print(f"Plotting {len(routes)} routes on {map_name}...")
        
        fig, ax = plt.subplots(figsize=(16, 16), dpi=100)
        ax.set_title(f"Route Coverage Map: {map_name} ({full_route_count} routes)", fontsize=16)
        ax.set_aspect('equal')
        ax.axis('off') 

        # Draw Map Background
        # Draw background lighter to make gaps obvious if any
        for xs, ys in map_lines:
            ax.plot(xs, ys, color='#DDDDDD', linewidth=3.0, alpha=1.0, zorder=1)

        # Draw Routes
        cmap = plt.cm.nipy_spectral 
        colors = cmap(np.linspace(0, 1, len(routes)))
        np.random.shuffle(colors)

        for i, pts in enumerate(routes):
            xs = [p[0] for p in pts]
            ys = [-p[1] for p in pts]
            
            ax.plot(xs, ys, 
                    color=colors[i], 
                    linewidth=3.5,     
                    alpha=0.8,         
                    solid_capstyle='round',
                    zorder=10)

        # plt.tight_layout()
        plt.savefig(Path(filename).with_suffix('.png'), bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def visualize_route(self, route_id, show_markers=True, marker_interval=5):
        """
        Visualize a specific route with waypoint order annotations.
        
        Args:
            route_id: The ID of the route to visualize
            show_markers: Whether to show waypoint markers and numbers
            marker_interval: Show marker every N waypoints (to avoid clutter)
        """
        filename = str(self.filename)
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return

        # Load map background
        map_file = self.filename.with_suffix('.pkl')
        map_lines = []
        map_name = "Unknown"
        
        if os.path.exists(map_file):
            print(f"Loading map background from {map_file}...")
            with open(map_file, 'rb') as f:
                data = pickle.load(f)
                map_lines = data.get('lines', [])
                map_name = data.get('town', 'Unknown')
        
        # Parse XML and find the specific route
        print(f"Parsing XML routes, looking for route {route_id}...")
        tree = ET.parse(filename)
        root = tree.getroot()
        
        target_route = None
        for route in root.findall('route'):
            if int(route.get('id')) == int(route_id):
                pts = []
                for pos in route.find('waypoints').findall('position'):
                    pts.append((float(pos.get('x')), float(pos.get('y'))))
                target_route = pts
                break
        
        if target_route is None:
            print(f"Route ID {route_id} not found in {filename}")
            return
        
        print(f"Plotting route {route_id} with {len(target_route)} waypoints on {map_name}...")
        
        fig, ax = plt.subplots(figsize=(16, 16), dpi=100)
        ax.set_title(f"Route {route_id} on {map_name} ({len(target_route)} waypoints)", fontsize=16)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw map background
        for xs, ys in map_lines:
            ax.plot(xs, ys, color='#DDDDDD', linewidth=2.0, alpha=1.0, zorder=1)

        # Prepare route data
        xs = [p[0] for p in target_route]
        ys = [-p[1] for p in target_route]

        # Draw route line with gradient color to show direction
        from matplotlib.collections import LineCollection
        
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Color gradient from green (start) to red (end)
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(segments)))
        lc = LineCollection(segments, colors=colors, linewidth=4, alpha=0.9, zorder=10)
        ax.add_collection(lc)

        if show_markers:
            # Mark start point (green triangle)
            ax.scatter(xs[0], ys[0], c='green', s=300, marker='^', 
                      edgecolors='white', linewidths=2, zorder=20, label='Start')
            ax.annotate('START', (xs[0], ys[0]), fontsize=10, fontweight='bold',
                       color='green', ha='center', va='bottom',
                       xytext=(0, 15), textcoords='offset points', zorder=25)
            
            # Mark end point (red square)
            ax.scatter(xs[-1], ys[-1], c='red', s=300, marker='s',
                      edgecolors='white', linewidths=2, zorder=20, label='End')
            ax.annotate('END', (xs[-1], ys[-1]), fontsize=10, fontweight='bold',
                       color='red', ha='center', va='bottom',
                       xytext=(0, 15), textcoords='offset points', zorder=25)
            
            # Mark intermediate waypoints with numbers
            for i in range(1, len(xs) - 1):
                if i % marker_interval == 0:
                    ax.scatter(xs[i], ys[i], c='blue', s=80, marker='o',
                              edgecolors='white', linewidths=1, zorder=15, alpha=0.8)
                    ax.annotate(str(i), (xs[i], ys[i]), fontsize=7,
                               color='darkblue', ha='center', va='center',
                               xytext=(8, 8), textcoords='offset points', zorder=25)

            # Add direction arrows along the route
            arrow_interval = max(1, len(xs) // 10)
            for i in range(0, len(xs) - 1, arrow_interval):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    ax.annotate('', xy=(xs[i] + dx * 0.6, ys[i] + dy * 0.6),
                               xytext=(xs[i] + dx * 0.4, ys[i] + dy * 0.4),
                               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                               zorder=12)

        ax.legend(loc='upper right', fontsize=12)
        
        # Add colorbar to show progress
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=20, pad=0.02)
        cbar.set_label('Route Progress (%)', fontsize=10)
        
        output_file = Path(filename).with_suffix(f'.route{route_id}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=150)
        print(f"Saved to {output_file}")
        plt.show()

if __name__ == '__main__':
    fire.Fire(AutoRouteTools)