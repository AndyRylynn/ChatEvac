# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from collections import deque, defaultdict
import tkinter as tk
from tkinter import Label
import random
import math
import os
import heapq
from sklearn.cluster import KMeans
import argparse
import sys

# Set output encoding to UTF-8
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Import social force model components
from SFM2 import PeopleList, SFMConfig, pixel_to_meter, meter_to_pixel
_here = os.path.dirname(os.path.abspath(__file__))
adress = os.path.join(_here, "Material", "Output.png")
# =======================================
# Hyperparameters - Navigation and Simulation
# =======================================
class SimulationConfig:
    """Simulation hyperparameter configuration class"""
    def __init__(self,
                 # Image and spatial parameters
                 image_width=512,           # Image pixel width
                 image_height=512,          # Image pixel height
                 space_width=30.0,          # Actual space width (meters)
                 space_height=30.0,         # Actual space height (meters)

                 # Navigation mesh parameters
                 target_points=150,         # Target node count
                 num_exit_seeds=5,          # Exit node count
                 safety_margin=10,          # Safety margin (pixels)
                 min_node_distance=16,      # Minimum node distance (pixels)
                 corridor_width_threshold=15,  # Narrow corridor width threshold
                 skeleton_priority_factor=2.0, # Skeleton priority factor
                 enable_edge_check=True,    # Enable edge checking

                 # Simulation parameters
                 num_people=30,             # Number of pedestrians
                 debug_visualization=True,  # Show debug visualization

                 # GUI parameters
                 window_width=1080,         # Window width
                 window_height=1130,        # Window height
                 canvas_width=1080,         # Canvas width
                 canvas_height=1080):       # Canvas height
        
        # Image and spatial parameters
        self.image_width = image_width
        self.image_height = image_height
        self.space_width = space_width
        self.space_height = space_height
        self.pixels_to_meters_x = space_width / image_width
        self.pixels_to_meters_y = space_height / image_height
        
        # Navigation mesh parameters
        self.target_points = target_points
        self.num_exit_seeds = num_exit_seeds
        self.safety_margin = safety_margin
        self.min_node_distance = min_node_distance
        self.corridor_width_threshold = corridor_width_threshold
        self.skeleton_priority_factor = skeleton_priority_factor
        self.enable_edge_check = enable_edge_check
        
        # Simulation parameters
        self.num_people = num_people
        self.debug_visualization = debug_visualization
        
        # GUI parameters
        self.window_width = window_width
        self.window_height = window_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

# Default configuration
default_sim_config = SimulationConfig()

# =======================================
# Command Line Argument Parser
# =======================================
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Pedestrian Evacuation Simulation with Social Force Model')
    
    # === SFM Parameters ===
    sfm_group = parser.add_argument_group('Social Force Model Parameters')
    sfm_group.add_argument('--social-force-a1', type=float, default=2000,
                          help='Pedestrian-pedestrian repulsion force strength (default: 2000)')
    sfm_group.add_argument('--social-force-a2', type=float, default=2000,
                          help='Pedestrian-wall repulsion force strength (default: 2000)')
    sfm_group.add_argument('--social-force-b', type=float, default=-0.08,
                          help='Repulsion force range coefficient (default: -0.08)')
    sfm_group.add_argument('--delta-time', type=float, default=0.1,
                          help='Simulation time step (default: 0.1)')
    sfm_group.add_argument('--people-mass-base', type=int, default=50,
                          help='Pedestrian mass base value in kg (default: 50)')
    sfm_group.add_argument('--people-mass-variation', type=int, default=20,
                          help='Pedestrian mass variation range in kg (default: 20)')
    sfm_group.add_argument('--people-radius-base', type=float, default=0.25,
                          help='Pedestrian radius base value. '
                               'Helbing engine: meters (e.g. 0.25); SFM2 engine: pixels (e.g. 35)')
    sfm_group.add_argument('--people-radius-variation', type=int, default=5,
                          help='Pedestrian radius variation range (default: 5)')
    sfm_group.add_argument('--people-radius-scale', type=int, default=200,
                          help='Pedestrian radius scale factor (default: 200)')
    sfm_group.add_argument('--people-speed-base', type=int, default=60,
                          help='Pedestrian desired speed base value (default: 60)')
    sfm_group.add_argument('--people-speed-variation', type=int, default=20,
                          help='Pedestrian desired speed variation range (default: 20)')
    sfm_group.add_argument('--people-speed-scale', type=int, default=100,
                          help='Pedestrian desired speed scale factor (default: 100)')
    sfm_group.add_argument('--relaxation-time', type=float, default=0.5,
                          help='Relaxation time constant (default: 0.5)')
    sfm_group.add_argument('--max-speed-factor', type=float, default=1.7,
                          help='Maximum speed factor (default: 1.2)')
    sfm_group.add_argument('--interaction-distance', type=float, default=1.4,
                          help='Pedestrian interaction distance (default: 1.4)')
    sfm_group.add_argument('--wall-check-distance', type=int, default=20,
                          help='Wall check distance in pixels (default: 20)')
    
    # === Simulation Parameters ===
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--image-width', type=int, default=512,
                          help='Image pixel width (default: 512)')
    sim_group.add_argument('--image-height', type=int, default=512,
                          help='Image pixel height (default: 512)')
    sim_group.add_argument('--space-width', type=float, default=30.0,
                          help='Actual space width in meters (default: 30.0)')
    sim_group.add_argument('--space-height', type=float, default=30.0,
                          help='Actual space height in meters (default: 30.0)')
    sim_group.add_argument('--target-points', type=int, default=150,
                          help='Target node count (default: 150)')
    sim_group.add_argument('--num-exit-seeds', type=int, default=5,
                          help='Exit node count (default: 5)')
    sim_group.add_argument('--safety-margin', type=int, default=10,
                          help='Safety margin in pixels (default: 10)')
    sim_group.add_argument('--min-node-distance', type=int, default=16,
                          help='Minimum node distance in pixels (default: 16)')
    sim_group.add_argument('--num-people', type=int, default=30,
                          help='Number of pedestrians (default: 30)')
    sim_group.add_argument('--no-debug', action='store_true',
                          help='Disable debug visualization')
    
    # === File Parameters ===
    file_group = parser.add_argument_group('File Parameters')
    file_group.add_argument('--image-path', type=str,
                          default=adress,
                          help='Floor plan image path')
    file_group.add_argument('--engine', choices=['helbing', 'sfm2'], default='helbing',
                          help='Evacuation engine: Helbing (strict Helbing 2000, default) '
                               'or SFM2 (legacy navigation-graph variant)')

    return parser.parse_args()

# =======================================
# Navigation Mesh Utility Functions
# =======================================
def check_line_walkable(p1, p2, walls):
    """Check if the line segment p1->p2 is walkable (no collision) in the walls binary image"""
    y1, x1 = max(0, int(p1[0])), max(0, int(p1[1]))
    y2, x2 = min(walls.shape[0] - 1, int(p2[0])), min(walls.shape[1] - 1, int(p2[1]))
    mask = np.zeros_like(walls)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    return np.count_nonzero(cv2.bitwise_and(walls, mask)) == 0

def detect_exit_regions(hsv_image):
    """Detect red exit areas using multiple HSV masks, with dilation"""
    red_ranges = [
    ([0, 150, 150], [5, 255, 255]),
    ([175, 150, 150], [180, 255, 255]),
    ]
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for lo, hi in red_ranges:
        mask |= cv2.inRange(hsv_image, np.array(lo), np.array(hi))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    return mask

def sample_points_in_mask(mask, num_points):
    """Randomly sample [y,x] coordinates from non-zero regions of a binary mask"""
    ys, xs = np.where(mask > 0)
    coords = np.vstack((ys, xs)).T
    if len(coords) == 0:
        return np.empty((0, 2), dtype=int)
    if len(coords) <= num_points:
        return coords
    idx = np.random.choice(len(coords), num_points, replace=False)
    return coords[idx]

# =======================================
# Advanced Node Placement Functions
# =======================================
def calculate_coverage_radius(walkable_area, target_points, min_node_distance):
    """Calculate theoretical coverage radius based on walkable area and target points"""
    area_per_point = walkable_area / target_points
    # Assume each point covers a circular area
    coverage_radius = np.sqrt(area_per_point / np.pi)
    return max(coverage_radius, min_node_distance)

def find_medial_axis_nodes(dist_transform, walkable_mask, num_samples=50):
    """
    Find medial axis points (corridor centerlines) via distance transform gradient; returns sparsely sampled.
    Does not depend on ximgproc.thinning.
    """
    gy, gx = np.gradient(dist_transform.astype(np.float64))
    grad_mag = np.sqrt(gy**2 + gx**2)
    medial_mask = (walkable_mask > 0) & (grad_mag < 0.5) & (dist_transform > 3)
    coords = np.argwhere(medial_mask).astype(np.float64)
    if len(coords) == 0:
        return np.array([]).reshape(0, 2)
    if len(coords) <= num_samples:
        return coords
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init=10)
    kmeans.fit(coords)
    return kmeans.cluster_centers_


def extract_skeleton_and_classify(walkable_mask):
    """Extract distance transform + narrow passage points, keeping original interface compatibility."""
    dist_transform = cv2.distanceTransform(walkable_mask, cv2.DIST_L2, 5)
    narrow_mask = (walkable_mask > 0) & (dist_transform > 0) & (dist_transform < 8)
    narrow_coords = np.argwhere(narrow_mask)
    if len(narrow_coords) > 80:
        kmeans = KMeans(n_clusters=80, random_state=42, n_init=10)
        kmeans.fit(narrow_coords.astype(np.float64))
        narrow_points = kmeans.cluster_centers_.astype(int)
    else:
        narrow_points = narrow_coords
    skeleton = np.zeros_like(walkable_mask, dtype=np.uint8)
    junction_points = np.array([]).reshape(0, 2)
    end_points = np.array([]).reshape(0, 2)
    return skeleton, junction_points, end_points, narrow_points, dist_transform


def poisson_disk_sampling(walkable_mask, radius, skeleton_points=None, max_attempts=30):
    """Improved Poisson disk sampling with skeleton point priority"""
    points = []
    active_list = []

    # Create grid for accelerated lookup
    cell_size = radius / np.sqrt(2)
    grid_width = int(np.ceil(walkable_mask.shape[1] / cell_size))
    grid_height = int(np.ceil(walkable_mask.shape[0] / cell_size))
    grid = {}

    def get_grid_coords(point):
        return int(point[1] / cell_size), int(point[0] / cell_size)

    def is_valid_point(point):
        y, x = int(point[0]), int(point[1])
        if y < 0 or y >= walkable_mask.shape[0] or x < 0 or x >= walkable_mask.shape[1]:
            return False
        if walkable_mask[y, x] == 0:
            return False

        # Check distance to existing points
        grid_x, grid_y = get_grid_coords(point)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                neighbor_grid_y = grid_y + dy
                neighbor_grid_x = grid_x + dx
                if (neighbor_grid_x, neighbor_grid_y) in grid:
                    neighbor = grid[(neighbor_grid_x, neighbor_grid_y)]
                    if np.linalg.norm(np.array(point) - np.array(neighbor)) < radius:
                        return False
        return True

    def add_point(point):
        points.append(point)
        active_list.append(len(points) - 1)
        grid_x, grid_y = get_grid_coords(point)
        grid[(grid_x, grid_y)] = point

    # If skeleton points exist, prioritize starting from skeleton points
    if skeleton_points is not None and len(skeleton_points) > 0:
        # Randomly select a skeleton point as starting point
        start_idx = np.random.randint(0, len(skeleton_points))
        start_point = skeleton_points[start_idx]
        if is_valid_point(start_point):
            add_point(start_point)
    else:
        # Randomly select starting point
        valid_coords = np.argwhere(walkable_mask > 0)
        if len(valid_coords) == 0:
            return np.array([])
        start_point = valid_coords[np.random.randint(0, len(valid_coords))]
        add_point(start_point)

    while active_list:
        active_idx = np.random.randint(0, len(active_list))
        current_point = points[active_list[active_idx]]

        found_valid = False
        for _ in range(max_attempts):
            # Generate new point in annular region
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(radius, 2 * radius)
            new_point = [
                current_point[0] + distance * np.sin(angle),
                current_point[1] + distance * np.cos(angle)
            ]

            if is_valid_point(new_point):
                add_point(new_point)
                found_valid = True
                break

        if not found_valid:
            active_list.pop(active_idx)

    return np.array(points)

# =======================================
# Corner-Guided Node Placement Functions
# =======================================
def detect_wall_corners(walls_mask, walkable_mask, projection_distance):
    """
    Detect polygonal vertices of wall/obstacle contours, project inward onto walkable area.
    Uses cv2.findContours + approxPolyDP to obtain geometric corner points.

    Args:
        walls_mask: Wall binary image (255=wall, 0=non-wall)
        walkable_mask: Eroded walkable area binary image
        projection_distance: Distance to project inward from corner

    Returns:
        corner_nodes: np.array of [y, x] corner node positions
    """
    contours, hierarchy = cv2.findContours(walls_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    h, w = walls_mask.shape
    corner_nodes = []

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 20:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.015 * perimeter, 1.5)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        for pt in approx:
            cy, cx = int(pt[0][1]), int(pt[0][0])

            # Scan walkable pixels around corner; centroid direction = "inward" direction
            scan_r = max(int(projection_distance * 1.5), 5)
            y1 = max(0, cy - scan_r)
            y2 = min(h, cy + scan_r + 1)
            x1 = max(0, cx - scan_r)
            x2 = min(w, cx + scan_r + 1)

            local_walk = walkable_mask[y1:y2, x1:x2]
            wy, wx = np.where(local_walk > 0)

            if len(wy) == 0:
                continue

            # Walkable pixel centroid -> inward direction
            wy_abs = wy + y1
            wx_abs = wx + x1
            mean_y = np.mean(wy_abs)
            mean_x = np.mean(wx_abs)

            dy = mean_y - cy
            dx = mean_x - cx
            dist = np.sqrt(dy**2 + dx**2)

            if dist > 1e-6:
                dy, dx = dy / dist, dx / dist
                ny = int(cy + dy * projection_distance)
                nx = int(cx + dx * projection_distance)

                if 0 <= ny < h and 0 <= nx < w and walkable_mask[ny, nx] > 0:
                    corner_nodes.append([ny, nx])

    if corner_nodes:
        return np.array(corner_nodes)
    return np.array([]).reshape(0, 2)


def place_corner_nodes(walkable_mask, walls_mask, projection_dist):
    """
    Place navigation nodes near wall corners. Deduplicate and merge nodes that are too close.
    """
    raw_corners = detect_wall_corners(walls_mask, walkable_mask, projection_dist)

    if len(raw_corners) == 0:
        return np.array([]).reshape(0, 2)

    # Greedy dedup: merge nodes closer than projection_dist
    kept = []
    for pt in raw_corners:
        too_close = False
        for k in kept:
            if np.linalg.norm(pt - k) < projection_dist * 0.5:
                too_close = True
                break
        if not too_close:
            kept.append(pt)

    return np.array(kept)


def find_open_area_centers(dist_transform, walkable_mask, min_distance, max_centers=15):
    """
    Place nodes at local maxima of distance transform (centers of large open spaces).
    Use KMeans clustering to avoid too many nodes.
    """
    from scipy import ndimage

    footprint_size = max(min_distance // 3, 3)
    footprint = np.ones((footprint_size, footprint_size))
    local_max = ndimage.maximum_filter(dist_transform, footprint=footprint)
    peaks = (dist_transform == local_max) & (walkable_mask > 0) & (dist_transform > min_distance)

    peak_coords = np.argwhere(peaks).astype(np.float64)

    if len(peak_coords) <= max_centers:
        return peak_coords

    n_clusters = min(len(peak_coords), max_centers)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(peak_coords)
    return kmeans.cluster_centers_


def refined_node_placement(walkable_mask, walls_mask, sim_config):
    """
    Corner-guided + skeleton + open area center node placement strategy.
    Replaces the original adaptive_node_placement + poisson_disk_sampling.

    Returns:
        refined_nodes: All placed nodes [y, x]
        coverage_radius: Theoretical coverage radius (for optional fill)
        skeleton: Skeleton visualization
        dist_transform: Distance transform
    """
    print("Starting refined corner-guided node placement...")

    projection_dist = sim_config.safety_margin + sim_config.min_node_distance

    # 1. Corner nodes — ensure connectivity at geometrically critical locations
    corner_nodes = place_corner_nodes(walkable_mask, walls_mask, projection_dist)
    print(f"  Corner nodes: {len(corner_nodes)}")

    # 2. Corridor medial axis nodes (based on distance transform gradient) + narrow passage points
    skeleton, _, _, narrow_points, dist_transform = extract_skeleton_and_classify(walkable_mask)

    # Medial axis nodes: sample along corridor centerlines
    num_medial = max(30, sim_config.target_points // 3)
    medial_nodes = find_medial_axis_nodes(dist_transform, walkable_mask, num_medial)

    skeleton_nodes = []
    if len(medial_nodes) > 0:
        skeleton_nodes.extend(medial_nodes.tolist())
    # Supplement with narrow passage points
    if len(narrow_points) > 0:
        n_narrow = min(len(narrow_points), 30)
        if len(narrow_points) > n_narrow:
            kmeans = KMeans(n_clusters=n_narrow, random_state=42, n_init=10)
            kmeans.fit(narrow_points.astype(np.float64))
            narrow_clustered = kmeans.cluster_centers_
            skeleton_nodes.extend(narrow_clustered.tolist())
        else:
            skeleton_nodes.extend(narrow_points.tolist())

    if skeleton_nodes:
        skeleton_nodes = np.unique(np.array(skeleton_nodes), axis=0)
        if len(corner_nodes) > 0:
            kept = []
            for sn in skeleton_nodes:
                if np.min(np.linalg.norm(corner_nodes - sn, axis=1)) >= projection_dist * 0.5:
                    kept.append(sn)
            skeleton_nodes = np.array(kept) if kept else np.array([]).reshape(0, 2)
    else:
        skeleton_nodes = np.array([]).reshape(0, 2)

    print(f"  Medial axis nodes: {len(skeleton_nodes)}")

    # 3. Open area center nodes
    open_nodes = find_open_area_centers(dist_transform, walkable_mask, projection_dist)

    all_so_far = np.vstack([corner_nodes, skeleton_nodes]) if len(skeleton_nodes) > 0 else corner_nodes
    if len(open_nodes) > 0 and len(all_so_far) > 0:
        kept_open = []
        for on in open_nodes:
            if np.min(np.linalg.norm(all_so_far - on, axis=1)) >= projection_dist:
                kept_open.append(on)
        open_nodes = np.array(kept_open) if kept_open else np.array([]).reshape(0, 2)

    print(f"  Open area nodes: {len(open_nodes)}")

    # 3.5 Uniform grid fill — give open areas enough "entry points"
    all_so_far = corner_nodes
    if len(skeleton_nodes) > 0:
        all_so_far = np.vstack([all_so_far, skeleton_nodes])
    if len(open_nodes) > 0:
        all_so_far = np.vstack([all_so_far, open_nodes.astype(np.float64)])

    # Grid spacing adapts to target node count (fills roughly to target_points)
    walkable_area = np.sum(walkable_mask > 0)
    target_density = sim_config.target_points / walkable_area
    grid_spacing = max(projection_dist, int(1.0 / np.sqrt(target_density)))
    fill_nodes = []
    h, w = walkable_mask.shape
    for gy in range(grid_spacing // 2, h, grid_spacing):
        for gx in range(grid_spacing // 2, w, grid_spacing):
            if walkable_mask[gy, gx] > 0:
                fill_nodes.append([gy, gx])

    if fill_nodes and len(all_so_far) > 0:
        fill_nodes = np.array(fill_nodes)
        kept_fill = []
        for fn in fill_nodes:
            if np.min(np.linalg.norm(all_so_far - fn, axis=1)) >= projection_dist:
                kept_fill.append(fn)
        fill_nodes = np.array(kept_fill) if kept_fill else np.array([]).reshape(0, 2)

    print(f"  Uniform fill nodes: {len(fill_nodes)}")

    # 4. Merge all nodes
    all_refined = []
    if len(corner_nodes) > 0:
        all_refined.append(corner_nodes)
    if len(skeleton_nodes) > 0:
        all_refined.append(skeleton_nodes)
    if len(open_nodes) > 0:
        all_refined.append(open_nodes.astype(np.float64))
    if len(fill_nodes) > 0:
        all_refined.append(fill_nodes.astype(np.float64))

    refined_nodes = np.vstack(all_refined) if all_refined else np.array([]).reshape(0, 2)

    # 5. Calculate coverage radius (for optional Poisson fill)
    walkable_area = np.sum(walkable_mask > 0)
    coverage_radius = calculate_coverage_radius(walkable_area, sim_config.target_points, sim_config.min_node_distance)

    print(f"  Total refined nodes: {len(refined_nodes)}")
    return refined_nodes, coverage_radius, skeleton, dist_transform

def add_visibility_edges(adjacency, all_points, walls, target_points):
    """
    Add "visibility edges" on top of Voronoi edges.
    Two nodes are directly connected if the straight line does not cross walls, allowing pedestrians to take shortcuts in open areas instead of detouring.

    Args:
        adjacency: Existing adjacency list (Voronoi edges)
        all_points: All nodes [y, x]
        walls: Wall binary image
        target_points: Target node count, used to estimate search range

    Returns:
        n_added: Number of newly added edges
    """
    n = len(all_points)
    if n < 2:
        return 0

    # Each node connects to at most 2 nearest visible neighbors (local shortcuts, no long-distance connections)
    k_nearest = min(2, n - 1)
    n_added = 0
    existing = {i: set(adjacency[i]) for i in range(n)}

    # Search distance cap: Voronoi median edge length * 2 (local shortcuts only)
    all_dists = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                d = np.sqrt((all_points[i][0]-all_points[j][0])**2 +
                            (all_points[i][1]-all_points[j][1])**2)
                all_dists.append(d)
    max_edge_dist = np.median(all_dists) * 2 if all_dists else float('inf')

    for i in range(n):
        yi, xi = all_points[i]
        candidates = []
        for j in range(n):
            if i == j or j in existing[i]:
                continue
            dist = np.sqrt((yi - all_points[j][0])**2 + (xi - all_points[j][1])**2)
            if dist <= max_edge_dist:
                candidates.append((dist, j))
        if not candidates:
            continue

        candidates.sort()
        added_for_i = 0
        for dist, j in candidates:
            if added_for_i >= k_nearest:
                break
            if check_line_walkable(all_points[i], all_points[j], walls):
                adjacency[i].append(j)
                adjacency[j].append(i)
                existing[i].add(j)
                existing[j].add(i)
                n_added += 1
                added_for_i += 1

    return n_added


def generate_navigation_mesh(image_path, sim_config, debug=False):
    """Generate navigation mesh using optimized approach"""
    # 1. Load image (handle RGBA) and convert to BGR + HSV
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Detect exit mask & walkable areas
    exit_mask = detect_exit_regions(hsv)
    black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    walk = cv2.bitwise_or(black, exit_mask)
    walls = cv2.dilate(255 - walk, np.ones((3, 3), np.uint8), iterations=1)
    eroded = cv2.erode(255 - walls, np.ones((sim_config.safety_margin, sim_config.safety_margin), np.uint8))

    # 3. Get refined nodes (corner + skeleton + open area)
    refined_nodes, coverage_radius, skeleton, dist_transform = refined_node_placement(eroded, walls, sim_config)

    # 4. Sample exit points
    exit_coords = np.argwhere(exit_mask > 0)
    if len(exit_coords) > 0:
        if len(exit_coords) <= sim_config.num_exit_seeds:
            exit_points = exit_coords
        else:
            kmeans = KMeans(n_clusters=sim_config.num_exit_seeds, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(exit_coords)
            exit_points = []
            for i in range(sim_config.num_exit_seeds):
                cluster_points = exit_coords[clusters == i]
                if len(cluster_points) > 0:
                    center = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_points - center, axis=1)
                    closest_idx = np.argmin(distances)
                    exit_points.append(cluster_points[closest_idx])
            exit_points = np.array(exit_points)
    else:
        exit_points = np.array([]).reshape(0, 2)

    print(f"Selected {len(exit_points)} exit points")

    # 5. Filter refined nodes: remove those too close to exits
    if len(refined_nodes) > 0 and len(exit_points) > 0:
        min_exit_dist = sim_config.safety_margin + sim_config.min_node_distance
        kept_nodes = []
        for rn in refined_nodes:
            if np.min(np.linalg.norm(exit_points - rn, axis=1)) >= min_exit_dist:
                kept_nodes.append(rn)
        refined_nodes = np.array(kept_nodes) if kept_nodes else np.array([]).reshape(0, 2)

    # 6. Merge: exit_points + refined_nodes = all_points
    all_points = []
    if len(exit_points) > 0:
        all_points.extend(exit_points.tolist())
    if len(refined_nodes) > 0:
        all_points.extend(refined_nodes.tolist())

    all_points = np.array(all_points)
    print(f"Finally generated {len(all_points)} nodes")

    if len(all_points) < 3:
        raise RuntimeError("Too few sample points to generate Voronoi")

    vor = Voronoi(all_points[:, [1, 0]])

    # 8. Build nodes & exit IDs
    nodes = {}
    exits = []
    for idx, (y, x) in enumerate(all_points):
        xm, ym = pixel_to_meter(x, y, sim_config.pixels_to_meters_x, sim_config.pixels_to_meters_y)
        is_ex = (idx < len(exit_points))
        nodes[idx] = {"pos": (xm, ym), "is_exit": is_ex}
        if is_ex:
            exits.append(idx)

    # 9. Build edges
    adjacency = defaultdict(list)
    seen = set()
    for u, v in vor.ridge_points:
        if -1 in (u, v):
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        if not sim_config.enable_edge_check or check_line_walkable(all_points[u], all_points[v], walls):
            seen.add(key)
            adjacency[u].append(v)
            adjacency[v].append(u)

    # 9b. Add visibility edges — directly connect two nodes if they can see each other, avoiding detours
    if sim_config.enable_edge_check:
        n_added = add_visibility_edges(adjacency, all_points, walls,
                                       sim_config.target_points)
        print(f"Added {n_added} visibility edges")

    # Connectivity check
    visited = set()
    queue = deque(exits)
    while queue:
        u = queue.popleft()
        if u in visited:
            continue
        visited.add(u)
        for w in adjacency[u]:
            if w not in visited:
                queue.append(w)
    all_connected = (len(visited) == len(nodes))
    unreachable = set(range(len(nodes))) - visited
    # Delete unreachable nodes and related edges
    if unreachable:
        print(f"Removing {len(unreachable)} unreachable nodes...")
        
        # Delete all edges of unreachable nodes
        for node_id in unreachable:
            if node_id in adjacency:
                # Remove references to this node from neighbor nodes
                for neighbor in adjacency[node_id]:
                    if neighbor in adjacency:
                        adjacency[neighbor] = [n for n in adjacency[neighbor] if n != node_id]
                # Delete this node's adjacency list
                del adjacency[node_id]
        
        # Delete unreachable nodes
        for node_id in sorted(unreachable, reverse=True):  # Delete from large to small to avoid index issues
            del nodes[node_id]
        
        # Rebuild node mapping (because indexes will change after deletion)
        old_to_new = {}
        new_nodes = {}
        new_all_points = []
        new_exits = []
        
        new_idx = 0
        for old_idx in range(len(nodes) + len(unreachable)):
            if old_idx not in unreachable:
                old_to_new[old_idx] = new_idx
                new_nodes[new_idx] = nodes[old_idx]
                new_all_points.append(all_points[old_idx])
                if old_idx in exits:
                    new_exits.append(new_idx)
                new_idx += 1
        
        # Update adjacency list indexes
        new_adjacency = defaultdict(list)
        for old_node, neighbors in adjacency.items():
            if old_node in old_to_new:
                new_node = old_to_new[old_node]
                for old_neighbor in neighbors:
                    if old_neighbor in old_to_new:
                        new_neighbor = old_to_new[old_neighbor]
                        new_adjacency[new_node].append(new_neighbor)
        
        # Update variables
        nodes = new_nodes
        adjacency = new_adjacency
        exits = new_exits
        all_points = np.array(new_all_points)
        
        # Update connectivity status
        all_connected = True
        unreachable = set()
        
        print(f"After cleanup: {len(nodes)} nodes, {len(exits)} exits")
        
    
    # Output diagnostic info
    print(f"Exits: {len(exits)}, Nodes: {len(nodes)}, Edges: {len(adjacency)}")
    print(f"All connected? {'Yes' if all_connected else 'No'}")
    if not all_connected:
        print(f"Unreachable sample IDs: {list(unreachable)[:10]}")

    # Visualization
    if debug:
        overlay = np.zeros_like(img)
        for u in adjacency:
            for v in adjacency[u]:
                if u < v:  # Only draw each edge once
                    y1, x1 = all_points[u]
                    y2, x2 = all_points[v]
                    cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        for idx, (y, x) in enumerate(all_points):
            col = (0, 0, 255) if nodes[idx]["is_exit"] else (0, 255, 0)
            cv2.circle(overlay, (int(x), int(y)), 3, col, -1)
        vis = cv2.addWeighted(img, 1, overlay, 1, 0)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Navigation Mesh with Optimized Node Placement")
        plt.show()

    return nodes, adjacency, exits, all_connected, unreachable, all_points, walls

# =======================================
# GUI Class
# =======================================
class GUI:
    def __init__(self, sim_config):
        """Set up the window"""
        self.sim_config = sim_config
        self.top = tk.Tk()
        self.top.title("Pedestrian Evacuation Simulation")
        self.top.geometry(f"{sim_config.window_width}x{sim_config.window_height}")
        self.top.resizable(width=False, height=False)

        """Set up the canvas"""
        self.c = tk.Canvas(self.top, width=sim_config.canvas_width, height=sim_config.canvas_height, bg="#A9A9A9")
        self.c.pack()

        """Set up status labels"""
        self.label_frame = tk.Frame(self.top)
        self.label_frame.pack(fill=tk.X)

        self.time_label = Label(self.label_frame, text="Time = 0.0 s")
        self.time_label.pack(side=tk.LEFT, padx=10)

        self.people_label = Label(self.label_frame, text="People remaining: 0")
        self.people_label.pack(side=tk.RIGHT, padx=10)

    def draw_navigation_mesh(self, nodes, adjacency, all_pts):
        """Draw the navigation mesh"""
        # Draw edges
        for node_id in adjacency:
            for neighbor in adjacency[node_id]:
                if node_id < neighbor:  # Only draw each edge once
                    y1, x1 = all_pts[node_id]
                    y2, x2 = all_pts[neighbor]
                    # Scale to GUI
                    x1_scaled = int(x1 * self.sim_config.canvas_width / self.sim_config.image_width)
                    y1_scaled = int(y1 * self.sim_config.canvas_height / self.sim_config.image_height)
                    x2_scaled = int(x2 * self.sim_config.canvas_width / self.sim_config.image_width)
                    y2_scaled = int(y2 * self.sim_config.canvas_height / self.sim_config.image_height)
                    self.c.create_line(x1_scaled, y1_scaled, x2_scaled, y2_scaled, fill="blue", width=1, tags="mesh")

        # Draw nodes
        for node_id, data in nodes.items():
            y, x = all_pts[node_id]
            # Scale to GUI
            x_scaled = int(x * self.sim_config.canvas_width / self.sim_config.image_width)
            y_scaled = int(y * self.sim_config.canvas_height / self.sim_config.image_height)

            if data["is_exit"]:
                color = "red"
                size = 5
            else:
                color = "green"
                size = 3

            self.c.create_oval(
                x_scaled - size, y_scaled - size,
                x_scaled + size, y_scaled + size,
                fill=color, outline=color, tags="mesh"
            )

    def draw_walls(self, walls):
        """Draw walls and obstacles"""
        # Scale to GUI
        h, w = walls.shape
        scale_x = self.sim_config.canvas_width / w
        scale_y = self.sim_config.canvas_height / h

        # Draw each wall pixel
        for y in range(h):
            for x in range(w):
                if walls[y, x] > 0:
                    x1 = int(x * scale_x)
                    y1 = int(y * scale_y)
                    x2 = int((x + 1) * scale_x)
                    y2 = int((y + 1) * scale_y)
                    self.c.create_rectangle(x1, y1, x2, y2, fill="#000000", outline="", tags="walls")

    def clear_mesh(self):
        """Clear navigation mesh"""
        self.c.delete("mesh")

    def clear_walls(self):
        """Clear walls"""
        self.c.delete("walls")

    def add_oval(self, x, y, r, oval_tag):
        """Draw a pedestrian with specified meter coordinates and radius"""
        # Convert from meters to pixels
        px, py = meter_to_pixel(x, y, self.sim_config.pixels_to_meters_x, self.sim_config.pixels_to_meters_y)

        # Scale to GUI
        x_scaled = int(px * self.sim_config.canvas_width / self.sim_config.image_width)
        y_scaled = int(py * self.sim_config.canvas_height / self.sim_config.image_height)

        # Convert radius to GUI scale
        r_scaled = int(r / self.sim_config.pixels_to_meters_x * self.sim_config.canvas_width / self.sim_config.image_width)

        self.c.create_oval(
            x_scaled - r_scaled, y_scaled - r_scaled,
            x_scaled + r_scaled, y_scaled + r_scaled,
            fill="#FFE4B5", tags=oval_tag
        )

    def del_oval(self, oval_tag):
        """Delete a pedestrian"""
        self.c.delete(oval_tag)

    def update_time(self, time_str):
        """Update displayed time"""
        self.time_label.config(text="Time = " + time_str + " s")

    def update_people_count(self, count):
        """Update displayed people count"""
        self.people_label.config(text=f"People remaining: {count}")

    def update_gui(self):
        """Update the GUI"""
        self.top.update()
        self.c.update()

    def start(self):
        """Start the GUI main loop"""
        self.top.mainloop()

# =======================================
# Main Function
# =======================================
def _run_helbing_engine(args):
    """New Helbing engine: first run headless to produce EvacT/EvacFlow/EvacLocation/traj.npz,
    then call data_process to generate heatmaps etc., finally replay trajectories with Tk.

    Note: SFM2 pixel/scale-unit physical params (B signed, radius in px, wall_check in px,
    speed with scale=100) are NOT forwarded — they are incompatible with Helbing meter/sec/positive-B
    units. Only num_people / image_path / space_width are forwarded; other physics use Helbing
    defaults (=helbing_test verified)."""
    import SFM
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "Material", "data")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(args.image_path):
        print(f"[launch] {args.image_path} not found; cannot run helbing engine.")
        return

    print("=== Helbing Engine ===")
    print(f"image_path  = {args.image_path}")
    print(f"output_dir  = {output_dir}")
    print(f"num_people  = {args.num_people}")
    print(f"space_width = {args.space_width} m")

    # people_radius_base unit detection: Helbing uses meters (~0.25), SFM2 uses pixels (~35)
    # Empirical threshold 5: < 5 treated as meters, otherwise pixels converted via space_width/image_width
    radius_arg = float(args.people_radius_base)
    if radius_arg >= 5.0:
        m_per_px = float(args.space_width) / float(args.image_width)
        radius_m = radius_arg * m_per_px
        print(f"radius      = {radius_arg} px -> {radius_m:.3f} m")
    else:
        radius_m = radius_arg
        print(f"radius      = {radius_m:.3f} m")

    result = SFM.simulate(
        image_path=args.image_path,
        output_dir=output_dir,
        num_people=int(args.num_people),
        space_width_m=float(args.space_width),
        radius=radius_m,
        # Remaining physics use SFM defaults (Helbing 2000 original, consistent with helbing_test)
    )

    # Reuse the exact same downstream analysis as SFM2
    try:
        import data_process
        print("\n=== Post-processing (data_process) ===")
        data_process.process_evacuation_data(
            output_path=output_dir,
            image_path=args.image_path,
        )
    except Exception as e:
        print(f"[launch] data_process failed: {e}")

    # Replay (blocking until window closed)
    print("\n=== Replay GUI ===")
    try:
        SFM.replay(traj_path=result["traj_path"], image_path=args.image_path)
    except Exception as e:
        print(f"[launch] replay failed: {e}")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # === Helbing engine branch (default) ===
    if args.engine == 'helbing':
        _run_helbing_engine(args)
        return

    # Create SFM configuration
    sfm_config = SFMConfig(
        social_force_a1=args.social_force_a1,
        social_force_a2=args.social_force_a2,
        social_force_b=args.social_force_b,
        delta_time=args.delta_time,
        people_mass_base=args.people_mass_base,
        people_mass_variation=args.people_mass_variation,
        people_radius_base=args.people_radius_base,
        people_radius_variation=args.people_radius_variation,
        people_radius_scale=args.people_radius_scale,
        people_speed_base=args.people_speed_base,
        people_speed_variation=args.people_speed_variation,
        people_speed_scale=args.people_speed_scale,
        relaxation_time=args.relaxation_time,
        max_speed_factor=args.max_speed_factor,
        interaction_distance=args.interaction_distance,
        wall_check_distance=args.wall_check_distance
    )
    
    # Create simulation configuration
    sim_config = SimulationConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        space_width=args.space_width,
        space_height=args.space_height,
        target_points=args.target_points,
        num_exit_seeds=args.num_exit_seeds,
        safety_margin=args.safety_margin,
        min_node_distance=args.min_node_distance,
        num_people=args.num_people,
        debug_visualization=not args.no_debug
    )
    
    # Print configuration information
    print("=== Simulation Configuration ===")
    print(f"Image dimensions: {sim_config.image_width}x{sim_config.image_height}")
    print(f"Space dimensions: {sim_config.space_width}x{sim_config.space_height}m")
    print(f"Number of people: {sim_config.num_people}")
    print(f"Target nodes: {sim_config.target_points}")
    print(f"Number of exits: {sim_config.num_exit_seeds}")
    print("\n=== Social Force Model Parameters ===")
    print(f"People-people repulsion force strength: {sfm_config.social_force_a1}")
    print(f"People-wall repulsion force strength: {sfm_config.social_force_a2}")
    print(f"Repulsion force range coefficient: {sfm_config.social_force_b}")
    print(f"Time step: {sfm_config.delta_time}")
    print(f"Relaxation time: {sfm_config.relaxation_time}")
    print()

    # Check image file
    if not os.path.exists(args.image_path):
        print(f"{args.image_path} not found. Creating a simple floor plan...")

        # Create a simple floor plan with exits
        floor_plan = np.ones((sim_config.image_width, sim_config.image_height, 3), dtype=np.uint8) * 255  # White background

        # Draw walls (black)
        cv2.rectangle(floor_plan, (50, 50), (450, 450), (0, 0, 0), -1)  # Outer wall
        cv2.rectangle(floor_plan, (100, 100), (400, 400), (255, 255, 255), -1)  # Inner space
        
        # Draw obstacles (black)
        cv2.rectangle(floor_plan, (200, 150), (250, 350), (0, 0, 0), -1)  # Obstacle 1
        cv2.rectangle(floor_plan, (300, 150), (350, 350), (0, 0, 0), -1)  # Obstacle 2

        # Draw exits (red)
        cv2.rectangle(floor_plan, (240, 100), (260, 110), (0, 0, 255), -1)  # Exit 1
        cv2.rectangle(floor_plan, (100, 240), (110, 260), (0, 0, 255), -1)  # Exit 2
        
        # Save the floor plan
        cv2.imwrite(args.image_path, floor_plan)
        print(f"Created {args.image_path}")

    # 1. Generate navigation mesh
    nodes, adjacency, exits, connected, unreachable, all_pts, walls = generate_navigation_mesh(
        args.image_path,
        sim_config,
        debug=sim_config.debug_visualization
    )

    if not connected:
        print("Warning: Not all areas are connected to exits!")

    # 2. Create GUI
    gui = GUI(sim_config)

    # 3. Draw navigation mesh and walls
    gui.draw_walls(walls)
    gui.draw_navigation_mesh(nodes, adjacency, all_pts)
    gui.update_gui()

    # 4. Create pedestrians using SFM module
    people_list = PeopleList(sim_config.num_people, nodes, adjacency, exits, walls, all_pts, 
                            sim_config.pixels_to_meters_x, sim_config.pixels_to_meters_y, sfm_config)

    # 5. Initialize pedestrians in GUI
    for person in people_list.list:
        gui.add_oval(person.loc[0], person.loc[1], person.r, person.id)

    # 6. Update people count
    gui.update_people_count(len(people_list.list))
    gui.update_gui()

    # 7. Simulation loop
    time = 0
    while people_list.list:
        # 7.1 Remove old pedestrian positions
        active_people = 0
        for i, person in enumerate(people_list.list):
            if not person.has_exited:
                gui.del_oval(person.id)
                active_people += 1

        # 7.2 Move pedestrians
        people_list.move()

        # 7.3 Draw new pedestrian positions
        for person in people_list.list:
            if not person.has_exited:
                gui.add_oval(person.loc[0], person.loc[1], person.r, person.id)

        # 7.4 Update time and GUI
        time += sfm_config.delta_time
        gui.update_time(str(round(time, 3)))
        gui.update_people_count(active_people)
        gui.update_gui()

        # 7.5 Early exit if all pedestrians have exited
        if active_people == 0:
            break

        # Slow down simulation for visualization
        if time % 0.05 < sfm_config.delta_time:  # Update display roughly every 0.05 seconds
            gui.top.after(1)  # Small delay for visualization

    print(f"Simulation completed in {round(time, 3)} seconds")

     # 8. Save evacuation data to file
    print("Saving evacuation data...")
    people_list.save_evacuation_data()
    print("Evacuation data saved to", people_list.output_path)
    
    # Display saved file information
    output_path = people_list.output_path
    print(f"\nGenerated files:")
    print(f"1. {os.path.join(output_path, 'EvacT.txt')} - Total evacuation time")
    print(f"2. {os.path.join(output_path, 'EvacFlow.csv')} - Evacuation flow data")
    print(f"3. {os.path.join(output_path, 'EvacLocation.csv')} - Pedestrian location data")
    
    # 9. Call data processing functionality - New addition
    print("\nStarting data processing and visualization...")
    try:
        import data_process
        data_process.process_evacuation_data(
            output_path=people_list.output_path,
            image_path=args.image_path
        )
        print("Data processing completed!")
    except Exception as e:
        print(f"Data processing error: {e}")
        import traceback
        traceback.print_exc()
    
        
    
    # Keep original gui.start() unchanged
    gui.start()

# =======================================
# Entry Point
# =======================================
if __name__ == "__main__":
    main()