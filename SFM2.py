import random
import math
import heapq
import os
import csv
import numpy as np
from scipy import ndimage

# =======================================
# Hyperparameters - Social Force Model
# =======================================
class SFMConfig:
    """Social force model hyperparameter configuration class"""
    def __init__(self, 
                 social_force_a1=2000,     # Pedestrian-pedestrian repulsion strength
                 social_force_a2=2000,      # Pedestrian-wall repulsion strength
                 social_force_b=-0.08,      # Repulsion force range coefficient
                 delta_time=0.1,            # Simulation time step
                 people_mass_base=50,       # Pedestrian mass base value (kg)
                 people_mass_variation=20,  # Pedestrian mass variation range (kg)
                 people_radius_base=35,     # Pedestrian radius base value
                 people_radius_variation=5, # Pedestrian radius variation range
                 people_radius_scale=200,   # Pedestrian radius scale factor
                 people_speed_base=60,      # Pedestrian desired speed base value
                 people_speed_variation=20, # Pedestrian desired speed variation range
                 people_speed_scale=100,    # Pedestrian desired speed scale factor
                 relaxation_time=0.5,       # Relaxation time constant
                 max_speed_factor=1.34,      # Maximum speed factor
                 interaction_distance=1.4,  # Pedestrian interaction distance
                 wall_check_distance=20):   # Wall check distance (pixels)
        
        # Social force model core parameters
        self.social_force_a1 = social_force_a1
        self.social_force_a2 = social_force_a2  
        self.social_force_b = social_force_b
        self.delta_time = delta_time
        
        # Pedestrian physical parameters
        self.people_mass_base = people_mass_base
        self.people_mass_variation = people_mass_variation
        self.people_radius_base = people_radius_base
        self.people_radius_variation = people_radius_variation
        self.people_radius_scale = people_radius_scale
        self.people_speed_base = people_speed_base
        self.people_speed_variation = people_speed_variation
        self.people_speed_scale = people_speed_scale
        
        # Dynamics parameters
        self.relaxation_time = relaxation_time
        self.max_speed_factor = max_speed_factor
        self.interaction_distance = interaction_distance
        self.wall_check_distance = wall_check_distance

# Configuration is created and passed by the caller (launch.py) from Agent.py command-line args.
# This script does not hold a default config; it only provides pure functions for external use.

# =======================================
# Coordinate Conversion Functions
# =======================================
def pixel_to_meter(x, y, pixels_to_meters_x, pixels_to_meters_y):
    """Convert pixel coordinates to real-world meter coordinates"""
    return x * pixels_to_meters_x, y * pixels_to_meters_y

def meter_to_pixel(x, y, pixels_to_meters_x, pixels_to_meters_y):
    """Convert real-world meter coordinates to pixel coordinates"""
    return int(x / pixels_to_meters_x), int(y / pixels_to_meters_y)

# =======================================
# Connected Component Analysis Functions
# =======================================
def find_connected_components(walls):
    """
    Find all connected walkable areas.
    Args:
        walls: Wall array, 0 = walkable, nonzero = wall
    Returns:
        labeled_array: Labeled array, each connected component has a unique label
        num_components: Number of connected components
    """
    # Create binary mask of walkable areas (0=walkable, 1=wall -> 1=walkable, 0=wall)
    walkable_mask = (walls == 0).astype(int)
    
    # Use 8-connectivity for connected component analysis
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled_array, num_components = ndimage.label(walkable_mask, structure=structure)
    
    return labeled_array, num_components

def check_component_has_nodes(labeled_array, component_id, nav_graph, pixels_to_meters_x, pixels_to_meters_y):
    """
    Check whether a connected component contains navigation nodes.
    Args:
        labeled_array: Connected component label array
        component_id: Connected component ID
        nav_graph: Navigation graph
        pixels_to_meters_x, pixels_to_meters_y: Pixel-to-meter conversion factors
    Returns:
        bool: Whether the component contains nodes
    """
    for node_id, data in nav_graph.items():
        node_pos_m = data["pos"]  # Node position (meters)
        # Convert to pixel coordinates
        node_x_px, node_y_px = meter_to_pixel(node_pos_m[0], node_pos_m[1], 
                                              pixels_to_meters_x, pixels_to_meters_y)
        
        # Check if node is within image bounds
        if (0 <= node_y_px < labeled_array.shape[0] and 
            0 <= node_x_px < labeled_array.shape[1]):
            # Check if node is in the current connected component
            if labeled_array[node_y_px, node_x_px] == component_id:
                return True
    return False

def get_component_pixels(labeled_array, component_id):
    """
    Get all pixel coordinates of a specified connected component.
    Args:
        labeled_array: Connected component label array
        component_id: Connected component ID
    Returns:
        list: Pixel coordinate list [(y, x), ...]
    """
    pixels = np.argwhere(labeled_array == component_id)
    return [(int(y), int(x)) for y, x in pixels]

def calculate_component_area(labeled_array, component_id):
    """
    Calculate connected component area (pixel count).
    Args:
        labeled_array: Connected component label array
        component_id: Connected component ID
    Returns:
        int: Area (pixel count)
    """
    return np.sum(labeled_array == component_id)

# =======================================
# Path Finding Algorithm
# =======================================
def find_path(nav_graph, adjacency, start_node, exits):
    """Find path from start_node to nearest exit using Dijkstra"""
    distances = {node: float('infinity') for node in adjacency}
    previous = {node: None for node in adjacency}
    distances[start_node] = 0
    pq = [(0, start_node)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        # If we've reached any exit, we're done
        if current_node in exits:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous[current_node]
            return list(reversed(path))

        # If we've processed this node already with a better path, skip
        if current_distance > distances[current_node]:
            continue

        for neighbor in adjacency[current_node]:
            # Calculate Euclidean distance
            pos1 = nav_graph[current_node]["pos"]
            pos2 = nav_graph[neighbor]["pos"]
            distance = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

            # Calculate total distance to reach neighbor
            distance = current_distance + distance

            # If we found a better path, update and add to queue
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    # If no path is found
    return None

# =======================================
# People Class
# =======================================
class People:
    def __init__(self, _id, _loc_x, _loc_y, nav_graph, adjacency, exits, all_pts, config=None):
        self.config = config
        self.id = _id  # Pedestrian ID
        
        # Generate pedestrian attributes using config parameters
        self.m = config.people_mass_base + random.randint(0, config.people_mass_variation)  # Pedestrian mass (kg)
        self.r = (config.people_radius_base + random.randint(0, config.people_radius_variation)) / config.people_radius_scale  # Pedestrian radius (m)
        self.d_v = (config.people_speed_base + random.randint(0, config.people_speed_variation)) / config.people_speed_scale  # Desired speed (m/s)
        
        self.loc = (_loc_x, _loc_y)  # Current position (m)
        self.v = (0, 0)  # Current velocity (m/s)
        self.a = (0, 0)  # Current acceleration (m/s²)
        self.has_exited = False  # Flag: pedestrian has exited
        self.exit_time = None  # Time when pedestrian exited

        # Navigation-related attributes
        self.nav_graph = nav_graph
        self.adjacency = adjacency
        self.exits = exits
        self.all_pts = all_pts

        # Find closest node and exit path
        self.path = []  # Initialize as empty first
        self.update_path()  # This sets current_node and generates the correct path

    def find_closest_node(self):
        """Find the closest navigation node to the pedestrian's current position"""
        min_dist = float('infinity')
        closest_node = None

        for node_id, data in self.nav_graph.items():
            pos = data["pos"]
            dist = ((self.loc[0] - pos[0]) ** 2 + (self.loc[1] - pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_node = node_id

        return closest_node

    def update_path(self):
        """Update path to nearest exit - only when necessary"""
        new_closest_node = self.find_closest_node()
        
        # If current node hasn't changed much, no need to replan entire path
        if (hasattr(self, 'current_node') and 
            self.current_node == new_closest_node and 
            self.path):
            return
        
        self.current_node = new_closest_node
        new_path = find_path(self.nav_graph, self.adjacency, self.current_node, self.exits)
        
        if new_path and len(new_path) > 1:
            # Remove first node in path (current closest node), start from next node
            self.path = new_path[1:]
        else:
            self.path = []

    def get_desired_direction(self):
        """Get desired direction vector based on path"""
        if not self.path:
            self.update_path()
            if not self.path:
                return (0, 0)  # No valid path

        # Use the first node in the path as the target (the node to head toward currently)
        target_node = self.path[0]

        # Calculate direction vector to target node
        current_pos = self.loc
        target_pos = self.nav_graph[target_node]["pos"]

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]

        # Normalize if not zero
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > 0:
            dx /= dist
            dy /= dist

        return (dx, dy)

    def check_reached_node(self):
        """Check if the pedestrian has reached the current target node"""
        if not self.path:
            return False

        # Check if reached the first node in the path (current target)
        target_node = self.path[0]
        target_pos = self.nav_graph[target_node]["pos"]
        current_pos = self.loc

        # Check if we're close enough to this node
        dist = ((target_pos[0] - current_pos[0]) ** 2 + (target_pos[1] - current_pos[1]) ** 2) ** 0.5

        if dist < self.r * 2:  # If within 2 radii of the node
            # Remove reached node
            self.path.pop(0)
            return True

        return False

    def check_exit_reached(self, current_time):
        """Check if pedestrian has reached an exit"""
        if self.current_node in self.exits:
            self.has_exited = True
            self.exit_time = current_time
            return True
        return False

# =======================================
# PeopleList Class
# =======================================
class PeopleList:
    def __init__(self, num_people, nav_graph, adjacency, exits, walls, all_pts, 
                 pixels_to_meters_x, pixels_to_meters_y, config=None):
        self.config = config
        self.list = []
        self.nav_graph = nav_graph
        self.adjacency = adjacency
        self.exits = exits
        self.walls = walls
        self.all_pts = all_pts
        self.pixels_to_meters_x = pixels_to_meters_x
        self.pixels_to_meters_y = pixels_to_meters_y
        
        # Data collection variables
        self.current_time = 0.0
        self.evacuation_flow_data = []  # Store evacuation flow data
        self.evacuation_location_data = []  # Store location data
        self.last_record_time = -1  # Last recorded time point

        # Output path (relative to this file's directory, portable across machines)
        _here = os.path.dirname(os.path.abspath(__file__))
        self.output_path = os.path.join(_here, "Material", "data")

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

        # =======================================
        # New: Connected component analysis & intelligent pedestrian allocation
        # =======================================
        
        print("Starting connected component analysis...")
        
        # 1. Perform connected component analysis
        labeled_array, num_components = find_connected_components(self.walls)
        print(f"Found {num_components} connected components")
        
        # 2. Find connected components containing nodes
        valid_components = []
        for component_id in range(1, num_components + 1):  # Component IDs start from 1
            if check_component_has_nodes(labeled_array, component_id, nav_graph, 
                                       pixels_to_meters_x, pixels_to_meters_y):
                area = calculate_component_area(labeled_array, component_id)
                valid_components.append({
                    'id': component_id,
                    'area': area,
                    'pixels': get_component_pixels(labeled_array, component_id)
                })
                print(f"Component {component_id}: area {area} pixels, contains navigation nodes")
            else:
                print(f"Component {component_id}: no navigation nodes, discarded")
        
        if not valid_components:
            raise ValueError("No valid connected components containing navigation nodes found!")
        
        # 3. Calculate total area and area proportion of each component
        total_area = sum(comp['area'] for comp in valid_components)
        print(f"Total valid component area: {total_area} pixels")
        
        # 4. Allocate pedestrians based on area proportion
        people_allocation = []
        allocated_people = 0
        
        for i, comp in enumerate(valid_components):
            if i == len(valid_components) - 1:  # Last component gets all remaining pedestrians
                people_count = num_people - allocated_people
            else:
                people_count = int(num_people * comp['area'] / total_area)
            
            people_allocation.append({
                'component': comp,
                'people_count': people_count
            })
            allocated_people += people_count
            
            print(f"Component {comp['id']}: allocated {people_count} pedestrians "
                  f"(area proportion: {comp['area']/total_area:.2%})")
        
        # 5. Generate pedestrians within each component
        people_id = 0
        for allocation in people_allocation:
            comp = allocation['component']
            people_count = allocation['people_count']
            available_pixels = comp['pixels']
            
            if people_count == 0:
                continue
                
            # Randomly generate specified number of pedestrians within current component
            for _ in range(people_count):
                if available_pixels:
                    # Randomly select a pixel position
                    idx = random.randint(0, len(available_pixels) - 1)
                    y, x = available_pixels[idx]
                    
                    # Convert to meter coordinates
                    x_m, y_m = pixel_to_meter(x, y, pixels_to_meters_x, pixels_to_meters_y)
                    
                    # Create pedestrian
                    self.list.append(People(f"p{people_id}", x_m, y_m, nav_graph, 
                                          adjacency, exits, all_pts, config))
                    people_id += 1
        
        print(f"Successfully created {len(self.list)} pedestrians")

    def record_data(self):
        """Record data at the current time point"""
        current_time_int = int(self.current_time)

        # Only record data at integer seconds
        if current_time_int > self.last_record_time:
            self.last_record_time = current_time_int

            # Count evacuated people
            evacuated_count = sum(1 for person in self.list if person.has_exited)

            # Record evacuation flow data
            self.evacuation_flow_data.append([current_time_int, evacuated_count])

            # Total image height (meters), used for Y-direction mirror flip
            image_height_m = self.walls.shape[0] * self.pixels_to_meters_y

            # Record positions of pedestrians still in the scene (origin at bottom-left)
            for person in self.list:
                if not person.has_exited:
                    flipped_y = image_height_m - person.loc[1]
                    self.evacuation_location_data.append([current_time_int, person.loc[0], flipped_y])
                
    def move(self):
        """Update positions of all pedestrians using the social force model"""
        # Record data
        self.record_data()
        
        # A. Calculate acceleration for each pedestrian
        for i in range(len(self.list)):
            person = self.list[i]

            # Skip if already exited
            if person.has_exited:
                continue

            # Check if reached current target node
            person.check_reached_node()

            # Check if reached exit
            if person.check_exit_reached(self.current_time):
                continue

            # 1. Calculate desired force - pointing toward next waypoint
            direction = person.get_desired_direction()
            desired_v = (person.d_v * direction[0], person.d_v * direction[1])

            # 2. Calculate pedestrian-pedestrian interaction forces
            sum_of_fij = (0, 0)
            for j in range(len(self.list)):
                if i == j or self.list[j].has_exited:
                    continue

                other = self.list[j]
                dx = person.loc[0] - other.loc[0]
                dy = person.loc[1] - other.loc[1]
                d = (dx ** 2 + dy ** 2) ** 0.5

                # Skip if too far away (using config parameters)
                if d >= self.config.interaction_distance:
                    continue

                # Prevent division by zero
                if d == 0:
                    d = 0.001

                # Calculate repulsive force (using config parameters)
                fij = self.config.social_force_a1 * 0.025 * math.exp((d - person.r - other.r) / self.config.social_force_b)

                # Add to total force
                sum_of_fij = (
                    sum_of_fij[0] + fij * dx / d,
                    sum_of_fij[1] + fij * dy / d
                )

            # 3. Calculate wall repulsion forces
            sum_of_fiw = (0, 0)

            # Convert pedestrian position to pixels for wall checks
            px, py = meter_to_pixel(person.loc[0], person.loc[1], self.pixels_to_meters_x, self.pixels_to_meters_y)

            # Check walls in 8 directions (using config parameters)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                for dist in range(1, self.config.wall_check_distance):  # Check up to wall_check_distance pixels away
                    nx, ny = px + dx * dist, py + dy * dist

                    # Skip if out of bounds
                    if nx < 0 or ny < 0 or nx >= self.walls.shape[1] or ny >= self.walls.shape[0]:
                        continue

                    # Check if this pixel is a wall
                    if self.walls[ny, nx] > 0:
                        # Calculate distance in meters
                        wall_distance = dist * self.pixels_to_meters_x

                        # Calculate repulsion force (using config parameters)
                        fiw = self.config.social_force_a2 * 0 * math.exp((wall_distance - person.r) / self.config.social_force_b)

                        # Add to total wall force
                        sum_of_fiw = (
                            sum_of_fiw[0] - fiw * dx / dist,
                            sum_of_fiw[1] - fiw * dy / dist
                        )
                        break

            # 4. Calculate total acceleration
            # Driving force term (using config parameters)
            driving_x = person.m * (desired_v[0] - person.v[0]) / self.config.relaxation_time
            driving_y = person.m * (desired_v[1] - person.v[1]) / self.config.relaxation_time

            # Calculate final acceleration
            ax = (driving_x + sum_of_fij[0] + sum_of_fiw[0]) / person.m
            ay = (driving_y + sum_of_fij[1] + sum_of_fiw[1]) / person.m

            # Update acceleration
            self.list[i].a = (ax, ay)

        # B. Update velocities and positions
        for i in range(len(self.list)):
            person = self.list[i]

            # Skip if already exited
            if person.has_exited:
                continue

            # Update velocity (using config parameters)
            vx = person.v[0] + person.a[0] * self.config.delta_time
            vy = person.v[1] + person.a[1] * self.config.delta_time

            # Limit speed to max desired velocity (using config parameters)
            speed = (vx ** 2 + vy ** 2) ** 0.5
            max_speed = person.d_v * self.config.max_speed_factor
            if speed > max_speed:
                vx = vx * max_speed / speed
                vy = vy * max_speed / speed

            person.v = (vx, vy)

            # Update position (using config parameters)
            new_x = person.loc[0] + vx * self.config.delta_time
            new_y = person.loc[1] + vy * self.config.delta_time

            person.loc = (new_x, new_y)
        
        # Update time
        self.current_time += self.config.delta_time

    def is_evacuation_complete(self):
        """Check if evacuation is complete"""
        return all(person.has_exited for person in self.list)

    def get_total_evacuation_time(self):
        """Get total evacuation time"""
        if not self.is_evacuation_complete():
            return None
        return max(person.exit_time for person in self.list if person.exit_time is not None)

    def save_evacuation_data(self):
        """Save evacuation data to files"""
        # 1. Save total evacuation time to EvacT.txt
        total_time = self.get_total_evacuation_time()
        if total_time is not None:
            evac_t_path = os.path.join(self.output_path, "EvacT.txt")
            with open(evac_t_path, 'w') as f:
                f.write(str(total_time))
        
        # 2. Save evacuation flow data to EvacFlow.csv
        evac_flow_path = os.path.join(self.output_path, "EvacFlow.csv")
        with open(evac_flow_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['T', 'N'])  # Write header row
            for row in self.evacuation_flow_data:
                writer.writerow(row)
        
        # 3. Save location data to EvacLocation.csv
        evac_location_path = os.path.join(self.output_path, "EvacLocation.csv")
        with open(evac_location_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['T', 'x', 'y'])  # Write header row
            for row in self.evacuation_location_data:
                writer.writerow(row)