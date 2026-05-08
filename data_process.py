##### Hollow-Out Processing #####
import os
from PIL import Image


def process_image(image_path, output_path, tolerance=0, threshold=128):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist. Creating: {output_dir}")
        os.makedirs(output_dir)

    img = Image.open(image_path).convert("RGBA")
    pixels = img.load()

    width, height = img.size

    # Preprocessing: binarize the image to pure black & white, but preserve red pixels
    print("Performing binarization (preserving red pixels)...")
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            # Check if pixel is red
            if r - g > tolerance and r - b > tolerance:
                # Keep red pixels unchanged
                continue

            # Compute grayscale value (using standard weights)
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)

            # Binarization: set non-red pixels to pure black or pure white based on threshold
            if gray > threshold:
                pixels[x, y] = (255, 255, 255, a)  # White
            else:
                pixels[x, y] = (0, 0, 0, a)  # Black

    print("Binarization complete, proceeding to next steps...")

    # Step 1: Skip red pixel processing, keep red unchanged
    # Step 2: Invert black & white (only process black/white pixels, skip red)    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]

            # Skip red pixels, keep red unchanged
            if r - g > tolerance and r - b > tolerance:
                pixels[x, y] = (255, 255, 255, a)
            else:
                # Step 2: Black & white inversion (non-red pixels)
                if (r, g, b) == (0, 0, 0):
                    pixels[x, y] = (255, 255, 255, a)
                elif (r, g, b) == (255, 255, 255):
                    pixels[x, y] = (0, 0, 0, a)

    # Step 3: Hollow out white pixels (set to transparent)
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if (r, g, b) == (255, 255, 255):
                pixels[x, y] = (255, 255, 255, 0)  # Set transparent

    img.save(output_path, "PNG")
    print(f"Processing complete, saved to {output_path}")


######### Heatmap Generation #####
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Global variable - max density value
DENSITY_MAX_VALUE = 7  # Modify this value to change the system-wide max density


def create_custom_colormap():
    """Create a rainbow colormap similar to Pathfinder, with 0 mapped to white"""
    colors = [
        '#FFFFFF',  # White (0) - zero density areas
        '#FFFFFF',  # White (0) - zero density areas
        '#0000FF',  # Blue
        '#00FFFF',  # Cyan
        '#00FFFF',  # Cyan
        '#00FF00',  # Green
        '#FFFF00',  # Yellow
        '#FFA500',  # Orange
        '#FF0000',  # Red
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('pathfinder_rainbow', colors, N=n_bins)
    return cmap


def load_boundary_mask(mask_path, grid_size=30, spatial_extent=15.0):
    """
    Load boundary mask from processed.png; black areas are walls

    Args:
    - mask_path: Path to processed.png
    - grid_size: Grid size
    - spatial_extent: Physical space width (m), used to determine physical size per grid cell
    """
    try:
        # Read 512x512 boundary image
        img = Image.open(mask_path).convert('RGB')
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.NEAREST)

        img_array = np.array(img)

        # Detect black areas (regions where all RGB values are very low)
        black_threshold = 30
        is_black = ((img_array[:, :, 0] < black_threshold) &
                    (img_array[:, :, 1] < black_threshold) &
                    (img_array[:, :, 2] < black_threshold))

        # Downsample 512x512 to grid_size x grid_size grid
        pixels_per_grid = 512 // grid_size
        boundary_mask = np.zeros((grid_size, grid_size), dtype=bool)

        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * pixels_per_grid
                y_end = min((i + 1) * pixels_per_grid, 512)
                x_start = j * pixels_per_grid
                x_end = min((j + 1) * pixels_per_grid, 512)

                grid_region = is_black[y_start:y_end, x_start:x_end]
                black_ratio = np.sum(grid_region) / grid_region.size

                # If black ratio < 50%, consider this grid cell walkable
                boundary_mask[i, j] = (black_ratio < 0.5)

        # Flip Y axis to match coordinate system (row=0 -> image bottom = CSV y=0)
        boundary_mask = np.flipud(boundary_mask)

        walkable_grids = np.sum(boundary_mask)
        total_grids = grid_size * grid_size
        print(f"Boundary mask loaded: walkable grids {walkable_grids}/{total_grids} ({walkable_grids / total_grids * 100:.1f}%)")

        return boundary_mask, img_array

    except Exception as e:
        print(f"Warning: could not load boundary mask ({e}), using fully open space")
        return np.ones((grid_size, grid_size), dtype=bool), None


def calculate_pathfinder_density(x_coords, y_coords, boundary_mask, grid_size=30, spatial_extent=15.0,
                                 influence_radius=0.8):
    """
    Calculate grid-based density using the Pathfinder method

    Args:
    - x_coords, y_coords: Pedestrian coordinates
    - boundary_mask: Boundary mask (30x30)
    - grid_size: Grid dimensions (30)
    - spatial_extent: Spatial extent (15m)
    - influence_radius: Influence radius (0.8m)
    """
    if len(x_coords) == 0:
        return np.zeros((grid_size, grid_size))

    grid_step = spatial_extent / grid_size  # 0.5 m per grid
    density_grid = np.zeros((grid_size, grid_size))

    print(f"  Computing grid density for {len(x_coords)} pedestrians using Pathfinder method...")

    # Compute influence for each pedestrian
    for person_idx, (px, py) in enumerate(zip(x_coords, y_coords)):
        if not (0 <= px <= spatial_extent and 0 <= py <= spatial_extent):
            continue

        # Determine which grid cell the pedestrian is in
        grid_x = int(px / grid_step)
        grid_y = int(py / grid_step)
        grid_x = min(grid_x, grid_size - 1)
        grid_y = min(grid_y, grid_size - 1)

        # Calculate influence range (in grid cells)
        influence_grids = int(np.ceil(influence_radius / grid_step)) + 1  # ~2 grid cells

        # Calculate contribution for each grid cell within influence range
        for dy in range(-influence_grids, influence_grids + 1):
            for dx in range(-influence_grids, influence_grids + 1):
                target_y = grid_y + dy
                target_x = grid_x + dx

                # Check bounds
                if not (0 <= target_y < grid_size and 0 <= target_x < grid_size):
                    continue

                # Check whether this grid cell is walkable
                if not boundary_mask[target_y, target_x]:
                    continue

                # Calculate distance from grid cell center to pedestrian
                grid_center_x = (target_x + 0.5) * grid_step
                grid_center_y = (target_y + 0.5) * grid_step
                distance = np.sqrt((grid_center_x - px) ** 2 + (grid_center_y - py) ** 2)

                if distance <= influence_radius:
                    # Calculate available space ratio at this location
                    available_space = calculate_available_space(
                        target_x, target_y, boundary_mask, influence_grids)

                    # Distance-based contribution (Gaussian decay)
                    sigma = influence_radius * 0.7
                    base_contribution = np.exp(-(distance ** 2) / (2 * sigma ** 2))

                    # Adjust density based on available space
                    space_factor = 1.0 / max(available_space, 0.2)

                    # Apply contribution
                    density_grid[target_y, target_x] += base_contribution * space_factor

    return density_grid


def calculate_available_space(grid_x, grid_y, boundary_mask, check_radius):
    """
    Calculate available space ratio around a grid cell.
    Used to simulate the effect of walls on density.
    """
    grid_size = boundary_mask.shape[0]
    total_checked = 0
    available_count = 0

    for dy in range(-check_radius, check_radius + 1):
        for dx in range(-check_radius, check_radius + 1):
            check_y = grid_y + dy
            check_x = grid_x + dx

            if 0 <= check_y < grid_size and 0 <= check_x < grid_size:
                total_checked += 1
                if boundary_mask[check_y, check_x]:
                    available_count += 1

    if total_checked == 0:
        return 1.0

    return available_count / total_checked


def create_high_resolution_contour_data(density_grid, boundary_mask, grid_size=30, spatial_extent=15.0,
                                        target_resolution=200):
    """
    Create high-resolution contour data.
    Uses a finer grid to generate smooth contours.
    """
    # Create original grid coordinates
    grid_step = spatial_extent / grid_size
    x_orig = np.linspace(grid_step / 2, spatial_extent - grid_step / 2, grid_size)
    y_orig = np.linspace(grid_step / 2, spatial_extent - grid_step / 2, grid_size)
    X_orig, Y_orig = np.meshgrid(x_orig, y_orig)

    # Only use walkable area data points for interpolation
    valid_mask = boundary_mask.astype(bool)
    x_valid = X_orig[valid_mask]
    y_valid = Y_orig[valid_mask]
    z_valid = density_grid[valid_mask]

    # If no valid data points, return empty grid
    if len(x_valid) == 0:
        x_high = np.linspace(0, spatial_extent, target_resolution)
        y_high = np.linspace(0, spatial_extent, target_resolution)
        X_high, Y_high = np.meshgrid(x_high, y_high)
        Z_high = np.zeros_like(X_high)
        return X_high, Y_high, Z_high

    # Create high-resolution grid
    x_high = np.linspace(0, spatial_extent, target_resolution)
    y_high = np.linspace(0, spatial_extent, target_resolution)
    X_high, Y_high = np.meshgrid(x_high, y_high)

    # Use cubic interpolation
    points = np.column_stack((x_valid, y_valid))
    xi = np.column_stack((X_high.ravel(), Y_high.ravel()))

    # Use cubic interpolation for smooth results
    Z_interp = griddata(points, z_valid, xi, method='cubic', fill_value=0)
    Z_high = Z_interp.reshape(X_high.shape)

    # Apply boundary mask to high-resolution grid
    # Create high-resolution boundary mask
    boundary_high = np.zeros_like(X_high, dtype=bool)
    for i in range(target_resolution):
        for j in range(target_resolution):
            x_pos = X_high[i, j]
            y_pos = Y_high[i, j]

            # Find corresponding original grid cell
            grid_x = int(x_pos / grid_step)
            grid_y = int(y_pos / grid_step)
            grid_x = min(max(grid_x, 0), grid_size - 1)
            grid_y = min(max(grid_y, 0), grid_size - 1)

            boundary_high[i, j] = boundary_mask[grid_y, grid_x]

    # Set areas outside boundary to 0
    Z_high[~boundary_high] = 0

    # Apply light smoothing for better contours
    Z_high = gaussian_filter(Z_high, sigma=1.0)

    return X_high, Y_high, Z_high


def create_contour_visualization(density_grid, boundary_mask, grid_size=30, spatial_extent=15.0,
                                  vmax=None):
    """
    Create Tecplot-style contour heatmap.
    vmax: Color scale upper bound (people/m²); falls back to global DENSITY_MAX_VALUE when None.
    """
    if vmax is None or vmax <= 0:
        vmax = DENSITY_MAX_VALUE

    # Generate high-resolution interpolated data
    X, Y, Z = create_high_resolution_contour_data(density_grid, boundary_mask, grid_size, spatial_extent)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')

    # Create colormap
    custom_cmap = create_custom_colormap()

    # Adaptively divide levels based on vmax (20 equal intervals)
    max_density = np.max(Z)
    if max_density > 0:
        levels = np.linspace(0, vmax, 20)
        # Keep levels not exceeding actual peak value; keep at least first 3
        kept = levels[levels <= max(max_density, vmax * 0.15)]
        if len(kept) < 3:
            kept = levels[:3]
        levels = np.unique(kept)
    else:
        levels = np.linspace(0, vmax, 20)

    # Draw filled contours
    contour_filled = ax.contourf(X, Y, Z,
                                 levels=levels,
                                 cmap=custom_cmap,
                                 vmin=0,
                                 vmax=vmax,
                                 extend='max')

    # Contour boundary lines
    contour_lines = ax.contour(X, Y, Z,
                               levels=levels[::2] if len(levels) >= 2 else levels,
                               colors='black',
                               linewidths=0.3,
                               alpha=0.3)

    # Set axes
    ax.set_xlim(0, spatial_extent)
    ax.set_ylim(0, spatial_extent)
    ax.set_aspect('equal')
    ax.axis('off')

    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig, ax


def generate_pathfinder_contour_heatmaps(csv_path, mask_path, output_dir, vmax=None,
                                          spatial_extent=None):
    """
    Generate Pathfinder/Tecplot-style contour heatmaps.
    vmax: Adaptive color scale upper bound. Auto-determined from 95th percentile of all timestep densities when None.
    spatial_extent: Physical space width (m). Auto-inferred from CSV coordinate range when None.
    Returns the final vmax used (for caller to generate corresponding colorbar).
    """
    # Read data
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)

    # Adaptive spatial extent: round up CSV coordinate max to next 5m multiple, minimum 15m
    # Note: must cover entire physical space, not just pedestrian activity range
    if spatial_extent is None or spatial_extent <= 0:
        max_xy = max(float(df['x'].max()), float(df['y'].max()))
        spatial_extent = max(15.0, math.ceil(max_xy / 5.0) * 5.0)
        print(f"  Adaptive spatial_extent = {spatial_extent} m "
              f"(data range x:{df['x'].min():.1f}..{df['x'].max():.1f}, "
              f"y:{df['y'].min():.1f}..{df['y'].max():.1f})")

    # Load boundary mask — pass same spatial_extent to ensure consistent physical scale
    print("Loading boundary mask...")
    boundary_mask, _ = load_boundary_mask(mask_path, spatial_extent=spatial_extent)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get time steps
    time_steps = sorted(df['T'].unique())
    print(f"Found {len(time_steps)} time steps")

    # Pass 1: adaptive vmax estimation
    if vmax is None or vmax <= 0:
        peak_densities = []
        for t in time_steps:
            cur = df[df['T'] == t]
            if len(cur) == 0:
                continue
            grid = calculate_pathfinder_density(cur['x'].values, cur['y'].values,
                                                 boundary_mask, spatial_extent=spatial_extent)
            if grid.size > 0 and grid.max() > 0:
                peak_densities.append(float(grid.max()))
        if peak_densities:
            arr = np.asarray(peak_densities)
            # 95th percentile + 10% margin, min 0.5 people/m², max 7 people/m²
            vmax = float(np.percentile(arr, 95)) * 1.1
            vmax = max(0.5, min(vmax, DENSITY_MAX_VALUE))
            print(f"  Adaptive vmax = {vmax:.2f} people/m² (peak 95% = {np.percentile(arr,95):.2f}, "
                  f"max={arr.max():.2f})")
        else:
            vmax = 1.0
            print(f"  No density data, using vmax={vmax}")

    # Image parameters
    dpi = 100

    print("Generating Pathfinder/Tecplot-style contour heatmaps...")

    for i, t in enumerate(time_steps):
        print(f"\nProcessing time step {t} ({i + 1}/{len(time_steps)})")

        # Get current time step data
        current_data = df[df['T'] == t]
        x_coords = current_data['x'].values
        y_coords = current_data['y'].values

        print(f"  Pedestrian count: {len(x_coords)}")

        # Compute grid density
        density_grid = calculate_pathfinder_density(x_coords, y_coords, boundary_mask,
                                                     spatial_extent=spatial_extent)

        # Create contour visualization
        fig, ax = create_contour_visualization(density_grid, boundary_mask,
                                                spatial_extent=spatial_extent, vmax=vmax)

        # Save
        filename = f"{t}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath,
                    dpi=dpi,
                    bbox_inches='tight',
                    pad_inches=0,
                    facecolor='white',
                    edgecolor='none',
                    transparent=False)

        plt.close()
        print(f"  Saved: {filename}")

    print(f"\nAll contour heatmaps generated! Saved in: {output_dir}")
    return vmax


def generate_dynamic_colorbar(vmax, save_path, label="Density (people/m²)"):
    """Dynamically generate a colorbar PNG matching the heatmap, based on vmax."""
    cmap = create_custom_colormap()
    fig, ax = plt.subplots(figsize=(1.6, 8), facecolor='white')
    fig.subplots_adjust(left=0.25, right=0.55, top=0.95, bottom=0.05)
    norm = plt.Normalize(vmin=0, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='vertical', extend='max')
    cb.set_label(label, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Dynamic colorbar saved: {save_path}  (vmax={vmax:.2f})")


################ Overlay Processing ###########
def overlay_image_on_folder(base_folder, overlay_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    overlay_img = Image.open(overlay_path).convert("RGBA")

    for filename in os.listdir(base_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_path = os.path.join(base_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                base_img = Image.open(base_path).convert("RGBA")

                # Resize overlay to match base image size if needed
                if overlay_img.size != base_img.size:
                    resized_overlay = overlay_img.resize(base_img.size, Image.Resampling.LANCZOS)
                else:
                    resized_overlay = overlay_img

                # Composite overlay onto base image
                combined = Image.alpha_composite(base_img, resized_overlay)

                # Determine format and convert if needed
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    combined.convert("RGB").save(output_path, "JPEG")
                else:
                    combined.save(output_path, "PNG")

                print(f"Saved overlaid image to: {output_path}")

            except Exception as e:
                print(f"Failed to process {base_path}: {e}")


########### Video Generation #################
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re


def generate_video(input_dir, output_path, colorbar_path=None, frame_duration=0.5):
    """Generate evacuation density video"""
    
    # Video parameters
    fps = int(1 / frame_duration)

    # Get all image files and sort in "natural order"
    def natural_key(filename):
        # Extract digit sequences as key; keep non-digit parts as-is
        parts = re.split(r'(\d+)', filename)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))
                   and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    image_files.sort(key=natural_key)

    if not image_files:
        raise RuntimeError(f"No images found in {input_dir}")

    # Read first image to determine dimensions
    first_img = Image.open(os.path.join(input_dir, image_files[0]))
    w, h = first_img.size

    # Process colorbar
    if colorbar_path and os.path.exists(colorbar_path):
        # Load colorbar image
        colorbar_img = Image.open(colorbar_path).convert("RGBA")
        # Calculate colorbar new size (scale proportionally to original image height)
        colorbar_aspect = colorbar_img.width / colorbar_img.height
        new_colorbar_height = h
        new_colorbar_width = int(new_colorbar_height * colorbar_aspect)
        colorbar_img = colorbar_img.resize((new_colorbar_width, new_colorbar_height), Image.LANCZOS)
    else:
        # No colorbar, create blank area
        new_colorbar_width = 100
        new_colorbar_height = h
        colorbar_img = Image.new("RGBA", (new_colorbar_width, new_colorbar_height), (255, 255, 255, 255))

    # Reserve extra height for text (e.g., 50 pixels)
    text_bar_height = 50
    # Calculate total width (original image width + left white area + right colorbar width)
    total_width = w + new_colorbar_width * 2
    # Calculate total height (original image height + top text bar + bottom white area)
    total_height = h + text_bar_height * 2
    video_size = (total_width, total_height)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, video_size)

    # Font settings
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except IOError:
        # Fall back to default font
        font = ImageFont.load_default()

    for img_name in image_files:
        # Open image
        img = Image.open(os.path.join(input_dir, img_name)).convert("RGB")

        # Create a new white-background canvas for placing original image, colorbar, and text bar
        canvas = Image.new("RGB", video_size, color=(255, 255, 255))

        # Paste original image in the center
        canvas.paste(img, (new_colorbar_width, text_bar_height))

        # Paste colorbar on the right (right of original image)
        canvas.paste(colorbar_img, (new_colorbar_width + w, text_bar_height),
                     colorbar_img if colorbar_img.mode == 'RGBA' else None)

        # Draw text in the top white bar
        draw = ImageDraw.Draw(canvas)
        # Extract filename without extension as seconds value
        n = os.path.splitext(img_name)[0]
        text = f"Evac_T= {n} s"

        # Use textbbox to calculate text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # Calculate text position, centered in top bar
        x = (total_width - text_w) // 2
        y = (text_bar_height - text_h) // 2  # Center within top white area
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

        # Convert to OpenCV BGR format and write to video
        frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    # Release resources
    video_writer.release()
    print(f"Video saved: {output_path}")


################# Congestion Analysis ###################
def create_congestion_time_colormap():
    """Create white to orange-red colormap"""
    colors = [
        '#FFFFFF',  # White (0) - No congestion
        '#FFF8DC',  # Light yellow
        '#FFE4B5',  # Light orange
        '#FFCC99',  # Orange
        '#FF9966',  # Deep orange
        '#FF6633',  # Red-orange
        '#FF3300',  # Red
        '#CC0000',  # Deep red
        '#990000'  # Dark red (Maximum congestion time)
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('congestion_time', colors, N=n_bins)
    return cmap


def calculate_congestion_time_map(csv_path, mask_path, congestion_threshold=None,
                                   output_path=None, vmax=None, spatial_extent=None):
    """
    Compute and generate congestion heatmap.

    Congestion definition: cumulative pedestrian throughput per grid cell over the entire evacuation (person-steps).
    This is better suited for sparse scenarios than instantaneous density thresholds — exit corridors and bottlenecks naturally accumulate more person-steps.
    The density_threshold parameter is retained for reference only but no longer used for filtering.
    """
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)

    # Adaptive spatial extent (consistent with generate_pathfinder_contour_heatmaps)
    if spatial_extent is None or spatial_extent <= 0:
        max_xy = max(float(df['x'].max()), float(df['y'].max()))
        spatial_extent = max(15.0, math.ceil(max_xy / 5.0) * 5.0)
    print(f"spatial_extent = {spatial_extent} m")

    print("Loading boundary mask...")
    boundary_mask, boundary_image = load_boundary_mask(mask_path, spatial_extent=spatial_extent)

    time_steps = sorted(df['T'].unique())
    print(f"Found {len(time_steps)} time steps")

    grid_size = 30
    grid_step = spatial_extent / grid_size

    # Cumulative pedestrian throughput: each pedestrian contributes 1 to their grid cell per time step
    cumulative_grid = np.zeros((grid_size, grid_size))

    print("Starting congestion analysis (cumulative pedestrian count)...")
    for i, t in enumerate(time_steps):
        print(f"\rProcessing time step {t} ({i + 1}/{len(time_steps)})", end="", flush=True)
        cur = df[df['T'] == t]
        for px, py in zip(cur['x'].values, cur['y'].values):
            if not (0 <= px <= spatial_extent and 0 <= py <= spatial_extent):
                continue
            gx = min(int(px / grid_step), grid_size - 1)
            gy = min(int(py / grid_step), grid_size - 1)
            cumulative_grid[gy, gx] += 1

    print(f"\nCongestion analysis completed!")

    # Only display walkable areas
    cumulative_grid = cumulative_grid * boundary_mask

    max_val = float(np.max(cumulative_grid))
    nonzero = int(np.sum(cumulative_grid > 0))
    total_walkable = int(np.sum(boundary_mask))
    print(f"Maximum cumulative count: {max_val:.0f} person-steps")
    print(f"Active grids: {nonzero}/{total_walkable}")

    if vmax is not None and vmax > 0:
        vmax_value = float(vmax)
    elif max_val > 0:
        vmax_value = max_val
        print(f"Using auto-adaptive vmax: {vmax_value:.1f}")
    else:
        vmax_value = 1.0
        print("No congestion detected, using default scale")

    # Smooth display
    display_image = create_smooth_visualization(cumulative_grid)
    congestion_cmap = create_congestion_time_colormap()

    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')

    im = ax.imshow(display_image,
                   extent=[0, spatial_extent, 0, spatial_extent],
                   cmap=congestion_cmap,
                   vmin=0,
                   vmax=vmax_value,
                   interpolation='bilinear',
                   aspect='equal',
                   origin='lower')

    # Overlay boundary image
    if boundary_image is not None:
        boundary_rgba = np.zeros((512, 512, 4))
        boundary_rgba[:, :, :3] = boundary_image / 255.0
        black_threshold = 30
        is_black = ((boundary_image[:, :, 0] < black_threshold) &
                    (boundary_image[:, :, 1] < black_threshold) &
                    (boundary_image[:, :, 2] < black_threshold))
        boundary_rgba[:, :, 3] = is_black.astype(float)
        boundary_rgba = np.flipud(boundary_rgba)
        ax.imshow(boundary_rgba,
                  extent=[0, spatial_extent, 0, spatial_extent],
                  aspect='equal',
                  origin='lower')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Congestion Time (time steps)', fontsize=12)

    # Set display
    ax.set_xlim(0, spatial_extent)
    ax.set_ylim(0, spatial_extent)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Congestion Heatmap\n(Cumulative pedestrian count, max={max_val:.0f})', fontsize=14)

    # Save image
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Congestion time heatmap saved: {output_path}")

    return congestion_time_grid


def create_smooth_visualization(congestion_grid, grid_size=30, display_size=512):
    """
    Convert 30x30 congestion time grid to 512x512 smooth display image
    """
    # Light smoothing to eliminate abrupt grid boundary transitions
    smoothed_grid = gaussian_filter(congestion_grid, sigma=0.8)

    # Use high-quality interpolation to upscale to target size
    from scipy.ndimage import zoom
    zoom_factor = display_size / grid_size
    display_image = zoom(smoothed_grid, zoom_factor, order=3)

    # Ensure correct dimensions
    if display_image.shape[0] != display_size:
        display_image = display_image[:display_size, :display_size]

    return display_image


def process_evacuation_data(output_path, image_path):
    """
    Main data processing function integrating all processing steps

    Args:
    - output_path: Evacuation data output path
    - image_path: Original image path
    """
    print("Starting data processing...")
    
    # Set file paths
    csv_file_path = os.path.join(output_path, "EvacLocation.csv")
    process_png_path = os.path.join(output_path, "Process.png")
    heatmap_dir = os.path.join(output_path, "heatmap")
    output_dir = os.path.join(output_path, "Output")
    video_path = os.path.join(output_path, "Density.mp4")
    congestion_path = os.path.join(output_path, "congestion.png")
    
    # Check required files exist
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found {csv_file_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found {image_path}")
        return
    
    try:
        # 1. Process image (hollow out)
        print("1. Processing image...")
        process_image(image_path, process_png_path)
        
        # 2. Generate heatmaps (adaptive vmax)
        print("2. Generating heatmaps...")
        adaptive_vmax = generate_pathfinder_contour_heatmaps(csv_file_path, process_png_path, heatmap_dir)

        # 2.5 Generate dynamic colorbar consistent with heatmaps using adaptive vmax
        dynamic_cb_path = os.path.join(output_path, "Colorbar.png")
        try:
            generate_dynamic_colorbar(adaptive_vmax, dynamic_cb_path)
        except Exception as e:
            print(f"  Dynamic colorbar generation failed (will fall back to pre-made): {e}")
        
        # 3. Overlay processing
        print("3. Overlay processing...")
        overlay_image_on_folder(heatmap_dir, process_png_path, output_dir)
        
        # 4. Generate video
        print("4. Generating video...")
        # Find colorbar file (prefer current adaptive one; fall back to pre-made)
        colorbar_paths = [
            os.path.join(output_path, "Colorbar.png"),
            os.path.join(os.path.dirname(__file__), "Colorbar.png"),
            os.path.join(os.path.dirname(image_path), "Colorbar.png")
        ]
        colorbar_path = None
        for path in colorbar_paths:
            if os.path.exists(path):
                colorbar_path = path
                break
        
        generate_video(output_dir, video_path, colorbar_path)
        
        # 5. Generate congestion analysis (cumulative pedestrian count, adaptive spatial extent)
        print(f"5. Generating congestion area analysis...")
        calculate_congestion_time_map(
            csv_file_path,
            process_png_path,
            congestion_threshold=None,   # New definition no longer uses density threshold
            output_path=congestion_path,
            vmax=None,
            spatial_extent=None          # Auto-adaptive inside the function
        )
        
        print("Data processing complete!")
        print(f"Output files:")
        print(f"- Processed image: {process_png_path}")
        print(f"- Heatmap directory: {heatmap_dir}")
        print(f"- Final output directory: {output_dir}")
        print(f"- Video file: {video_path}")
        print(f"- Congestion analysis: {congestion_path}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()


# Keep the original main block for standalone execution
if __name__ == "__main__":
    # This code runs when the file is executed directly, but not when imported
    print("data_process.py executed directly")
    print("Please use the process_evacuation_data function for data processing")