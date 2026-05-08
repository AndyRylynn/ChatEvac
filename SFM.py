# -*- coding: utf-8 -*-
"""
SFM.py — Helbing(2000) Social Force Model Evacuation Simulation Engine (based on floor plan PNG)
=================================================================================================
PNG three-color convention:
  Black (R,G,B < 60)           → Walkable area
  Red (R > 200, G,B < 80)      → Exit (pedestrian evacuates upon stepping in)
  White (R,G,B > 200)          → Wall (impenetrable)

Public interface:
  simulate(image_path, output_dir, num_people=50, space_width_m=30.0, ...)
      Run simulation headless, write EvacT.txt / EvacFlow.csv / EvacLocation.csv / traj.npz
      Return dict: {evac_time, num_evac, completed, traj_path, ...}
  replay(traj_path, image_path, sc_px_max=800)
      Read NPZ trajectory, replay with Tk GUI (Space to pause, speed slider)

Note: EvacLocation.csv y coordinates are flipped to bottom-left origin (consistent with
    SFM2.PeopleList.save_evacuation_data), making them compatible with the full
    data_process.py / analysis_gui.py analysis pipeline.
"""

from __future__ import annotations
import math, os, random, time, csv, heapq
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import distance_transform_edt, label as cc_label, gaussian_filter
import cv2


# ============================================================
# Scene: PNG parsing + distance field + EDT wall distance + gradient field (one-time precomputation)
# ============================================================
class Scene:
    def __init__(self, image_path: str, space_width_m: float = 30.0,
                 grad_smooth_sigma: float = 2.0,
                 wall_erode_px: int = 0):
        """wall_erode_px: Erosion pixels for wall mask (expands walkable area); default 0 = no erosion.
        If diffusion output has overly narrow corridors causing pedestrian deadlock, set to 1~2."""
        self.image_path = image_path
        self.img_pil = Image.open(image_path).convert("RGB")
        arr = np.asarray(self.img_pil)
        self.H_PX, self.W_PX = arr.shape[:2]
        self.M_PER_PX = space_width_m / self.W_PX
        self.WIDTH_M = space_width_m
        self.HEIGHT_M = self.H_PX * self.M_PER_PX

        R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
        # Consistent with SFM2 (launch.detect_exit_regions): detect red exits in HSV space, then dilate
        # This tolerates sparse red pixels / anti-aliased transitions in diffusion output
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red_ranges = [
            ([0, 150, 150],   [5, 255, 255]),
            ([175, 150, 150], [180, 255, 255]),
        ]
        ex = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in red_ranges:
            ex |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        ex = cv2.dilate(ex, np.ones((5, 5), np.uint8), iterations=2)
        self.exit_mask = ex.astype(bool)

        # Walls: white (high V, low S in HSV), complement of SFM2 walkable
        # SFM2 uses inRange(hsv,[0,0,0],[180,255,50]) to get black as walkable;
        # Same semantics here: black = walkable base, white = wall
        black_hsv = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        self.black_m = black_hsv.astype(bool)
        # Wall = remaining bright area that is neither black nor red (mostly white); relaxed "bright and low saturation" test
        white_hsv = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
        self.wall_mask = white_hsv.astype(bool) & ~self.exit_mask
        # Remove exit pixels from black (exits are red, not black; this just ensures no overlap)
        self.black_m = self.black_m & ~self.exit_mask & ~self.wall_mask
        # Erode walls: eliminate wall lines narrower than (2*wall_erode_px), making overly narrow corridors in diffusion output walkable
        if wall_erode_px > 0:
            k = np.ones((2 * wall_erode_px + 1, 2 * wall_erode_px + 1), np.uint8)
            wall_eroded = cv2.erode(self.wall_mask.astype(np.uint8), k, iterations=1).astype(bool)
            # Eroded wall becomes walkable
            new_walk = self.wall_mask & ~wall_eroded
            self.wall_mask = wall_eroded
            self.black_m = self.black_m | new_walk
        self.walk_mask = self.black_m | self.exit_mask

        classified = self.wall_mask | self.exit_mask | self.black_m
        unclass = 1.0 - classified.mean()
        if unclass > 0.05:
            print(f"[SFM] Warning: Unclassified pixels in PNG: {unclass*100:.1f}% > 5%")
        if not self.exit_mask.any():
            raise ValueError(f"[SFM] No red exit pixels found in PNG: {image_path}")
        if not self.walk_mask.any():
            raise ValueError(f"[SFM] No walkable pixels found in PNG: {image_path}")

        # Connected components: mark walkable pixels that can reach any exit
        lab, _ = cc_label(self.walk_mask, structure=np.ones((3, 3), dtype=int))
        reach_labels = set(np.unique(lab[self.exit_mask]))
        reach_labels.discard(0)
        self.reach_mask = np.isin(lab, list(reach_labels))
        unreach_px = int((self.walk_mask & ~self.reach_mask).sum())
        if unreach_px > 0:
            print(f"[SFM] Warning: {unreach_px} walkable pixels not connected to any exit (isolated islands)")

        # ---- Distance field (Dijkstra 8-neighbor, multi-source seeds = red pixels) ----
        INF = -1.0
        dist = np.full((self.H_PX, self.W_PX), INF, dtype=np.float32)
        ey, ex = np.where(self.exit_mask)
        DIRS8 = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.41421356), (1, -1, 1.41421356),
                 (-1, 1, 1.41421356), (-1, -1, 1.41421356))
        pq = []
        for r, c in zip(ey, ex):
            heapq.heappush(pq, (0.0, int(r), int(c)))
            dist[r, c] = 0.0
        while pq:
            d0, r, c = heapq.heappop(pq)
            if d0 > dist[r, c]: continue
            for dr, dc, w in DIRS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H_PX and 0 <= nc < self.W_PX and self.walk_mask[nr, nc]:
                    nd = d0 + w
                    if dist[nr, nc] < 0 or nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(pq, (nd, nr, nc))

        # Fill unwalkable pixels with distance to nearest walkable pixel, preventing bilinear sampling from hitting -1
        invalid = dist < 0
        if invalid.any() and (~invalid).any():
            _, (iy_idx, ix_idx) = distance_transform_edt(invalid, return_indices=True)
            dist = dist[iy_idx, ix_idx]

        # Gaussian smooth distance field -> eliminate gradient singularities at wall convex corners
        dist_smooth = gaussian_filter(dist, sigma=grad_smooth_sigma)
        self.GY_D, self.GX_D = np.gradient(dist_smooth)
        self.DIST = dist

        # ---- Wall distance EDT + normal ----
        self.DIST_W_PX = distance_transform_edt(~self.wall_mask).astype(np.float32)
        self.GY_W, self.GX_W = np.gradient(self.DIST_W_PX)

        # ---- Exit distance field (meters) ----
        self.EXIT_DIST_M = distance_transform_edt(~self.exit_mask).astype(np.float32) * self.M_PER_PX
        # Near-exit threshold: 1.5 m (fixed, consistent with helbing_test, does not scale with resolution)
        self.EXIT_NEAR_M = 1.5

    def _bilinear(self, arr, cx, cy):
        ix, iy = int(cx), int(cy)
        if ix < 0: ix = 0
        elif ix > self.W_PX - 2: ix = self.W_PX - 2
        if iy < 0: iy = 0
        elif iy > self.H_PX - 2: iy = self.H_PX - 2
        fx, fy = cx - ix, cy - iy
        v00 = arr[iy, ix]; v10 = arr[iy, ix + 1]
        v01 = arr[iy + 1, ix]; v11 = arr[iy + 1, ix + 1]
        return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11

    def get_gradient(self, px, py):
        cx = px / self.M_PER_PX
        cy = py / self.M_PER_PX
        if cx < 0 or cx >= self.W_PX - 1 or cy < 0 or cy >= self.H_PX - 1:
            return 0.0, 0.0
        gx_d = self._bilinear(self.GX_D, cx, cy)
        gy_d = self._bilinear(self.GY_D, cx, cy)
        dx, dy = -gx_d, -gy_d
        m = math.hypot(dx, dy)
        if m < 1e-9: return 0.0, 0.0
        return dx / m, dy / m

    def nw(self, px, py):
        """Distance to nearest wall (meters) + unit normal (pointing away from wall)"""
        cx, cy = px / self.M_PER_PX, py / self.M_PER_PX
        d_px = self._bilinear(self.DIST_W_PX, cx, cy)
        wd = d_px * self.M_PER_PX
        gx = self._bilinear(self.GX_W, cx, cy)
        gy = self._bilinear(self.GY_W, cx, cy)
        m = math.hypot(gx, gy)
        if m < 1e-9: return wd, 0.0, 0.0
        return wd, gx / m, gy / m

    def wk(self, x, y):
        cx, cy = int(x / self.M_PER_PX), int(y / self.M_PER_PX)
        if cx < 0 or cx >= self.W_PX or cy < 0 or cy >= self.H_PX: return False
        return bool(self.walk_mask[cy, cx])

    def in_exit(self, x, y, radius):
        cx, cy = int(x / self.M_PER_PX), int(y / self.M_PER_PX)
        if cx < 0 or cx >= self.W_PX or cy < 0 or cy >= self.H_PX: return False
        if self.exit_mask[cy, cx]: return True
        return self.EXIT_DIST_M[cy, cx] < radius

    def exit_dist(self, x, y):
        cx, cy = int(x / self.M_PER_PX), int(y / self.M_PER_PX)
        if cx < 0 or cx >= self.W_PX or cy < 0 or cy >= self.H_PX: return 1e9
        return float(self.EXIT_DIST_M[cy, cx])


# ============================================================
# Person + Dynamics
# ============================================================
class Person:
    __slots__ = ("x", "y", "vx", "vy", "ex")
    def __init__(self, x, y):
        self.x = x; self.y = y; self.vx = 0.0; self.vy = 0.0; self.ex = False


def _compute_forces(scene: Scene, pp, prm):
    n = len(pp)
    ac = [(0.0, 0.0)] * n
    A_PED = prm["A_ped"]; A_WALL = prm["A_wall"]; B = prm["B"]; B_WALL = prm["B_wall"]
    K_BODY = prm["k_body"]; K_FRIC = prm["k_fric"]
    TAU = prm["tau"]; V_DES = prm["v_des"]; M = prm["mass"]; R = prm["radius"]
    INTERACT = prm["interact_range"]; WALL_RANGE = prm["wall_range"]
    EXIT_NEAR = scene.EXIT_NEAR_M

    for i, p in enumerate(pp):
        if p.ex: continue
        dx, dy = scene.get_gradient(p.x, p.y)
        # Wall tangential sliding: attenuate the "toward-wall" component of driving direction, avoiding deadlock in concave corners
        wd0, wnx0, wny0 = scene.nw(p.x, p.y)
        if wd0 < WALL_RANGE and (dx != 0.0 or dy != 0.0):
            dn = dx * wnx0 + dy * wny0
            if dn < 0.0:
                w = (wd0 / WALL_RANGE) ** 2
                dx_new = dx - (1.0 - w) * dn * wnx0
                dy_new = dy - (1.0 - w) * dn * wny0
                m = math.hypot(dx_new, dy_new)
                if m > 1e-9:
                    dx, dy = dx_new / m, dy_new / m

        drx = M * (V_DES * dx - p.vx) / TAU
        dry = M * (V_DES * dy - p.vy) / TAU
        rx = 0.0; ry = 0.0

        # Person-person force (full Helbing)
        for j, o in enumerate(pp):
            if i == j or o.ex: continue
            ddx = p.x - o.x; ddy = p.y - o.y
            d = math.hypot(ddx, ddy)
            if d >= INTERACT: continue
            if d < 1e-9:
                d = 0.001
                a = random.uniform(0, 6.283)
                ddx = math.cos(a) * d; ddy = math.sin(a) * d
            rs = R * 2; ov = rs - d
            fn = A_PED * math.exp((rs - d) / B)
            if ov > 0: fn += K_BODY * ov
            nx = ddx / d; ny = ddy / d
            ft = 0.0
            if ov > 0:
                dvt = (o.vx - p.vx) * (-ny) + (o.vy - p.vy) * nx
                ft = K_FRIC * ov * dvt
            rx += fn * nx - ft * ny
            ry += fn * ny + ft * nx

        # Wall force (attenuated near exits, prevents pushing people away from exits)
        wd, wnx, wny = scene.nw(p.x, p.y)
        ed = scene.exit_dist(p.x, p.y)
        wall_atten = 1.0 if ed > EXIT_NEAR else max(0.0, ed / EXIT_NEAR) * 0.1
        if wd < WALL_RANGE:
            ow = R - wd
            fw = A_WALL * math.exp((R - wd) / B_WALL)
            if ow > 0: fw += K_BODY * ow
            rx += fw * wnx * wall_atten
            ry += fw * wny * wall_atten

        ac[i] = ((drx + rx) / M, (dry + ry) / M)
    return ac


def _integrate(scene: Scene, pp, ac, prm):
    DT = prm["dt"]; MAX_SPD = prm["max_spd"]; R = prm["radius"]
    for i, p in enumerate(pp):
        if p.ex: continue
        ax, ay = ac[i]
        p.vx += ax * DT; p.vy += ay * DT
        sp = math.hypot(p.vx, p.vy)
        if sp > MAX_SPD:
            p.vx *= MAX_SPD / sp; p.vy *= MAX_SPD / sp
        nx = p.x + p.vx * DT; ny = p.y + p.vy * DT
        if not scene.wk(nx, ny):
            if scene.wk(p.x, ny): nx = p.x
            elif scene.wk(nx, p.y): ny = p.y
            else: nx, ny = p.x, p.y
            p.vx = 0.0; p.vy = 0.0
        p.x = nx; p.y = ny
        if scene.in_exit(p.x, p.y, R): p.ex = True


def _init_people(scene: Scene, n: int, radius: float, seed: int):
    rng = random.Random(seed)
    pp = []
    tries = 0
    max_tries = max(200, n * 200)
    min_sep_sq = (2 * radius) ** 2
    bm = scene.black_m
    while len(pp) < n and tries < max_tries:
        tries += 1
        cx = rng.randint(0, scene.W_PX - 1)
        cy = rng.randint(0, scene.H_PX - 1)
        if not bm[cy, cx]: continue
        if not scene.reach_mask[cy, cx]: continue   # Skip isolated islands
        x = (cx + 0.5) * scene.M_PER_PX
        y = (cy + 0.5) * scene.M_PER_PX
        if scene.DIST_W_PX[cy, cx] * scene.M_PER_PX < radius + 0.05: continue
        ok = True
        for q in pp:
            if (q.x - x) ** 2 + (q.y - y) ** 2 < min_sep_sq:
                ok = False; break
        if ok: pp.append(Person(x, y))
    if len(pp) < n:
        print(f"[SFM] Warning: Sampled {max_tries} times but only placed {len(pp)}/{n} people")
    return pp


# ============================================================
# Public interface: simulate
# ============================================================
def simulate(image_path: str,
             output_dir: str,
             num_people: int = 50,
             space_width_m: float = 30.0,
             seed: int = 42,
             max_sim_time_s: float = 600.0,
             A_ped: float = 2000.0, A_wall: float = 2000.0,
             B: float = 0.08, B_wall: float = 0.08,
             k_body: float = 12000.0, k_fric: float = 24000.0,
             tau: float = 0.5, v_des: float = 1.34,
             mass: float = 80.0, radius: float = 0.25,
             dt: float = 0.01, max_spd: float = 2.0,
             interact_range: float = 3.0, wall_range: float = 1.0,
             grad_smooth_sigma: float = 2.0,
             wall_erode_px: int = 0,
             verbose: bool = True) -> dict:
    """Run simulation headless, write CSV triple + traj.npz, return result dict."""
    os.makedirs(output_dir, exist_ok=True)
    scene = Scene(image_path, space_width_m=space_width_m,
                  grad_smooth_sigma=grad_smooth_sigma,
                  wall_erode_px=wall_erode_px)
    pp = _init_people(scene, num_people, radius, seed)
    N0 = len(pp)
    if verbose:
        print(f"[SFM] Image {scene.W_PX}x{scene.H_PX} px, "
              f"Physical {scene.WIDTH_M:.1f} x {scene.HEIGHT_M:.1f} m, "
              f"{scene.M_PER_PX:.4f} m/px")
        print(f"[SFM] Pedestrians {N0}/{num_people}")

    prm = dict(A_ped=A_ped, A_wall=A_wall, B=B, B_wall=B_wall,
               k_body=k_body, k_fric=k_fric,
               tau=tau, v_des=v_des, mass=mass, radius=radius,
               dt=dt, max_spd=max_spd,
               interact_range=interact_range, wall_range=wall_range)

    # Integer-second sampling buffer (compatible with downstream)
    evac_flow_rows = [(0, 0)]
    evac_loc_rows = []   # (T, x, y_flipped)
    # Full-frame trajectory buffer (for replay)
    max_steps = int(math.ceil(max_sim_time_s / dt)) + 1
    traj = np.full((max_steps, N0, 3), np.nan, dtype=np.float32)  # x, y, ex

    sim_t = 0.0
    step = 0
    last_int_sec = 0
    completed = False
    t_wall0 = time.time()

    while sim_t < max_sim_time_s:
        ac = _compute_forces(scene, pp, prm)
        _integrate(scene, pp, ac, prm)
        sim_t += dt
        for i, p in enumerate(pp):
            traj[step, i, 0] = p.x
            traj[step, i, 1] = p.y
            traj[step, i, 2] = 1.0 if p.ex else 0.0
        step += 1

        sec = int(sim_t + 1e-6)
        if sec > last_int_sec:
            last_int_sec = sec
            n_evac = sum(1 for q in pp if q.ex)
            evac_flow_rows.append((sec, n_evac))
            for q in pp:
                if not q.ex:
                    evac_loc_rows.append((sec, q.x, scene.HEIGHT_M - q.y))
            if verbose and sec % 5 == 0:
                print(f"[SFM] t={sec}s, evacuated {n_evac}/{N0}")

        if all(p.ex for p in pp):
            completed = True
            # Add a row to EvacFlow for the "all-evacuated" instant (integer second)
            sec = int(math.ceil(sim_t))
            if sec > last_int_sec:
                evac_flow_rows.append((sec, N0))
                last_int_sec = sec
            break

    evac_time = sim_t
    n_evac_final = sum(1 for q in pp if q.ex)

    # ===== Write CSV / TXT =====
    evac_t_path = os.path.join(output_dir, "EvacT.txt")
    with open(evac_t_path, "w", encoding="utf-8") as f:
        f.write(f"{evac_time:.3f}\n")

    flow_path = os.path.join(output_dir, "EvacFlow.csv")
    with open(flow_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["T", "N"])
        for t, n in evac_flow_rows:
            w.writerow([t, n])

    loc_path = os.path.join(output_dir, "EvacLocation.csv")
    with open(loc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["T", "x", "y"])
        for t, x, y in evac_loc_rows:
            w.writerow([t, f"{x:.4f}", f"{y:.4f}"])

    # ===== Write NPZ trajectory (for replay) =====
    traj = traj[:step]   # Trim to actual simulation steps
    traj_path = os.path.join(output_dir, "traj.npz")
    np.savez_compressed(
        traj_path,
        traj=traj, dt=np.float32(dt),
        space_width_m=np.float32(space_width_m),
        height_m=np.float32(scene.HEIGHT_M),
        radius=np.float32(radius),
        n_people=np.int32(N0),
    )

    if verbose:
        print(f"[SFM] Done completed={completed} evac_time={evac_time:.2f}s "
              f"n_evac={n_evac_final}/{N0} wall_clock={time.time()-t_wall0:.1f}s")
        print(f"[SFM] Output: {evac_t_path}")
        print(f"[SFM] Output: {flow_path}")
        print(f"[SFM] Output: {loc_path}")
        print(f"[SFM] Output: {traj_path}")

    return dict(
        evac_time=evac_time,
        num_evac=n_evac_final,
        num_total=N0,
        completed=completed,
        traj_path=traj_path,
        evac_t_path=evac_t_path,
        evac_flow_path=flow_path,
        evac_loc_path=loc_path,
    )


# ============================================================
# Public interface: replay
# ============================================================
def replay(traj_path: str, image_path: str, sc_px_max: int = 800):
    """Read NPZ trajectory, replay with Tk GUI. Blocks until window closed."""
    import tkinter as tk
    data = np.load(traj_path)
    traj = data["traj"]   # (T, N, 3)
    dt = float(data["dt"])
    space_width_m = float(data["space_width_m"])
    radius = float(data["radius"])
    T_steps = traj.shape[0]
    N = int(data["n_people"])

    img_pil = Image.open(image_path).convert("RGB")
    W_PX, H_PX = img_pil.size
    scale = sc_px_max / max(W_PX, H_PX)
    cw = max(1, int(W_PX * scale))
    ch = max(1, int(H_PX * scale))
    px_per_m = cw / space_width_m

    root = tk.Tk()
    root.title("SFM Replay — Space: Pause/Resume")
    canvas = tk.Canvas(root, width=cw, height=ch, bg="white")
    canvas.pack()
    bg = img_pil.resize((cw, ch), Image.NEAREST)
    bgimg = ImageTk.PhotoImage(bg)
    canvas.create_image(0, 0, image=bgimg, anchor="nw")
    label = tk.Label(root, text="Ready", font=("Consolas", 11))
    label.pack()
    sv = tk.DoubleVar(value=1.0)
    tk.Scale(root, from_=0.1, to=10, resolution=0.1, orient=tk.HORIZONTAL,
             label="Speed", variable=sv).pack(fill=tk.X)

    state = {"running": True, "paused": False, "frame": 0}
    def toggle(ev=None): state["paused"] = not state["paused"]
    def close(): state["running"] = False; root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<space>", toggle)

    r = max(2, int(radius * px_per_m))
    frame_skip = 5  # Same as original Helbing GUI (draw every 5 steps)

    while state["running"]:
        if not state["paused"]:
            t0 = time.time()
            f = state["frame"]
            if f >= T_steps:
                label.config(text=f"Replay finished  t={(T_steps-1)*dt:.1f}s")
                state["paused"] = True
            else:
                canvas.delete("p")
                row = traj[f]
                n_remain = 0
                for i in range(N):
                    ex = row[i, 2]
                    if ex >= 0.5: continue
                    x = row[i, 0]; y = row[i, 1]
                    if not (np.isfinite(x) and np.isfinite(y)): continue
                    sx = x * px_per_m; sy = y * px_per_m
                    canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                       fill="blue", outline="", tags="p")
                    n_remain += 1
                sim_t = f * dt
                label.config(text=f"t={sim_t:.1f}s | Remaining:{n_remain}/{N} | Speed:{sv.get():.1f}x")
                state["frame"] += frame_skip
            elapsed = time.time() - t0
            target = (dt * frame_skip) / sv.get()
            if elapsed < target:
                root.after(int((target - elapsed) * 1000))
        root.update()
        if not state["running"]: break


# ============================================================
# CLI self-test entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-people", type=int, default=30)
    ap.add_argument("--space-width", type=float, default=30.0)
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()
    res = simulate(args.image_path, args.output_dir,
                   num_people=args.num_people,
                   space_width_m=args.space_width)
    if not args.no_replay:
        replay(res["traj_path"], args.image_path)
