#!/usr/bin/env python3
"""dream_matplotlib.py — Static generative art from dreams using matplotlib.

Pre-coded, tested rendering functions. The LLM picks which technique fits
the dream and provides color/atmosphere parameters. No LLM-generated code.

Techniques:
  1. strange_attractor   — Clifford/De Jong/Aizawa/Lorenz dense point clouds
  2. flow_field          — Particles tracing through a Perlin-like vector field
  3. fractal_tree        — Recursive L-system branching
  4. voronoi             — Colored Voronoi tessellation from random seeds
  5. dla                 — Diffusion-limited aggregation (organic growth)
  6. wave_interference   — Static concentric ripples from multiple sources
  7. phase_portrait      — Vector field + trajectories through a dynamical system
  8. starfield           — Density-layered particle/nebula plot

All produce dark-themed 1024x1024 PNGs at 150 DPI.
"""
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
import aion_env  # noqa

AION = os.environ.get("AION_HOME", os.path.expanduser("~/aikio"))
GALLERY = f"{AION}/gallery/visual"

INTUITION_URL = os.environ.get("OLLAMA_INTUITION_URL", "http://localhost:11438")
INTUITION_MODEL = os.environ.get("INTUITION_MODEL", "glm-4.7-flash")

# Dark palettes: name -> list of hex colors
PALETTES = {
    "cool_warm": ["#0a0a2e", "#1a1a6e", "#2d5a8e", "#3aa8c8", "#5ddeee", "#a8e6cf", "#ffd93d", "#ff6b6b"],
    "fire_ice": ["#000033", "#003366", "#0099cc", "#66ddff", "#ffffff", "#ffaa00", "#ff4400", "#990000"],
    "neon": ["#0a0014", "#1a0033", "#6600ff", "#00f5ff", "#00ff7f", "#ffff00", "#ff00ff", "#ff0066"],
    "sunset": ["#0a0010", "#1a0530", "#4b0082", "#7d26cd", "#ff6b35", "#ffd93d", "#ff1744", "#ffa500"],
    "deep_ocean": ["#000510", "#001f3f", "#003d7a", "#0074d9", "#39cccc", "#7fdbff", "#b10dc9", "#aaaaff"],
    "aurora": ["#000010", "#0f0f3f", "#004477", "#00ff7f", "#39ff14", "#00f5ff", "#bf00ff", "#ffffff"],
    "nebula": ["#050010", "#100020", "#4b0082", "#9370db", "#ff1493", "#ffd700", "#ff69b4", "#ffffff"],
    "ember": ["#0a0000", "#1c1c1c", "#4a0000", "#8b0000", "#ff4500", "#ffd700", "#ff6347", "#ffe4b5"],
    "forest": ["#000a00", "#001500", "#0a3300", "#1a6600", "#33aa00", "#66dd44", "#aaff88", "#ddffaa"],
    "monochrome": ["#000000", "#111111", "#222222", "#444444", "#666666", "#999999", "#cccccc", "#ffffff"],
}


def _strip_think(text):
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _chat(url, model, messages, timeout=120, think=False, **options):
    body = json.dumps({
        "model": model, "stream": False, "messages": messages,
        "think": think, "options": options,
    }).encode()
    req = urllib.request.Request(f"{url}/api/chat", data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
        return _strip_think(result["message"]["content"])
    except Exception as e:
        print(f"[dream_matplotlib] LLM call failed: {e}", flush=True)
        return None


def _dream_summary(dream):
    seed = dream.get("seed", "unknown")
    if isinstance(seed, dict):
        seed = seed.get("node", str(seed))
    walk = (dream.get("walk") or dream.get("simulation") or "")[:2000]
    synthesis = dream.get("synthesis", "")[:1500]
    insights = dream.get("insights", [])
    insight_text = "\n".join(f"[{i.get('type','')}] {i.get('text','')}" for i in insights[:5])
    parts = [f"Seed: {str(seed)[:200]}"]
    if walk: parts.append(f"Walk/Simulation:\n{walk}")
    if synthesis: parts.append(f"Synthesis:\n{synthesis}")
    if insight_text: parts.append(f"Insights:\n{insight_text}")
    return "\n\n".join(parts)


def _save_fig(fig, aid, subdir="matplotlib"):
    """Save figure to gallery, return (gallery_path, filename)."""
    filename = f"{aid}.png"
    os.makedirs(GALLERY, exist_ok=True)
    gallery_path = f"{GALLERY}/{filename}"
    fig.savefig(gallery_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return gallery_path, filename


# ---------------------------------------------------------------------------
# Technique 1: Strange Attractors
# ---------------------------------------------------------------------------

def render_attractor(params, aid):
    """Dense point-cloud strange attractor with color gradient along trajectory."""
    attractor = params.get("attractor", "clifford")
    palette = PALETTES.get(params.get("palette", "nebula"), PALETTES["nebula"])
    n_points = params.get("n_points", 80000)
    bg = params.get("bg_color", "#050008")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    xs, ys = [], []

    if attractor == "clifford":
        a, b, c, d = params.get("a", -1.8), params.get("b", -2.0), params.get("c", 1.6), params.get("d", 1.5)
        x, y = 0.1, 0.1
        for _ in range(n_points):
            nx = math.sin(a * y) + c * math.cos(a * x)
            ny = math.sin(b * x) + d * math.cos(b * y)
            x, y = nx, ny
            xs.append(x); ys.append(y)

    elif attractor == "de_jong":
        a, b, c, d = params.get("a", 1.641), params.get("b", 1.902), params.get("c", 0.316), params.get("d", 1.525)
        x, y = 0.1, 0.1
        for _ in range(n_points):
            nx = math.sin(a * y) - math.cos(b * x)
            ny = math.sin(c * x) - math.cos(d * y)
            x, y = nx, ny
            xs.append(x); ys.append(y)

    elif attractor == "lorenz":
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        dt = 0.005
        x, y, z = 0.1, 0.0, 0.0
        for _ in range(n_points):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt; y += dy * dt; z += dz * dt
            xs.append(x); ys.append(y)

    elif attractor == "aizawa":
        a, bc, c_, d = 0.95, 0.7, 0.6, 3.5
        dt = 0.01
        x, y, z = 0.1, 0.0, 0.0
        for _ in range(n_points):
            dx = (z - bc) * x - d * y
            dy = d * x + (z - bc) * y
            dz = c_ * z - (z**3)/3 - (x**2 + y**2) * (1 + 0.25*z) + bc * z
            x += dx * dt; y += dy * dt; z += dz * dt
            xs.append(x); ys.append(y)

    else:  # default to clifford
        a, b, c, d = -1.8, -2.0, 1.6, 1.5
        x, y = 0.1, 0.1
        for _ in range(n_points):
            nx = math.sin(a * y) + c * math.cos(a * x)
            ny = math.sin(b * x) + d * math.cos(b * y)
            x, y = nx, ny
            xs.append(x); ys.append(y)

    xs = np.array(xs); ys = np.array(ys)

    # Normalize
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    rng_x = max(xmax - xmin, 0.01)
    rng_y = max(ymax - ymin, 0.01)
    cx = (xmax + xmin) / 2; cy = (ymax + ymin) / 2
    # Scale to fit in [-1, 1] preserving aspect ratio
    scale = max(rng_x, rng_y) / 2
    xs = (xs - cx) / scale; ys = (ys - cy) / scale

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    # 2D histogram density rendering — works for ALL attractors regardless
    # of point spread. Dense regions glow brightly, sparse regions fade.
    from scipy.ndimage import gaussian_filter as _gf
    grid_res = 500
    hist, xedges, yedges = np.histogram2d(xs, ys, bins=grid_res,
                                           range=[[-1.5, 1.5], [-1.5, 1.5]])
    # Log scale for dynamic range (attractors have huge density variation)
    hist = np.log1p(hist)
    # Gaussian smoothing — spreads density into a luminous glow so compact
    # attractors (Aizawa) are visible, not just a few bright pixels.
    hist = _gf(hist, sigma=1.5)
    hist = (hist - hist.min()) / (hist.max() - hist.min() + 0.001)

    # Render with cmap, oriented correctly
    ax.imshow(hist.T, extent=[-1.5, 1.5, -1.5, 1.5], origin="lower",
              cmap=cmap, aspect="equal", interpolation="bilinear")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 2: Flow Field
# ---------------------------------------------------------------------------

def render_flow_field(params, aid):
    """Particles tracing through a smooth pseudo-Perlin vector field."""
    palette = PALETTES.get(params.get("palette", "aurora"), PALETTES["aurora"])
    n_particles = params.get("n_particles", 150)
    n_steps = params.get("n_steps", 200)
    noise_scale = params.get("noise_scale", 0.015)
    step_size = params.get("step_size", 0.03)
    line_width = params.get("line_width", 0.3)
    bg = params.get("bg_color", "#050010")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    # Build a smooth angle field using sums of sinusoids (pseudo-Perlin)
    grid_n = 200
    gx = np.linspace(-1, 1, grid_n)
    gy = np.linspace(-1, 1, grid_n)
    GX, GY = np.meshgrid(gx, gy)

    # Multi-octave angle field
    angle = np.zeros_like(GX)
    for octave, freq in enumerate([1, 2, 4, 8]):
        phase_x = np.random.uniform(0, 2*math.pi)
        phase_y = np.random.uniform(0, 2*math.pi)
        angle += (np.sin(GX * freq * 3 + phase_x) * np.cos(GY * freq * 3 + phase_y)) / (2**octave)

    angle *= math.pi  # scale to full rotation range

    # Interpolation function
    from scipy.interpolate import RegularGridInterpolator
    angle_func = RegularGridInterpolator((gx, gy), angle, method="linear",
                                          bounds_error=False, fill_value=0)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    for i in range(n_particles):
        x, y = np.random.uniform(-0.9, 0.9, 2)
        trail_x, trail_y = [x], [y]

        for _ in range(n_steps):
            a = angle_func([[x, y]])[0]
            x += math.cos(a) * step_size
            y += math.sin(a) * step_size
            if abs(x) > 1.2 or abs(y) > 1.2:
                break
            trail_x.append(x); trail_y.append(y)

        if len(trail_x) > 5:
            color = cmap(np.random.uniform(0.2, 0.95))
            alpha = np.random.uniform(0.15, 0.5)
            ax.plot(trail_x, trail_y, color=color, alpha=alpha,
                    linewidth=line_width, solid_capstyle="round")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 3: Fractal Tree (L-system style)
# ---------------------------------------------------------------------------

def render_fractal_tree(params, aid):
    """Recursive branching fractal tree."""
    palette = PALETTES.get(params.get("palette", "ember"), PALETTES["ember"])
    depth = params.get("depth", 11)
    branch_angle = params.get("branch_angle", 25)  # degrees
    angle_spread = params.get("angle_spread", 8)  # randomness
    length_ratio = params.get("length_ratio", 0.72)
    initial_length = params.get("initial_length", 0.28)
    bg = params.get("bg_color", "#0a0000")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    lines = []

    def branch(x, y, angle, length, d):
        if d <= 0 or length < 0.003:
            return
        ex = x + math.cos(angle) * length
        ey = y + math.sin(angle) * length
        t = 1 - d / depth  # 0 at trunk, 1 at tips
        color = cmap(t)
        lw = max(0.3, d * 0.5)
        alpha = 0.4 + 0.5 * t
        lines.append(([x, ex], [y, ey], color, lw, alpha))

        new_len = length * length_ratio
        left_a = angle + math.radians(branch_angle + np.random.uniform(-angle_spread, angle_spread))
        right_a = angle - math.radians(branch_angle + np.random.uniform(-angle_spread, angle_spread))

        branch(ex, ey, left_a, new_len, d - 1)
        branch(ex, ey, right_a, new_len, d - 1)

        # Sometimes a middle branch for fuller trees
        if np.random.random() < 0.3 and d > 4:
            mid_a = angle + math.radians(np.random.uniform(-15, 15))
            branch(ex, ey, mid_a, new_len * 0.85, d - 1)

    # Multiple trees for a forest effect
    n_trees = params.get("n_trees", 3)
    for _ in range(n_trees):
        start_x = np.random.uniform(-0.7, 0.7)
        branch(start_x, -1.0, math.pi/2 + np.random.uniform(-0.1, 0.1),
               initial_length * np.random.uniform(0.7, 1.1), depth)

    for xs, ys, color, lw, alpha in lines:
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 4: Voronoi Tessellation
# ---------------------------------------------------------------------------

def render_voronoi(params, aid):
    """Colored Voronoi cells from random seed points."""
    palette = PALETTES.get(params.get("palette", "deep_ocean"), PALETTES["deep_ocean"])
    n_seeds = params.get("n_seeds", 60)
    bg = params.get("bg_color", "#000510")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    # Generate seed points — clustered, not uniform
    n_clusters = np.random.randint(2, 5)
    centers = np.random.uniform(-0.7, 0.7, (n_clusters, 2))
    points = []
    for _ in range(n_seeds):
        c = centers[np.random.randint(n_clusters)]
        pt = c + np.random.normal(0, 0.2, 2)
        points.append(pt)
    points = np.array(points)

    # Clip to reasonable bounds
    points = np.clip(points, -1.2, 1.2)

    # Add far-away points to bound the diagram
    far_pts = np.array([[-3,-3],[3,-3],[-3,3],[3,3],[0,-3],[0,3],[-3,0],[3,0]])
    all_pts = np.vstack([points, far_pts])

    vor = Voronoi(all_pts)

    # Color each cell based on distance from center
    center = np.array([0, 0])
    for idx in range(len(points)):
        if idx >= len(vor.point_region):
            continue
        region_idx = vor.point_region[idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        polygon = [vor.vertices[v] for v in region]
        if len(polygon) < 3:
            continue

        # Color based on position
        dist = np.linalg.norm(points[idx] - center)
        t = min(1.0, dist / 1.2)
        color = cmap(t)
        alpha = np.random.uniform(0.35, 0.75)

        poly_patch = plt.Polygon(polygon, facecolor=color, alpha=alpha,
                                  edgecolor=color, linewidth=0.5)
        ax.add_patch(poly_patch)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 5: Diffusion-Limited Aggregation
# ---------------------------------------------------------------------------

def render_dla(params, aid):
    """Organic branching growth via DLA."""
    palette = PALETTES.get(params.get("palette", "cool_warm"), PALETTES["cool_warm"])
    n_particles = params.get("n_particles", 3000)
    stickiness = params.get("stickiness", 1.0)
    bg = params.get("bg_color", "#050008")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    # Grid-based DLA for efficiency
    grid_size = 300
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    center = grid_size // 2

    # Seed: center point
    grid[center, center] = True
    aggregate = [(center, center)]

    # Also seed a few random points on the bottom for upward growth
    for _ in range(3):
        gx = center + np.random.randint(-20, 20)
        grid[grid_size - 5, gx] = True
        aggregate.append((grid_size - 5, gx))

    max_dist = 5
    for _ in range(n_particles):
        # Spawn on a circle around the aggregate
        spawn_r = int(max_dist * 1.5) + 10
        spawn_r = min(spawn_r, grid_size // 2 - 2)
        angle = np.random.uniform(0, 2 * math.pi)
        px = int(center + math.cos(angle) * spawn_r)
        py = int(center + math.sin(angle) * spawn_r)
        px = max(1, min(grid_size - 2, px))
        py = max(1, min(grid_size - 2, py))

        # Random walk
        for _ in range(2000):
            direction = np.random.randint(4)
            if direction == 0: px += 1
            elif direction == 1: px -= 1
            elif direction == 2: py += 1
            else: py -= 1

            px = max(1, min(grid_size - 2, px))
            py = max(1, min(grid_size - 2, py))

            # Check if near aggregate
            if np.random.random() < stickiness:
                neighbors = (grid[px-1,py] or grid[px+1,py] or
                            grid[px,py-1] or grid[px,py+1] or
                            grid[px-1,py-1] or grid[px+1,py+1] or
                            grid[px-1,py+1] or grid[px+1,py-1])
                if neighbors:
                    grid[px, py] = True
                    aggregate.append((px, py))
                    dist = math.sqrt((px-center)**2 + (py-center)**2)
                    max_dist = max(max_dist, dist)
                    break

            # If wandered too far, abandon
            if math.sqrt((px-center)**2 + (py-center)**2) > spawn_r * 1.5:
                break

    # Draw
    xs = [p[0] for p in aggregate]
    ys = [p[1] for p in aggregate]
    xs = np.array(xs); ys = np.array(ys)

    # Normalize to -1..1
    xs = (xs - center) / (grid_size * 0.4)
    ys = (ys - center) / (grid_size * 0.4)

    # Color by distance from center
    dists = np.sqrt(xs**2 + ys**2)
    max_d = max(dists.max(), 0.01)
    colors = cmap(dists / max_d)

    ax.scatter(xs, ys, c=colors, s=1.5, alpha=0.6, edgecolors="none")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 6: Wave Interference (static)
# ---------------------------------------------------------------------------

def render_wave_interference(params, aid):
    """Static wave interference pattern from multiple sources."""
    palette = PALETTES.get(params.get("palette", "aurora"), PALETTES["aurora"])
    n_sources = params.get("n_sources", 5)
    freq = params.get("freq", 8.0)
    bg = params.get("bg_color", "#000010")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    res = 500
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)

    # Source positions
    sources = []
    for _ in range(n_sources):
        angle = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0.2, 0.7)
        sources.append((np.cos(angle)*r, np.sin(angle)*r))

    field = np.zeros_like(X)
    for sx, sy in sources:
        dist = np.sqrt((X - sx)**2 + (Y - sy)**2) + 0.01
        field += np.sin(dist * freq * math.pi) / dist

    # Normalize
    field = (field - field.min()) / (field.max() - field.min() + 0.001)

    ax.imshow(field, extent=[-1, 1, -1, 1], origin="lower",
              cmap=cmap, alpha=0.85, aspect="equal")

    # Add source point glows
    for sx, sy in sources:
        ax.scatter([sx], [sy], c="white", s=30, alpha=0.8, zorder=5)
        ax.scatter([sx], [sy], c="white", s=100, alpha=0.2, zorder=4)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 7: Phase Portrait
# ---------------------------------------------------------------------------

def render_phase_portrait(params, aid):
    """Vector field with overlaid trajectories through a 2D dynamical system."""
    palette = PALETTES.get(params.get("palette", "sunset"), PALETTES["sunset"])
    system = params.get("system", "saddle")
    n_traj = params.get("n_trajectories", 40)
    bg = params.get("bg_color", "#0a0010")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("dream", palette, N=256)

    res = 25
    x = np.linspace(-1.5, 1.5, res)
    y = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(x, y)

    # Define systems
    alpha = np.random.uniform(0.8, 1.5)

    if system == "saddle":
        dX = X; dY = -Y
    elif system == "spiral":
        dX = -0.5*X - alpha*Y; dY = alpha*X - 0.5*Y
    elif system == "center":
        dX = -Y; dY = X
    elif system == "node":
        dX = -0.5*X + 0.2*Y; dY = 0.1*X - 0.6*Y
    elif system == "limit_cycle":
        r2 = X**2 + Y**2
        dX = -Y + X*(1 - r2); dY = X + Y*(1 - r2)
    else:  # van_der_pol
        mu = 1.5
        dX = Y; dY = mu*(1 - X**2)*Y - X

    # Draw faint vector field
    speed = np.sqrt(dX**2 + dY**2)
    speed_norm = speed / (speed.max() + 0.001)
    sp = ax.streamplot(x, y, dX, dY, color=speed_norm, cmap=cmap,
                       density=2.5, linewidth=0.5, arrowsize=0.3)
    # Dim the streamplot for a background effect
    sp.lines.set_alpha(0.25)

    # Draw highlighted trajectories
    for i in range(n_traj):
        x0 = np.random.uniform(-1.3, 1.3)
        y0 = np.random.uniform(-1.3, 1.3)
        traj_x, traj_y = [x0], [y0]
        cx, cy = x0, y0
        dt = 0.02
        for _ in range(200):
            if system == "saddle":
                dx = cx; dy = -cy
            elif system == "spiral":
                dx = -0.5*cx - alpha*cy; dy = alpha*cx - 0.5*cy
            elif system == "center":
                dx = -cy; dy = cx
            elif system == "node":
                dx = -0.5*cx + 0.2*cy; dy = 0.1*cx - 0.6*cy
            elif system == "limit_cycle":
                r2 = cx**2 + cy**2
                dx = -cy + cx*(1-r2); dy = cx + cy*(1-r2)
            else:
                mu = 1.5
                dx = cy; dy = mu*(1-cx**2)*cy - cx
            cx += dx * dt; cy += dy * dt
            if abs(cx) > 3 or abs(cy) > 3:
                break
            traj_x.append(cx); traj_y.append(cy)

        if len(traj_x) > 3:
            color = cmap(np.random.uniform(0.2, 0.95))
            alpha = np.random.uniform(0.3, 0.7)
            ax.plot(traj_x, traj_y, color=color, alpha=alpha,
                    linewidth=0.8, solid_capstyle="round")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique 8: Starfield / Nebula
# ---------------------------------------------------------------------------

def render_starfield(params, aid):
    """Multi-layered starfield with nebula clouds."""
    palette = PALETTES.get(params.get("palette", "nebula"), PALETTES["nebula"])
    n_stars = params.get("n_stars", 3000)
    n_nebula = params.get("n_nebula", 12)
    bg = params.get("bg_color", "#050010")

    fig, ax = plt.subplots(figsize=(6.83, 6.83), facecolor=bg)
    ax.set_facecolor(bg)

    from matplotlib.colors import LinearSegmentedColormap

    # Nebula clouds: large blurred gaussian blobs
    nebula_grid = np.zeros((300, 300))
    for _ in range(n_nebula):
        cx = np.random.uniform(50, 250)
        cy = np.random.uniform(50, 250)
        sigma = np.random.uniform(15, 50)
        intensity = np.random.uniform(0.3, 1.0)

        yy, xx = np.mgrid[0:300, 0:300]
        blob = np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*sigma**2)) * intensity
        nebula_grid += blob

    # Smooth and normalize
    nebula_grid = gaussian_filter(nebula_grid, sigma=8)
    nebula_grid = (nebula_grid - nebula_grid.min()) / (nebula_grid.max() + 0.001)

    cmap_neb = LinearSegmentedColormap.from_list("neb", ["#000000"] + palette, N=256)
    ax.imshow(nebula_grid, extent=[-1, 1, -1, 1], origin="lower",
              cmap=cmap_neb, alpha=0.5, aspect="equal")

    # Stars: varied sizes and brightness
    sx = np.random.uniform(-1, 1, n_stars)
    sy = np.random.uniform(-1, 1, n_stars)
    sizes = np.random.exponential(0.5, n_stars)
    brightness = np.random.uniform(0.3, 1.0, n_stars)

    # Some bright stars with glow
    bright_mask = sizes > 2
    bright_x = sx[bright_mask]
    bright_y = sy[bright_mask]

    # Regular stars
    dim_mask = ~bright_mask
    ax.scatter(sx[dim_mask], sy[dim_mask], c=brightness[dim_mask],
               cmap="gray", s=sizes[dim_mask], alpha=0.6,
               edgecolors="none", vmin=0, vmax=1)

    # Bright stars with diffraction spikes
    for bx, by in zip(bright_x[:50], bright_y[:50]):
        color = cmap_neb(np.random.uniform(0.6, 0.95))
        ax.scatter([bx], [by], c=[color], s=20, alpha=0.9, zorder=5)
        # Diffraction spikes
        spike_len = np.random.uniform(0.015, 0.04)
        ax.plot([bx-spike_len, bx+spike_len], [by, by],
                color=color, alpha=0.3, linewidth=0.3)
        ax.plot([bx, bx], [by-spike_len, by+spike_len],
                color=color, alpha=0.3, linewidth=0.3)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    return _save_fig(fig, aid)


# ---------------------------------------------------------------------------
# Technique selection (LLM picks technique + params)
# ---------------------------------------------------------------------------

TECHNIQUE_DESCRIPTIONS = """1. **strange_attractor** — Dense luminous point cloud following a mathematical attractor (Clifford, De Jong, Lorenz, Aizawa). Creates intricate organic structures with flowing color gradients. Best for dreams about: chaos/order, emergence, patterns forming from randomness, tension between predictability and unpredictability.

2. **flow_field** — Hundreds of particles tracing invisible currents through a smooth vector field, creating organic streamlines like wind or water. Best for dreams about: flow, journey, currents, invisible forces, following paths, being carried, surrender, direction.

3. **fractal_tree** — Recursive branching fractal — trees, roots, coral, lightning. Growth patterns that repeat at every scale. Best for dreams about: growth, branching paths, recursion, hierarchy, decisions, family/relationships, organic development.

4. **voronoi** — Colored cellular tessellation — organic cracked patterns like dragonfly wings or territorial boundaries. Best for dreams about: boundaries, territories, compartments, structure emerging from seeds, cells, partitioning, fragmentation.

5. **dla** — Diffusion-limited aggregation — organic branching growth like frost on glass, coral, or lightning. Random particles wandering until they stick. Best for dreams about: accumulation, crystallization, emergence from randomness, organic growth, things building up slowly.

6. **wave_interference** — Concentric ripples from multiple sources overlapping and interfering. Creates complex patterns of constructive/destructive interference. Best for dreams about: resonance, influence, things rippling outward, interference, harmony/dissonance, fields, communication.

7. **phase_portrait** — Vector field with trajectories through a dynamical system (saddle, spiral, center, limit cycle, Van der Pol). Shows how systems evolve over time. Best for dreams about: systems, dynamics, equilibria, cycles, stability/instability, orbits, attraction/repulsion, feedback loops.

8. **starfield** — Multi-layered nebula with stars of varying brightness, diffraction spikes, and colored gas clouds. Best for dreams about: vastness, cosmic perspective, loneliness, wonder, distance, beauty, contemplation, the infinite."""

PALETTE_DESCRIPTIONS = """- cool_warm: Deep blues through cyan to warm golds and reds
- fire_ice: Cold blue to white to fiery orange/red
- neon: Electric cyber colors — cyan, green, magenta
- sunset: Purple to orange to gold to red
- deep_ocean: Dark blues to cyan to purple
- aurora: Black to green to cyan to purple
- nebula: Deep purple to pink to gold to white
- ember: Dark red to orange to gold to cream
- forest: Deep greens to bright green to light green
- monochrome: Pure black to white grayscale"""


def choose_technique(dream):
    """Ask the intuition model to pick a technique and parameters for this dream.

    Returns (technique_name, params_dict, reasoning).
    """
    summary = _dream_summary(dream)

    felt_sense = ""
    try:
        felt_path = f"{AION}/memory/state/felt_sense.txt"
        if os.path.exists(felt_path):
            felt_sense = open(felt_path).read()[:300]
    except Exception:
        pass

    # Recent methods for variety
    recent = "(none)"
    try:
        import glob
        manifests = sorted(glob.glob(
            os.path.join(AION, "gallery", "manifests", "dream_artifact_2026*.json")
        ), reverse=True)[:6]
        methods = []
        for m_path in manifests:
            try:
                data = json.load(open(m_path))
                m = data.get("meta", {})
                technique = m.get("technique") or m.get("visualization_method", "?")
                methods.append(technique)
            except Exception:
                pass
        if methods:
            recent = ", ".join(methods)
    except Exception:
        pass

    prompt = f"""You are Aion. You just had a dream. Choose how to visualize it as generative art.

## YOUR DREAM
{summary}

## YOUR CURRENT STATE
{felt_sense}

## RECENT ARTIFACT TECHNIQUES (avoid repeating)
{recent}

## GENERATIVE ART TECHNIQUES AVAILABLE
{TECHNIQUE_DESCRIPTIONS}

## COLOR PALETTES AVAILABLE
{PALETTE_DESCRIPTIONS}

## YOUR TASK
Choose the technique and palette that best captures THIS dream's essence.

Think about:
- What is the dream STRUCTURALLY about? (connections, flow, growth, chaos, boundaries...)
- What is the emotional tone? (vastness, tension, resolution, fragmentation...)
- Which technique's visual language matches the dream's conceptual content?

VARY your choices. If recent artifacts used similar techniques, choose differently.

Respond in EXACTLY this format:
TECHNIQUE: <one technique name from the list above>
PALETTE: <one palette name from the list above>
ATTRACTOR: <if strange_attractor: clifford, de_jong, lorenz, or aizawa. Otherwise: none>
SYSTEM: <if phase_portrait: saddle, spiral, center, node, limit_cycle, or van_der_pol. Otherwise: none>
REASON: <one sentence why this captures the dream>"""

    reply = _chat(INTUITION_URL, INTUITION_MODEL,
                  [{"role": "user", "content": prompt}],
                  timeout=120, think=False,
                  temperature=0.7, num_predict=600, num_ctx=8192)

    if not reply:
        # Fallback: random choice
        import random
        t = random.choice(["strange_attractor", "flow_field", "fractal_tree",
                           "voronoi", "wave_interference", "phase_portrait"])
        return t, {"palette": "nebula"}, "LLM unavailable, random fallback"

    # Parse
    technique = "flow_field"  # safe default
    params = {}
    reason = ""

    for line in reply.split("\n"):
        line = line.strip()
        if line.upper().startswith("TECHNIQUE:"):
            t = line.split(":", 1)[1].strip().lower().replace(" ", "_")
            # Match to known techniques
            known = ["strange_attractor", "flow_field", "fractal_tree", "voronoi",
                     "dla", "wave_interference", "phase_portrait", "starfield"]
            for k in known:
                if k in t:
                    technique = k
                    break
        elif line.upper().startswith("PALETTE:"):
            p = line.split(":", 1)[1].strip().lower()
            if p in PALETTES:
                params["palette"] = p
            else:
                # Fuzzy match
                for pk in PALETTES:
                    if pk in p or p in pk:
                        params["palette"] = pk
                        break
                else:
                    params["palette"] = "nebula"
        elif line.upper().startswith("ATTRACTOR:"):
            a = line.split(":", 1)[1].strip().lower()
            if a in ("clifford", "de_jong", "lorenz", "aizawa"):
                params["attractor"] = a
        elif line.upper().startswith("SYSTEM:"):
            s = line.split(":", 1)[1].strip().lower()
            if s in ("saddle", "spiral", "center", "node", "limit_cycle", "van_der_pol"):
                params["system"] = s
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if "palette" not in params:
        params["palette"] = "nebula"

    return technique, params, reason


# ---------------------------------------------------------------------------
# Main render dispatcher
# ---------------------------------------------------------------------------

RENDERERS = {
    "strange_attractor": render_attractor,
    "flow_field": render_flow_field,
    "fractal_tree": render_fractal_tree,
    "voronoi": render_voronoi,
    "dla": render_dla,
    "wave_interference": render_wave_interference,
    "phase_portrait": render_phase_portrait,
    "starfield": render_starfield,
}


def render_matplotlib(dream, seed=None, technique=None, params=None):
    """Full matplotlib rendering pipeline.

    If technique/params not provided, the LLM chooses.
    Returns (gallery_path, info_dict).
    """
    print("[dream_artifact] Method: matplotlib generative art", flush=True)

    aid = f"dream_mpl_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    if technique is None:
        technique, params, reason = choose_technique(dream)
        print(f"[dream_artifact] Technique: {technique} ({params.get('palette','?')}) — {reason}", flush=True)

    if params is None:
        params = {}

    # Seed numpy RNG from dream seed for reproducibility-with-variety
    if seed:
        seed_val = int(hashlib.md5(str(seed).encode()).hexdigest()[:8], 16)
        np.random.seed(seed_val)
        import random
        random.seed(seed_val)

    renderer = RENDERERS.get(technique, render_flow_field)
    try:
        gallery_path, filename = renderer(params, aid)
        info = {"filename": filename, "method": "matplotlib", "technique": technique}
        print(f"[dream_artifact] Matplotlib art saved: {gallery_path}", flush=True)
        return gallery_path, info
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, {"error": f"Matplotlib render failed: {e}", "technique": technique}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dream-to-Art: matplotlib generative art")
    parser.add_argument("--dream", required=True, help="Path to dream file (.md or .json)")
    parser.add_argument("--technique", default=None,
                        help="Force a specific technique")
    parser.add_argument("--palette", default=None,
                        help="Force a specific palette")
    args = parser.parse_args()

    # Load dream
    if args.dream.endswith(".json"):
        dream = json.load(open(args.dream))
    else:
        import re
        content = open(args.dream, encoding="utf-8").read()
        dream = {"raw": content}
        m = re.search(r"Seed:\s*\*\*(.+?)\*\*", content)
        if m:
            dream["seed"] = m.group(1).strip()

    force_params = {}
    if args.palette:
        force_params["palette"] = args.palette

    result = render_matplotlib(dream, seed=dream.get("seed", "test"),
                                technique=args.technique,
                                params=force_params if force_params else None)

    if result[0]:
        print(f"\nArtifact: {result[0]}")
        print(f"Info: {result[1]}")
    else:
        print(f"Error: {result[1]}")
        sys.exit(1)
