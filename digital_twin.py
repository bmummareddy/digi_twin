# -*- coding: utf-8 -*-
# digital_twin.py
# Digital Twin module for BJAM - STL slicing, particle packing, and visualization
# Separate module to keep main app clean and modular

from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# Conditional imports for geometry
try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False

try:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    from shapely import wkb
    HAVE_SHAPELY = True
except ImportError:
    HAVE_SHAPELY = False

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ------------------------------- Colors & Constants ---------------------------
BINDER_COLORS = {
    "water_based": "#F2D06F",
    "solvent_based": "#F2B233",
    "furan": "#F5C07A",
    "acrylic": "#FFD166",
    "other": "#F4B942"
}
PARTICLE = "#2F6CF6"
EDGE = "#1f2937"
BORDER = "#111111"
VOID = "#FFFFFF"

def binder_color(name: str) -> str:
    """Get binder color from name"""
    key = (name or "").lower()
    for k, v in BINDER_COLORS.items():
        if k in key:
            return v
    return BINDER_COLORS["other"]

# ------------------------------- PSD Sampling ---------------------------------
@st.cache_data(show_spinner=False)
def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], 
                  d90_um: Optional[float], seed: int) -> np.ndarray:
    """Generate particle size distribution using lognormal"""
    rng = np.random.default_rng(seed)
    med = max(1e-9, float(d50_um))
    if d10_um and d90_um and d90_um > d10_um > 0:
        m = np.log(med)
        s = (np.log(d90_um) - np.log(d10_um)) / (2*1.2815515655446004)
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    d = np.exp(rng.normal(m, s, size=n))
    return np.clip(d, 0.30*med, 3.00*med)

# ------------------------------- Mesh Loading ---------------------------------
@st.cache_resource(show_spinner="Loading mesh...")
def load_mesh_from_bytes(file_bytes: bytes):
    """Load and cache mesh from STL bytes"""
    if not HAVE_TRIMESH:
        raise ImportError("trimesh required for STL loading")
    try:
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type="stl", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

@st.cache_resource(show_spinner="Creating cube...")
def get_cube_mesh():
    """Get built-in 10mm cube mesh"""
    if not HAVE_TRIMESH:
        raise ImportError("trimesh required for cube generation")
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

# ------------------------------- Mesh Slicing ---------------------------------
@st.cache_data(show_spinner=False)
def slice_mesh_at_z(_mesh_hash, mesh_verts, mesh_faces, z) -> List[bytes]:
    """
    Slice mesh at Z height and return WKB polygons.
    Cached per mesh + Z position.
    """
    if not HAVE_TRIMESH or not HAVE_SHAPELY:
        raise ImportError("trimesh and shapely required for slicing")
    
    try:
        mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
        
        # Try slicing with small tolerance offsets if exact fails
        sect = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
        if sect is None:
            for offset in [0.0001, -0.0001, 0.001, -0.001]:
                sect = mesh.section(plane_origin=(0, 0, z+offset), plane_normal=(0, 0, 1))
                if sect is not None:
                    break
        
        if sect is None:
            return []
        
        # Convert to 2D polygons
        planar, _ = sect.to_planar()
        polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        valid = [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
        return [p.wkb for p in valid]
    except Exception:
        return []

# ------------------------------- FOV Cropping ---------------------------------
@st.cache_data(show_spinner=False)
def crop_to_fov(_polys_wkb_tuple, fov_size):
    """Crop polygons to field of view"""
    if not HAVE_SHAPELY:
        raise ImportError("shapely required for FOV cropping")
    
    if not _polys_wkb_tuple:
        return [], (0.0, 0.0)
    
    polys = [wkb.loads(p) for p in _polys_wkb_tuple]
    dom = unary_union(polys)
    cx, cy = dom.centroid.x, dom.centroid.y
    half = fov_size / 2.0
    xmin, ymin = cx - half, cy - half
    win = box(xmin, ymin, xmin + fov_size, ymin + fov_size)
    res = dom.intersection(win)
    
    if getattr(res, "is_empty", True):
        return [], (xmin, ymin)
    
    geoms = [res] if isinstance(res, Polygon) else [g for g in res.geoms if isinstance(g, Polygon)]
    return [g.wkb for g in geoms], (xmin, ymin)

@st.cache_data(show_spinner=False)
def to_local_coords(_polys_wkb_tuple, origin_xy):
    """Convert polygons to local coordinate system"""
    if not HAVE_SHAPELY:
        raise ImportError("shapely required for coordinate conversion")
    
    if not _polys_wkb_tuple:
        return []
    
    ox, oy = origin_xy
    polys = [wkb.loads(p) for p in _polys_wkb_tuple]
    out = []
    for p in polys:
        x, y = p.exterior.xy
        out.append(Polygon(np.c_[np.array(x) - ox, np.array(y) - oy]).wkb)
    return out

# ------------------------------- Particle Packing -----------------------------
def pack_particles_no_cache(polys_wkb_list, diam_units, phi2D_target, 
                           max_particles, layer_idx):
    """
    Pack particles in polygon domain using RSA algorithm.
    NOT cached - computes fresh each time to ensure unique particles per layer.
    
    Args:
        polys_wkb_list: List of WKB-encoded polygons
        diam_units: Array of particle diameters (in mesh units)
        phi2D_target: Target 2D packing fraction
        max_particles: Maximum number of particles to place
        layer_idx: Layer index (used for random seed)
    
    Returns:
        centers: Nx2 array of particle centers
        radii: N array of particle radii
        phi2D: Achieved packing fraction
    """
    if not HAVE_SHAPELY:
        raise ImportError("shapely required for particle packing")
    
    if not polys_wkb_list:
        return np.empty((0, 2)), np.empty((0,)), 0.0
    
    # Deserialize polygons
    polys = [wkb.loads(p) for p in polys_wkb_list]
    dom_all = unary_union(polys)
    
    if dom_all.is_empty:
        return np.empty((0, 2)), np.empty((0,)), 0.0
    
    minx, miny, maxx, maxy = dom_all.bounds
    area_dom = dom_all.area
    
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    
    # CRITICAL: Unique random seed per layer
    rng = np.random.default_rng(20_000 + layer_idx)
    
    # Spatial grid for fast collision detection
    cell = max(diam.max() / 2.0, (maxx - minx + maxy - miny) / 400.0)
    grid: Dict[Tuple[int, int], List[int]] = {}
    
    def no_overlap(x, y, r):
        gx, gy = int(x // cell), int(y // cell)
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for j in grid.get((ix, iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return True
    
    trials = 0
    max_trials = 480_000
    
    for d in diam:
        r = d / 2.0
        fit_dom = dom_all.buffer(-r)
        if getattr(fit_dom, "is_empty", True):
            continue
        fminx, fminy, fmaxx, fmaxy = fit_dom.bounds
        
        for _ in range(600):
            trials += 1
            if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
                break
            
            x = rng.uniform(fminx, fmaxx)
            y = rng.uniform(fminy, fmaxy)
            
            if not fit_dom.contains(Point(x, y)):
                continue
            if not no_overlap(x, y, r):
                continue
            
            idx = len(placed_xy)
            placed_xy.append((x, y))
            placed_r.append(r)
            gx, gy = int(x // cell), int(y // cell)
            grid.setdefault((gx, gy), []).append(idx)
            area_circ += math.pi * r * r
        
        if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
            break
    
    centers = np.array(placed_xy) if placed_xy else np.empty((0, 2))
    radii = np.array(placed_r) if placed_r else np.empty((0,))
    phi2D = area_circ / area_dom if area_dom > 0 else 0.0
    
    return centers, radii, float(phi2D)

# ------------------------------- Rasterization --------------------------------
@st.cache_data(show_spinner=False)
def raster_particle_mask(_centers_hash, centers, radii, fov, px=900):
    """
    Rasterize particles to binary mask using vectorized numpy.
    10x faster than PIL-based approach.
    """
    y_grid, x_grid = np.mgrid[0:px, 0:px]
    sx = fov / px
    x_phys = x_grid * sx
    y_phys = (px - y_grid) * sx
    
    mask = np.zeros((px, px), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        dist_sq = (x_phys - cx)**2 + (y_phys - cy)**2
        mask |= (dist_sq <= r**2)
    
    return mask

# ------------------------------- Void Generation ------------------------------
@st.cache_data(show_spinner=False)
def generate_void_mask(_pore_hash, pore_mask, saturation, seed):
    """Generate void mask based on binder saturation"""
    rng = np.random.default_rng(seed)
    pore = int(pore_mask.sum())
    if pore <= 0:
        return np.zeros_like(pore_mask, bool)
    target = int(round((1.0 - saturation) * pore))
    if target <= 0:
        return np.zeros_like(pore_mask, bool)
    
    if HAVE_SCIPY:
        # Distance transform + noise approach
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18 * noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat) - target)[len(flat) - target]
        vm = np.zeros_like(pore_mask, bool)
        vm[pore_mask] = field[pore_mask] >= kth
        vm = ndi.binary_opening(vm, iterations=1)
        vm = ndi.binary_closing(vm, iterations=1)
        return vm
    
    # Fallback: random dots
    h, w = pore_mask.shape
    vm = np.zeros_like(pore_mask, bool)
    area, tries = 0, 0
    while area < target and tries < 120000:
        tries += 1
        r = int(np.clip(rng.normal(3.0, 1.2), 1.0, 6.0))
        x = rng.integers(r, w - r)
        y = rng.integers(r, h - r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            vm[add] = True
            area = int(vm.sum())
    return vm

# ------------------------------- Visualization Helpers ------------------------
def draw_scale_bar(ax, fov_mm, length_um=500):
    """Draw scale bar on matplotlib axis"""
    length_mm = length_um / 1000.0
    if length_mm >= fov_mm:
        return
    pad = 0.06 * fov_mm
    x0 = fov_mm - pad - length_mm
    x1 = fov_mm - pad
    y = pad * 0.6
    ax.plot([x0, x1], [y, y], lw=3.5, color="#111111")
    ax.text((x0 + x1) / 2, y + 0.02 * fov_mm, f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=9, color="#111111")

def render_particles_only(ax, centers, radii, fov):
    """Render particles-only view"""
    ax.add_patch(Rectangle((0, 0), fov, fov, facecolor="white", 
                          edgecolor=BORDER, linewidth=1.2))
    for (x, y), r in zip(centers, radii):
        ax.add_patch(Circle((x, y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, fov)
    ax.set_ylim(0, fov)
    ax.set_xticks([])
    ax.set_yticks([])
    draw_scale_bar(ax, fov)

def render_with_binder(ax, centers, radii, pore_mask, saturation, 
                      binder_hex, fov, seed_offset):
    """Render particles + binder + voids"""
    # Generate void mask
    pore_hash = hash((pore_mask.tobytes(), saturation, seed_offset))
    vmask = generate_void_mask(pore_hash, pore_mask, saturation, seed_offset)
    
    # Background: binder
    ax.add_patch(Rectangle((0, 0), fov, fov, facecolor=binder_hex, 
                          edgecolor=BORDER, linewidth=1.2))
    
    # Voids (white dots)
    ys, xs = np.where(vmask)
    if len(xs):
        xm = xs * (fov / vmask.shape[1])
        ym = (vmask.shape[0] - ys) * (fov / vmask.shape[0])
        ax.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidth=0)
    
    # Particles
    for (x, y), r in zip(centers, radii):
        ax.add_patch(Circle((x, y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, fov)
    ax.set_ylim(0, fov)
    ax.set_xticks([])
    ax.set_yticks([])
    draw_scale_bar(ax, fov)
    
    # Legend
    ax.text(0.02 * fov, 0.97 * fov, "SiC", color=PARTICLE, fontsize=9, va="top")
    ax.text(0.12 * fov, 0.97 * fov, "Binder", color="#805a00", fontsize=9, va="top")
    ax.text(0.24 * fov, 0.97 * fov, "Void", color="#666", fontsize=9, va="top")

# ------------------------------- Capability Checks ----------------------------
def check_dependencies():
    """Check if required dependencies are available"""
    missing = []
    if not HAVE_TRIMESH:
        missing.append("trimesh")
    if not HAVE_SHAPELY:
        missing.append("shapely")
    if not HAVE_SCIPY:
        missing.append("scipy (optional, but recommended)")
    return missing
