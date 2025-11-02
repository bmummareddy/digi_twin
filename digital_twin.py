# -*- coding: utf-8 -*-
# digital_twin.py — OPTIMIZED: Full STL view + visible particle size distribution
# Fixes: (1) Shows entire cross-section with 10% margin, (2) Wider PSD (sigma=0.40) for visual variety

from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image

# --------------------------- Optional/required deps ---------------------------
try:
    import trimesh
    HAVE_TRIMESH = True
except Exception:
    HAVE_TRIMESH = False

try:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    from shapely import wkb
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ------------------------------- Colors & UI ---------------------------------
BINDER_COLORS = {
    "water_based":  "#F2D06F",
    "solvent_based":"#F2B233",
    "furan":        "#F5C07A",
    "acrylic":      "#FFD166",
    "other":        "#F4B942",
}
COLOR_PARTICLE = "#2F6CF6"
COLOR_EDGE     = "#1f2937"
COLOR_BORDER   = "#111111"
COLOR_VOID     = "#FFFFFF"

def _binder_hex(name: str) -> str:
    k = (name or "").lower()
    if "water"   in k: return BINDER_COLORS["water_based"]
    if "solvent" in k: return BINDER_COLORS["solvent_based"]
    return BINDER_COLORS["other"]

# ------------------------------- PSD sampling (OPTIMIZED) --------------------
def _psd_um(n: int, d50_um: float, seed: int) -> np.ndarray:
    """
    Generate particle size distribution with WIDER variance for visual variety.
    Uses lognormal with sigma=0.40 (was 0.25) and wider clipping range.
    """
    rng = np.random.default_rng(seed)
    mu = np.log(max(d50_um, 1e-6))
    sigma = 0.40  # INCREASED from 0.25 for more visible size variation
    d = np.exp(rng.normal(mu, sigma, size=n))
    # WIDER clipping range: 0.15x to 4.0x (was 0.30x to 3.0x)
    return np.clip(d, 0.15*d50_um, 4.0*d50_um)

# ----------------------------- Mesh I/O + slicing ----------------------------
@st.cache_resource(show_spinner=False)
def _load_mesh_from_bytes(file_bytes: bytes):
    if not HAVE_TRIMESH:
        raise RuntimeError("trimesh not installed")
    m = trimesh.load(io.BytesIO(file_bytes), file_type="stl", force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    return m

@st.cache_resource(show_spinner=False)
def _cube_mesh():
    if not HAVE_TRIMESH:
        raise RuntimeError("trimesh not installed")
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

@st.cache_data(show_spinner=False)
def _slice_at_z(mesh_key: Tuple[int,int], verts: bytes, faces: bytes, z: float) -> List[bytes]:
    """
    Return a list of WKB polygons for the cross-section at z.
    Inputs are bytes so Streamlit can cache.
    """
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        return []
    v = np.frombuffer(verts, dtype=np.float64).reshape(-1, 3)
    f = np.frombuffer(faces, dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    sec = m.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
    if sec is None:
        # try tiny offsets to catch exact-vertex planes
        for off in (1e-4, -1e-4, 1e-3, -1e-3):
            sec = m.section(plane_origin=(0, 0, z + off), plane_normal=(0, 0, 1))
            if sec is not None:
                break
    if sec is None:
        return []
    planar, _ = sec.to_planar()
    polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    valid = [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
    return [p.wkb for p in valid]

# ------------------------------- Packing utils --------------------------------
def _bitmap_mask_vectorized(centers: np.ndarray, radii: np.ndarray, fov: float, px: int=900) -> np.ndarray:
    """Vectorized raster of circles → boolean mask (True = particle)."""
    if len(centers) == 0:
        return np.zeros((px, px), dtype=bool)
    yy, xx = np.mgrid[0:px, 0:px]
    sx = fov / px
    x_phys = xx * sx
    y_phys = (px - yy) * sx
    mask = np.zeros((px, px), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        mask |= (x_phys - cx)**2 + (y_phys - cy)**2 <= r**2
    return mask

def _void_mask_from_saturation(pore_mask: np.ndarray, saturation: float, seed: int) -> np.ndarray:
    """Generate white voids (unfilled pores) to match (1 - saturation) of pore area."""
    pore = int(pore_mask.sum())
    if pore <= 0:
        return np.zeros_like(pore_mask, bool)
    target = int(round((1.0 - float(saturation)) * pore))
    if target <= 0:
        return np.zeros_like(pore_mask, bool)

    if HAVE_SCIPY:
        rng = np.random.default_rng(seed)
        dist  = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18 * noise
        flat = field[pore_mask]
        kth  = np.partition(flat, len(flat) - target)[len(flat) - target]
        vm   = np.zeros_like(pore_mask, bool)
        vm[pore_mask] = field[pore_mask] >= kth
        vm = ndi.binary_opening(vm, iterations=1)
        vm = ndi.binary_closing(vm, iterations=1)
        return vm

    # Fallback: dotted
    h, w = pore_mask.shape
    rng = np.random.default_rng(seed)
    vm = np.zeros_like(pore_mask, bool)
    area = 0
    tries = 0
    while area < target and tries < 120_000:
        tries += 1
        r = int(np.clip(rng.normal(3.0, 1.2), 1.0, 6.0))
        x = rng.integers(r, w - r)
        y = rng.integers(r, h - r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h - y, -x:w - x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            vm[add] = True
            area = int(vm.sum())
    return vm

def _pack(polys_wkb_list: List[bytes], diam_units: np.ndarray, phi_target: float,
          max_particles: int, max_trials: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Greedy RSA with spatial grid and adaptive attempt cap."""
    if not (HAVE_SHAPELY and polys_wkb_list):
        return np.empty((0, 2)), np.empty((0,)), 0.0

    dom = unary_union([wkb.loads(p) for p in polys_wkb_list])
    if getattr(dom, "is_empty", True):
        return np.empty((0, 2)), np.empty((0,)), 0.0

    minx, miny, maxx, maxy = dom.bounds
    area_dom = max(dom.area, 1e-12)

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy: List[Tuple[float, float]] = []
    placed_r:  List[float] = []
    area_circ = 0.0
    target_area = float(np.clip(phi_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)

    med_r = float(np.median(diam)) / 2.0
    cell = max(med_r / 1.5, (maxx - minx + maxy - miny) / 480.0)
    grid: Dict[Tuple[int, int], List[int]] = {}

    def _free(x: float, y: float, r: float) -> bool:
        gx, gy = int(x // cell), int(y // cell)
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for j in grid.get((ix, iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return True

    trials = 0
    for d in diam:
        if area_circ >= target_area or len(placed_xy) >= max_particles or trials >= max_trials:
            break
        r = d / 2.0
        fit = dom.buffer(-r)
        if getattr(fit, "is_empty", True):
            continue
        fx0, fy0, fx1, fy1 = fit.bounds

        # Adaptive attempts based on particle size
        per_size = 360 if r >= med_r else 200
        if (target_area - area_circ) / target_area < 0.08:
            per_size = 120

        for _ in range(per_size):
            trials += 1
            if area_circ >= target_area or len(placed_xy) >= max_particles or trials >= max_trials:
                break
            x = rng.uniform(fx0, fx1)
            y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x, y)):
                continue
            if not _free(x, y, r):
                continue
            idx = len(placed_xy)
            placed_xy.append((x, y))
            placed_r.append(r)
            gx, gy = int(x // cell), int(y // cell)
            grid.setdefault((gx, gy), []).append(idx)
            area_circ += math.pi * r * r

    centers = np.array(placed_xy) if placed_xy else np.empty((0, 2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi     = area_circ / area_dom
    return centers, radii, float(phi)

# ----------------------------- Plot helpers -----------------------------------
def _scale_bar(ax, fov_mm: float, length_um: int = 100):
    length_mm = length_um / 1000.0
    if length_mm >= fov_mm:
        return
    pad = 0.06 * fov_mm
    x0 = fov_mm - pad - length_mm
    x1 = fov_mm - pad
    y = pad * 0.6
    ax.plot([x0, x1], [y, y], lw=3.0, color=COLOR_BORDER)
    ax.text((x0 + x1) / 2, y + 0.02 * fov_mm, f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=9, color=COLOR_BORDER)

# --------------------------------- Public UI ----------------------------------
def render(material: str, d50_um: float, layer_um: float):
    """Render the Digital Twin tab; callable from streamlit_app.py"""
    st.subheader("Digital Twin — Full STL view + particle size distribution")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        miss = []
        if not HAVE_TRIMESH: miss.append("trimesh")
        if not HAVE_SHAPELY: miss.append("shapely")
        st.error("Missing deps: " + ", ".join(miss))
        return

    # pull Top-5 (from main app)
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is not None and not getattr(top5, "empty", True):
        top5 = top5.reset_index(drop=True)
    else:
        top5 = None

    L, R = st.columns([1.2, 1])
    with L:
        if top5 is not None:
            rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
            picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
        else:
            st.info("No Top-5 in session; using sidebar values.")
            rec_id, picks = None, []

    with R:
        stl_units = st.selectbox("STL units", ["mm", "m"], index=0)
        um2unit = 1e-3 if stl_units == "mm" else 1e-6

        view_mode = st.selectbox(
            "View mode",
            ["Full cross-section (auto FOV)", "Manual FOV crop"],
            index=0,
            help="Full cross-section shows entire part slice; Manual FOV lets you zoom in"
        )
        
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 20.0, 3.0, 0.05, 
                          disabled=(view_mode.startswith("Full")))

        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.95 * phi_TPD, 0.45, 0.90))

        fast = st.toggle("Fast mode (coarser packing)", value=True, 
                        help="Limits tries and sizes to speed up")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2:
        use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3:
        show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    # recipe params
    if top5 is not None and rec_id is not None:
        row = top5[top5["id"] == rec_id].iloc[0]
        d50_r   = float(row.get("d50_um",  d50_um))
        layer_r = float(row.get("layer_um", layer_um))
        sat_pct = float(row.get("saturation_pct", 80.0))
        binder  = str(row.get("binder_type", "water_based"))
    else:
        d50_r, layer_r, sat_pct, binder = float(d50_um), float(layer_um), 80.0, "water_based"

    # mesh load
    mesh = None
    if use_cube:
        mesh = _cube_mesh()
    elif stl_file is not None:
        mesh = _load_mesh_from_bytes(stl_file.read())

    # layer selection
    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * um2unit
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"**Layers: {n_layers}** · Z span: [{minz:.3f}, {maxz:.3f}] {stl_units} · Thickness: {thickness:.4f}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness
    else:
        n_layers, layer_idx, z = 1, 1, 0.0
        st.info("No STL — using square window fallback.")

    # optional mesh preview
    if mesh is not None and show_mesh:
        import plotly.graph_objects as go
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # ==================== OPTIMIZED FOV CALCULATION ====================
    # FIX: Create bounding square that shows ENTIRE cross-section
    if mesh is not None and HAVE_SHAPELY:
        verts = mesh.vertices.astype(np.float64, copy=False)
        faces = mesh.faces.astype(np.int64, copy=False)
        key = (verts.size, faces.size)
        polys_wkb = _slice_at_z(key, verts.tobytes(), faces.tobytes(), float(z))

        if polys_wkb:
            dom = unary_union([wkb.loads(p) for p in polys_wkb])
            xmin, ymin, xmax, ymax = dom.bounds
            
            if view_mode.startswith("Full"):
                # OPTIMIZED: Use actual bounding box with 10% margin
                width = xmax - xmin
                height = ymax - ymin
                side = max(width, height) * 1.10  # 10% margin for visibility
                
                # Center the square on the part centroid
                cx, cy = dom.centroid.x, dom.centroid.y
                half = side / 2.0
                win = box(cx - half, cy - half, cx + half, cy + half)
                
                # Clip part to window
                clipped = dom.intersection(win)
            else:
                # Manual FOV mode
                cx, cy = dom.centroid.x, dom.centroid.y
                half_fov = fov_mm / 2.0
                win = box(cx - half_fov, cy - half_fov, cx + half_fov, cy + half_fov)
                clipped = dom.intersection(win)
            
            # Convert to list of polygons
            geoms = [clipped] if isinstance(clipped, Polygon) else \
                    [g for g in clipped.geoms if isinstance(g, Polygon)]
            polys_clip_wkb = [g.wkb for g in geoms]

            # Convert to local coordinates (origin at window lower-left)
            ox, oy = win.bounds[0], win.bounds[1]
            def _to_local_wkb(pw):
                p = wkb.loads(pw)
                x, y = p.exterior.xy
                return Polygon(np.c_[np.array(x) - ox, np.array(y) - oy]).wkb

            polys_local = [_to_local_wkb(pw) for pw in polys_clip_wkb]
            render_fov = float(win.bounds[2] - win.bounds[0])
            
            st.success(f"✅ Layer {layer_idx}: Showing full cross-section ({render_fov:.2f} mm FOV)")
        else:
            # Empty slice
            st.warning(f"⚠️ No geometry at layer {layer_idx} (Z={z:.4f})")
            side = 2.0
            polys_local = [box(0, 0, side, side).wkb]
            render_fov = side
    else:
        # No mesh - fallback
        side = float(fov_mm if view_mode.startswith("Manual") else 2.0)
        polys_local = [box(0, 0, side, side).wkb]
        render_fov = side

    # ==================== PARTICLE PACKING ====================
    # Dynamic particle cap based on FOV and D50
    est = (render_fov * 1000.0 / max(d50_r, 1e-6))**2 * 0.35
    cap = int(np.clip(est, 1000, 5000))

    # Generate PSD with WIDER distribution for visual variety
    n_psd = 8000 if fast else 12000
    diam_um = _psd_um(n_psd, d50_r, seed=9991)
    diam_units = diam_um * um2unit

    # Pack particles
    centers, radii, phi2D = _pack(
        polys_local, diam_units, phi2D_target,
        max_particles=cap, max_trials=(220_000 if fast else 500_000),
        seed=20_000 + int(layer_idx),
    )

    if len(centers) == 0:
        st.error("⚠️ No particles packed! Try increasing FOV or reducing D50.")
        return

    # Show particle size statistics
    if len(radii) > 0:
        diams_placed = radii * 2.0 * (1000.0 / um2unit)  # Convert to µm
        st.caption(f"**Particles placed: {len(centers)}** · "
                  f"Sizes: {diams_placed.min():.1f}-{diams_placed.max():.1f} µm · "
                  f"Median: {np.median(diams_placed):.1f} µm")

    # ==================== VISUALIZATION ====================
    cA, cB = st.columns(2)

    with cA:
        figA, axA = plt.subplots(figsize=(5.3, 5.3), dpi=190)
        axA.add_patch(Rectangle((0, 0), render_fov, render_fov, facecolor="white",
                                edgecolor=COLOR_BORDER, linewidth=1.2))
        for (x, y), r in zip(centers, radii):
            axA.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, 
                                edgecolor=COLOR_EDGE, linewidth=0.25))
        axA.set_aspect('equal', 'box')
        axA.set_xlim(0, render_fov)
        axA.set_ylim(0, render_fov)
        axA.set_xticks([])
        axA.set_yticks([])
        _scale_bar(axA, render_fov)
        axA.set_title("Particles only", fontsize=11)
        st.pyplot(figA, use_container_width=True)

    with cB:
        px = 900
        pores = ~_bitmap_mask_vectorized(centers, radii, render_fov, px)
        vmask = _void_mask_from_saturation(pores, saturation=sat_pct/100.0, 
                                           seed=1234 + int(layer_idx))

        figB, axB = plt.subplots(figsize=(5.3, 5.3), dpi=190)
        axB.add_patch(Rectangle((0, 0), render_fov, render_fov,
                                facecolor=_binder_hex(binder), 
                                edgecolor=COLOR_BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (render_fov / vmask.shape[1])
            ym = (vmask.shape[0] - ys) * (render_fov / vmask.shape[0])
            axB.scatter(xm, ym, s=0.32, c=COLOR_VOID, alpha=0.96, linewidth=0)
        for (x, y), r in zip(centers, radii):
            axB.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, 
                                edgecolor=COLOR_EDGE, linewidth=0.25))
        axB.set_aspect('equal', 'box')
        axB.set_xlim(0, render_fov)
        axB.set_ylim(0, render_fov)
        axB.set_xticks([])
        axB.set_yticks([])
        _scale_bar(axB, render_fov)
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=11)
        # Add legend
        axB.text(0.02*render_fov, 0.97*render_fov, "SiC", 
                color=COLOR_PARTICLE, fontsize=9, va="top")
        axB.text(0.12*render_fov, 0.97*render_fov, "Binder", 
                color="#805a00", fontsize=9, va="top")
        axB.text(0.24*render_fov, 0.97*render_fov, "Void", 
                color="#666", fontsize=9, va="top")
        st.pyplot(figB, use_container_width=True)

    st.caption(
        f"⚡ **Layer {layer_idx}** · FOV={render_fov:.2f} mm · "
        f"φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · "
        f"Porosity₂D≈{max(0.0,1.0-float(phi2D)):.2f}"
    )

    # ==================== COMPARISON VIEW ====================
    if top5 is not None and picks and len(picks) > 0:
        st.subheader("Compare trials")
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"] == rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0))
            bnd = str(row.get("binder_type", "water_based"))
            
            px = 800
            pores = ~_bitmap_mask_vectorized(centers, radii, render_fov, px)
            vm = _void_mask_from_saturation(pores, saturation=sat/100.0, 
                                           seed=987 + int(layer_idx) + i)
            
            figC, axC = plt.subplots(figsize=(5.0, 5.0), dpi=180)
            axC.add_patch(Rectangle((0, 0), render_fov, render_fov,
                                    facecolor=_binder_hex(bnd), 
                                    edgecolor=COLOR_BORDER, linewidth=1.2))
            ys, xs = np.where(vm)
            if len(xs):
                xm = xs * (render_fov / vm.shape[1])
                ym = (vm.shape[0] - ys) * (render_fov / vm.shape[0])
                axC.scatter(xm, ym, s=0.30, c=COLOR_VOID, alpha=0.96, linewidth=0)
            for (x, y), r in zip(centers, radii):
                axC.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, 
                                    edgecolor=COLOR_EDGE, linewidth=0.25))
            axC.set_aspect('equal', 'box')
            axC.set_xlim(0, render_fov)
            axC.set_ylim(0, render_fov)
            axC.set_xticks([])
            axC.set_yticks([])
            axC.set_title(f'{row["id"]}: {bnd} · Sat {int(sat)}%', fontsize=10)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
