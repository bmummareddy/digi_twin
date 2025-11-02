# -*- coding: utf-8 -*-
# digital_twin.py — STL slice → FOV → fast RSA packing → binder/void render (polydisperse & outline)

from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Polygon, Point, box  # imported lazily below if not installed
from shapely.ops import unary_union
from shapely import wkb

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

# ------------------------------- PSD sampling --------------------------------
def _psd_um(n: int, d50_um: float, cv_pct: float, seed: int) -> np.ndarray:
    """
    Lognormal PSD controlled by CV (%). CV=0 → monodisperse; higher CV → broader.
    """
    rng = np.random.default_rng(seed)
    cv = max(0.0, float(cv_pct)) / 100.0
    if cv <= 0:
        d = np.full(n, float(d50_um))
    else:
        sigma = float(np.sqrt(np.log(1.0 + cv**2)))
        d = float(d50_um) * rng.lognormal(mean=0.0, sigma=sigma, size=n)
    return np.clip(d, 0.30*float(d50_um), 3.00*float(d50_um))

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
    """
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        return []
    v = np.frombuffer(verts, dtype=np.float64).reshape(-1, 3)
    f = np.frombuffer(faces, dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    sec = m.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
    if sec is None:
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
    """Greedy RSA with a spatial grid and per-size attempt cap (fast)."""
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
def _scale_bar(ax, fov_mm: float, length_um: int = 500):
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
    st.subheader("Digital Twin — STL slice + qualitative packing")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        miss = []
        if not HAVE_TRIMESH: miss.append("trimesh")
        if not HAVE_SHAPELY: miss.append("shapely")
        st.error("Missing deps: " + ", ".join(miss))
        return

    # pull Top-5 (from your main app); we won't fail if absent
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

        pack_fit = st.selectbox(
            "FOV mode",
            ["Centered square (auto)", "Manual square", "Fit full XY bounds (not square)"],
            index=0,
            help="Choose how the field of view is defined on the slice."
        )
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 40.0, 8.0, 0.1,
                           disabled=(pack_fit != "Manual square"))
        pack_scope = st.selectbox(
            "Packing scope",
            ["Pack inside PART only", "Pack full SQUARE build box"],
            index=0,
            help="Packing in part preserves shape; build box fills the whole square."
        )

        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.95 * phi_TPD, 0.45, 0.90))

        psd_cv = st.slider("PSD spread (CV %)", 0, 60, 25, 5,
                           help="0 = mono; higher = broader particle-size spread")
        vis_scale = st.slider("Plot radius scale (visual-only)", 0.6, 1.6, 1.0, 0.05)

        fast = st.toggle("Fast mode", value=True, help="Limits tries/sizes for speed")

        show_outline = st.checkbox("Show part outline overlay", value=True)

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
        binder  = str(  row.get("binder_type", "water_based"))
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
        thickness = layer_r * (1e-3 if stl_units == "mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz - minz:.3f} {stl_units}")
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

    # slice → FOV window(s)
    dom = None
    polys_local: List[bytes] = []
    render_w = render_h = 0.0

    if mesh is not None and HAVE_SHAPELY:
        verts = mesh.vertices.astype(np.float64, copy=False)
        faces = mesh.faces.astype(np.int64, copy=False)
        key = (verts.size, faces.size)
        polys_wkb = _slice_at_z(key, verts.tobytes(), faces.tobytes(), float(z))

        if polys_wkb:
            dom = unary_union([wkb.loads(p) for p in polys_wkb])
            xmin, ymin, xmax, ymax = dom.bounds
            cx, cy = dom.centroid.x, dom.centroid.y

            if pack_fit == "Fit full XY bounds (not square)":
                win = box(xmin, ymin, xmax, ymax)
            elif pack_fit == "Manual square":
                half = float(fov_mm) / 2.0
                win  = box(cx - half, cy - half, cx + half, cy + half)
            else:  # Centered square (auto)
                side = max(xmax - xmin, ymax - ymin)
                half = side / 2.0
                win  = box(cx - half, cy - half, cx + half, cy + half)

            clipped = dom.intersection(win)
            geoms = [clipped] if isinstance(clipped, Polygon) else [g for g in clipped.geoms if isinstance(g, Polygon)]
            polys_clip_wkb = [g.wkb for g in geoms]

            ox, oy, x2, y2 = win.bounds
            render_w = x2 - ox
            render_h = y2 - oy

            def _to_local_wkb(pw):
                p = wkb.loads(pw)
                x, y = p.exterior.xy
                return Polygon(np.c_[np.array(x) - ox, np.array(y) - oy]).wkb

            polys_local = [_to_local_wkb(pw) for pw in polys_clip_wkb]
        else:
            side = 2.0
            polys_local = [box(0, 0, side, side).wkb]
            render_w = render_h = side
    else:
        side = 2.0
        polys_local = [box(0, 0, side, side).wkb]
        render_w = render_h = side

    # packing domain for "build box" mode
    if pack_scope.endswith("build box"):
        polys_local = [box(0, 0, render_w, render_h).wkb]

    # dynamic particle cap ~ (FOV / D50)^2 scaled
    est = (max(render_w, render_h) * 1000.0 / max(d50_r, 1e-6))**2 * 0.34
    cap = int(np.clip(est, 900, 5200))

    # PSD & units
    n_psd = 7000 if fast else 10000
    diam_um = _psd_um(n_psd, d50_r, cv_pct=psd_cv, seed=9991)
    unit_scale = (1e-3 if stl_units == "mm" else 1e-6)
    diam_units = diam_um * unit_scale

    centers, radii, phi2D = _pack(
        polys_local, diam_units, phi2D_target,
        max_particles=cap, max_trials=(180_000 if fast else 420_000),
        seed=20_000 + int(layer_idx),
    )

    # visual-only radius scale (does not affect packing / porosity)
    radii_vis = radii * float(vis_scale)

    # panels
    cA, cB = st.columns(2)

    with cA:
        figA, axA = plt.subplots(figsize=(5.2, 5.2), dpi=185)
        axA.add_patch(Rectangle((0, 0), render_w, render_h, facecolor="white",
                                edgecolor=COLOR_BORDER, linewidth=1.2))
        for (x, y), r in zip(centers, radii_vis):
            axA.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, edgecolor=COLOR_EDGE, linewidth=0.25))
        if show_outline and dom is not None:
            # draw outline in local coordinates
            ox0, oy0, _, _ = (dom.bounds[0], dom.bounds[1], dom.bounds[2], dom.bounds[3])
            outline = dom.buffer(0).exterior
            if outline is not None:
                xg, yg = outline.xy
                xg = np.array(xg) - (dom.bounds[0] if pack_fit!="Manual square" else (dom.centroid.x - render_w/2))
                yg = np.array(yg) - (dom.bounds[1] if pack_fit!="Manual square" else (dom.centroid.y - render_h/2))
                axA.plot(xg, yg, color="#222", lw=1.0, alpha=0.35)
        axA.set_aspect('equal', 'box'); axA.set_xlim(0, render_w); axA.set_ylim(0, render_h)
        axA.set_xticks([]); axA.set_yticks([])
        _scale_bar(axA, max(render_w, render_h))
        axA.set_title("Particles only", fontsize=12)
        st.pyplot(figA, use_container_width=True)

    with cB:
        px = 900
        pores = ~_bitmap_mask_vectorized(centers, radii_vis, max(render_w, render_h), px)
        vmask = _void_mask_from_saturation(pores, saturation=sat_pct/100.0, seed=1234 + int(layer_idx))

        figB, axB = plt.subplots(figsize=(5.2, 5.2), dpi=185)
        axB.add_patch(Rectangle((0, 0), render_w, render_h,
                                facecolor=_binder_hex(binder), edgecolor=COLOR_BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (max(render_w, render_h) / vmask.shape[1])
            ym = (vmask.shape[0] - ys) * (max(render_w, render_h) / vmask.shape[0])
            axB.scatter(xm, ym, s=0.30, c=COLOR_VOID, alpha=0.96, linewidth=0)
        for (x, y), r in zip(centers, radii_vis):
            axB.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, edgecolor=COLOR_EDGE, linewidth=0.25))
        axB.set_aspect('equal', 'box'); axB.set_xlim(0, render_w); axB.set_ylim(0, render_h)
        axB.set_xticks([]); axB.set_yticks([])
        _scale_bar(axB, max(render_w, render_h))
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=12)
        st.pyplot(figB, use_container_width=True)

    st.caption(
        f"FOV≈{max(render_w, render_h):.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · "
        f"φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-float(phi2D)):.2f}"
    )

    # comparison uses the same particle layout (so differences are binder/sat only)
    if top5 is not None and picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"] == rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0))
            bnd = str(  row.get("binder_type", "water_based"))
            px = 780
            pores = ~_bitmap_mask_vectorized(centers, radii_vis, max(render_w, render_h), px)
            vm = _void_mask_from_saturation(pores, saturation=sat/100.0, seed=987 + int(layer_idx))
            figC, axC = plt.subplots(figsize=(5.0, 5.0), dpi=185)
            axC.add_patch(Rectangle((0, 0), render_w, render_h,
                                    facecolor=_binder_hex(bnd), edgecolor=COLOR_BORDER, linewidth=1.2))
            ys, xs = np.where(vm)
            if len(xs):
                xm = xs * (max(render_w, render_h) / vm.shape[1])
                ym = (vm.shape[0] - ys) * (max(render_w, render_h) / vm.shape[0])
                axC.scatter(xm, ym, s=0.28, c=COLOR_VOID, alpha=0.96, linewidth=0)
            for (x, y), r in zip(centers, radii_vis):
                axC.add_patch(Circle((x, y), r, facecolor=COLOR_PARTICLE, edgecolor=COLOR_EDGE, linewidth=0.25))
            axC.set_aspect('equal', 'box'); axC.set_xlim(0, render_w); axC.set_ylim(0, render_h)
            axC.set_xticks([]); axC.set_yticks([])
            axC.set_title(f'{row["id"]}: {bnd} · Sat {int(sat)}%', fontsize=10)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
