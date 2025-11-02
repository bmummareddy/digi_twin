# -*- coding: utf-8 -*-
# digital_twin.py — STL slice → polygon window → particle packing → binder/void render
from __future__ import annotations
import io, math, hashlib
from typing import List, Tuple, Dict, Optional
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw
import streamlit as st

# Optional geometry deps
try:
    import trimesh
    HAVE_TRIMESH = True
except Exception:
    HAVE_TRIMESH = False

try:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    from shapely import wkb as shp_wkb
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

try:
    from scipy import ndimage as ndi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Colors
PARTICLE = "#2F6CF6"; EDGE = "#1f2937"; BORDER = "#111111"; VOID = "#FFFFFF"
BINDER = {"water_based": "#F2D06F", "solvent_based": "#F2B233", "other": "#F4B942"}
def binder_hex(name: str) -> str:
    k = (name or "").lower()
    if "water" in k: return BINDER["water_based"]
    if "solvent" in k: return BINDER["solvent_based"]
    return BINDER["other"]

# ---------------- Mesh helpers (no st.cache_*) ----------------
def _mesh_key(vertices: np.ndarray, faces: np.ndarray) -> str:
    return hashlib.sha1(vertices[:64].tobytes() + faces[:64].tobytes()).hexdigest()

def load_mesh_from_bytes(data: bytes):
    if not HAVE_TRIMESH:
        raise RuntimeError("trimesh not installed")
    m = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    return m

def cube_mesh():
    if not HAVE_TRIMESH:
        raise RuntimeError("trimesh not installed")
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

@lru_cache(maxsize=64)
def slice_at_z(mesh_key: str, verts_bytes: bytes, faces_bytes: bytes, z: float) -> List[bytes]:
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        return []
    verts = np.frombuffer(verts_bytes, dtype=np.float64).reshape(-1, 3)
    faces = np.frombuffer(faces_bytes, dtype=np.int64).reshape(-1, 3)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
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
    valid = [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-9]
    return [p.wkb for p in valid]

# ---------------- PSD / packing ----------------
def sample_psd_um(n: int, d50_um: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, sigma = np.log(max(d50_um, 1e-9)), 0.25
    d = np.exp(rng.normal(mu, sigma, size=n))
    return np.clip(d, 0.30 * d50_um, 3.0 * d50_um)

def pack_particles(polys_wkb: List[bytes], diam_units: np.ndarray, phi_target: float,
                   cap: int, layer_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    if not HAVE_SHAPELY or not polys_wkb:
        return np.empty((0, 2)), np.empty((0,)), 0.0
    polys = [shp_wkb.loads(p) for p in polys_wkb]
    dom = unary_union(polys)
    if getattr(dom, "is_empty", True):
        return np.empty((0, 2)), np.empty((0,)), 0.0

    minx, miny, maxx, maxy = dom.bounds
    area_dom = dom.area
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ = 0.0
    target_area = float(np.clip(phi_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(20_000 + int(layer_idx))

    cell = max(diam.max() / 2.0, (maxx - minx + maxy - miny) / 450.0)
    grid: Dict[Tuple[int, int], List[int]] = {}

    def ok(x, y, r) -> bool:
        gx, gy = int(x // cell), int(y // cell)
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for j in grid.get((ix, iy), []):
                    dx = x - placed_xy[j][0]; dy = y - placed_xy[j][1]
                    if dx * dx + dy * dy < (r + placed_r[j]) ** 2:
                        return False
        return True

    trials = 0; MAX_TRIALS = 200_000
    for d in diam:
        r = d / 2.0
        fit = dom.buffer(-r)
        if getattr(fit, "is_empty", True): continue
        fx0, fy0, fx1, fy1 = fit.bounds
        for _ in range(480):
            trials += 1
            if trials > MAX_TRIALS or area_circ >= target_area or len(placed_xy) >= cap:
                break
            x = rng.uniform(fx0, fx1); y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x, y)) or not ok(x, y, r): continue
            idx = len(placed_xy)
            placed_xy.append((x, y)); placed_r.append(r)
            gx, gy = int(x // cell), int(y // cell)
            grid.setdefault((gx, gy), []).append(idx)
            area_circ += math.pi * r * r
        if trials > MAX_TRIALS or area_circ >= target_area or len(placed_xy) >= cap:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0, 2))
    radii = np.array(placed_r) if placed_r else np.empty((0,))
    phi2D = area_circ / area_dom if area_dom > 0 else 0.0
    return centers, radii, float(phi2D)

# ---------------- raster + voids ----------------
def bitmap_mask(centers: np.ndarray, radii: np.ndarray, fov: float, px: int = 900) -> np.ndarray:
    img = Image.new("L", (px, px), color=0); d = ImageDraw.Draw(img)
    sx = px / fov; sy = px / fov
    for (x, y), r in zip(centers, radii):
        x0 = int((x - r) * sx); y0 = int((fov - (y + r)) * sy)
        x1 = int((x + r) * sx); y1 = int((fov - (y - r)) * sy)
        d.ellipse([x0, y0, x1, y1], fill=255)
    return (np.array(img) > 0)

def voids_from_saturation(pore_mask: np.ndarray, sat: float, seed: int = 0) -> np.ndarray:
    pore = int(pore_mask.sum())
    if pore <= 0: return np.zeros_like(pore_mask, bool)
    target = int(round((1.0 - float(sat)) * pore))
    if target <= 0: return np.zeros_like(pore_mask, bool)
    if HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(np.random.default_rng(seed).standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18 * noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat) - target)[len(flat) - target]
        vm = np.zeros_like(pore_mask, bool); vm[pore_mask] = field[pore_mask] >= kth
        vm = ndi.binary_opening(vm, iterations=1); vm = ndi.binary_closing(vm, iterations=1)
        return vm
    # dotted fallback
    h, w = pore_mask.shape; vm = np.zeros_like(pore_mask, bool)
    area = 0; tries = 0; rng = np.random.default_rng(seed)
    while area < target and tries < 90000:
        tries += 1
        r = int(np.clip(rng.normal(3.0, 1.2), 1.0, 6.0))
        x = rng.integers(r, w - r); y = rng.integers(r, h - r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx * xx + yy * yy) <= r * r
            add = np.logical_and(disk, pore_mask); vm[add] = True; area = int(vm.sum())
    return vm

def scale_bar(ax, fov_mm: float, length_um: int = 500):
    L = length_um / 1000.0
    if L >= fov_mm: return
    pad = 0.06 * fov_mm
    x0 = fov_mm - pad - L; x1 = fov_mm - pad; y = pad * 0.6
    ax.plot([x0, x1], [y, y], lw=3.0, color=BORDER)
    ax.text((x0 + x1) / 2, y + 0.02 * fov_mm, f"{int(length_um)} µm", ha="center", va="bottom", fontsize=9, color=BORDER)

# ---------------- public entry ----------------
def render(material: str, d50_um: float, layer_um: float):
    st.subheader("Digital Twin — STL slice + qualitative packing")
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Please add 'trimesh' and 'shapely' to requirements.txt to use this tab.")
        return

    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run Predict (Top-5) first.")
        return
    top5 = top5.reset_index(drop=True).copy()

    L, R = st.columns([1.2, 1.0])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm", "m"], index=0)
        um2unit = 1e-3 if stl_units == "mm" else 1e-6
        pack_full = st.checkbox("Pack full slice (bounding square)", value=True)
        auto_micro = st.checkbox("Auto micro-FOV to reach φ with cap", value=True)
        fov_mm = st.slider("Manual FOV (mm)", 0.2, 12.0, 1.5, 0.05, disabled=pack_full)
        phi_TPD = st.slider("Target φ (theoretical packing density)", 0.70, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90 * phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 400, 12000, 2400, 100)
        show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: up = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: fast = st.checkbox("Fast packing", value=True)

    rec = top5[top5["id"] == rec_id].iloc[0]
    d50_r = float(rec.get("d50_um", d50_um))
    layer_r = float(rec.get("layer_um", layer_um))

    n_psd = 7000 if fast else 10000
    diam_um = sample_psd_um(n_psd, d50_r, seed=9991)
    diam_units = diam_um * um2unit
    r2_mean = float(np.mean((0.5 * diam_units) ** 2)) if diam_units.size else 0.0

    mesh = None
    if use_cube:
        mesh = cube_mesh()
    elif up is not None:
        mesh = load_mesh_from_bytes(up.read())

    layer_idx = 1
    polys_local_wkb: List[bytes] = []
    render_fov = float(fov_mm)

    if mesh is not None:
        verts = mesh.vertices.astype(np.float64, copy=False)
        faces = mesh.faces.astype(np.int64, copy=False)
        key = _mesh_key(verts, faces)
        minz, maxz = float(np.min(verts[:, 2])), float(np.max(verts[:, 2]))
        thickness = layer_r * (1e-3 if stl_units == "mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz - minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness

        polys_wkb = slice_at_z(key, verts.tobytes(), faces.tobytes(), float(z))
        if not polys_wkb:
            st.warning("No section at this Z. Try a different layer.")
            return

        polys = [shp_wkb.loads(p) for p in polys_wkb]
        dom = unary_union(polys)
        xmin, ymin, xmax, ymax = dom.bounds
        fov_slice = max(xmax - xmin, ymax - ymin)

        if auto_micro and r2_mean > 0 and phi2D_target > 0:
            area_needed = (float(cap) * math.pi * r2_mean) / float(phi2D_target)
            fov_auto = float(np.sqrt(max(area_needed, 1e-12)))
            fov_pick = float(np.clip(fov_auto, 0.2, fov_slice))
            cx, cy = dom.centroid.x, dom.centroid.y
            half = fov_pick / 2.0
            win = box(cx - half, cy - half, cx + half, cy + half)
            clip = dom.intersection(win)
            polys_local_wkb = [clip.wkb] if getattr(clip, "area", 0) > 0 else []
            render_fov = fov_pick
        else:
            if pack_full:
                win = box(xmin, ymin, xmin + fov_slice, ymin + fov_slice)
                clip = dom.intersection(win)
                polys_local_wkb = [clip.wkb] if getattr(clip, "area", 0) > 0 else []
                render_fov = fov_slice
            else:
                cx, cy = dom.centroid.x, dom.centroid.y
                half = float(fov_mm) / 2.0
                win = box(cx - half, cy - half, cx + half, cy + half)
                clip = dom.intersection(win)
                polys_local_wkb = [clip.wkb] if getattr(clip, "area", 0) > 0 else []
                render_fov = float(fov_mm)

        if show_mesh:
            import plotly.graph_objects as go
            figm = go.Figure(data=[go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightgray", opacity=0.55, flatshading=True
            )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=360)
            st.plotly_chart(figm, use_container_width=True)
    else:
        if HAVE_SHAPELY:
            demo = box(0, 0, 1.8, 1.8)
            polys_local_wkb = [demo.wkb]
            render_fov = 1.8
        else:
            polys_local_wkb = []
            render_fov = float(fov_mm)

    centers, radii, phi2D = pack_particles(polys_local_wkb, diam_units, float(phi2D_target),
                                           cap=int(cap), layer_idx=int(layer_idx))

    rec = top5[top5["id"] == rec_id].iloc[0]
    sat_pct = float(rec.get("saturation_pct", 80.0))
    binder = str(rec.get("binder_type", "water_based"))

    cA, cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.2, 5.2), dpi=185)
        axA.add_patch(Rectangle((0, 0), render_fov, render_fov, facecolor="white",
                                edgecolor=BORDER, linewidth=1.2))
        for (x, y), r in zip(centers, radii):
            axA.add_patch(Circle((x, y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        axA.set_aspect('equal', 'box'); axA.set_xlim(0, render_fov); axA.set_ylim(0, render_fov)
        axA.set_xticks([]); axA.set_yticks([])
        scale_bar(axA, render_fov); axA.set_title("Particles only", fontsize=11)
        st.pyplot(figA, use_container_width=True)

    with cB:
        pores = ~bitmap_mask(centers, radii, render_fov, px=900)
        vm = voids_from_saturation(pores, sat=sat_pct/100.0, seed=123 + int(layer_idx))
        figB, axB = plt.subplots(figsize=(5.2, 5.2), dpi=185)
        axB.add_patch(Rectangle((0, 0), render_fov, render_fov, facecolor=binder_hex(binder),
                                edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vm)
        if len(xs):
            xm = xs * (render_fov / vm.shape[1]); ym = (vm.shape[0] - ys) * (render_fov / vm.shape[0])
            axB.scatter(xm, ym, s=0.30, c=VOID, alpha=0.96, linewidth=0)
        for (x, y), r in zip(centers, radii):
            axB.add_patch(Circle((x, y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        axB.set_aspect('equal', 'box'); axB.set_xlim(0, render_fov); axB.set_ylim(0, render_fov)
        axB.set_xticks([]); axB.set_yticks([])
        scale_bar(axB, render_fov); axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=11)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"] == rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0)); hexc = binder_hex(str(row.get("binder_type", "water_based")))
            pores = ~bitmap_mask(centers, radii, render_fov, px=800)
            vm = voids_from_saturation(pores, sat=sat/100.0, seed=987 + int(layer_idx) + i)
            figC, axC = plt.subplots(figsize=(4.9, 4.9), dpi=185)
            axC.add_patch(Rectangle((0, 0), render_fov, render_fov, facecolor=hexc,
                                    edgecolor=BORDER, linewidth=1.2))
            ys, xs = np.where(vm)
            if len(xs):
                xm = xs * (render_fov / vm.shape[1]); ym = (vm.shape[0] - ys) * (render_fov / vm.shape[0])
                axC.scatter(xm, ym, s=0.30, c=VOID, alpha=0.96, linewidth=0)
            for (x, y), r in zip(centers, radii):
                axC.add_patch(Circle((x, y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
            axC.set_aspect('equal', 'box'); axC.set_xlim(0, render_fov); axC.set_ylim(0, render_fov)
            axC.set_xticks([]); axC.set_yticks([])
            axC.set_title(f'{row["id"]}: {row["binder_type"]} · Sat {int(sat)}%', fontsize=10)
            with cols[i]: st.pyplot(figC, use_container_width=True)
