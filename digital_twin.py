# -*- coding: utf-8 -*-
# digital_twin.py — STL slicing, particle packing, and visualization (cached)
from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# ------------------------------- Optional deps -------------------------------
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

# ------------------------------- Colors & constants --------------------------
BINDER_COLORS = {
    "water_based":  "#F2D06F",
    "solvent_based":"#F2B233",
    "furan":        "#F5C07A",
    "acrylic":      "#FFD166",
    "other":        "#F4B942",
}
PARTICLE = "#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"

def binder_color(name: str) -> str:
    key = (name or "").lower()
    for k, v in BINDER_COLORS.items():
        if k in key: return v
    return BINDER_COLORS["other"]

# ------------------------------- Capability check ----------------------------
def check_dependencies():
    missing = []
    if not HAVE_TRIMESH: missing.append("trimesh")
    if not HAVE_SHAPELY: missing.append("shapely")
    # SciPy optional (faster voids)
    return missing

# ------------------------------- PSD sampling --------------------------------
@st.cache_data(show_spinner=False)
def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
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

# ------------------------------- Mesh loading --------------------------------
@st.cache_resource(show_spinner="Loading mesh…")
def load_mesh_from_bytes(file_bytes: bytes):
    if not HAVE_TRIMESH:
        raise ImportError("trimesh required for STL loading")
    m = trimesh.load(io.BytesIO(file_bytes), file_type="stl", force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    return m

@st.cache_resource(show_spinner="Creating cube…")
def get_cube_mesh():
    if not HAVE_TRIMESH:
        raise ImportError("trimesh required for cube generation")
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

# ------------------------------- Slicing & FOV --------------------------------
@st.cache_data(show_spinner=False)
def slice_mesh_at_z(mesh_hash: int, verts: np.ndarray, faces: np.ndarray, z: float) -> List[bytes]:
    """Return WKB polygons of the cross-section at z."""
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        return []
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    sect = mesh.section(plane_origin=(0,0,float(z)), plane_normal=(0,0,1))
    if sect is None:
        for dz in (1e-4, -1e-4, 1e-3, -1e-3):
            sect = mesh.section(plane_origin=(0,0,float(z)+dz), plane_normal=(0,0,1))
            if sect is not None: break
    if sect is None: return []
    planar,_ = sect.to_planar()
    polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    valid = [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
    return [p.wkb for p in valid]

@st.cache_data(show_spinner=False)
def crop_to_fov(polys_wkb: Tuple[bytes, ...], fov_size: float):
    if not HAVE_SHAPELY or not polys_wkb:
        return [], (0.0, 0.0)
    polys = [shp_wkb.loads(p) for p in polys_wkb]
    dom = unary_union(polys)
    cx, cy = dom.centroid.x, dom.centroid.y
    half = float(fov_size)/2.0
    xmin, ymin = cx-half, cy-half
    win = box(xmin, ymin, xmin+fov_size, ymin+fov_size)
    res = dom.intersection(win)
    if getattr(res, "is_empty", True):
        return [], (xmin, ymin)
    geoms = [res] if isinstance(res, Polygon) else [g for g in res.geoms if isinstance(g, Polygon)]
    return [g.wkb for g in geoms], (xmin, ymin)

@st.cache_data(show_spinner=False)
def to_local_coords(polys_wkb: Tuple[bytes, ...], origin_xy: Tuple[float, float]):
    if not HAVE_SHAPELY or not polys_wkb:
        return []
    ox, oy = origin_xy
    out = []
    for p in polys_wkb:
        poly = shp_wkb.loads(p)
        x, y = poly.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-ox, np.array(y)-oy]).wkb)
    return out

# ------------------------------- Packing --------------------------------------
def pack_particles_no_cache(polys_wkb: List[bytes], diam_units: np.ndarray, phi2D_target: float,
                            max_particles: int, layer_idx: int):
    """Greedy RSA with grid accel. No cache → unique per layer."""
    if not HAVE_SHAPELY or not polys_wkb:
        return np.empty((0,2)), np.empty((0,)), 0.0
    polys = [shp_wkb.loads(p) for p in polys_wkb]
    dom_all = unary_union(polys)
    if dom_all.is_empty: return np.empty((0,2)), np.empty((0,)), 0.0

    minx, miny, maxx, maxy = dom_all.bounds
    area_dom = dom_all.area
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy: List[Tuple[float,float]] = []; placed_r: List[float] = []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(20000 + int(layer_idx))

    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/450.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def ok(x, y, r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1, gx+2):
            for iy in range(gy-1, gy+2):
                for j in grid.get((ix, iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return True

    trials = 0; MAX_TRIALS = 480_000
    for d in diam:
        r = d/2.0
        fit = dom_all.buffer(-r)
        if getattr(fit, "is_empty", True): continue
        fx0, fy0, fx1, fy1 = fit.bounds
        for _ in range(480):
            trials += 1
            if trials > MAX_TRIALS or area_circ >= target_area or len(placed_xy) >= max_particles: break
            x = rng.uniform(fx0, fx1); y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x,y)) or not ok(x,y,r): continue
            idx = len(placed_xy)
            placed_xy.append((x,y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx,gy), []).append(idx)
            area_circ += math.pi*r*r
        if trials > MAX_TRIALS or area_circ >= target_area or len(placed_xy) >= max_particles: break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi2D   = area_circ/area_dom if area_dom>0 else 0.0
    return centers, radii, float(phi2D)

# ------------------------------- Raster & voids -------------------------------
@st.cache_data(show_spinner=False)
def raster_particle_mask(centers_hash: int, centers: np.ndarray, radii: np.ndarray, fov: float, px: int=900):
    if centers.size == 0 or radii.size == 0:
        return np.zeros((px, px), dtype=bool)
    y, x = np.mgrid[0:px, 0:px]
    sx = float(fov) / px
    x_phys = x * sx
    y_phys = (px - y) * sx
    mask = np.zeros((px, px), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        dist2 = (x_phys - cx)**2 + (y_phys - cy)**2
        mask |= (dist2 <= r**2)
    return mask

@st.cache_data(show_spinner=False)
def generate_void_mask(pore_hash: int, pore_mask: np.ndarray, saturation: float, seed: int):
    rng = np.random.default_rng(seed)
    pore = int(pore_mask.sum())
    if pore <= 0: return np.zeros_like(pore_mask, bool)
    target = int(round((1.0 - float(saturation)) * pore))
    if target <= 0: return np.zeros_like(pore_mask, bool)
    if HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18*noise
        flat  = field[pore_mask]
        kth   = np.partition(flat, len(flat)-target)[len(flat)-target]
        vm = np.zeros_like(pore_mask, bool)
        vm[pore_mask] = field[pore_mask] >= kth
        vm = ndi.binary_opening(vm, iterations=1)
        vm = ndi.binary_closing(vm, iterations=1)
        return vm
    # dotted fallback
    h,w = pore_mask.shape
    vm = np.zeros_like(pore_mask, bool); area=0; tries=0
    while area < target and tries < 120000:
        tries += 1
        r = int(np.clip(rng.normal(3.0,1.2),1.0,6.0))
        x = rng.integers(r, w-r); y = rng.integers(r, h-r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            vm[add] = True
            area = int(vm.sum())
    return vm

# ------------------------------- Drawing helpers ------------------------------
def draw_scale_bar(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm
    x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad
    y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.0, color=BORDER)
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=9, color=BORDER)

def render_particles_only(ax, centers, radii, fov):
    ax.add_patch(Rectangle((0,0), fov, fov, facecolor="white", edgecolor=BORDER, linewidth=1.2))
    for (x,y), r in zip(centers, radii):
        ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
    ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov)
    ax.set_xticks([]); ax.set_yticks([])
    draw_scale_bar(ax, fov)

def render_with_binder(ax, centers, radii, pore_mask, saturation, binder_hex, fov, seed_offset):
    pore_hash = hash((pore_mask.shape, saturation, seed_offset))
    vmask = generate_void_mask(pore_hash, pore_mask, float(saturation), int(seed_offset))
    ax.add_patch(Rectangle((0,0), fov, fov, facecolor=binder_hex, edgecolor=BORDER, linewidth=1.2))
    ys, xs = np.where(vmask)
    if len(xs):
        xm = xs * (fov/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (fov/vmask.shape[0])
        ax.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidth=0)
    for (x,y), r in zip(centers, radii):
        ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
    ax.set_aspect('equal','box'); ax.set_xlim(0,fov); ax.set_ylim(0,fov)
    ax.set_xticks([]); ax.set_yticks([])
    draw_scale_bar(ax, fov)

# ============================== ADAPTER RENDER ================================
def _mesh_to_arrays(mesh):
    if mesh is None: return None, None, None
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mhash = hash((verts.tobytes(), faces.tobytes()))
    return verts, faces, mhash

def render(material: str, d50_um: float, layer_um: float):
    """One-call UI used by streamlit_app.py"""
    st.subheader("Digital Twin (Beta) — STL slice + qualitative packing")

    missing = check_dependencies()
    if "trimesh" in missing or "shapely" in missing:
        st.error("Digital Twin requires: " + ", ".join(missing))
        st.info("Add to requirements.txt (SciPy optional for faster voids).")
        return

    # Pull Top-5 from main app
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run the Recommend tab first to produce Top-5 recipes.")
        return
    top5 = top5.reset_index(drop=True)

    # Controls
    L, R = st.columns([1.2, 1])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        pack_full = st.checkbox("Pack full slice (auto FOV)", value=True)
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 6.0, 1.5, 0.05, disabled=pack_full)
        phi_TPD = st.slider("Target φ (theoretical packing density)", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 200, 3500, 1600, 50)
        fast = st.toggle("Fast mode", value=True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    # Mesh
    mesh=None
    if use_cube:
        try: mesh = get_cube_mesh()
        except Exception as e: st.error(f"Cube error: {e}"); return
    elif stl_file is not None:
        try: mesh = load_mesh_from_bytes(stl_file.read())
        except Exception as e: st.error(f"STL error: {e}"); return

    # Recipe & PSD
    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    n_psd = 7000 if fast else 10000
    diam_um = sample_psd_um(n_psd, d50_r, None, None, seed=9991)
    diam_units = diam_um * um2unit

    # Layer choice
    if mesh is not None:
        v, f, mhash = _mesh_to_arrays(mesh)
        minz, maxz = float(np.min(v[:,2])), float(np.max(v[:,2]))
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness
    else:
        v, f, mhash = None, None, None
        n_layers, layer_idx, z = 1, 1, 0.0

    # Slice → FOV
    polys_local_wkb = []; render_fov = float(fov_mm)
    if mesh is not None:
        polys_z_wkb = slice_mesh_at_z(mhash, v, f, float(z))
        if not polys_z_wkb:
            st.warning("No section at this Z. Try another layer."); return

        if pack_full:
            polys = [shp_wkb.loads(p) for p in polys_z_wkb]
            dom = unary_union(polys)
            xmin, ymin, xmax, ymax = dom.bounds
            fov = max(xmax-xmin, ymax-ymin)
            win = box(xmin, ymin, xmin+fov, ymin+fov)
            clip = dom.intersection(win)
            polys_local_wkb = [clip.wkb] if clip.area>0 else []
            render_fov = fov
        else:
            cropped_wkb, origin = crop_to_fov(tuple(polys_z_wkb), float(fov_mm))
            polys_local_wkb = to_local_coords(tuple(cropped_wkb), origin)
            render_fov = float(fov_mm)
    else:
        if HAVE_SHAPELY:
            polys_local_wkb = [box(0,0,1.8,1.8).wkb]; render_fov = 1.8
        else:
            polys_local_wkb = []; render_fov = float(fov_mm)

    # Pack
    centers, radii, phi2D = pack_particles_no_cache(
        polys_local_wkb, diam_units, float(phi2D_target),
        max_particles=int(cap), layer_idx=int(layer_idx)
    )

    # Raster → pore mask
    if centers.size and radii.size:
        centers_hash = hash((centers.tobytes(), radii.tobytes(), int(render_fov)))
        solid = raster_particle_mask(centers_hash, centers, radii, float(render_fov), px=900)
        pore_mask = ~solid
    else:
        pore_mask = np.zeros((900,900), dtype=bool)

    # Render single trial
    rec_sat = float(rec.get("saturation_pct", 85.0))
    rec_binder = str(rec.get("binder_type", "water_based"))
    hexc = binder_color(rec_binder)

    cA,cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.1,5.1), dpi=185)
        render_particles_only(axA, centers, radii, float(render_fov))
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with cB:
        figB, axB = plt.subplots(figsize=(5.1,5.1), dpi=185)
        seed_off = 123 + int(layer_idx) + int(rec_sat)
        render_with_binder(axB, centers, radii, pore_mask, rec_sat/100.0, hexc, float(render_fov), seed_off)
        axB.set_title(f"{rec_binder} · Sat {int(rec_sat)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · "
               f"φ₂D(achieved)≈{min(float(phi2D),1.0):.2f} · Porosity₂D≈{max(0.0,1.0-float(phi2D)):.2f}")

    # Optional comparison
    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", rec_sat))
            bndr = str(row.get("binder_type", rec_binder))
            hexc2 = binder_color(bndr)
            figC, axC = plt.subplots(figsize=(4.9,4.9), dpi=185)
            seed_off2 = 987 + int(layer_idx) + int(sat)
            render_with_binder(axC, centers, radii, pore_mask, sat/100.0, hexc2, float(render_fov), seed_off2)
            axC.set_title(f'{row["id"]}: {bndr} · Sat {int(sat)}%', fontsize=9)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
