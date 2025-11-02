# -*- coding: utf-8 -*-
# Digital Twin module for BJAM — STL slicing -> 2D packing -> visuals

from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# Optional deps (loaded lazily so main app can run without STL)
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

# -------------------- Colors --------------------
BINDER_COLORS = {
    "water_based": "#F2D06F",
    "solvent_based": "#F2B233",
    "other": "#F4B942",
}
PARTICLE = "#2F6CF6"; EDGE = "#1f2937"; BORDER = "#111111"; VOID = "#FFFFFF"

def _binder_hex(name: str) -> str:
    k = (name or "").lower()
    if "water" in k: return BINDER_COLORS["water_based"]
    if "solvent" in k: return BINDER_COLORS["solvent_based"]
    return BINDER_COLORS["other"]

# -------------------- PSD ----------------------
def _psd_um(n: int, d50_um: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, sigma = np.log(max(d50_um, 1e-6)), 0.25
    d = np.exp(rng.normal(mu, sigma, size=n))
    return np.clip(d, 0.3*d50_um, 3.0*d50_um)

# -------------------- Mesh I/O ------------------
def _load_mesh_from_bytes(st, file_bytes: bytes):
    if not HAVE_TRIMESH:
        st.error("trimesh not installed (add to requirements.txt).")
        return None
    try:
        m = trimesh.load(io.BytesIO(file_bytes), file_type="stl",
                         force="mesh", process=False)
        if not isinstance(m, trimesh.Trimesh):
            m = m.dump(concatenate=True)
        return m
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

def _cube_mesh():
    if not HAVE_TRIMESH:
        return None
    return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

# -------------------- Slice -> polygons --------
def _slice_polys(mesh, z) -> List["Polygon"]:
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        return []
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None:  # try tiny z offsets to catch plane vertices
            for dz in (1e-3, -1e-3, 1e-4, -1e-4):
                sec = mesh.section(plane_origin=(0,0,z+dz), plane_normal=(0,0,1))
                if sec is not None:
                    break
        if sec is None:
            return []
        planar,_ = sec.to_planar()
        polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]
    except Exception:
        return []

# -------------------- Crop / local coords ------
def _crop_to_square(polys: List["Polygon"]) -> Tuple[List["Polygon"], float, Tuple[float,float]]:
    """
    Make a square FOV that fully contains the unioned slice; return
    localised polygons, render_fov, and origin (xmin,ymin).
    """
    if not polys:
        return [], 0.0, (0.0, 0.0)
    dom = unary_union(polys)
    xmin, ymin, xmax, ymax = dom.bounds
    w, h = xmax - xmin, ymax - ymin
    fov = max(w, h)
    # Square window anchored at (xmin,ymin) so we can localize by subtracting
    win = box(xmin, ymin, xmin + fov, ymin + fov)
    clip = dom.intersection(win)
    geoms = [clip] if isinstance(clip, Polygon) else [g for g in getattr(clip, "geoms", []) if isinstance(g, Polygon)]
    # local coords
    out = []
    for g in geoms:
        x, y = g.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-xmin, np.array(y)-ymin]))
    return out, float(fov), (float(xmin), float(ymin))

# -------------------- Packing ------------------
def _pack(polys, diam_units, phi2D_target, max_particles, max_trials, seed):
    if not (HAVE_SHAPELY and polys):
        return np.empty((0,2)), np.empty((0,)), 0.0
    dom_all = unary_union(polys)
    minx, miny, maxx, maxy = dom_all.bounds
    area_dom = dom_all.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)

    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/450.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def _ok(x, y, r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1, gx+2):
            for iy in range(gy-1, gy+2):
                for j in grid.get((ix, iy), []):
                    dx, dy = x - placed_xy[j][0], y - placed_xy[j][1]
                    if dx*dx + dy*dy < (r + placed_r[j])**2:
                        return False
        return True

    trials = 0
    for d in diam:
        r = d/2.0
        fit = dom_all.buffer(-r)
        if getattr(fit, "is_empty", True):
            continue
        fx0, fy0, fx1, fy1 = fit.bounds
        for _ in range(480):
            trials += 1
            if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
                break
            x = rng.uniform(fx0, fx1); y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x,y)) or not _ok(x,y,r): 
                continue
            idx = len(placed_xy)
            placed_xy.append((x,y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx,gy), []).append(idx)
            area_circ += math.pi*r*r
        if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi2D   = area_circ/area_dom if area_dom>0 else 0.0
    return centers, radii, float(phi2D)

# -------------------- Raster helpers ----------
def _bitmap_mask(centers, radii, fov, px=900):
    img = Image.new("L",(px,px), color=0); drw = ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def _voids_from_sat(pore_mask, sat, seed=0):
    pore = int(pore_mask.sum())
    if pore<=0: return np.zeros_like(pore_mask,bool)
    target = int(round((1.0 - float(sat)) * pore))
    if target<=0: return np.zeros_like(pore_mask,bool)
    if HAVE_SCIPY:
        rng = np.random.default_rng(seed)
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18*noise
        flat = field[pore_mask]
        kth  = np.partition(flat, len(flat)-target)[len(flat)-target]
        vm = np.zeros_like(pore_mask,bool); vm[pore_mask] = field[pore_mask] >= kth
        vm = ndi.binary_opening(vm, iterations=1); vm = ndi.binary_closing(vm, iterations=1)
        return vm
    # dotted fallback
    h,w = pore_mask.shape; vm=np.zeros_like(pore_mask,bool)
    area=0; tries=0; rng=np.random.default_rng(seed)
    while area<target and tries<90000:
        tries+=1; r=int(np.clip(rng.normal(3.0,1.2),1.0,6.0))
        x=rng.integers(r,w-r); y=rng.integers(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]; disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def _scale_bar(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm; x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad; y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.0, color=BORDER)
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=9, color=BORDER)

# ---- outline for Polygon or MultiPolygon
def _iter_outline_xy(geom):
    g = geom.buffer(0)  # clean
    b = g.boundary
    if b.geom_type == "LineString":
        yield np.asarray(b.xy[0]), np.asarray(b.xy[1])
    elif b.geom_type == "MultiLineString":
        for ls in b.geoms:
            yield np.asarray(ls.xy[0]), np.asarray(ls.xy[1])

# -------------------- Public entry ----------------
def render(st, *, material: str, d50_um: float, layer_um: float):
    """
    Call as: digital_twin.render(st, material=..., d50_um=..., layer_um=...)
    """
    st.subheader("Digital Twin (Beta) — full-slice packing")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Install 'trimesh' and 'shapely' (see requirements.txt).")
        return

    # Top-5 from main app (used for binder/saturation defaults)
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run the Predict tab first to generate Top-5.")
        return
    top5 = top5.reset_index(drop=True)

    # Controls
    L, R = st.columns([1.2,1])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        pack_cap = st.slider("Max particles (visual cap)", 200, 5000, 2200, 50)
        fast = st.toggle("Fast mode", True, help="Limits attempts for speed")
        show_outline = st.toggle("Show part outline", True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("3D preview", value=False)

    # Mesh
    mesh = None
    if use_cube:
        mesh = _cube_mesh()
    elif stl is not None:
        mesh = _load_mesh_from_bytes(st, stl.read())
    if mesh is None:
        st.info("No mesh provided; using a 1.8 mm square window for a quick demo.")
        polys_local = [box(0,0,1.8,1.8)]
        render_fov = 1.8
        dom = None
    else:
        # Layer index
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = float(layer_um) * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness

        # Optional 3D preview
        if show_mesh:
            import plotly.graph_objects as go
            figm = go.Figure(data=[go.Mesh3d(
                x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                color="lightgray", opacity=0.55, flatshading=True, name="Part"
            )]).update_layout(scene=dict(aspectmode="data"),
                              margin=dict(l=0,r=0,t=0,b=0), height=360)
            st.plotly_chart(figm, use_container_width=True)

        # Slice and build a square FOV that contains the entire section
        polys_world = _slice_polys(mesh, z)
        polys_local, render_fov, origin = _crop_to_square(polys_world)
        dom = unary_union(polys_world) if polys_world else None

    # PSD & packing
    rec = top5[top5["id"]==rec_id].iloc[0]
    sat_pct = float(rec.get("saturation_pct", 80.0))
    binder  = str(rec.get("binder_type",  "water_based"))
    d50_r   = float(rec.get("d50_um", d50_um))

    diam_um = _psd_um(9000 if fast else 12000, d50_r, seed=9991)
    diam_units = diam_um * (1e-3 if stl_units=="mm" else 1e-6)  # µm -> mesh units

    centers, radii, phi2D = _pack(
        polys_local if polys_local else [box(0,0,render_fov,render_fov)],
        diam_units,
        phi2D_target=float(np.clip(0.90*0.90, 0.40, 0.88)),  # ~90% TPD target -> 2D
        max_particles=int(pack_cap),
        max_trials=200_000 if fast else 480_000,
        seed=20_000 + 1
    )

    # Render: particles vs binder+voids
    colA, colB = st.columns(2)

    def _panel_particles(ax):
        ax.add_patch(Rectangle((0,0), render_fov, render_fov,
                               facecolor="white", edgecolor=BORDER, linewidth=1.2))
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        # outline (if available)
        if show_outline and dom is not None:
            for xs, ys in _iter_outline_xy(dom):
                ax.plot(xs - dom.bounds[0], ys - dom.bounds[1], lw=1.0, color="#111111", alpha=0.8)
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov)
        ax.set_xticks([]); ax.set_yticks([])
        _scale_bar(ax, render_fov)

    def _panel_binder(ax, sat_pct, binder_hex):
        pores = ~_bitmap_mask(centers, radii, render_fov, px=900)
        vm = _voids_from_sat(pores, sat=float(sat_pct)/100.0, seed=1234)
        ax.add_patch(Rectangle((0,0), render_fov, render_fov,
                               facecolor=binder_hex, edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vm)
        if len(xs):
            xm = xs*(render_fov/vm.shape[1]); ym = (vm.shape[0]-ys)*(render_fov/vm.shape[0])
            ax.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidth=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        if show_outline and dom is not None:
            for xs, ys in _iter_outline_xy(dom):
                ax.plot(xs - dom.bounds[0], ys - dom.bounds[1], lw=1.0, color="#111111", alpha=0.8)
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov)
        ax.set_xticks([]); ax.set_yticks([])
        _scale_bar(ax, render_fov)

    with colA:
        figA, axA = plt.subplots(figsize=(5.0,5.0), dpi=185)
        _panel_particles(axA)
        axA.set_title("Particles only", fontsize=11)
        st.pyplot(figA, use_container_width=True)

    with colB:
        figB, axB = plt.subplots(figsize=(5.0,5.0), dpi=185)
        _panel_binder(axB, sat_pct, _binder_hex(binder))
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=11)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")
