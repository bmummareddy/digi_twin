# digital_twin.py — STL slice → qualitative packing (fast), designed to be imported by streamlit_app.py
from __future__ import annotations
import io, math, importlib.util
from typing import List, Tuple, Dict
import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

_HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
_HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
_HAVE_SCIPY   = importlib.util.find_spec("scipy")   is not None
if _HAVE_TRIMESH:
    import trimesh  # type: ignore
if _HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
if _HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore

# Colors
_COLOR_PARTICLE = "#2F6CF6"
_COLOR_EDGE     = "#1f2937"
_COLOR_BORDER   = "#111111"
_COLOR_VOID     = "#FFFFFF"
_COLOR_BINDERS  = {"water":"#F2D06F","solvent":"#F2B233","other":"#F4B942"}

def _binder_hex(name:str)->str:
    k=(name or "").lower()
    if "water" in k: return _COLOR_BINDERS["water"]
    if "solvent" in k: return _COLOR_BINDERS["solvent"]
    return _COLOR_BINDERS["other"]

@st.cache_data(show_spinner=False)
def _dt_load_mesh(data: bytes):
    if not _HAVE_TRIMESH:
        raise RuntimeError("trimesh not installed")
    m = trimesh.load(io.BytesIO(data), file_type="stl", force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    return m

def _dt_slice_polys(mesh, z)->List["Polygon"]:
    if not _HAVE_SHAPELY: return []
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        out = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in out if p.is_valid and p.area>1e-8]
    except Exception:
        return []

def _dt_crop_to_fov(polys, fov):
    if not polys: return [], (0.0, 0.0)
    dom = unary_union(polys); cx,cy = dom.centroid.x, dom.centroid.y
    half=fov/2.0; xmin, ymin = cx-half, cy-half
    win = box(xmin, ymin, xmin+fov, ymin+fov)
    res = dom.intersection(win)
    if getattr(res,"is_empty",True): return [], (xmin, ymin)
    geoms = [res] if isinstance(res, Polygon) else [g for g in res.geoms if isinstance(g, Polygon)]
    return geoms, (xmin, ymin)

def _dt_to_local(polys, origin_xy):
    if not polys: return []
    ox, oy = origin_xy
    out=[]
    for p in polys:
        x,y = p.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-ox, np.array(y)-oy]))
    return out

def _dt_psd_um(n:int, d50_um:float, seed:int)->np.ndarray:
    rng = np.random.default_rng(seed)
    mu, sigma = np.log(max(d50_um,1e-6)), 0.25
    d = np.exp(rng.normal(mu, sigma, size=n))
    return np.clip(d, 0.3*d50_um, 3.0*d50_um)

def _dt_pack(polys, diam_units, phi_target, max_particles, max_trials, seed):
    if not _HAVE_SHAPELY or not polys:
        return np.empty((0,2)), np.empty((0,)), 0.0
    dom = unary_union(polys)
    minx, miny, maxx, maxy = dom.bounds
    area_dom = dom.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ, target_area = 0.0, float(np.clip(phi_target,0.05,0.90))*area_dom
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
        fit = dom.buffer(-r)
        if getattr(fit,"is_empty",True): continue
        fx0, fy0, fx1, fy1 = fit.bounds
        for _ in range(480):
            trials += 1
            if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles: break
            x = rng.uniform(fx0, fx1); y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x,y)) or not _ok(x,y,r): continue
            idx = len(placed_xy)
            placed_xy.append((x,y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx,gy), []).append(idx)
            area_circ += math.pi*r*r
        if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles: break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi     = area_circ/area_dom if area_dom>0 else 0.0
    return centers, radii, float(phi)

def _dt_bitmap_mask(centers, radii, fov, px=900):
    img=Image.new("L",(px,px),color=0); drw=ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def _dt_voids(pore_mask, sat, seed=0):
    if pore_mask.sum()==0: return np.zeros_like(pore_mask,bool)
    want = int(round((1.0 - float(sat))*int(pore_mask.sum())))
    if want<=0: return np.zeros_like(pore_mask,bool)
    if _HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(np.random.default_rng(seed).standard_normal(pore_mask.shape), sigma=2.0)
        field = dist + 0.18*noise
        flat  = field[pore_mask]
        kth   = np.partition(flat, len(flat)-want)[len(flat)-want]
        vm    = np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm    = ndi.binary_opening(vm, iterations=1); vm = ndi.binary_closing(vm, iterations=1)
        return vm
    # dotted fallback
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    rng=np.random.default_rng(seed)
    while area<want and tries<90000:
        tries+=1; r=int(np.clip(rng.normal(3.0,1.2),1.0,6.0))
        x=rng.integers(r,w-r); y=rng.integers(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]; disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def _dt_scale(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm
    x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad
    y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.0, color=_COLOR_BORDER)
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=9, color=_COLOR_BORDER)

def render(material:str, d50_um:float, layer_um:float):
    st.subheader("Digital Twin (Beta) — STL slice + qualitative packing")
    if not (_HAVE_TRIMESH and _HAVE_SHAPELY):
        st.error("Install 'trimesh' and 'shapely' (see requirements.txt)."); return

    # Expect Top-k from main app
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Generate recommendations first (click Recommend)."); return
    top5 = top5.reset_index(drop=True)

    L, R = st.columns([1.2, 1])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        pack_full = st.checkbox("Pack full slice (auto FOV)", value=True)
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 6.0, 1.5, 0.05, disabled=pack_full)
        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 200, 3000, 1500, 50)
        fast = st.toggle("Fast mode (coarser packing)", value=True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    mesh=None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = _dt_load_mesh(stl_file.read())

    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    diam_um = _dt_psd_um(7000 if fast else 10000, d50_r, seed=9991)
    diam_units = diam_um * um2unit

    # Pick layer
    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness
    else:
        n_layers, layer_idx, z = 1, 1, 0.0

    # Preview mesh
    if mesh is not None and show_mesh:
        import plotly.graph_objects as go
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # Build slice area
    if mesh is not None and _HAVE_SHAPELY:
        polys_world = _dt_slice_polys(mesh, z)
        if pack_full and polys_world:
            dom = unary_union(polys_world)
            xmin, ymin, xmax, ymax = dom.bounds
            fov = max(xmax-xmin, ymax-ymin)
            win = box(xmin, ymin, xmin+fov, ymin+fov)
            polys_clip = [dom.intersection(win)]
            polys_local = _dt_to_local(polys_clip, (xmin, ymin))
            render_fov = fov
        else:
            polys_clip, origin = _dt_crop_to_fov(polys_world, float(fov_mm))
            polys_local = _dt_to_local(polys_clip, origin)
            render_fov = float(fov_mm)
    else:
        half = (1.8)/2.0
        polys_local=[box(0,0,2*half,2*half)]
        render_fov = 2*half

    # Pack particles
    centers, radii, phi2D = _dt_pack(
        polys_local, diam_units, phi2D_target,
        max_particles=int(cap),
        max_trials=200_000 if fast else 480_000,
        seed=20_000 + layer_idx
    )

    # Render
    def _panel_particles(ax):
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor="white", edgecolor=_COLOR_BORDER, linewidth=1.2))
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=_COLOR_PARTICLE, edgecolor=_COLOR_EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov); ax.set_xticks([]); ax.set_yticks([])

    def _panel_binder(ax, sat_pct, binder):
        px=900; pores=~_dt_bitmap_mask(centers, radii, render_fov, px)
        vm = _dt_voids(pores, sat=float(sat_pct)/100.0, seed=1234)
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor=_binder_hex(binder), edgecolor=_COLOR_BORDER, linewidth=1.2))
        ys,xs=np.where(vm)
        if len(xs):
            xm=xs*(render_fov/vm.shape[1]); ym=(vm.shape[0]-ys)*(render_fov/vm.shape[0])
            ax.scatter(xm, ym, s=0.32, c=_COLOR_VOID, alpha=0.96, linewidth=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=_COLOR_PARTICLE, edgecolor=_COLOR_EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov); ax.set_xticks([]); ax.set_yticks([])

    rec = top5[top5["id"]==rec_id].iloc[0]
    sat_pct = float(rec.get("saturation_pct", 85.0))
    binder  = str(rec.get("binder_type","water_based"))

    cA,cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.1,5.1), dpi=185)
        _panel_particles(axA); _dt_scale(axA, render_fov); axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with cB:
        figB, axB = plt.subplots(figsize=(5.1,5.1), dpi=185)
        _panel_binder(axB, sat_pct, binder); _dt_scale(axB, render_fov); axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 85.0)); bndr=str(row.get("binder_type","water_based"))
            figC, axC = plt.subplots(figsize=(4.9,4.9), dpi=185)
            _panel_binder(axC, sat, bndr); _dt_scale(axC, render_fov)
            axC.set_title(f'{row["id"]}: {bndr} · Sat {int(sat)}%', fontsize=9)
            with cols[i]: st.pyplot(figC, use_container_width=True)
