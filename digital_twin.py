# -*- coding: utf-8 -*-
# digital_twin.py — clean module, no top-level Streamlit calls.
from __future__ import annotations
import io, math
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# Optional/geometry deps
try:
    import trimesh
    HAVE_TRIMESH = True
except Exception:
    HAVE_TRIMESH = False

try:
    from shapely.geometry import Polygon, MultiPolygon, Point, box
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

# -------------------- colors --------------------
BINDER = {"water":"#F2D06F","solvent":"#F2B233","other":"#F4B942"}
PARTICLE="#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"

def _binder_hex(name:str)->str:
    k=(name or "").lower()
    if "water" in k: return BINDER["water"]
    if "solvent" in k: return BINDER["solvent"]
    return BINDER["other"]

# -------------------- PSD -----------------------
def sample_psd_um(n:int, d50_um:float, seed:int=1234, sigma:float=0.28)->np.ndarray:
    """Lognormal PSD with visible variance; clipped to [0.3, 3.0]×D50."""
    rng = np.random.default_rng(seed)
    mu = np.log(max(1e-6, float(d50_um)))
    d = np.exp(rng.normal(mu, float(sigma), size=int(n)))
    d = np.clip(d, 0.30*d50_um, 3.0*d50_um)
    return d

# -------------------- mesh I/O -------------------
def _load_mesh_from_bytes(file_bytes: bytes):
    if not HAVE_TRIMESH:
        raise ImportError("trimesh is required for STL loading")
    m = trimesh.load(io.BytesIO(file_bytes), file_type="stl", force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump(concatenate=True)
    return m

def _cube_mm():
    if not HAVE_TRIMESH:
        raise ImportError("trimesh is required for cube generation")
    return trimesh.creation.box(extents=(10.0,10.0,10.0))

# -------------------- slicing --------------------
def _slice_polys(mesh, z)->List[Polygon]:
    """Return valid 2D polygons at plane z (robust to tiny z offsets)."""
    if not (HAVE_TRIMESH and HAVE_SHAPELY): return []
    sec = mesh.section(plane_origin=(0,0,float(z)), plane_normal=(0,0,1))
    if sec is None:
        # gentle nudge if slicing falls on a vertex
        for dz in (1e-4, -1e-4, 1e-3, -1e-3):
            sec = mesh.section(plane_origin=(0,0,float(z)+dz), plane_normal=(0,0,1))
            if sec is not None:
                break
    if sec is None:
        return []
    planar,_ = sec.to_planar()
    polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
    return [p.buffer(0) for p in polys if p.is_valid and p.area>1e-8]

def _square_fit_over(dom)->Polygon:
    """Square that covers the part's XY span (used when 'pack full slice' is on)."""
    xmin,ymin,xmax,ymax = dom.bounds
    L = max(xmax-xmin, ymax-ymin)
    return box(xmin, ymin, xmin+L, ymin+L)

def _crop_center(dom, fov_mm:float)->Tuple[List[Polygon], Tuple[float,float]]:
    """Centered square crop with side=fov_mm."""
    cx,cy = dom.centroid.x, dom.centroid.y
    half = float(fov_mm)/2.0
    win = box(cx-half, cy-half, cx+half, cy+half)
    res = dom.intersection(win)
    if getattr(res,"is_empty",True): return [], (cx-half, cy-half)
    if isinstance(res, Polygon): return [res], (cx-half, cy-half)
    return [g for g in res.geoms if isinstance(g, Polygon)], (cx-half, cy-half)

def _to_local(polys:List[Polygon], origin_xy:Tuple[float,float])->List[Polygon]:
    if not polys: return []
    ox,oy = origin_xy
    out=[]
    for p in polys:
        x,y = p.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-ox, np.array(y)-oy]))
    return out

# -------------------- outlines -------------------
def _exterior_rings(geom:Polygon|MultiPolygon):
    """Yield numpy (x,y) ring coordinates for Polygon or MultiPolygon boundaries."""
    if isinstance(geom, Polygon):
        yield np.array(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            if not g.is_empty:
                yield np.array(g.exterior.coords)

# -------------------- packing --------------------
def _pack(polys:List[Polygon], diam_units:np.ndarray, phi2D_target:float,
          max_particles:int, max_trials:int, seed:int):
    if not (HAVE_SHAPELY and polys): 
        return np.empty((0,2)), np.empty((0,)), 0.0

    dom_all = unary_union(polys)
    if getattr(dom_all,"is_empty",True):
        return np.empty((0,2)), np.empty((0,)), 0.0

    area_dom = dom_all.area
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy:List[Tuple[float,float]] = []
    placed_r :List[float] = []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)

    minx, miny, maxx, maxy = dom_all.bounds
    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/450.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def ok(x,y,r):
        gx,gy = int(x//cell), int(y//cell)
        for ix in range(gx-1,gx+2):
            for iy in range(gy-1,gy+2):
                for j in grid.get((ix,iy),[]):
                    dx=x-placed_xy[j][0]; dy=y-placed_xy[j][1]
                    if dx*dx+dy*dy < (r+placed_r[j])**2: return False
        return True

    trials=0
    for d in diam:
        r=d/2.0
        fit = dom_all.buffer(-r)
        if getattr(fit,"is_empty",True): continue
        fx0,fy0,fx1,fy1 = fit.bounds
        for _ in range(440):   # tight inner loop for speed
            trials += 1
            if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles:
                break
            x=rng.uniform(fx0,fx1); y=rng.uniform(fy0,fy1)
            if not fit.contains(Point(x,y)) or not ok(x,y,r): 
                continue
            idx=len(placed_xy)
            placed_xy.append((x,y)); placed_r.append(r)
            gx,gy=int(x//cell),int(y//cell)
            grid.setdefault((gx,gy),[]).append(idx)
            area_circ += math.pi*r*r
        if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi2D = area_circ/area_dom if area_dom>0 else 0.0
    return centers, radii, float(phi2D), dom_all

# -------------------- raster/voids -------------
def _bitmap_mask(centers, radii, fov, px=900):
    img=Image.new("L",(px,px),color=0); drw=ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def _voids_from_sat(pore_mask, sat, seed):
    pore=int(pore_mask.sum())
    if pore<=0: return np.zeros_like(pore_mask,bool)
    target=int(round((1.0-float(sat))*pore))
    if target<=0: return np.zeros_like(pore_mask,bool)
    if HAVE_SCIPY:
        dist=ndi.distance_transform_edt(pore_mask)
        rng=np.random.default_rng(seed)
        noise=ndi.gaussian_filter(rng.standard_normal(pore_mask.shape),sigma=2.0)
        field=dist+0.18*noise
        flat=field[pore_mask]
        kth=np.partition(flat,len(flat)-target)[len(flat)-target]
        vm=np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm=ndi.binary_opening(vm,iterations=1); vm=ndi.binary_closing(vm,iterations=1)
        return vm
    # fallback speckles
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    rng=np.random.default_rng(seed)
    while area<target and tries<90000:
        tries+=1; r=int(np.clip(rng.normal(3.0,1.2),1.0,6.0))
        x=rng.integers(r,w-r); y=rng.integers(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]; disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def _scale_bar(ax, fov_mm, length_um=500):
    L=length_um/1000.0
    if L>=fov_mm: return
    pad=0.06*fov_mm; x0=fov_mm-pad-L; x1=fov_mm-pad; y=pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.0, color=BORDER)
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm", ha="center", va="bottom", fontsize=9, color=BORDER)

# -------------------- public entry ------------
def render(st, *, material:str, d50_um:float, layer_um:float):
    """Embed inside an existing Streamlit tab. Pass the imported `st` explicitly."""
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Install 'trimesh' and 'shapely' to use the Digital Twin tab.")
        return

    st.subheader("Digital Twin — STL slice + qualitative packing")

    # Pull recipes from main app (created earlier)
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5,"empty",True):
        st.info("Run the Predict tab first to generate Top-5 recipes.")
        return
    top5 = top5.reset_index(drop=True)

    L,R = st.columns([1.2,1])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        pack_fit = st.selectbox("Packing window", ["Full part (auto)", "Centered FOV (manual)"], index=0)
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 12.0, 2.0, 0.05, disabled=(pack_fit!="Centered FOV (manual)"))
        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 200, 5000, 2200, 50)
        show_outline = st.toggle("Overlay part outline", True)
        fast = st.toggle("Fast mode", True)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    # mesh
    mesh=None
    if use_cube:
        mesh=_cube_mm()
    elif stl_file is not None:
        mesh=_load_mesh_from_bytes(stl_file.read())

    # trial params
    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    diam_um = sample_psd_um(10000 if not fast else 7000, d50_r, seed=9991, sigma=0.30)
    diam_units = diam_um * um2unit

    # layer selection
    if mesh is not None:
        minz,maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz-minz)/max(thickness,1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(3,n_layers))
        z = minz + (layer_idx-0.5)*thickness
    else:
        st.info("No STL — using a square microstructure window.")
        layer_idx, z = 1, 0.0

    # optional 3D preview
    if mesh is not None and show_mesh:
        import plotly.graph_objects as go
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # slice polygons
    if mesh is not None and HAVE_SHAPELY:
        polys_world = _slice_polys(mesh, z)
        if polys_world:
            dom = unary_union(polys_world)
            if pack_fit=="Full part (auto)":
                # square covering full part, then intersect with part to stay inside
                win = _square_fit_over(dom)
                clipped = dom.intersection(win)
                polys_local = _to_local([clipped], (win.bounds[0], win.bounds[1]))
                render_fov = win.bounds[2]-win.bounds[0]
                origin = (win.bounds[0], win.bounds[1])
            else:
                cropped, origin = _crop_center(dom, fov_mm)
                polys_local = _to_local(cropped, origin)
                render_fov = float(fov_mm)
        else:
            dom=None
            half = 1.8/2.0
            polys_local=[box(0,0,2*half,2*half)]
            render_fov=2*half; origin=(0.0,0.0)
    else:
        dom=None
        half = 1.8/2.0
        polys_local=[box(0,0,2*half,2*half)]
        render_fov=2*half; origin=(0.0,0.0)

    # pack
    centers, radii, phi2D, dom_all = _pack(
        polys_local, diam_units, phi2D_target,
        max_particles=int(cap),
        max_trials=200_000 if fast else 480_000,
        seed=20_000 + int(layer_idx if mesh is not None else 0)
    )

    # panels
    def _panel_particles(ax):
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor="white", edgecolor=BORDER, linewidth=1.2))
        for (x,y),r in zip(centers,radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.18, alpha=0.95))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov); ax.set_xticks([]); ax.set_yticks([])

    def _panel_binder(ax, sat_pct, binder_key):
        px=900; pores=~_bitmap_mask(centers, radii, render_fov, px)
        vm=_voids_from_sat(pores, sat=float(sat_pct)/100.0, seed=1234)
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor=_binder_hex(binder_key), edgecolor=BORDER, linewidth=1.2))
        ys,xs=np.where(vm)
        if len(xs):
            xm=xs*(render_fov/vm.shape[1]); ym=(vm.shape[0]-ys)*(render_fov/vm.shape[0])
            ax.scatter(xm, ym, s=0.28, c=VOID, alpha=0.96, linewidth=0)
        for (x,y),r in zip(centers,radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.18, alpha=0.95))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov); ax.set_xticks([]); ax.set_yticks([])

    # single trial
    trial = top5[top5["id"]==rec_id].iloc[0]
    sat_pct = float(trial.get("saturation_pct", 80.0))
    bname   = str(trial.get("binder_type","water_based"))

    cA,cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.2,5.2), dpi=180)
        _panel_particles(axA)
        if show_outline and dom_all is not None:
            # draw outline(s) in local coords
            ox0, oy0 = (origin[0], origin[1])
            for ring in _exterior_rings(dom_all.buffer(0)):
                xg = ring[:,0] - ox0; yg = ring[:,1] - oy0
                axA.plot(xg, yg, color="#111", linewidth=0.8, alpha=0.6)
        _scale_bar(axA, render_fov)
        axA.set_title("Particles only", fontsize=11)
        st.pyplot(figA, use_container_width=True)

    with cB:
        figB, axB = plt.subplots(figsize=(5.2,5.2), dpi=180)
        _panel_binder(axB, sat_pct, bname)
        if show_outline and dom_all is not None:
            ox0, oy0 = (origin[0], origin[1])
            for ring in _exterior_rings(dom_all.buffer(0)):
                xg = ring[:,0] - ox0; yg = ring[:,1] - oy0
                axB.plot(xg, yg, color="#111", linewidth=0.8, alpha=0.6)
        _scale_bar(axB, render_fov)
        axB.set_title(f"{bname} · Sat {int(sat_pct)}%", fontsize=11)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

    # compare
    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0)); bndr=str(row.get("binder_type","water_based"))
            figC, axC = plt.subplots(figsize=(4.9,4.9), dpi=175)
            _panel_binder(axC, sat, bndr)
            if show_outline and dom_all is not None:
                ox0, oy0 = (origin[0], origin[1])
                for ring in _exterior_rings(dom_all.buffer(0)):
                    xg = ring[:,0] - ox0; yg = ring[:,1] - oy0
                    axC.plot(xg, yg, color="#111", linewidth=0.8, alpha=0.6)
            _scale_bar(axC, render_fov)
            axC.set_title(f'{row["id"]}: {bndr} · Sat {int(sat)}%', fontsize=10)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
