# -*- coding: utf-8 -*-
# BJAM — Binder-Jet AM Parameter Recommender + Digital Twin (Beta)
# Relies on your existing shared.py for data/model logic.

from __future__ import annotations
import io, os, math, importlib.util
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# ------------------------------- Page config & theme --------------------------
st.set_page_config(page_title="BJAM Predictions", page_icon="🟨", layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#FFFDF8 0%,#FFF6E9 50%,#FFF1DD 100%)}
[data-testid="stSidebar"]{background:#fffdfa;border-right:1px solid #f3e8d9}
.stTabs [data-baseweb="tab-list"]{gap:12px}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #f3e8d9;border-radius:10px;padding:6px 10px}
.block-container{padding-top:1.1rem;padding-bottom:1.1rem}
</style>
""", unsafe_allow_html=True)

# ------------------------------- Optional geometry deps -----------------------
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
HAVE_SCIPY   = importlib.util.find_spec("scipy")   is not None
if HAVE_TRIMESH: import trimesh  # type: ignore
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
if HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore

# ------------------------------- Colors --------------------------------------
BINDER_COLORS = {"PVOH":"#F2B233","PEG":"#F2D06F","Furan":"#F5C07A","Acrylic":"#FFD166","Other":"#F4B942"}
PARTICLE="#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"
def binder_color(name:str)->str:
    key=(name or "").lower()
    for k,v in BINDER_COLORS.items():
        if k.lower() in key: return v
    return BINDER_COLORS["Other"]

# ------------------------------- Packing helpers (defined FIRST) --------------
def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
    """Log-normal PSD with gentle tail clipping (0.3×–3× D50)."""
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

def load_mesh(fileobj):
    try:
        mesh = trimesh.load(io.BytesIO(fileobj.read()), file_type="stl", force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh): mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}"); return None

def slice_polys(mesh, z)->List["Polygon"]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        out = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in out if p.is_valid and p.area>1e-8]
    except Exception:
        return []

def crop_fov(polys, fov):
    if not polys: return []
    dom = unary_union(polys); cx,cy = dom.centroid.x, dom.centroid.y
    half=fov/2.0; win = box(cx-half, cy-half, cx+half, cy+half)
    res = dom.intersection(win)
    if getattr(res, "is_empty", True): return []
    if isinstance(res, Polygon): return [res]
    return [g for g in res.geoms if isinstance(g, Polygon) and g.is_valid and g.area>1e-8]

def pack_in_domain(polys, diam_units, phi2D_target, max_particles, max_trials, seed):
    """Greedy circle packing; each circle must fit entirely inside domain (erode by r)."""
    if not HAVE_SHAPELY:
        raise RuntimeError("This view requires 'shapely'. Please add it to requirements.txt.")
    if not polys: return np.empty((0,2)), np.empty((0,)), 0.0

    dom_all = unary_union(polys)
    minx, miny, maxx, maxy = dom_all.bounds
    area_dom = dom_all.area

    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circ = 0.0
    target_area = float(np.clip(phi2D_target, 0.05, 0.90)) * area_dom
    rng = np.random.default_rng(seed)

    cell = max(diam.max()/2.0, (maxx-minx+maxy-miny)/400.0)
    grid: Dict[Tuple[int,int], List[int]] = {}

    def no_overlap(x, y, r):
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
        fit_dom = dom_all.buffer(-r)  # key: full circle fit
        if getattr(fit_dom, "is_empty", True):
            continue
        fminx, fminy, fmaxx, fmaxy = fit_dom.bounds

        for _ in range(180):
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
            placed_xy.append((x, y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx, gy), []).append(idx)
            area_circ += math.pi * r * r

        if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
            break

    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii   = np.array(placed_r)  if placed_r  else np.empty((0,))
    phi2D   = area_circ / area_dom if area_dom > 0 else 0.0
    return centers, radii, float(phi2D)

def raster_mask(centers, radii, fov, px=900):
    img=Image.new("L",(px,px),color=0); drw=ImageDraw.Draw(img)
    sx=px/fov; sy=px/fov
    for (x,y),r in zip(centers,radii):
        x0=int((x-r)*sx); y0=int((fov-(y+r))*sy); x1=int((x+r)*sx); y1=int((fov-(y-r))*sy)
        drw.ellipse([x0,y0,x1,y1], fill=255)
    return (np.array(img)>0)

def voids_from_saturation(pore_mask, saturation, rng=None):
    if rng is None: rng=np.random.default_rng(0)
    pore=int(pore_mask.sum())
    if pore<=0: return np.zeros_like(pore_mask,bool)
    target=int(round((1.0 - saturation)*pore))
    if target<=0: return np.zeros_like(pore_mask,bool)
    if HAVE_SCIPY:
        dist=ndi.distance_transform_edt(pore_mask)
        noise=ndi.gaussian_filter(rng.standard_normal(pore_mask.shape),sigma=2.2)
        field=dist+0.18*noise
        flat=field[pore_mask]
        kth=np.partition(flat,len(flat)-target)[len(flat)-target]
        vm=np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm=ndi.binary_opening(vm,iterations=1); vm=ndi.binary_closing(vm,iterations=1)
        return vm
    # fallback dotted
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    while area<target and tries<120000:
        tries+=1; r=int(np.clip(np.random.normal(3.0,1.2),1.0,6.0))
        x=np.random.randint(r,w-r); y=np.random.randint(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]
            disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def draw_scale_bar(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm
    x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad
    y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.5, color="#111111")
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm", ha="center", va="bottom", fontsize=9, color="#111111")

# ------------------------------- Import your model utils ----------------------
if importlib.util.find_spec("shared") is None:
    st.error("shared.py not found. Place shared.py next to this file.")
    st.stop()

from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    guardrail_ranges,
    copilot,
)

# ------------------------------- Data & models --------------------------------
df_base, src_path = load_dataset(".")
models, meta = train_green_density_models(df_base)

# ------------------------------- Header ---------------------------------------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided few-shot models from your dataset (shared.py). Digital Twin added. Generated with help of ChatGPT.")

# ------------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base.columns else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", "Silicon Carbide (SiC)")
    d50_um  = st.number_input("D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_green = st.slider("Target green density (% of theoretical)", 80, 98, 90, 1)
    guardrails_on = st.toggle("Guardrails", value=True)

    st.divider()
    st.markdown("Data / Models")
    st.write("Source:", os.path.basename(src_path) if src_path else "—")
    st.write("Rows:", f"{len(df_base):,}")
    st.write("Models:", "trained" if models else "—")

# ------------------------------- Tabs -----------------------------------------
tab_pred, tab_heat, tab_sens, tab_pack, tab_form, tab_twin = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)"]
)

# ==============================================================================
# Predict (Top-5)
# ==============================================================================
with tab_pred:
    st.subheader("Top-5 parameter sets (your quantile models)")
    top_k = st.selectbox("How many?", [3, 5, 7, 10], index=1)

    recs = copilot(
        material=material,
        d50_um=float(d50_um),
        df_source=df_base,
        models=models,
        guardrails_on=guardrails_on,
        target_green=float(target_green),
        top_k=int(top_k),
    )

    if recs is None or len(recs) == 0:
        st.warning("No recommendations were returned. Try widening guardrails or lowering the target.")
        st.session_state.pop("top5_recipes_df", None)
    else:
        recs = recs.reset_index(drop=True).copy()
        if "id" not in recs.columns:
            recs["id"] = [f"Opt-{i+1}" for i in range(len(recs))]
        else:
            recs["id"] = recs["id"].astype(str)
        st.session_state["top5_recipes_df"] = recs
        st.dataframe(recs, use_container_width=True, hide_index=True)

# ==============================================================================
# Heatmap (journal-style)
# ==============================================================================
with tab_heat:
    st.subheader("Predicted green %TD (q50) — speed × saturation")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = gr["binder_saturation_pct"]; spd_lo, spd_hi = gr["roller_speed_mm_s"]
    Xs = np.linspace(float(sat_lo), float(sat_hi), 65)
    Ys = np.linspace(float(spd_lo), float(spd_hi), 55)

    grid = pd.DataFrame(
        [(b, v, layer_um, d50_um, material) for b in Xs for v in Ys],
        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"],
    )
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    pred = predict_quantiles(models, grid)
    Z = pred.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=Z, x=Xs, y=Ys, zsmooth=False,
        colorscale=[ [0.0,"#2c7fb8"], [0.35,"#7fcdbb"], [0.6,"#c7e9b4"], [0.8,"#fefebd"], [1.0,"#fdae61"] ],
        colorbar=dict(title="%TD (q50)", len=0.82, ticks="outside"),
    ))

    thr = float(target_green)
    fig.add_trace(go.Contour(
        z=Z, x=Xs, y=Ys,
        contours=dict(start=thr, end=thr, size=1, coloring="lines"),
        line=dict(color="#C21807", dash="dash", width=3.0),
        showscale=False, hoverinfo="skip", name="Target",
    ))

    x0, x1 = np.percentile(Xs, [35, 65]); y0, y1 = np.percentile(Ys, [35, 65])
    fig.update_layout(shapes=[dict(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                                   line=dict(color="#2bb7b3", width=3, dash="dash"),
                                   fillcolor="rgba(0,0,0,0)")])
    cx, cy = (x0+x1)/2, (y0+y1)/2
    fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers",
                             marker=dict(symbol="circle-open-dot", size=14, line=dict(width=2, color="#1f2937")),
                             hoverinfo="skip", name=""))
    fig.add_shape(type="line", x0=cx-1.6, y0=cy, x1=cx+1.6, y1=cy, line=dict(color="#1f2937", width=2))
    fig.add_shape(type="line", x0=cx, y0=cy-0.08, x1=cx, y1=cy+0.08, line=dict(color="#1f2937", width=2))

    fig.update_layout(
        title="Predicted Green Density (% Theoretical Density)",
        xaxis_title="Binder Saturation (%)", yaxis_title="Roller Speed (mm/s)",
        margin=dict(l=10, r=20, t=40, b=35), height=470,
        xaxis=dict(ticks="outside", showgrid=False), yaxis=dict(ticks="outside", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Saturation sensitivity
# ==============================================================================
with tab_sens:
    st.subheader("Saturation sensitivity at representative speed")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 50)

    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "layer_thickness_um": np.full_like(sat_axis, layer_um),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material]*len(sat_axis),
        "material_class":"metal",
        "binder_type_rec":"solvent_based",
    })
    pred = predict_quantiles(models, grid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=pred["binder_saturation_pct"], y=pred["td_q90"], name="q90", mode="lines"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                      height=380, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Qualitative packing — poster-style layer (requires shapely)
# ==============================================================================
with tab_pack:
    st.subheader("Qualitative packing (illustrative)")
    if not HAVE_SHAPELY:
        st.error("This view requires 'shapely'. Please add it to requirements.txt.")
    else:
        rng = np.random.default_rng(1234)
        frame_mm = 1.8
        px = 1400

        diam_um = sample_psd_um(6500, d50_um, None, None, seed=42)
        diam_mm = diam_um / 1000.0
        phi2D_target = float(np.clip(0.90 * (target_green/100.0), 0.40, 0.88))
        dom = box(0, 0, frame_mm, frame_mm)

        centers, radii, phi2D = pack_in_domain([dom], diam_mm, phi2D_target,
                                               max_particles=2200, max_trials=300_000, seed=1001)

        def raster_particle_mask_layer():
            img = Image.new("L", (px, px), color=0)
            d = ImageDraw.Draw(img)
            sx = px / frame_mm; sy = px / frame_mm
            for (x,y), r in zip(centers, radii):
                x0 = int((x - r)*sx); y0 = int((frame_mm - (y + r))*sy)
                x1 = int((x + r)*sx); y1 = int((frame_mm - (y - r))*sy)
                d.ellipse([x0,y0,x1,y1], fill=255)
            return (np.array(img) > 0)

        pore_mask = ~raster_particle_mask_layer()
        sat_frac = float(np.clip(target_green/100.0, 0.6, 0.98))
        vmask = voids_from_saturation(pore_mask, saturation=sat_frac, rng=rng)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
        ax.add_patch(Rectangle((0, 0), frame_mm, frame_mm, facecolor=binder_color("PVOH"), edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (frame_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (frame_mm/vmask.shape[0])
            ax.scatter(xm, ym, s=0.25, c=VOID, alpha=0.95, linewidths=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0, frame_mm); ax.set_ylim(0, frame_mm)
        ax.set_xticks([]); ax.set_yticks([])

        draw_scale_bar(ax, frame_mm, length_um=500)
        ax.text(0.02*frame_mm, 0.98*frame_mm, "SiC", color=PARTICLE, fontsize=9, va="top")
        ax.text(0.12*frame_mm, 0.98*frame_mm, "Binder", color="#805a00", fontsize=9, va="top")
        ax.text(0.26*frame_mm, 0.98*frame_mm, "Void", color="#666", fontsize=9, va="top")

        ax.set_title(f"Silicon Carbide (SiC) · D50≈{d50_um:.0f} µm · Binder Sat≈{int(sat_frac*100)}% · Layer≈{layer_um:.0f} µm",
                     fontsize=10, pad=10)
        st.pyplot(fig, clear_figure=True)

# ==============================================================================
# Digital Twin (Beta) — true-scale packing, STL slicing, compare
# ==============================================================================
with tab_twin:
    st.subheader("Digital Twin — recipe-true layer preview & compare")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Digital Twin needs 'trimesh' and 'shapely' (see requirements.txt).")
        st.stop()

    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run the Predict tab first to generate Top-5 recipes.")
        st.stop()
    top5 = top5.reset_index(drop=True).copy()
    if "id" not in top5.columns:
        top5["id"] = [f"Opt-{i+1}" for i in range(len(top5))]
    else:
        top5["id"] = top5["id"].astype(str)

    left,right = st.columns([1.2,1])
    with left:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with right:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        # FOV tied to D50 so the view contains many particles
        min_fov_mm = max(0.20, 0.02 * d50_um)
        default_fov = max(0.80, 0.02 * d50_um)
        fov_mm = st.slider("Field of view (mm)", float(min_fov_mm), 3.0, float(default_fov), 0.05)
        phi_TPD = st.slider("Target φ_TPD", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 100, 2500, 1200, 50)

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    mesh=None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = load_mesh(stl_file)

    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    diam_um = sample_psd_um(7500, d50_r, None, None, seed=9991)
    diam_units = diam_um * um2unit

    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.markdown(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, 1)
        z = minz + (layer_idx - 0.5) * thickness
    else:
        n_layers, layer_idx, z = 1, 1, 0.0
        st.info("No STL — using a centered square FOV for microstructure.")

    if mesh is not None and show_mesh:
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    if mesh is not None and HAVE_SHAPELY:
        polys = crop_fov(slice_polys(mesh, z), fov_mm)
    else:
        polys = []
    if not polys:
        half=fov_mm/2.0
        polys=[box(-half,-half,half,half)]

    centers, radii, phi2D = pack_in_domain(polys, diam_units, phi2D_target,
                                           max_particles=cap, max_trials=240_000, seed=20_000+layer_idx)

    def panel_particles(ax):
        ax.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor="white", edgecolor=BORDER, linewidth=1.2))
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,fov_mm); ax.set_ylim(0,fov_mm); ax.set_xticks([]); ax.set_yticks([])
        draw_scale_bar(ax, fov_mm)

    def panel_binder(ax, sat_pct, binder_hex):
        px = 900
        pores = ~raster_mask(centers, radii, fov_mm, px)
        vmask = voids_from_saturation(pores, saturation=float(sat_pct)/100.0,
                                      rng=np.random.default_rng(123+layer_idx))
        ax.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=binder_hex, edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (fov_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (fov_mm/vmask.shape[0])
            ax.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidths=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,fov_mm); ax.set_ylim(0,fov_mm); ax.set_xticks([]); ax.set_yticks([])
        draw_scale_bar(ax, fov_mm)
        ax.text(0.02*fov_mm, 0.97*fov_mm, "SiC", color=PARTICLE, fontsize=9, va="top")
        ax.text(0.12*fov_mm, 0.97*fov_mm, "Binder", color="#805a00", fontsize=9, va="top")
        ax.text(0.24*fov_mm, 0.97*fov_mm, "Void", color="#666", fontsize=9, va="top")

    st.subheader("Single trial")
    sat_pct = float(rec.get("saturation_pct", 80.0)); binder = str(rec.get("binder_type","PVOH"))
    col1, col2 = st.columns(2)
    with col1:
        figA, axA = plt.subplots(figsize=(5.3,5.3), dpi=188)
        panel_particles(axA)
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with col2:
        figB, axB = plt.subplots(figsize=(5.3,5.3), dpi=188)
        panel_binder(axB, sat_pct, binder_color(binder))
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(
        f"FOV={fov_mm:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}"
    )

    st.subheader("Compare trials")
    if not picks:
        st.info("Pick one or more trials above to compare.")
    else:
        cols = st.columns(min(3, len(picks))) if len(picks)<=3 else None
        tabs = None if cols else st.tabs(picks)
        for i, rid in enumerate(picks):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 80.0)); hexc=binder_color(str(row.get("binder_type","PVOH")))
            px=780; pores=~raster_mask(centers, radii, fov_mm, px)
            vm = voids_from_saturation(pores, saturation=sat/100.0,
                                       rng=np.random.default_rng(987+int(sat)+layer_idx))
            figC, axC = plt.subplots(figsize=(5.1,5.1), dpi=185)
            axC.add_patch(Rectangle((0,0), fov_mm, fov_mm, facecolor=hexc, edgecolor=BORDER, linewidth=1.2))
            ys,xs=np.where(vm)
            if len(xs):
                xm=xs*(fov_mm/vm.shape[1]); ym=(vm.shape[0]-ys)*(fov_mm/vm.shape[0])
                axC.scatter(xm, ym, s=0.32, c=VOID, alpha=0.96, linewidth=0)
            for (x,y), r in zip(centers, radii):
                axC.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
            axC.set_aspect('equal','box'); axC.set_xlim(0,fov_mm); axC.set_ylim(0,fov_mm); axC.set_xticks([]); axC.set_yticks([])
            axC.set_title(f'{row["id"]}: {row["binder_type"]} · Sat {int(sat)}% · Layer {int(row.get("layer_um", layer_r))} µm', fontsize=10)
            if cols: 
                with cols[i]: st.pyplot(figC, use_container_width=True)
            else:
                with tabs[i]: st.pyplot(figC, use_container_width=True)
