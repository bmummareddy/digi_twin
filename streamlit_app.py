# -*- coding: utf-8 -*-
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

# Local utils
from shared import load_dataset, train_green_density_models, predict_quantiles, guardrail_ranges

# ------------------------------- Page config & style --------------------------
st.set_page_config(page_title="BJAM Predictions", page_icon="🟨", layout="wide")
BUILD = "2025-11-01 — Original flow + Digital Twin (Beta)"
st.caption(f"Build: {BUILD}")
st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#FFFDF8 0%,#FFF6E9 50%,#FFF1DD 100%)}
[data-testid="stSidebar"]{background:#fffdfa;border-right:1px solid #f3e8d9}
.stTabs [data-baseweb="tab-list"]{gap:12px}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #f3e8d9;border-radius:10px;padding:6px 10px}
.block-container{padding-top:1.0rem;padding-bottom:1.0rem}
</style>
""", unsafe_allow_html=True)

# Quick cache reset button (handy on Streamlit Cloud)
with st.sidebar:
    if st.button("Clear cached data & restart", use_container_width=True):
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        for k in list(st.session_state.keys()): del st.session_state[k]
        st.success("Cache cleared. Reloading…")
        st.experimental_rerun()

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
from scipy.ndimage import gaussian_filter

# ------------------------------- Colors --------------------------------------
BINDER_COLORS = {"water_based":"#F2D06F","solvent_based":"#F2B233","furan":"#F5C07A","acrylic":"#FFD166","other":"#F4B942"}
PARTICLE="#2F6CF6"; EDGE="#1f2937"; BORDER="#111111"; VOID="#FFFFFF"
def binder_color(name:str)->str:
    key=(name or "").lower()
    for k,v in BINDER_COLORS.items():
        if k in key: return v
    return BINDER_COLORS["other"]

# ------------------------------- Packing helpers ------------------------------
def sample_psd_um(n: int, d50_um: float, d10_um: Optional[float], d90_um: Optional[float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    med = max(1e-9, float(d50_um))
    if d10_um and d90_um and d90_um > d10_um > 0:
        m = np.log(med); s = (np.log(d90_um) - np.log(d10_um)) / (2*1.2815515655446004)
        s = float(max(s, 0.05))
    else:
        m, s = np.log(med), 0.25
    d = np.exp(rng.normal(m, s, size=n))
    return np.clip(d, 0.30*med, 3.00*med)

def draw_scale_bar(ax, fov_mm, length_um=500):
    length_mm = length_um/1000.0
    if length_mm >= fov_mm: return
    pad = 0.06*fov_mm
    x0 = fov_mm - pad - length_mm; x1 = fov_mm - pad
    y = pad*0.6
    ax.plot([x0,x1],[y,y], lw=3.2, color="#111111")
    ax.text((x0+x1)/2, y+0.02*fov_mm, f"{int(length_um)} µm", ha="center", va="bottom", fontsize=9, color="#111111")

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
        noise=ndi.gaussian_filter(rng.standard_normal(pore_mask.shape),sigma=2.0)
        field=dist+0.18*noise
        flat=field[pore_mask]
        kth=np.partition(flat,len(flat)-target)[len(flat)-target]
        vm=np.zeros_like(pore_mask,bool); vm[pore_mask]=field[pore_mask]>=kth
        vm=ndi.binary_opening(vm,iterations=1); vm=ndi.binary_closing(vm,iterations=1)
        return vm
    # dotted fallback
    h,w=pore_mask.shape; vm=np.zeros_like(pore_mask,bool); area=0; tries=0
    while area<target and tries<120000:
        tries+=1; r=int(np.clip(np.random.normal(3.0,1.2),1.0,6.0))
        x=np.random.randint(r,w-r); y=np.random.randint(r,h-r)
        if pore_mask[y,x]:
            yy,xx=np.ogrid[-y:h-y,-x:w-x]
            disk=(xx*xx+yy*yy)<=r*r
            add=np.logical_and(disk,pore_mask); vm[add]=True; area=int(vm.sum())
    return vm

def slice_polys(mesh, z)->List["Polygon"]:
    try:
        sec = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sec is None: return []
        planar,_ = sec.to_planar()
        out = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in out if p.is_valid and p.area>1e-8]
    except Exception:
        return []

def crop_window(polys, fov):
    if not polys: return [], (0.0, 0.0)
    dom = unary_union(polys); cx,cy = dom.centroid.x, dom.centroid.y
    half=fov/2.0; xmin, ymin = cx-half, cy-half
    win = box(xmin, ymin, xmin+fov, ymin+fov)
    res = dom.intersection(win)
    if getattr(res, "is_empty", True): return [], (xmin, ymin)
    geoms = [res] if isinstance(res, Polygon) else [g for g in res.geoms if isinstance(g, Polygon)]
    return geoms, (xmin, ymin)

def to_local(polys, origin_xy):
    if not polys: return []
    ox, oy = origin_xy
    out=[]
    for p in polys:
        x,y = p.exterior.xy
        out.append(Polygon(np.c_[np.array(x)-ox, np.array(y)-oy]))
    return out

def pack_in_domain(polys, diam_units, phi2D_target, max_particles, max_trials, seed):
    """Greedy circle packing with erosion-by-radius admissibility and a light grid for speed."""
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
        fit_dom = dom_all.buffer(-r)
        if getattr(fit_dom, "is_empty", True): continue
        fminx, fminy, fmaxx, fmaxy = fit_dom.bounds

        for _ in range(600):
            trials += 1
            if trials > max_trials or area_circ >= target_area or len(placed_xy) >= max_particles:
                break
            x = rng.uniform(fminx, fmaxx); y = rng.uniform(fminy, fmaxy)
            if not fit_dom.contains(Point(x, y)): continue
            if not no_overlap(x, y, r): continue

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

# ------------------------------- Data & models --------------------------------
df_base, src_path = load_dataset(".")
models, meta = train_green_density_models(df_base)  # cached & robust

# ------------------------------- Sidebar --------------------------------------
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base.columns else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", "Alumina")
    d50_um  = st.number_input("D50 (µm)", min_value=1.0, value=60.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)
    target_green = st.slider("Target green density (%TD)", 80, 98, 90, 1)
    guardrails_on = st.toggle("Guardrails", value=True)

    st.divider()
    st.markdown("Data / Models")
    st.write("Source:", os.path.basename(src_path) if src_path else "—")
    st.write("Rows:", f"{len(df_base):,}")
    st.write("Models:", "trained" if models else "—")

# ------------------------------- Balanced Top-5 -------------------------------
def recommend_balanced_top5(material, d50_um, layer_um, target_green, guardrails_on, models, df_source):
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = [float(x) for x in gr["binder_saturation_pct"]]
    spd_lo, spd_hi = [float(x) for x in gr["roller_speed_mm_s"]]

    def candidates(binder_type):
        Xs = np.linspace(sat_lo, sat_hi, 36)
        Ys = np.linspace(spd_lo, spd_hi, 28)
        g = pd.DataFrame([(b, v, d50_um) for v in Ys for b in Xs],
                         columns=["binder_saturation_pct","roller_speed_mm_s","d50_um"])
        g["material"] = material
        pr = predict_quantiles(models, g)
        g = g.join(pr[["td_q10","td_q50","td_q90"]])
        g["binder_type_rec"] = binder_type
        g["score"] = (g["td_q50"] - float(target_green)).abs() + 0.10*np.clip(float(target_green)-g["td_q10"], 0, None)
        return g.sort_values("score", ascending=True)

    gw = candidates("water_based")
    gs = candidates("solvent_based")

    pick_w = gw.head(3).copy()
    pick_s = gs.head(2).copy()
    if len(pick_w) < 3:
        need = 3 - len(pick_w)
        pick_w = pd.concat([pick_w, gw.iloc[len(pick_w):len(pick_w)+need]], ignore_index=True)
    if len(pick_s) < 2:
        need = 2 - len(pick_s)
        pick_s = pd.concat([pick_s, gs.iloc[len(pick_s):len(pick_s)+need]], ignore_index=True)

    out = pd.concat([pick_w, pick_s], ignore_index=True).reset_index(drop=True)
    out = out.rename(columns={
        "binder_saturation_pct":"saturation_pct",
        "roller_speed_mm_s":"roller_speed",
        "td_q50":"pred_q50","td_q10":"pred_q10","td_q90":"pred_q90",
        "binder_type_rec":"binder_type"
    })
    out["layer_um"] = float(layer_um)
    out["id"] = [f"Opt-{i+1}" for i in range(len(out))]
    cols = ["id","binder_type","saturation_pct","roller_speed","layer_um","pred_q10","pred_q50","pred_q90","d50_um","material"]
    return out[cols]

# ------------------------------- Tabs -----------------------------------------
tab_pred, tab_heat, tab_sens, tab_pack, tab_form, tab_twin, tab_health = st.tabs(
    ["Predict (Top-5)", "Heatmap", "Saturation sensitivity", "Qualitative packing", "Formulae", "Digital Twin (Beta)", "Data health"]
)

# ==============================================================================
# Predict (Top-5)
# ==============================================================================
with tab_pred:
    st.subheader("Top-5 parameter sets (3 aqueous + 2 solvent)")
    recs = recommend_balanced_top5(material=material, d50_um=float(d50_um), layer_um=float(layer_um),
                                   target_green=float(target_green), guardrails_on=guardrails_on,
                                   models=models, df_source=df_base)
    st.session_state["top5_recipes_df"] = recs.copy()
    st.dataframe(recs, use_container_width=True, hide_index=True)

# ==============================================================================
# Heatmap (q50 with smoothing)
# ==============================================================================
with tab_heat:
    st.subheader("Predicted Green %TD (q50) — speed × saturation")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = gr["binder_saturation_pct"]; spd_lo, spd_hi = gr["roller_speed_mm_s"]
    Xs = np.linspace(float(sat_lo), float(sat_hi), 85)
    Ys = np.linspace(float(spd_lo), float(spd_hi), 70)

    grid = pd.DataFrame([(b, v, d50_um) for v in Ys for b in Xs], columns=["binder_saturation_pct","roller_speed_mm_s","d50_um"])
    grid["material"] = material
    pr = predict_quantiles(models, grid)
    dfZ = pd.DataFrame({"sat": pr["binder_saturation_pct"], "spd": pr["roller_speed_mm_s"], "z": pr["td_q50"]}).astype(float)
    Z = dfZ.pivot_table(index="spd", columns="sat", values="z").sort_index().sort_index(axis=1)

    # Optional smoothing
    sigma = st.slider("Smoothing σ", 0.0, 2.0, 1.0, 0.1, help="Set to 0 for raw model output.")
    Zv = Z.values.copy()
    if sigma > 0: Zv = gaussian_filter(Zv, sigma=sigma, mode="nearest")

    zmin, zmax = float(np.nanmin(Zv)), float(np.nanmax(Zv))
    z0, z1 = max(40.0, zmin), min(100.0, zmax if zmax > zmin else zmin + 1.0)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=Zv, x=Z.columns.values, y=Z.index.values, zmin=z0, zmax=z1,
                             zsmooth="best", colorscale="Viridis",
                             colorbar=dict(title="%TD (q50)", len=0.82)))
    thr = float(target_green)
    fig.add_trace(go.Contour(z=Zv, x=Z.columns.values, y=Z.index.values,
                             contours=dict(start=thr,end=thr,size=1,coloring="lines"),
                             line=dict(color="#C21807", dash="dash", width=3.0),
                             showscale=False, hoverinfo="skip"))
    fig.update_layout(xaxis_title="Binder Saturation (%)", yaxis_title="Roller Speed (mm/s)",
                      margin=dict(l=10,r=20,t=30,b=35), height=470)
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Saturation sensitivity (q10–q90 band + q50)
# ==============================================================================
with tab_sens:
    st.subheader("Saturation sensitivity at representative speed")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 75)
    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material]*len(sat_axis),
    })
    pred = predict_quantiles(models, grid)
    q10,q50,q90 = pred["td_q10"].astype(float).values, pred["td_q50"].astype(float).values, pred["td_q90"].astype(float).values

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sat_axis, y=q90, line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=sat_axis, y=q10, fill='tonexty', name="q10–q90", mode="lines",
                              line=dict(color="rgba(56,161,105,0.0)"), fillcolor="rgba(56,161,105,0.22)"))
    fig2.add_trace(go.Scatter(x=sat_axis, y=q50, name="q50 (median)", mode="lines", line=dict(width=3, color="#2563eb")))
    fig2.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD",
                       height=420, margin=dict(l=10,r=10,t=10,b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# Qualitative packing (square window, illustrative)
# ==============================================================================
with tab_pack:
    st.subheader("Qualitative packing (illustrative)")
    if not HAVE_SHAPELY:
        st.error("This view requires 'shapely'. Please add it to requirements.txt.")
    else:
        frame_mm = 1.8; px = 1400
        diam_um = sample_psd_um(6500, d50_um, None, None, seed=42)
        diam_mm = diam_um / 1000.0
        phi2D_target = float(np.clip(0.90 * (target_green/100.0), 0.40, 0.88))
        dom = box(0, 0, frame_mm, frame_mm)

        centers, radii, phi2D = pack_in_domain([dom], diam_mm, phi2D_target,
                                               max_particles=2600, max_trials=600_000, seed=1001)

        pores = ~raster_mask(centers, radii, frame_mm, px)
        sat_frac = float(np.clip(target_green/100.0, 0.6, 0.98))
        vmask = voids_from_saturation(pores, saturation=sat_frac)

        fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
        ax.add_patch(Rectangle((0, 0), frame_mm, frame_mm, facecolor=binder_color("water_based"),
                               edgecolor=BORDER, linewidth=1.2))
        ys, xs = np.where(vmask)
        if len(xs):
            xm = xs * (frame_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (frame_mm/vmask.shape[0])
            ax.scatter(xm, ym, s=0.25, c=VOID, alpha=0.95, linewidths=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0, frame_mm); ax.set_ylim(0, frame_mm)
        ax.set_xticks([]); ax.set_yticks([])
        draw_scale_bar(ax, frame_mm, length_um=500)
        ax.set_title(f"D50≈{d50_um:.0f} µm · Sat≈{int(sat_frac*100)}% · Layer≈{layer_um:.0f} µm",
                     fontsize=10, pad=10)
        st.pyplot(fig, clear_figure=True)

# ==============================================================================
# Formulae (placeholder)
# ==============================================================================
with tab_form:
    st.subheader("Formulae")
    st.write("Add your reference equations/notes here (kept for continuity with the original app).")

# ==============================================================================
# Digital Twin (Beta) — STL slicing + recipe-true packing
# ==============================================================================
with tab_twin:
    st.subheader("Digital Twin — recipe-true layer preview")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Digital Twin needs 'trimesh' and 'shapely' (see requirements.txt).")
    else:
        # Use Top-5 from Predict tab
        top5 = st.session_state.get("top5_recipes_df")
        if top5 is None or getattr(top5, "empty", True):
            st.info("Run the Predict tab first to generate Top-5 recipes.")
        else:
            top5 = top5.reset_index(drop=True).copy()

            # Perf & viewing controls
            cfast, cmesh = st.columns([1,1])
            with cfast:
                fast_mode = st.toggle("Fast mode (recommended)", value=True,
                                      help="Reduces particle count/resolution for smoother interaction.")
            with cmesh:
                show_mesh = st.toggle("Show 3D mesh preview", value=not fast_mode)

            left,right = st.columns([1.2,1])
            with left:
                rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
            with right:
                stl_units = st.selectbox("STL units", ["mm","m"], index=0)
                um2unit = 1e-3 if stl_units=="mm" else 1e-6

            c1,c2,c3 = st.columns([2,1,1])
            with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
            with c2: pack_full = st.checkbox("Pack full slice (auto FOV)", value=True)
            with c3: cap = st.slider("Visual cap (particles)", 200, 4000, 1200 if fast_mode else 2200, 50)

            # Chosen recipe
            rec = top5[top5["id"]==rec_id].iloc[0]
            d50_r   = float(rec.get("d50_um",  d50_um))
            layer_r = float(rec.get("layer_um", layer_um))
            sat_pct = float(rec.get("saturation_pct", 85.0))
            binder  = str(rec.get("binder_type","water_based"))

            # Target 2D packing ~ fraction of green target (qualitative)
            phi2D_target = float(np.clip(0.90 * (sat_pct/100.0 + 0.05), 0.40, 0.88))

            # Particle sizes
            diam_um   = sample_psd_um(9000 if not fast_mode else 5000, d50_r, None, None, seed=9991)
            diam_unit = diam_um * um2unit

            mesh=None
            if stl_file is not None:
                try:
                    mesh = trimesh.load(io.BytesIO(stl_file.read()), file_type="stl", force="mesh", process=False)
                    if not isinstance(mesh, trimesh.Trimesh):
                        mesh = mesh.dump(concatenate=True)
                except Exception as e:
                    st.error(f"Could not read STL: {e}")
            else:
                st.info("No STL — using a centered square FOV for microstructure.")

            if mesh is not None:
                minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
                thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
                n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
                st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
                layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
                z = minz + (layer_idx - 0.5) * thickness
                if show_mesh:
                    figm = go.Figure(data=[go.Mesh3d(
                        x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                        color="lightgray", opacity=0.55, flatshading=True, name="Part"
                    )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
                    st.plotly_chart(figm, use_container_width=True)
                # Slice polygons
                polys_world = slice_polys(mesh, z)
            else:
                polys_world = []
                n_layers, layer_idx, z = 1, 1, 0.0

            # Determine local window/FOV
            if pack_full and polys_world:
                dom = unary_union(polys_world)
                xmin, ymin, xmax, ymax = dom.bounds
                fov_x = xmax - xmin; fov_y = ymax - ymin
                fov = max(fov_x, fov_y)
                win = box(xmin, ymin, xmin+fov, ymin+fov)
                clip = dom.intersection(win)
                geoms = [clip] if isinstance(clip, Polygon) else [g for g in clip.geoms if g.area > 1e-9]
                origin = (xmin, ymin)
                polys_local = to_local(geoms, origin)
                render_fov = fov
            else:
                default_fov = max(0.80, 0.02 * d50_r)
                fov_mm = st.slider("Field of view (mm)", float(max(0.20, 0.02 * d50_r)), 6.0, float(default_fov), 0.05)
                if polys_world:
                    polys_clip, origin = crop_window(polys_world, float(fov_mm))
                    polys_local = to_local(polys_clip, origin)
                else:
                    half=fov_mm/2.0; polys_local=[box(0,0,2*half,2*half)]
                render_fov = float(fov_mm)

            # Pack circles
            centers, radii, phi2D = pack_in_domain(polys_local, diam_unit, phi2D_target,
                                                   max_particles=int(cap),
                                                   max_trials=300_000 if fast_mode else 480_000,
                                                   seed=20_000 + (0 if mesh is None else layer_idx))

            # Render two panels
            col1, col2 = st.columns(2)
            with col1:
                figA, axA = plt.subplots(figsize=(5.2,5.2), dpi=170 if fast_mode else 188)
                axA.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor="white",
                                        edgecolor=BORDER, linewidth=1.2))
                for (x,y), r in zip(centers, radii):
                    axA.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
                axA.set_aspect('equal','box'); axA.set_xlim(0,render_fov); axA.set_ylim(0,render_fov); axA.set_xticks([]); axA.set_yticks([])
                draw_scale_bar(axA, render_fov)
                axA.set_title("Particles only", fontsize=10)
                st.pyplot(figA, use_container_width=True)

            with col2:
                px = 680 if fast_mode else 900
                pores = ~raster_mask(centers, radii, render_fov, px)
                vmask = voids_from_saturation(pores, saturation=float(sat_pct)/100.0,
                                             rng=np.random.default_rng(123 + (0 if mesh is None else layer_idx)))
                figB, axB = plt.subplots(figsize=(5.2,5.2), dpi=170 if fast_mode else 188)
                axB.add_patch(Rectangle((0,0), render_fov, render_fov,
                                        facecolor=binder_color(binder), edgecolor=BORDER, linewidth=1.2))
                ys, xs = np.where(vmask)
                if len(xs):
                    xm = xs * (render_fov/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (render_fov/vmask.shape[0])
                    axB.scatter(xm, ym, s=0.32 if not fast_mode else 0.28, c=VOID, alpha=0.96, linewidth=0)
                for (x,y), r in zip(centers, radii):
                    axB.add_patch(Circle((x,y), r, facecolor=PARTICLE, edgecolor=EDGE, linewidth=0.25))
                axB.set_aspect('equal','box'); axB.set_xlim(0,render_fov); axB.set_ylim(0,render_fov); axB.set_xticks([]); axB.set_yticks([])
                draw_scale_bar(axB, render_fov)
                axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
                st.pyplot(figB, use_container_width=True)

            st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

# ==============================================================================
# Data health — coverage & ≥90%TD evidence near D50
# ==============================================================================
with tab_health:
    st.subheader("Training coverage & ≥90%TD evidence near this D50")
    d = df_base.copy()
    if "material" in d.columns:
        d = d[d["material"].astype(str)==str(material)]
    if "d50_um" in d.columns:
        lo, hi = 0.8*float(d50_um), 1.2*float(d50_um)
        d = d[(d["d50_um"]>=lo) & (d["d50_um"]<=hi)]
    c1, c2 = st.columns([1,2])
    with c1:
        if "green_pct_td" in d.columns:
            st.metric("Rows in ±20% D50 window", len(d))
            st.metric("Seen ≥90%TD cases", int((d["green_pct_td"]>=90).sum()))
            if len(d): st.metric("Best %TD", f"{float(d['green_pct_td'].max()):.1f}")
        else:
            st.info("No green %TD column found after normalization.")
    with c2:
        if not d.empty and "d50_um" in d.columns and "green_pct_td" in d.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["d50_um"], y=d["green_pct_td"], mode="markers", name="train pts"))
            fig.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
            fig.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD",
                              height=360, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training points in this window.")
