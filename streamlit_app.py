# streamlit_app.py — BJAM Recommender + Digital Twin (robust STL packing, guardrails hardened)
from __future__ import annotations
import io, math, importlib.util
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Optional deps for Digital Twin
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH: import trimesh
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    from shapely import wkb

# Project utilities
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
    suggest_binder_family,
)

# ================= Page =================
st.set_page_config(page_title="BJAM Predictions", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
:root{--font:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif}
.stApp{background:#FFFDF7!important}
html,body,[class*="css"]{font-family:var(--font)!important;color:#111827!important}
.block-container{max-width:1200px}
.kpi{background:#fff;border-radius:12px;padding:16px 18px;border:1px solid rgba(0,0,0,.06);box-shadow:0 1px 2px rgba(0,0,0,.03)}
.kpi .kpi-label{font-weight:600}
.kpi .kpi-value{font-weight:800;font-size:2.1rem}
.kpi .kpi-unit{font-weight:700}
.badge{display:inline-block;padding:.2rem .5rem;border:1px solid #ddd;border-radius:6px;background:#fff}
.footer{margin:24px 0 6px;text-align:center}
</style>
""", unsafe_allow_html=True)

# ================= Data/Models =================
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# ================= Helpers =================
def safe_guardrails(d50_um: float, on: bool):
    try:
        gr = guardrail_ranges(float(d50_um), on=bool(on))
        b0,b1 = [float(x) for x in gr["binder_saturation_pct"]]
        v0,v1 = [float(x) for x in gr["roller_speed_mm_s"]]
        t0,t1 = [float(x) for x in gr["layer_thickness_um"]]
    except Exception:
        pri = physics_priors(float(d50_um), binder_type_guess=None)
        b = float(np.clip(pri["binder_saturation_pct"], 55, 105))
        v = float(np.clip(pri["roller_speed_mm_s"], 0.6, 5.0))
        t = float(np.clip(pri["layer_thickness_um"], 2.0, 300.0))
        b0,b1 = b-15, b+15
        v0,v1 = v-1.0, v+1.0
        t0,t1 = 0.6*t, 1.4*t

    def _fix(a,b,eps):
        a=float(a); b=float(b)
        if not np.isfinite(a): a=0.0
        if not np.isfinite(b): b=a+eps
        if b <= a + eps: b = a + eps
        return a,b
    b0,b1 = _fix(b0,b1,1.0)
    v0,v1 = _fix(v0,v1,0.05)
    t0,t1 = _fix(t0,t1,1.0)
    return {
        "binder_saturation_pct": (max(40.0,b0), min(120.0,b1)),
        "roller_speed_mm_s": (max(0.2,v0), min(6.0,v1)),
        "layer_thickness_um": (max(2.0,t0), min(400.0,t1)),
    }

def binder_hex(name: str) -> str:
    s = (name or "").lower()
    if "water" in s: return "#F2D06F"
    if "solvent" in s: return "#F2B233"
    if "acryl" in s: return "#FFD166"
    if "furan" in s: return "#F5C07A"
    return "#F4B942"

# ================= Sidebar =================
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data: {Path(src).name} · rows={len(df_base):,}")
        st.download_button("Download dataset (CSV)",
                           data=df_base.to_csv(index=False).encode("utf-8"),
                           file_name=Path(src).name, mime="text/csv")
    else:
        st.warning("No dataset found; physics-only priors will be used.")

    st.divider()
    guardrails_on = st.toggle("Guardrails", True,
                              help="ON = stable windows (binder ~60–110%, speed ≈1.2–3.5 mm/s, layer ≈3–5×D50)")
    target_green = st.slider("Target green %TD", 80, 98, 90, 1)
    st.caption("Selector prefers q10 ≥ target for conservatism.")

# ================= Header =================
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot • Guardrails hardened • Digital Twin for STL slices")

with st.expander("Preview source data", expanded=False):
    if len(df_base): st.dataframe(df_base.head(25), use_container_width=True)
    else: st.info("No rows to preview.")

st.divider()

# ================= Inputs =================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Inputs")
    mode = st.radio("Material source", ["From dataset","Custom"], horizontal=True)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []

    if mode=="From dataset" and materials:
        material = st.selectbox("Material", materials, index=0)
        d50_default = float(df_base.loc[df_base["material"].astype(str)==material, "d50_um"].dropna().median()) if "d50_um" in df_base else 30.0
        material_class = (
            df_base.loc[df_base["material"].astype(str)==material, "material_class"]
            .dropna().astype(str).iloc[0]
            if {"material","material_class"}.issubset(df_base.columns) and
               (df_base["material"].astype(str)==material).any()
            else "metal"
        )
    else:
        material = st.text_input("Material (custom)", "Al2O3")
        material_class = st.selectbox("Material class", ["metal","oxide","carbide","other"], index=1)
        d50_default = 30.0

    d50_um = st.number_input("D50 (µm)", 1.0, 150.0, float(d50_default), 1.0)
    pri = physics_priors(d50_um, binder_type_guess=None)
    gr = safe_guardrails(d50_um, on=guardrails_on)

    t_lo, t_hi = gr["layer_thickness_um"]
    def_layer = float(np.clip(pri["layer_thickness_um"], t_lo, t_hi))
    layer_um = st.slider("Layer thickness (µm)", float(round(t_lo)), float(round(t_hi)),
                         float(round(def_layer)), 1.0)

    auto_binder = suggest_binder_family(material, material_class)
    binder_choice = st.selectbox("Binder family", [f"auto ({auto_binder})","solvent_based","water_based"])
    binder_family = auto_binder if binder_choice.startswith("auto") else binder_choice

with right:
    st.subheader("Priors")
    def kpi(col, label, value, unit="", sub=""):
        col.markdown(f"""
        <div class="kpi">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    kpi(c1, "Prior binder", f"{pri['binder_saturation_pct']:.0f}", "%")
    kpi(c2, "Prior speed", f"{pri['roller_speed_mm_s']:.2f}", "mm/s")
    kpi(c3, "Layer/D50", f"{layer_um/d50_um:.2f}", "×")

st.divider()

# ================= Recommendations =================
st.subheader("Recommended parameters")
colL, colR = st.columns([1,1])
top_k = colL.slider("How many to show", 3, 8, 5, 1)
diverse_pick = colR.toggle("Use diverse 3-water + 2-solvent", False,
                           help="Force binder % diversity and two solvent trials.")

def dense_candidates(models, d50_um, layer_um, material, material_class, binder_family, gr, nx=55, ny=41):
    b_lo,b_hi=gr["binder_saturation_pct"]; s_lo,s_hi=gr["roller_speed_mm_s"]
    Xs=np.linspace(float(b_lo),float(b_hi),nx); Ys=np.linspace(float(s_lo),float(s_hi),ny)
    df=pd.DataFrame([(b,v,layer_um,d50_um,material) for b in Xs for v in Ys],
                    columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    df["material_class"]=material_class; df["binder_type_rec"]=binder_family
    pred=predict_quantiles(models, df)
    return df.reset_index(drop=True).join(pred[["td_q10","td_q50","td_q90"]].reset_index(drop=True))

def score_and_pick(df, target):
    s = 3.0*np.clip(float(target)-df["td_q10"],0,None) + (df["td_q50"]-float(target)).abs()
    return df.assign(_score=s).sort_values("_score")

def pick_diverse(df_sorted: pd.DataFrame, k: int, min_sat_gap=3.0, min_spd_gap=0.12):
    chosen=[]
    for _,row in df_sorted.iterrows():
        sat=float(row["binder_saturation_pct"]); spd=float(row["roller_speed_mm_s"])
        if all(abs(sat-float(r["binder_saturation_pct"]))>=min_sat_gap and
               abs(spd-float(r["roller_speed_mm_s"]))>=min_spd_gap for r in chosen):
            chosen.append(row)
        if len(chosen)>=k: break
    if len(chosen)<k:
        extra = df_sorted.iloc[:(k-len(chosen))]
        chosen.extend([extra.iloc[i] for i in range(len(extra))])
    return pd.DataFrame(chosen)

btn = st.button("Recommend", type="primary", use_container_width=True)

if btn:
    if not diverse_pick:
        recs = copilot(material=material, d50_um=float(d50_um), df_source=df_base, models=models,
                       guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k))
        recs["binder_type"] = binder_family
        out = recs.rename(columns={
            "binder_type":"Binder","binder_%":"Binder sat (%)","speed_mm_s":"Speed (mm/s)",
            "layer_um":"Layer (µm)","predicted_%TD_q10":"q10 %TD","predicted_%TD_q50":"q50 %TD",
            "predicted_%TD_q90":"q90 %TD","meets_target_q10":f"Meets target (q10 ≥ {target_green}%)",
        })
        st.session_state["top_recipes_df"]=recs.copy()
    else:
        recs_list=[]
        for fam, need in [("water_based",3),("solvent_based",2)]:
            cand = dense_candidates(models, float(d50_um), float(layer_um), material, material_class, fam, gr)
            sorted_c = score_and_pick(cand, float(target_green))
            picked = pick_diverse(sorted_c, need, min_sat_gap=3.0, min_spd_gap=0.12)
            picked["binder_type"]=fam
            recs_list.append(picked)
        recs = pd.concat(recs_list, ignore_index=True)
        recs = recs.rename(columns={
            "binder_saturation_pct":"binder_%", "roller_speed_mm_s":"speed_mm_s",
            "layer_thickness_um":"layer_um", "td_q10":"predicted_%TD_q10",
            "td_q50":"predicted_%TD_q50","td_q90":"predicted_%TD_q90"
        })
        recs["meets_target_q10"]=recs["predicted_%TD_q10"]>=float(target_green)
        recs["material"]=material; recs["d50_um"]=float(d50_um)
        st.session_state["top_recipes_df"]=recs.copy()
        out = recs.rename(columns={
            "binder_type":"Binder","binder_%":"Binder sat (%)","speed_mm_s":"Speed (mm/s)",
            "layer_um":"Layer (µm)","predicted_%TD_q10":"q10 %TD","predicted_%TD_q50":"q50 %TD",
            "predicted_%TD_q90":"q90 %TD","meets_target_q10":f"Meets target (q10 ≥ {target_green}%)",
        })
    st.dataframe(out, use_container_width=True)
    st.download_button("Download recommendations (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name="bjam_recommendations.csv", use_container_width=True, type="secondary")
else:
    st.info("Click Recommend to generate top-k parameter sets.")

st.divider()

# ================= Visuals (kept) =================
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing (2D slice)",
    "Pareto frontier",
    "Formulae",
    "Digital Twin",
])

def grid_for_context(gr, layer_um, d50_um, material, material_class, binder_family, nx=55, ny=45):
    b_lo,b_hi=gr["binder_saturation_pct"]; s_lo,s_hi=gr["roller_speed_mm_s"]
    Xs=np.linspace(float(b_lo),float(b_hi),nx); Ys=np.linspace(float(s_lo),float(s_hi),ny)
    grid=pd.DataFrame([(b,v,layer_um,d50_um,material) for b in Xs for v in Ys],
                      columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    grid["material_class"]=material_class; grid["binder_type_rec"]=binder_family
    return grid, Xs, Ys

with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD (q50)")
    grid, Xs, Ys = grid_for_context(gr, layer_um, d50_um, material, material_class, binder_family)
    sc = predict_quantiles(models, grid)
    Z = sc.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs),len(Ys)).T
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=list(Xs),y=list(Ys),z=Z,colorscale="Viridis",colorbar=dict(title="%TD")))
    fig.add_trace(go.Contour(x=list(Xs),y=list(Ys),z=Z,contours=dict(start=90,end=90,size=1,coloring="none"),
                             line=dict(width=3),showscale=False,name="90% TD"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Roller speed (mm/s)",
                      height=520, margin=dict(l=10,r=10,t=40,b=10),
                      title=f"Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · {material} ({material_class}) · Source={Path(src).name if src else '—'}")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90)")
    sats = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 61)
    curve = pd.DataFrame({
        "binder_saturation_pct":sats,
        "roller_speed_mm_s": float(np.mean(gr["roller_speed_mm_s"])),
        "layer_thickness_um": float(layer_um),
        "d50_um": float(d50_um),
        "material": material,
        "material_class": material_class,
        "binder_type_rec": binder_family,
    })
    P = predict_quantiles(models, curve)
    fig2, ax = plt.subplots(figsize=(7.0,4.2), dpi=170)
    ax.plot(P["binder_saturation_pct"], P["td_q50"], lw=2.0, color="#1f77b4", label="q50")
    ax.fill_between(P["binder_saturation_pct"], P["td_q10"], P["td_q90"], alpha=0.18, label="q10–q90")
    ax.axhline(target_green, ls="--", lw=1.2, color="#374151", label=f"Target {target_green}%")
    ax.set_xlabel("Binder saturation (%)"); ax.set_ylabel("Predicted green %TD")
    ax.grid(True, axis="y", alpha=0.18); ax.legend(frameon=False)
    st.pyplot(fig2, clear_figure=True)

with tabs[2]:
    st.subheader("Packing — 2D slice (toy)")
    c1,c2,c3,c4=st.columns(4)
    side_mult=c1.slider("Square side (×D50)",10,60,20,2)
    cv_pct=c2.slider("Polydispersity (CV %)",0,60,20,5)
    densify=c3.toggle("Densify packing", False)
    seed=c4.number_input("Seed",0,9999,0,1)
    W=float(int(side_mult)); rng=np.random.default_rng(int(seed))
    N=int((520 if densify else 260)*(W/20)**2)
    if cv_pct<=0: diam=np.full(N, float(d50_um))
    else:
        sigma=np.sqrt(np.log(1+(cv_pct/100)**2)); diam=float(d50_um)*rng.lognormal(0.0, sigma, N)
        diam=np.clip(diam, 0.4*float(d50_um), 1.8*float(d50_um))
    rad=0.5*np.sort(diam/d50_um)[::-1]
    pts=[]; MAX=40000; att=0
    def can_place(x,y,r):
        if x-r<0 or x+r>W or y-r<0 or y+r>W: return False
        for (px,py,pr) in pts:
            if (x-px)**2+(y-py)**2<(r+pr)**2: return False
        return True
    for r in rad:
        for _ in range(280 if densify else 240):
            x=rng.uniform(r,W-r); y=rng.uniform(r,W-r)
            if can_place(x,y,r): pts.append((x,y,r)); break
        att+=1
        if att>MAX: break
    rs=np.array([r for (_,_,r) in pts]); phi=(np.pi*np.sum(rs**2))/(W*W) if W>0 else 0.0
    xx=np.linspace(0,W,int(21*W)); yy=np.linspace(0,W,int(21*W)); X,Y=np.meshgrid(xx,yy)
    solid=np.zeros_like(X,dtype=bool)
    for (x,y,r) in pts: solid |= (X-x)**2+(Y-y)**2<=r**2
    figA, axA = plt.subplots(figsize=(1.6,1.6), dpi=300)
    axA.set_aspect('equal','box'); axA.add_patch(plt.Rectangle((0,0),W,W,fill=False, lw=1.1, color='#111827'))
    for (x,y,r) in pts: axA.add_patch(plt.Circle((x,y),r,fc='#3b82f6',ec='#111827',lw=0.3,alpha=0.92))
    axA.set_xlim(0,W); axA.set_ylim(0,W); axA.set_xticks([]); axA.set_yticks([])
    st.pyplot(figA, clear_figure=True); st.caption(f"φ≈{phi*100:.1f}% • side≈{W*d50_um:.0f} µm")

with tabs[3]:
    st.subheader("Pareto frontier — Binder vs green %TD (q50)")
    gridP,_,_ = grid_for_context(gr, layer_um, d50_um, material, material_class, binder_family, nx=80, ny=1)
    scP = predict_quantiles(models, gridP)[["binder_saturation_pct","td_q50"]].dropna().sort_values("binder_saturation_pct")
    pts_line=scP.values; idx=[]; best=-1
    for i,(b,td) in enumerate(pts_line[::-1]):
        if td>best: idx.append(len(pts_line)-1-i); best=td
    idx=sorted(idx)
    fig4=go.Figure()
    fig4.add_trace(go.Scatter(x=scP["binder_saturation_pct"],y=scP["td_q50"],mode="markers",marker=dict(size=6,color="#1f77b4"),name="Candidates"))
    fig4.add_trace(go.Scatter(x=scP.iloc[idx]["binder_saturation_pct"],y=scP.iloc[idx]["td_q50"],
                              mode="lines+markers",marker=dict(size=7,color="#111827"),
                              line=dict(width=2,color="#111827"),name="Pareto"))
    fig4.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD (q50)",
                       height=460, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)

with tabs[4]:
    st.subheader("Formulae (symbols)")
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")

# ================= Digital Twin (robust) =================
@st.cache_data(show_spinner=False)
def _slice_polys_wkb(_mesh_key, z: float) -> Tuple[bytes, ...]:
    try:
        mesh = st.session_state.get("_dtw_mesh")
        if mesh is None: return tuple()
        zmin,zmax = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        span=max(zmax-zmin,1e-6); eps=1e-6*span
        for zi in (z, z+eps, z-eps, z+2*eps, z-2*eps):
            sec = mesh.section(plane_origin=(0,0,float(zi)), plane_normal=(0,0,1))
            if sec is None: continue
            planar,_=sec.to_planar()
            rings = getattr(planar,"polygons_full",None) or getattr(planar,"polygons_closed",None)
            if not rings: continue
            polys=[]
            for ring in rings:
                try:
                    p=Polygon(ring)
                    if p.is_valid and p.area>1e-9: polys.append(p.buffer(0))
                except Exception: pass
            if polys: return tuple(p.wkb for p in polys)
        return tuple()
    except Exception:
        return tuple()

@st.cache_data(show_spinner=False)
def _crop_local(_polys_wkb: Tuple[bytes, ...], desired_fov: float | None):
    if not _polys_wkb: return tuple(), (0.0,0.0), 0.0
    polys=[wkb.loads(p) for p in _polys_wkb]; dom=unary_union(polys)
    xmin,ymin,xmax,ymax=dom.bounds; bbox=max(xmax-xmin,ymax-ymin)
    fov=float(bbox) if (desired_fov is None or desired_fov<=0) else float(min(desired_fov,bbox))
    cx,cy=dom.centroid.x, dom.centroid.y; half=fov/2
    x0,y0=cx-half, cy-half; win=box(x0,y0,x0+fov,y0+fov)
    clip=dom.intersection(win)
    if getattr(clip,"is_empty",True): return tuple(), (x0,y0), fov
    geoms=[clip] if isinstance(clip,Polygon) else [g for g in clip.geoms if isinstance(g,Polygon)]
    local=[]
    for g in geoms:
        x,y=g.exterior.xy
        local.append(Polygon(np.c_[np.array(x)-x0, np.array(y)-y0]).wkb)
    return tuple(local), (x0,y0), fov

@st.cache_data(show_spinner=False)
def _hex_pack_target(_key, polys_wkb: Tuple[bytes, ...], d50_unit: float,
                     phi_target: float, fov: float, cap: int, jitter: float):
    """
    Robust packer:
      • Stage 1: hex grid + erosion (true fit)
      • Stage 2: center-in-domain (no erosion)
      • Stage 3: compute radius from φ target, domain area, cap (guaranteed fill)
    Returns: centers, radii, phi2D, stage_used (1/2/3)
    """
    if not HAVE_SHAPELY or not polys_wkb:
        return np.empty((0,2)), np.empty((0,)), 0.0, 0
    polys=[wkb.loads(p) for p in polys_wkb]; dom=unary_union(polys)
    if getattr(dom,"is_empty",True):
        return np.empty((0,2)), np.empty((0,)), 0.0, 0

    target=float(np.clip(phi_target,0.40,0.88))
    area_dom=float(dom.area)
    rng = np.random.default_rng(1234)

    def _grid(radius):
        s=2.0*radius; dy=radius*np.sqrt(3.0)
        xs=np.arange(radius, fov-radius, s); ys=np.arange(radius, fov-radius, dy)
        if len(xs)==0 or len(ys)==0: return np.empty((0,2))
        pts=[]
        for j,yy in enumerate(ys):
            xoff=0.0 if (j%2==0) else radius
            for xx in xs:
                x0=xx+xoff
                if x0>fov-radius: continue
                pts.append((x0,yy))
        return np.array(pts,float)

    def _stage1_try(k):
        r=max(1e-12, d50_unit/2.0)*k
        C=_grid(r)
        if C.size==0: return np.empty((0,2)), np.empty((0,)), 0.0
        if jitter>0: C += rng.uniform(-jitter*r, jitter*r, C.shape)
        try:
            fit=dom.buffer(-r)
            if getattr(fit,"is_empty",True): return np.empty((0,2)), np.empty((0,)), 0.0
        except Exception:
            fit=dom
        keep=[i for i,(cx,cy) in enumerate(C)
              if fit.contains(Polygon([(cx+r,cy),(cx,cy+r),(cx-r,cy),(cx,cy-r)]))]
        C=C[keep]
        if len(C)>cap: C=C[:cap]
        R=np.full(len(C), r, float)
        phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
        return C,R,phi

    # Stage 1: bisection on k
    bestC,bestR,bestPhi=np.empty((0,2)),np.empty((0,)),0.0
    lo,hi=0.15,2.0
    for _ in range(24):
        mid=(lo+hi)/2
        C,R,phi=_stage1_try(mid)
        if R.size==0: hi=mid; continue
        bestC,bestR,bestPhi=C,R,phi
        if phi<target: lo=mid
        else: hi=mid
    if bestR.size>0:
        return bestC,bestR,bestPhi,1

    # Stage 2: center-in-domain (no erosion)
    r2=max(1e-12, d50_unit/2.0)*0.6  # smaller to encourage placement
    C=_grid(r2)
    if C.size>0:
        if jitter>0: C += rng.uniform(-jitter*r2, jitter*r2, C.shape)
        keep=[i for i,(cx,cy) in enumerate(C) if dom.contains(Point(cx,cy))]
        C=C[keep]
        if len(C)>cap: C=C[:cap]
        R=np.full(len(C), r2, float)
        phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
        if R.size>0:
            return C,R,phi,2

    # Stage 3: solve radius from φ target and cap ⇒ r = sqrt(target*Area/(N*pi))
    N=max(1,int(cap))
    r3=math.sqrt(max(target*area_dom/(N*math.pi), 1e-18))
    # don’t exceed FOV/4 to avoid degenerate zeros on skinny slices
    r3=min(r3, 0.25*max(1e-12,fov))
    C=_grid(r3)
    if C.size==0:
        # final minimal sprinkle
        C = rng.uniform(r3, fov-r3, size=(min(N, max(64,int(0.15*cap))),2))
    if jitter>0: C += rng.uniform(-jitter*r3, jitter*r3, C.shape)
    keep=[i for i,(cx,cy) in enumerate(C) if dom.contains(Point(cx,cy))]
    C=C[keep]
    if len(C)>cap: C=C[:cap]
    R=np.full(len(C), r3, float)
    phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
    return C,R,phi,3

@st.cache_data(show_spinner=False)
def _raster_solids(_key, centers: np.ndarray, radii: np.ndarray, fov: float, px: int):
    if centers.size==0: return np.zeros((px,px),bool)
    y,x=np.mgrid[0:px,0:px]; s=fov/px; xx=x*s; yy=(px-y)*s
    mask=np.zeros((px,px),bool)
    for (cx,cy),r in zip(centers,radii):
        d2=(xx-cx)**2+(yy-cy)**2; mask |= (d2<=r*r)
    return mask

with tabs[5]:
    st.subheader("Digital Twin — STL slice + particle packing (robust)")

    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Please add 'trimesh' and 'shapely' to requirements.txt")
    else:
        trials=st.session_state.get("top_recipes_df")
        use_trials=st.checkbox("Use generated trials (when available)", value=True)
        if use_trials and isinstance(trials, pd.DataFrame) and not trials.empty:
            idx=st.selectbox("Pick a trial", list(range(len(trials))), format_func=lambda i: f"Trial-{i+1}", index=0)
            row=trials.iloc[idx]
            binder_for_twin=str(row.get("binder_type","water_based"))
            sat_pct_for_twin=float(row.get("binder_%", 80.0))
            layer_um_for_twin=float(row.get("layer_um", layer_um))
            d50_um_for_twin=float(row.get("d50_um", d50_um))
        else:
            binder_for_twin=binder_family
            sat_pct_for_twin=st.slider("Binder saturation for visualization (%)", 50, 100, 80, 1)
            layer_um_for_twin=layer_um; d50_um_for_twin=d50_um

        c0,c1,c2,c3,c4 = st.columns([2,1,1,1,1])
        with c0: stl=st.file_uploader("Upload STL", type=["stl"])
        with c1: use_cube=st.checkbox("Use 10 mm cube", value=(stl is None))
        with c2:
            stl_unit = st.selectbox("Model units", ["mm","m","inch","custom"], index=0)
        with c3:
            custom_mm_per_unit = st.number_input("Custom: mm per unit", 0.001, 10000.0, 1.0, 0.001)
        with c4:
            show_mesh = st.checkbox("Show 3D mesh preview", value=True)

        if stl_unit=="mm": um2unit = 1e-3
        elif stl_unit=="m": um2unit = 1e-6
        elif stl_unit=="inch": um2unit = (1.0/25.4)*1e-3
        else: um2unit = (1.0/custom_mm_per_unit) * 1e-3

        mesh=None
        if use_cube:
            mesh=trimesh.creation.box(extents=(10.0,10.0,10.0))
        elif stl is not None:
            try:
                mesh=trimesh.load(io.BytesIO(stl.read()), file_type="stl", force="mesh", process=False)
                if not isinstance(mesh,trimesh.Trimesh): mesh=mesh.dump(concatenate=True)
            except Exception as e:
                st.error(f"Could not read STL: {e}")

        if mesh is None:
            st.info("Upload an STL or select the sample cube.")
        else:
            st.session_state["_dtw_mesh"]=mesh
            thickness=float(layer_um_for_twin)*um2unit
            d50_unit=float(d50_um_for_twin)*um2unit

            zmin,zmax=float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
            n_layers=max(1, int((zmax-zmin)/max(thickness,1e-12)))
            st.caption(f"Layers: {n_layers} · Z span: {zmax-zmin:.3f} {stl_unit}")

            lcol,rcol=st.columns([2,1])
            with lcol: layer_idx=st.slider("Layer index",1,n_layers,1)
            with rcol: px_user=st.slider("Render resolution (px)", 300, 1400, 800, 50)

            if show_mesh:
                figm = go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                    color="lightgray", opacity=0.55, flatshading=True, name="Part"
                )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=320)
                st.plotly_chart(figm, use_container_width=True)

            z=zmin+(layer_idx-0.5)*thickness
            mkey=hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))
            polys_wkb=_slice_polys_wkb(mkey, z)
            if not polys_wkb:
                st.warning("Empty slice at this layer (try another layer or adjust units).")
            else:
                # target φ2D and auto FOV
                phi_TPD=0.90; phi2D_target=float(np.clip(0.90*phi_TPD, 0.40, 0.88))
                auto_fov=st.checkbox("Auto FOV to hit φ target", True)
                cap=st.slider("Particle cap", 500, 20000, 2200, 100)

                polys_tmp=[wkb.loads(p) for p in polys_wkb]; dom_tmp=unary_union(polys_tmp)
                bx0,by0,bx1,by1=dom_tmp.bounds; slice_side=float(max(bx1-bx0, by1-by0))

                r0=max(1e-12, d50_unit/2.0)
                est_cell=np.pi*(r0**2)/phi2D_target
                fov_auto=float(np.sqrt(max(cap*est_cell, 1e-9)))
                if auto_fov:
                    desired_fov=float(np.clip(fov_auto, 20.0*d50_unit, slice_side))
                else:
                    desired_fov=st.slider("FOV (model units)", float(max(5.0*d50_unit, 0.2)), float(slice_side),
                                          float(min(max(fov_auto, 10.0*d50_unit), slice_side)), 0.05)

                local_wkb, origin, fov = _crop_local(polys_wkb, desired_fov)
                px_auto=int(np.ceil((fov/max(d50_unit,1e-12))*6.0)); px_eff=int(max(px_user, px_auto, 400))

                centers,radii,phi2D,stage = _hex_pack_target(
                    (hash(local_wkb), round(d50_unit,9), round(phi2D_target,4), round(fov,6), cap),
                    local_wkb, d50_unit, phi2D_target, fov, cap, jitter=0.12
                )

                solids=_raster_solids((hash(centers.tobytes()) if centers.size else 0, px_eff, round(fov,6)),
                                      centers, radii, fov, px_eff)
                pores=~solids
                sat=float(np.clip(sat_pct_for_twin/100.0, 0.01, 0.99))
                pore_idx=np.flatnonzero(pores.ravel()); rng=np.random.default_rng(42+layer_idx+int(sat_pct_for_twin))
                k=int((1.0-sat)*len(pore_idx)); voids=np.zeros_like(pores,bool)
                if k>0 and len(pore_idx)>0:
                    choose=rng.choice(pore_idx, size=min(k,len(pore_idx)), replace=False)
                    voids.ravel()[choose]=True

                # Render
                img_particles=np.ones((px_eff,px_eff,3),float); img_particles[solids]=np.array([0.18,0.38,0.96])
                b_rgb=tuple(int(binder_hex(binder_for_twin)[i:i+2],16)/255.0 for i in (1,3,5))
                img_layer=np.ones((px_eff,px_eff,3),float); img_layer[:]=b_rgb
                img_layer[voids]=np.array([1,1,1]); img_layer[solids]=np.array([0.18,0.38,0.96])

                colA,colB=st.columns(2)
                with colA:
                    st.caption("Particles only")
                    figA=go.Figure(go.Image(z=(img_particles*255).astype(np.uint8)))
                    figA.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                    st.plotly_chart(figA, use_container_width=True)
                with colB:
                    st.caption(f"{binder_for_twin} · Sat {int(sat_pct_for_twin)}%")
                    figB=go.Figure(go.Image(z=(img_layer*255).astype(np.uint8)))
                    figB.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                    st.plotly_chart(figB, use_container_width=True)

                reason = {1:"fit (eroded)", 2:"center-in-domain", 3:"radius-from-φ"}
                st.caption(
                    f"<span class='badge'>Layer {layer_idx}/{n_layers} • FOV={fov:.3f} {stl_unit} • "
                    f"d50={d50_unit:.5g} {stl_unit} • φ₂D(target)≈{phi2D_target:.2f} • "
                    f"φ₂D(achieved)≈{phi2D:.2f} • particles={len(radii)} • pack={reason.get(stage,'—')}</span>",
                    unsafe_allow_html=True
                )

# ================= Footer =================
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note":"No trained models (physics-only)."})
    st.write("Trials cached:", isinstance(st.session_state.get("top_recipes_df"), pd.DataFrame))

st.markdown(f"""
<div class="footer">
  <strong>© {datetime.now().year} Bhargavi Mummareddy</strong> ·
  <a href="mailto:mummareddybhargavi@gmail.com">mummareddybhargavi@gmail.com</a>
</div>
""", unsafe_allow_html=True)
