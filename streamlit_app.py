# -*- coding: utf-8 -*-
# BJAM — Binder-Jet AM Recommender + Digital Twin (STL-first, robust slicing/packing)
# Keeps all original tabs; only adds a dependable STL Digital Twin.
from __future__ import annotations
import io, math, importlib.util, os
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

# Optional geometry libs
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH: import trimesh
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box
    from shapely.ops import unary_union
    from shapely import wkb

# ---------- Shared project utils (unchanged from your repo) ----------
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    guardrail_ranges,
)

# ---------- Page ----------
st.set_page_config(page_title="BJAM Predictions", page_icon="🟨", layout="wide")

st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#FFFDF8 0%,#FFF7ED 45%,#FFF1DE 100%)}
[data-testid="stSidebar"]{background:#fffdfa;border-right:1px solid #f3e8d9}
.stTabs [data-baseweb="tab-list"]{gap:10px}
.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #f3e8d9;border-radius:10px;padding:6px 10px}
.kpi{background:#fff;border-radius:12px;padding:12px 14px;border:1px solid rgba(0,0,0,.06)}
.kpi .v{font-weight:800;font-size:1.8rem}
.badge{display:inline-block;padding:.18rem .5rem;border:1px solid #e6dccc;border-radius:6px;background:#fff}
</style>
""", unsafe_allow_html=True)

# ---------- Data & models ----------
df_base, src_path = load_dataset(".")
models, meta = train_green_density_models(df_base)

# ---------- Small helpers ----------
def _binder_hex(name: str) -> str:
    s=(name or "").lower()
    if "water"   in s: return "#F2D06F"
    if "solvent" in s: return "#F2B233"
    if "acryl"   in s: return "#FFD166"
    if "furan"   in s: return "#F5C07A"
    return "#F4B942"

def _safe_guardrails(d50_um: float, on: bool) -> Dict[str, Tuple[float,float]]:
    # Prevents Streamlit slider errors if ranges collapse
    try:
        gr = guardrail_ranges(float(d50_um), on=bool(on))
        b0,b1 = [float(x) for x in gr["binder_saturation_pct"]]
        v0,v1 = [float(x) for x in gr["roller_speed_mm_s"]]
        t0,t1 = [float(x) for x in gr["layer_thickness_um"]]
    except Exception:
        # conservative fallback
        b0,b1 = 60.0, 110.0
        v0,v1 = 1.2, 3.5
        t0,t1 = max(3.0*d50_um,8.0), min(5.0*d50_um,300.0)

    def fix(a,b,eps):
        a=float(a); b=float(b)
        if not np.isfinite(a): a=0.0
        if not np.isfinite(b): b=a+eps
        if b <= a + eps: b = a + eps
        return a,b
    b0,b1 = fix(b0,b1,1.0)
    v0,v1 = fix(v0,v1,0.05)
    t0,t1 = fix(t0,t1,1.0)
    return {"binder_saturation_pct":(b0,b1), "roller_speed_mm_s":(v0,v1), "layer_thickness_um":(t0,t1)}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []
    material = st.selectbox("Material", materials, index=0) if materials else st.text_input("Material", "Silicon Carbide (SiC)")
    d50_um = st.number_input("D50 (µm)", 1.0, 200.0, 30.0, 1.0)
    guardrails_on = st.toggle("Guardrails", True)
    gr = _safe_guardrails(d50_um, guardrails_on)
    t0,t1 = gr["layer_thickness_um"]
    layer_um = st.slider("Layer thickness (µm)", float(round(t0)), float(round(t1)),
                         float(round(max(10.0, min(200.0, 4.0*d50_um)))), 1.0)
    target_green = st.slider("Target green density (%TD)", 80, 98, 90, 1)

    st.divider()
    if src_path:
        st.caption(f"Dataset: {Path(src_path).name} · Rows: {len(df_base):,}")

# ---------- Header ----------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · Guardrails hardened · Digital Twin for STL slices")

# ---------- Recommender (Top-5: 3 water + 2 solvent) ----------
@st.cache_data(show_spinner=False)
def _balanced_top5(_material, _d50, _layer, _target, _gr, _models_hash):
    b0,b1 = _gr["binder_saturation_pct"]; v0,v1 = _gr["roller_speed_mm_s"]
    def sweep(binder_type):
        Xs=np.linspace(float(b0), float(b1), 40)
        Ys=np.linspace(float(v0), float(v1), 28)
        g=pd.DataFrame([(b,v,_layer,_d50,_material,binder_type,"metal") for v in Ys for b in Xs],
                       columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um",
                                "d50_um","material","binder_type_rec","material_class"])
        pred=predict_quantiles(models,g)
        g=g.reset_index(drop=True).join(pred[["td_q10","td_q50","td_q90"]].reset_index(drop=True))
        g["score"]=(g["td_q50"]-float(_target)).abs() + 0.10*np.clip(float(_target)-g["td_q10"],0,None)
        return g.sort_values("score")
    gw=sweep("water_based"); gs=sweep("solvent_based")
    pick=pd.concat([gw.head(3), gs.head(2)], ignore_index=True)
    out=pick.rename(columns={
        "binder_saturation_pct":"saturation_pct","roller_speed_mm_s":"roller_speed",
        "layer_thickness_um":"layer_um","td_q10":"pred_q10","td_q50":"pred_q50","td_q90":"pred_q90",
        "binder_type_rec":"binder_type"
    })
    out["id"]=[f"Opt-{i+1}" for i in range(len(out))]
    cols=["id","binder_type","saturation_pct","roller_speed","layer_um","pred_q10","pred_q50","pred_q90","d50_um","material"]
    return out[cols]

models_hash = hash(str(meta)) if meta else 0
recs = _balanced_top5(material, float(d50_um), float(layer_um), float(target_green), gr, models_hash)
st.session_state["top5_recipes_df"] = recs.copy()

# ---------- Tabs (unchanged set) ----------
tabs = st.tabs([
    "Predict (Top-5)",
    "Heatmap",
    "Saturation sensitivity",
    "Qualitative packing",
    "Formulae",
    "Digital Twin",
    "Data health"
])

# TAB 1 — Predict
with tabs[0]:
    st.subheader("Top-5 parameter sets (3 water-based + 2 solvent-based)")
    st.dataframe(recs, use_container_width=True, hide_index=True)

# TAB 2 — Heatmap
with tabs[1]:
    st.subheader("Predicted green %TD (q50) — speed × saturation")
    b0,b1 = gr["binder_saturation_pct"]; v0,v1 = gr["roller_speed_mm_s"]
    Xs=np.linspace(float(b0), float(b1), 70)
    Ys=np.linspace(float(v0), float(v1), 56)
    grid=pd.DataFrame([(b,v,layer_um,d50_um,material,"water_based","metal") for v in Ys for b in Xs],
                      columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um",
                               "d50_um","material","binder_type_rec","material_class"])
    pred=predict_quantiles(models,grid)
    dfZ=pd.DataFrame({"sat":pred["binder_saturation_pct"].astype(float),
                      "spd":pred["roller_speed_mm_s"].astype(float),
                      "z":pred["td_q50"].astype(float)})
    Z=dfZ.pivot_table(index="spd", columns="sat", values="z").sort_index().sort_index(axis=1)
    fig=go.Figure()
    fig.add_trace(go.Heatmap(z=Z.values, x=Z.columns.values, y=Z.index.values,
                             colorscale="Viridis", colorbar=dict(title="%TD", len=0.82)))
    fig.add_hline(y=float(np.median(Ys)), line=dict(color="#aaa", dash="dot"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Roller speed (mm/s)",
                      height=480, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# TAB 3 — Sensitivity
with tabs[2]:
    st.subheader("Saturation sensitivity at representative speed")
    v_mid = float(np.mean(gr["roller_speed_mm_s"]))
    sat_axis=np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 75)
    curve=pd.DataFrame({
        "binder_saturation_pct": sat_axis, "roller_speed_mm_s": v_mid,
        "layer_thickness_um": layer_um, "d50_um": d50_um,
        "material": material, "material_class":"metal", "binder_type_rec":"water_based",
    })
    pred=predict_quantiles(models, curve)
    fig2, ax=plt.subplots(figsize=(7.0,4.2), dpi=170)
    ax.plot(pred["binder_saturation_pct"], pred["td_q50"], lw=2.0, color="#2563eb", label="q50")
    ax.fill_between(pred["binder_saturation_pct"], pred["td_q10"], pred["td_q90"], alpha=0.20, label="q10–q90")
    ax.axhline(target_green, ls="--", lw=1.2, color="#374151", label=f"Target {target_green}%")
    ax.set_xlabel("Binder saturation (%)"); ax.set_ylabel("Predicted green %TD"); ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.18)
    st.pyplot(fig2, clear_figure=True)

# TAB 4 — Qualitative packing (simple)
with tabs[3]:
    st.subheader("Qualitative packing (illustrative)")
    rng=np.random.default_rng(1234)
    W=1.8; N=550
    pts=rng.uniform(0,W,size=(N,2))
    r=0.012*W
    figP, axP=plt.subplots(figsize=(5.0,5.0), dpi=190)
    axP.add_patch(Rectangle((0,0),W,W, fill=False, lw=1.0, ec="#111"))
    for (x,y) in pts: axP.add_patch(Circle((x,y), r, fc="#2F6CF6", ec="#1f2937", lw=0.25))
    axP.set_aspect('equal','box'); axP.set_xlim(0,W); axP.set_ylim(0,W); axP.set_xticks([]); axP.set_yticks([])
    st.pyplot(figP, clear_figure=True)

# TAB 5 — Formulae
with tabs[4]:
    st.subheader("Formulae & Physics Relations")
    st.latex(r"\text{Furnas packing:}\quad \phi_{\max} \approx 1-\prod_i (1-\phi_i)")
    st.latex(r"\text{Washburn penetration:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta} r t}")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")

# ===================== DIGITAL TWIN (STL-first) =====================
# Caches
@st.cache_resource(show_spinner="Loading STL…")
def _load_mesh_bytes(stl_bytes: bytes):
    try:
        m=trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh", process=False)
        if not isinstance(m, trimesh.Trimesh): m=m.dump(concatenate=True)
        return m
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

@st.cache_resource(show_spinner="Creating cube…")
def _cube_mesh(): return trimesh.creation.box(extents=(10.0,10.0,10.0))

@st.cache_data(show_spinner=False)
def _slice_polys_wkb(_mesh_key, z: float) -> Tuple[bytes,...]:
    try:
        mesh=st.session_state.get("_dt_mesh")
        if mesh is None: return tuple()
        zmin,zmax=float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        span=max(zmax-zmin,1e-6); eps=1e-6*span
        for zi in (z, z+eps, z-eps, z+2*eps, z-2*eps):
            sec=mesh.section(plane_origin=(0,0,float(zi)), plane_normal=(0,0,1))
            if sec is None: continue
            planar,_=sec.to_planar()
            rings=getattr(planar,"polygons_full", None) or getattr(planar,"polygons_closed", None)
            polys=[]
            for ring in (rings or []):
                try:
                    p=Polygon(ring)
                    if p.is_valid and p.area>1e-9: polys.append(p.buffer(0))
                except Exception: pass
            if polys: return tuple(p.wkb for p in polys)
        return tuple()
    except Exception: return tuple()

@st.cache_data(show_spinner=False)
def _crop_local(_polys_wkb: Tuple[bytes,...], desired_fov: Optional[float]):
    if not _polys_wkb: return tuple(), (0.0,0.0), 0.0
    polys=[wkb.loads(p) for p in _polys_wkb]; dom=unary_union(polys)
    xmin,ymin,xmax,ymax=dom.bounds; side=max(xmax-xmin,ymax-ymin)
    fov = side if (desired_fov is None or desired_fov<=0) else float(min(desired_fov, side))
    cx,cy=dom.centroid.x, dom.centroid.y; half=fov/2
    x0,y0=cx-half, cy-half; win=box(x0,y0,x0+fov,y0+fov)
    clip=dom.intersection(win)
    if getattr(clip,"is_empty",True): return tuple(), (x0,y0), fov
    geoms=[clip] if isinstance(clip,Polygon) else [g for g in clip.geoms if isinstance(g,Polygon)]
    local=[Polygon(np.c_[np.array(g.exterior.xy[0])-x0, np.array(g.exterior.xy[1])-y0]).wkb for g in geoms]
    return tuple(local), (x0,y0), fov

@st.cache_data(show_spinner=False)
def _hex_pack_target(_key, polys_wkb: Tuple[bytes,...], d50_unit: float,
                     phi_target: float, fov: float, cap: int, jitter: float):
    if not HAVE_SHAPELY or not polys_wkb:
        return np.empty((0,2)), np.empty((0,)), 0.0, 0
    polys=[wkb.loads(p) for p in polys_wkb]; dom=unary_union(polys)
    if getattr(dom,"is_empty",True): return np.empty((0,2)), np.empty((0,)), 0.0, 0
    area_dom=float(dom.area); target=float(np.clip(phi_target,0.40,0.88))
    rng=np.random.default_rng(1234)

    def grid(r):
        s=2.0*r; dy=r*np.sqrt(3.0)
        xs=np.arange(r, fov-r, s); ys=np.arange(r, fov-r, dy)
        pts=[]
        for j,yy in enumerate(ys):
            xoff=0.0 if (j%2==0) else r
            for xx in xs:
                x0=xx+xoff
                if x0>fov-r: continue
                pts.append((x0,yy))
        return np.array(pts,float)

    # Stage 1: fit (eroded)
    def try_fit(k):
        r=max(1e-12, d50_unit/2.0)*k; C=grid(r)
        if C.size==0: return np.empty((0,2)), np.empty((0,)), 0.0
        if jitter>0: C += rng.uniform(-jitter*r, jitter*r, C.shape)
        try: fit=dom.buffer(-r)
        except Exception: fit=dom
        keep=[i for i,(cx,cy) in enumerate(C)
              if fit.contains(Polygon([(cx+r,cy),(cx,cy+r),(cx-r,cy),(cx,cy-r)]))]
        C=C[keep][:cap]; R=np.full(len(C), r, float)
        phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
        return C,R,phi

    bestC,bestR,bestPhi=np.empty((0,2)),np.empty((0,)),0.0
    lo,hi=0.15,2.2
    for _ in range(22):
        mid=(lo+hi)/2
        C,R,phi=try_fit(mid)
        if R.size==0: hi=mid; continue
        bestC,bestR,bestPhi=C,R,phi
        if phi<target: lo=mid
        else: hi=mid
    if bestR.size>0: return bestC,bestR,bestPhi,1

    # Stage 2: center-in
    r2=max(1e-12,d50_unit/2.0)*0.6; C=grid(r2)
    if C.size>0:
        if jitter>0: C += rng.uniform(-jitter*r2, jitter*r2, C.shape)
        keep=[i for i,(cx,cy) in enumerate(C) if dom.contains(Point(cx,cy))]
        C=C[keep][:cap]; R=np.full(len(C), r2, float)
        phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
        if R.size>0: return C,R,phi,2

    # Stage 3: constant radius from φ
    N=max(1,int(cap)); r3=math.sqrt(max(target*area_dom/(N*math.pi), 1e-18)); r3=min(r3,0.25*max(1e-12,fov))
    C=grid(r3); 
    if C.size==0: C=rng.uniform(r3, fov-r3, size=(min(N, max(64,int(0.15*cap))),2))
    if jitter>0: C += rng.uniform(-jitter*r3, jitter*r3, C.shape)
    keep=[i for i,(cx,cy) in enumerate(C) if dom.contains(Point(cx,cy))]
    C=C[keep][:cap]; R=np.full(len(C), r3, float)
    phi=(float(np.sum(np.pi*R**2))/area_dom) if area_dom>0 else 0.0
    return C,R,phi,3

@st.cache_data(show_spinner=False)
def _raster_solids(_key, centers: np.ndarray, radii: np.ndarray, fov: float, px: int):
    if centers.size==0: return np.zeros((px,px), bool)
    y,x=np.mgrid[0:px,0:px]; s=fov/px; xx=x*s; yy=(px-y)*s
    mask=np.zeros((px,px), bool)
    for (cx,cy),r in zip(centers,radii):
        d2=(xx-cx)**2 + (yy-cy)**2
        mask |= (d2 <= r*r)
    return mask

@st.cache_data(show_spinner=False)
def _voids_from_sat(_key, pores: np.ndarray, sat: float, seed: int):
    rng=np.random.default_rng(seed)
    idx=np.flatnonzero(pores.ravel()); k=int((1.0-sat)*len(idx))
    vm=np.zeros_like(pores,bool)
    if k>0 and len(idx)>0:
        choose=rng.choice(idx, size=min(k,len(idx)), replace=False)
        vm.ravel()[choose]=True
    return vm

with tabs[5]:
    st.subheader("Digital Twin — STL slice + packing (STL first, no fallback)")
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Please add 'trimesh' and 'shapely' to requirements.txt")
    else:
        top5 = st.session_state.get("top5_recipes_df", pd.DataFrame())
        rec_id = st.selectbox("Pick one of the Top-5 to visualize", list(top5["id"]) if not top5.empty else ["—"], index=0, disabled=top5.empty)
        if not top5.empty:
            rec = top5[top5["id"]==rec_id].iloc[0]
            sat_pct = float(rec.get("saturation_pct", 80.0))
            binder  = str(rec.get("binder_type", "water_based"))
            d50_r   = float(rec.get("d50_um", d50_um))
            layer_r = float(rec.get("layer_um", layer_um))
        else:
            sat_pct, binder, d50_r, layer_r = 80.0, "water_based", d50_um, layer_um

        c1,c2,c3,c4 = st.columns([2,1,1,1])
        with c1:
            stl_file = st.file_uploader("Upload STL (required to override cube)", type=["stl"])
        with c2:
            # IMPORTANT: cube toggle is *ignored* if an STL is uploaded
            use_cube = st.checkbox("Use 10 mm cube", value=(stl_file is None), disabled=(stl_file is not None))
        with c3:
            units = st.selectbox("Model units", ["mm","m","inch","custom"], index=0)
        with c4:
            mm_per_unit = st.number_input("Custom: mm per unit", 0.001, 10000.0, 1.0, 0.001, disabled=(units!="custom"))

        # um -> model unit
        if units=="mm": um2unit=1e-3
        elif units=="m": um2unit=1e-6
        elif units=="inch": um2unit=(1.0/25.4)*1e-3
        else: um2unit=(1.0/float(mm_per_unit))*1e-3

        mesh=None
        if stl_file is not None:
            mesh=_load_mesh_bytes(stl_file.read())
        elif use_cube:
            mesh=_cube_mesh()

        if mesh is None:
            st.warning("Upload an STL or use the 10 mm cube.")
        else:
            # persist once for slicer cache
            st.session_state["_dt_mesh"]=mesh
            minz,maxz=float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
            thickness=float(layer_r)*um2unit
            n_layers=max(1, int((maxz-minz)/max(thickness,1e-12)))
            layer_idx=st.slider("Layer index", 1, n_layers, 1)
            z=minz+(layer_idx-0.5)*thickness

            show_mesh = st.checkbox("Show 3D preview", value=True)
            if show_mesh:
                figm=go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                    color="lightgray", opacity=0.55, flatshading=True
                )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=320)
                st.plotly_chart(figm, use_container_width=True)

            # Slice (multi-epsilon robust) → auto-FOV to square bbox (model units)
            mkey=hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))
            polys_wkb=_slice_polys_wkb(mkey, z)
            if not polys_wkb:
                st.warning("Empty slice at this layer. Try another layer or check model units.")
            else:
                polys=[wkb.loads(p) for p in polys_wkb]; dom=unary_union(polys)
                xmin,ymin,xmax,ymax=dom.bounds; slice_side=float(max(xmax-xmin,ymax-ymin))
                local_wkb, origin_xy, fov = _crop_local(polys_wkb, desired_fov=None)  # full slice square
                # Pack (guaranteed visible)
                phi_TPD=0.90; phi2D_target=float(np.clip(0.90*phi_TPD,0.40,0.88))
                d50_unit=float(d50_r)*um2unit
                cap = st.slider("Particle cap", 500, 20000, 2200, 100)
                centers,radii,phi2D,stage=_hex_pack_target(
                    (hash(local_wkb), round(d50_unit,9), round(phi2D_target,4), round(fov,6), cap),
                    local_wkb, d50_unit, phi2D_target, fov, cap, jitter=0.12
                )
                # Raster → pores → voids from saturation
                px=st.slider("Render resolution (px)", 300, 1400, 800, 50)
                solids=_raster_solids((hash(centers.tobytes()) if centers.size else 0, px, round(fov,6)),
                                      centers, radii, fov, px)
                pores=~solids
                sat=float(np.clip(sat_pct/100.0,0.01,0.99))
                voids=_voids_from_sat((hash(pores.tobytes()), round(sat,4), layer_idx),
                                      pores, sat, 42+layer_idx+int(sat_pct))

                # Panels
                img_particles=np.ones((px,px,3),float); img_particles[solids]=np.array([0.18,0.38,0.96])
                b_rgb=tuple(int(_binder_hex(binder)[i:i+2],16)/255.0 for i in (1,3,5))
                img_layer=np.ones((px,px,3),float); img_layer[:]=b_rgb
                img_layer[voids]=np.array([1,1,1]); img_layer[solids]=np.array([0.18,0.38,0.96])

                colA,colB=st.columns(2)
                with colA:
                    st.caption("Particles only")
                    figA=go.Figure(go.Image(z=(img_particles*255).astype(np.uint8)))
                    figA.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                    st.plotly_chart(figA, use_container_width=True)
                with colB:
                    st.caption(f"{binder} · Sat {int(sat_pct)}%")
                    figB=go.Figure(go.Image(z=(img_layer*255).astype(np.uint8)))
                    figB.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                    st.plotly_chart(figB, use_container_width=True)

                why={1:"fit (eroded)",2:"center-in",3:"φ-based"}
                st.caption(
                    f"<span class='badge'>Layer {layer_idx}/{n_layers} • FOV={fov:.3f} {units} • d50={d50_unit:.5g} {units} • "
                    f"φ₂D(target)≈{phi2D_target:.2f} • φ₂D(achieved)≈{phi2D:.2f} • particles={len(radii)} • pack={why.get(stage,'—')}</span>",
                    unsafe_allow_html=True
                )

# TAB 7 — Data health
def _data_health(df: pd.DataFrame, material: str, d50_um: float) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d=df.copy()
    if "material" in d.columns:
        d=d[d["material"].astype(str)==str(material)]
    if "d50_um" in d.columns:
        lo,hi=0.8*d50_um, 1.2*d50_um
        d=d[(d["d50_um"]>=lo)&(d["d50_um"]<=hi)]
    cols=[]
    if "green_pct_td" in d.columns and len(d):
        cols.append(("≥90% cases", int((d["green_pct_td"]>=90).sum())))
        cols.append(("n rows (±20% D50)", int(len(d))))
        cols.append(("best %TD", float(d["green_pct_td"].max())))
    return pd.DataFrame(cols, columns=["metric","value"])

with tabs[6]:
    st.subheader("Training coverage & 90%TD evidence near this D50")
    rep=_data_health(df_base, material, d50_um)
    if rep.empty:
        st.info("No rows found for this material / D50 window")
    else:
        c1,c2=st.columns([1,2])
        with c1: st.dataframe(rep, use_container_width=True, hide_index=True)
        with c2:
            if {"d50_um","green_pct_td","material"}.issubset(df_base.columns):
                lo,hi=0.8*d50_um, 1.2*d50_um
                sub=df_base[(df_base["material"].astype(str)==str(material)) & (df_base["d50_um"].between(lo,hi))]
                if not sub.empty:
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=sub["d50_um"], y=sub["green_pct_td"], mode="markers", name="train pts"))
                    fig.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
                    fig.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD", height=360, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True)

# Footer / diagnostics
with st.expander("Diagnostics", expanded=False):
    st.write("Source:", os.path.basename(src_path) if src_path else "—")
    st.write("Models meta:", meta if meta else {"note":"No trained models"})
    st.write("Guardrails:", gr)
