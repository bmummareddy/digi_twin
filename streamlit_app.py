# streamlit_app.py — BJAM Binder-Jet AM Recommender
# - Ivory light theme with forced dark text (no raw CSS text)
# - Heatmap, sensitivity, Pareto, formulae
# - Packing tab:
#     • Square side (×D50) slider + densify toggle
#     • Side-by-side Packing slice + Printed-layer (pixelated), same size
#     • Small captions (no large titles)

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ---- Project utilities (already in repo)
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
    suggest_binder_family,
)

# ======================= Page & Theme =======================
st.set_page_config(
    page_title="BJAM Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global CSS (inject safely; prevents raw CSS text)
CSS = """
<style>
  /* Font & base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  :root { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
  .stApp { background: #FFFDF7 !important; }
  html, body, [class*="css"] { font-family: var(--font) !important; }

  /* Force dark text on light bg */
  html, body, h1, h2, h3, h4, h5, h6, p, span, label, div, li, code, pre,
  .stMarkdown, .stText, .stCaption, .stAlert, .stMetric {
    color: #111827 !important;
  }

  /* Inputs */
  .stTextInput input, .stNumberInput input { color:#111827 !important; }
  .stSelectbox [data-baseweb="select"] * { color:#111827 !important; }
  .stRadio > div label { color:#111827 !important; }
  .stSlider { color:#111827 !important; }

  /* Layout tweaks */
  .block-container { max-width: 1200px; }
  .stTabs [data-baseweb="tab"] { font-weight:600; color:#111827 !important; }
  .stDataFrame { background: rgba(255,255,255,.65); }

  /* KPI cards */
  .kpi {
    background:#fff; border-radius:12px; padding:16px 18px;
    border:1px solid rgba(0,0,0,0.06); box-shadow:0 1px 2px rgba(0,0,0,0.03);
  }
  .kpi .kpi-label { color:#1f2937; font-weight:600; font-size:1.0rem; opacity:.9; white-space:nowrap; }
  .kpi .kpi-value-line { display:flex; align-items:baseline; gap:.35rem; white-space:nowrap; }
  .kpi .kpi-value { color:#111827; font-weight:800; font-size:2.2rem; line-height:1.05;
                    font-variant-numeric: tabular-nums; letter-spacing:.2px; }
  .kpi .kpi-unit  { color:#111827; font-weight:700; font-size:1.1rem; opacity:.85; }
  .kpi .kpi-sub   { color:#374151; opacity:.65; font-size:.9rem; margin-top:.25rem; white-space:nowrap; }

  /* Footer */
  .footer { text-align:center; margin: 28px 0 6px; color:#1f2937; opacity:.9; font-size:0.95rem; }
  .footer a { color:#0d6efd; text-decoration:none; }
  .footer a:hover { text-decoration:underline; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================= Data & Models =======================
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# ======================= Sidebar =======================
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data source: {Path(src).name} · rows={len(df_base):,}")
        st.download_button(
            "Download source dataset (CSV)",
            data=df_base.to_csv(index=False).encode("utf-8"),
            file_name=Path(src).name,
            mime="text/csv",
        )
    else:
        st.warning("No dataset found. App will use physics priors only (few-shot disabled).")

    st.divider()
    guardrails_on = st.toggle(
        "Guardrails", True,
        help="ON: stable windows (binder 60–110%, speed ≈1.2–3.5 mm/s, layer ≈3–5×D50). OFF: wider exploration."
    )
    target_green = st.slider("Target green %TD", 80, 98, 90, 1)
    st.caption("Recommendations prefer **q10 ≥ target** for conservatism.")

# ======================= Header =======================
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot • Custom materials supported • Guardrails toggle")

with st.expander("Preview source data", expanded=False):
    if len(df_base): st.dataframe(df_base.head(25), use_container_width=True)
    else: st.info("No rows to preview.")

st.divider()

# ======================= Inputs =======================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Inputs")

    mode = st.radio("Material source", ["From dataset", "Custom"], horizontal=True)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []

    if mode == "From dataset" and materials:
        material = st.selectbox("Material (from dataset)", options=materials, index=0)
        d50_default = 30.0
        if "d50_um" in df_base.columns:
            sel = df_base["material"].astype(str) == material
            if sel.any():
                d50_default = float(df_base.loc[sel, "d50_um"].dropna().median() or 30.0)
        material_class = (
            df_base.loc[df_base["material"].astype(str) == material, "material_class"]
            .dropna().astype(str).iloc[0]
            if {"material","material_class"}.issubset(df_base.columns) and
               (df_base["material"].astype(str) == material).any()
            else "metal"
        )
    else:
        material = st.text_input("Material (custom)", value="Al2O3")
        material_class = st.selectbox("Material class", ["metal","oxide","carbide","other"], index=1)
        d50_default = 30.0

    d50_um = st.number_input("D50 (µm)", 1.0, 150.0, float(d50_default), 1.0,
                             help="Layer guidance typically ≈ 3–5×D50.")
    pri = physics_priors(d50_um, binder_type_guess=None)
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    t_lo, t_hi = gr["layer_thickness_um"]
    layer_um = st.slider("Layer thickness (µm)", float(round(t_lo)), float(round(t_hi)),
                         float(round(pri["layer_thickness_um"])), 1.0)

    auto_binder = suggest_binder_family(material, material_class)
    binder_choice = st.selectbox(
        "Binder family",
        [f"auto ({auto_binder})", "solvent_based", "water_based"],
        help="Auto uses material class: water for oxide/carbide; solvent otherwise."
    )
    binder_family = auto_binder if binder_choice.startswith("auto") else binder_choice

with right:
    st.subheader("Priors (for intuition)")
    k1, k2, k3 = st.columns(3)

    def kpi_num(col, label: str, value: str, unit: str = "", sub: str = ""):
        col.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value-line">
                <div class="kpi-value">{value}</div>
                <div class="kpi-unit">{unit}</div>
              </div>
              <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi_num(k1, "Prior binder", f"{pri['binder_saturation_pct']:.0f}", "%")
    kpi_num(k2, "Prior speed",  f"{pri['roller_speed_mm_s']:.2f}", "mm/s")
    kpi_num(k3, "Layer/D50",    f"{layer_um/d50_um:.2f}", "×")

st.divider()

# ======================= Recommendations =======================
st.subheader("Recommended parameters")

colL, colR = st.columns([1, 1])
top_k = colL.slider("How many to show", 3, 8, 5, 1)
run_recs = colR.button("Recommend", type="primary", use_container_width=True)

if run_recs:
    recs = copilot(
        material=material, d50_um=float(d50_um), df_source=df_base, models=models,
        guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k)
    )
    recs["binder_type"] = binder_family
    pretty = recs.rename(columns={
        "binder_type": "Binder",
        "binder_%": "Binder sat (%)",
        "speed_mm_s": "Speed (mm/s)",
        "layer_um": "Layer (µm)",
        "predicted_%TD_q10": "q10 %TD",
        "predicted_%TD_q50": "q50 %TD",
        "predicted_%TD_q90": "q90 %TD",
        "meets_target_q10": f"Meets target (q10 ≥ {target_green}%)",
    })
    st.dataframe(
        pretty,
        use_container_width=True,
        column_config={
            "Binder sat (%)": st.column_config.NumberColumn(format="%.1f"),
            "Speed (mm/s)":   st.column_config.NumberColumn(format="%.2f"),
            "Layer (µm)":     st.column_config.NumberColumn(format="%.0f"),
            "q10 %TD":        st.column_config.NumberColumn(format="%.2f"),
            "q50 %TD":        st.column_config.NumberColumn(format="%.2f"),
            "q90 %TD":        st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button(
        "Download recommendations (CSV)",
        data=pretty.to_csv(index=False).encode("utf-8"),
        file_name="bjam_recommendations.csv",
        type="secondary",
        use_container_width=True,
    )
else:
    st.info("Click **Recommend** to generate top-k parameter sets aimed at your target green %TD.")

st.divider()

# ======================= Visuals =======================
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing (2D slice)",
    "Pareto frontier",
    "Formulae",
])

def _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=55, ny=45):
    sats = np.linspace(float(b_lo), float(b_hi), nx)
    spds = np.linspace(float(s_lo), float(s_hi), ny)
    grid = pd.DataFrame([(b,v,layer_um,d50_um,material) for b in sats for v in spds],
                        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    grid["material_class"] = material_class
    grid["binder_type_rec"] = binder_family
    return grid, sats, spds

# ---- Heatmap
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD")
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    grid, Xs, Ys = _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family)
    scored = predict_quantiles(models, grid)
    Z = scored.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=list(Xs), y=list(Ys), z=Z, colorscale="Viridis", colorbar=dict(title="%TD")))
    fig.add_trace(go.Contour(x=list(Xs), y=list(Ys), z=Z,
                             contours=dict(start=90, end=90, size=1, coloring="none"),
                             line=dict(width=3), showscale=False, name="90% TD"))
    fig.add_trace(go.Scatter(x=[80], y=[1.6], mode="markers+text",
                             marker=dict(size=10, symbol="x", color="#111827"),
                             text=["prior"], textposition="top center"))
    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        title=f"Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · Material={material} ({material_class}) · Source={Path(src).name if src else '—'}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Sensitivity
with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90)")
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    sats = np.linspace(float(b_lo), float(b_hi), 61)
    curve_df = pd.DataFrame({
        "binder_saturation_pct": sats,
        "roller_speed_mm_s": 1.6,
        "layer_thickness_um": float(layer_um),
        "d50_um": float(d50_um),
        "material": material,
        "material_class": material_class,
        "binder_type_rec": binder_family,
    })
    cs = predict_quantiles(models, curve_df)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.1), dpi=175)
    ax2.plot(cs["binder_saturation_pct"], cs["td_q50"], color="#1f77b4", linewidth=2.0, label="q50")
    ax2.fill_between(cs["binder_saturation_pct"], cs["td_q10"], cs["td_q90"], alpha=0.18, label="q10–q90")
    ax2.axhline(target_green, linestyle="--", linewidth=1.2, color="#374151", label=f"Target {target_green}%")
    ax2.set_xlabel("Binder saturation (%)"); ax2.set_ylabel("Predicted green %TD")
    ax2.grid(True, axis="y", alpha=0.18); ax2.legend(frameon=False)
    st.pyplot(fig2, clear_figure=True)

# ---- Packing (side-by-side, small, pixelated)
with tabs[2]:
    st.subheader("Packing — 2D slice")
    st.caption("Square slice sized in ×D50. Toggle densification; preview a printed layer with binder fill.")

    # Controls
    c1, c2, c3, c4 = st.columns(4)
    side_mult = c1.slider("Square side (× D50)", 10, 60, 20, 2,
                          help="Side length of the slice in multiples of D50.")
    cv_pct   = c2.slider("Polydispersity (CV %)", 0, 60, 20, 5,
                         help="Coefficient of variation of particle diameter (lognormal).")
    densify  = c3.toggle("Densify packing", False,
                         help="OFF: baseline packing; ON: more tries ⇒ tighter fill.")
    seed     = c4.number_input("Seed", 0, 9999, 0, 1)

    # Derived
    W_mult = int(side_mult)
    base_ref, dense_ref = 260, 520     # ref counts at 20×D50
    baseline_particles   = int(base_ref  * (W_mult/20)**2)
    densified_particles  = int(dense_ref * (W_mult/20)**2)
    Npx = int(21 * W_mult)             # keep ~constant pixels per D50

    # RSA helper
    def rsa_pack(max_particles: int, D50_um: float, cv_pct: float, W_mult: int, seed: int, densify: bool):
        rng = np.random.default_rng(seed)
        cv = cv_pct/100.0
        if cv <= 0:
            diam_um = np.full(max_particles, float(D50_um))
        else:
            sigma = float(np.sqrt(np.log(1.0 + cv**2)))
            diam_um = float(D50_um) * rng.lognormal(mean=0.0, sigma=sigma, size=max_particles)
            diam_um = np.clip(diam_um, 0.4*float(D50_um), 1.8*float(D50_um))
        diam_u = diam_um / float(D50_um)
        radii = 0.5 * np.sort(diam_u)[::-1]
        W = float(W_mult)

        pts = []
        MAX_ATTEMPTS = 40000; attempts = 0

        def can_place(x,y,r):
            if x-r<0 or x+r>W or y-r<0 or y+r>W: return False
            for (px,py,pr) in pts:
                dx = x - px; dy = y - py
                if dx*dx + dy*dy < (r+pr)**2: return False
            return True

        for r in radii:
            tries = 240 if not densify else 280
            for _ in range(tries):
                x = rng.uniform(r, W-r); y = rng.uniform(r, W-r)
                if can_place(x,y,r):
                    pts.append((x,y,r)); break
            attempts += 1
            if attempts > MAX_ATTEMPTS: break

        rs = np.array([r for (_,_,r) in pts])
        phi = (np.pi*np.sum(rs**2))/(W*W) if W>0 else 0.0
        return pts, phi, W

    # Build packing
    num_p = densified_particles if densify else baseline_particles
    pts, phi_area, W = rsa_pack(num_p, float(d50_um), float(cv_pct), W_mult, int(seed), densify)

    # Small controls row for printed-layer preview (kept above both plots)
    ctrlL, ctrlR = st.columns([3, 1])
    binder_sat_pct = ctrlL.slider("Binder saturation (%)", 50, 100, 80, 1)
    t_over_D50 = float(layer_um) / float(d50_um)
    ctrlR.metric("Layer/D50 used", f"{t_over_D50:.2f}×")

    # Rasterize solids & binder mask (for right plot)
    xx = np.linspace(0, W, Npx); yy = np.linspace(0, W, Npx)
    X, Y = np.meshgrid(xx, yy)
    solid = np.zeros((Npx, Npx), dtype=bool)
    for (x, y, r) in pts:
        solid |= (X - x) ** 2 + (Y - y) ** 2 <= r ** 2
    void = ~solid

    rng_local = np.random.default_rng(int(seed) + 12345)
    idx_void = np.flatnonzero(void.ravel())
    k = int(len(idx_void) * (binder_sat_pct / 100.0))
    chosen = rng_local.choice(idx_void, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    binder_mask = np.zeros_like(void, dtype=bool)
    if k > 0: binder_mask.ravel()[chosen] = True

    # === Side-by-side plots (identical sizes) ===
    FIGSIZE = (1.6, 1.6)  # smaller, consistent squares
    DPI = 300

    colA, colB = st.columns(2)

    # LEFT: packing slice (circles)
    with colA:
        figP, axP = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axP.set_aspect('equal', 'box')
        axP.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        for (x, y, r) in pts:
            axP.add_patch(plt.Circle((x, y), r, facecolor='#3b82f6', edgecolor='#111827',
                                     linewidth=0.40, alpha=0.92))
        axP.set_xlim(0, W); axP.set_ylim(0, W)
        axP.set_xticks([]); axP.set_yticks([])
        plt.tight_layout(pad=0.1)
        st.pyplot(figP, clear_figure=True)
        side_um = W * float(d50_um)
        st.caption(f"Packing {'(densified)' if densify else '(baseline)'} • φ≈{phi_area*100:.1f}% • side≈{side_um:.0f} µm")

    # RIGHT: printed layer (pixelated)
    with colB:
        figL, axL = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axL.set_aspect('equal', 'box')
        img = np.zeros_like(solid, dtype=float)
        img[solid] = 0.60          # solids
        img[binder_mask] = 0.95    # binder fill
        axL.imshow(
            img, extent=[0, W, 0, W], origin='lower', vmin=0, vmax=1,
            interpolation='nearest'  # pixelate
        )
        axL.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        axL.set_xlim(0, W); axL.set_ylim(0, W)
        axL.set_xticks([]); axL.set_yticks([])
        plt.tight_layout(pad=0.1)
        st.pyplot(figL, clear_figure=True)
        st.caption(f"Printed layer • binder≈{binder_sat_pct}% • t/D50={t_over_D50:.2f}")

# ---- Pareto
with tabs[3]:
    st.subheader("Pareto frontier — Binder vs green %TD (fixed layer & D50)")
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    grid_p, _, _ = _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=80, ny=1)
    sc_p = predict_quantiles(models, grid_p)[["binder_saturation_pct","td_q50"]].dropna().sort_values("binder_saturation_pct")

    pts_line = sc_p.values; idx=[]; best=-1
    for i,(b,td) in enumerate(pts_line[::-1]):
        if td>best: idx.append(len(pts_line)-1-i); best=td
    idx = sorted(idx)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=sc_p["binder_saturation_pct"], y=sc_p["td_q50"], mode="markers",
                              marker=dict(size=6, color="#1f77b4"), name="Candidates"))
    fig4.add_trace(go.Scatter(x=sc_p.iloc[idx]["binder_saturation_pct"], y=sc_p.iloc[idx]["td_q50"],
                              mode="lines+markers", marker=dict(size=7, color="#111827"),
                              line=dict(width=2, color="#111827"), name="Pareto frontier"))
    fig4.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD (q50)",
                       height=460, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig4, use_container_width=True)

# ---- Formulae
with tabs[4]:
    st.subheader("Formulae (symbols)")
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    st.caption("Few-shot model refines these physics-guided priors using your dataset.")

# ======================= Diagnostics & Footer =======================
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})

st.markdown(f"""
<div class="footer">
<strong>© {datetime.now().year} Bhargavi Mummareddy</strong> • Contact:
<a href="mailto:mummareddybhargavi@gmail.com">mummareddybhargavi@gmail.com</a><br/>
</div>
""", unsafe_allow_html=True)
# -*- coding: utf-8 -*-
# Digital-twin tab: STL → layer slice → fast 2D qualitative packing
import io, math, importlib.util
from typing import List, Tuple, Dict, Optional

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from PIL import Image, ImageDraw

# Optional deps
_HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
_HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
_HAVE_SCIPY   = importlib.util.find_spec("scipy")   is not None
if _HAV E_TRIMESH:
    import trimesh  # type: ignore
if _HAVE_SHAPELY:
    from shapely.geometry import Polygon, Point, box  # type: ignore
    from shapely.ops import unary_union  # type: ignore
if _HAVE_SCIPY:
    from scipy import ndimage as ndi  # type: ignore

# ------- minimal, private helpers (namespaced to avoid collisions) -------
_COLOR_PARTICLE = "#2F6CF6"
_COLOR_EDGE     = "#1f2937"
_COLOR_BORDER   = "#111111"
_COLOR_VOID     = "#FFFFFF"
_COLOR_BINDERS  = {"water":"#F2D06F", "solvent":"#F2B233", "other":"#F4B942"}

def _binder_hex(name:str)->str:
    k = (name or "").lower()
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
    if not _HAVE_SHAPELY:
        return []
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
    if getattr(res, "is_empty", True): return [], (xmin, ymin)
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
        for _ in range(480):  # tight loop cap → faster
            trials += 1
            if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles:
                break
            x = rng.uniform(fx0, fx1); y = rng.uniform(fy0, fy1)
            if not fit.contains(Point(x,y)) or not _ok(x,y,r):
                continue
            idx = len(placed_xy)
            placed_xy.append((x,y)); placed_r.append(r)
            gx, gy = int(x//cell), int(y//cell)
            grid.setdefault((gx,gy), []).append(idx)
            area_circ += math.pi*r*r
        if trials>max_trials or area_circ>=target_area or len(placed_xy)>=max_particles:
            break

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

# ============================ PUBLIC ENTRY ====================================
def render(material:str, d50_um:float, layer_um:float):
    st.subheader("Digital Twin (Beta) — STL slice + qualitative packing")

    if not (_HAVE_TRIMESH and _HAVE_SHAPELY):
        st.error("Install 'trimesh' and 'shapely' to use this tab (see requirements.txt).")
        return

    # pull Top-5 from the main app (unchanged)
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run your app’s Predict tab first to generate Top-5 recipes.")
        return
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
        fast = st.toggle("Fast mode (coarser packing)", value=True, help="Speeds up packing by limiting attempts")

    c1,c2,c3 = st.columns([2,1,1])
    with c1: stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2: use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3: show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    mesh=None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        mesh = _dt_load_mesh(stl_file.read())

    # use selected trial’s D50/layer if present; else fallback to sidebar values
    rec = top5[top5["id"]==rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    diam_um = _dt_psd_um(7000 if fast else 10000, d50_r, seed=9991)
    diam_units = diam_um * um2unit

    # choose layer
    if mesh is not None:
        minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        thickness = layer_r * (1e-3 if stl_units=="mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz-minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness
    else:
        n_layers, layer_idx, z = 1, 1, 0.0

    if mesh is not None and show_mesh:
        import plotly.graph_objects as go
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=360)
        st.plotly_chart(figm, use_container_width=True)

    # build slice polygon(s) and local packing domain
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
        # fallback: square window if no STL
        half = (1.8)/2.0
        polys_local=[box(0,0,2*half,2*half)]
        render_fov = 2*half

    # pack
    centers, radii, phi2D = _dt_pack(
        polys_local, diam_units, phi2D_target,
        max_particles=int(cap),
        max_trials=200_000 if fast else 480_000,
        seed=20_000 + layer_idx
    )

    # render two panels
    def _panel_particles(ax):
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor="white", edgecolor=_COLOR_BORDER, linewidth=1.2))
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=_COLOR_PARTICLE, edgecolor=_COLOR_EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov)
        ax.set_xticks([]); ax.set_yticks([])

    def _panel_binder(ax, sat_pct, binder):
        px=900; pores=~_dt_bitmap_mask(centers, radii, render_fov, px)
        vm=_dt_voids(pores, sat=float(sat_pct)/100.0, seed=1234)
        ax.add_patch(Rectangle((0,0), render_fov, render_fov, facecolor=_binder_hex(binder), edgecolor=_COLOR_BORDER, linewidth=1.2))
        ys,xs=np.where(vm)
        if len(xs):
            xm=xs*(render_fov/vm.shape[1]); ym=(vm.shape[0]-ys)*(render_fov/vm.shape[0])
            ax.scatter(xm, ym, s=0.32, c=_COLOR_VOID, alpha=0.96, linewidth=0)
        for (x,y), r in zip(centers, radii):
            ax.add_patch(Circle((x,y), r, facecolor=_COLOR_PARTICLE, edgecolor=_COLOR_EDGE, linewidth=0.25))
        ax.set_aspect('equal','box'); ax.set_xlim(0,render_fov); ax.set_ylim(0,render_fov); ax.set_xticks([]); ax.set_yticks([])

    # single trial (respects main app’s Top-5)
    rec = top5[top5["id"]==rec_id].iloc[0]
    sat_pct = float(rec.get("saturation_pct", 85.0))
    binder  = str(rec.get("binder_type","water_based"))

    cA,cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.1,5.1), dpi=185)
        _panel_particles(axA)
        _dt_scale(axA, render_fov)
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)
    with cB:
        figB, axB = plt.subplots(figsize=(5.1,5.1), dpi=185)
        _panel_binder(axB, sat_pct, binder)
        _dt_scale(axB, render_fov)
        axB.set_title(f"{binder} · Sat {int(sat_pct)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · φ₂D(achieved)≈{min(phi2D,1.0):.2f} · Porosity₂D≈{max(0.0,1.0-phi2D):.2f}")

    # compare multiple trials visually (same slice/particles; binder & sat vary)
    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"]==rid].iloc[0]
            sat = float(row.get("saturation_pct", 85.0)); bndr=str(row.get("binder_type","water_based"))
            figC, axC = plt.subplots(figsize=(4.9,4.9), dpi=185)
            _panel_binder(axC, sat, bndr)
            _dt_scale(axC, render_fov)
            axC.set_title(f'{row["id"]}: {bndr} · Sat {int(sat)}%', fontsize=9)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
