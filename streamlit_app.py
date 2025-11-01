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
# digital_twin_tab_light.py
# Lightweight, fast Digital Twin tab that ONLY consumes existing Top-5 trials from the original app.
# No re-training, no new predictions—keeps your current app's behavior intact.

from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go

# Optional deps
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH: import trimesh
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from shapely import wkb

# -------------------- Fast utilities --------------------
def _binder_hex(binder_type: str) -> str:
    name = (binder_type or "").lower()
    if "water" in name:   return "#F2D06F"
    if "solvent" in name: return "#F2B233"
    if "acryl" in name:   return "#FFD166"
    if "furan" in name:   return "#F5C07A"
    return "#F4B942"

# Vectorized rasterization of circles (for pores)
@st.cache_data(show_spinner=False)
def _raster_solids_mask(_key, centers: np.ndarray, radii: np.ndarray, fov: float, px: int=700) -> np.ndarray:
    if centers.size == 0: return np.zeros((px, px), dtype=bool)
    y, x = np.mgrid[0:px, 0:px]
    s = fov / px
    xx = x * s
    yy = (px - y) * s
    mask = np.zeros((px, px), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        d2 = (xx - cx)**2 + (yy - cy)**2
        mask |= (d2 <= r*r)
    return mask

# Fast hex/triangular packing into a square window, then clip to polygon (approx, fast)
@st.cache_data(show_spinner=False)
def _hex_pack_into_polys(_key, polys_wkb: Tuple[bytes, ...], d50_unit: float, phi2D_target: float,
                         fov: float, max_particles: int, jitter: float) -> Tuple[np.ndarray, np.ndarray, float]:
    if not polys_wkb: 
        return np.empty((0,2)), np.empty((0,)), 0.0
    if not HAVE_SHAPELY:
        # Simple square fill fallback
        n = min(800, max_particles)
        rng = np.random.default_rng(0)
        centers = rng.uniform(0, fov, size=(n,2))
        radii = np.full(n, max(1e-6, d50_unit/2.0))
        return centers, radii, 0.55

    polys = [wkb.loads(p) for p in polys_wkb]
    dom = unary_union(polys)
    if getattr(dom, "is_empty", True):
        return np.empty((0,2)), np.empty((0,)), 0.0

    # Hex spacing for target area fraction (2D): φ_hex ≈ π/(2√3) ≈ 0.9069 for equal circles
    # We scale the nominal radius so that achieved φ2D ≈ phi2D_target (bounded)
    phi = float(np.clip(phi2D_target, 0.40, 0.88))
    r = max(1e-9, d50_unit/2.0)
    # scale to hit phi
    # φ ~ (area_circle / area_cell) with cell area ≈ (√3/2)*s^2; s ≈ 2r_scale
    # derive a scale factor k on radius: k ≈ sqrt(phi / 0.9069)
    k = float(np.sqrt(phi / 0.9069))
    r_eff = r * k

    # Hex grid extents
    xmin, ymin, xmax, ymax = 0.0, 0.0, fov, fov
    s = 2.0 * r_eff
    dy = r_eff * np.sqrt(3.0)
    # Build grid
    xs = np.arange(xmin + r_eff, xmax, s)
    ys = np.arange(ymin + r_eff, ymax, dy)
    pts = []
    for j, yy in enumerate(ys):
        xoff = 0.0 if (j % 2 == 0) else r_eff
        for xx in xs:
            x0 = xx + xoff
            if x0 > xmax - r_eff: 
                continue
            pts.append((x0, yy))
    if not pts:
        return np.empty((0,2)), np.empty((0,)), 0.0
    centers = np.array(pts, float)

    # Clip to polygon & cap
    # Jitter for realism
    if jitter > 0:
        rng = np.random.default_rng(1234)
        centers += rng.uniform(-jitter*r_eff, jitter*r_eff, size=centers.shape)

    # Keep points whose circle stays inside dom (fast approximation via erode)
    try:
        dom_fit = dom.buffer(-r_eff)
        if getattr(dom_fit, "is_empty", True):
            return np.empty((0,2)), np.empty((0,)), 0.0
    except Exception:
        dom_fit = dom

    keep = []
    for idx, (cx, cy) in enumerate(centers):
        if dom_fit.contains(Polygon([(cx+r_eff,cy),(cx,cy+r_eff),(cx-r_eff,cy),(cx,cy-r_eff)])):
            keep.append(idx)
    if not keep:
        return np.empty((0,2)), np.empty((0,)), 0.0
    centers = centers[keep]
    if len(centers) > max_particles:
        centers = centers[:max_particles]

    radii = np.full(len(centers), r_eff, float)
    # Estimate φ2D achieved quickly: solids area over dom area
    area_solids = float(np.sum(np.pi * radii**2))
    phi2D = area_solids / dom.area if dom.area > 0 else 0.0
    return centers, radii, float(np.clip(phi2D, 0.0, 1.0))

# Robust slicer with epsilon offsets (avoid vertex/edge degeneracy)
@st.cache_data(show_spinner=False)
def _slice_polys_wkb(_mesh_key, z: float) -> Tuple[bytes, ...]:
    try:
        mesh = st.session_state.get("_dtw_loaded_mesh")
        if mesh is None: return tuple()
        zmin, zmax = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
        span = max(zmax - zmin, 1e-6)
        eps = 1e-6 * span
        for zi in (z, z+eps, z-eps, z+2*eps, z-2*eps):
            sec = mesh.section(plane_origin=(0.0,0.0,float(zi)), plane_normal=(0.0,0.0,1.0))
            if sec is None: 
                continue
            planar, _ = sec.to_planar()
            rings = getattr(planar, "polygons_full", None) or getattr(planar, "polygons_closed", None)
            if not rings: 
                continue
            polys = []
            for ring in rings:
                try:
                    p = Polygon(ring)
                    if p.is_valid and p.area > 1e-9:
                        polys.append(p.buffer(0))
                except Exception:
                    pass
            if polys:
                return tuple(p.wkb for p in polys)
        return tuple()
    except Exception:
        return tuple()

@st.cache_data(show_spinner=False)
def _crop_and_local(_polys_wkb: Tuple[bytes, ...], fov: float):
    if not _polys_wkb: return tuple(), (0.0, 0.0), 0.0
    polys = [wkb.loads(p) for p in _polys_wkb]
    dom = unary_union(polys)
    cx, cy = dom.centroid.x, dom.centroid.y
    half = fov/2.0
    xmin, ymin = cx - half, cy - half
    win = box(xmin, ymin, xmin+fov, ymin+fov)
    clip = dom.intersection(win)
    if getattr(clip, "is_empty", True): 
        return tuple(), (xmin, ymin), fov
    geoms = [clip] if isinstance(clip, Polygon) else [g for g in clip.geoms if isinstance(g, Polygon)]
    if not geoms: 
        return tuple(), (xmin, ymin), fov
    local = []
    for g in geoms:
        x, y = g.exterior.xy
        local.append(Polygon(np.c_[np.array(x)-xmin, np.array(y)-ymin]).wkb)
    return tuple(local), (xmin, ymin), fov

# -------------------- Public tab API --------------------
def render_digital_twin_tab():
    st.markdown("### Digital Twin (lightweight) — STL slice + particle packing")
    if not HAVE_TRIMESH or not HAVE_SHAPELY:
        st.error("Please add 'trimesh' and 'shapely' to requirements.txt")
        return

    # Require precomputed trials from the original app
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Open your original Predict tab first to generate the 5 trials.")
        return
    top5 = top5.reset_index(drop=True)

    # Controls: choose which of your existing trials to visualize (no predictions here)
    left, right = st.columns([1.6, 1])
    with left:
        trial_id = st.selectbox("Select trial to visualize (from Top-5)", list(top5["id"]), index=0)
        row = top5[top5["id"]==trial_id].iloc[0]
        binder = str(row.get("binder_type", "water_based"))
        sat_pct = float(row.get("saturation_pct", 80.0))
        layer_um = float(row.get("layer_um", 4.0*float(row.get("d50_um", 35.0))))
        d50_um = float(row.get("d50_um", 35.0))
        st.caption(f"Using your trial {trial_id}: {binder}, Sat={int(sat_pct)}%, Layer={int(layer_um)} µm, D50≈{int(d50_um)} µm")
    with right:
        stl_units = st.selectbox("STL units", ["mm","m"], index=0)
        um2unit = 1e-3 if stl_units=="mm" else 1e-6
        px = st.slider("Render resolution (px)", 400, 1200, 700, 50)
        jitter = st.slider("Packing jitter", 0.0, 0.35, 0.12, 0.01)
        max_particles = st.slider("Particle cap per layer", 200, 6000, 1800, 100)
        phi_TPD = st.slider("Target post-sinter %TD", 85, 95, 90, 1) / 100.0
        phi2D_target = float(np.clip(0.90*phi_TPD, 0.40, 0.88))

    # STL upload or sample cube
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2:
        use_cube = st.checkbox("Use 10 mm cube", value=(stl_file is None))
    with c3:
        precompute = st.checkbox("Precompute all layers (cache)", value=False)

    mesh = None
    if use_cube:
        mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
    elif stl_file is not None:
        try:
            mesh = trimesh.load(io.BytesIO(stl_file.read()), file_type="stl", force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump(concatenate=True)
        except Exception as e:
            st.error(f"Could not read STL: {e}")
            return

    if mesh is None:
        st.info("Upload an STL or select the sample cube.")
        return

    # Stash once for slicer cache
    st.session_state["_dtw_loaded_mesh"] = mesh
    d50_unit = d50_um * um2unit
    thickness = layer_um * um2unit

    # Layer selection
    zmin, zmax = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    n_layers = max(1, int((zmax - zmin) / max(thickness, 1e-12)))
    st.markdown(f"**Layers:** {n_layers} · Z span: {zmax - zmin:.3f} {stl_units}")
    layer_idx = st.slider("Layer index", 1, n_layers, 1)
    z = zmin + (layer_idx - 0.5) * thickness

    # 3D mesh preview
    with st.expander("3D preview", expanded=False):
        figm = go.Figure(data=[go.Mesh3d(
            x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
            i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
            color="lightgray", opacity=0.55, flatshading=True, name="Part"
        )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=320)
        st.plotly_chart(figm, use_container_width=True)

    # --- Slice → Crop (FOV) → Pack → Render
    mesh_key = hash((mesh.vertices.tobytes(), mesh.faces.tobytes()))
    polys_wkb = _slice_polys_wkb(mesh_key, z)
    if not polys_wkb:
        st.warning("No cross-section at this layer (empty slice). Try another layer.")
        return

    # Choose FOV as square bounding box side for full slice view
    polys = [wkb.loads(p) for p in polys_wkb]
    dom = unary_union(polys)
    xmin, ymin, xmax, ymax = dom.bounds
    fov = max(xmax - xmin, ymax - ymin)
    # center-crop square window and transform to local coords
    local_wkb, origin_xy, fov = _crop_and_local(polys_wkb, fov)

    # Hex-pack (fast), clip, and render
    pack_key = (hash(local_wkb), round(d50_unit, 9), round(phi2D_target, 4), fov, max_particles, round(jitter,3))
    centers, radii, phi2D = _hex_pack_into_polys(pack_key, local_wkb, d50_unit, phi2D_target, fov, max_particles, jitter)

    # Raster solids → pores → show binder/void overlay based on your chosen trial’s saturation
    solids_key = (hash(centers.tobytes()) if centers.size else 0, round(fov,6), px)
    solids_mask = _raster_solids_mask(solids_key, centers, radii, fov, px)
    pores = ~solids_mask
    sat = float(np.clip(sat_pct/100.0, 0.01, 0.99))

    # Approximate void selection as (1 - saturation) fraction of pore pixels (random but stable)
    pore_idx = np.flatnonzero(pores.ravel())
    rng = np.random.default_rng(42 + layer_idx + int(sat_pct))
    k = int((1.0 - sat) * len(pore_idx))
    if k > 0:
        choose = rng.choice(pore_idx, size=k, replace=False)
        voids = np.zeros_like(pores, bool)
        voids.ravel()[choose] = True
    else:
        voids = np.zeros_like(pores, bool)

    # Plot two panels with Plotly images (fast)
    # panel 1: particles only
    img1 = np.ones((px, px, 3), dtype=float)
    # draw particles as dark blue where solids_mask True
    img1[solids_mask] = np.array([0.18, 0.38, 0.96])
    # panel 2: binder background + particles + voids
    img2 = np.ones((px, px, 3), dtype=float)
    # binder color
    bhex = _binder_hex(binder)
    b_rgb = tuple(int(bhex[i:i+2], 16)/255.0 for i in (1,3,5))
    img2[:] = b_rgb
    # voids = white
    img2[voids] = np.array([1.0, 1.0, 1.0])
    # particles overlaid (dark edge)
    img2[solids_mask] = np.array([0.18, 0.38, 0.96])

    colA, colB = st.columns(2)
    with colA:
        st.markdown("Particles only")
        figA = go.Figure(go.Image(z=(img1*255).astype(np.uint8)))
        figA.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
        st.plotly_chart(figA, use_container_width=True)
    with colB:
        st.markdown(f"{binder} · Sat {int(sat_pct)}%")
        figB = go.Figure(go.Image(z=(img2*255).astype(np.uint8)))
        figB.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
        st.plotly_chart(figB, use_container_width=True)

    st.caption(f"Layer {layer_idx}/{n_layers} · FOV={fov:.3f} {stl_units} · φ₂D≈{phi2D:.2f} · particles={len(radii)}")

    # Optional: precompute/cache all layers (warn if heavy)
    if precompute:
        with st.spinner("Precomputing cached slices…"):
            for li in range(1, n_layers+1):
                zi = zmin + (li - 0.5) * thickness
                _ = _slice_polys_wkb(mesh_key, zi)  # cached
        st.success("Layer slices cached.")
