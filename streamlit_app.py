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
