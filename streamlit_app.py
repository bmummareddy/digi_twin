# streamlit_app.py — BJAM Binder-Jet AM Recommender (+ Digital Twin)
# Tightened guardrails applied locally (no change to shared.py)
# Improved heatmap visualization

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

# --- Project utilities (unchanged signatures in shared.py) ---
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
    suggest_binder_family,
)

# --- Digital Twin as separate module ---
import digital_twin

# ======================= Page & Theme =======================
st.set_page_config(
    page_title="BJAM Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  :root { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
  .stApp { background: #FFFDF7 !important; }
  html, body, [class*="css"] { font-family: var(--font) !important; }
  html, body, h1, h2, h3, h4, h5, h6, p, span, label, div, li, code, pre,
  .stMarkdown, .stText, .stCaption, .stAlert, .stMetric { color: #111827 !important; }
  .stTextInput input, .stNumberInput input { color:#111827 !important; }
  .stSelectbox [data-baseweb="select"] * { color:#111827 !important; }
  .stRadio > div label { color:#111827 !important; }
  .stSlider { color:#111827 !important; }
  .block-container { max-width: 1200px; }
  .stTabs [data-baseweb="tab"] { font-weight:600; color:#111827 !important; }
  .stDataFrame { background: rgba(255,255,255,.65); }
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
        help="ON: use stable windows. We further tighten them locally for safety."
    )
    # NEW: optional extra squeeze (default ON)
    extra_tight = st.toggle(
        "Extra-tight window", True,
        help="Narrows guardrails around the center of the base window (no change to shared.py)."
    )
    target_green = st.slider("Target green %TD", 80, 98, 90, 1)
    st.caption("Recommendations favor q10 ≥ target for conservatism.")

# ======================= Header =======================
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot • Custom materials supported • Guardrails tightened locally")

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

    # Base windows from shared.py
    gr_base = guardrail_ranges(d50_um, on=guardrails_on)

    # -------- Tighten guardrails LOCALLY (no change to shared.py) --------
    def _tighten(lo, hi, shrink=0.65):
        """Shrink the span around the midpoint. shrink=0.65 keeps 65% of the base range."""
        lo, hi = float(lo), float(hi)
        mid = 0.5*(lo+hi)
        half = 0.5*(hi-lo)*shrink
        return max(lo, mid-half), min(hi, mid+half)

    if guardrails_on and extra_tight:
        b_lo_t, b_hi_t = _tighten(*gr_base["binder_saturation_pct"], shrink=0.60)   # keep 60% of base range
        s_lo_t, s_hi_t = _tighten(*gr_base["roller_speed_mm_s"],     shrink=0.60)
        # layer control remains from slider; show hint
        st.caption(f"Tight window: binder {b_lo_t:.0f}–{b_hi_t:.0f}% · speed {s_lo_t:.2f}–{s_hi_t:.2f} mm/s")
    else:
        b_lo_t, b_hi_t = [float(x) for x in gr_base["binder_saturation_pct"]]
        s_lo_t, s_hi_t = [float(x) for x in gr_base["roller_speed_mm_s"]]

    # Layer slider stays as before (using shared.py guidance for bounds)
    t_lo, t_hi = gr_base["layer_thickness_um"]
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

    kpi_num(k1, "Prior binder", f"{pri['binder_saturation_pct']:.0f}", "%", "Physics-informed")
    kpi_num(k2, "Prior speed",  f"{pri['roller_speed_mm_s']:.2f}", "mm/s", "Physics-informed")
    kpi_num(k3, "Layer/D50",    f"{layer_um/d50_um:.2f}", "×", "Rule-of-thumb 3–5×")

st.divider()

# ======================= Recommendations =======================
st.subheader("Recommended parameters")
colL, colR = st.columns([1, 1])
top_k = colL.slider("How many to show", 3, 8, 5, 1)
run_recs = colR.button("Recommend", type="primary", use_container_width=True)

def _within_tight(b, v):
    return (b_lo_t <= float(b) <= b_hi_t) and (s_lo_t <= float(v) <= s_hi_t)

if run_recs:
    recs = copilot(
        material=material, d50_um=float(d50_um), df_source=df_base, models=models,
        guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k)
    )
    # Filter to tight window if enabled; back-fill if fewer than top_k remain
    if guardrails_on and extra_tight and not recs.empty:
        tight_mask = recs.apply(lambda r: _within_tight(r["binder_%"], r["speed_mm_s"]), axis=1)
        recs_tight = recs[tight_mask].copy()
        if len(recs_tight) < int(top_k):
            backfill = recs[~tight_mask].head(int(top_k)-len(recs_tight))
            recs = pd.concat([recs_tight, backfill], ignore_index=True)
        else:
            recs = recs_tight

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

    # Save compact Top-k for Digital Twin
    dt = recs.rename(columns={
        "binder_%": "saturation_pct",
        "speed_mm_s": "roller_speed",
        "layer_um": "layer_um",
    }).copy()
    if "d50_um" not in dt.columns: dt["d50_um"] = float(d50_um)
    if "id" not in dt.columns:     dt["id"] = [f"Opt-{i+1}" for i in range(len(dt))]
    cols_keep = ["id","binder_type","saturation_pct","roller_speed","layer_um","d50_um"]
    st.session_state["top5_recipes_df"] = dt[cols_keep].reset_index(drop=True)
else:
    st.info("Click Recommend to generate top-k parameter sets aimed at your target green %TD.")

st.divider()

# ======================= Visuals =======================
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing (2D slice)",
    "Pareto frontier",
    "Formulae",
    "Digital Twin",
])

def _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=55, ny=45):
    sats = np.linspace(float(b_lo), float(b_hi), nx)
    spds = np.linspace(float(s_lo), float(s_hi), ny)
    grid = pd.DataFrame([(b,v,layer_um,d50_um,material) for b in sats for v in spds],
                        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    grid["material_class"] = material_class
    grid["binder_type_rec"] = binder_family
    return grid, sats, spds

# ---- Heatmap (tightened window + overlays)
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD (tight guardrails)")

    # Use the TIGHT window when guardrails_on
    b_lo, b_hi = (b_lo_t, b_hi_t) if guardrails_on else gr_base["binder_saturation_pct"]
    s_lo, s_hi = (s_lo_t, s_hi_t) if guardrails_on else gr_base["roller_speed_mm_s"]

    grid, Xs, Ys = _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family, nx=80, ny=60)
    scored = predict_quantiles(models, grid)
    Z = scored.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    z0, z1 = max(40.0, zmin), min(100.0, zmax if zmax > zmin else zmin+1.0)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=list(Xs), y=list(Ys), z=Z,
        zmin=z0, zmax=z1, colorscale="Viridis",
        zsmooth="best",
        colorbar=dict(title="%TD (q50)", len=0.85)
    ))
    # 90% & 95% contours
    for thr, col in [(90, "#C21807"), (95, "#0c7c59")]:
        fig.add_trace(go.Contour(
            x=list(Xs), y=list(Ys), z=Z,
            contours=dict(start=thr, end=thr, size=1, coloring="none"),
            line=dict(color=col, width=3, dash="dash"),
            showscale=False, name=f"{thr}% TD"
        ))

    # Prior marker
    fig.add_trace(go.Scatter(
        x=[pri["binder_saturation_pct"]], y=[pri["roller_speed_mm_s"]],
        mode="markers+text",
        marker=dict(size=11, symbol="x", color="#111827"),
        text=["prior"], textposition="top center",
        name="Prior"
    ))

    # “Focus box” around the center of the tight window
    xb0, xb1 = np.percentile(Xs, [35, 65]); yb0, yb1 = np.percentile(Ys, [35, 65])
    fig.update_layout(shapes=[
        dict(type="rect", x0=xb0, x1=xb1, y0=yb0, y1=yb1,
             line=dict(color="#2bb7b3", width=3, dash="dash"), fillcolor="rgba(0,0,0,0)")
    ])

    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        title=f"Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm · Material={material} ({material_class}) · Window: {float(b_lo):.0f}–{float(b_hi):.0f}% / {float(s_lo):.2f}–{float(s_hi):.2f} mm/s",
        legend=dict(orientation="h", y=1.02, x=1.0, xanchor="right")
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Sensitivity (uses tight window mid-speed)
with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90) in tight window")
    v_mid = 0.5*(float(s_lo_t)+float(s_hi_t)) if (guardrails_on and extra_tight) else 1.6
    sats = np.linspace(float(b_lo_t if (guardrails_on and extra_tight) else gr_base["binder_saturation_pct"][0]),
                       float(b_hi_t if (guardrails_on and extra_tight) else gr_base["binder_saturation_pct"][1]),
                       75)
    curve_df = pd.DataFrame({
        "binder_saturation_pct": sats,
        "roller_speed_mm_s": np.full_like(sats, v_mid),
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

# ---- Packing (unchanged from your working version)
with tabs[2]:
    st.subheader("Packing — 2D slice")
    st.caption("Square slice sized in ×D50. Toggle densification; preview a printed layer with binder fill.")

    c1, c2, c3, c4 = st.columns(4)
    side_mult = c1.slider("Square side (× D50)", 10, 60, 20, 2)
    cv_pct   = c2.slider("Polydispersity (CV %)", 0, 60, 20, 5)
    densify  = c3.toggle("Densify packing", False)
    seed     = c4.number_input("Seed", 0, 9999, 0, 1)

    W_mult = int(side_mult)
    base_ref, dense_ref = 260, 520
    baseline_particles   = int(base_ref  * (W_mult/20)**2)
    densified_particles  = int(dense_ref * (W_mult/20)**2)
    Npx = int(21 * W_mult)

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

    num_p = densified_particles if densify else baseline_particles
    pts, phi_area, W = rsa_pack(num_p, float(d50_um), float(cv_pct), W_mult, int(seed), densify)

    ctrlL, ctrlR = st.columns([3, 1])
    binder_sat_pct = ctrlL.slider("Binder saturation (%)", 50, 100, 80, 1)
    t_over_D50 = float(layer_um) / float(d50_um)
    ctrlR.metric("Layer/D50 used", f"{t_over_D50:.2f}×")

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

    FIGSIZE = (1.6, 1.6); DPI = 300
    colA, colB = st.columns(2)

    with colA:
        figP, axP = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axP.set_aspect('equal', 'box')
        axP.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        for (x, y, r) in pts:
            axP.add_patch(plt.Circle((x, y), r, facecolor='#3b82f6', edgecolor='#111827', linewidth=0.40, alpha=0.92))
        axP.set_xlim(0, W); axP.set_ylim(0, W)
        axP.set_xticks([]); axP.set_yticks([])
        plt.tight_layout(pad=0.1)
        st.pyplot(figP, clear_figure=True)
        side_um = W * float(d50_um)
        st.caption(f"Packing {'(densified)' if densify else '(baseline)'} • φ≈{phi_area*100:.1f}% • side≈{side_um:.0f} µm")

    with colB:
        figL, axL = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axL.set_aspect('equal', 'box')
        img = np.zeros_like(solid, dtype=float)
        img[solid] = 0.60          # solids
        img[binder_mask] = 0.95    # binder fill
        axL.imshow(img, extent=[0, W, 0, W], origin='lower', vmin=0, vmax=1, interpolation='nearest')
        axL.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        axL.set_xlim(0, W); axL.set_ylim(0, W)
        axL.set_xticks([]); axL.set_yticks([])
        plt.tight_layout(pad=0.1)
        st.pyplot(figL, clear_figure=True)
        st.caption(f"Printed layer • binder≈{binder_sat_pct}% • t/D50={t_over_D50:.2f}")

# ---- Pareto (computed in the tight window)
with tabs[3]:
    st.subheader("Pareto frontier — Binder vs green %TD (tight window)")
    b_lo, b_hi = (b_lo_t, b_hi_t) if guardrails_on else gr_base["binder_saturation_pct"]
    grid_p = pd.DataFrame({
        "binder_saturation_pct": np.linspace(float(b_lo), float(b_hi), 80),
        "roller_speed_mm_s": np.full(80, 0.5*(float(s_lo_t)+float(s_hi_t)) if (guardrails_on and extra_tight) else 1.6),
        "layer_thickness_um": np.full(80, float(layer_um)),
        "d50_um": np.full(80, float(d50_um)),
        "material": [material]*80,
        "material_class": [material_class]*80,
        "binder_type_rec": [binder_family]*80,
    })
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

# ---- Digital Twin
with tabs[5]:
    digital_twin.render(material=material, d50_um=float(d50_um), layer_um=float(layer_um))

# ======================= Diagnostics & Footer =======================
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on, "| Extra-tight:", extra_tight)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})

st.markdown(f"""
<div class="footer">
<strong>© {datetime.now().year} Bhargavi Mummareddy</strong> • Contact:
<a href="mailto:mummareddybhargavi@gmail.com">mummareddybhargavi@gmail.com</a>
</div>
""", unsafe_allow_html=True)
