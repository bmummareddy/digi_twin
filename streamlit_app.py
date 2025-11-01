# streamlit_app.py — BJAM Binder-Jet AM Recommender (+ fast Digital Twin + tighter optimizer option)
# - Keeps your original UI & logic
# - Stores trials to st.session_state["top_recipes_df"]
# - Digital Twin tab with robust STL slicing, auto-FOV, cached hex-packing
# - Optional "Tighter optimizer" (diverse binder saturation & speeds)

from __future__ import annotations
import io, importlib.util, math
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# Optional deps for Digital Twin
HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
if HAVE_TRIMESH: import trimesh
if HAVE_SHAPELY:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from shapely import wkb

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
  .kpi { background:#fff; border-radius:12px; padding:16px 18px;
         border:1px solid rgba(0,0,0,0.06); box-shadow:0 1px 2px rgba(0,0,0,0.03); }
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
        help="ON: stable windows (binder ~60–110%, speed ≈1.2–3.5 mm/s, layer ≈3–5×D50). OFF: wider exploration."
    )
    target_green = st.slider("Target green %TD", 80, 98, 90, 1)
    st.caption("Recommendations prefer q10 ≥ target for conservatism.")

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
use_tight = colR.toggle("Use tighter optimizer (diverse binder sat & speed)", False,
                        help="Search a dense grid and add diversity pressure on saturation/speed so the 5 trials are not identical in binder %.")

def _dense_grid_and_score(models, d50_um, layer_um, material, material_class, binder_family, guardrails_on, target_green):
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    sat_lo, sat_hi = [float(x) for x in gr["binder_saturation_pct"]]
    spd_lo, spd_hi = [float(x) for x in gr["roller_speed_mm_s"]]
    # dense but still fast
    Xs = np.linspace(sat_lo, sat_hi, 51)
    Ys = np.linspace(spd_lo, spd_hi, 37)
    df = pd.DataFrame([(b, v, layer_um, d50_um, material) for b in Xs for v in Ys],
                      columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    df["material_class"] = material_class
    df["binder_type_rec"] = binder_family
    pred = predict_quantiles(models, df)
    cand = df.reset_index(drop=True).join(pred[["td_q10","td_q50","td_q90"]].reset_index(drop=True))
    # main score: hit target on q10, stay close on q50
    score = (np.clip(target_green - cand["td_q10"], 0, None)) * 3.0 + (cand["td_q50"] - target_green).abs()
    cand["score"] = score
    return cand.sort_values("score", ascending=True).reset_index(drop=True)

def _select_diverse(cand: pd.DataFrame, k: int, min_sat_gap: float = 3.0, min_speed_gap: float = 0.15):
    """Greedy pick with diversity gaps on saturation (%) and speed (mm/s)."""
    picked = []
    for _, row in cand.iterrows():
        sat = float(row["binder_saturation_pct"])
        spd = float(row["roller_speed_mm_s"])
        if all(abs(sat - float(p["binder_saturation_pct"])) >= min_sat_gap and
               abs(spd - float(p["roller_speed_mm_s"])) >= min_speed_gap for p in picked):
            picked.append(row)
        if len(picked) >= k: break
    if len(picked) < k:
        # top up ignoring diversity to always reach k
        need = k - len(picked)
        extra = cand.iloc[:need]
        picked.extend([extra.iloc[i] for i in range(len(extra))])
    out = pd.DataFrame(picked).copy()
    out["binder_type"] = cand.get("binder_type_rec", "water_based")
    out.rename(columns={
        "binder_saturation_pct":"binder_%",
        "roller_speed_mm_s":"speed_mm_s",
        "layer_thickness_um":"layer_um",
        "td_q10":"predicted_%TD_q10",
        "td_q50":"predicted_%TD_q50",
        "td_q90":"predicted_%TD_q90",
    }, inplace=True)
    out["meets_target_q10"] = out["predicted_%TD_q10"] >= float(target_green)
    return out[["binder_type","binder_%","speed_mm_s","layer_um","d50_um","material",
                "predicted_%TD_q10","predicted_%TD_q50","predicted_%TD_q90","meets_target_q10"]]

btn = st.button("Recommend", type="primary", use_container_width=True)

if btn:
    if not use_tight:
        # Original path: your copilot()
        recs = copilot(
            material=material, d50_um=float(d50_um), df_source=df_base, models=models,
            guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k)
        )
        # Preserve binder family choice (your original behavior)
        recs["binder_type"] = binder_family
    else:
        # Tighter optimizer: dense grid + diversity pressure
        cand = _dense_grid_and_score(models, float(d50_um), float(layer_um), material, material_class, binder_family, guardrails_on, float(target_green))
        recs = _select_diverse(cand, int(top_k), min_sat_gap=3.0, min_speed_gap=0.12)

    # Store for Digital Twin tab
    st.session_state["top_recipes_df"] = recs.copy()

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
    st.info("Click Recommend to generate top-k parameter sets aimed at your target green %TD.")

st.divider()

# ======================= Visuals =======================
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing (2D slice)",
    "Pareto frontier",
    "Formulae",
    "Digital Twin",            # NEW
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
    gr = guardrail_ranges(d50_um, on=guardrails_on)
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
    gr = guardrail_ranges(d50_um, on=guardrails_on)
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
    st.caption(
