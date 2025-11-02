# streamlit_app.py — BJAM Binder-Jet AM Recommender (+ Digital Twin)
# Hardened guardrails: safe windows for sliders & plots when D50 is OOD
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# --- Project utilities (unchanged) ---
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
    suggest_binder_family,
)

# --- Digital Twin module (unchanged) ---
import digital_twin

# ---------------- UI shell ----------------
st.set_page_config(page_title="BJAM Predictions", layout="wide", initial_sidebar_state="expanded")
st.title("Physics guided Binder Jet AM Studio")
st.caption("Physics-guided • stl aware • Practical UX •Predicts process parameters based on powder particle size and type of binder")

# ---------------- Data & models ----------------
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data: {Path(src).name} · rows={len(df_base):,}")
        st.download_button("Download dataset (CSV)", df_base.to_csv(index=False).encode("utf-8"),
                           file_name=Path(src).name, mime="text/csv", use_container_width=True)
    else:
        st.warning("No dataset found. Physics-only fallbacks in use.")
    st.divider()
    guardrails_on = st.toggle("Guardrails", True)
    extra_tight   = st.toggle("Extra-tight window", True, help="Narrows guardrails locally (no shared.py change).")
    target_green  = st.slider("Target green %TD", 80, 98, 90, 1)

# ---------------- Inputs ----------------
left, right = st.columns([1.2, 1])

def _finite(x, default=None):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default

def _window_ok(lo, hi):
    return (lo is not None) and (hi is not None) and math.isfinite(lo) and math.isfinite(hi) and (hi > lo)

def _tighten(lo, hi, shrink=0.60):
    lo, hi = float(lo), float(hi)
    mid = 0.5*(lo+hi); half = 0.5*(hi-lo)*shrink
    return (mid-half, mid+half)

def _safe_guardrails(d50_um: float, guardrails_on: bool, extra_tight: bool, pri: dict):
    """
    Returns sanitized windows:
    binder_range (%), speed_range (mm/s), layer_range (um)
    Always ensures finite floats and min < max. Falls back to physics if needed.
    """
    gr = guardrail_ranges(d50_um, on=guardrails_on)

    b_lo = _finite(gr.get("binder_saturation_pct", [None,None])[0], None)
    b_hi = _finite(gr.get("binder_saturation_pct", [None,None])[1], None)
    v_lo = _finite(gr.get("roller_speed_mm_s", [None,None])[0], None)
    v_hi = _finite(gr.get("roller_speed_mm_s", [None,None])[1], None)
    t_lo = _finite(gr.get("layer_thickness_um", [None,None])[0], None)
    t_hi = _finite(gr.get("layer_thickness_um", [None,None])[1], None)

    # Physics fallbacks when base guardrails are missing/bad
    if not _window_ok(b_lo, b_hi):
        # binder prior ±20% points, clamped to [50,120]
        b0 = _finite(pri["binder_saturation_pct"], 80.0)
        b_lo, b_hi = max(50.0, b0-20.0), min(120.0, b0+20.0)
    if not _window_ok(v_lo, v_hi):
        # speed prior ±0.8 mm/s, clamped to [0.2, 6]
        v0 = _finite(pri["roller_speed_mm_s"], 1.6)
        v_lo, v_hi = max(0.2, v0-0.8), min(6.0, v0+0.8)
    if not _window_ok(t_lo, t_hi):
        # 3–5×D50 fallback
        t_lo, t_hi = max(3.0*d50_um, 5.0), 5.0*d50_um

    # Extra-tight shrink (optional)
    if guardrails_on and extra_tight:
        b_lo, b_hi = _tighten(b_lo, b_hi, 0.60)
        v_lo, v_hi = _tighten(v_lo, v_hi, 0.60)
        # Keep layer range as-is; user sets exact layer via slider.

    # Ensure slider-friendly (avoid equality)
    def _degen_fix(lo, hi, pad):
        if hi <= lo:
            lo, hi = lo - pad, hi + pad
        if hi - lo < pad:
            mid = 0.5*(lo+hi); lo, hi = mid - 0.5*pad, mid + 0.5*pad
        return float(lo), float(hi)

    b_lo, b_hi = _degen_fix(b_lo, b_hi, 2.0)   # percent
    v_lo, v_hi = _degen_fix(v_lo, v_hi, 0.1)   # mm/s
    t_lo, t_hi = _degen_fix(t_lo, t_hi, 2.0)   # um

    return (b_lo, b_hi), (v_lo, v_hi), (t_lo, t_hi)

with left:
    st.subheader("Inputs")
    mode = st.radio("Material source", ["From dataset", "Custom"], horizontal=True)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []

    if mode == "From dataset" and materials:
        material = st.selectbox("Material (dataset)", options=materials, index=0)
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

    d50_um = st.number_input("D50 (µm)", 1.0, 500.0, float(d50_default), 1.0, help="Layer guidance ≈ 3–5×D50.")
    pri = physics_priors(d50_um, binder_type_guess=None)

    # Build SAFE windows (never NaN/degenerate)
    (b_lo, b_hi), (v_lo, v_hi), (t_lo, t_hi) = _safe_guardrails(d50_um, guardrails_on, extra_tight, pri)

    # Layer slider (now guaranteed valid)
    layer_default = float(np.clip(pri["layer_thickness_um"], t_lo, t_hi))
    layer_um = st.slider("Layer thickness (µm)", float(round(t_lo)), float(round(t_hi)),
                         float(round(layer_default)), 1.0)

    auto_binder = suggest_binder_family(material, material_class)
    binder_choice = st.selectbox("Binder family", [f"auto ({auto_binder})", "solvent_based", "water_based"])
    binder_family = auto_binder if binder_choice.startswith("auto") else binder_choice

with right:
    st.subheader("Priors (for intuition)")
    k1, k2, k3 = st.columns(3)
    def _kpi(c, label, val, unit=""):
        c.metric(label, f"{val}{unit}")
    _kpi(k1, "Prior binder", f"{pri['binder_saturation_pct']:.0f}", "%")
    _kpi(k2, "Prior speed",  f"{pri['roller_speed_mm_s']:.2f}", " mm/s")
    _kpi(k3, "Layer/D50",    f"{layer_um/d50_um:.2f}", "×")

st.caption(f"Active window: binder {b_lo:.0f}–{b_hi:.0f}% · speed {v_lo:.2f}–{v_hi:.2f} mm/s")
st.divider()

# ---------------- Recommendations ----------------
st.subheader("Recommended parameters")
colL, colR = st.columns([1, 1])
top_k = colL.slider("How many to show", 3, 8, 5, 1)
run_recs = colR.button("Recommend", type="primary", use_container_width=True)

def _within_tight(b, v):
    try:
        return (b_lo <= float(b) <= b_hi) and (v_lo <= float(v) <= v_hi)
    except Exception:
        return False

if run_recs:
    recs = copilot(material=material, d50_um=float(d50_um), df_source=df_base, models=models,
                   guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k))
    # Post-filter to sanitized window; backfill if needed
    if guardrails_on and not recs.empty:
        tight_mask = recs.apply(lambda r: _within_tight(r["binder_%"], r["speed_mm_s"]), axis=1)
        recs_tight = recs[tight_mask].copy()
        if len(recs_tight) < int(top_k):
            backfill = recs[~tight_mask].head(int(top_k) - len(recs_tight))
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
    st.dataframe(pretty, use_container_width=True)
    st.download_button("Download recommendations (CSV)", pretty.to_csv(index=False).encode("utf-8"),
                       file_name="bjam_recommendations.csv", use_container_width=True)

    # handoff to Digital Twin (minimal columns)
    dt = recs.rename(columns={"binder_%":"saturation_pct","speed_mm_s":"roller_speed"})
    if "d50_um" not in dt.columns: dt["d50_um"] = float(d50_um)
    if "id" not in dt.columns: dt["id"] = [f"Opt-{i+1}" for i in range(len(dt))]
    keep = ["id","binder_type","saturation_pct","roller_speed","layer_um","d50_um"]
    st.session_state["top5_recipes_df"] = dt[keep].reset_index(drop=True)
else:
    st.info("Click Recommend to generate top-k sets.")

st.divider()

# ---------------- Tabs ----------------
tabs = st.tabs([
    "Heatmap (speed × saturation)",
    "Saturation sensitivity",
    "Packing (2D slice)",
    "Pareto frontier",
    "Formulae",
    "stl UX for particle distribution",
])

def _grid_for_context(b_lo,b_hi,v_lo,v_hi,layer_um,d50_um,material,material_class,binder_family, nx=80, ny=60):
    # Always finite ranges by this point
    sats = np.linspace(float(b_lo), float(b_hi), nx)
    spds = np.linspace(float(v_lo), float(v_hi), ny)
    grid = pd.DataFrame([(b,v,layer_um,d50_um,material) for b in sats for v in spds],
                        columns=["binder_saturation_pct","roller_speed_mm_s","layer_thickness_um","d50_um","material"])
    grid["material_class"] = material_class
    grid["binder_type_rec"] = binder_family
    return grid, sats, spds

# Heatmap (uses sanitized window; graceful failure path)
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD (q50)")

    try:
        grid, Xs, Ys = _grid_for_context(b_lo,b_hi,v_lo,v_hi,layer_um,d50_um,material,material_class,binder_family)
        pred = predict_quantiles(models, grid)
        # robust pivot
        dfZ = pd.DataFrame({
            "sat": pred["binder_saturation_pct"].astype(float),
            "spd": pred["roller_speed_mm_s"].astype(float),
            "z":   pred["td_q50"].astype(float),
        })
        Z = dfZ.pivot_table(index="spd", columns="sat", values="z").sort_index().sort_index(axis=1)
        X = Z.columns.values; Y = Z.index.values
        zmin, zmax = float(np.nanmin(Z.values)), float(np.nanmax(Z.values))
        z0, z1 = max(40.0, zmin), min(100.0, zmax if zmax > zmin else zmin+1.0)

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=Z.values, x=X, y=Y, zmin=z0, zmax=z1, zsmooth='best',
                                 colorscale='Viridis', colorbar=dict(title="%TD (q50)", len=0.85)))
        for thr, col in [(90, "#C21807"), (95, "#0c7c59")]:
            fig.add_trace(go.Contour(z=Z.values, x=X, y=Y,
                                     contours=dict(start=thr,end=thr,size=1, coloring="none"),
                                     line=dict(color=col, dash="dash", width=3), showscale=False))
        fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Roller speed (mm/s)",
                          height=520, margin=dict(l=10,r=10,t=40,b=10),
                          title=f"Window: {b_lo:.0f}–{b_hi:.0f}% / {v_lo:.2f}–{v_hi:.2f} mm/s · Layer={layer_um:.0f} µm · D50={d50_um:.0f} µm")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render heatmap with the current inputs: {e}")

# Sensitivity (mid-speed of sanitized window)
with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90)")
    v_mid = 0.5*(v_lo+v_hi)
    sats = np.linspace(b_lo, b_hi, 75)
    curve_df = pd.DataFrame({
        "binder_saturation_pct": sats,
        "roller_speed_mm_s": np.full_like(sats, v_mid),
        "layer_thickness_um": float(layer_um),
        "d50_um": float(d50_um),
        "material": material,
        "material_class": material_class,
        "binder_type_rec": binder_family,
    })
    try:
        cs = predict_quantiles(models, curve_df)
        fig2, ax2 = plt.subplots(figsize=(7.0, 4.1), dpi=175)
        ax2.plot(cs["binder_saturation_pct"], cs["td_q50"], color="#1f77b4", linewidth=2.0, label="q50")
        ax2.fill_between(cs["binder_saturation_pct"], cs["td_q10"], cs["td_q90"], alpha=0.18, label="q10–q90")
        ax2.axhline(target_green, linestyle="--", linewidth=1.2, color="#374151", label=f"Target {target_green}%")
        ax2.set_xlabel("Binder saturation (%)"); ax2.set_ylabel("Predicted green %TD")
        ax2.grid(True, axis="y", alpha=0.18); ax2.legend(frameon=False)
        st.pyplot(fig2, clear_figure=True)
    except Exception as e:
        st.warning(f"Sensitivity curve unavailable: {e}")

# Packing (your working implementation kept as-is for look/feel)
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
        MAX_ATTEMPTS = 40000
        for r in radii:
            tries = 240 if not densify else 280
            for _ in range(tries):
                x = rng.uniform(r, W-r); y = rng.uniform(r, W-r)
                if x-r<0 or x+r>W or y-r<0 or y+r>W: continue
                coll = False
                for (px,py,pr) in pts:
                    dx = x - px; dy = y - py
                    if dx*dx + dy*dy < (r+pr)**2: coll=True; break
                if not coll:
                    pts.append((x,y,r)); break
            if len(pts) >= max_particles: break

        rs = np.array([r for (_,_,r) in pts]) if pts else np.array([])
        phi = (np.pi*np.sum(rs**2))/(W*W) if W>0 and rs.size else 0.0
        return pts, phi, W

    num_p = densified_particles if densify else baseline_particles
    pts, phi_area, W = rsa_pack(num_p, float(d50_um), float(cv_pct), W_mult, int(seed), densify)

    ctrlL, ctrlR = st.columns([3, 1])
    binder_sat_pct = ctrlL.slider("Binder saturation (%)", int(b_lo), int(b_hi), min(100, int((b_lo+b_hi)/2)), 1)
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
    binder_mask = np.zeros_like(void, dtype=bool)
    if k > 0 and len(idx_void) > 0:
        chosen = rng_local.choice(idx_void, size=min(k, len(idx_void)), replace=False)
        binder_mask.ravel()[chosen] = True

    FIGSIZE = (1.6, 1.6); DPI = 300
    colA, colB = st.columns(2)

    with colA:
        figP, axP = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axP.set_aspect('equal', 'box')
        axP.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        for (x, y, r) in pts:
            axP.add_patch(plt.Circle((x, y), r, facecolor='#3b82f6', edgecolor='#111827', linewidth=0.40, alpha=0.92))
        axP.set_xlim(0, W); axP.set_ylim(0, W); axP.set_xticks([]); axP.set_yticks([])
        st.pyplot(figP, clear_figure=True)
        side_um = W * float(d50_um)
        st.caption(f"Packing {'(densified)' if densify else '(baseline)'} • φ≈{phi_area*100:.1f}% • side≈{side_um:.0f} µm")

    with colB:
        figL, axL = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        axL.set_aspect('equal', 'box')
        img = np.zeros_like(solid, dtype=float)
        img[solid] = 0.60; img[binder_mask] = 0.95
        axL.imshow(img, extent=[0, W, 0, W], origin='lower', vmin=0, vmax=1, interpolation='nearest')
        axL.add_patch(plt.Rectangle((0, 0), W, W, fill=False, linewidth=1.1, color='#111827'))
        axL.set_xlim(0, W); axL.set_ylim(0, W); axL.set_xticks([]); axL.set_yticks([])
        st.pyplot(figL, clear_figure=True)
        st.caption(f"Printed layer • binder≈{binder_sat_pct}% • t/D50={t_over_D50:.2f}")

# Pareto (tight window, robust)
with tabs[3]:
    st.subheader("Pareto frontier — Binder vs green %TD (tight window)")
    grid_p = pd.DataFrame({
        "binder_saturation_pct": np.linspace(float(b_lo), float(b_hi), 80),
        "roller_speed_mm_s": np.full(80, 0.5*(float(v_lo)+float(v_hi))),
        "layer_thickness_um": np.full(80, float(layer_um)),
        "d50_um": np.full(80, float(d50_um)),
        "material": [material]*80,
        "material_class": [material_class]*80,
        "binder_type_rec": [binder_family]*80,
    })
    try:
        sc_p = predict_quantiles(models, grid_p)[["binder_saturation_pct","td_q50"]].dropna().sort_values("binder_saturation_pct")
        pts_line = sc_p.values; idx=[]; best=-1
        for i,(b,td) in enumerate(pts_line[::-1]):
            if td>best: idx.append(len(pts_line)-1-i); best=td
        idx = sorted(idx)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=sc_p["binder_saturation_pct"], y=sc_p["td_q50"], mode="markers",
                                  marker=dict(size=6, color="#1f77b4"), name="Candidates"))
        if idx:
            fig4.add_trace(go.Scatter(x=sc_p.iloc[idx]["binder_saturation_pct"], y=sc_p.iloc[idx]["td_q50"],
                                      mode="lines+markers", marker=dict(size=7, color="#111827"),
                                      line=dict(width=2, color="#111827"), name="Pareto frontier"))
        fig4.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD (q50)",
                           height=460, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.warning(f"Pareto plot unavailable: {e}")

# Formulae
with tabs[4]:
    st.subheader("Formulae (symbols)")
    st.latex(r"\%TD = \frac{\rho_{\mathrm{bulk}}}{\rho_{\mathrm{theoretical}}}\times 100\%")
    st.latex(r"3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    st.caption("Few-shot model refines physics-guided priors using your dataset.")

# Digital Twin
with tabs[5]:
    digital_twin.render(material=material, d50_um=float(d50_um), layer_um=float(layer_um))

# Diagnostics / footer
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails:", guardrails_on, "| Extra-tight:", extra_tight)
    st.write("Windows:", dict(binder=(b_lo,b_hi), speed=(v_lo,v_hi), layer=(t_lo,t_hi)))
    st.write("Source:", src or "—")
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})
st.markdown(f"<div style='text-align:center;margin:24px 0;'>© {datetime.now().year} Bhargavi Mummareddy</div>", unsafe_allow_html=True)
