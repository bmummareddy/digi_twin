# -*- coding: utf-8 -*-
# BJAM — Binder-Jet AM Parameter Recommender + Digital Twin (surgical add-on)
# - Keeps existing app structure/UX
# - Uses your shared.py models as-is (no override)
# - Digital Twin tab: STL-first, layer-by-layer slices, fast hex-packing, cached

from __future__ import annotations
import io, importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

# ---- Project utilities you already have
from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
)

# ======================= Page & Theme =======================
st.set_page_config(
    page_title="BJAM Predictions",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
  :root { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; }
  .stApp { background: linear-gradient(180deg, #FFFDF7 0%, #FFF8EC 40%, #FFF4E2 100%) !important; }
  html, body, [class*="css"] { font-family: var(--font) !important; color: #111827 !important; }
  [data-testid="stSidebar"] { background: #fffdfa; border-right: 1px solid #f3e8d9; }
  .stTabs [data-baseweb="tab-list"] { gap: 12px; }
  .stTabs [data-baseweb="tab"] { background: #fff; border: 1px solid #f3e8d9; padding: 6px 10px; border-radius: 10px; }
  .block-container { max-width: 1200px; }
  .kpi { background:#fff; border-radius:12px; padding:16px 18px; border:1px solid rgba(0,0,0,.06); box-shadow:0 1px 2px rgba(0,0,0,.03); }
  .kpi .kpi-label { color:#1f2937; font-weight:600; font-size:1.0rem; opacity:.9; white-space:nowrap; }
  .kpi .kpi-value-line { display:flex; align-items:baseline; gap:.35rem; white-space:nowrap; }
  .kpi .kpi-value { color:#111827; font-weight:800; font-size:2.2rem; line-height:1.05; font-variant-numeric:tabular-nums; letter-spacing:.2px; }
  .kpi .kpi-unit  { color:#111827; font-weight:700; font-size:1.1rem; opacity:.85; }
  .kpi .kpi-sub   { color:#374151; opacity:.65; font-size:.9rem; margin-top:.25rem; white-space:nowrap; }
</style>
""", unsafe_allow_html=True)

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
    # Robust slider bounds to avoid Streamlit range errors
    t_lo, t_hi = [float(gr["layer_thickness_um"][0]), float(gr["layer_thickness_um"][1])]
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_lo >= t_hi:
        t_lo, t_hi = max(5.0, 3.0*d50_um), min(300.0, 5.0*d50_um)
        if t_lo >= t_hi: t_lo, t_hi = 3.0*d50_um, 5.1*d50_um
    layer_um = st.slider("Layer thickness (µm)",
                         float(round(t_lo)), float(round(t_hi)),
                         float(round(pri["layer_thickness_um"])), 1.0)

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
    # ensure an 'id' column exists for Digital Twin selection
    if "id" not in recs.columns:
        recs = recs.copy()
        recs["id"] = [f"Opt-{i+1}" for i in range(len(recs))]
    st.session_state["top5_recipes_df"] = recs.copy()

    pretty = recs.rename(columns={
        "binder_type": "Binder",
        "binder_%": "Binder sat (%)",
        "saturation_pct": "Binder sat (%)",  # accommodate both schemas
        "speed_mm_s": "Speed (mm/s)",
        "roller_speed": "Speed (mm/s)",
        "layer_um": "Layer (µm)",
        "predicted_%TD_q10": "q10 %TD",
        "predicted_%TD_q50": "q50 %TD",
        "predicted_%TD_q90": "q90 %TD",
        "td_q10": "q10 %TD",
        "td_q50": "q50 %TD",
        "td_q90": "q90 %TD",
    })
    st.dataframe(
        pretty,
        use_container_width=True,
        hide_index=True,
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
    "Heatmap",
    "Saturation sensitivity",
    "Qualitative packing",
    "Formulae",
    "Digital Twin",
    "Data health",
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
    st.subheader("Predicted green %TD (q50) — speed × saturation")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    b_lo,b_hi = gr["binder_saturation_pct"]; s_lo,s_hi = gr["roller_speed_mm_s"]
    binder_family = "water_based" if material_class in ("oxide","carbide") else "solvent_based"

    grid, Xs, Ys = _grid_for_context(b_lo,b_hi,s_lo,s_hi,layer_um,d50_um,material,material_class,binder_family)
    scored = predict_quantiles(models, grid).copy()
    Z = scored.sort_values(["binder_saturation_pct","roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=list(Xs), y=list(Ys), z=Z, colorscale="Viridis", colorbar=dict(title="%TD")))
    fig.add_hline(y=float((s_lo+s_hi)/2.0), line=dict(color="#444", dash="dot", width=1))
    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        height=520, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Sensitivity
with tabs[1]:
    st.subheader("Saturation sensitivity (q10–q90)")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 75)
    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "layer_thickness_um": np.full_like(sat_axis, layer_um),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material]*len(sat_axis),
        "material_class": material_class,
        "binder_type_rec": "water_based" if material_class in ("oxide","carbide") else "solvent_based",
    })
    pred = predict_quantiles(models, grid)
    q10 = pred["td_q10"].astype(float).values
    q50 = pred["td_q50"].astype(float).values
    q90 = pred["td_q90"].astype(float).values
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.1), dpi=175)
    ax2.plot(sat_axis, q50, color="#1f77b4", linewidth=2.0, label="q50")
    ax2.fill_between(sat_axis, q10, q90, alpha=0.18, label="q10–q90")
    ax2.axhline(target_green, linestyle="--", linewidth=1.2, color="#374151", label=f"Target {target_green}%")
    ax2.set_xlabel("Binder saturation (%)"); ax2.set_ylabel("Predicted green %TD")
    ax2.grid(True, axis="y", alpha=0.18); ax2.legend(frameon=False)
    st.pyplot(fig2, clear_figure=True)

# ---- Qualitative packing
with tabs[2]:
    st.subheader("Qualitative packing (illustrative)")
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, size=(160, 2))
    r = (float(target_green) / 100.0) * 0.03 + 0.005
    fig3, ax3 = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax3.set_aspect("equal", "box")
    for (x, y) in pts:
        circ = plt.Circle((x, y), r, alpha=0.75)
        ax3.add_patch(circ)
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(f"Slice ~{target_green:.0f}% effective packing")
    st.pyplot(fig3, clear_figure=True)

# ---- Formulae
with tabs[3]:
    st.subheader("Formulae & Physics Relations")
    st.latex(r"\text{Furnas packing:}\quad \phi_{\max} \approx 1-\prod_i (1-\phi_i)")
    st.latex(r"\text{Washburn penetration:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta} r t}")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\text{Packing fraction:}\quad \phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")

# ==============================================================================
# TAB 5: Digital Twin (STL-first, per-layer packing; does NOT alter originals)
# ==============================================================================
with tabs[4]:
    st.subheader("Digital Twin — STL slice + per-layer packing (cached)")

    # lazy import: the rest of the app works even if these are missing
    HAVE_TRIMESH = importlib.util.find_spec("trimesh") is not None
    HAVE_SHAPELY = importlib.util.find_spec("shapely") is not None
    if not (HAVE_TRIMESH and HAVE_SHAPELY):
        st.error("Please add 'trimesh' and 'shapely' to requirements.txt for Digital Twin.")
    else:
        import trimesh as _tm
        from shapely.geometry import Polygon as _Poly, Point as _Pt, box as _box
        from shapely.ops import unary_union as _uun
        from shapely import wkb as _wkb

        # ---- helpers (cached) ------------------------------------------------
        @st.cache_resource(show_spinner=False)
        def _load_mesh_from_bytes(stl_bytes: bytes):
            try:
                m = _tm.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh", process=False)
                if not isinstance(m, _tm.Trimesh):
                    m = m.dump(concatenate=True)
                m.rezero()
                return m
            except Exception as e:
                st.error(f"Could not read STL: {e}")
                return None

        @st.cache_resource(show_spinner=False)
        def _decimate_for_plot(mesh: "_tm.Trimesh", max_faces: int = 60_000) -> "_tm.Trimesh":
            try:
                if mesh.faces.shape[0] <= max_faces:
                    return mesh
                m = mesh.copy()
                try:
                    m = m.simplify_quadratic_decimation(max_faces)
                except Exception:
                    step = int(np.ceil(mesh.faces.shape[0] / max_faces))
                    m = _tm.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces[::step].copy(), process=False)
                return m
            except Exception:
                return mesh

        @st.cache_data(show_spinner=False)
        def _slice_polys_wkb(mesh_verts, mesh_faces, z: float) -> Tuple[bytes, ...]:
            m = _tm.Trimesh(vertices=mesh_verts, faces=mesh_faces, process=False)
            zmin, zmax = float(m.bounds[0][2]), float(m.bounds[1][2])
            span = max(zmax - zmin, 1e-9); eps = 1e-6 * span
            for zi in (z, z+eps, z-eps, z+2*eps, z-2*eps):
                sec = m.section(plane_origin=(0,0,float(zi)), plane_normal=(0,0,1))
                if sec is None: 
                    continue
                planar, _ = sec.to_planar()
                rings = getattr(planar, "polygons_full", None) or getattr(planar, "polygons_closed", None)
                if not rings: 
                    continue
                polys = []
                for ring in rings:
                    try:
                        p = _Poly(ring)
                        if p.is_valid and p.area > 1e-9:
                            polys.append(p.buffer(0))
                    except Exception:
                        pass
                if polys:
                    return tuple(p.wkb for p in polys)
            return tuple()

        @st.cache_data(show_spinner=False)
        def _crop_local(_polys_wkb: Tuple[bytes, ...]):
            if not _polys_wkb:
                return tuple(), (0.0, 0.0), 0.0
            polys = [_wkb.loads(p) for p in _polys_wkb]
            dom = _uun(polys)
            xmin, ymin, xmax, ymax = dom.bounds
            side = float(max(xmax - xmin, ymax - ymin))
            cx, cy = dom.centroid.x, dom.centroid.y
            half = side/2.0
            x0, y0 = cx - half, cy - half
            win = _box(x0, y0, x0+side, y0+side)
            clip = dom.intersection(win)
            if getattr(clip, "is_empty", True):
                return tuple(), (x0, y0), side
            geoms = [clip] if isinstance(clip, _Poly) else [g for g in clip.geoms if isinstance(g, _Poly)]
            local = []
            for g in geoms:
                x, y = g.exterior.xy
                local.append(_Poly(np.c_[np.array(x)-x0, np.array(y)-y0]).wkb)
            return tuple(local), (x0, y0), side

        @st.cache_data(show_spinner=False)
        def _hex_pack(_key, polys_wkb: Tuple[bytes, ...], d50_unit: float, phi_target: float,
                      fov: float, cap: int, jitter: float):
            if not polys_wkb:
                return np.empty((0,2)), np.empty((0,)), 0.0
            polys = [_wkb.loads(p) for p in polys_wkb]
            dom = _uun(polys)
            if getattr(dom, "is_empty", True):
                return np.empty((0,2)), np.empty((0,)), 0.0
            phi = float(np.clip(phi_target, 0.40, 0.88))
            r_nom = max(1e-12, d50_unit/2.0)
            k = float(np.sqrt(phi / 0.9069))  # hex close packing
            r = r_nom * k
            s = 2.0 * r; dy = r * np.sqrt(3.0)
            xs = np.arange(r, fov - r, s); ys = np.arange(r, fov - r, dy)
            pts = []
            for j, yy in enumerate(ys):
                xoff = 0.0 if (j % 2 == 0) else r
                for xx in xs:
                    x0 = xx + xoff
                    if x0 > fov - r: continue
                    pts.append((x0, yy))
            if not pts:
                return np.empty((0,2)), np.empty((0,)), 0.0
            C = np.array(pts, float)
            if jitter > 0:
                C += np.random.default_rng(123).uniform(-jitter*r, jitter*r, size=C.shape)
            try:
                fit = dom.buffer(-r)
                if getattr(fit, "is_empty", True):
                    return np.empty((0,2)), np.empty((0,)), 0.0
            except Exception:
                fit = dom
            keep = []
            for i, (cx, cy) in enumerate(C):
                if fit.contains(_Poly([(cx+r,cy),(cx,cy+r),(cx-r,cy),(cx,cy-r)])):
                    keep.append(i)
            C = C[keep]
            if C.shape[0] > cap: C = C[:cap]
            R = np.full(C.shape[0], r, float)
            area_solids = float(np.sum(np.pi * R**2))
            phi2D = area_solids / dom.area if dom.area > 0 else 0.0
            return C, R, float(np.clip(phi2D, 0.0, 1.0))

        @st.cache_data(show_spinner=False)
        def _raster_solids(_key, centers: np.ndarray, radii: np.ndarray, fov: float, px: int):
            if centers.size == 0: return np.zeros((px, px), dtype=bool)
            y, x = np.mgrid[0:px, 0:px]; s = fov / px
            xx = x * s; yy = (px - y) * s
            mask = np.zeros((px, px), dtype=bool)
            for (cx, cy), r in zip(centers, radii):
                d2 = (xx - cx)**2 + (yy - cy)**2
                mask |= (d2 <= r*r)
            return mask

        @st.cache_data(show_spinner=False)
        def _voids_from_sat(_key, pore_mask: np.ndarray, sat_frac: float, seed: int):
            rng = np.random.default_rng(seed)
            idx = np.flatnonzero(pore_mask.ravel())
            k = int((1.0 - sat_frac) * len(idx))
            vm = np.zeros_like(pore_mask, dtype=bool)
            if k > 0 and len(idx) > 0:
                vm.ravel()[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
            return vm

        def _binder_hex(binder_type: str) -> str:
            name = (binder_type or "").lower()
            if "water" in name:   return "#F2D06F"
            if "solvent" in name: return "#F2B233"
            if "acryl" in name:   return "#FFD166"
            if "furan" in name:   return "#F5C07A"
            return "#F4B942"

        # ---- UI --------------------------------------------------------------
        top5 = st.session_state.get("top5_recipes_df", pd.DataFrame())
        if top5 is None or getattr(top5, "empty", True):
            st.info("Generate Top-5 on Predict tab first.")
        else:
            left, right = st.columns([1.4, 1])
            with left:
                rec_id = st.selectbox("Pick one Top-5 recipe", list(top5["id"]), index=0)
                row = top5[top5["id"]==rec_id].iloc[0]
                sat_pct = float(row.get("saturation_pct", row.get("binder_%", 80.0)))
                binder  = str(row.get("binder_type", row.get("binder_type_rec", "water_based")))
                d50_r   = float(row.get("d50_um", d50_um))
                layer_r = float(row.get("layer_um", layer_um))
                st.caption(f"{rec_id}: {binder}, Sat={int(sat_pct)}%, Layer={int(layer_r)} µm, D50≈{int(d50_r)} µm")
            with right:
                units = st.selectbox("Model units", ["mm", "m", "inch", "custom"], index=0)
                mm_per_unit = st.number_input("Custom: mm per unit", 0.001, 10_000.0, 1.0, 0.001, disabled=(units!="custom"))
                if units == "mm":   um2unit = 1e-3
                elif units == "m":  um2unit = 1e-6
                elif units == "inch": um2unit = (1.0/25.4)*1e-3
                else: um2unit = (1.0/float(mm_per_unit))*1e-3
                px = st.slider("Render resolution (px)", 300, 1400, 800, 50)
                cap = st.slider("Particle cap", 300, 15000, 2200, 100)

            c1, c2 = st.columns([2,1])
            with c1:
                stl_file = st.file_uploader("Upload STL (required to override cube)", type=["stl"])
            with c2:
                use_cube = st.checkbox("Use 10 mm cube", value=(stl_file is None), disabled=(stl_file is not None))

            if stl_file is not None:
                mesh = _load_mesh_from_bytes(stl_file.read())
            elif use_cube:
                mesh = _tm.creation.box(extents=(10.0,10.0,10.0))
            else:
                mesh = None

            if mesh is None:
                st.warning("Upload an STL or use the 10 mm cube.")
            else:
                mesh_light = _decimate_for_plot(mesh, max_faces=60_000)
                V = mesh_light.vertices; F = mesh_light.faces
                figm = go.Figure(data=[go.Mesh3d(
                    x=V[:,0].tolist(), y=V[:,1].tolist(), z=V[:,2].tolist(),
                    i=F[:,0].tolist(), j=F[:,1].tolist(), k=F[:,2].tolist(),
                    color="lightgray", opacity=0.58, flatshading=True
                )]).update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0), height=340)
                st.plotly_chart(figm, use_container_width=True)

                minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
                thickness = layer_r * um2unit
                n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
                st.markdown(f"**Layers:** {n_layers}   |   Z span: {maxz - minz:.5f} {units}")
                layer_idx = st.slider("Layer index", 1, n_layers, 1)
                z = minz + (layer_idx - 0.5) * thickness

                polys = _slice_polys_wkb(mesh.vertices, mesh.faces, z)
                if not polys:
                    st.warning("Empty slice at this layer. Try neighboring layers or check units.")
                else:
                    local_wkb, origin, fov = _crop_local(polys)
                    d50_unit = d50_r * um2unit
                    phi_target = float(np.clip(0.90*0.90, 0.40, 0.88))  # aim φ2D~0.81 for 90% TPD goal

                    centers, radii, phi2D = _hex_pack(
                        (hash(local_wkb), round(d50_unit,9), round(fov,6), cap, layer_idx),
                        local_wkb, d50_unit, phi_target, fov, cap, jitter=0.12
                    )

                    solids = _raster_solids((hash(centers.tobytes()) if centers.size else 0, px, round(fov,6)),
                                            centers, radii, fov, px)
                    pores = ~solids
                    sat = float(np.clip(sat_pct/100.0, 0.01, 0.99))
                    voids = _voids_from_sat((hash(pores.tobytes()), round(sat,4), layer_idx), pores, sat, 123+layer_idx)

                    # image panels for speed
                    def _hex_to_rgb(h): return tuple(int(h[i:i+2],16)/255.0 for i in (1,3,5))
                    img1 = np.ones((px,px,3), float); img1[solids] = np.array([0.18, 0.38, 0.96])
                    img2 = np.ones((px,px,3), float); img2[:] = _hex_to_rgb(_binder_hex(binder))
                    img2[voids] = np.array([1.0, 1.0, 1.0]); img2[solids] = np.array([0.18, 0.38, 0.96])

                    colA, colB = st.columns(2)
                    with colA:
                        st.caption("Particles only")
                        figA = go.Figure(go.Image(z=(img1*255).astype(np.uint8)))
                        figA.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                        st.plotly_chart(figA, use_container_width=True)
                    with colB:
                        st.caption(f"{binder} · Sat {int(sat_pct)}%")
                        figB = go.Figure(go.Image(z=(img2*255).astype(np.uint8)))
                        figB.update_layout(margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), height=420)
                        st.plotly_chart(figB, use_container_width=True)

                    st.caption(f"Layer {layer_idx}/{n_layers} · FOV={fov:.5f} {units} · φ₂D≈{phi2D:.2f} · particles={radii.size} · D50={d50_unit:.5g} {units}")

# ==============================================================================
# TAB 6: Data health
# ==============================================================================
def data_health_report(df: pd.DataFrame, material: str, d50_um: float) -> pd.DataFrame:
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    if "material" in d.columns:
        d = d[d["material"].astype(str)==str(material)]
    if "d50_um" in d.columns:
        lo, hi = 0.8*d50_um, 1.2*d50_um
        d = d[(d["d50_um"]>=lo) & (d["d50_um"]<=hi)]
    cols = []
    if "green_pct_td" in d.columns:
        cols.append(("≥90% cases", int((d["green_pct_td"]>=90).sum())))
        cols.append(("n rows (±20% D50)", int(len(d))))
        if len(d): cols.append(("best %TD", float(d["green_pct_td"].max())))
    return pd.DataFrame(cols, columns=["metric","value"])

with tabs[5]:
    st.subheader("Training coverage & 90%TD evidence near this D50")
    rep = data_health_report(df_base, material, d50_um)
    if rep.empty:
        st.info("No rows found for this material / D50 window")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(rep, use_container_width=True, hide_index=True)
        with c2:
            if "d50_um" in df_base.columns and "green_pct_td" in df_base.columns:
                lo, hi = 0.8*d50_um, 1.2*d50_um
                sub = df_base[
                    (df_base["material"].astype(str)==str(material)) &
                    (df_base["d50_um"].between(lo, hi, inclusive="both"))
                ]
                if not sub.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=sub["d50_um"], y=sub["green_pct_td"], mode="markers", name="train pts"))
                    fig.add_hline(y=90, line=dict(color="#C21807", dash="dash"))
                    fig.update_layout(xaxis_title="D50 (µm)", yaxis_title="Green %TD",
                                      height=360, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No training points within ±20% of this D50")

# ======================= Diagnostics & Footer =======================
with st.expander("Diagnostics", expanded=False):
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})
