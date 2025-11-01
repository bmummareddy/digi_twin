# streamlit_app.py
# BJAM — Binder-Jet AM Parameter Recommender (bright, friendly UI)
# Uses shared.py for: loading dataset, guardrails, physics priors, quantile models, and copilot.

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for servers
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import streamlit as st

from shared import (
    load_dataset,
    train_green_density_models,
    predict_quantiles,
    physics_priors,
    guardrail_ranges,
    copilot,
)

# -----------------------------------------------------------------------------
# Page setup (bright, inviting)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BJAM Predictions",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A light, “ivory + bright accents” feel without exposing theme explicitly
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #FFFDF7 0%, #FFF8EC 40%, #FFF4E2 100%);
    }
    [data-testid="stSidebar"] {
        background: #fffdfa;
        border-right: 1px solid #f3e8d9;
    }
    .metric-container div {
        background: #fffefa !important;
        border: 1px solid #f3e8d9 !important;
        border-radius: 12px !important;
        padding: 8px 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #fff;
        border: 1px solid #f3e8d9;
        padding: 6px 10px;
        border-radius: 10px;
    }
    div[data-testid="stStatusWidget"] { opacity: .85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading (single source of truth; uses BJAM_All_Deep_Fill_v9.csv via shared.py)
# -----------------------------------------------------------------------------
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# Sidebar — data + guardrails
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data: {Path(src).name}  ·  Rows: {len(df_base):,}")
        st.download_button(
            "Download source dataset (CSV)",
            data=df_base.to_csv(index=False).encode("utf-8"),
            file_name=Path(src).name,
            mime="text/csv",
            help="Exports the exact dataset driving optimization & visuals.",
        )
    else:
        st.warning("No dataset found. Running on physics priors only (few-shot disabled).")

    st.divider()

    guardrails_on = st.toggle(
        "Guardrails",
        value=True,
        help=(
            "Keep binder saturation, roller speed, and layer thickness within physically sensible ranges "
            "derived from Furnas packing, Washburn penetration, and vendor envelopes."
        ),
    )
    st.caption("Turn this off if you want to explore outside recommended ranges.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · bright, friendly UI · toggle guardrails to explore")

# Top metrics / preview
m1, m2, m3 = st.columns(3)
m1.metric("Rows in dataset", f"{len(df_base):,}")
m2.metric("Materials", f"{df_base['material'].nunique() if 'material' in df_base else 0:,}")
m3.metric("Quantile models", "Trained" if models else "Physics-only")

with st.expander("Preview source data", expanded=False):
    if len(df_base):
        st.dataframe(df_base.head(25), use_container_width=True)
    else:
        st.info("No rows to preview.")

st.divider()

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
left, right = st.columns([1.1, 1])

# Material & D50
with left:
    st.subheader("Inputs", help="Set material & D50; layer default ≈ 4×D50, adjustable below.")

    # Material options from data (fallback to text input if empty)
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []
    use_list = st.toggle("Pick material from dataset", value=bool(materials))

    if use_list and materials:
        material = st.selectbox("Material", materials, index=0)
    else:
        material = st.text_input("Material", value="Silicon Carbide (SiC)")

    d50_um = st.number_input("D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*d50_um))), 1)

    # Target green density (%TD)
    target_green = st.slider("Target green density (% of theoretical)", 80, 98, 90, 1)

# Guarded ranges + Top-k
with right:
    st.subheader("Recommend")
    top_k = st.selectbox("How many recipes?", [3, 5, 7, 10], index=1)

    # Show active guardrails
    with st.expander("Active ranges", expanded=False):
        gr = guardrail_ranges(d50_um, on=guardrails_on)
        st.json({k: [float(v[0]), float(v[1])] for k, v in gr.items()}, expanded=False)

    recommend_btn = st.button("Recommend", use_container_width=True, type="primary")

    if recommend_btn:
        recs = copilot(
            material=material,
            d50_um=float(d50_um),
            df_source=df_base,
            models=models,
            guardrails_on=guardrails_on,
            target_green=float(target_green),
            top_k=int(top_k),
        )
        st.dataframe(recs, use_container_width=True)
        st.session_state['top5_recipes_df'] = recs
        st.caption(
            "Ranking favors **q10 ≥ target** (conservative), then **q50**; light penalty for extreme binder/speed. "
            "Use the visuals below to see *why* these are suggested."
        )
    else:
        st.info("Click **Recommend** to generate top-k parameter sets.")

st.divider()

# -----------------------------------------------------------------------------
# Visuals (Heatmap, Sensitivity, Packing)
# -----------------------------------------------------------------------------
tabs = st.tabs(
    [
        "Heatmap (speed × saturation)",
        "Saturation sensitivity",
        "Qualitative packing",
        "Formulae (symbols)",
        "Digital Twin (Beta)",
    ]
)

# -- Heatmap (speed × saturation, fixed layer & d50) --------------------------
with tabs[0]:
    st.subheader("Heatmap — Predicted green %TD", help="Layer fixed to your slider; explore speed × saturation.")
    # Build grid
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    b_lo, b_hi = gr["binder_saturation_pct"]
    s_lo, s_hi = gr["roller_speed_mm_s"]

    s_vals = np.linspace(float(b_lo), float(b_hi), 55)   # saturation axis
    v_vals = np.linspace(float(s_lo), float(s_hi), 45)   # speed axis

    grid = pd.DataFrame(
        [(b, v, layer_um, d50_um, material) for b in s_vals for v in v_vals],
        columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um", "material"],
    )
    # Fill categorical hints
    grid["material_class"] = "metal"  # best-effort; shared.predict_quantiles tolerates this
    grid["binder_type_rec"] = "solvent_based"

    scored = predict_quantiles(models, grid)
    # Pivot to matrices for z
    Xs = sorted(scored["binder_saturation_pct"].unique())
    Ys = sorted(scored["roller_speed_mm_s"].unique())
    z = scored.sort_values(["binder_saturation_pct", "roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure(data=go.Heatmap(z=z, x=Xs, y=Ys, colorscale="YlOrRd"))
    fig.update_layout(
        xaxis_title="Binder saturation (%)",
        yaxis_title="Roller speed (mm/s)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use guardrails to limit to recommended ranges; darker = higher predicted %TD.")

# -- Saturation sensitivity (1D at fixed speed) --------------------------------
with tabs[1]:
    st.subheader("Saturation sensitivity", help="1D slice at representative roller speed; see sensitivity.")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    s_lo, s_hi = gr["roller_speed_mm_s"]
    v_mid = float((s_lo + s_hi) / 2.0)

    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 50)
    grid = pd.DataFrame(
        {
            "binder_saturation_pct": sat_axis,
            "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
            "layer_thickness_um": np.full_like(sat_axis, layer_um),
            "d50_um": np.full_like(sat_axis, d50_um),
            "material": [material] * len(sat_axis),
        }
    )
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    sc = predict_quantiles(models, grid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q90"], name="q90", mode="lines"))
    fig.update_layout(xaxis_title="Binder saturation (%)", yaxis_title="Predicted green %TD", height=380, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# -- Qualitative packing (toy 2D) ---------------------------------------------
with tabs[2]:
    st.subheader("Qualitative packing", help="Toy 2D random close packing sketch to build intuition.")
    np.random.seed(42)
    target_green = float(target_green)
    pts = np.random.rand(150, 2)
    r = (target_green / 100.0) * 0.03 + 0.005

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax3.set_aspect("equal", "box")
    for (x, y) in pts:
        circ = plt.Circle((x, y), r, alpha=0.75)
        ax3.add_patch(circ)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([]); ax3.set_yticks([])
    ax3.set_title(f"Qualitative packing slice (~{target_green:.0f}% effective packing)")
    st.pyplot(fig3, clear_figure=True)

# -- Formulae (symbols, LaTeX) ------------------------------------------------
with tabs[3]:
    st.subheader("Formulae (symbols)", help="Key symbolic relations used for intuition and display.")
    st.latex(r"\text{Furnas multi-size packing:}\quad \phi_{\max} \approx 1-\prod_i (1-\phi_i)")
    st.latex(r"\text{Washburn penetration:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta} \, r \, t}")
    st.latex(r"\text{Layer guidance:}\quad 3 \le \frac{t}{D_{50}} \le 5")
    st.latex(r"\text{Packing fraction:}\quad \phi = \frac{V_{\text{solids}}}{V_{\text{total}}}")
    st.caption(
        "These relations guide priors and plots. The few-shot model refines predictions from your dataset."
    )

# -- Digital Twin (Beta) -------------------------------------------------------
with tabs[4]:
    st.subheader("Digital Twin — Layer Viewer & Compare")

    # Soft imports so missing wheels don't crash the whole app
    try:
        import trimesh
        from shapely.geometry import Polygon, Point, box
        from shapely.ops import unary_union
        HAVE_GEOM = True
    except Exception as _e:
        HAVE_GEOM = False
        st.error("Digital Twin requires extra packages: 'trimesh' and 'shapely'. "
                 "Add them to requirements.txt and redeploy.")

    try:
        from scipy import ndimage as ndi
        HAVE_SCIPY = True
    except Exception:
        HAVE_SCIPY = False

    # Helper functions are local to this tab to avoid name clashes
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Circle as _Circle, Rectangle as _Rectangle
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    import plotly.graph_objects as _go

    def _lognormal_from_quantiles(_d50, _d10=None, _d90=None):
        _med = max(1e-9, float(_d50))
        if _d10 and _d90 and _d90 > _d10 > 0:
            _m = _np.log(_med)
            _s = ( _np.log(_d90) - _np.log(_d10) ) / (2*1.2815515655446004)
            _s = float(max(_s, 0.05))
        else:
            _m, _s = _np.log(_med), 0.25
        return _m, _s

    def _sample_psd_um(_n, _d50, _d10=None, _d90=None, _seed=42):
        _rng = _np.random.default_rng(_seed)
        _m, _s = _lognormal_from_quantiles(_d50, _d10, _d90)
        return _np.exp(_rng.normal(_m, _s, size=_n))

    def _slice_mesh_polygons(mesh, z):
        sect = mesh.section(plane_origin=(0,0,z), plane_normal=(0,0,1))
        if sect is None: return []
        planar, _ = sect.to_planar()
        polys = [Polygon(p) for p in getattr(planar, "polygons_full", [])]
        return [p.buffer(0) for p in polys if p.is_valid and p.area > 1e-8]

    def _apply_fov_crop(polys, fov_size):
        if not polys: return []
        domain = unary_union(polys)
        cx, cy = domain.centroid.x, domain.centroid.y
        half = fov_size/2.0
        window = box(cx-half, cy-half, cx+half, cy+half)
        cropped = domain.intersection(window)
        if cropped.is_empty: return []
        if isinstance(cropped, Polygon): return [cropped]
        return [g for g in cropped.geoms if isinstance(g, Polygon) and g.is_valid and g.area>1e-8]

    def _pack_particles(polys, diam_units, target_phi2D=0.80, max_particles=1200, max_trials=220000, seed=0):
        if not polys: return _np.empty((0,2)), _np.empty((0,)), 0.0
        domain = unary_union(polys)
        minx, miny, maxx, maxy = domain.bounds
        area_domain = domain.area
        diam = _np.sort(_np.asarray(diam_units))[::-1]
        placed_xy, placed_r = [], []
        area_circles = 0.0
        target_area = float(_np.clip(target_phi2D, 0.05, 0.90)) * area_domain
        rng = _np.random.default_rng(seed)
        cell = max((diam.max()/2.0), (maxx-minx+maxy-miny)/400.0)
        grid = {}
        def ok(x,y,r):
            gx, gy = int(x//cell), int(y//cell)
            for ix in range(gx-1,gx+2):
                for iy in range(gy-1,gy+2):
                    for j in grid.get((ix,iy), []):
                        dx, dy = x-placed_xy[j][0], y-placed_xy[j][1]
                        if dx*dx + dy*dy < (r+placed_r[j])**2: return False
            return domain.contains(Point(x,y))
        trials=0
        for d in diam:
            r=d/2.0
            for _ in range(180):
                trials+=1
                if trials>max_trials or area_circles>=target_area or len(placed_xy)>=max_particles: break
                x=rng.uniform(minx,maxx); y=rng.uniform(miny,maxy)
                if ok(x,y,r):
                    idx=len(placed_xy); placed_xy.append((x,y)); placed_r.append(r)
                    gx, gy = int(x//cell), int(y//cell)
                    grid.setdefault((gx,gy), []).append(idx)
                    area_circles += math.pi*r*r
            if trials>max_trials or area_circles>=target_area or len(placed_xy)>=max_particles: break
        centers = _np.array(placed_xy) if placed_xy else _np.empty((0,2))
        radii   = _np.array(placed_r)  if placed_r  else _np.empty((0,))
        return centers, radii, (area_circles/area_domain)

    def _raster_particle_mask(centers, radii, fov_size, px=780):
        img = _Image.new("L", (px,px), color=0)
        drw = _ImageDraw.Draw(img)
        sx = px / fov_size; sy = px / fov_size
        for (x,y), r in zip(centers, radii):
            x0 = int((x - r)*sx); y0 = int((fov_size - (y + r))*sy)
            x1 = int((x + r)*sx); y1 = int((fov_size - (y - r))*sy)
            drw.ellipse([x0,y0,x1,y1], fill=255)
        return (_np.array(img) > 0)

    def _void_mask_from_saturation(pore_mask, saturation, rng=None):
        if rng is None: rng = _np.random.default_rng(0)
        pore_area = int(pore_mask.sum())
        if pore_area <= 0: return _np.zeros_like(pore_mask, dtype=bool)
        target_void = int(round((1.0 - saturation) * pore_area))
        if target_void <= 0: return _np.zeros_like(pore_mask, dtype=bool)
        if HAVE_SCIPY:
            dist = ndi.distance_transform_edt(pore_mask)
            noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.3)
            field = dist + 0.18*noise
            flat = field[pore_mask]
            kth = _np.partition(flat, len(flat)-target_void)[len(flat)-target_void]
            void_mask = _np.zeros_like(pore_mask, dtype=bool)
            void_mask[pore_mask] = field[pore_mask] >= kth
            return void_mask
        # fallback dots
        h,w = pore_mask.shape
        void = _np.zeros_like(pore_mask, dtype=bool); area=0; tries=0
        while area < target_void and tries < 100000:
            tries += 1
            r = int(np.clip(np.random.normal(3.0,1.2), 1.0, 6.0))
            x = np.random.randint(r, w-r); y = np.random.randint(r, h-r)
            if pore_mask[y, x]:
                yy, xx = np.ogrid[-y:h-y, -x:w-x]
                disk = (xx*xx + yy*yy) <= r*r
                add = np.logical_and(disk, pore_mask)
                void[add] = True
                area = int(void.sum())
        return void

    if HAVE_GEOM:
        # Controls
        colA, colB, colC = st.columns([1.2,1,1])
        with colA:
            dt_d50 = st.number_input("D50 (µm)", min_value=1.0, value=float(d50_um), step=1.0, key="dt_d50")
            dt_d10 = st.number_input("D10 (µm, opt)", min_value=0.0, value=float(0.0), step=1.0, key="dt_d10")
            dt_d90 = st.number_input("D90 (µm, opt)", min_value=0.0, value=float(0.0), step=1.0, key="dt_d90")
        with colB:
            stl_units = st.selectbox("STL units", ["mm","m"], index=0, key="dt_units")
            um_to_unit = 1e-3 if stl_units=="mm" else 1e-6
            fov_mm = st.slider("Field of view (mm)", 0.10, 2.0, float(max(0.1, min(2.0, 0.5))), 0.05, key="dt_fov")
        with colC:
            layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 0.9*float(d50_um)))), 1, key="dt_layer")
            phi2D_target = float(np.clip(0.9 * st.slider("Target green packing φ_TPD", 0.85, 0.95, 0.90, 0.01, key="dt_tpd"), 0.40, 0.88))
            cap = st.slider("Visual cap (particles)", 100, 2500, 1200, 50, key="dt_cap")

        # STL upload or cube
        c1, c2 = st.columns([2,1])
        with c1:
            stl = st.file_uploader("Upload STL", type=["stl"], key="dt_upl")
        with c2:
            use_cube = st.checkbox("Use built-in 10 mm cube", value=False, key="dt_cube")

        mesh = None
        if use_cube:
            mesh = trimesh.creation.box(extents=(10.0,10.0,10.0))
        elif stl is not None:
            try:
                mesh = trimesh.load(io.BytesIO(stl.read()), file_type="stl", force="mesh")
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump(concatenate=True)
            except Exception as e:
                st.error(f"Could not read STL: {e}")

        if mesh is not None:
            st.plotly_chart(_go.Figure(data=[_go.Mesh3d(x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                                                        i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                                                        color='lightgray', opacity=0.55, flatshading=True, name='Part')]
                                     ).update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0), height=360),
                            use_container_width=True)

            minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
            thickness = layer_um * (1e-3 if stl_units=='mm' else 1e-6)
            n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
            st.markdown(f"Layers: {n_layers}  |  Z span: {maxz - minz:.3f} {stl_units}")
            layer_idx = st.slider("Layer index", 1, n_layers, 1, key="dt_idx")

            # Slice & FOV
            z = minz + (layer_idx - 0.5) * thickness
            polys_full = _slice_mesh_polygons(mesh, z)
            polys = _apply_fov_crop(polys_full, fov_mm)
            if not polys:
                st.warning("No cross-section at this layer (or FOV too small). Try another layer or increase FOV.")
            else:
                # Build microstructure (true-scale)
                diam_um = _sample_psd_um(7500, dt_d50, dt_d10 if dt_d10>0 else None, dt_d90 if dt_d90>0 else None, seed=10000+layer_idx)
                diam_units = diam_um * um_to_unit
                centers, radii, phi2D = _pack_particles(polys, diam_units, phi2D_target,
                                                        max_particles=cap, max_trials=240000, seed=20000+layer_idx)

                # Single vs Compare
                t1, t2 = st.tabs(["Single Recipe", "Compare Top-5"])

                with t1:
                    binder = st.selectbox("Binder type", ["PVOH","PEG","Acrylic","Furan","Other"], index=0, key="dt_binder")
                    sat = st.slider("Binder saturation (%)", 50, 95, 80, 1, key="dt_sat")
                    px = 840
                    pores = ~_raster_particle_mask(centers, radii, fov_mm, px)
                    vmask = _void_mask_from_saturation(pores, saturation=sat/100.0, rng=_np.random.default_rng(123+layer_idx))
                    fig, ax = _plt.subplots(figsize=(5.8,5.8), dpi=188)
                    # background binder
                    ax.add_patch(_Rectangle((0,0), fov_mm, fov_mm, facecolor="#F4B942", edgecolor="#111111", linewidth=1.4))
                    # voids
                    ys, xs = _np.where(vmask)
                    if len(xs):
                        xm = xs * (fov_mm/vmask.shape[1]); ym = (vmask.shape[0]-ys) * (fov_mm/vmask.shape[0])
                        ax.scatter(xm, ym, s=0.35, c="#FFFFFF", alpha=0.95, linewidths=0)
                    # particles
                    for (x,y), r in zip(centers, radii):
                        ax.add_patch(_Circle((x,y), r, facecolor="#2F6CF6", edgecolor="#1f2937", linewidth=0.25))
                    ax.set_aspect('equal','box'); ax.set_xlim(0,fov_mm); ax.set_ylim(0,fov_mm); ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f'{binder} · Sat {int(sat)}% · Layer {int(layer_um)} µm', fontsize=10)
                    st.pyplot(fig, use_container_width=True)

                with t2:
                    # Load top-5 from main page if available
                    top5 = st.session_state.get("top5_recipes_df")
                    if top5 is None or getattr(top5,'empty', True):
                        st.info("No Top-5 table found yet — generate on the Predict tab, or use Single Recipe here.")
                    else:
                        ids = list(top5["id"])
                        sel = st.multiselect("Pick recipes to view", ids, default=ids, key="dt_sel")
                        sub = top5[top5["id"].isin(sel)].reset_index(drop=True)
                        if sub.empty:
                            st.warning("Select at least one recipe.")
                        else:
                            px = 760
                            pores = ~_raster_particle_mask(centers, radii, fov_mm, px)
                            n = len(sub)
                            if n <= 3:
                                cols = st.columns(n)
                                for col, (_, row) in zip(cols, sub.iterrows()):
                                    with col:
                                        fig, ax = _plt.subplots(figsize=(5.0,5.0), dpi=185)
                                        vm = _void_mask_from_saturation(pores, float(row["saturation_pct"])/100.0,
                                                                        rng=_np.random.default_rng(123 + int(row["saturation_pct"]) + layer_idx))
                                        # render minimal
                                        ax.add_patch(_Rectangle((0,0), fov_mm, fov_mm, facecolor="#F4B942", edgecolor="#111111", linewidth=1.4))
                                        ys,xs = _np.where(vm)
                                        if len(xs):
                                            xm = xs * (fov_mm/vm.shape[1]); ym = (vm.shape[0]-ys) * (fov_mm/vm.shape[0])
                                            ax.scatter(xm, ym, s=0.35, c="#FFFFFF", alpha=0.95, linewidths=0)
                                        for (x,y), r in zip(centers, radii):
                                            ax.add_patch(_Circle((x,y), r, facecolor="#2F6CF6", edgecolor="#1f2937", linewidth=0.25))
                                        ax.set_aspect('equal','box'); ax.set_xlim(0,fov_mm); ax.set_ylim(0,fov_mm); ax.set_xticks([]); ax.set_yticks([])
                                        ax.set_title(f'{row["id"]}: {row["binder_type"]} · Sat {int(row["saturation_pct"])}% · Layer {int(row["layer_um"])} µm', fontsize=10)
                                        st.pyplot(fig, use_container_width=True)
                            else:
                                tabs2 = st.tabs(list(sub["id"]))
                                for tb, (_, row) in zip(tabs2, sub.iterrows()):
                                    with tb:
                                        fig, ax = _plt.subplots(figsize=(5.2,5.2), dpi=185)
                                        vm = _void_mask_from_saturation(pores, float(row["saturation_pct"])/100.0,
                                                                        rng=_np.random.default_rng(123 + int(row["saturation_pct"]) + layer_idx))
                                        ax.add_patch(_Rectangle((0,0), fov_mm, fov_mm, facecolor="#F4B942", edgecolor="#111111", linewidth=1.4))
                                        ys,xs = _np.where(vm)
                                        if len(xs):
                                            xm = xs * (fov_mm/vm.shape[1]); ym = (vm.shape[0]-ys) * (fov_mm/vm.shape[0])
                                            ax.scatter(xm, ym, s=0.35, c="#FFFFFF", alpha=0.95, linewidths=0)
                                        for (x,y), r in zip(centers, radii):
                                            ax.add_patch(_Circle((x,y), r, facecolor="#2F6CF6", edgecolor="#1f2937", linewidth=0.25))
                                        ax.set_aspect('equal','box'); ax.set_xlim(0,fov_mm); ax.set_ylim(0,fov_mm); ax.set_xticks([]); ax.set_yticks([])
                                        ax.set_title(f'{row["id"]}: {row["binder_type"]} · Sat {int(row["saturation_pct"])}% · Layer {int(row["layer_um"])} µm', fontsize=10)
                                        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Add 'trimesh' and 'shapely' to requirements.txt to enable this tab.")

# -----------------------------------------------------------------------------
# Footer / Debug
# -----------------------------------------------------------------------------
with st.expander("Diagnostics", expanded=False):
    st.write("Models meta:", meta if meta else {"note": "No trained models (physics-only)."})
    st.write("Guardrails on:", guardrails_on)
    st.write("Source file:", src or "—")
    if not len(df_base):
        st.info("No dataset rows found. Upload or add BJAM_All_Deep_Fill_v9.csv to enable few-shot.")
