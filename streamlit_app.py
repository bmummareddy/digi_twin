# streamlit_app.py (FIXED - Layer Slicing & Visualization)
# Fixes: "No cross-section" errors and matches reference visualization style
# Shows particles with binder (yellow) and voids (white) like the reference image

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

# Page setup
st.set_page_config(
    page_title="BJAM Predictions",
    page_icon="🟨",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# FIXED: Optimized helper functions with better Z-handling
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _lognormal_from_quantiles(_d50, _d10=None, _d90=None):
    _med = max(1e-9, float(_d50))
    if _d10 and _d90 and _d90 > _d10 > 0:
        _m = np.log(_med)
        _s = (np.log(_d90) - np.log(_d10)) / (2*1.2815515655446004)
        _s = float(max(_s, 0.05))
    else:
        _m, _s = np.log(_med), 0.25
    return _m, _s

@st.cache_data(show_spinner=False)
def _sample_psd_um(_n, _d50, _d10=None, _d90=None, _seed=42):
    _rng = np.random.default_rng(_seed)
    _m, _s = _lognormal_from_quantiles(_d50, _d10, _d90)
    return np.exp(_rng.normal(_m, _s, size=_n))

@st.cache_resource(show_spinner="Loading mesh...")
def _load_mesh_from_bytes(stl_bytes):
    try:
        import trimesh
        mesh = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    except Exception as e:
        st.error(f"Could not read STL: {e}")
        return None

@st.cache_resource(show_spinner="Creating cube...")
def _get_cube_mesh():
    import trimesh
    return trimesh.creation.box(extents=(10.0,10.0,10.0))

@st.cache_data(show_spinner=False)
def _slice_mesh_polygons_fixed(_mesh_hash, _mesh_verts, _mesh_faces, z_target, tolerance=1e-6):
    """FIXED: Better mesh slicing with tolerance and validation"""
    try:
        import trimesh
        from shapely.geometry import Polygon
        
        # Reconstruct mesh
        mesh = trimesh.Trimesh(vertices=_mesh_verts, faces=_mesh_faces)
        
        # CRITICAL FIX: Use plane_origin at z_target
        sect = mesh.section(plane_origin=(0, 0, z_target), plane_normal=(0, 0, 1))
        
        if sect is None:
            return []
        
        # Convert to 2D
        planar, _ = sect.to_planar()
        
        # Extract polygons
        if hasattr(planar, 'polygons_full'):
            polys = [Polygon(p) for p in planar.polygons_full]
        elif hasattr(planar, 'polygons'):
            polys = [Polygon(p) for p in planar.polygons]
        else:
            return []
        
        # Validate and clean
        valid_polys = []
        for p in polys:
            if p.is_valid and p.area > tolerance:
                valid_polys.append(p.buffer(0))  # Clean topology
        
        return valid_polys
        
    except Exception as e:
        st.warning(f"Slice error at z={z_target:.3f}: {e}")
        return []

@st.cache_data(show_spinner=False)
def _apply_fov_crop_fixed(_polys_wkb, fov_size):
    """FIXED: FOV cropping with better handling"""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import box, wkb
    
    if not _polys_wkb:
        return []
    
    # Deserialize
    polys = [wkb.loads(p) for p in _polys_wkb]
    if not polys:
        return []
    
    domain = unary_union(polys)
    if domain.is_empty or domain.area < 1e-9:
        return []
    
    cx, cy = domain.centroid.x, domain.centroid.y
    half = fov_size / 2.0
    window = box(cx-half, cy-half, cx+half, cy+half)
    
    cropped = domain.intersection(window)
    
    if cropped.is_empty:
        return []
    if isinstance(cropped, Polygon):
        return [cropped.wkb]
    
    return [g.wkb for g in cropped.geoms if isinstance(g, Polygon) and g.is_valid and g.area > 1e-9]

@st.cache_data(show_spinner="Packing particles...")
def _pack_particles(_polys_wkb, _diam_hash, diam_units, target_phi2D=0.80, 
                   max_particles=1200, max_trials=220000, seed=0):
    """Optimized particle packing"""
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely import wkb
    
    if not _polys_wkb:
        return np.empty((0,2)), np.empty((0,)), 0.0
    
    polys = [wkb.loads(p) for p in _polys_wkb]
    domain = unary_union(polys)
    minx, miny, maxx, maxy = domain.bounds
    area_domain = domain.area
    
    diam = np.sort(np.asarray(diam_units))[::-1]
    placed_xy, placed_r = [], []
    area_circles = 0.0
    target_area = float(np.clip(target_phi2D, 0.05, 0.90)) * area_domain
    
    rng = np.random.default_rng(seed)
    cell = max((diam.max()/2.0), (maxx-minx+maxy-miny)/400.0)
    grid = {}
    
    def ok(x,y,r):
        gx, gy = int(x//cell), int(y//cell)
        for ix in range(gx-1,gx+2):
            for iy in range(gy-1,gy+2):
                for j in grid.get((ix,iy), []):
                    dx, dy = x-placed_xy[j][0], y-placed_xy[j][1]
                    if dx*dx + dy*dy < (r+placed_r[j])**2:
                        return False
        return domain.contains(Point(x,y))
    
    trials = 0
    for d in diam:
        r = d/2.0
        for _ in range(180):
            trials += 1
            if trials > max_trials or area_circles >= target_area or len(placed_xy) >= max_particles:
                break
            x = rng.uniform(minx,maxx)
            y = rng.uniform(miny,maxy)
            if ok(x,y,r):
                idx = len(placed_xy)
                placed_xy.append((x,y))
                placed_r.append(r)
                gx, gy = int(x//cell), int(y//cell)
                grid.setdefault((gx,gy), []).append(idx)
                area_circles += np.pi*r*r
        if trials > max_trials or area_circles >= target_area or len(placed_xy) >= max_particles:
            break
    
    centers = np.array(placed_xy) if placed_xy else np.empty((0,2))
    radii = np.array(placed_r) if placed_r else np.empty((0,))
    return centers, radii, (area_circles/area_domain)

@st.cache_data(show_spinner=False)
def _raster_particle_mask_fast(_centers_hash, centers, radii, fov_size, px=780):
    """Vectorized numpy rasterization"""
    y_grid, x_grid = np.mgrid[0:px, 0:px]
    sx = fov_size / px
    x_phys = x_grid * sx
    y_phys = (px - y_grid) * sx
    
    mask = np.zeros((px, px), dtype=bool)
    for (cx, cy), r in zip(centers, radii):
        dist_sq = (x_phys - cx)**2 + (y_phys - cy)**2
        mask |= (dist_sq <= r**2)
    
    return mask

@st.cache_data(show_spinner=False)
def _void_mask_from_saturation(_pore_hash, pore_mask, saturation, seed=0):
    """Generate void mask from binder saturation"""
    try:
        from scipy import ndimage as ndi
        HAVE_SCIPY = True
    except:
        HAVE_SCIPY = False
    
    rng = np.random.default_rng(seed)
    pore_area = int(pore_mask.sum())
    if pore_area <= 0:
        return np.zeros_like(pore_mask, dtype=bool)
    
    target_void = int(round((1.0 - saturation) * pore_area))
    if target_void <= 0:
        return np.zeros_like(pore_mask, dtype=bool)
    
    if HAVE_SCIPY:
        dist = ndi.distance_transform_edt(pore_mask)
        noise = ndi.gaussian_filter(rng.standard_normal(pore_mask.shape), sigma=2.3)
        field = dist + 0.18*noise
        flat = field[pore_mask]
        kth = np.partition(flat, len(flat)-target_void)[len(flat)-target_void]
        void_mask = np.zeros_like(pore_mask, dtype=bool)
        void_mask[pore_mask] = field[pore_mask] >= kth
        return void_mask
    
    # Fallback
    h, w = pore_mask.shape
    void = np.zeros_like(pore_mask, dtype=bool)
    area = 0
    tries = 0
    
    while area < target_void and tries < 50000:
        tries += 1
        r = int(np.clip(rng.normal(3.0, 1.2), 1.0, 6.0))
        x = rng.integers(r, w-r)
        y = rng.integers(r, h-r)
        if pore_mask[y, x]:
            yy, xx = np.ogrid[-y:h-y, -x:w-x]
            disk = (xx*xx + yy*yy) <= r*r
            add = np.logical_and(disk, pore_mask)
            void[add] = True
            area = int(void.sum())
    
    return void

# Data loading
df_base, src = load_dataset(".")
models, meta = train_green_density_models(df_base)

# Sidebar
with st.sidebar:
    st.header("BJAM Controls")
    if src and len(df_base):
        st.success(f"Data: {Path(src).name}  ·  Rows: {len(df_base):,}")
        st.download_button(
            "Download source dataset (CSV)",
            data=df_base.to_csv(index=False).encode("utf-8"),
            file_name=Path(src).name,
            mime="text/csv",
        )
    else:
        st.warning("No dataset found. Physics priors only.")

    st.divider()

    guardrails_on = st.toggle("Guardrails", value=True)
    st.caption("Turn off to explore wider parameter ranges.")

# Header
st.title("BJAM — Binder-Jet AM Parameter Recommender")
st.caption("Physics-guided + few-shot · **FIXED: Layer slicing & visualization**")

# Metrics
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

# Inputs
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Inputs")
    materials = sorted(df_base["material"].dropna().astype(str).unique().tolist()) if "material" in df_base else []
    use_list = st.toggle("Pick material from dataset", value=bool(materials))

    if use_list and materials:
        material = st.selectbox("Material", materials, index=0)
    else:
        material = st.text_input("Material", value="Silicon Carbide (SiC)")

    d50_um = st.number_input("D50 (µm)", min_value=1.0, value=30.0, step=1.0)
    layer_um = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 4.0*d50_um))), 1)
    target_green = st.slider("Target green density (%TD)", 80, 98, 90, 1)

with right:
    st.subheader("Recommend")
    top_k = st.selectbox("How many recipes?", [3, 5, 7, 10], index=1)

    with st.expander("Active ranges", expanded=False):
        gr = guardrail_ranges(d50_um, on=guardrails_on)
        st.json({k: [float(v[0]), float(v[1])] for k, v in gr.items()}, expanded=False)

    recommend_btn = st.button("Recommend", use_container_width=True, type="primary")

    if recommend_btn:
        recs = copilot(material=material, d50_um=float(d50_um), df_source=df_base, models=models,
                      guardrails_on=guardrails_on, target_green=float(target_green), top_k=int(top_k))
        st.dataframe(recs, use_container_width=True)
        st.session_state['top5_recipes_df'] = recs
    else:
        st.info("Click **Recommend** to generate recipes.")

st.divider()

# Tabs
tabs = st.tabs([
    "Heatmap", "Sensitivity", "Packing", "Formulae", "Digital Twin (FIXED)"
])

# Heatmap
with tabs[0]:
    st.subheader("Heatmap")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    s_vals = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 55)
    v_vals = np.linspace(float(gr["roller_speed_mm_s"][0]), float(gr["roller_speed_mm_s"][1]), 45)

    grid = pd.DataFrame(
        [(b, v, layer_um, d50_um, material) for b in s_vals for v in v_vals],
        columns=["binder_saturation_pct", "roller_speed_mm_s", "layer_thickness_um", "d50_um", "material"],
    )
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    scored = predict_quantiles(models, grid)
    Xs = sorted(scored["binder_saturation_pct"].unique())
    Ys = sorted(scored["roller_speed_mm_s"].unique())
    z = scored.sort_values(["binder_saturation_pct", "roller_speed_mm_s"])["td_q50"].to_numpy().reshape(len(Xs), len(Ys)).T

    fig = go.Figure(data=go.Heatmap(z=z, x=Xs, y=Ys, colorscale="YlOrRd"))
    fig.update_layout(xaxis_title="Binder (%)", yaxis_title="Speed (mm/s)", margin=dict(l=10,r=10,t=10,b=10), height=440)
    st.plotly_chart(fig, use_container_width=True)

# Sensitivity
with tabs[1]:
    st.subheader("Saturation sensitivity")
    gr = guardrail_ranges(d50_um, on=guardrails_on)
    v_mid = float((gr["roller_speed_mm_s"][0] + gr["roller_speed_mm_s"][1]) / 2.0)
    sat_axis = np.linspace(float(gr["binder_saturation_pct"][0]), float(gr["binder_saturation_pct"][1]), 50)
    
    grid = pd.DataFrame({
        "binder_saturation_pct": sat_axis,
        "roller_speed_mm_s": np.full_like(sat_axis, v_mid),
        "layer_thickness_um": np.full_like(sat_axis, layer_um),
        "d50_um": np.full_like(sat_axis, d50_um),
        "material": [material] * len(sat_axis),
    })
    grid["material_class"] = "metal"
    grid["binder_type_rec"] = "solvent_based"

    sc = predict_quantiles(models, grid)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q10"], name="q10", mode="lines"))
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q50"], name="q50", mode="lines"))
    fig.add_trace(go.Scatter(x=sc["binder_saturation_pct"], y=sc["td_q90"], name="q90", mode="lines"))
    fig.update_layout(xaxis_title="Binder (%)", yaxis_title="Green %TD", height=380, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Packing
with tabs[2]:
    st.subheader("Qualitative packing")
    np.random.seed(42)
    pts = np.random.rand(150, 2)
    r = (float(target_green) / 100.0) * 0.03 + 0.005

    fig3, ax3 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax3.set_aspect("equal", "box")
    for (x, y) in pts:
        ax3.add_patch(plt.Circle((x, y), r, alpha=0.75))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title(f"Qualitative packing (~{target_green:.0f}% packing)")
    st.pyplot(fig3, clear_figure=True)

# Formulae
with tabs[3]:
    st.subheader("Formulae")
    st.latex(r"\text{Furnas:}\quad \phi_{\max} \approx 1-\prod_i (1-\phi_i)")
    st.latex(r"\text{Washburn:}\quad L=\sqrt{\frac{\gamma \cos\theta}{2\eta} r t}")
    st.latex(r"\text{Layer:}\quad 3 \le \frac{t}{D_{50}} \le 5")

# Digital Twin (FIXED)
with tabs[4]:
    st.subheader("Digital Twin — FIXED Layer Viewer")
    st.success("✅ FIXED: Layer slicing validation, Z-position calculation, and visualization style")

    try:
        import trimesh
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from shapely import box, wkb
        HAVE_GEOM = True
    except:
        HAVE_GEOM = False
        st.error("Requires 'trimesh' and 'shapely'")

    if HAVE_GEOM:
        from matplotlib.patches import Circle as _Circle, Rectangle as _Rectangle

        # Controls
        colA, colB, colC = st.columns([1.2,1,1])
        with colA:
            dt_d50 = st.number_input("D50 (µm)", min_value=1.0, value=float(d50_um), step=1.0, key="dt_d50")
            dt_d10 = st.number_input("D10 (µm, opt)", min_value=0.0, value=0.0, step=1.0, key="dt_d10")
            dt_d90 = st.number_input("D90 (µm, opt)", min_value=0.0, value=0.0, step=1.0, key="dt_d90")
        with colB:
            stl_units = st.selectbox("STL units", ["mm","m"], index=0, key="dt_units")
            um_to_unit = 1e-3 if stl_units=="mm" else 1e-6
            fov_mm = st.slider("Field of view (mm)", 0.10, 2.0, 0.6, 0.05, key="dt_fov")
        with colC:
            layer_um_dt = st.slider("Layer thickness (µm)", 5, 300, int(max(10, min(200, 4.0*float(dt_d50)))), 1, key="dt_layer")
            phi2D_target = float(np.clip(0.9 * st.slider("Target φ", 0.85, 0.95, 0.90, 0.01, key="dt_tpd"), 0.40, 0.88))
            cap = st.slider("Particle cap", 100, 2500, 1200, 50, key="dt_cap")

        # STL upload or cube
        c1, c2 = st.columns([2,1])
        with c1:
            stl = st.file_uploader("Upload STL", type=["stl"], key="dt_upl")
        with c2:
            use_cube = st.checkbox("Use 10 mm cube", value=False, key="dt_cube")

        mesh = None
        if use_cube:
            mesh = _get_cube_mesh()
        elif stl is not None:
            stl_bytes = stl.read()
            mesh = _load_mesh_from_bytes(stl_bytes)

        if mesh is not None:
            # Show 3D
            st.plotly_chart(
                go.Figure(data=[go.Mesh3d(
                    x=mesh.vertices[:,0], y=mesh.vertices[:,1], z=mesh.vertices[:,2],
                    i=mesh.faces[:,0], j=mesh.faces[:,1], k=mesh.faces[:,2],
                    color='lightgray', opacity=0.55, flatshading=True
                )]).update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,t=0,b=0), height=360),
                use_container_width=True
            )

            # FIXED: Better Z calculation
            minz, maxz = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
            thickness = layer_um_dt * um_to_unit
            
            # Add small margin to ensure layers are within bounds
            effective_height = maxz - minz
            n_layers = max(1, int(np.floor(effective_height / thickness)))
            
            st.markdown(f"**Layers: {n_layers}** | Z: {minz:.3f} to {maxz:.3f} {stl_units} | Thickness: {thickness:.4f} {stl_units}")
            
            layer_idx = st.slider("Layer index", 1, n_layers, min(1, n_layers), key="dt_idx")

            # FIXED: Z calculation with center-of-layer
            z = minz + (layer_idx - 0.5) * thickness
            
            # Validate Z is within bounds
            if z < minz or z > maxz:
                st.error(f"Layer {layer_idx} at Z={z:.4f} is outside part bounds [{minz:.4f}, {maxz:.4f}]")
            else:
                st.info(f"Slicing at Z = {z:.4f} {stl_units} (layer {layer_idx}/{n_layers})")
                
                # Slice
                mesh_hash = hash((mesh.vertices.tobytes(), mesh.faces.tobytes(), z))
                polys_full = _slice_mesh_polygons_fixed(mesh_hash, mesh.vertices, mesh.faces, z)
                
                if not polys_full:
                    st.warning(f"⚠️ No cross-section at Z={z:.4f}. Try layers {max(1, layer_idx-5)}-{min(n_layers, layer_idx+5)}")
                else:
                    st.success(f"✅ Found {len(polys_full)} polygon(s) at this layer")
                    
                    # FOV crop
                    polys_full_wkb = [p.wkb for p in polys_full]
                    polys_wkb = _apply_fov_crop_fixed(polys_full_wkb, fov_mm)
                    
                    if not polys_wkb:
                        st.warning("FOV too small. Increase field of view.")
                    else:
                        # Pack particles
                        diam_um = _sample_psd_um(7500, dt_d50, 
                                                dt_d10 if dt_d10>0 else None, 
                                                dt_d90 if dt_d90>0 else None, 
                                                seed=10000+layer_idx)
                        diam_units = diam_um * um_to_unit
                        diam_hash = hash(diam_units.tobytes())
                        
                        centers, radii, phi2D = _pack_particles(
                            polys_wkb, diam_hash, diam_units, phi2D_target,
                            max_particles=cap, max_trials=240000, seed=20000+layer_idx
                        )

                        if len(centers) == 0:
                            st.warning("No particles packed. Try increasing particle cap or FOV.")
                        else:
                            # Render (like reference image)
                            st.subheader(f"Packing: φ≈{phi2D*100:.1f}% · {len(centers)} particles")
                            
                            # Single recipe visualization
                            binder = st.selectbox("Binder", ["PVOH","PEG","Acrylic","Furan"], index=0, key="dt_binder")
                            sat = st.slider("Saturation (%)", 50, 95, 80, 1, key="dt_sat")
                            
                            # Adaptive resolution
                            px = 840 if fov_mm < 1.0 else 680
                            
                            # Compute masks
                            centers_hash = hash(centers.tobytes())
                            particle_mask = _raster_particle_mask_fast(centers_hash, centers, radii, fov_mm, px)
                            pores = ~particle_mask
                            
                            pore_hash = hash((pores.tobytes(), sat))
                            vmask = _void_mask_from_saturation(pore_hash, pores, saturation=sat/100.0, seed=123+layer_idx)
                            
                            # Create figure (matching reference style)
                            fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
                            
                            # Background: binder (yellow like reference)
                            ax.add_patch(_Rectangle((0,0), fov_mm, fov_mm, 
                                                   facecolor="#F4B942", edgecolor="#222222", linewidth=2))
                            
                            # Voids (white like reference)
                            ys, xs = np.where(vmask)
                            if len(xs):
                                xm = xs * (fov_mm/vmask.shape[1])
                                ym = (vmask.shape[0]-ys) * (fov_mm/vmask.shape[0])
                                ax.scatter(xm, ym, s=0.4, c="#FFFFFF", alpha=0.98, linewidths=0)
                            
                            # Particles (teal/green like reference)
                            for (x,y), r in zip(centers, radii):
                                ax.add_patch(_Circle((x,y), r, facecolor="#2B9B84", 
                                                    edgecolor="#1a5a4f", linewidth=0.3))
                            
                            ax.set_aspect('equal','box')
                            ax.set_xlim(0,fov_mm)
                            ax.set_ylim(0,fov_mm)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title(f'Printed layer · binder≈{int(sat)}% · t/D50={layer_um_dt/dt_d50:.2f}', 
                                       fontsize=11, pad=10)
                            
                            st.pyplot(fig, use_container_width=True)
                            
                            # Stats
                            st.caption(f"Resolution: {px}×{px} | Packing: {phi2D*100:.1f}% | " +
                                     f"Voids: {vmask.sum()/(px*px)*100:.1f}% of pore space")

# Diagnostics
with st.expander("Diagnostics", expanded=False):
    st.write("**Models:**", meta if meta else {"note": "Physics-only"})
    st.write("**Guardrails:**", guardrails_on)
    st.write("**Source:**", src or "—")
    st.write("**Cache stats:**", {
        "Mesh slicing": "Cached per layer",
        "Particle packing": "Cached per geometry + PSD",
        "Rasterization": "Cached per configuration",
        "Void masks": "Cached per saturation"
    })
