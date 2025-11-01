
# streamlit_app_DIGITAL_TWIN.py
# Combined app: retains dataset usage and adds a Digital Twin tab for STL slicing & packing visualization.
# Safe to drop-in on Streamlit Cloud. If you used a single-file app before, replace it with this one.
# Libraries: streamlit, pandas, numpy, plotly, trimesh (optional), numpy-stl (fallback)

import os
import io
import math
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Try trimesh first for robust STL ops; fall back to numpy-stl if unavailable
_TRIMESH_OK = True
try:
    import trimesh
    from trimesh.voxel import creation as voxel_creation
except Exception:
    _TRIMESH_OK = False

_STL_OK = True
try:
    from stl import mesh as numpy_stl_mesh  # from numpy-stl
except Exception:
    _STL_OK = False

VERSION = "2025.11.01-digital-twin"

# ---------- Utilities ----------

@st.cache_data(show_spinner=False)
def load_master_dataset(csv_path: str = "/mnt/data/BJAM_All_Deep_Fill_v9.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Try relative or working-dir versions as fallback
        candidates = [
            "BJAM_All_Deep_Fill_v9.csv",
            os.path.join(os.getcwd(), "BJAM_All_Deep_Fill_v9.csv"),
        ]
        df = None
        for c in candidates:
            if os.path.exists(c):
                df = pd.read_csv(c)
                break
        if df is None:
            df = pd.DataFrame()
    return df

def default_layer_thickness_um(d50_um: float) -> float:
    # Common guardrail: 3–5× D50. We'll pick 4× as default.
    return max(20.0, 4.0 * float(d50_um))

def suggest_binder_settings(material: str, d50_um: float, target_td=0.90):
    # Simple physics-informed placeholder: start with base green packing and map to target post-sinter density.
    # Assumptions (tunable):
    # - Monodisperse green packing ~0.60–0.64 (random close packing ~0.64). Use 0.62 baseline.
    # - Sinter densification gain depends on material class and particle size.
    base_green = 0.62
    # Fudge: smaller particles densify more (higher surface area). Clip to [0.05, 0.18] gain.
    densify_gain = np.clip(0.18 - 0.0015 * (d50_um - 20.0), 0.05, 0.18)
    post_sinter = base_green + densify_gain

    # If not enough to reach target, instruct higher saturation and slower roller to increase packing
    needs_boost = post_sinter < target_td
    # Nominal ranges
    binder_types = ["Water-based PVOH", "PEG-water", "Furan-Solvent", "Acrylate-Solvent"]
    # Heuristic pick: ceramics -> water, metals -> solvent; SI-based simplicity
    mat_lower = (material or "").lower()
    if any(k in mat_lower for k in ["steel", "inconel", "nickel", "ti", "co", "aluminum", "aluminium", "copper", "bronze"]):
        binder_default = "Acrylate-Solvent"
    else:
        binder_default = "PVOH-Water"

    # Saturation heuristic: 70–95%
    sat = 0.78 if not needs_boost else 0.88
    # Roller traverse speed heuristic inversely scaled with D50 (slower for finer powders), clamp 50–200 mm/s
    vr = float(np.clip(220.0 - 2.0 * d50_um, 50.0, 200.0))
    # Post-sinter temperature suggestion (very rough): based on general class; user should override from literature.
    if "sic" in mat_lower or "carbide" in mat_lower:
        tsint = 2150  # °C, PIP/CVI cycles typical > 1800 °C; placeholder
    elif any(k in mat_lower for k in ["alumina", "al2o3"]):
        tsint = 1600
    elif any(k in mat_lower for k in ["zirconia", "zro2"]):
        tsint = 1450
    elif any(k in mat_lower for k in ["steel", "inconel", "nickel"]):
        tsint = 1250
    else:
        tsint = 1400

    return {
        "binder_default": binder_default,
        "binder_options": binder_types,
        "saturation": round(float(sat), 2),
        "roller_speed_mm_s": int(vr),
        "post_sinter_temp_C": tsint,
        "estimated_green_packing": round(float(base_green), 3),
        "estimated_post_sinter": round(float(post_sinter), 3),
        "meets_target": bool(post_sinter >= target_td),
    }

def _mesh_from_stl_bytes(file_bytes: bytes):
    if _TRIMESH_OK:
        m = trimesh.load(io.BytesIO(file_bytes), file_type="stl", force="mesh")
        if isinstance(m, trimesh.Scene):
            # Combine into a single mesh if needed
            m = trimesh.util.concatenate([g for g in m.geometry.values()])
        return m
    elif _STL_OK:
        # numpy-stl fallback (limited functionality)
        stl_mesh = numpy_stl_mesh.Mesh.from_file(io.BytesIO(file_bytes))
        # Wrap minimal attributes for downstream plotting
        class SimpleMesh:
            def __init__(self, m):
                self.vectors = m.vectors  # (N, 3, 3)
                self.bounds = np.array([self.vectors.min(axis=(0,1)), self.vectors.max(axis=(0,1))])
                self.vertices = np.unique(self.vectors.reshape(-1,3), axis=0)
                # Fake faces by indexing triplets
                self.faces = np.arange(self.vertices.shape[0]).reshape(-1,3)
        return SimpleMesh(stl_mesh)
    else:
        raise RuntimeError("Neither trimesh nor numpy-stl is available. Please add 'trimesh' (preferred) or 'numpy-stl' to requirements.")

def _plot_mesh_plotly(mesh_obj):
    # Build Plotly Mesh3d
    if _TRIMESH_OK and isinstance(mesh_obj, trimesh.Trimesh):
        V = mesh_obj.vertices
        F = mesh_obj.faces
    else:
        V = mesh_obj.vertices
        F = mesh_obj.faces

    x, y, z = V[:,0], V[:,1], V[:,2]
    i, j, k = F[:,0], F[:,1], F[:,2]
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.5, name="mesh")])
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=30,b=0))
    return fig

def _voxelize_mesh(mesh_obj, pitch: float):
    """Voxelize the mesh on a grid with given pitch (same units as mesh). Returns a boolean occupancy grid and grid metadata.
    """
    if _TRIMESH_OK and isinstance(mesh_obj, trimesh.Trimesh):
        vox = voxel_creation.voxelize(mesh_obj, pitch=pitch)
        occ = vox.matrix  # (nx, ny, nz) boolean
        # Build origin and pitch for indexing
        origin = vox.origin
        return occ, origin, pitch
    else:
        # Very coarse fallback: just bound-box filled for demo
        bounds = np.array([mesh_obj.vertices.min(axis=0), mesh_obj.vertices.max(axis=0)])
        size = bounds[1] - bounds[0]
        nx, ny, nz = np.maximum((size / pitch).astype(int), 2)
        occ = np.ones((nx, ny, nz), dtype=bool)
        origin = bounds[0]
        return occ, origin, pitch

def layer_indices(z_coords, layer_thickness):
    zmin, zmax = float(np.min(z_coords)), float(np.max(z_coords))
    edges = np.arange(zmin, zmax + layer_thickness, layer_thickness)
    return edges

def layer_packing_estimator(occ_slice_2d, target_green=0.62):
    """Quick estimator: occupied voxels -> local packing factor map, scaled to target_green baseline.
    """
    # Occ slice is boolean where True=filled mesh; we assume powder fully covers XY inside contour, so use a smoothed map
    occ = occ_slice_2d.astype(float)
    if occ.size == 0:
        return occ, 0.0
    # Smooth by a small kernel to mimic roller-induced packing uniformity
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(occ, sigma=1.0)
    # Normalize so mean equals target_green, clip to [0.45, 0.68] for green state
    m = sm.mean() if sm.mean() > 0 else 0.001
    pf_map = np.clip(sm * (target_green / m), 0.45, 0.68)
    return pf_map, float(pf_map.mean())

def ensure_example_cube(edge_mm=10.0):
    # Generates a simple cube using trimesh primitives if no STL provided
    if not _TRIMESH_OK:
        # Fallback: return simple numpy cube vertices/faces
        V = np.array([[0,0,0],[edge_mm,0,0],[edge_mm,edge_mm,0],[0,edge_mm,0],
                      [0,0,edge_mm],[edge_mm,0,edge_mm],[edge_mm,edge_mm,edge_mm],[0,edge_mm,edge_mm]], float)
        F = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]], int)
        class Simple:
            def __init__(self):
                self.vertices = V; self.faces = F
        return Simple()
    cube = trimesh.creation.box(extents=[edge_mm, edge_mm, edge_mm])
    return cube

# ---------- App UI ----------

st.set_page_config(page_title="BJAM Predictions + Digital Twin", layout="wide")
st.title("BJAM Predictions + Digital Twin")
st.caption(f"Build: {VERSION}")

# Keep dataset access from legacy app, if present
df_master = load_master_dataset()
if not df_master.empty:
    with st.expander("Dataset preview (kept from legacy streamlit_app.py)", expanded=False):
        st.dataframe(df_master.head(20), use_container_width=True)
else:
    st.info("Master dataset not found in default path. You can still use the Digital Twin tab.")

# Tabs: keep your original features in 'Predictions' and add 'Digital Twin'
tabs = st.tabs(["Predictions (existing)", "Digital Twin (new)"])

with tabs[0]:
    st.markdown("This tab preserves your existing prediction workflow. If you had a separate app before, keep those controls here.")
    st.write("If you share specific functions/controls, we can drop them in verbatim. For now this placeholder keeps the dataset loaded and provides the same parameter entry fields used by Digital Twin below.")
    # Minimal parameter inputs (reused in Digital Twin defaults)
    material = st.text_input("Material powder (e.g., SiC, Al2O3, 316L)", value="SiC")
    d50_um = st.number_input("Particle size D50 (μm)", min_value=1.0, max_value=200.0, value=35.0, step=1.0)
    layer_thk_um = st.number_input("Layer thickness (μm)", min_value=10.0, max_value=500.0, value=default_layer_thickness_um(35.0), step=5.0)
    recs = suggest_binder_settings(material, d50_um, target_td=0.90)
    st.subheader("Suggested process window (to reach ≥90% post-sinter density)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Binder", recs["binder_default"])
    c2.metric("Saturation", f"{int(recs['saturation']*100)} %")
    c3.metric("Roller speed", f"{recs['roller_speed_mm_s']} mm/s")
    c4.metric("Post-sinter T", f"{recs['post_sinter_temp_C']} °C")
    c5.metric("Est. post-sinter", f"{int(recs['estimated_post_sinter']*100)} %")
    st.caption("Heuristics are placeholders. Replace with your model outputs; I kept dataset wiring intact.")

with tabs[1]:
    st.header("Digital Twin: STL slicing, layer packing, and per-layer display")

    with st.sidebar:
        st.subheader("Digital Twin Controls")
        material_dt = st.text_input("Material powder", value="SiC")
        d50_um_dt = st.number_input("Particle size D50 (μm)", min_value=1.0, max_value=200.0, value=35.0, step=1.0, key="d50_dt")
        layer_thk_um_dt = st.number_input("Layer thickness (μm)", min_value=10.0, max_value=500.0, value=default_layer_thickness_um(35.0), step=5.0, key="lt_dt")
        voxel_pitch_um = st.number_input("Voxel pitch for slicing (μm)", min_value=10.0, max_value=500.0, value=default_layer_thickness_um(35.0)/2.0, step=5.0, key="vp_dt")
        st.caption("Smaller voxel pitch → finer slice maps but higher compute time.")

    cL, cR = st.columns([1.2, 1.0])

    with cL:
        st.subheader("Upload STL or use sample cube")
        f = st.file_uploader("STL file", type=["stl"])
        if f is not None:
            bytes_data = f.read()
            try:
                mesh_obj = _mesh_from_stl_bytes(bytes_data)
                st.success("STL loaded.")
            except Exception as e:
                st.error(f"Failed to load STL: {e}")
                mesh_obj = ensure_example_cube(edge_mm=10.0)
        else:
            st.info("No STL uploaded. Using a 10 mm cube sample.")
            mesh_obj = ensure_example_cube(edge_mm=10.0)

        # Basic mesh preview
        try:
            fig_mesh = _plot_mesh_plotly(mesh_obj)
            st.plotly_chart(fig_mesh, use_container_width=True)
        except Exception as e:
            st.warning(f"3D preview unavailable: {e}")

    with cR:
        st.subheader("Process Targets & Suggestions")
        recs_dt = suggest_binder_settings(material_dt, d50_um_dt, target_td=0.90)
        st.write("Target theoretical packing (post-sinter): ≥ 90%")
        st.json(recs_dt, expanded=False)

    # Convert layer thickness and voxel pitch to mesh units (assume mesh units are mm; user can rescale if needed)
    lt = float(layer_thk_um_dt) / 1000.0  # mm if STL is in mm
    vp = float(voxel_pitch_um) / 1000.0

    st.subheader("Voxelization & Slicing")
    try:
        with st.spinner("Voxelizing mesh..."):
            occ, origin, pitch = _voxelize_mesh(mesh_obj, pitch=vp)
        nx, ny, nz = occ.shape
        st.write(f"Voxel grid: {nx} × {ny} × {nz}, pitch ≈ {pitch*1000:.1f} μm")

        # Build z coordinates for slices (voxel centers)
        z_coords = origin[2] + np.arange(nz) * pitch
        # Layer edges
        edges = layer_indices(z_coords, lt)
        st.write(f"Computed {len(edges)-1} layers (thickness ≈ {lt*1000:.1f} μm).")

        # Pre-compute per-layer packing
        from scipy.ndimage import zoom  # resize slices for smoother heatmaps
        layer_means = []
        layer_maps = []
        # Map z-index to layer bin
        z_centers = z_coords + 0.5 * pitch
        layer_id_for_z = np.digitize(z_centers, edges) - 1  # 0..L-1

        L = int(layer_id_for_z.max() + 1)
        for lid in range(L):
            # Combine all slices whose centers fall in this layer
            z_idx = np.where(layer_id_for_z == lid)[0]
            if z_idx.size == 0:
                layer_maps.append(np.zeros((nx, ny)))
                layer_means.append(0.0)
                continue
            occ_layer = occ[:, :, z_idx].any(axis=2)  # occupancy mask for this layer
            pf_map, pf_mean = layer_packing_estimator(occ_layer, target_green=0.62)
            # Upscale for nicer display if small
            scale =  min(2, max(1, int(200 / max(occ_layer.shape[0], occ_layer.shape[1]))))
            if scale > 1:
                pf_map = zoom(pf_map, scale, order=1)
            layer_maps.append(pf_map)
            layer_means.append(pf_mean)

        # Layer selector
        st.subheader("Layer View")
        sel = st.slider("Layer index", 0, max(0, L-1), 0, 1)
        pf_map = layer_maps[sel]
        fig2 = go.Figure(data=go.Heatmap(z=pf_map, coloraxis="coloraxis"))
        fig2.update_layout(coloraxis={"colorscale":"Viridis"}, margin=dict(l=0,r=0,t=20,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig2, use_container_width=True)
        st.write(f"Estimated green packing (layer {sel}): {layer_means[sel]*100:.1f}%")

        # Per-layer profile
        st.subheader("Per-layer average green packing")
        fig3 = go.Figure(data=go.Scatter(y=[m*100 for m in layer_means], mode="lines+markers", name="Green packing %"))
        fig3.update_layout(yaxis_title="Green packing (%)", xaxis_title="Layer index", margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True)

        # Simple check against target post-sinter density
        g_mean = float(np.mean([m for m in layer_means if m > 0]))
        # Assume densification gain as earlier heuristic
        densify_gain = np.clip(0.18 - 0.0015 * (d50_um_dt - 20.0), 0.05, 0.18)
        post_sinter_est = g_mean + float(densify_gain)
        st.info(f"Avg. green packing (all layers): {g_mean*100:.1f}% → Est. post-sinter: {post_sinter_est*100:.1f}%")

        if post_sinter_est < 0.90:
            st.warning("Estimated post-sinter density falls short of 90%. Consider higher saturation, slower roller, finer particle size, and/or lower layer thickness.")
        else:
            st.success("Target ≥90% post-sinter density appears achievable with current settings (heuristic).")

    except Exception as e:
        st.error(f"Digital Twin failed: {e}")
        st.stop()

st.caption("Notes: The Digital Twin layer packing is a fast, heuristic estimator. For publication or production use, replace with calibrated models (Furnas packing, Washburn infiltration, sintering kinetics) and validated parameters from your dataset.")
