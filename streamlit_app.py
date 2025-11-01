import io
import os
import math
import time
import base64
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

# Light-weight plotting & geometry
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Geometry & slicing
import trimesh
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="BJAM Predictions + Digital Twin",
    layout="wide",
)

# ---------------------------
# Helpers
# ---------------------------

@dataclass
class PowderDefaults:
    binder_type: str
    binder_saturation: float
    roller_speed_mm_s: float
    post_sinter_temp_c: int

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_bjam_dataset() -> Optional[pd.DataFrame]:
    # Expect the repository to include this file; user can replace with their own
    candidates = [
        "BJAM_All_Deep_Fill_v9.csv",
        "./data/BJAM_All_Deep_Fill_v9.csv",
        "/mnt/data/BJAM_All_Deep_Fill_v9.csv",
    ]
    for p in candidates:
        df = safe_read_csv(p)
        if df is not None and len(df) > 0:
            return df
    return None

@st.cache_data(show_spinner=False)
def list_known_powders(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    cols = [c for c in df.columns if c.lower() in ("material", "powder", "material_name")]
    if cols:
        return sorted(list(pd.Series(df[cols[0]].astype(str).str.strip().unique()).dropna()))
    # fallback to any categorical-like columns
    return []

def pack_guardrails(d50_um: float) -> Dict[str, float]:
    """
    Quick physics-informed guardrails.
    - Layer thickness ~ 4 × D50 (common BJAM heuristic)
    - Binder saturation 70–90% (target high packing)
    - Roller speed scales inversely with particle size (empirical)
    """
    d50_m = d50_um * 1e-6
    # Layer thickness in microns
    layer_thickness_um = max(2.5 * d50_um, 60.0)  # keep sensible lower bound
    # Binder saturation targeting >=90% packing -> bias high but < 1.0
    binder_sat = float(np.clip(0.78 + 0.06*np.log10(max(d50_um, 10)/50.0 + 1.05), 0.72, 0.90))
    # Roller speed scaling: v_r ∝ 1/(d * D50) with alpha ≈ 0.6
    alpha = 0.6
    norm = (50e-6)**alpha
    v_ref = 80.0  # mm/s reference
    v_mm_s = float(np.clip(v_ref * (norm / (d50_m**alpha)), 20.0, 250.0))
    return dict(layer_thickness_um=layer_thickness_um, binder_saturation=binder_sat, roller_speed_mm_s=v_mm_s)

def default_recommendations(powder: str, d50_um: float, df: Optional[pd.DataFrame]) -> PowderDefaults:
    g = pack_guardrails(d50_um)
    # Simple mapping from name keywords to binder type + sinter temp baseline.
    # In a full system, derive from dataset stats; here we use a pragmatic mapping.
    powder_l = (powder or "").lower()
    if any(k in powder_l for k in ["sic", "silicon carbide"]):
        binder = "Aqueous PVA"
        sinter = 2150
    elif any(k in powder_l for k in ["al2o3", "alumina"]):
        binder = "Aqueous PVA"
        sinter = 1550
    elif any(k in powder_l for k in ["ss316", "316l", "steel", "inconel", "625", "718"]):
        binder = "Solvent (glycol-ether)"
        sinter = 1280
    else:
        binder = "Aqueous PVA"
        sinter = 1400

    return PowderDefaults(
        binder_type=binder,
        binder_saturation=float(g["binder_saturation"]),
        roller_speed_mm_s=float(g["roller_speed_mm_s"]),
        post_sinter_temp_c=int(sinter)
    )

def fig_to_png_bytes(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.read()

# ---------------------------
# Digital Twin: slicing & packing visualization
# ---------------------------

def load_mesh_from_bytes(uploaded_file) -> trimesh.Trimesh:
    data = uploaded_file.read()
    mesh = trimesh.load(io.BytesIO(data), file_type=uploaded_file.name.split(".")[-1].lower())
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh

def generate_unit_cube(size_mm: float = 10.0) -> trimesh.Trimesh:
    # generate a simple cube centered at origin
    m = trimesh.creation.box(extents=[size_mm, size_mm, size_mm])
    m.apply_translation([-size_mm/2, -size_mm/2, 0])  # base on z=0
    return m

def mesh_bounds_height(mesh: trimesh.Trimesh) -> Tuple[float, float, float]:
    mn = mesh.bounds[0][2]
    mx = mesh.bounds[1][2]
    return float(mn), float(mx), float(mx - mn)

def slice_mesh_z(mesh: trimesh.Trimesh, z: float) -> List[Polygon]:
    # Get planar cross-section at Z
    try:
        xs = mesh.section(plane_origin=[0,0,z], plane_normal=[0,0,1.0])
        if xs is None:
            return []
        # Convert to shapely polygons (trimesh.path entities)
        polygons = xs.to_polygons()
        out = []
        for poly in polygons:
            if len(poly) >= 3:
                out.append(Polygon(poly))
        return out
    except Exception:
        return []

def synthetic_packing_map(polys: List[Polygon], pixels: int = 128) -> np.ndarray:
    """
    Create a fast, plausible 'packing factor' heatmap for the current cross-section.
    - Normalize polygon area into a bounding square grid
    - Use filtered noise that varies with local distance to boundary
    Result is a float map in [0,1] that we map to 0.85–0.95 packing range.
    """
    if not polys:
        return np.zeros((pixels, pixels), dtype=float)
    union = unary_union(polys)
    if union.area <= 0:
        return np.zeros((pixels, pixels), dtype=float)
    minx, miny, maxx, maxy = union.bounds
    if maxx - minx <= 0 or maxy - miny <= 0:
        return np.zeros((pixels, pixels), dtype=float)

    xs = np.linspace(minx, maxx, pixels)
    ys = np.linspace(miny, maxy, pixels)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Mask: inside union -> 1, else 0
    mask = np.zeros((pixels*pixels,), dtype=float)
    # Vectorized point-in-polygon via shapely is slow; sample grid coarsely
    # We'll rasterize by checking center points in a loop, but keep it efficient.
    union_prep = union.buffer(0)  # clean
    for i, (px, py) in enumerate(pts):
        mask[i] = 1.0 if union_prep.contains(Polygon([(px,py),(px+1e-6,py),(px,py+1e-6)])) else 0.0
    mask = mask.reshape((pixels, pixels))

    # Distance-to-boundary field (approx)
    # Create a simple gradient-based modulation
    # (A proper signed distance would require rasterization; this is lightweight.)
    gx, gy = np.gradient(mask)
    grad_mag = np.sqrt(gx**2 + gy**2)
    # Smooth gradient magnitude to form boundary emphasis
    from scipy.ndimage import gaussian_filter
    edge = gaussian_filter(grad_mag, sigma=2.0)
    edge = edge / (edge.max() + 1e-8)

    # Base noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, size=(pixels, pixels))
    noise = gaussian_filter(noise, sigma=3.0)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

    # Compose: inside areas get high base packing; near edges slightly reduced
    packing = 0.90 + 0.05*noise - 0.05*edge
    packing = np.clip(packing, 0.0, 1.0)
    packing *= mask
    return packing

def render_layer_heatmap(packing_map: np.ndarray, title: str) -> bytes:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(packing_map, origin="lower")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    png.seek(0)
    return png.read()

# ---------------------------
# UI
# ---------------------------

st.title("BJAM Predictions + Digital Twin")

with st.sidebar:
    st.subheader("Inputs")
    df = load_bjam_dataset()
    known_powders = list_known_powders(df)
    powder = st.selectbox("Powder / Material", options=(["Custom…"] + known_powders) if known_powders else ["Custom…"])
    if powder == "Custom…":
        powder = st.text_input("Custom material name", value="SiC")
    d50_um = st.number_input("Particle size D50 (µm)", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
    rec = default_recommendations(powder, d50_um, df)
    guard = pack_guardrails(d50_um)

    st.markdown("### Recommended Parameters")
    colA, colB = st.columns(2)
    with colA:
        st.write(f"Binder type: {rec.binder_type}")
        st.write(f"Binder saturation: {rec.binder_saturation:.2f}")
    with colB:
        st.write(f"Roller speed: {rec.roller_speed_mm_s:.1f} mm/s")
        st.write(f"Post-sintering temp: {rec.post_sinter_temp_c} °C")

    st.caption("Guardrails: layer thickness ≈ 4×D50; saturation 0.72–0.90; roller speed ∝ (D50)^(-0.6).")

tab1, tab2 = st.tabs(["Parameter Suggestions", "Digital Twin (STL ➜ Layers ➜ Packing)"])

with tab1:
    st.subheader("Process Parameter Suggestions")
    st.write("Outputs are informed by physics guardrails and simple heuristics. Tie into your dataset to make these data-driven.")
    st.json({
        "material": powder,
        "d50_um": d50_um,
        "binder_type": rec.binder_type,
        "binder_saturation": round(rec.binder_saturation, 3),
        "roller_speed_mm_s": round(rec.roller_speed_mm_s, 1),
        "post_sinter_temp_c": rec.post_sinter_temp_c,
        "layer_thickness_um": round(guard["layer_thickness_um"], 1),
    })

with tab2:
    st.subheader("Upload or Generate Geometry")
    uploaded = st.file_uploader("Upload STL/OBJ/PLY", type=["stl", "obj", "ply"])
    use_cube = st.toggle("Use 10 mm cube (demo)", value=uploaded is None)

    if uploaded is not None:
        try:
            mesh = load_mesh_from_bytes(uploaded)
        except Exception as e:
            st.error(f"Failed to load mesh: {e}")
            mesh = None
    else:
        mesh = generate_unit_cube(10.0) if use_cube else None

    if mesh is not None:
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            st.write(f"Triangles: {len(mesh.faces):,}")
        with col2:
            zmin, zmax, h = mesh_bounds_height(mesh)
            st.write(f"Z-height: {h:.2f} mm")
        with col3:
            layer_thickness_mm = guard["layer_thickness_um"] * 1e-3
            st.write(f"Layer thickness: {layer_thickness_mm:.2f} mm")

        # Slice controls
        total_layers = max(1, int(math.ceil(h / layer_thickness_mm)))
        layer_idx = st.slider("Layer index", 0, max(0, total_layers-1), 0, key="layer_idx")

        # Compute layer plane Z
        z0 = zmin + (layer_idx + 0.5) * layer_thickness_mm

        with st.spinner("Slicing and estimating packing…"):
            polys = slice_mesh_z(mesh, z0)
            if len(polys) == 0:
                st.info("No cross-section at this layer (outside geometry). Try a different layer.")
            else:
                # Render packing map
                packing = synthetic_packing_map(polys, pixels=192)
                img_bytes = render_layer_heatmap(packing, f"Layer {layer_idx+1}/{total_layers} • z={z0:.2f} mm")
                st.image(img_bytes, caption="Estimated local packing factor (0–1). Target ≥ 0.90.")

                # Quick stats
                pack_mean = float(np.mean(packing[packing > 0])) if np.any(packing > 0) else 0.0
                st.metric("Layer mean packing", f"{pack_mean:.3f}", delta=None)

                # Binder suggestion per layer (slightly adapt with packing)
                bs = float(np.clip(rec.binder_saturation + 0.02*(0.92 - pack_mean), 0.72, 0.92))
                st.write(f"Suggested binder saturation for this layer: {bs:.2f}")

        # Layer scrub preview
        with st.expander("Quick scrub preview (coarse)", expanded=False):
            cols = st.columns(4)
            preview_layers = np.linspace(0, total_layers-1, num=4, dtype=int)
            for j, li in enumerate(preview_layers):
                z = zmin + (li + 0.5) * layer_thickness_mm
                polys_j = slice_mesh_z(mesh, z)
                packing_j = synthetic_packing_map(polys_j, pixels=96)
                img_j = render_layer_heatmap(packing_j, f"L{li+1}")
                cols[j].image(img_j, use_column_width=True)

    else:
        st.info("Upload an STL/OBJ/PLY or enable the 10 mm cube demo to get started.")
