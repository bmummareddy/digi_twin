# ============================== ADAPTER RENDER ==============================
# This adapter recreates the earlier one-call UI so streamlit_app.py can just call
# digital_twin.render(material, d50_um, layer_um) without changing its code.

def _mesh_to_arrays(_mesh):
    """Return (verts, faces, mesh_hash) for caching-friendly calls."""
    if _mesh is None:
        return None, None, None
    verts = np.asarray(_mesh.vertices, dtype=np.float64)
    faces = np.asarray(_mesh.faces, dtype=np.int64)
    mhash = hash((verts.tobytes(), faces.tobytes()))
    return verts, faces, mhash

def render(material: str, d50_um: float, layer_um: float):
    """Digital Twin — STL slice + qualitative packing with the old, simple API."""
    st.subheader("Digital Twin (Beta) — STL slice + qualitative packing")

    # Capability checks (clear, early, and non-crashing)
    missing = check_dependencies()
    if "trimesh" in missing or "shapely" in missing:
        st.error("Digital Twin requires: " + ", ".join(missing))
        st.info("Ask your admin to add these to requirements.txt (SciPy optional but recommended).")
        return

    # Pull Top-5 recipes from the main app (same key used earlier)
    top5 = st.session_state.get("top5_recipes_df")
    if top5 is None or getattr(top5, "empty", True):
        st.info("Run the Predict/Recommend tab first to generate Top-5 recipes.")
        return
    top5 = top5.reset_index(drop=True)

    # ----------------- UI controls (kept compact and stable) -----------------
    L, R = st.columns([1.2, 1])
    with L:
        rec_id = st.selectbox("Pick one trial", list(top5["id"]), index=0)
        picks  = st.multiselect("Compare trials", list(top5["id"]), default=list(top5["id"])[:3])
    with R:
        stl_units = st.selectbox("STL units", ["mm", "m"], index=0)
        um2unit = 1e-3 if stl_units == "mm" else 1e-6  # µm → model unit
        pack_full = st.checkbox("Pack full slice (auto FOV)", value=True)
        fov_mm = st.slider("Manual FOV (mm)", 0.5, 6.0, 1.5, 0.05, disabled=pack_full)
        phi_TPD = st.slider("Target φ (theoretical packing density)", 0.85, 0.95, 0.90, 0.01)
        phi2D_target = float(np.clip(0.90 * phi_TPD, 0.40, 0.88))
        cap = st.slider("Visual cap (particles)", 200, 3500, 1600, 50)
        fast = st.toggle("Fast mode (coarser packing)", value=True,
                         help="Speeds up packing by limiting attempts & sample size.")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        stl_file = st.file_uploader("Upload STL", type=["stl"])
    with c2:
        use_cube = st.checkbox("Use built-in 10 mm cube", value=False)
    with c3:
        show_mesh = st.checkbox("Show 3D mesh preview", value=True)

    # ----------------- Mesh load (cached resources) -----------------
    mesh = None
    if use_cube:
        try:
            mesh = get_cube_mesh()
        except Exception as e:
            st.error(f"Could not create cube mesh: {e}")
            return
    elif stl_file is not None:
        try:
            mesh = load_mesh_from_bytes(stl_file.read())
        except Exception as e:
            st.error(f"Could not read STL: {e}")
            return

    # Select recipe + PSD
    rec = top5[top5["id"] == rec_id].iloc[0]
    d50_r = float(rec.get("d50_um",  d50_um))
    layer_r = float(rec.get("layer_um", layer_um))
    n_psd = 7000 if fast else 10000
    diam_um = sample_psd_um(n_psd, d50_r, None, None, seed=9991)
    diam_units = diam_um * um2unit  # convert µm → mesh unit

    # ----------------- Layer selection -----------------
    if mesh is not None:
        v, f, mhash = _mesh_to_arrays(mesh)
        if v is None:
            st.error("Mesh arrays could not be created.")
            return
        minz, maxz = float(np.min(v[:, 2])), float(np.max(v[:, 2]))
        thickness = layer_r * (1e-3 if stl_units == "mm" else 1e-6)
        n_layers = max(1, int((maxz - minz) / max(thickness, 1e-12)))
        st.caption(f"Layers: {n_layers} · Z span: {maxz - minz:.3f} {stl_units}")
        layer_idx = st.slider("Layer index", 1, n_layers, min(5, n_layers))
        z = minz + (layer_idx - 0.5) * thickness
    else:
        # No STL → fallback square domain
        v, f, mhash = None, None, None
        n_layers, layer_idx, z = 1, 1, 0.0

    # ----------------- Slice → FOV crop → local coords (all cached) -----------------
    polys_local_wkb = []
    render_fov = float(fov_mm)
    if mesh is not None:
        # Slice
        polys_z_wkb = slice_mesh_at_z(mhash, v, f, float(z))
        if not polys_z_wkb:
            st.warning("No section found at this Z. Try another layer.")
            return

        # Full-slice packing or view-window crop
        if pack_full:
            # auto square window around polygon union bounds
            try:
                from shapely import wkb as _shp_wkb
                from shapely.ops import unary_union
                polys = [_shp_wkb.loads(p) for p in polys_z_wkb]
                dom = unary_union(polys)
                xmin, ymin, xmax, ymax = dom.bounds
                fov = max(xmax - xmin, ymax - ymin)
                win = box(xmin, ymin, xmin + fov, ymin + fov)
                clip = dom.intersection(win)
                polys_local_wkb = [clip.wkb] if clip.area > 0 else []
                render_fov = fov
            except Exception:
                polys_local_wkb = polys_z_wkb[:]  # fallback: use original slice in world coords
                # Choose a visible FOV if not pack_full crop worked
                try:
                    from shapely import wkb as _shp_wkb
                    dom = unary_union([_shp_wkb.loads(p) for p in polys_local_wkb])
                    xmin, ymin, xmax, ymax = dom.bounds
                    render_fov = max(xmax - xmin, ymax - ymin)
                except Exception:
                    render_fov = float(fov_mm)
        else:
            # center crop to user FOV
            cropped_wkb, origin = crop_to_fov(tuple(polys_z_wkb), float(fov_mm))
            polys_local_wkb = to_local_coords(tuple(cropped_wkb), origin)
            render_fov = float(fov_mm)
    else:
        # No mesh → centered square (1.8 mm default)
        try:
            from shapely.geometry import box as _box
            polys_local_wkb = [_box(0, 0, 1.8, 1.8).wkb]
            render_fov = 1.8
        except Exception:
            polys_local_wkb = []
            render_fov = float(fov_mm)

    # ----------------- Pack particles (fresh each layer; deterministic seed) -----------------
    centers, radii, phi2D = pack_particles_no_cache(
        polys_local_wkb, diam_units, float(phi2D_target),
        max_particles=int(cap),
        layer_idx=int(layer_idx)
    )

    # Render panels
    rec_sat = float(rec.get("saturation_pct", 85.0))
    rec_binder = str(rec.get("binder_type", "water_based"))
    binder_hex = binder_color(rec_binder)

    # Raster → pore mask → voids (cached)
    if centers.size and radii.size:
        centers_hash = hash((centers.tobytes(), radii.tobytes(), int(render_fov)))
        solid_mask = raster_particle_mask(centers_hash, centers, radii, float(render_fov), px=900)
        pore_mask = ~solid_mask
    else:
        pore_mask = np.zeros((900, 900), dtype=bool)

    # Layout
    cA, cB = st.columns(2)
    with cA:
        figA, axA = plt.subplots(figsize=(5.1, 5.1), dpi=185)
        render_particles_only(axA, centers, radii, float(render_fov))
        axA.set_title("Particles only", fontsize=10)
        st.pyplot(figA, use_container_width=True)

    with cB:
        figB, axB = plt.subplots(figsize=(5.1, 5.1), dpi=185)
        # seed_offset keeps void pattern stable per layer & trial choice
        seed_offset = 123 + int(layer_idx) + int(rec_sat)
        render_with_binder(axB, centers, radii, pore_mask, rec_sat/100.0, binder_hex, float(render_fov), seed_offset)
        axB.set_title(f"{rec_binder} · Sat {int(rec_sat)}%", fontsize=10)
        st.pyplot(figB, use_container_width=True)

    st.caption(
        f"FOV={render_fov:.2f} mm · φ₂D(target)≈{phi2D_target:.2f} · "
        f"φ₂D(achieved)≈{min(float(phi2D),1.0):.2f} · Porosity₂D≈{max(0.0,1.0-float(phi2D)):.2f}"
    )

    # Optional: compare multiple trials visually (same particles; binder/sat vary)
    if picks:
        cols = st.columns(min(3, len(picks)))
        for i, rid in enumerate(picks[:len(cols)]):
            row = top5[top5["id"] == rid].iloc[0]
            sat = float(row.get("saturation_pct", rec_sat))
            bndr = str(row.get("binder_type", rec_binder))
            hexc = binder_color(bndr)
            figC, axC = plt.subplots(figsize=(4.9, 4.9), dpi=185)
            seed_off = 987 + int(layer_idx) + int(sat)
            render_with_binder(axC, centers, radii, pore_mask, sat/100.0, hexc, float(render_fov), seed_off)
            axC.set_title(f'{row["id"]}: {bndr} · Sat {int(sat)}%', fontsize=9)
            with cols[i]:
                st.pyplot(figC, use_container_width=True)
