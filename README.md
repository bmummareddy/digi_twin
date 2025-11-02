# Physics-Guided Binder Jet AM Studio (BJAM)

Physics-guided, few-shot recommendations for **Binder Jet Additive Manufacturing (BJAM)** — with an STL-aware Digital Twin for qualitative layer packing.

**Live app:** https://bjampredictions.streamlit.app/

**What it does**
- Given a material and **D50** (µm), the app suggests:
  - **Binder family/type** (water-based vs solvent-based)
  - **Binder saturation (%)**
  - **Roller traverse speed (mm/s)**
  - **Layer thickness (µm)**
- Visual diagnostics to assess feasibility of **≥90% theoretical density (%TD)** at green (and an illustrative post-sinter view).

**Tagline:** Physics-guided • STL-aware • Practical UX • Predicts process parameters from powder size and binder type

---

## Why this app?

BJAM tuning is multi-factor and often data-sparse. This tool blends:
- **Physics-guided baselines** (packing intuition; Washburn-style infiltration),
- **Few-shot learning** (leveraging small datasets), and
- **Guardrails** (stable, empirically sane windows)

…to propose sensible starting points and visualize trade-offs before you waste powder or time.

---

## Key features

- **One-click recommendations** targeting a **green %TD** (default 90%)
- **Guardrails toggle**
  - **ON (recommended):** narrow, stable windows + conservative clipping
  - **OFF:** explore wider ranges (still bounded within 0–100% TD)
- **Visual diagnostics**
  - **Heatmap:** predicted %TD vs speed × saturation with ~90% contour
  - **Saturation sensitivity:** q50 curve with q10–q90 band
  - **Packing slice (qualitative):** polydisperse circles + pixelated binder/void fill
  - **Pareto view:** binder vs density trade-off
  - **Formulae** and quick references
- **Binder family suggestion** by material class (override anytime)
- **Digital Twin (Beta)**
  - Upload an **STL** (units: mm/m) or use a built-in 10 mm cube
  - Slice at a chosen **layer thickness**; pack particles inside the actual cross-section
  - **Full-part auto FOV** or **centered manual FOV**
  - Compare multiple Top-5 trials on the same slice (binder/saturation differences)

> Note: The Digital Twin is **qualitative** (fast packing + heuristic infiltration sketch). It’s ideal for intuition and side-by-side comparisons, not yet a calibrated predictor of local porosity or sintering distortion.

---

## Repository layout

```
.
├─ streamlit_app.py        # Main UI & recommender tabs
├─ digital_twin.py         # STL→slice→packing→visuals (imported by the app)
├─ shared.py               # Data I/O, guardrails, model train/predict utils
├─ BJAM_All_Deep_Fill_v9.csv   # Dataset (source of truth for training/inference)
├─ requirements.txt
└─ README.md
```

---

## Quick start

### A) Use the hosted app
Open: https://bjampredictions.streamlit.app/

### B) Run locally
```bash
# 1) Clone
git clone https://github.com/<your-username>/BJAM.git
cd BJAM

# 2) (Recommended) Python 3.11 virtual env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 3) Install deps
pip install -r requirements.txt

# 4) Launch
streamlit run streamlit_app.py
```

---

## Requirements

Minimal `requirements.txt` (pin versions on Streamlit Cloud if builds fail):
```txt
streamlit
numpy
pandas
matplotlib
plotly
trimesh
shapely
scipy
rtree
Pillow
```

Tips
- `rtree` improves Shapely performance; some Linux hosts need `libspatialindex` via OS packages.
- If cloud builds are flaky, pin: `shapely==2.0.*`, `trimesh==4.*`, `rtree==1.2.*`.

---

## Using the app (typical flow)

1) **Inputs (sidebar)**  
   Choose material (from dataset or custom), set **D50 (µm)**, **layer thickness (µm)**, and **target green %TD**. Keep **Guardrails** ON.

2) **Recommend**  
   Click “Recommend” to get Top-K parameter sets with **q10/q50/q90** predicted density. Download CSV if needed.

3) **Explore tabs**
   - **Heatmap:** scan the landscape; use the 90% contour to anchor trials.
   - **Sensitivity:** see how saturation alone shifts q10/q50/q90 at a representative speed.
   - **Packing (slice):** view polydisperse circles + binder/void raster (qualitative).
   - **Pareto:** pick a balanced trial across binder vs density.

4) **Digital Twin (Beta)**
   - Upload an STL or use the built-in cube; select a layer index.
   - Toggle **Full-part auto FOV** or set a **manual FOV**.
   - Set a particle cap (**Fast mode** improves responsiveness).
   - Pick a Top-5 trial; optionally compare multiple trials on the same slice.

---

## Troubleshooting

- **“Bad message format / Tried to use SessionInfo before it was initialized”**  
  Ensure `digital_twin.py` has **no** Streamlit calls at import time. It should only define functions and be invoked from inside a tab in `streamlit_app.py`. Clear the Streamlit cache and redeploy.

- **`ModuleNotFoundError: trimesh/shapely/rtree`**  
  Add to `requirements.txt`, clear cache, redeploy.

- **STL slice looks cropped**  
  Use **Full-part auto FOV** to cover the entire cross-section; the packing is clipped to part outlines.

- **Particles look identical**  
  Make sure you’re on the latest code; the packing uses a lognormal PSD around D50 for visible polydispersity.

- **Slow rendering**  
  Turn on **Fast mode**, reduce particle cap, or test with a smaller manual FOV.

---

## Data expectations

`BJAM_All_Deep_Fill_v9.csv` typically includes:
- `material`, `material_class` (metal/oxide/carbide/other)  
- `d50_um`, `binder_saturation_pct`, `roller_speed_mm_s`, `layer_thickness_um`  
- `green_pct_td` (for training/validation)

Sparse data is fine: the app falls back to physics-guided priors (few-shot behavior).


## License & contact
 
Maintainer: **Bhargavi Mummareddy** — mummareddybhargavi@gmail.com

