# BJAM — Binder-Jet AM Parameter Recommender

Physics-guided, few-shot recommendations for **Binder Jet Additive Manufacturing (BJAM)**.

Given a material powder and **D50** (µm), the app suggests:
- **Binder family/type** (e.g., solvent-based vs water-based)  
- **Binder saturation (%)**  
- **Roller traverse speed (mm/s)**  
- **Layer thickness (µm)**  
- Diagnostics to assess if you can hit **≥ 90% theoretical density (%TD)** at green and after sintering (illustrative)

**Live app:** https://bjampredictions.streamlit.app/

---

## Why this app?

Tuning BJAM is multi-factor and data-sparse. This tool blends:
- a **physics-guided baseline** (random-close-packing intuition, Washburn-type flow),
- **few-shot learning** (when only a handful of runs exist), and
- **guardrails** (keep settings in empirically stable windows)

to propose sensible starting points and visualize trade-offs.

---

## Features

- **One-click recommendations** targeting a **green %TD** (default 90%)
- **Guardrails toggle**
  - **ON (recommended):** narrow, stable input windows + conservative clipping of predictions
  - **OFF:** explore wide ranges (still physically bounded 0–100% TD)
- **Visual diagnostics** (all-in-one expander):
  - **Process window**: layer thickness vs **3–5×D50** band
  - **Saturation sensitivity**: median + uncertainty band (q10–q90)
  - **Speed × Saturation heatmap** with ≈90% TD contour
  - **Few-shot uplift**: proxy vs refined model (q10/q50/q90)
  - **Qualitative packing slice** at target φ (~90%)
  - **Sintered density vs temperature** (illustrative S-curve)
  - **Pareto frontier** (binder vs density)
  - **Local importance** (which inputs move density most)
  - **Residuals/outliers** (once you upload measured runs)
- **Binder family suggestion** by material class (override as needed)
- Bright, engaging UI (Ivory/Soft palette) for demos and stakeholder buy-in

---

## Quick Start

### A) Run the hosted app (no install)
Open **https://bjampredictions.streamlit.app/**

### B) Run locally
```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/BJAM.git
cd BJAM

# 2) (Recommended) Python 3.11 environment
#    Any venv tool is fine; then install deps:
pip install -r requirements.txt

# 3) Launch
streamlit run streamlit_app.py
