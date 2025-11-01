# data_preprocessing.py - Clean and standardize BJAM dataset
# Handles the specific format of BJAM_All_Deep_Fill_v9.csv

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def extract_numeric_from_text(text: str) -> Optional[float]:
    """
    Extract numeric value from text with units or ranges.
    Handles formats like: "35 µm", "0.85-1.45 µm", "100%", "~30%", ">60%", etc.
    For ranges, returns the midpoint.
    """
    if pd.isna(text) or text == "":
        return None
    
    text = str(text).strip()
    
    # Handle "Not specified" or similar
    if text.lower() in ["not specified", "nan", "n/a", "na", ""]:
        return None
    
    # Remove common units and symbols
    text = text.replace("µm", "").replace("μm", "").replace("mm/s", "").replace("mm/min", "")
    text = text.replace("%", "").replace("°C", "").replace(" ", "")
    
    # Handle comparison operators
    text = text.replace("~", "").replace(">", "").replace("<", "").replace("≥", "").replace("≤", "")
    
    # Try to extract number(s)
    # Handle ranges like "0.85-1.45" or "15-45"
    range_match = re.search(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', text)
    if range_match:
        val1 = float(range_match.group(1))
        val2 = float(range_match.group(2))
        return (val1 + val2) / 2.0  # Return midpoint
    
    # Single number
    number_match = re.search(r'(\d+\.?\d*)', text)
    if number_match:
        return float(number_match.group(1))
    
    return None


def extract_density_and_state(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract density value and state (green/sintered) from text.
    Examples:
        "65.2% (Sintered)" -> (65.2, "sintered")
        ">60% (Green)" -> (60.0, "green")
        "Green: 58%, Sintered: 96%" -> (58.0, "green") [prioritizes green]
    """
    if pd.isna(text) or text == "":
        return None, None
    
    text = str(text).strip()
    
    if text.lower() in ["not specified", "nan", "n/a", "na", ""]:
        return None, None
    
    # Try to extract green density specifically if both are present
    green_match = re.search(r'green[:\s=]*([~<>≥≤]?\s*\d+\.?\d*)\s*%', text, re.IGNORECASE)
    if green_match:
        density = extract_numeric_from_text(green_match.group(1))
        return density, "green"
    
    # Check for state indicators
    state = None
    if "green" in text.lower():
        state = "green"
    elif "sintered" in text.lower():
        state = "sintered"
    
    # Extract any numeric value
    density = extract_numeric_from_text(text)
    
    return density, state


def convert_speed_to_mm_s(text: str) -> Optional[float]:
    """
    Convert speed to mm/s regardless of input unit.
    Handles: "2 mm/s", "20 mm/min", etc.
    """
    if pd.isna(text) or text == "":
        return None
    
    text = str(text).strip().lower()
    
    if "not specified" in text:
        return None
    
    # Extract numeric value
    num = extract_numeric_from_text(text)
    if num is None:
        return None
    
    # Convert based on unit
    if "mm/min" in text or "min" in text:
        return num / 60.0  # Convert mm/min to mm/s
    elif "mm/s" in text or "s" in text:
        return num
    else:
        # Assume mm/s if no unit specified but value seems reasonable for mm/s
        return num if 0.1 <= num <= 10 else None


def infer_material_class(material: str) -> str:
    """
    Infer material class from material name.
    """
    if pd.isna(material):
        return "other"
    
    material = str(material).lower()
    
    # Metals
    metal_keywords = [
        "steel", "stainless", "316l", "420", "17-4", 
        "titanium", "ti-6al-4v", "aluminum", "copper", "iron",
        "inconel", "nickel", "alloy", "cobalt", "chromium",
        "tungsten", "magnesium", "neodymium", "manganese"
    ]
    if any(kw in material for kw in metal_keywords):
        return "metal"
    
    # Ceramics/Oxides
    ceramic_keywords = [
        "alumina", "al2o3", "al₂o₃", "zirconia", "zro2", "oxide",
        "barium titanate", "batio", "hydroxyapatite", "calcium",
        "porcelain", "ceramic", "spinel", "gypsum"
    ]
    if any(kw in material for kw in ceramic_keywords):
        return "ceramic"
    
    # Carbides
    carbide_keywords = [
        "carbide", "wc-co", "sic", "silicon carbide", "tungsten carbide",
        "silicon nitride"
    ]
    if any(kw in material for kw in carbide_keywords):
        return "carbide"
    
    # Polymers
    polymer_keywords = [
        "nylon", "polymer", "graphite/nylon", "silk"
    ]
    if any(kw in material for kw in polymer_keywords):
        return "polymer"
    
    return "other"


def normalize_binder_type(binder: str) -> str:
    """
    Normalize binder type to standard categories.
    """
    if pd.isna(binder):
        return "water_based"  # Default assumption
    
    binder = str(binder).lower()
    
    if "water" in binder or "aqueous" in binder:
        return "water_based"
    elif "polymer" in binder or "uv" in binder:
        return "polymer_based"
    elif "solvent" in binder:
        return "solvent_based"
    else:
        return "water_based"  # Default


def clean_bjam_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the BJAM_All_Deep_Fill_v9.csv dataset.
    
    Args:
        df: Raw dataframe from CSV
        
    Returns:
        Cleaned dataframe with standardized columns
    """
    # The dataset has TWO formats mixed:
    # Format 1: Columns 1-8 (Unnamed: 1 through Unnamed: 8) for rows 3-78
    # Format 2: Columns 9-16 for rows 79+
    
    data_rows = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Try Format 1 first (columns 1-8)
        material = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None
        
        # Check if this looks like actual data (not a header)
        if material and str(material).lower() not in ['material name', 'paper', 'nan', '']:
            particle_size = row.iloc[2] if len(row) > 2 else None
            binder_sat = row.iloc[3] if len(row) > 3 else None
            binder_type = row.iloc[4] if len(row) > 4 else None
            layer_thick = row.iloc[5] if len(row) > 5 else None
            speed = row.iloc[6] if len(row) > 6 else None
            packing = row.iloc[7] if len(row) > 7 else None
            density = row.iloc[8] if len(row) > 8 else None
            
            data_rows.append({
                "material": material,
                "particle_size_raw": particle_size,
                "binder_saturation_raw": binder_sat,
                "binder_type_raw": binder_type,
                "layer_thickness_raw": layer_thick,
                "speed_raw": speed,
                "packing_density_raw": packing,
                "final_density_raw": density,
            })
            continue
        
        # Try Format 2 (columns 9-16)
        if len(row) > 9:
            material_alt = row.iloc[9] if pd.notna(row.iloc[9]) else None
            if material_alt and str(material_alt) not in ['', 'nan']:
                particle_size = row.iloc[10] if len(row) > 10 else None
                binder_sat = row.iloc[11] if len(row) > 11 else None
                binder_type = row.iloc[12] if len(row) > 12 else None
                layer_thick = row.iloc[13] if len(row) > 13 else None
                speed = row.iloc[14] if len(row) > 14 else None
                packing = row.iloc[15] if len(row) > 15 else None
                density = row.iloc[16] if len(row) > 16 else None
                
                data_rows.append({
                    "material": material_alt,
                    "particle_size_raw": particle_size,
                    "binder_saturation_raw": binder_sat,
                    "binder_type_raw": binder_type,
                    "layer_thickness_raw": layer_thick,
                    "speed_raw": speed,
                    "packing_density_raw": packing,
                    "final_density_raw": density,
                })
    
    if not data_rows:
        return pd.DataFrame()
    
    df_work = pd.DataFrame(data_rows)
    
    # Process each column
    df_work["d50_um"] = df_work["particle_size_raw"].apply(extract_numeric_from_text)
    df_work["binder_saturation_pct"] = df_work["binder_saturation_raw"].apply(extract_numeric_from_text)
    df_work["layer_thickness_um"] = df_work["layer_thickness_raw"].apply(extract_numeric_from_text)
    df_work["roller_speed_mm_s"] = df_work["speed_raw"].apply(convert_speed_to_mm_s)
    
    # Extract density and state from both packing and final density columns
    density_state_final = df_work["final_density_raw"].apply(extract_density_and_state)
    density_state_packing = df_work["packing_density_raw"].apply(extract_density_and_state)
    
    df_work["final_density_pct"] = density_state_final.apply(lambda x: x[0])
    df_work["final_density_state"] = density_state_final.apply(lambda x: x[1])
    
    # If final_density is NaN but packing density exists, use that as green density
    mask = df_work["final_density_pct"].isna() & df_work["packing_density_raw"].notna()
    if mask.any():
        packing_vals = density_state_packing[mask].apply(lambda x: x[0])
        df_work.loc[mask, "final_density_pct"] = packing_vals
        df_work.loc[mask, "final_density_state"] = "green"
    
    # Material class
    df_work["material_class"] = df_work["material"].apply(infer_material_class)
    
    # Binder type
    df_work["binder_type_rec"] = df_work["binder_type_raw"].apply(normalize_binder_type)
    
    # Keep only standardized columns
    output_cols = [
        "material", "material_class", "d50_um", "layer_thickness_um",
        "roller_speed_mm_s", "binder_saturation_pct", "binder_type_rec",
        "final_density_pct", "final_density_state"
    ]
    
    result = df_work[output_cols].copy()
    
    # Remove rows with missing critical values
    # At minimum we need: material, d50, binder_saturation, and some density measurement
    result = result[result["material"].notna()].copy()
    result = result[result["d50_um"].notna()].copy()
    result = result[result["binder_saturation_pct"].notna()].copy()
    
    # Ensure numeric columns are float
    numeric_cols = ["d50_um", "layer_thickness_um", "roller_speed_mm_s", 
                    "binder_saturation_pct", "final_density_pct"]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    
    # Fill missing layer thickness with 4*D50 heuristic
    mask = result["layer_thickness_um"].isna() & result["d50_um"].notna()
    if mask.any():
        result.loc[mask, "layer_thickness_um"] = result.loc[mask, "d50_um"] * 4.0
    
    # Fill missing speed with 1.6 mm/s heuristic
    result.loc[result["roller_speed_mm_s"].isna(), "roller_speed_mm_s"] = 1.6
    
    return result.reset_index(drop=True)


def load_and_clean_bjam_data(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load and clean the BJAM dataset from file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        (cleaned_df, metadata_dict)
    """
    try:
        # Read the raw CSV
        df_raw = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            df_raw = pd.read_csv(filepath, encoding='latin-1')
        except:
            df_raw = pd.read_csv(filepath, encoding='cp1252')
    
    # Clean the data
    df_clean = clean_bjam_dataset(df_raw)
    
    # Generate metadata
    metadata = {
        "source_file": str(Path(filepath).name),
        "raw_rows": len(df_raw),
        "clean_rows": len(df_clean),
        "materials": sorted(df_clean["material"].unique().tolist()) if len(df_clean) > 0 else [],
        "n_materials": df_clean["material"].nunique() if len(df_clean) > 0 else 0,
        "green_density_samples": int(len(df_clean[df_clean["final_density_state"] == "green"])) if len(df_clean) > 0 else 0,
        "sintered_density_samples": int(len(df_clean[df_clean["final_density_state"] == "sintered"])) if len(df_clean) > 0 else 0,
        "d50_range": (
            float(df_clean["d50_um"].min()), 
            float(df_clean["d50_um"].max())
        ) if len(df_clean) > 0 and df_clean["d50_um"].notna().any() else (None, None),
    }
    
    return df_clean, metadata
