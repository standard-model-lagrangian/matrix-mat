"""
Material-level aggregation and statistical summaries (all-data vs clean-only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("matrix_mat.reporting.summaries")


def compute_summary_by_material(
    per_image_rows: List[Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Compute per-material summary metrics:
    - n_images_total, n_images_clean, n_gels, flagged_image_rate
    - mean ± SD planar nuclei density (nuclei_per_mm2 for all data and clean only)
    - median nuclear volume
    - mean spheroid volume
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_image_rows)
    if df.empty:
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_path, index=False)
        return empty_df

    df["is_clean"] = df["flag_summary"] == "clean"

    materials = sorted(df["material"].unique())
    summary_rows = []

    for mat in materials:
        m_df = df[df["material"] == mat]
        clean_df = m_df[m_df["is_clean"]]

        n_total = len(m_df)
        n_clean = len(clean_df)
        n_gels = int(m_df["replicate"].nunique()) if "replicate" in m_df.columns else n_total
        flag_rate = (n_total - n_clean) / float(n_total) if n_total > 0 else 0.0

        # Planar areal density (nuclei/mm2)
        if "nuclei_per_mm2" in m_df.columns:
            all_densities = m_df["nuclei_per_mm2"].dropna().values
        elif "nuclei_density_per_mm3" in m_df.columns:
            all_densities = m_df["nuclei_density_per_mm3"].dropna().values
        else:
            all_densities = np.array([])

        mean_dens_all = float(np.mean(all_densities)) if len(all_densities) > 0 else 0.0
        sd_dens_all = float(np.std(all_densities, ddof=1)) if len(all_densities) > 1 else 0.0

        # Clean-only density (nuclei/mm2)
        if "nuclei_per_mm2" in clean_df.columns:
            clean_densities = clean_df["nuclei_per_mm2"].dropna().values
        elif "nuclei_density_per_mm3" in clean_df.columns:
            clean_densities = clean_df["nuclei_density_per_mm3"].dropna().values
        else:
            clean_densities = np.array([])

        mean_dens_clean = float(np.mean(clean_densities)) if len(clean_densities) > 0 else 0.0
        sd_dens_clean = float(np.std(clean_densities, ddof=1)) if len(clean_densities) > 1 else 0.0

        # Spheroid volume and nuclear volume
        mean_sph_vol = float(np.mean(m_df["spheroid_volume_um3"].dropna().values)) if n_total > 0 and "spheroid_volume_um3" in m_df.columns and len(m_df["spheroid_volume_um3"].dropna()) > 0 else 0.0
        med_nuc_vol = float(np.median(m_df["median_nuclear_volume_um3"].dropna().values)) if n_total > 0 and "median_nuclear_volume_um3" in m_df.columns and len(m_df["median_nuclear_volume_um3"].dropna()) > 0 else 0.0

        stat_note = "" if n_gels >= 2 else "insufficient replicates for statistical test (n_gels=1)"

        summary_rows.append({
            "material": mat,
            "n_images_total": n_total,
            "n_images_clean": n_clean,
            "n_gels": n_gels,
            "flagged_image_rate": round(flag_rate, 3),
            "mean_nuclei_per_mm2_all": round(mean_dens_all, 2),
            "sd_nuclei_per_mm2_all": round(sd_dens_all, 2),
            "mean_nuclei_per_mm2_clean": round(mean_dens_clean, 2),
            "sd_nuclei_per_mm2_clean": round(sd_dens_clean, 2),
            # Backwards compatibility column aliases
            "mean_nuclei_density_all": round(mean_dens_all, 2),
            "sd_nuclei_density_all": round(sd_dens_all, 2),
            "mean_nuclei_density_clean": round(mean_dens_clean, 2),
            "sd_nuclei_density_clean": round(sd_dens_clean, 2),
            "median_nuclear_volume_um3": round(med_nuc_vol, 2),
            "mean_spheroid_volume_um3": round(mean_sph_vol, 1),
            "statistical_note": stat_note,
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(output_path, index=False)
    logger.info(f"Saved primary biological summary ({len(sum_df)} materials) to {output_path}")
    return sum_df


__all__ = [
    "compute_summary_by_material",
]
