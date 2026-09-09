"""
Material-level aggregation and statistical summaries (all-data vs clean-only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger("spheroid_if_sweep.summaries")


def compute_summary_by_material(
    per_image_rows: List[Dict[str, Any]],
    output_path: Path,
) -> pd.DataFrame:
    """
    Compute per-material summary metrics:
    - n_images_total, n_images_clean, flagged_image_rate
    - mean ± SD nuclei density (all data and clean only)
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
        flag_rate = (n_total - n_clean) / float(n_total) if n_total > 0 else 0.0

        # All-data density
        all_densities = m_df["nuclei_density_per_mm3"].values
        mean_dens_all = float(np.mean(all_densities)) if len(all_densities) > 0 else 0.0
        sd_dens_all = float(np.std(all_densities, ddof=1)) if len(all_densities) > 1 else 0.0

        # Clean-only density
        clean_densities = clean_df["nuclei_density_per_mm3"].values
        mean_dens_clean = float(np.mean(clean_densities)) if len(clean_densities) > 0 else 0.0
        sd_dens_clean = float(np.std(clean_densities, ddof=1)) if len(clean_densities) > 1 else 0.0

        # Spheroid volume and nuclear volume
        mean_sph_vol = float(np.mean(m_df["spheroid_volume_um3"].values)) if n_total > 0 else 0.0
        med_nuc_vol = float(np.median(m_df["median_nuclear_volume_um3"].values)) if n_total > 0 else 0.0

        summary_rows.append({
            "material": mat,
            "n_images_total": n_total,
            "n_images_clean": n_clean,
            "flagged_image_rate": round(flag_rate, 3),
            "mean_nuclei_density_all": round(mean_dens_all, 2),
            "sd_nuclei_density_all": round(sd_dens_all, 2),
            "mean_nuclei_density_clean": round(mean_dens_clean, 2),
            "sd_nuclei_density_clean": round(sd_dens_clean, 2),
            "median_nuclear_volume_um3": round(med_nuc_vol, 2),
            "mean_spheroid_volume_um3": round(mean_sph_vol, 1),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(output_path, index=False)
    logger.info(f"Saved primary biological summary ({len(sum_df)} materials) to {output_path}")
    return sum_df
