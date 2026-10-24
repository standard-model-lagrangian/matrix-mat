"""
Material-level aggregation and statistical summaries (all-data vs clean-only).

Inference is at gel level: per-field planar densities are first averaged within each gel, and each
material is compared with the control by an exact (or Monte Carlo) gel-label permutation test,
Holm-adjusted across materials. Field-level means/SDs are reported as descriptive statistics only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from matrix_mat.core.stats import adjust_pvalues, cluster_permutation_test

logger = logging.getLogger("matrix_mat.reporting.summaries")

DENSITY_COL = "nuclei_per_mm2"


def _mean(vals: np.ndarray) -> float:
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def _sd(vals: np.ndarray) -> float:
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def _stars(p: float) -> str:
    if np.isnan(p):
        return "untested"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def compute_summary_by_material(
    per_image_rows: List[Dict[str, Any]],
    output_path: Path,
    control_material: str = "Mat",
    n_perm: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute per-material summary metrics:
    - n_images_total, n_images_clean, n_gels, flagged_image_rate
    - field-level mean ± SD planar density (nuclei/mm2) for all and clean fields (descriptive)
    - gel-level mean ± SD planar density (the inferential unit)
    - gel-level permutation p-value vs control, Holm-adjusted across materials
    - median nuclear volume, mean spheroid volume
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_image_rows)
    if df.empty:
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_path, index=False)
        return empty_df
    if DENSITY_COL not in df.columns:
        raise ValueError(
            f"per-image rows have no '{DENSITY_COL}' column; outputs from pre-0.2 runs report a pseudo-volumetric "
            "density in different units and must be regenerated rather than mixed in"
        )

    df["is_clean"] = df["flag_summary"] == "clean"
    has_gels = "replicate" in df.columns

    gel_means = (
        df.dropna(subset=[DENSITY_COL]).groupby(["material", "replicate"])[DENSITY_COL].mean().reset_index()
        if has_gels else pd.DataFrame(columns=["material", "replicate", DENSITY_COL])
    )

    materials = sorted(df["material"].unique())
    summary_rows = []
    raw_p: Dict[str, float] = {}
    min_p: Dict[str, float] = {}

    for mat in materials:
        m_df = df[df["material"] == mat]
        clean_df = m_df[m_df["is_clean"]]
        n_total = len(m_df)
        n_clean = len(clean_df)
        n_gels = int(m_df["replicate"].nunique()) if has_gels else 0
        flag_rate = (n_total - n_clean) / float(n_total) if n_total > 0 else float("nan")

        all_d = m_df[DENSITY_COL].dropna().values
        clean_d = clean_df[DENSITY_COL].dropna().values
        gel_d = gel_means[gel_means["material"] == mat][DENSITY_COL].values

        sph_area = m_df["spheroid_area_um2"].dropna().values if "spheroid_area_um2" in m_df.columns else np.array([])
        nuc_vol = m_df["median_nuclear_volume_um3"].dropna().values if "median_nuclear_volume_um3" in m_df.columns else np.array([])

        note = ""
        if mat == control_material:
            note = "control"
        elif n_gels < 2:
            note = f"insufficient replicates for statistical test (n_gels={n_gels})"
        elif has_gels and control_material in materials:
            pair_df = df[df["material"].isin([control_material, mat])].dropna(subset=[DENSITY_COL])
            p, mp = cluster_permutation_test(
                pair_df, DENSITY_COL, "material", cluster_col="replicate",
                ref_group=control_material, target_group=mat, n_perm=n_perm, seed=random_seed,
            )
            raw_p[mat], min_p[mat] = float(p), float(mp)

        summary_rows.append({
            "material": mat,
            "n_images_total": n_total,
            "n_images_clean": n_clean,
            "n_gels": n_gels,
            "flagged_image_rate": round(flag_rate, 3),
            "mean_nuclei_per_mm2_all": _mean(all_d),
            "sd_nuclei_per_mm2_all": _sd(all_d),
            "mean_nuclei_per_mm2_clean": _mean(clean_d),
            "sd_nuclei_per_mm2_clean": _sd(clean_d),
            "gel_mean_nuclei_per_mm2": _mean(gel_d),
            "gel_sd_nuclei_per_mm2": _sd(gel_d),
            "median_nuclear_volume_um3": float(np.median(nuc_vol)) if len(nuc_vol) else float("nan"),
            "mean_spheroid_area_um2": _mean(sph_area),
            "statistical_note": note,
        })

    tested = list(raw_p.keys())
    adj = dict(zip(tested, adjust_pvalues([raw_p[m] for m in tested], method="holm"))) if tested else {}
    for row in summary_rows:
        mat = row["material"]
        row["p_raw"] = raw_p.get(mat, float("nan"))
        row["p_adj"] = float(adj.get(mat, float("nan")))
        row["p_adj_method"] = "holm"
        row["min_achievable_p"] = min_p.get(mat, float("nan"))
        row["significance"] = "control" if mat == control_material else _stars(row["p_adj"])

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(output_path, index=False)
    logger.info(f"Saved gel-level biological summary ({len(sum_df)} materials) to {output_path}")
    return sum_df


__all__ = [
    "compute_summary_by_material",
]
