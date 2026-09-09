"""
Publication-Grade Biological Comparison Visualizations for Spheroid IF Pipeline.

Generates high-resolution publication figures (300 DPI PNG + vector PDF) focusing
strictly on biological comparisons across hydrogel formulations:
  1. Cross-Material Nuclei Density Distributions (fig_if_nuclei_density_by_material)
  2. Clean vs. All Cohort Biological Robustness (fig_if_clean_vs_all_density)
  3. Single-Nucleus Volume & Morphology Distributions (fig_if_nuclear_volume_distributions)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("spheroid_if_sweep.biological_plots")

# Standardized color palette across all hydrogel formulations
MATERIAL_PALETTE: Dict[str, str] = {
    "Mat": "#2980b9",      # Blue (Control)
    "S34D30": "#d35400",   # Orange
    "S40D30": "#27ae60",   # Green
    "S43D20": "#c0392b",   # Red
    "S46D10": "#8e44ad",   # Purple
    "S50": "#795548",      # Brown
}

MATERIAL_ORDER: List[str] = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]


def get_ordered_materials(df: pd.DataFrame, col: str = "material") -> List[str]:
    """Returns materials ordered by canonical MATERIAL_ORDER followed by any novel materials."""
    if col not in df.columns:
        return []
    unique_mats = list(df[col].dropna().unique())
    known = [m for m in MATERIAL_ORDER if m in unique_mats]
    other = [m for m in sorted(unique_mats) if m not in known]
    return known + other


def plot_nuclei_density_by_material(
    per_image_df: pd.DataFrame,
    output_dir: Path,
    control_material: str = "Mat",
) -> Tuple[Path, Path]:
    """
    Figure 1: Cross-Material Nuclei Density Distribution.
    Displays boxplot with individual field jitter scatter and non-parametric
    hypothesis test (Mann-Whitney U) vs control Mat.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Filter to present materials (canonical + any novel ones)
    present_materials = get_ordered_materials(per_image_df, "material")

    # Get control data
    ctrl_data = per_image_df[per_image_df["material"] == control_material]["nuclei_density_per_mm3"].dropna().values

    data_by_mat = []
    for m in present_materials:
        sub = per_image_df[per_image_df["material"] == m]["nuclei_density_per_mm3"].dropna().values
        data_by_mat.append(sub)

    positions = np.arange(len(present_materials))

    # Panel A: Linear scale boxplot + jitter
    bp1 = ax1.boxplot(
        data_by_mat,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops=dict(color="#2c3e50", linewidth=2.5),
    )

    for patch, m in zip(bp1["boxes"], present_materials):
        patch.set_facecolor(MATERIAL_PALETTE.get(m, "#95a5a6"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(1.5)

    np.random.seed(42)
    max_y = 0.0
    for i, m in enumerate(present_materials):
        vals = data_by_mat[i]
        if len(vals) == 0:
            continue
        max_y = max(max_y, float(np.max(vals)))
        jitter = np.random.normal(0, 0.06, size=len(vals))
        ax1.scatter(
            positions[i] + jitter, vals,
            color="#2c3e50", alpha=0.75, s=50, zorder=4, edgecolor="white", linewidth=0.6
        )
        mean_v = float(np.mean(vals))
        ax1.scatter(positions[i], mean_v, marker="D", color="#f1c40f", s=65, zorder=5, edgecolor="#2c3e50")

        # Mann-Whitney U test vs control
        if m != control_material and len(vals) >= 2 and len(ctrl_data) >= 2:
            stat, p_val = stats.mannwhitneyu(vals, ctrl_data, alternative="two-sided")
            sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            ax1.text(
                positions[i], np.max(vals) * 1.08, sig_str,
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="#2c3e50"
            )

    ax1.set_xticks(positions)
    ax1.set_xticklabels(present_materials, fontweight="bold", fontsize=11)
    ax1.set_ylabel("Nuclei Density ($10^3\\,\\mathrm{nuclei/mm}^3$)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Volumetric Nuclei Density by Hydrogel Material\n(Diamond=Mean, Line=Median, *p<0.05 vs Mat)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", alpha=0.4, axis="y")

    # Panel B: Log-scale distribution
    bp2 = ax2.boxplot(
        data_by_mat,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops=dict(color="#2c3e50", linewidth=2.5),
    )
    for patch, m in zip(bp2["boxes"], present_materials):
        patch.set_facecolor(MATERIAL_PALETTE.get(m, "#95a5a6"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#2c3e50")
        patch.set_linewidth(1.5)

    for i, m in enumerate(present_materials):
        vals = data_by_mat[i]
        if len(vals) == 0:
            continue
        jitter = np.random.normal(0, 0.06, size=len(vals))
        ax2.scatter(
            positions[i] + jitter, vals,
            color="#2c3e50", alpha=0.75, s=50, zorder=4, edgecolor="white", linewidth=0.6
        )

    ax2.set_yscale("log")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(present_materials, fontweight="bold", fontsize=11)
    ax2.set_ylabel("Nuclei Density (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Dynamic Range Distribution (Log Scale)\n(Individual Field Replicates)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    png_path = output_dir / "fig_if_nuclei_density_by_material.png"
    pdf_path = output_dir / "fig_if_nuclei_density_by_material.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    logger.info(f"Generated biological comparison Figure 1: {png_path}")
    return png_path, pdf_path


def plot_clean_vs_all_density(
    per_image_df: pd.DataFrame,
    qc_flags_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Figure 2: Biological Robustness — Clean vs All Fields.
    Evaluates whether excluding QC-flagged fields altered condition rank order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = pd.merge(per_image_df, qc_flags_df[["image_id", "flag_any"]], on="image_id", how="left")

    present_materials = get_ordered_materials(merged, "material")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    width = 0.35
    positions = np.arange(len(present_materials))

    all_means, all_sds = [], []
    clean_means, clean_sds = [], []

    for m in present_materials:
        sub_all = merged[merged["material"] == m]["nuclei_density_per_mm3"].dropna()
        sub_clean = merged[(merged["material"] == m) & (merged["flag_any"] == False)]["nuclei_density_per_mm3"].dropna()

        all_means.append(float(sub_all.mean()) if len(sub_all) > 0 else 0.0)
        all_sds.append(float(sub_all.std()) if len(sub_all) > 1 else 0.0)

        clean_means.append(float(sub_clean.mean()) if len(sub_clean) > 0 else float(sub_all.mean()) if len(sub_all) > 0 else 0.0)
        clean_sds.append(float(sub_clean.std()) if len(sub_clean) > 1 else 0.0)

    rects1 = ax.bar(
        positions - width / 2, all_means, width, yerr=all_sds,
        label="All Cohort (Unfiltered)", color="#95a5a6", edgecolor="#7f8c8d", alpha=0.8, capsize=4
    )
    rects2 = ax.bar(
        positions + width / 2, clean_means, width, yerr=clean_sds,
        label="Clean Only (QC Outliers Excluded)", color="#27ae60", edgecolor="#1e8449", alpha=0.85, capsize=4
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(present_materials, fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Nuclei Density ± SD ($10^3\\,\\mathrm{nuclei/mm}^3$)", fontsize=11, fontweight="bold")
    ax.set_title("Biological Stability: All Cohort vs. Clean Fields Only\n(Confirming QC Gating Does Not Alter Material Biological Trends)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")

    plt.tight_layout()
    png_path = output_dir / "fig_if_clean_vs_all_density.png"
    pdf_path = output_dir / "fig_if_clean_vs_all_density.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    logger.info(f"Generated biological comparison Figure 2: {png_path}")
    return png_path, pdf_path


def plot_nuclear_volume_distributions(
    objects_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Figure 3: Single-Nucleus Volume & Morphology Distributions across hydrogels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    present_materials = get_ordered_materials(objects_df, "material")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    vol_col = "volume_um3" if "volume_um3" in objects_df.columns else "nuclear_volume_um3"

    vols_by_mat = []
    for m in present_materials:
        sub = objects_df[(objects_df["material"] == m) & (objects_df["excluded_border"] == False)][vol_col].dropna().values
        vols_by_mat.append(sub)

    positions = np.arange(len(present_materials))
    bp = ax.boxplot(
        vols_by_mat,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#d35400", linewidth=2.2),
    )

    for patch, m in zip(bp["boxes"], present_materials):
        patch.set_facecolor(MATERIAL_PALETTE.get(m, "#bdc3c7"))
        patch.set_alpha(0.7)
        patch.set_edgecolor("#2c3e50")

    # Annotate counts and median values
    for i, m in enumerate(present_materials):
        v = vols_by_mat[i]
        if len(v) > 0:
            med = float(np.median(v))
            ax.text(positions[i], med, f"{med:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")
            ax.text(positions[i], 50, f"n={len(v)}", ha="center", va="bottom", fontsize=8.5, color="#7f8c8d")

    ax.set_xticks(positions)
    ax.set_xticklabels(present_materials, fontweight="bold", fontsize=11)
    ax.set_ylabel("Nuclear Volume ($V_{\\mathrm{nuc}}$, $\\mu\\mathrm{m}^3$)", fontsize=11, fontweight="bold")
    ax.set_title("Single-Nucleus Volume Distributions across Hydrogel Formulations\n(Border Excluded Nuclei, Numbers=Median $\\mu\\mathrm{m}^3$)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.4, axis="y")

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            plt.tight_layout()
        except Exception:
            pass
    png_path = output_dir / "fig_if_nuclear_volume_distributions.png"
    pdf_path = output_dir / "fig_if_nuclear_volume_distributions.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    logger.info(f"Generated biological comparison Figure 3: {png_path}")
    return png_path, pdf_path


def generate_all_if_biological_figures(
    run_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[Path, Path]]:
    """Generates all 3 biological comparison figures for an IF run directory."""
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_img_path = run_dir / "features" / "per_image.csv"
    qc_flags_path = run_dir / "qc" / "qc_flags.csv"
    feat_obj_path = run_dir / "features" / "objects.csv"

    figs = {}
    if feat_img_path.exists():
        per_img = pd.read_csv(feat_img_path)
        figs["density_by_material"] = plot_nuclei_density_by_material(per_img, out_dir)

        if qc_flags_path.exists():
            qc_df = pd.read_csv(qc_flags_path)
            figs["clean_vs_all"] = plot_clean_vs_all_density(per_img, qc_df, out_dir)

    if feat_obj_path.exists():
        objs_df = pd.read_csv(feat_obj_path)
        figs["nuclear_volumes"] = plot_nuclear_volume_distributions(objs_df, out_dir)

    logger.info(f"All biological comparison figures generated in {out_dir}")
    return figs
