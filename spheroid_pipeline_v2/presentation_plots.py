"""
Integrated Presentation Figures & Well Growth Analysis Module for Spheroid Pipeline v2.

Provides parameterized, production-ready generation of:
  - well_growth_comparison.csv: Well-replicate level pooling (biomass, count density, hypertrophy).
  - Figure 1: Primary Biomass & Proliferation Dashboard (presentation_fig1_primary_biomass_dashboard).
  - Figure 2: Hypertrophy vs Hyperplasia Decomposition (presentation_fig2_hypertrophy_vs_hyperplasia).
  - Figure 3: Per-Spheroid Volume Distributions (presentation_fig3_per_spheroid_volume_distributions).
  - Figure 4: Pipeline QC Yield & Forensic Exclusion Breakdown (presentation_fig4_qc_pipeline_yield).
  - Figure 5: Hungarian-Tracked Paired Trajectories (presentation_fig5_paired_trajectories).
  - Figure 6: Raw Total Spheroid Volume with Anti-Collision Labels (presentation_fig6_raw_well_volumes).
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("spheroid_pipeline_v2.presentation_plots")

COND_ORDER = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]

COND_COLORS = {
    "Mat": "#2980b9",      # Blue (Control)
    "S34D30": "#d35400",   # Orange
    "S40D30": "#27ae60",   # Green
    "S43D20": "#c0392b",   # Red
    "S46D10": "#8e44ad",   # Purple
    "S50": "#795548",      # Brown
}


def get_effective_fov_counts(df_all: pd.DataFrame) -> pd.DataFrame:
    """Computes effective physical FOVs per well by clustering multi-focal redundant planes."""
    eff_counts = []
    for (c, r, tp), grp in df_all.groupby(["condition", "replicate", "timepoint"]):
        fovs = sorted(grp["fov"].unique())
        if len(fovs) <= 1:
            eff_counts.append({"condition": c, "replicate": r, "timepoint": tp, "n_fovs": len(fovs)})
            continue
        pair_matches = {}
        dups = grp[grp["qc_reasons"].str.contains("duplicate_focal_plane", na=False)]
        for _, row in dups.iterrows():
            m = re.search(r"in fov (\d+)", str(row["qc_reasons"]))
            if m:
                other_fov = int(m.group(1))
                curr_fov = int(row["fov"])
                pair = tuple(sorted([curr_fov, other_fov]))
                pair_matches[pair] = pair_matches.get(pair, 0) + 1
        adj = {f: set([f]) for f in fovs}
        for (f1, f2), count in pair_matches.items():
            n1 = len(grp[grp["fov"] == f1])
            n2 = len(grp[grp["fov"] == f2])
            min_n = min(n1, n2)
            if (min_n > 0 and count / min_n >= 0.65) or count >= 3:
                adj[f1].add(f2)
                adj[f2].add(f1)
        visited = set()
        n_comp = 0
        for f in fovs:
            if f not in visited:
                n_comp += 1
                q = [f]
                visited.add(f)
                while q:
                    curr = q.pop(0)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            q.append(nbr)
        eff_counts.append({"condition": c, "replicate": r, "timepoint": tp, "n_fovs": n_comp})
    return pd.DataFrame(eff_counts)


def compute_well_growth_comparison(
    objects_df: pd.DataFrame,
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Computes well-level growth metrics: pooled volume fold change, mean volume, and count density."""
    pass_df = objects_df[objects_df["qc_flag"] == "PASS"].copy()
    eff_fov_df = get_effective_fov_counts(objects_df)

    rows = []
    wells_list = pass_df[["condition", "replicate"]].drop_duplicates().values

    for c, r in wells_list:
        sub_t0 = pass_df[(pass_df["condition"] == c) & (pass_df["replicate"] == r) & (pass_df["timepoint"] == "t0")]
        sub_t7 = pass_df[(pass_df["condition"] == c) & (pass_df["replicate"] == r) & (pass_df["timepoint"] == "t7")]

        v0_tot = sub_t0["volume_sphere_um3"].sum()
        v7_tot = sub_t7["volume_sphere_um3"].sum()
        fc_tot = (v7_tot / v0_tot) if v0_tot > 0 else np.nan

        n0 = len(sub_t0)
        n7 = len(sub_t7)
        fc_count = (n7 / n0) if n0 > 0 else np.nan

        v0_mean = sub_t0["volume_sphere_um3"].mean() if n0 > 0 else np.nan
        v7_mean = sub_t7["volume_sphere_um3"].mean() if n7 > 0 else np.nan
        fc_mean_vol = (v7_mean / v0_mean) if (v0_mean and v0_mean > 0) else np.nan

        fov_t0 = eff_fov_df[(eff_fov_df["condition"] == c) & (eff_fov_df["replicate"] == r) & (eff_fov_df["timepoint"] == "t0")]
        fov_t7 = eff_fov_df[(eff_fov_df["condition"] == c) & (eff_fov_df["replicate"] == r) & (eff_fov_df["timepoint"] == "t7")]

        nf0 = fov_t0["n_fovs"].values[0] if len(fov_t0) > 0 else 1
        nf7 = fov_t7["n_fovs"].values[0] if len(fov_t7) > 0 else 1

        dens0 = n0 / nf0 if nf0 > 0 else np.nan
        dens7 = n7 / nf7 if nf7 > 0 else np.nan
        fc_dens = (dens7 / dens0) if (dens0 and dens0 > 0) else np.nan

        vol_dens0 = v0_tot / nf0 if nf0 > 0 else np.nan
        vol_dens7 = v7_tot / nf7 if nf7 > 0 else np.nan
        fc_vol_dens = (vol_dens7 / vol_dens0) if (vol_dens0 and vol_dens0 > 0) else np.nan

        rows.append({
            "condition": c,
            "replicate": r,
            "total_volume_um3_t0": v0_tot,
            "total_volume_um3_t7": v7_tot,
            "fc_total_volume": fc_tot,
            "count_t0": n0,
            "count_t7": n7,
            "fc_count": fc_count,
            "mean_volume_um3_t0": v0_mean,
            "mean_volume_um3_t7": v7_mean,
            "fc_mean_volume": fc_mean_vol,
            "effective_fovs_t0": nf0,
            "effective_fovs_t7": nf7,
            "count_density_t0": dens0,
            "count_density_t7": dens7,
            "fc_count_density": fc_dens,
            "pooled_volume_density_t0": vol_dens0,
            "pooled_volume_density_t7": vol_dens7,
            "fc_pooled_vol_density": fc_vol_dens,
        })

    res_df = pd.DataFrame(rows)
    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_p, index=False)
        logger.info(f"Saved well growth comparison table to {out_p}")
    return res_df


def stagger_log_positions(y_vals: Sequence[float], min_log_gap: float = 0.12) -> np.ndarray:
    """Adjusts y-positions on log scale to prevent overlapping labels."""
    safe_y = np.maximum(1e-4, np.array(y_vals, dtype=float))
    order = np.argsort(safe_y)
    log_y = np.log10(safe_y)
    for i in range(1, len(order)):
        prev_idx = order[i - 1]
        curr_idx = order[i]
        if log_y[curr_idx] - log_y[prev_idx] < min_log_gap:
            log_y[curr_idx] = log_y[prev_idx] + min_log_gap
    shift = (np.mean(log_y) - np.mean(np.log10(safe_y))) * 0.5
    log_y -= shift
    return 10 ** log_y


def generate_all_presentation_figures(
    output_dir: Union[str, Path],
    objects_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    wells_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Tuple[Path, Path]]:
    """Generates all 6 publication presentation figures in output_dir/figures/."""
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if wells_df is None:
        wells_csv = out_dir / "well_growth_comparison.csv"
        wells_df = compute_well_growth_comparison(objects_df, wells_csv)

    pass_objs = objects_df[objects_df["qc_flag"] == "PASS"].copy()
    pass_pairs = pairs_df[pairs_df["combined_qc_flag"] == "PASS"].copy() if not pairs_df.empty else pd.DataFrame()
    known_conds = [c for c in COND_ORDER if c in wells_df["condition"].unique()]
    other_conds = [c for c in sorted(wells_df["condition"].unique()) if c not in known_conds]
    cond_order = known_conds + other_conds
    if not cond_order:
        logger.warning("No conditions found in wells_df; skipping presentation figure generation.")
        return {}
    positions = np.arange(len(cond_order))

    generated_figs = {}

    # =========================================================================
    # FIGURE 1: Primary Biomass & Proliferation Dashboard
    # =========================================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    box_data = []
    for c in cond_order:
        sub = wells_df[wells_df["condition"] == c]
        fcs = sub["fc_pooled_vol_density"].replace([np.inf, -np.inf], np.nan).dropna()
        box_data.append(fcs.values)

    bp1 = ax1.boxplot(
        box_data, positions=positions, widths=0.5, patch_artist=True,
        showfliers=False, medianprops=dict(color="#d35400", linewidth=2.5)
    )
    for patch, c in zip(bp1["boxes"], cond_order):
        patch.set_facecolor(COND_COLORS.get(c, "#95a5a6"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#2c3e50")

    np.random.seed(42)
    for i, c in enumerate(cond_order):
        sub = wells_df[wells_df["condition"] == c]
        fcs = sub["fc_pooled_vol_density"].replace([np.inf, -np.inf], np.nan).dropna().values
        jitter = np.random.normal(0, 0.07, size=len(fcs))
        ax1.scatter(positions[i] + jitter, fcs, color="#2c3e50", alpha=0.75, s=45, zorder=4, edgecolor="white", linewidth=0.5)
        ax1.scatter(positions[i], np.mean(fcs), marker="D", color="#f1c40f", s=65, zorder=5, edgecolor="#2c3e50")

        if len(fcs) >= 5:
            res_wilc = stats.wilcoxon(fcs - 1.0, alternative="greater")
            sig_a = "**" if res_wilc.pvalue < 0.01 else ("*" if res_wilc.pvalue < 0.05 else "ns")
        else:
            sig_a = "ns"
        ax1.text(positions[i], max(fcs) * 1.15, sig_a, ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2c3e50")

    ax1.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8, label="No Growth (FC=1.0)")
    ax1.set_yscale("log")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(cond_order, fontweight="bold", fontsize=11)
    ax1.set_ylabel("Total Pooled Biomass Fold Change\n($\\Sigma V_{t7} / \\Sigma V_{t0}$) [Log Scale]", fontsize=11, fontweight="bold")
    ax1.set_title("A. Total Pooled Well Biomass Fold Change\n(Replicates vs $H_0: \\mathrm{FC}=1.0$)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, which="both", linestyle=":", alpha=0.4, axis="y")
    ax1.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Panel B: t0 vs t7 population volumes
    box_t0, box_t7 = [], []
    pos_t0, pos_t7 = [], []
    width = 0.35
    for i, c in enumerate(cond_order):
        grp = pass_objs[pass_objs["condition"] == c]
        box_t0.append(grp[grp["timepoint"] == "t0"]["volume_sphere_um3"].values)
        box_t7.append(grp[grp["timepoint"] == "t7"]["volume_sphere_um3"].values)
        pos_t0.append(i - width / 2.0)
        pos_t7.append(i + width / 2.0)

    bp_t0 = ax2.boxplot(box_t0, positions=pos_t0, widths=width * 0.85, patch_artist=True, showfliers=False, medianprops=dict(color="#1b4f72", linewidth=2.0))
    bp_t7 = ax2.boxplot(box_t7, positions=pos_t7, widths=width * 0.85, patch_artist=True, showfliers=False, medianprops=dict(color="#7e5109", linewidth=2.0))
    for p in bp_t0["boxes"]:
        p.set_facecolor("#5dade2")
        p.set_alpha(0.75)
    for p in bp_t7["boxes"]:
        p.set_facecolor("#f39c12")
        p.set_alpha(0.75)

    ax2.set_yscale("log")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(cond_order, fontweight="bold", fontsize=11)
    ax2.set_ylabel("Spheroid Volume ($V_{\\mathrm{sphere}}$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
    ax2.set_title("B. Population Volume Shift\n(All PASS Spheroids: $t_0$ vs $t_7$)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.4, axis="y")

    for c in cond_order:
        sub_p = pass_pairs[pass_pairs["condition"] == c] if not pass_pairs.empty else pd.DataFrame()
        if not sub_p.empty:
            if "fold_change_volume" in sub_p.columns:
                fcs = sub_p["fold_change_volume"].replace([np.inf, -np.inf], np.nan).dropna()
            elif "fold_change_v7_v0" in sub_p.columns:
                fcs = sub_p["fold_change_v7_v0"].replace([np.inf, -np.inf], np.nan).dropna()
            else:
                fcs = (sub_p["t7_volume_sphere_um3"] / np.maximum(1e-6, sub_p["t0_volume_sphere_um3"])).replace([np.inf, -np.inf], np.nan).dropna()
            ax3.scatter(sub_p["t0_volume_sphere_um3"], fcs, label=f"{c} (n={len(sub_p)})", color=COND_COLORS.get(c, "#95a5a6"), alpha=0.75, s=45)
    ax3.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Baseline Volume ($V_0$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Matched Pair Fold Change ($V_7 / V_0$) [Log Scale]", fontsize=11, fontweight="bold")
    ax3.set_title("C. Individual Spheroid Growth vs Initial Size\n(Paired Centroid Tracking)", fontsize=12, fontweight="bold", pad=10)
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    p1_png = fig_dir / "presentation_fig1_primary_biomass_dashboard.png"
    p1_pdf = fig_dir / "presentation_fig1_primary_biomass_dashboard.pdf"
    fig1.savefig(p1_png, dpi=300)
    fig1.savefig(p1_pdf)
    plt.close(fig1)
    generated_figs["fig1"] = (p1_png, p1_pdf)

    # =========================================================================
    # FIGURE 2: Hypertrophy vs Hyperplasia Decomposition
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(10, 7), dpi=300)
    for c in cond_order:
        sub = wells_df[wells_df["condition"] == c]
        ax.scatter(sub["fc_mean_volume"], sub["fc_count_density"], color=COND_COLORS.get(c, "#95a5a6"), label=c, s=70, alpha=0.85, edgecolor="#2c3e50")
    ax.axvline(1.0, color="#7f8c8d", linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="#7f8c8d", linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean Spheroid Volume Fold Change (Hypertrophy)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Spheroid Count Density Fold Change (Hyperplasia)", fontsize=11, fontweight="bold")
    ax.set_title("Hypertrophy vs Hyperplasia Decomposition\n(Separating Spheroid Enlargement from Colony Proliferation)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5)
    plt.tight_layout()
    p2_png = fig_dir / "presentation_fig2_hypertrophy_vs_hyperplasia.png"
    p2_pdf = fig_dir / "presentation_fig2_hypertrophy_vs_hyperplasia.pdf"
    fig2.savefig(p2_png, dpi=300)
    fig2.savefig(p2_pdf)
    plt.close(fig2)
    generated_figs["fig2"] = (p2_png, p2_pdf)

    # =========================================================================
    # FIGURE 3: Per-Spheroid Volume Distributions
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(12, 6), dpi=300)
    all_vols = [pass_objs[pass_objs["condition"] == c]["volume_sphere_um3"].values for c in cond_order]
    bp3 = ax.boxplot(all_vols, positions=positions, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="#d35400", linewidth=2.5))
    for patch, c in zip(bp3["boxes"], cond_order):
        patch.set_facecolor(COND_COLORS.get(c, "#95a5a6"))
        patch.set_alpha(0.65)
        patch.set_edgecolor("#2c3e50")
    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(cond_order, fontweight="bold", fontsize=11)
    ax.set_ylabel("Spheroid Volume ($V_{\\mathrm{sphere}}$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_title("Per-Spheroid Volume Distributions across All Valid Detected Objects", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4, axis="y")
    plt.tight_layout()
    p3_png = fig_dir / "presentation_fig3_per_spheroid_volume_distributions.png"
    p3_pdf = fig_dir / "presentation_fig3_per_spheroid_volume_distributions.pdf"
    fig3.savefig(p3_png, dpi=300)
    fig3.savefig(p3_pdf)
    plt.close(fig3)
    generated_figs["fig3"] = (p3_png, p3_pdf)

    # =========================================================================
    # FIGURE 4: Pipeline Quality Control Yield
    # =========================================================================
    fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    pass_pct, rev_pct, fail_pct = [], [], []
    for c in cond_order:
        grp = objects_df[objects_df["condition"] == c]
        tot = len(grp)
        pass_pct.append((len(grp[grp["qc_flag"] == "PASS"]) / tot) * 100.0 if tot else 0)
        rev_pct.append((len(grp[grp["qc_flag"] == "REVIEW"]) / tot) * 100.0 if tot else 0)
        fail_pct.append((len(grp[grp["qc_flag"] == "FAIL"]) / tot) * 100.0 if tot else 0)

    pass_pct = np.array(pass_pct)
    rev_pct = np.array(rev_pct)
    fail_pct = np.array(fail_pct)
    y_pos = np.arange(len(cond_order))

    ax4_1.barh(y_pos, pass_pct, color="#2ecc71", edgecolor="#27ae60", height=0.55, label="PASS")
    ax4_1.barh(y_pos, rev_pct, left=pass_pct, color="#f39c12", edgecolor="#d68910", height=0.55, label="REVIEW")
    ax4_1.barh(y_pos, fail_pct, left=pass_pct + rev_pct, color="#e74c3c", edgecolor="#c0392b", height=0.55, label="FAIL")
    ax4_1.set_yticks(y_pos)
    ax4_1.set_yticklabels(cond_order, fontweight="bold", fontsize=11)
    ax4_1.set_xlabel("Percentage of Detected Candidate Instances (%)", fontsize=11, fontweight="bold")
    ax4_1.set_title("A. Pipeline Quality Control Yield", fontsize=12, fontweight="bold", pad=10)
    ax4_1.set_xlim(0, 100)
    ax4_1.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax4_1.invert_yaxis()

    fail_objs = objects_df[objects_df["qc_flag"] == "FAIL"]
    reasons = {
        "Border Clipped": fail_objs["qc_reasons"].str.contains("touches_image_border", na=False).sum() if not fail_objs.empty else 0,
        "Low Contrast": fail_objs["qc_reasons"].str.contains("insufficient_dark_contrast", na=False).sum() if not fail_objs.empty else 0,
        "Undersized": fail_objs["qc_reasons"].str.contains("debris_undersized", na=False).sum() if not fail_objs.empty else 0,
        "Multi-Focal Duplicate": fail_objs["qc_reasons"].str.contains("duplicate_focal_plane", na=False).sum() if not fail_objs.empty else 0,
        "Oversized": fail_objs["qc_reasons"].str.contains("oversized", na=False).sum() if not fail_objs.empty else 0,
    }
    ax4_2.bar(list(reasons.keys()), list(reasons.values()), color="#34495e", edgecolor="#1a252f", width=0.5, alpha=0.85)
    ax4_2.set_ylabel("Excluded Objects Count", fontsize=11, fontweight="bold")
    ax4_2.set_title(f"B. Forensic Breakdown of Excluded Objects (N={len(fail_objs)})", fontsize=12, fontweight="bold", pad=10)
    ax4_2.grid(True, linestyle=":", alpha=0.4, axis="y")

    plt.tight_layout()
    p4_png = fig_dir / "presentation_fig4_qc_pipeline_yield.png"
    p4_pdf = fig_dir / "presentation_fig4_qc_pipeline_yield.pdf"
    fig4.savefig(p4_png, dpi=300)
    fig4.savefig(p4_pdf)
    plt.close(fig4)
    generated_figs["fig4"] = (p4_png, p4_pdf)

    # =========================================================================
    # FIGURE 5: Paired Spheroid Trajectories
    # =========================================================================
    fig5, (ax5_1, ax5_2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    if not pass_pairs.empty:
        for c in cond_order:
            sub = pass_pairs[pass_pairs["condition"] == c]
            if len(sub) > 0:
                ax5_1.scatter(
                    sub["t0_volume_sphere_um3"], sub["t7_volume_sphere_um3"],
                    color=COND_COLORS.get(c, "#95a5a6"), label=f"{c} (n={len(sub)})",
                    s=75, alpha=0.85, edgecolor="#2c3e50", linewidth=0.8, zorder=4
                )
        min_v = min(pass_pairs["t0_volume_sphere_um3"].min(), pass_pairs["t7_volume_sphere_um3"].min()) * 0.7
        max_v = max(pass_pairs["t0_volume_sphere_um3"].max(), pass_pairs["t7_volume_sphere_um3"].max()) * 1.5
        line_x = np.array([min_v, max_v])
        ax5_1.plot(line_x, line_x, color="#7f8c8d", linestyle="--", linewidth=1.5, label="1:1 (No Change)")
        ax5_1.plot(line_x, line_x * 2.0, color="#27ae60", linestyle=":", linewidth=1.5, label="2:1 (Doubling)")
        ax5_1.plot(line_x, line_x * 4.0, color="#e67e22", linestyle=":", linewidth=1.5, label="4:1 (Quadrupling)")
        ax5_1.set_xscale("log")
        ax5_1.set_yscale("log")
        ax5_1.set_xlim(min_v, max_v)
        ax5_1.set_ylim(min_v, max_v)
        ax5_1.set_xlabel("Baseline Volume ($V_0$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
        ax5_1.set_ylabel("Endpoint Volume ($V_7$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
        ax5_1.set_title("A. Tracked Individual Pairs ($V_0$ vs $V_7$)", fontsize=12, fontweight="bold", pad=10)
        ax5_1.grid(True, which="both", linestyle=":", alpha=0.4)
        ax5_1.legend(loc="upper left", frameon=True, fontsize=9.5)

        for c in cond_order:
            sub = pass_pairs[pass_pairs["condition"] == c]
            if len(sub) > 0:
                med_v0 = sub["t0_volume_sphere_um3"].median()
                med_v7 = sub["t7_volume_sphere_um3"].median()
                ax5_2.plot([0, 1], [med_v0, med_v7], color=COND_COLORS.get(c, "#95a5a6"), linewidth=3.5, label=f"{c} Median ({med_v7/med_v0:.2f}x)")
        ax5_2.set_yscale("log")
        ax5_2.set_xticks([0, 1])
        ax5_2.set_xticklabels(["Day 0 ($t_0$)", "Day 7 ($t_7$)"], fontweight="bold", fontsize=11)
        ax5_2.set_ylabel("Spheroid Volume ($V_{\\mathrm{sphere}}$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=11, fontweight="bold")
        ax5_2.set_title("B. Paired Growth Trajectories (Condition Medians)", fontsize=12, fontweight="bold", pad=10)
        ax5_2.grid(True, which="both", linestyle=":", alpha=0.4)
        ax5_2.legend(loc="upper left", frameon=True, fontsize=9.5)

    plt.tight_layout()
    p5_png = fig_dir / "presentation_fig5_paired_trajectories.png"
    p5_pdf = fig_dir / "presentation_fig5_paired_trajectories.pdf"
    fig5.savefig(p5_png, dpi=300)
    fig5.savefig(p5_pdf)
    plt.close(fig5)
    generated_figs["fig5"] = (p5_png, p5_pdf)

    # =========================================================================
    # FIGURE 6: Raw Volume Numbers Per Well (t0 vs t7) with Anti-Collision Labels
    # =========================================================================
    n_conds = len(cond_order)
    n_cols = min(3, n_conds)
    n_rows = int(np.ceil(n_conds / n_cols))
    fig6, axes6 = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 6 * n_rows), dpi=300, sharey=True)
    axes_flat = np.array(axes6).flatten()

    for idx, c in enumerate(cond_order):
        ax = axes_flat[idx]
        sub = wells_df[wells_df["condition"] == c].sort_values("replicate").reset_index(drop=True)
        v0_raw = (sub["total_volume_um3_t0"] / 1e6).values
        v7_raw = (sub["total_volume_um3_t7"] / 1e6).values
        ratios = v7_raw / np.maximum(1e-6, v0_raw)
        reps = sub["replicate"].values

        stag_v0 = stagger_log_positions(v0_raw, min_log_gap=0.12) if len(v0_raw) > 1 else v0_raw
        stag_v7 = stagger_log_positions(v7_raw, min_log_gap=0.12) if len(v7_raw) > 1 else v7_raw

        for i in range(len(sub)):
            v0, v7 = v0_raw[i], v7_raw[i]
            ratio = ratios[i]
            rep = reps[i]
            line_color = "#27ae60" if ratio >= 1.2 else ("#c0392b" if ratio <= 0.9 else "#7f8c8d")
            ax.plot([0, 1], [v0, v7], color=line_color, lw=2.2, alpha=0.75, zorder=3)
            ax.scatter(0, v0, color="#5dade2", edgecolor="#1b4f72", s=55, zorder=5)
            ax.scatter(1, v7, color="#f39c12", edgecolor="#7e5109", s=55, zorder=5)

            if len(v0_raw) > 1 and abs(stag_v0[i] - v0) / max(1e-6, v0) > 0.05:
                ax.plot([-0.05, 0], [stag_v0[i], v0], color="#bdc3c7", lw=0.7, linestyle=":", zorder=2)
            if len(v7_raw) > 1 and abs(stag_v7[i] - v7) / max(1e-6, v7) > 0.05:
                ax.plot([1, 1.05], [v7, stag_v7[i]], color="#bdc3c7", lw=0.7, linestyle=":", zorder=2)

            ax.text(-0.06, stag_v0[i], f"{rep}: {v0:.1f}", ha="right", va="center", fontsize=8.5, color="#2c3e50", fontweight="bold")
            ax.text(1.06, stag_v7[i], f"{rep}: {v7:.1f} M ({ratio:.2f}x)", ha="left", va="center", fontsize=8.5, color=line_color, fontweight="bold")

        mean_v0 = np.mean(v0_raw)
        mean_v7 = np.mean(v7_raw)
        mean_ratio = mean_v7 / max(1e-6, mean_v0)
        ax.plot([0, 1], [mean_v0, mean_v7], color="#1c2833", lw=3.5, linestyle="--", label=f"Mean: {mean_v0:.1f} -> {mean_v7:.1f} M ({mean_ratio:.2f}x)", zorder=6)
        ax.set_yscale("log")
        ax.set_ylim(0.15, 260)
        ax.set_xlim(-0.42, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Day 0 ($t_0$)", "Day 7 ($t_7$)"], fontweight="bold", fontsize=11.5)
        ax.set_title(f"{c} (N = {len(sub)} Replicate Wells)", fontsize=13, fontweight="bold", color=COND_COLORS.get(c, "#2c3e50"), pad=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    for r in range(n_rows):
        axes6[r, 0].set_ylabel("Raw Spheroid Volume per Well\n($10^6\\,\\mu\\mathrm{m}^3 = 1\\,\\mathrm{nL}$) [Log Scale]", fontsize=11.5, fontweight="bold")

    fig6.suptitle("Raw Total Spheroid Volume per Replicate Well (Day 0 vs Day 7, Deduplicated)\nValues in $10^6\\,\\mu\\mathrm{m}^3$ (Nanoliters) with Individual Well Raw Ratios ($V_{t7} / V_{t0}$)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p6_png = fig_dir / "presentation_fig6_raw_well_volumes.png"
    p6_pdf = fig_dir / "presentation_fig6_raw_well_volumes.pdf"
    fig6.savefig(p6_png, dpi=300)
    fig6.savefig(p6_pdf)
    plt.close(fig6)
    generated_figs["fig6"] = (p6_png, p6_pdf)

    logger.info(f"Generated all 6 presentation figures in {fig_dir}")
    return generated_figs
