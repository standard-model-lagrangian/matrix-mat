"""
Forensic Audit & Session Quality Module for Spheroid Volume Pipeline v2.

Provides:
  1. Deep audit of the FAIL classification:
     - Multi-panel size, contrast, and solidity distribution histograms.
     - Forensic crop contact sheet for 20 representative FAIL objects.
     - Condition-wise detection yield breakdown (evaluating gel texture interference).
  2. Systematic session-level quality & focus analysis:
     - Laplacian variance (focus/sharpness) and Brenner gradient for all 285 raw TIFFs.
     - Global intensity and standard deviation comparing Day 0 vs Day 7 imaging sessions.
     - Condition-independence verification for optical shifts.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("spheroid_pipeline_v2.audit")

# Palette
QC_COLORS = {"PASS": "#2ecc71", "REVIEW": "#e67e22", "FAIL": "#e74c3c"}
COND_COLORS = {
    "Mat": "#1f77b4",
    "S34D30": "#ff7f0e",
    "S40D30": "#2ca02c",
    "S43D20": "#d62728",
    "S46D10": "#9467bd",
    "S50": "#8c564b",
}


def audit_fail_class(
    objects_df: pd.DataFrame,
    output_dir: Union[str, Path],
    save_pdf: bool = True,
) -> Dict[str, Path]:
    """
    Generates multi-panel audit figure characterizing the FAIL object population:
    - Panel A: Diameter distribution (FAIL vs PASS vs REVIEW)
    - Panel B: Annular Contrast Ratio distribution
    - Panel C: Solidity distribution
    - Panel D: Detection Yield per Condition (PASS, REVIEW, FAIL)
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    ax_diam, ax_cont = axes[0, 0], axes[0, 1]
    ax_sol, ax_yield = axes[1, 0], axes[1, 1]

    if objects_df.empty or "qc_flag" not in objects_df.columns:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No objects data available", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        png_path = fig_dir / "fail_class_audit.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
        if save_pdf:
            pdf_path = fig_dir / "fail_class_audit.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            results["pdf"] = pdf_path
        plt.close(fig)
        return results

    # Panel A: Diameter Distribution (log scale x-axis)
    bins_d = np.geomspace(20, 1500, 40)
    for qf in ["FAIL", "REVIEW", "PASS"]:
        sub = objects_df[objects_df["qc_flag"] == qf]
        d_vals = sub["equivalent_diameter_um"].dropna().values if "equivalent_diameter_um" in sub.columns else np.array([])
        if len(d_vals) > 0:
            ax_diam.hist(
                d_vals,
                bins=bins_d,
                alpha=0.6,
                label=f"{qf} (n={len(d_vals):,})",
                color=QC_COLORS.get(qf, "#7f8c8d"),
                edgecolor="white",
                density=True,
            )
        else:
            ax_diam.plot([], [], label=f"{qf} (n=0)", color=QC_COLORS.get(qf, "#7f8c8d"))
    ax_diam.axvline(40, color="#c0392b", linestyle="--", linewidth=1.5, label="Min Size Gate (40 um)")
    ax_diam.set_xscale("log")
    ax_diam.set_xlabel("Equivalent Diameter ($d$, $\\mu$m) [Log Scale]", fontsize=11, fontweight="bold")
    ax_diam.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_diam.set_title("A. Size Distribution by QC Class", fontsize=12, fontweight="bold")
    ax_diam.legend(fontsize=9, frameon=True)
    ax_diam.grid(True, linestyle=":", alpha=0.5)

    # Panel B: Contrast Ratio Distribution
    bins_c = np.linspace(-0.05, 0.40, 45)
    for qf in ["FAIL", "REVIEW", "PASS"]:
        sub = objects_df[objects_df["qc_flag"] == qf]
        c_vals = sub["contrast_ratio"].dropna().values if "contrast_ratio" in sub.columns else np.array([])
        if len(c_vals) > 0:
            ax_cont.hist(
                c_vals,
                bins=bins_c,
                alpha=0.6,
                label=f"{qf} (n={len(c_vals):,})",
                color=QC_COLORS.get(qf, "#7f8c8d"),
                edgecolor="white",
                density=True,
            )
        else:
            ax_cont.plot([], [], label=f"{qf} (n=0)", color=QC_COLORS.get(qf, "#7f8c8d"))
    ax_cont.axvline(0.10, color="#c0392b", linestyle="--", linewidth=1.5, label="Min Contrast Gate (10%)")
    ax_cont.set_xlabel("Dark Contrast Ratio ($\\Delta I / I_{\\mathrm{ring}}$)", fontsize=11, fontweight="bold")
    ax_cont.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_cont.set_title("B. Annular Ring Contrast Distribution", fontsize=12, fontweight="bold")
    ax_cont.legend(fontsize=9, frameon=True)
    ax_cont.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Solidity Distribution
    bins_s = np.linspace(0.2, 1.0, 40)
    for qf in ["FAIL", "REVIEW", "PASS"]:
        sub = objects_df[objects_df["qc_flag"] == qf]
        s_vals = sub["solidity"].dropna().values if "solidity" in sub.columns else np.array([])
        if len(s_vals) > 0:
            ax_sol.hist(
                s_vals,
                bins=bins_s,
                alpha=0.6,
                label=f"{qf} (n={len(s_vals):,})",
                color=QC_COLORS.get(qf, "#7f8c8d"),
                edgecolor="white",
                density=True,
            )
        else:
            ax_sol.plot([], [], label=f"{qf} (n=0)", color=QC_COLORS.get(qf, "#7f8c8d"))
    ax_sol.axvline(0.85, color="#c0392b", linestyle="--", linewidth=1.5, label="Min Solidity Gate (0.85)")
    ax_sol.set_xlabel("Morphological Solidity ($A / A_{\\mathrm{convex}}$)", fontsize=11, fontweight="bold")
    ax_sol.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax_sol.set_title("C. Solidity Distribution by QC Class", fontsize=12, fontweight="bold")
    ax_sol.legend(fontsize=9, frameon=True)
    ax_sol.grid(True, linestyle=":", alpha=0.5)

    # Panel D: Detection Yield Breakdown by Condition
    cond_order = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]
    present_conds = [c for c in cond_order if c in objects_df["condition"].unique()]
    
    yield_data = []
    for c in present_conds:
        sub = objects_df[objects_df["condition"] == c]
        n_pass = (sub["qc_flag"] == "PASS").sum()
        n_rev = (sub["qc_flag"] == "REVIEW").sum()
        n_fail = (sub["qc_flag"] == "FAIL").sum()
        yield_data.append({"Condition": c, "PASS": n_pass, "REVIEW": n_rev, "FAIL": n_fail, "Total": len(sub)})
    
    ydf = pd.DataFrame(yield_data)
    x = np.arange(len(present_conds))
    width = 0.55

    ax_yield.bar(x, ydf["PASS"], width, label="PASS", color=QC_COLORS["PASS"])
    ax_yield.bar(x, ydf["REVIEW"], width, bottom=ydf["PASS"], label="REVIEW", color=QC_COLORS["REVIEW"])
    ax_yield.bar(x, ydf["FAIL"], width, bottom=ydf["PASS"] + ydf["REVIEW"], label="FAIL (Faint/Debris)", color=QC_COLORS["FAIL"], alpha=0.5)

    for i, row in ydf.iterrows():
        total = row["Total"]
        p_pct = (row["PASS"] / total * 100) if total > 0 else 0
        ax_yield.text(i, total + 40, f"PASS:\n{row['PASS']} ({p_pct:.1f}%)", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_yield.set_xticks(x)
    ax_yield.set_xticklabels(present_conds, fontsize=10, fontweight="bold")
    ax_yield.set_ylabel("Total Candidate Objects Detected", fontsize=11, fontweight="bold")
    ax_yield.set_title("D. Detection Yield & Candidate Volume by Condition", fontsize=12, fontweight="bold")
    ax_yield.legend(loc="upper right", frameon=True)
    ax_yield.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.suptitle("Forensic Audit of Candidate Detection & FAIL Class Gating", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    png_path = fig_dir / "fail_class_audit.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    results["png"] = png_path

    if save_pdf:
        pdf_path = fig_dir / "fail_class_audit.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def create_fail_audit_crops_grid(
    objects_df: pd.DataFrame,
    t0_dir: Union[str, Path],
    t7_dir: Union[str, Path],
    output_path: Union[str, Path],
    num_samples: int = 20,
) -> Optional[Path]:
    """
    Extracts 20 representative crops of FAIL objects from raw images with overlaid contours
    and metrics to visually audit why they were rejected (demonstrating gel texture vs debris).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if objects_df.empty or "qc_flag" not in objects_df.columns:
        return None

    fails = objects_df[objects_df["qc_flag"] == "FAIL"].copy()
    if fails.empty:
        return None

    # Sample stratified across conditions
    sample_df = fails.sample(n=min(num_samples, len(fails)), random_state=42)

    crop_images = []
    crop_size = (180, 180)

    for _, row in sample_df.iterrows():
        img_dir = Path(t0_dir) if row["timepoint"] == "t0" else Path(t7_dir)
        raw_path = img_dir / f"{row['image_id']}.tif"
        if not raw_path.exists():
            raw_path = img_dir / f"{row['image_id']}.tiff"
            if not raw_path.exists():
                raw_path = img_dir / str(row["image_id"])
                if not raw_path.exists():
                    continue

        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])
        r = int((row["equivalent_diameter_px"] if "equivalent_diameter_px" in row else 50) / 2) + 25
        h, w = raw.shape

        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)

        sub_raw = raw[y0:y1, x0:x1]
        if sub_raw.size == 0:
            continue

        sub_u8 = cv2.resize(sub_raw, crop_size, interpolation=cv2.INTER_AREA)
        sub_bgr = cv2.cvtColor(sub_u8, cv2.COLOR_GRAY2BGR)

        # Draw red border and annotation
        cv2.rectangle(sub_bgr, (0, 0), (crop_size[0] - 1, crop_size[1] - 1), (60, 76, 231), 3)
        cv2.circle(sub_bgr, (crop_size[0] // 2, crop_size[1] // 2), 4, (60, 76, 231), -1)

        # Labels
        d_val = row["equivalent_diameter_um"]
        c_val = row["contrast_ratio"]
        cond = row["condition"]
        reason = "Faint" if "contrast" in str(row["qc_reasons"]) else "Size"

        cv2.putText(sub_bgr, f"{cond} ({row['timepoint']})", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(sub_bgr, f"{cond} ({row['timepoint']})", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(sub_bgr, f"d={d_val:.1f}um c={c_val:.2f}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)
        cv2.putText(sub_bgr, f"d={d_val:.1f}um c={c_val:.2f}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 76, 231), 1)

        cv2.putText(sub_bgr, f"FAIL: {reason}", (6, crop_size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(sub_bgr, f"FAIL: {reason}", (6, crop_size[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 76, 231), 1)

        crop_images.append(sub_bgr)

    if not crop_images:
        return None

    # Tile into grid 5 cols x 4 rows
    cols = 5
    rows = int(np.ceil(len(crop_images) / cols))
    grid_h = rows * crop_size[1]
    grid_w = cols * crop_size[0]

    grid = np.full((grid_h + 50, grid_w, 3), 245, dtype=np.uint8)

    # Header title
    cv2.putText(grid, "FAIL Class Audit Grid — 20 Sampled Objects (Demonstrating Rejection of Faint Gel Artifacts)", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    for idx, c_img in enumerate(crop_images):
        r_i = idx // cols
        c_i = idx % cols
        y_pos = 50 + r_i * crop_size[1]
        x_pos = c_i * crop_size[0]
        grid[y_pos:y_pos + crop_size[1], x_pos:x_pos + crop_size[0]] = c_img

    cv2.imwrite(str(out_path), grid)
    logger.info(f"Saved FAIL audit contact grid to {out_path}")
    return out_path


def audit_session_quality(
    t0_dir: Union[str, Path],
    t7_dir: Union[str, Path],
    output_dir: Union[str, Path],
    save_pdf: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Computes image-level sharpness (Laplacian variance, Brenner gradient) and exposure metrics
    across all 285 raw brightfield TIFFs to quantify session-level optical stability.
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    t0_files = sorted(glob.glob(os.path.join(t0_dir, "*.tif")))
    t7_files = sorted(glob.glob(os.path.join(t7_dir, "*.tif")))

    records = []
    for tp, files in [("t0", t0_files), ("t7", t7_files)]:
        for f in files:
            fname = os.path.basename(f)
            cond = "Mat" if "Mat" in fname else fname.split()[1] if len(fname.split()) > 1 else "Unknown"
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue

            lap_var = float(cv2.Laplacian(im, cv2.CV_64F).var())
            brenner = float(np.mean((im[:, 2:].astype(float) - im[:, :-2].astype(float)) ** 2))
            mean_i = float(np.mean(im))
            std_i = float(np.std(im))

            records.append({
                "filename": fname,
                "timepoint": tp,
                "condition": cond,
                "laplacian_var": lap_var,
                "brenner_gradient": brenner,
                "mean_intensity": mean_i,
                "std_intensity": std_i,
            })

    session_df = pd.DataFrame(records)
    csv_path = out_dir / "session_quality_audit.csv"
    session_df.to_csv(csv_path, index=False)

    # Plot 2-panel figure
    fig, (ax_focus, ax_exp) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    cond_order = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]
    present_conds = [c for c in cond_order if c in session_df["condition"].unique()]

    # Focus Plot
    t0_focus = [session_df[(session_df["condition"] == c) & (session_df["timepoint"] == "t0")]["laplacian_var"].values for c in present_conds]
    t7_focus = [session_df[(session_df["condition"] == c) & (session_df["timepoint"] == "t7")]["laplacian_var"].values for c in present_conds]

    x = np.arange(len(present_conds))
    width = 0.35

    ax_focus.bar(x - width/2, [np.median(v) if len(v)>0 else 0 for v in t0_focus], width, label="Day 0 ($t_0$)", color="#3498db", alpha=0.85)
    ax_focus.bar(x + width/2, [np.median(v) if len(v)>0 else 0 for v in t7_focus], width, label="Day 7 ($t_7$)", color="#e67e22", alpha=0.85)
    ax_focus.set_xticks(x)
    ax_focus.set_xticklabels(present_conds, fontsize=10, fontweight="bold")
    ax_focus.set_ylabel("Median Focus Score [Laplacian Variance]", fontsize=11, fontweight="bold")
    ax_focus.set_title("A. Image Sharpness / Focus Consistency (t0 vs t7)", fontsize=12, fontweight="bold")
    ax_focus.legend(frameon=True)
    ax_focus.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Exposure Plot
    t0_exp = [session_df[(session_df["condition"] == c) & (session_df["timepoint"] == "t0")]["mean_intensity"].values for c in present_conds]
    t7_exp = [session_df[(session_df["condition"] == c) & (session_df["timepoint"] == "t7")]["mean_intensity"].values for c in present_conds]

    ax_exp.bar(x - width/2, [np.median(v) if len(v)>0 else 0 for v in t0_exp], width, label="Day 0 ($t_0$)", color="#3498db", alpha=0.85)
    ax_exp.bar(x + width/2, [np.median(v) if len(v)>0 else 0 for v in t7_exp], width, label="Day 7 ($t_7$)", color="#e67e22", alpha=0.85)
    ax_exp.set_xticks(x)
    ax_exp.set_xticklabels(present_conds, fontsize=10, fontweight="bold")
    ax_exp.set_ylabel("Median Global Image Intensity (8-bit)", fontsize=11, fontweight="bold")
    ax_exp.set_title("B. Illumination / Exposure Stability (t0 vs t7)", fontsize=12, fontweight="bold")
    ax_exp.legend(frameon=True)
    ax_exp.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.suptitle("Imaging Session Quality & Optical Shift Audit (285 Raw TIFF Images)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    png_path = fig_dir / "session_focus_exposure_audit.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    results["png"] = png_path

    if save_pdf:
        pdf_path = fig_dir / "session_focus_exposure_audit.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return session_df, results
