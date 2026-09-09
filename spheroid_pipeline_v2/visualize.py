"""
Visualization & Supervision Artifacts Module for Spheroid Volume Pipeline v2.

Provides:
  - High-contrast RGB supervision overlays with labeled contours (Green=PASS, Amber=REVIEW, Red=FAIL),
    centroid indicators, and legible multi-line text annotations (ID, diameter d in um, QC flag).
  - Multi-image contact sheets grouping FOVs by condition and replicate.
  - Publication-quality figures (both high-res 300 DPI PNG and vector PDF):
    * fold_change_violin.png / .pdf: Log-scale violin plots of PASS-only pairs across conditions.
    * population_volume_distributions.png / .pdf: Unpaired population-level V0 vs V7 volume distributions
      with Mann-Whitney U test significance and growth ratios.
    * v0_vs_v7_scatter.png / .pdf: Log-log scatter plot of V0 vs V7 with y = x identity line.
    * growth_slopegraph.png / .pdf: Paired trajectory slopegraphs (t0 -> t7).
    * circularity_sensitivity_audit.png / .pdf: Volume discrepancy ratio vs circularity audit.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import stats

from spheroid_pipeline_v2.config import ArtifactConfig, PipelineConfig
from spheroid_pipeline_v2.measure import SpheroidObjectRecord
from spheroid_pipeline_v2.pair import SpheroidPairRecord

logger = logging.getLogger("spheroid_pipeline_v2.visualize")

# Condition color palette (colorblind-friendly and publication standard)
CONDITION_COLORS: Dict[str, str] = {
    "Mat": "#1f77b4",       # Control Blue
    "S34D30": "#ff7f0e",    # Orange
    "S40D30": "#2ca02c",    # Green
    "S43D20": "#d62728",    # Red
    "S46D10": "#9467bd",    # Purple
    "S50": "#8c564b",       # Brown
}

# Standard QC colors in BGR (for OpenCV) and RGB (for PIL/Matplotlib)
QC_COLORS_BGR = {
    "PASS": (46, 204, 113),     # Emerald Green
    "REVIEW": (34, 126, 230),   # Amber/Orange
    "FAIL": (60, 76, 231),      # Crimson Red
}

QC_COLORS_RGB = {
    "PASS": "#2ecc71",     # Emerald Green
    "REVIEW": "#e67e22",   # Amber/Orange
    "FAIL": "#e74c3c",      # Crimson Red
}


def create_overlay(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    objects: Sequence[Union[SpheroidObjectRecord, Dict[str, Any]]],
    output_path: Union[str, Path],
    pixel_size_um: float = 1.518817,
) -> Path:
    """
    Creates a high-contrast RGB supervision overlay image with color-coded contours,
    centroid markers, and legible text annotations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize image to 8-bit uint8 grayscale
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0 + 1e-4:
            img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            p_low, p_high = np.percentile(image, (1.0, 99.0))
            if p_high > p_low:
                norm = np.clip((image - p_low) / (p_high - p_low) * 255.0, 0, 255)
                img_u8 = norm.astype(np.uint8)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    if img_u8.ndim == 3:
        bgr = img_u8.copy()
    else:
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    h, w = bgr.shape[:2]

    # Convert objects to dicts
    obj_dicts = []
    for obj in objects:
        if isinstance(obj, dict):
            obj_dicts.append(obj)
        elif hasattr(obj, "__dict__"):
            obj_dicts.append(obj.__dict__)
        else:
            obj_dicts.append({})

    label_to_obj: Dict[int, Dict[str, Any]] = {}
    for obj in obj_dicts:
        l_idx = obj.get("label_idx", -1)
        if l_idx > 0:
            label_to_obj[l_idx] = obj

    max_label = int(np.max(labeled_mask)) if labeled_mask.size > 0 else 0

    for label_val in range(1, max_label + 1):
        bin_mask = (labeled_mask == label_val).astype(np.uint8)
        if np.sum(bin_mask) == 0:
            continue

        obj_info = label_to_obj.get(label_val, {})
        qc_flag = obj_info.get("qc_flag", "PASS")
        color_bgr = QC_COLORS_BGR.get(qc_flag, (255, 255, 255))

        # Contours
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, color_bgr, 2, lineType=cv2.LINE_AA)

        # Centroid
        cx = obj_info.get("centroid_x")
        cy = obj_info.get("centroid_y")
        if cx is None or cy is None or not (np.isfinite(cx) and np.isfinite(cy)):
            M = cv2.moments(bin_mask)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                coords = np.argwhere(bin_mask > 0)
                cy, cx = np.mean(coords, axis=0) if len(coords) > 0 else (h // 2, w // 2)

        cx_int, cy_int = int(round(cx)), int(round(cy))
        cv2.circle(bgr, (cx_int, cy_int), 4, color_bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(bgr, (cx_int, cy_int), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Metrics
        d_um = obj_info.get("equivalent_diameter_um")
        if d_um is None:
            area_px = float(np.sum(bin_mask))
            d_um = 2.0 * math.sqrt(area_px / math.pi) * pixel_size_um

        obj_id = str(obj_info.get("object_id", f"Obj{label_val}"))
        short_id = obj_id.split("_")[-1] if "_" in obj_id else obj_id

        text_line1 = f"{short_id}:{qc_flag[0]}"
        text_line2 = f"{d_um:.0f}um"

        tx = min(max(5, cx_int + 8), w - 65)
        ty1 = min(max(15, cy_int - 5), h - 25)
        ty2 = ty1 + 14

        for ty, txt in [(ty1, text_line1), (ty2, text_line2)]:
            cv2.putText(bgr, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(bgr, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_bgr, 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), bgr)
    return output_path


def create_contact_sheet(
    overlay_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    grid_cols: int = 4,
    thumb_size: Tuple[int, int] = (400, 300),
    title: Optional[str] = None,
) -> Optional[Path]:
    """Generates multi-image montage contact sheet."""
    valid_paths = [Path(p) for p in overlay_paths if Path(p).exists()]
    if not valid_paths:
        return None

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_images = len(valid_paths)
    grid_cols = max(1, min(grid_cols, n_images))
    grid_rows = int(math.ceil(n_images / grid_cols))

    tw, th = thumb_size
    margin = 8
    header_height = 40 if title else 0

    total_w = grid_cols * tw + (grid_cols + 1) * margin
    total_h = header_height + grid_rows * th + (grid_rows + 1) * margin

    canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((margin + 5, 10), title, fill=(240, 240, 240))

    for idx, img_path in enumerate(valid_paths):
        r = idx // grid_cols
        c = idx % grid_cols
        x = margin + c * (tw + margin)
        y = header_height + margin + r * (th + margin)

        try:
            with Image.open(img_path) as im:
                thumb = im.resize(thumb_size, Image.Resampling.BILINEAR)
                canvas.paste(thumb, (x, y))
                draw.rectangle([x, y, x + tw - 1, y + th - 1], outline=(100, 100, 100), width=1)
                fname = img_path.stem.replace("_overlay", "")
                draw.text((x + 6, y + th - 18), fname, fill=(255, 255, 255))
        except Exception as e:
            logger.warning(f"Failed to paste {img_path} into contact sheet: {e}")

    canvas.save(str(out_path), quality=92)
    return out_path


def generate_all_contact_sheets(
    overlays_dir: Union[str, Path],
    output_dir: Union[str, Path],
    conditions: Sequence[str] = ("Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"),
) -> Dict[str, Path]:
    """Generates condition-specific contact sheets."""
    ov_dir = Path(overlays_dir)
    out_dir = Path(output_dir) / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pngs = sorted(ov_dir.glob("*.png"))
    results: Dict[str, Path] = {}
    grouped: Dict[str, List[Path]] = {c: [] for c in conditions}

    for p in all_pngs:
        stem = p.name
        matched = "Other"
        for cond in conditions:
            if cond in stem:
                matched = cond
                break
        if matched in grouped:
            grouped[matched].append(p)

    for cond, files in grouped.items():
        if not files:
            continue
        cs_path = out_dir / f"{cond}_contact_sheet.png"
        res = create_contact_sheet(
            overlay_paths=files,
            output_path=cs_path,
            grid_cols=4,
            thumb_size=(400, 300),
            title=f"Spheroid Quality Control Supervision Grid — Condition: {cond} ({len(files)} FOVs)",
        )
        if res is not None:
            results[cond] = res

    return results


def plot_fold_change_violin(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
) -> Dict[str, Path]:
    """
    Generates publication violin plot of volume fold change (V7 / V0) across conditions
    strictly from PASS-only mutually-paired objects passing the denominator gate.
    """
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Filter strictly to PASS-only and denominator gate passed
    valid_pairs = pairs_df.copy()
    if pass_only and "combined_qc_flag" in valid_pairs.columns:
        valid_pairs = valid_pairs[valid_pairs["combined_qc_flag"] == "PASS"]
    if "denominator_gate_pass" in valid_pairs.columns:
        valid_pairs = valid_pairs[valid_pairs["denominator_gate_pass"] == True]

    preferred_order = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]
    if "condition" in valid_pairs.columns and not valid_pairs.empty:
        present_conditions = [c for c in preferred_order if c in valid_pairs["condition"].unique()]
        other_conditions = [c for c in sorted(valid_pairs["condition"].unique()) if c not in present_conditions]
        conditions = present_conditions + other_conditions
    else:
        conditions = []

    if not conditions or valid_pairs.empty:
        ax.text(0.5, 0.5, "No PASS-verified matched pairs data available\n(See Population-Level Distribution Analysis for full cohort)", ha="center", va="center", transform=ax.transAxes, fontsize=11)
    else:
        dataset = []
        labels = []
        for c in conditions:
            sub = valid_pairs[valid_pairs["condition"] == c]
            fc_vals = sub["fold_change_volume"].dropna().values
            fc_vals = fc_vals[np.isfinite(fc_vals) & (fc_vals > 0)]
            dataset.append(fc_vals if len(fc_vals) > 0 else np.array([1.0]))
            labels.append(f"{c}\n(n={len(fc_vals)})")

        parts = ax.violinplot(
            dataset,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )

        for idx, pc in enumerate(parts["bodies"]):
            cond_name = conditions[idx]
            color = CONDITION_COLORS.get(cond_name, "#3498db")
            pc.set_facecolor(color)
            pc.set_edgecolor("#2c3e50")
            pc.set_alpha(0.65)
            pc.set_linewidth(1.2)

        if "cmedians" in parts:
            parts["cmedians"].set_color("#2c3e50")
            parts["cmedians"].set_linewidth(2.0)
        if "cbars" in parts:
            parts["cbars"].set_color("#7f8c8d")
        if "cmaxes" in parts:
            parts["cmaxes"].set_color("#7f8c8d")
        if "cmins" in parts:
            parts["cmins"].set_color("#7f8c8d")

        # Jittered scatter points
        np.random.seed(42)
        for i, c in enumerate(conditions, 1):
            sub = valid_pairs[valid_pairs["condition"] == c]
            fc_vals = sub["fold_change_volume"].dropna().values
            fc_vals = fc_vals[np.isfinite(fc_vals) & (fc_vals > 0)]
            if len(fc_vals) > 0:
                jitter = np.random.normal(0, 0.05, size=len(fc_vals))
                ax.scatter(
                    i + jitter,
                    fc_vals,
                    alpha=0.75,
                    color="#2c3e50",
                    s=32,
                    zorder=4,
                    edgecolor="white",
                    linewidth=0.5,
                )

        ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="No Growth ($V_7 = V_0$)", zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(conditions) + 1))
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.set_ylabel("Volume Fold Change ($V_7 / V_0$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title("Paired Spheroid Volume Fold Change (PASS-Only Trajectories, $d_0 \\geq 60\\,\\mu\\mathrm{m}$)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def plot_population_volume_distributions(
    objects_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
) -> Dict[str, Path]:
    """
    Generates publication-quality figure comparing full unpaired spheroid volume distributions
    at Day 0 vs Day 7 for each hydrogel formulation, with Mann-Whitney U test p-values and population growth ratios.
    """
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    sub_df = objects_df[objects_df["qc_flag"] == "PASS"].copy() if pass_only and "qc_flag" in objects_df.columns else objects_df.copy()

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    if sub_df.empty or "condition" not in sub_df.columns or "volume_sphere_um3" not in sub_df.columns or "timepoint" not in sub_df.columns:
        ax.text(0.5, 0.5, "No population volume data available", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        if save_png:
            png_path = base_path.with_suffix(".png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            results["png"] = png_path
        if save_pdf:
            pdf_path = base_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            results["pdf"] = pdf_path
        plt.close(fig)
        return results

    preferred_order = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]
    present_conditions = [c for c in preferred_order if c in sub_df["condition"].unique()]

    x_positions = []
    box_data = []
    box_colors = []
    labels = []

    for idx, cond in enumerate(present_conditions):
        c_df = sub_df[sub_df["condition"] == cond]
        v0 = c_df[c_df["timepoint"] == "t0"]["volume_sphere_um3"].dropna().values
        v7 = c_df[c_df["timepoint"] == "t7"]["volume_sphere_um3"].dropna().values

        pos_t0 = idx * 2.5 + 1.0
        pos_t7 = idx * 2.5 + 1.8

        x_positions.extend([pos_t0, pos_t7])
        box_data.extend([v0 if len(v0)>0 else np.array([1e5]), v7 if len(v7)>0 else np.array([1e5])])
        box_colors.extend(["#3498db", "#e67e22"])

        # Mann-Whitney U test
        if len(v0) > 0 and len(v7) > 0:
            _, mwu_p = stats.mannwhitneyu(v7, v0, alternative="two-sided")
            v0_med = np.median(v0)
            v7_med = np.median(v7)
            fc = v7_med / v0_med if v0_med > 0 else 1.0
            p_str = f"p={mwu_p:.1e}" if mwu_p < 0.001 else f"p={mwu_p:.3f}"
            star = "****" if mwu_p < 0.0001 else "***" if mwu_p < 0.001 else "**" if mwu_p < 0.01 else "*" if mwu_p < 0.05 else "ns"
            
            # Annotate growth above condition
            y_max = max(np.percentile(v0, 95), np.percentile(v7, 95)) * 2.0
            mid_x = (pos_t0 + pos_t7) / 2
            ax.plot([pos_t0, pos_t7], [y_max, y_max], color="#2c3e50", linewidth=1.2)
            ax.text(mid_x, y_max * 1.15, f"{star}\nFC={fc:.2f}x\n({p_str})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#2c3e50")

        labels.append(cond)

    bplot = ax.boxplot(
        box_data,
        positions=x_positions,
        patch_artist=True,
        widths=0.6,
        showmeans=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.3, markerfacecolor="#7f8c8d", markeredgecolor="none"),
    )

    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#2c3e50")

    for median in bplot["medians"]:
        median.set_color("#2c3e50")
        median.set_linewidth(2.0)

    # X axis
    mid_positions = [idx * 2.5 + 1.4 for idx in range(len(present_conditions))]
    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", alpha=0.7, edgecolor="#2c3e50", label="Day 0 ($t_0$)"),
        Patch(facecolor="#e67e22", alpha=0.7, edgecolor="#2c3e50", label="Day 7 ($t_7$)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("Spheroid Volume ($V_{\\mathrm{sphere}}$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title("Unpaired Population-Level Spheroid Volume Distributions (Day 0 vs Day 7)", fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    plt.tight_layout()

    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def plot_v0_vs_v7_scatter(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
) -> Dict[str, Path]:
    """Generates publication log-log scatter plot of V0 vs V7 with identity line."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    sub_df = pairs_df.copy()
    if pass_only and "combined_qc_flag" in sub_df.columns:
        sub_df = sub_df[sub_df["combined_qc_flag"] == "PASS"]

    req_cols = {"t0_volume_sphere_um3", "t7_volume_sphere_um3", "condition"}
    if sub_df.empty or not req_cols.issubset(sub_df.columns):
        ax.text(0.5, 0.5, "No PASS pairs data available", ha="center", va="center", transform=ax.transAxes)
    else:
        sub_df = sub_df[
            (sub_df["t0_volume_sphere_um3"] > 0)
            & (sub_df["t7_volume_sphere_um3"] > 0)
            & np.isfinite(sub_df["t0_volume_sphere_um3"])
            & np.isfinite(sub_df["t7_volume_sphere_um3"])
        ]
        if sub_df.empty:
            ax.text(0.5, 0.5, "No PASS pairs data available", ha="center", va="center", transform=ax.transAxes)
        else:
            conditions = sorted(sub_df["condition"].unique())
        for cond in conditions:
            c_df = sub_df[sub_df["condition"] == cond]
            color = CONDITION_COLORS.get(cond, "#3498db")
            ax.scatter(
                c_df["t0_volume_sphere_um3"],
                c_df["t7_volume_sphere_um3"],
                label=f"{cond} (n={len(c_df)})",
                color=color,
                alpha=0.75,
                s=45,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

        all_v = np.concatenate([sub_df["t0_volume_sphere_um3"].values, sub_df["t7_volume_sphere_um3"].values])
        min_val = max(1e4, np.min(all_v) * 0.5)
        max_val = np.max(all_v) * 2.0

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Identity ($V_7 = V_0$)", zorder=2)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Baseline Volume at Day 0 ($V_0$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Endpoint Volume at Day 7 ($V_7$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title("Spheroid Growth Tracking: Baseline vs Endpoint Volume ($V_0$ vs $V_7$)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, which="both")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def plot_growth_slopegraph(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
) -> Dict[str, Path]:
    """Generates paired slopegraph trajectories."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    sub_df = pairs_df.copy()
    if pass_only and "combined_qc_flag" in sub_df.columns:
        sub_df = sub_df[sub_df["combined_qc_flag"] == "PASS"]

    req_cols = {"t0_volume_sphere_um3", "t7_volume_sphere_um3", "condition"}
    if sub_df.empty or not req_cols.issubset(sub_df.columns):
        conditions = []
    else:
        conditions = sorted(sub_df["condition"].unique())
    n_conds = max(1, len(conditions))

    fig, axes = plt.subplots(1, n_conds, figsize=(3.2 * n_conds, 5.5), dpi=300, sharey=True)
    if n_conds == 1:
        axes = [axes]

    if not conditions or sub_df.empty:
        axes[0].text(0.5, 0.5, "No PASS pairs data", ha="center", va="center")
    else:
        for idx, cond in enumerate(conditions):
            ax = axes[idx]
            c_sub = sub_df[
                (sub_df["condition"] == cond)
                & (sub_df["t0_volume_sphere_um3"] > 0)
                & (sub_df["t7_volume_sphere_um3"] > 0)
            ]
            color = CONDITION_COLORS.get(cond, "#3498db")

            for _, row in c_sub.iterrows():
                v0 = row["t0_volume_sphere_um3"]
                v7 = row["t7_volume_sphere_um3"]
                is_growth = v7 >= v0
                line_color = color if is_growth else "#95a5a6"
                ax.plot([0, 7], [v0, v7], color=line_color, alpha=0.7, linewidth=1.2, marker="o", markersize=5)

            ax.set_title(f"{cond}\n(n={len(c_sub)})", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 7])
            ax.set_xticklabels(["Day 0", "Day 7"], fontweight="bold")
            ax.set_yscale("log")
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[0].set_ylabel("Volume ($\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")

    fig.suptitle("Individual Spheroid Volume Trajectories ($t_0 \\to t_7$ PASS-Only)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def plot_circularity_sensitivity_audit(
    objects_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
) -> Dict[str, Path]:
    """Generates circularity sensitivity audit scatter plot."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    if objects_df.empty:
        ax.text(0.5, 0.5, "No objects data available", ha="center", va="center")
    else:
        valid_objs = objects_df[
            np.isfinite(objects_df["circularity"])
            & np.isfinite(objects_df["volume_discrepancy_ratio"])
            & (objects_df["circularity"] > 0)
        ].copy()

        pass_objs = valid_objs[valid_objs["qc_flag"] == "PASS"]
        review_objs = valid_objs[valid_objs["qc_flag"] == "REVIEW"]

        if not review_objs.empty:
            ax.scatter(
                review_objs["circularity"],
                review_objs["volume_discrepancy_ratio"] * 100.0,
                alpha=0.5,
                color=QC_COLORS_RGB["REVIEW"],
                s=25,
                label=f"REVIEW (n={len(review_objs)})",
                zorder=2,
            )

        if not pass_objs.empty:
            ax.scatter(
                pass_objs["circularity"],
                pass_objs["volume_discrepancy_ratio"] * 100.0,
                alpha=0.7,
                color=QC_COLORS_RGB["PASS"],
                s=35,
                label=f"PASS (n={len(pass_objs)})",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        ax.axvline(0.65, color="#e67e22", linestyle="--", linewidth=1.5, label="Circularity Gate (0.65)", zorder=1)
        ax.axhline(15.0, color="#e74c3c", linestyle=":", linewidth=1.5, label="15% Discrepancy Margin", zorder=1)

        ax.set_xlabel("Morphological Circularity ($4\\pi A / P^2$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Volume Discrepancy $|V_{\\mathrm{sph}} - V_{\\mathrm{ell}}| / V_{\\mathrm{sph}}$ (%)", fontsize=11, fontweight="bold")
        ax.set_title("Circularity Assumption Audit: Spherical vs Ellipsoidal Volume", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path

    plt.close(fig)
    return results


def generate_all_figures(
    pairs_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    output_dir: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
) -> Dict[str, Path]:
    """
    Master figure generation function producing all publication-quality plots:
      1. fold_change_violin.png/.pdf (PASS-only paired fold changes)
      2. population_volume_distributions.png/.pdf (Unpaired V0 vs V7 distributions with Mann-Whitney U test stats)
      3. v0_vs_v7_scatter.png/.pdf (Log-log scatter of V0 vs V7 with identity line)
      4. growth_slopegraph.png/.pdf (Paired slopegraph trajectories)
      5. circularity_sensitivity_audit.png/.pdf (Volume discrepancy vs circularity)
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 1. Fold change violin plot (PASS-only)
    fc_paths = plot_fold_change_violin(
        pairs_df=pairs_df,
        output_base_path=fig_dir / "fold_change_violin",
        save_pdf=save_pdf,
        save_png=save_png,
        pass_only=True,
    )

    # 2. Population volume distributions (Unpaired V0 vs V7)
    pop_paths = plot_population_volume_distributions(
        objects_df=objects_df,
        output_base_path=fig_dir / "population_volume_distributions",
        save_pdf=save_pdf,
        save_png=save_png,
        pass_only=True,
    )

    # 3. V0 vs V7 scatter plot
    v0_v7_paths = plot_v0_vs_v7_scatter(
        pairs_df=pairs_df,
        output_base_path=fig_dir / "v0_vs_v7_scatter",
        save_pdf=save_pdf,
        save_png=save_png,
        pass_only=True,
    )

    # 4. Growth slopegraph
    slope_paths = plot_growth_slopegraph(
        pairs_df=pairs_df,
        output_base_path=fig_dir / "growth_slopegraph",
        save_pdf=save_pdf,
        save_png=save_png,
        pass_only=True,
    )

    # 5. Circularity sensitivity audit
    circ_paths = plot_circularity_sensitivity_audit(
        objects_df=objects_df,
        output_base_path=fig_dir / "circularity_sensitivity_audit",
        save_pdf=save_pdf,
        save_png=save_png,
    )

    all_figs: Dict[str, Dict[str, Path]] = {
        "fold_change_violin": fc_paths,
        "population_volume_distributions": pop_paths,
        "v0_vs_v7_scatter": v0_v7_paths,
        "growth_slopegraph": slope_paths,
        "circularity_sensitivity_audit": circ_paths,
    }

    return all_figs


def generate_condition_contact_sheets(
    overlays_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_df: pd.DataFrame,
    conditions: Sequence[str] = ("Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"),
) -> Dict[str, Path]:
    """Generates tiled multi-image contact sheets per condition."""
    return generate_all_contact_sheets(overlays_dir, output_dir, conditions)


class Visualizer:
    """Helper class providing object-oriented visualization methods."""

    def __init__(self, config: Optional[ArtifactConfig] = None):
        self.config = config or ArtifactConfig()

    @staticmethod
    def create_overlay(
        image: np.ndarray,
        labeled_mask: np.ndarray,
        objects: Sequence[Union[SpheroidObjectRecord, Dict[str, Any]]],
        output_path: Union[str, Path],
        pixel_size_um: float = 1.518817,
    ) -> Path:
        return create_overlay(image, labeled_mask, objects, output_path, pixel_size_um)

    @staticmethod
    def create_contact_sheet(
        overlay_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        grid_cols: int = 4,
        thumb_size: Tuple[int, int] = (400, 300),
        title: Optional[str] = None,
    ) -> Optional[Path]:
        return create_contact_sheet(overlay_paths, output_path, grid_cols, thumb_size, title)

    @staticmethod
    def generate_all_figures(
        pairs_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        output_dir: Union[str, Path],
        save_pdf: bool = True,
        save_png: bool = True,
    ) -> Dict[str, Path]:
        return generate_all_figures(pairs_df, objects_df, output_dir, save_pdf, save_png)
