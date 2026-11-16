"""
Visualization & Supervision Artifacts Module for Spheroid Volume Pipeline v2.

Provides:
  - High-contrast RGB supervision overlays with labeled contours (Green=PASS, Amber=REVIEW, Red=FAIL),
    centroid indicators, and legible multi-line text annotations (ID, diameter d in um, QC flag).
  - Multi-image contact sheets grouping FOVs by condition and replicate.
  - Publication-quality figures (both high-res 300 DPI PNG and vector PDF):
    * fold_change_violin.png / .pdf: Log-scale violin plots of PASS-only pairs across conditions.
    * population_volume_distributions.png / .pdf: Unpaired population-level volume distributions per
      timepoint, annotated with the condition summary test (cluster permutation) and growth ratios.
    * v0_vs_v7_scatter.png / .pdf: Log-log scatter plot of V0 vs V7 with y = x identity line.
    * growth_slopegraph.png / .pdf: Paired trajectory slopegraphs (t0 -> t7).
    * circularity_sensitivity_audit.png / .pdf: Volume discrepancy ratio vs circularity audit.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from matrix_mat.brightfield.config import ArtifactConfig, PipelineConfig  # noqa: F401  (re-exported by legacy shims)
from matrix_mat.brightfield.measure import SpheroidObjectRecord
from matrix_mat.brightfield.pair import SpheroidPairRecord  # noqa: F401  (re-exported by legacy shims)
from matrix_mat.reporting._figures import closes_new_figures, ordered_values, timepoint_order

logger = logging.getLogger("matrix_mat.reporting.visualize")

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
    img_u8: np.ndarray
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
    elif image.dtype == np.uint8:
        img_u8 = np.asarray(image, dtype=np.uint8)
    else:
        # 16-bit / signed / bool images: scale by the actual intensity range (astype would wrap).
        arr = image.astype(np.float64)
        lo, hi = float(np.min(arr)) if arr.size else 0.0, float(np.max(arr)) if arr.size else 0.0
        if hi > lo:
            img_u8 = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(arr, 0, 255).astype(np.uint8)

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

    max_label = int(np.max(labeled_mask)) if labeled_mask.size > 0 else 0
    present_labels = set(int(v) for v in np.unique(labeled_mask)) if labeled_mask.size > 0 else set()

    # Map records to mask labels. FAIL records and records whose label is not in this mask are
    # ignored, so a rejected candidate can never re-colour a label that belongs to another object.
    label_to_obj: Dict[int, Dict[str, Any]] = {}
    for obj in obj_dicts:
        try:
            l_idx = int(obj.get("label_idx", -1))
        except (TypeError, ValueError):
            continue
        if l_idx <= 0 or l_idx not in present_labels:
            continue
        if str(obj.get("qc_flag", "")).upper() == "FAIL":
            continue
        label_to_obj[l_idx] = obj

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
    dpi: Optional[int] = None,
) -> Optional[Path]:
    """Generates multi-image montage contact sheet (``dpi`` is stored in the PNG metadata)."""
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

    save_kw: Dict[str, Any] = {"quality": 92}
    if dpi:
        save_kw["dpi"] = (int(dpi), int(dpi))
    canvas.save(str(out_path), **save_kw)
    return out_path


def _condition_of_overlay(path: Path) -> Optional[str]:
    from matrix_mat.core.naming import parse_image_filename

    try:
        return str(parse_image_filename(_overlay_image_id(path) + ".tif")["condition"])
    except ValueError:
        return None


def _overlay_image_id(path: Path) -> str:
    stem = path.stem
    for suffix in ("_overlay", "_supervision"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def generate_all_contact_sheets(
    overlays_dir: Union[str, Path],
    output_dir: Union[str, Path],
    conditions: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    grid_cols: int = 4,
    overlay_paths: Optional[Sequence[Union[str, Path]]] = None,
    image_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Path]:
    """Generates condition-specific contact sheets.

    ``overlay_paths`` (this run's overlays) is used when given; otherwise ``overlays_dir`` is
    globbed, restricted to ``image_ids`` when those are given (so stale overlays from earlier runs
    do not leak in). Conditions are parsed from the overlay file names; ``conditions`` only sets
    the display order (others found in the data follow). Unparseable names go to an ``Other`` sheet.
    """
    ov_dir = Path(overlays_dir)
    out_dir = Path(output_dir) / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if overlay_paths is not None:
        candidates = [Path(p) for p in overlay_paths]
    else:
        candidates = sorted(ov_dir.glob("*.png"))
        if image_ids is not None:
            wanted = {str(i) for i in image_ids}
            candidates = [p for p in candidates if _overlay_image_id(p) in wanted]

    grouped: Dict[str, List[Path]] = {}
    for p in candidates:
        grouped.setdefault(_condition_of_overlay(p) or "Other", []).append(p)

    order = ordered_values([k for k in grouped if k != "Other"], known=conditions)
    if "Other" in grouped:
        order.append("Other")

    results: Dict[str, Path] = {}
    for cond in order:
        files = grouped[cond]
        cs_path = out_dir / f"{cond}_contact_sheet.png"
        res = create_contact_sheet(
            overlay_paths=files,
            output_path=cs_path,
            grid_cols=grid_cols,
            thumb_size=(400, 300),
            title=f"Spheroid Quality Control Supervision Grid — Condition: {cond} ({len(files)} FOVs)",
            dpi=dpi,
        )
        if res is not None:
            results[cond] = res

    return results


def _save(fig: Any, base_path: Path, save_png: bool, save_pdf: bool, dpi: Optional[int]) -> Dict[str, Path]:
    results: Dict[str, Path] = {}
    if save_png:
        png_path = base_path.with_suffix(".png")
        fig.savefig(png_path, dpi=int(dpi or 300), bbox_inches="tight")
        results["png"] = png_path
    if save_pdf:
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        results["pdf"] = pdf_path
    return results


def _finite_positive(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def _gate_from_config(config: Any) -> Optional[float]:
    for section in ("pairing", "stats"):
        v = getattr(getattr(config, section, None), "min_t0_d_um_denominator", None)
        if v is not None:
            return float(v)
    return None


@closes_new_figures
def plot_fold_change_violin(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
    min_t0_d_um: Optional[float] = None,
    dpi: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Generates publication violin plot of volume fold change (V7 / V0) across conditions
    strictly from PASS-only mutually-paired objects passing the denominator gate.
    ``min_t0_d_um`` is the run's Day-0 denominator gate, shown in the title when given.
    Conditions without data are labelled ``n=0`` and get no violin.
    """
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=int(dpi or 300))

    valid_pairs = pairs_df.copy()
    if pass_only and "combined_qc_flag" in valid_pairs.columns:
        valid_pairs = valid_pairs[valid_pairs["combined_qc_flag"] == "PASS"]
    if "denominator_gate_pass" in valid_pairs.columns:
        valid_pairs = valid_pairs[valid_pairs["denominator_gate_pass"] == True]  # noqa: E712

    if "condition" in valid_pairs.columns and not valid_pairs.empty and "fold_change_volume" in valid_pairs.columns:
        conditions = ordered_values(valid_pairs["condition"].unique())
    else:
        conditions = []

    if not conditions:
        ax.text(0.5, 0.5, "No PASS-verified matched pairs data available\n(See Population-Level Distribution Analysis for full cohort)", ha="center", va="center", transform=ax.transAxes, fontsize=11)
    else:
        per_cond = {c: _finite_positive(valid_pairs.loc[valid_pairs["condition"] == c, "fold_change_volume"]) for c in conditions}
        labels = [f"{c}\n(n={len(per_cond[c])})" for c in conditions]
        positions = [i for i, c in enumerate(conditions, 1) if len(per_cond[c]) > 0]
        dataset = [per_cond[c] for c in conditions if len(per_cond[c]) > 0]

        if dataset:
            parts = ax.violinplot(dataset, positions=positions, showmeans=False, showmedians=True, showextrema=True)
            drawn = [c for c in conditions if len(per_cond[c]) > 0]
            for idx, pc in enumerate(list(parts["bodies"])):  # type: ignore[call-overload]
                pc.set_facecolor(CONDITION_COLORS.get(drawn[idx], "#3498db"))
                pc.set_edgecolor("#2c3e50")
                pc.set_alpha(0.65)
                pc.set_linewidth(1.2)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#2c3e50")
                parts["cmedians"].set_linewidth(2.0)
            for key in ("cbars", "cmaxes", "cmins"):
                if key in parts:
                    parts[key].set_color("#7f8c8d")

        rng = np.random.default_rng(42)
        for i, c in enumerate(conditions, 1):
            fc_vals = per_cond[c]
            if len(fc_vals) > 0:
                jitter = rng.normal(0, 0.05, size=len(fc_vals))
                ax.scatter(i + jitter, fc_vals, alpha=0.75, color="#2c3e50", s=32, zorder=4, edgecolor="white", linewidth=0.5)
            else:
                ax.annotate("n=0", xy=(i, 0.5), xycoords=("data", "axes fraction"), ha="center", va="center",
                            fontsize=10, color="#7f8c8d")

        ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="No Growth ($V_7 = V_0$)", zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(0.4, len(conditions) + 0.6)
        ax.set_xticks(range(1, len(conditions) + 1))
        ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    gate = f"$d_0 \\geq {min_t0_d_um:g}\\,\\mu\\mathrm{{m}}$" if min_t0_d_um is not None else "denominator-gated"
    ax.set_ylabel("Volume Fold Change ($V_7 / V_0$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title(f"Paired Spheroid Volume Fold Change (PASS-Only Trajectories, {gate})", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    return _save(fig, base_path, save_png, save_pdf, dpi)


@closes_new_figures
def plot_population_volume_distributions(
    objects_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
    condition_df: Optional[pd.DataFrame] = None,
    dpi: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Generates publication-quality figure comparing full unpaired spheroid volume distributions
    per timepoint for each hydrogel formulation, annotating precomputed condition summary statistics.

    Conditions (known order first, then any others) and timepoints (by day number) are taken
    from the data. Empty condition/timepoint groups get an ``n=0`` label, never a fake box.
    """
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = objects_df[objects_df["qc_flag"] == "PASS"].copy() if pass_only and "qc_flag" in objects_df.columns else objects_df.copy()

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=int(dpi or 300))

    if sub_df.empty or not {"condition", "volume_sphere_um3", "timepoint"}.issubset(sub_df.columns):
        ax.text(0.5, 0.5, "No population volume data available", ha="center", va="center", transform=ax.transAxes, fontsize=11)
        return _save(fig, base_path, save_png, save_pdf, dpi)

    present_conditions = ordered_values(sub_df["condition"].unique())
    tps = timepoint_order(sub_df["timepoint"].unique())
    tp_palette = ["#3498db", "#e67e22", "#27ae60", "#8e44ad", "#c0392b", "#7f8c8d"]
    tp_colors = {tp: tp_palette[i % len(tp_palette)] for i, tp in enumerate(tps)}
    step = 0.8
    group_w = step * len(tps) + 0.9

    x_positions: List[float] = []
    box_data: List[np.ndarray] = []
    box_colors: List[str] = []
    labels: List[str] = []
    mid_positions: List[float] = []

    for idx, cond in enumerate(present_conditions):
        c_df = sub_df[sub_df["condition"] == cond]
        per_tp = {tp: _finite_positive(c_df.loc[c_df["timepoint"] == tp, "volume_sphere_um3"]) for tp in tps}
        positions = {tp: idx * group_w + 1.0 + j * step for j, tp in enumerate(tps)}
        mid_positions.append(float(np.mean(list(positions.values()))))

        for tp in tps:
            if len(per_tp[tp]) > 0:
                x_positions.append(positions[tp])
                box_data.append(per_tp[tp])
                box_colors.append(tp_colors[tp])
            else:
                ax.annotate("n=0", xy=(positions[tp], 0.02), xycoords=("data", "axes fraction"), ha="center",
                            va="bottom", fontsize=7, color="#7f8c8d")

        v_first, v_last = per_tp[tps[0]], per_tp[tps[-1]]
        if len(tps) >= 2 and len(v_first) > 0 and len(v_last) > 0:
            first_med = float(np.median(v_first))
            last_med = float(np.median(v_last))
            fc = last_med / first_med if first_med > 0 else float("nan")

            annot_lines = [f"FC={fc:.2f}x ({tps[-1]}/{tps[0]})"]
            if condition_df is not None and not condition_df.empty and "condition" in condition_df.columns:
                cond_row = condition_df[condition_df["condition"] == cond]
                if not cond_row.empty:
                    def _get(col: str, row: pd.DataFrame = cond_row) -> Any:
                        if col in row.columns and pd.notna(row[col].values[0]):
                            return row[col].values[0]
                        return None

                    star = str(_get("significance") or "")
                    p_adj = _get("p_adj")
                    p_raw = _get("p_raw")
                    p_val, p_kind = (p_adj, "Holm-adjusted") if p_adj is not None else (
                        (p_raw, "unadjusted") if p_raw is not None else (None, ""))
                    n_gels = _get("n_gels")
                    if n_gels is None:
                        n_gels = len(c_df["replicate"].dropna().unique()) if "replicate" in c_df.columns else 1
                    if star and star != "ns":
                        annot_lines.insert(0, star)
                    if p_val is not None:
                        p_f = float(p_val)
                        p_str = "p<0.001" if p_f < 0.001 else f"p={p_f:.3f}"
                        annot_lines.append(f"({p_str}, {p_kind})")
                    if int(n_gels) >= 2:
                        annot_lines.append(f"Cluster Perm (n={int(n_gels)} gels)")
                    else:
                        annot_lines.append(f"insufficient rep (n={int(n_gels)})")

            y_max = max(np.percentile(v, 95) for v in per_tp.values() if len(v) > 0) * 2.0
            x0, x1 = positions[tps[0]], positions[tps[-1]]
            ax.plot([x0, x1], [y_max, y_max], color="#2c3e50", linewidth=1.2)
            ax.text((x0 + x1) / 2, y_max * 1.15, "\n".join(annot_lines), ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2c3e50")

        labels.append(str(cond))

    if box_data:
        bplot = ax.boxplot(
            box_data,
            positions=x_positions,
            patch_artist=True,
            widths=0.6,
            showmeans=False,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.3, markerfacecolor="#7f8c8d", markeredgecolor="none"),
        )
        for patch, color in zip(bplot["boxes"], box_colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("#2c3e50")
        for median in bplot["medians"]:
            median.set_color("#2c3e50")
            median.set_linewidth(2.0)
        ax.set_yscale("log")

    ax.set_xticks(mid_positions)
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tp_colors[tp], alpha=0.7, edgecolor="#2c3e50", label=str(tp)) for tp in tps
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    ax.set_ylabel("Spheroid Volume ($V_{\\mathrm{sphere}}$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title(f"Unpaired Population-Level Spheroid Volume Distributions ({', '.join(map(str, tps))})", fontsize=13, fontweight="bold", pad=14)
    ax.grid(True, linestyle=":", alpha=0.5, which="both")
    plt.tight_layout()
    return _save(fig, base_path, save_png, save_pdf, dpi)


@closes_new_figures
def plot_v0_vs_v7_scatter(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
    dpi: Optional[int] = None,
) -> Dict[str, Path]:
    """Generates publication log-log scatter plot of V0 vs V7 with identity line."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=int(dpi or 300))

    sub_df = pairs_df.copy()
    if pass_only and "combined_qc_flag" in sub_df.columns:
        sub_df = sub_df[sub_df["combined_qc_flag"] == "PASS"]

    req_cols = {"t0_volume_sphere_um3", "t7_volume_sphere_um3", "condition"}
    if not sub_df.empty and req_cols.issubset(sub_df.columns):
        v0 = pd.to_numeric(sub_df["t0_volume_sphere_um3"], errors="coerce")
        v7 = pd.to_numeric(sub_df["t7_volume_sphere_um3"], errors="coerce")
        sub_df = sub_df[(v0 > 0) & (v7 > 0) & np.isfinite(v0) & np.isfinite(v7)]

    if sub_df.empty or not req_cols.issubset(sub_df.columns):
        ax.text(0.5, 0.5, "No PASS pairs data available", ha="center", va="center", transform=ax.transAxes)
    else:
        for cond in ordered_values(sub_df["condition"].unique()):
            c_df = sub_df[sub_df["condition"] == cond]
            ax.scatter(
                c_df["t0_volume_sphere_um3"],
                c_df["t7_volume_sphere_um3"],
                label=f"{cond} (n={len(c_df)})",
                color=CONDITION_COLORS.get(cond, "#3498db"),
                alpha=0.75,
                s=45,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )

        all_v = np.concatenate([sub_df["t0_volume_sphere_um3"].values, sub_df["t7_volume_sphere_um3"].values]).astype(float)
        min_val = max(1e4, np.min(all_v) * 0.5)
        max_val = np.max(all_v) * 2.0

        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, label="Identity ($V_7 = V_0$)", zorder=2)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    ax.set_xlabel("Baseline Volume at Day 0 ($V_0$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Endpoint Volume at Day 7 ($V_7$, $\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title("Spheroid Growth Tracking: Baseline vs Endpoint Volume ($V_0$ vs $V_7$)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6, which="both")
    plt.tight_layout()
    return _save(fig, base_path, save_png, save_pdf, dpi)


@closes_new_figures
def plot_growth_slopegraph(
    pairs_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    pass_only: bool = True,
    dpi: Optional[int] = None,
) -> Dict[str, Path]:
    """Generates paired slopegraph trajectories."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    sub_df = pairs_df.copy()
    if pass_only and "combined_qc_flag" in sub_df.columns:
        sub_df = sub_df[sub_df["combined_qc_flag"] == "PASS"]

    req_cols = {"t0_volume_sphere_um3", "t7_volume_sphere_um3", "condition"}
    if sub_df.empty or not req_cols.issubset(sub_df.columns):
        conditions: List[Any] = []
    else:
        conditions = ordered_values(sub_df["condition"].unique())
    n_conds = max(1, len(conditions))

    fig, axes = plt.subplots(1, n_conds, figsize=(3.2 * n_conds, 5.5), dpi=int(dpi or 300), sharey=True)
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
                line_color = color if v7 >= v0 else "#95a5a6"
                ax.plot([0, 7], [v0, v7], color=line_color, alpha=0.7, linewidth=1.2, marker="o", markersize=5)

            ax.set_title(f"{cond}\n(n={len(c_sub)})", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 7])
            ax.set_xticklabels(["Day 0", "Day 7"], fontweight="bold")
            if len(c_sub):
                ax.set_yscale("log")
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[0].set_ylabel("Volume ($\\mu\\mathrm{m}^3$) [Log Scale]", fontsize=12, fontweight="bold")

    fig.suptitle("Individual Spheroid Volume Trajectories ($t_0 \\to t_7$ PASS-Only)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, base_path, save_png, save_pdf, dpi)


@closes_new_figures
def plot_circularity_sensitivity_audit(
    objects_df: pd.DataFrame,
    output_base_path: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    dpi: Optional[int] = None,
) -> Dict[str, Path]:
    """Generates circularity sensitivity audit scatter plot."""
    base_path = Path(output_base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=int(dpi or 300))

    needed = {"circularity", "volume_discrepancy_ratio", "qc_flag"}
    if objects_df.empty or not needed.issubset(objects_df.columns):
        ax.text(0.5, 0.5, "No objects data available", ha="center", va="center")
    else:
        circ = pd.to_numeric(objects_df["circularity"], errors="coerce")
        disc = pd.to_numeric(objects_df["volume_discrepancy_ratio"], errors="coerce")
        valid_objs = objects_df[np.isfinite(circ) & np.isfinite(disc) & (circ > 0)].copy()

        pass_objs = valid_objs[valid_objs["qc_flag"] == "PASS"]
        review_objs = valid_objs[valid_objs["qc_flag"] == "REVIEW"]

        if not review_objs.empty:
            ax.scatter(review_objs["circularity"], review_objs["volume_discrepancy_ratio"] * 100.0, alpha=0.5,
                       color=QC_COLORS_RGB["REVIEW"], s=25, label=f"REVIEW (n={len(review_objs)})", zorder=2)
        if not pass_objs.empty:
            ax.scatter(pass_objs["circularity"], pass_objs["volume_discrepancy_ratio"] * 100.0, alpha=0.7,
                       color=QC_COLORS_RGB["PASS"], s=35, label=f"PASS (n={len(pass_objs)})", edgecolor="white",
                       linewidth=0.5, zorder=3)

        ax.axvline(0.65, color="#e67e22", linestyle="--", linewidth=1.5, label="Circularity Gate (0.65)", zorder=1)
        ax.axhline(15.0, color="#e74c3c", linestyle=":", linewidth=1.5, label="15% Discrepancy Margin", zorder=1)

        ax.set_xlabel("Morphological Circularity ($4\\pi A / P^2$)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Volume Discrepancy $|V_{\\mathrm{sph}} - V_{\\mathrm{ell}}| / V_{\\mathrm{sph}}$ (%)", fontsize=11, fontweight="bold")
        ax.set_title("Circularity Assumption Audit: Spherical vs Ellipsoidal Volume", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    return _save(fig, base_path, save_png, save_pdf, dpi)


def generate_all_figures(
    pairs_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    output_dir: Union[str, Path],
    save_pdf: bool = True,
    save_png: bool = True,
    condition_df: Optional[pd.DataFrame] = None,
    config: Optional[Any] = None,
    min_t0_d_um: Optional[float] = None,
    dpi: Optional[int] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    Master figure generation function producing all publication-quality plots:
      1. fold_change_violin.png/.pdf (PASS-only paired fold changes)
      2. population_volume_distributions.png/.pdf (Unpaired volume distributions per timepoint)
      3. v0_vs_v7_scatter.png/.pdf (Log-log scatter of V0 vs V7 with identity line)
      4. growth_slopegraph.png/.pdf (Paired slopegraph trajectories)
      5. circularity_sensitivity_audit.png/.pdf (Volume discrepancy vs circularity)

    ``config`` (a PipelineConfig) supplies the denominator gate and ``artifacts.figure_dpi``;
    explicit ``min_t0_d_um`` / ``dpi`` arguments take precedence.
    """
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if config is not None:
        if min_t0_d_um is None:
            min_t0_d_um = _gate_from_config(config)
        if dpi is None:
            dpi = getattr(getattr(config, "artifacts", None), "figure_dpi", None)
    if min_t0_d_um is None and "min_t0_d_um_denominator" in getattr(pairs_df, "columns", []):
        vals = pd.to_numeric(pairs_df["min_t0_d_um_denominator"], errors="coerce").dropna().unique()
        if len(vals) == 1:
            min_t0_d_um = float(vals[0])

    if condition_df is None:
        cond_csv = out_dir / "condition_summary.csv"
        if cond_csv.exists():
            try:
                condition_df = pd.read_csv(cond_csv)
            except Exception:
                condition_df = None

    common: Dict[str, Any] = dict(save_pdf=save_pdf, save_png=save_png, dpi=dpi)
    return {
        "fold_change_violin": plot_fold_change_violin(
            pairs_df=pairs_df, output_base_path=fig_dir / "fold_change_violin", pass_only=True,
            min_t0_d_um=min_t0_d_um, **common),
        "population_volume_distributions": plot_population_volume_distributions(
            objects_df=objects_df, output_base_path=fig_dir / "population_volume_distributions", pass_only=True,
            condition_df=condition_df, **common),
        "v0_vs_v7_scatter": plot_v0_vs_v7_scatter(
            pairs_df=pairs_df, output_base_path=fig_dir / "v0_vs_v7_scatter", pass_only=True, **common),
        "growth_slopegraph": plot_growth_slopegraph(
            pairs_df=pairs_df, output_base_path=fig_dir / "growth_slopegraph", pass_only=True, **common),
        "circularity_sensitivity_audit": plot_circularity_sensitivity_audit(
            objects_df=objects_df, output_base_path=fig_dir / "circularity_sensitivity_audit", **common),
    }


def generate_condition_contact_sheets(
    overlays_dir: Union[str, Path],
    output_dir: Union[str, Path],
    manifest_df: pd.DataFrame,
    conditions: Optional[Sequence[str]] = None,
    dpi: Optional[int] = None,
    overlay_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> Dict[str, Path]:
    """Generates tiled multi-image contact sheets per condition (conditions taken from the data).

    Pass ``overlay_paths`` (e.g. ``run_state["overlay_paths"]``) to use exactly this run's overlays;
    otherwise overlays are globbed and restricted to ``manifest_df["image_id"]``.
    """
    cols = getattr(manifest_df, "columns", [])
    if conditions is None and manifest_df is not None and "condition" in cols:
        conditions = ordered_values(manifest_df["condition"].unique())
    image_ids = list(manifest_df["image_id"].astype(str)) if (overlay_paths is None and "image_id" in cols) else None
    return generate_all_contact_sheets(overlays_dir, output_dir, conditions, dpi=dpi,
                                       overlay_paths=overlay_paths, image_ids=image_ids)


class Visualizer:
    """Helper class providing object-oriented visualization methods."""

    def __init__(self, config: Optional[ArtifactConfig] = None):
        self.config = config or ArtifactConfig()

    def generate_figures(
        self,
        pairs_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        output_dir: Union[str, Path],
        condition_df: Optional[pd.DataFrame] = None,
        min_t0_d_um: Optional[float] = None,
    ) -> Dict[str, Dict[str, Path]]:
        """Generate figures respecting ArtifactConfig options (save_pdf/save_png/figure_dpi)."""
        return generate_all_figures(
            pairs_df,
            objects_df,
            output_dir,
            save_pdf=self.config.save_pdf,
            save_png=self.config.save_png,
            condition_df=condition_df,
            min_t0_d_um=min_t0_d_um,
            dpi=self.config.figure_dpi,
        )

    def create_contact_sheet_with_config(
        self,
        overlay_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        title: Optional[str] = None,
    ) -> Optional[Path]:
        """Create contact sheet using ArtifactConfig settings (columns and DPI)."""
        return create_contact_sheet(
            overlay_paths,
            output_path,
            grid_cols=self.config.contact_sheet_cols,
            title=title,
            dpi=self.config.contact_sheet_dpi,
        )

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
        condition_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Path]]:
        return generate_all_figures(pairs_df, objects_df, output_dir, save_pdf, save_png, condition_df=condition_df)
