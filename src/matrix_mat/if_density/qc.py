"""
Quality Control (QC) gating, artifact flagging, and mask deduplication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from skimage import measure

logger = logging.getLogger("matrix_mat.if_density.qc")


@dataclass
class ImageQCMetrics:
    image_id: str
    material: str
    replicate: str
    n_nuclei_raw: int
    n_nuclei_dedup: int
    n_duplicates_removed: int
    nuclear_snr: float
    nuclear_laplacian_var: float
    nuclear_saturation_frac: float
    actin_saturation_frac: float
    edge_touching_frac: float
    debris_count: int
    merged_count: int
    # Boolean Flags
    flag_empty_seg: bool
    flag_count_outlier: bool
    flag_dup_masks: bool
    flag_debris_or_merged: bool
    flag_edge_heavy: bool
    flag_saturated: bool
    flag_blurry: bool
    flag_low_snr: bool
    flag_any: bool
    flags_list: str
    flag_density_outlier: bool = False
    nuclei_per_mm2: float = float("nan")
    n_nuclei_valid: int = 0
    n_border_excluded: int = 0
    debris_count_pre: int = 0
    merged_count_pre: int = 0


def deduplicate_masks(
    masks: Union[np.ndarray, List[np.ndarray]],
    iou_threshold: float = 0.50,
) -> Tuple[np.ndarray, int]:
    """
    Check for overlapping mask instances with IoU > iou_threshold.
    Deduplicates by discarding the smaller instance and keeping the larger.
    Supports candidate proposals passed as a List[np.ndarray], a 3D boolean/integer
    array of shape (N, H, W), or a 2D integer label image.

    Returns (deduplicated_masks, n_duplicates_removed).
    """
    if isinstance(masks, list):
        if not masks:
            return np.zeros((100, 100), dtype=np.int32), 0
        h, w = masks[0].shape[:2]
        masks_list = [(m > 0) for m in masks]
    elif isinstance(masks, np.ndarray) and masks.ndim == 3:
        n, h, w = masks.shape
        masks_list = [(masks[i] > 0) for i in range(n)]
    elif isinstance(masks, np.ndarray) and masks.ndim == 2:
        if masks.max() == 0:
            return masks.copy(), 0
        unique_labels = [int(lbl) for lbl in np.unique(masks) if lbl > 0]
        if len(unique_labels) <= 1:
            return masks.copy(), 0
        # On standard single-layer 2D label masks, pixels are mutually exclusive
        return masks.copy(), 0
    else:
        raise ValueError(f"Unsupported mask format: {type(masks)}")

    n_candidates = len(masks_list)
    if n_candidates <= 1:
        clean = np.zeros((h, w), dtype=np.int32)
        if n_candidates == 1:
            clean[masks_list[0]] = 1
        return clean, 0

    candidate_props = []
    for idx, m in enumerate(masks_list):
        coords = np.argwhere(m)
        if coords.size == 0:
            continue
        minr, minc = coords.min(axis=0)
        maxr, maxc = coords.max(axis=0) + 1
        area = int(np.sum(m))
        candidate_props.append({
            "idx": idx,
            "mask": m,
            "bbox": (minr, minc, maxr, maxc),
            "area": area,
        })

    candidate_props.sort(key=lambda p: p["area"], reverse=True)
    duplicates_to_remove: Set[int] = set()

    for i in range(len(candidate_props)):
        p1 = candidate_props[i]
        idx1 = p1["idx"]
        if idx1 in duplicates_to_remove:
            continue
        minr1, minc1, maxr1, maxc1 = p1["bbox"]
        area1 = p1["area"]

        for j in range(i + 1, len(candidate_props)):
            p2 = candidate_props[j]
            idx2 = p2["idx"]
            if idx2 in duplicates_to_remove:
                continue
            minr2, minc2, maxr2, maxc2 = p2["bbox"]
            area2 = p2["area"]

            if not (minr1 < maxr2 and maxr1 > minr2 and minc1 < maxc2 and maxc1 > minc2):
                continue

            sub_minr = max(minr1, minr2)
            sub_maxr = min(maxr1, maxr2)
            sub_minc = max(minc1, minc2)
            sub_maxc = min(maxc1, maxc2)

            crop1 = p1["mask"][sub_minr:sub_maxr, sub_minc:sub_maxc]
            crop2 = p2["mask"][sub_minr:sub_maxr, sub_minc:sub_maxc]

            intersection = int(np.logical_and(crop1, crop2).sum())
            if intersection == 0:
                continue

            union = area1 + area2 - intersection
            iou = intersection / float(union) if union > 0 else 0.0

            if iou > iou_threshold:
                duplicates_to_remove.add(idx2)

    clean_masks = np.zeros((h, w), dtype=np.int32)
    new_lbl = 1
    for p in candidate_props:
        if p["idx"] not in duplicates_to_remove:
            unclaimed = (clean_masks == 0) & p["mask"]
            if np.any(unclaimed):
                clean_masks[unclaimed] = new_lbl
                new_lbl += 1

    n_removed = len(duplicates_to_remove)
    if n_removed > 0:
        logger.info(f"Deduplicated {n_removed} masks with IoU > {iou_threshold}")

    return clean_masks, n_removed


def compute_image_qc_metrics(
    image_id: str,
    material: str,
    replicate: str,
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    raw_masks: np.ndarray,
    dedup_masks: np.ndarray,
    n_dups_removed: int,
    pixel_size_um: float,
    qc_config: Dict[str, Any],
    nuclei_per_mm2: float = float("nan"),
    n_nuclei_valid: Optional[int] = None,
    n_border_excluded: int = 0,
    debris_count_pre: Optional[int] = None,
    merged_count_pre: Optional[int] = None,
) -> ImageQCMetrics:
    """Compute per-image numeric QC metrics and individual flags (except cohort-relative outlier flags)."""
    h, w = nuclear_img.shape[:2]

    # 1. Saturation
    nuc_sat = float(np.mean(nuclear_img >= 255))
    act_sat = float(np.mean(actin_img >= 255))
    sat_thresh = float(qc_config.get("saturation_fraction_max", 0.005))
    flag_saturated = (nuc_sat > sat_thresh) or (act_sat > sat_thresh)

    # 2. Focus / Blur (Laplacian variance)
    lap_var = float(cv2.Laplacian(nuclear_img, cv2.CV_64F).var())

    # 3. SNR (Signal to Background Ratio)
    p50 = float(np.percentile(nuclear_img, 50))
    p95 = float(np.percentile(nuclear_img, 95))
    snr = float((p95 + 1e-4) / (p50 + 1e-4))
    snr_thresh = float(qc_config.get("snr_min", 1.5))
    flag_low_snr = snr < snr_thresh

    # 4. Count & Empty Segmentation
    n_raw = int(raw_masks.max())
    n_dedup = int(dedup_masks.max())
    if n_nuclei_valid is None:
        n_nuclei_valid = n_dedup - n_border_excluded

    flag_empty_seg = (n_dedup == 0) and (snr >= snr_thresh)
    dup_thresh = float(qc_config.get("duplicate_iou_threshold", 0.50))
    flag_dup_masks = n_dups_removed > 0

    # 5. Mask size / debris / merged & edge touching
    min_vol = float(qc_config.get("min_nuclear_vol_um3", 150.0))
    max_vol = float(qc_config.get("max_nuclear_vol_um3", 4000.0))
    margin = int(qc_config.get("border_margin_px", 5))

    props = measure.regionprops(dedup_masks)
    debris_count = 0
    merged_count = 0
    edge_touching_count = 0

    from matrix_mat.core.units import compute_spherical_volume
    for p in props:
        area_px = p.area
        area_um2 = area_px * (pixel_size_um ** 2)
        eq_diam_um = 2.0 * np.sqrt(area_um2 / np.pi)
        vol_um3 = compute_spherical_volume(eq_diam_um)

        if vol_um3 < min_vol:
            debris_count += 1
        elif vol_um3 > max_vol:
            merged_count += 1

        minr, minc, maxr, maxc = p.bbox
        if minr <= margin or minc <= margin or maxr >= (h - margin) or maxc >= (w - margin):
            edge_touching_count += 1

    # Pre-post debris/merged counts
    if debris_count_pre is None or merged_count_pre is None:
        raw_props = measure.regionprops(raw_masks)
        raw_debris = 0
        raw_merged = 0
        for rp in raw_props:
            r_area_um2 = rp.area * (pixel_size_um ** 2)
            r_eq_diam_um = 2.0 * np.sqrt(r_area_um2 / np.pi)
            r_vol_um3 = compute_spherical_volume(r_eq_diam_um)
            if r_vol_um3 < min_vol:
                raw_debris += 1
            elif r_vol_um3 > max_vol:
                raw_merged += 1
        debris_count_pre = raw_debris
        merged_count_pre = raw_merged

    edge_touching_frac = edge_touching_count / max(1, n_dedup) if n_dedup > 0 else 0.0
    edge_thresh = float(qc_config.get("edge_heavy_fraction", 0.25))
    flag_edge_heavy = edge_touching_frac > edge_thresh
    flag_debris_or_merged = (debris_count > 0) or (merged_count > 0)

    return ImageQCMetrics(
        image_id=image_id,
        material=material,
        replicate=replicate,
        n_nuclei_raw=n_raw,
        n_nuclei_dedup=n_dedup,
        n_duplicates_removed=n_dups_removed,
        nuclear_snr=round(snr, 3),
        nuclear_laplacian_var=round(lap_var, 3),
        nuclear_saturation_frac=round(nuc_sat, 5),
        actin_saturation_frac=round(act_sat, 5),
        edge_touching_frac=round(edge_touching_frac, 3),
        debris_count=debris_count,
        merged_count=merged_count,
        flag_empty_seg=flag_empty_seg,
        flag_count_outlier=False,  # Evaluated in cohort pass
        flag_dup_masks=flag_dup_masks,
        flag_debris_or_merged=flag_debris_or_merged,
        flag_edge_heavy=flag_edge_heavy,
        flag_saturated=flag_saturated,
        flag_blurry=False,  # Evaluated in cohort pass
        flag_low_snr=flag_low_snr,
        flag_any=False,
        flags_list="",
        flag_density_outlier=False,  # Evaluated in cohort pass
        nuclei_per_mm2=round(nuclei_per_mm2, 2) if not np.isnan(nuclei_per_mm2) else float("nan"),
        n_nuclei_valid=n_nuclei_valid,
        n_border_excluded=n_border_excluded,
        debris_count_pre=debris_count_pre,
        merged_count_pre=merged_count_pre,
    )


def apply_cohort_qc_flags(
    metrics_list: List[ImageQCMetrics],
    qc_config: Dict[str, Any],
) -> List[ImageQCMetrics]:
    """
    Apply cohort-relative flags:
    - blurry: Laplacian variance below blur_percentile_cutoff (default 10%) of cohort
    - density_outlier (primary): Nuclei density outside 1.5 * IQR within material cohort (skipped if n < 3)
    - count_outlier (reference): Raw count outside 1.5 * IQR within material cohort (skipped if n < 3)
    """
    if not metrics_list:
        return metrics_list

    # 1. Blurry threshold (cohort-wide Laplacian variance percentile or absolute threshold)
    blur_mode = qc_config.get("blur_mode", "absolute")
    blur_abs_threshold = qc_config.get("blur_abs_threshold", None)
    if blur_mode == "cohort_percentile":
        blur_cutoff_pct = float(qc_config.get("blur_percentile_cutoff", 10.0))
        all_laps = [m.nuclear_laplacian_var for m in metrics_list]
        blur_threshold = float(np.percentile(all_laps, blur_cutoff_pct)) if all_laps else 0.0
        logger.warning(
            f"blur_mode=cohort_percentile flags the lowest {blur_cutoff_pct:.0f}% of fields by construction"
        )
    elif blur_abs_threshold is None:
        raise ValueError(
            "qc.blur_abs_threshold must be set (absolute Laplacian variance of the raw nuclear channel), "
            "or choose qc.blur_mode: cohort_percentile explicitly"
        )
    else:
        blur_threshold = float(blur_abs_threshold)

    # 2. Density and count outlier by material
    by_material: Dict[str, List[ImageQCMetrics]] = {}
    for m in metrics_list:
        by_material.setdefault(m.material, []).append(m)

    density_outlier_ids: Set[str] = set()
    count_outlier_ids: Set[str] = set()
    outlier_iqr_factor = float(qc_config.get("outlier_iqr_factor", 1.5))

    for mat, group in by_material.items():
        if len(group) < 3:
            logger.info(
                f"Skipping IQR outlier flags for material '{mat}' (n={len(group)} < 3 images; insufficient for IQR)."
            )
            continue

        # Density outlier (reported separately; evaluated on planar areal density nuclei_per_mm2)
        dens_vals = []
        for m in group:
            val = m.nuclei_per_mm2
            dens_vals.append(val if (val is not None and not np.isnan(val) and val > 0) else np.nan)
        valid_densities = [d for d in dens_vals if not np.isnan(d)]
        if len(valid_densities) >= 3:
            q25_d, q75_d = np.percentile(valid_densities, [25, 75])
            iqr_d = q75_d - q25_d
            low_d = q25_d - outlier_iqr_factor * iqr_d
            high_d = q75_d + outlier_iqr_factor * iqr_d
            for idx, m in enumerate(group):
                d = dens_vals[idx]
                if not np.isnan(d) and (d < low_d or d > high_d):
                    density_outlier_ids.add(m.image_id)

        # Count outlier (for reference)
        counts = [m.n_nuclei_dedup for m in group]
        q25_c, q75_c = np.percentile(counts, [25, 75])
        iqr_c = q75_c - q25_c
        low_c = q25_c - outlier_iqr_factor * iqr_c
        high_c = q75_c + outlier_iqr_factor * iqr_c
        for m in group:
            if m.n_nuclei_dedup < low_c or m.n_nuclei_dedup > high_c:
                count_outlier_ids.add(m.image_id)

    # 3. Apply flags
    for m in metrics_list:
        m.flag_blurry = bool(m.nuclear_laplacian_var < blur_threshold)
        m.flag_density_outlier = bool(m.image_id in density_outlier_ids)
        m.flag_count_outlier = bool(m.image_id in count_outlier_ids)

        active_flags = []
        artifact_flags = []
        if m.flag_empty_seg:
            active_flags.append("empty_seg")
            artifact_flags.append("empty_seg")
        if m.flag_density_outlier:
            active_flags.append("density_outlier")
        if m.flag_count_outlier:
            active_flags.append("count_outlier")
        if m.flag_dup_masks:
            active_flags.append("dup_masks")
            artifact_flags.append("dup_masks")
        if m.flag_debris_or_merged:
            active_flags.append("debris_or_merged")
            artifact_flags.append("debris_or_merged")
        if m.flag_edge_heavy:
            active_flags.append("edge_heavy")
            artifact_flags.append("edge_heavy")
        if m.flag_saturated:
            active_flags.append("saturated")
            artifact_flags.append("saturated")
        if m.flag_blurry:
            active_flags.append("blurry")
            artifact_flags.append("blurry")
        if m.flag_low_snr:
            active_flags.append("low_snr")
            artifact_flags.append("low_snr")

        # flag_any excludes endpoint flags (density_outlier, count_outlier)
        m.flag_any = len(artifact_flags) > 0
        m.flags_list = ";".join(active_flags)

    return metrics_list


def export_qc_flags_csv(
    metrics_list: List[ImageQCMetrics],
    output_path: Path,
) -> pd.DataFrame:
    """Save qc_flags.csv containing all numeric metrics and boolean flags."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.__dict__ for m in metrics_list]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved QC metrics for {len(df)} images to {output_path}")
    return df
