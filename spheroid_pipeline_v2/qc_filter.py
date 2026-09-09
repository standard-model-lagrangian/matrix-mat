"""
Quality Control (QC) and Filtering Module for Spheroid Volume Pipeline v2 (spheroid_pipeline_v2).
Provides:
  - Overlap resolution for candidate segmentation proposals
  - Physical size gating in calibrated um: [min_d_um, max_d_um] (default 40 to 1500 um)
  - Border margin clearance gating: rejects objects touching or within border_margin_px (default 15 px)
  - Interior vs background annular ring contrast gating:
    R_i = Dilate(M_i) \\ (bigcup M_j)
    Rejects objects whose mean interior intensity is not at least N% (default 10%) darker than surrounding ring
  - Single mask max area plausibility gating (< 25% FOV area)
  - Full separation between FAIL (rejected from clean label map), REVIEW (retained, flagged for manual check), and PASS
  - Master entry point: filter_and_measure_objects
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from skimage import measure

from spheroid_pipeline_v2.config import PipelineConfig, QCConfig
from spheroid_pipeline_v2.measure import (
    SpheroidObjectRecord,
    evaluate_morphological_qc,
    measure_single_object,
)

logger = logging.getLogger("spheroid_pipeline_v2.qc_filter")


def resolve_overlaps(
    candidate_masks: Union[List[np.ndarray], np.ndarray],
    scores: Optional[List[float]] = None,
    overlap_threshold: float = 0.50,
) -> np.ndarray:
    """
    Resolve overlapping candidate instance proposals by area or confidence score.
    Proposals with >= overlap_threshold (default 50%) overlap with already claimed pixels
    are discarded as redundant duplicates. Proposals with < 50% overlap claim unclaimed pixels.

    Args:
        candidate_masks: List of 2D binary bool/uint8 masks, 3D array (N, H, W), or 2D labeled integer mask.
        scores: Optional list of confidence scores for each proposal. If None, sorts by mask area.
        overlap_threshold: Fractional overlap above which candidate is rejected (default 0.50).

    Returns:
        Partitioned 2D int32 label mask of shape (H, W) where 0=background, 1..K=distinct instances.
    """
    # Case 1: 2D labeled integer mask
    if isinstance(candidate_masks, np.ndarray) and candidate_masks.ndim == 2:
        h, w = candidate_masks.shape
        unique_labels = [int(lbl) for lbl in np.unique(candidate_masks) if lbl > 0]
        if not unique_labels:
            return np.zeros((h, w), dtype=np.int32)
        masks_list = [(candidate_masks == lbl) for lbl in unique_labels]
    # Case 2: 3D array (N, H, W)
    elif isinstance(candidate_masks, np.ndarray) and candidate_masks.ndim == 3:
        n, h, w = candidate_masks.shape
        masks_list = [(candidate_masks[i] > 0) for i in range(n)]
    # Case 3: List of 2D boolean/integer masks
    elif isinstance(candidate_masks, list):
        if not candidate_masks:
            return np.zeros((100, 100), dtype=np.int32)
        h, w = candidate_masks[0].shape[:2]
        masks_list = [(m > 0) for m in candidate_masks]
    else:
        raise ValueError(f"Unsupported candidate_masks format: {type(candidate_masks)}")

    if not masks_list:
        return np.zeros((h, w), dtype=np.int32)

    # Determine sorting order
    if scores is not None and len(scores) == len(masks_list):
        order = np.argsort(scores)[::-1]
    else:
        areas = [np.sum(m) for m in masks_list]
        order = np.argsort(areas)[::-1]

    master = np.zeros((h, w), dtype=np.int32)
    current_label = 1

    for idx in order:
        mask = masks_list[idx]
        total_px = np.sum(mask)
        if total_px == 0:
            continue

        overlap_px = np.sum(mask & (master > 0))
        overlap_frac = float(overlap_px) / float(total_px)

        # Reject duplicate if overlap exceeds threshold
        if overlap_frac >= overlap_threshold:
            continue

        # Claim unclaimed pixels
        unclaimed = mask & (master == 0)
        if np.sum(unclaimed) > 0:
            master[unclaimed] = current_label
            current_label += 1

    return master


def compute_annular_ring_contrast(
    image: np.ndarray,
    obj_mask: np.ndarray,
    all_objects_mask: np.ndarray,
    dilation_px: int = 15,
) -> Tuple[float, float, float]:
    """
    Compute local annular background ring R_i = Dilate(M_i) \\ (bigcup M_j) and contrast metrics.

    Args:
        image: 2D float32 image in range [0.0, 1.0].
        obj_mask: 2D binary mask for current object instance M_i.
        all_objects_mask: 2D binary mask of all segmented object instances bigcup M_j.
        dilation_px: Radius of structuring element for ring dilation (default 15 px).

    Returns:
        (mean_interior_intensity, mean_ring_intensity, contrast_ratio)
        where contrast_ratio = (mean_ring - mean_interior) / max(mean_ring, 1e-4)
    """
    img_f32 = image.astype(np.float32)
    m_i = (obj_mask > 0).astype(np.uint8)
    all_m = (all_objects_mask > 0).astype(np.uint8)

    # Structuring element for annular ring
    kernel_size = max(3, int(dilation_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(m_i, kernel)

    # Background ring: pixels in dilated mask excluding all object interiors
    ring_mask = (dilated > 0) & (all_m == 0)

    # Interior intensity
    if np.sum(m_i) > 0:
        mean_interior = float(np.mean(img_f32[m_i > 0]))
    else:
        mean_interior = 1.0

    # Ring intensity (with fallback to global median if no ring pixels available)
    if np.sum(ring_mask) > 0:
        mean_ring = float(np.mean(img_f32[ring_mask]))
    else:
        mean_ring = float(np.median(img_f32))

    denom = max(mean_ring, 1e-4)
    contrast_ratio = float((mean_ring - mean_interior) / denom)

    return mean_interior, mean_ring, contrast_ratio


def check_image_plausibility(
    clean_label_mask: np.ndarray,
    raw_label_mask: Optional[np.ndarray] = None,
    single_mask_max_area_pct: float = 0.25,
) -> Tuple[bool, str]:
    """
    Evaluate FOV plausibility:
    1. Must contain >= 1 valid object after QC gating.
    2. No single mask may occupy > single_mask_max_area_pct (default 25%) of FOV area.

    Returns:
        (is_plausible, status_reason)
    """
    h, w = clean_label_mask.shape[:2]
    total_px = float(h * w)
    max_allowed_px = single_mask_max_area_pct * total_px

    n_clean = int(np.max(clean_label_mask))
    if n_clean == 0:
        return False, "SEG_FAIL: 0 objects passed QC gating"

    # Check clean mask areas
    for obj_id in range(1, n_clean + 1):
        area = float(np.sum(clean_label_mask == obj_id))
        if area > max_allowed_px:
            pct = (area / total_px) * 100.0
            return False, f"SEG_FAIL: Single mask #{obj_id} occupies {pct:.1f}% FOV (> {single_mask_max_area_pct*100:.0f}% limit)"

    # Check raw mask areas if provided
    if raw_label_mask is not None:
        n_raw = int(np.max(raw_label_mask))
        for obj_id in range(1, n_raw + 1):
            area = float(np.sum(raw_label_mask == obj_id))
            if area > max_allowed_px:
                pct = (area / total_px) * 100.0
                return False, f"SEG_FAIL: Raw mask #{obj_id} occupies {pct:.1f}% FOV (> {single_mask_max_area_pct*100:.0f}% limit)"

    return True, f"PASS: {n_clean} plausible objects"


class QCFilter:
    """Quality Control and Morphological Filtering Engine."""

    def __init__(self, config: Optional[QCConfig] = None):
        self.config = config if config is not None else QCConfig()

    def filter_and_measure(
        self,
        image: np.ndarray,
        raw_label_mask: np.ndarray,
        image_metadata: Optional[Dict[str, Any]] = None,
        pixel_size_um: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[SpheroidObjectRecord]]:
        """
        Execute full QC filtering and measurement pipeline on segmented image.

        Args:
            image: 2D image (float32 [0, 1] or uint8/uint16).
            raw_label_mask: 2D labeled integer mask from segmentation stage.
            image_metadata: Metadata dictionary containing image IDs and well descriptors.
            pixel_size_um: Optional calibrated pixel size in um/px.

        Returns:
            clean_label_mask: (H, W) int32 array containing only PASS and REVIEW objects.
            object_records: Full list of 32-column SpheroidObjectRecord instances (including FAILs for auditing).
        """
        meta = dict(image_metadata or {})
        h, w = raw_label_mask.shape[:2]

        # Ensure image is 2D float32 normalized in [0, 1]
        if image.ndim == 3:
            if image.shape[2] == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                img_gray = image[:, :, 0]
        else:
            img_gray = image

        if img_gray.dtype == np.uint8:
            img_f32 = img_gray.astype(np.float32) / 255.0
        elif img_gray.dtype == np.uint16:
            img_f32 = img_gray.astype(np.float32) / 65535.0
        elif np.issubdtype(img_gray.dtype, np.floating):
            max_v = float(np.max(img_gray)) if np.max(img_gray) > 0 else 1.0
            if max_v > 1.0:
                img_f32 = (img_gray / max_v).astype(np.float32)
            else:
                img_f32 = img_gray.astype(np.float32)
        else:
            img_f32 = np.clip(img_gray.astype(np.float32) / 255.0, 0.0, 1.0)

        # Scale determination
        scale = pixel_size_um or meta.get("pixel_size_um", 1.518817)
        try:
            scale = float(scale)
            if scale <= 0 or math.isnan(scale):
                scale = 1.518817
        except Exception:
            scale = 1.518817

        # Overlap resolution / region property extraction
        props = measure.regionprops(raw_label_mask)
        if not props:
            return np.zeros((h, w), dtype=np.int32), []

        all_objects_binary = (raw_label_mask > 0).astype(np.uint8)
        records: List[SpheroidObjectRecord] = []
        clean_mask = np.zeros((h, w), dtype=np.int32)
        clean_label_idx = 1

        for prop in props:
            obj_lbl = prop.label
            obj_bin = (raw_label_mask == obj_lbl)

            # Compute annular background ring & contrast
            mean_int, mean_ring, contrast_ratio = compute_annular_ring_contrast(
                image=img_f32,
                obj_mask=obj_bin,
                all_objects_mask=all_objects_binary,
                dilation_px=self.config.background_ring_dilation_px,
            )

            # Measure full 32-column attributes
            rec = measure_single_object(
                prop=prop,
                image_shape=(h, w),
                pixel_size_um=scale,
                mean_interior_intensity=mean_int,
                mean_ring_intensity=mean_ring,
                contrast_ratio=contrast_ratio,
                image_metadata=meta,
                config=self.config,
            )

            # Check single mask max area gate (25% FOV)
            total_fov_px = float(h * w)
            if (rec.area_px / total_fov_px) > self.config.single_mask_max_area_pct:
                rec.qc_flag = "FAIL"
                reason_area = (
                    f"exceeds_max_area_limit ({(rec.area_px / total_fov_px)*100:.1f}% > "
                    f"{self.config.single_mask_max_area_pct*100:.0f}%)"
                )
                if rec.qc_reasons == "None" or not rec.qc_reasons:
                    rec.qc_reasons = reason_area
                else:
                    rec.qc_reasons = f"{reason_area}; {rec.qc_reasons}"

            # Only PASS and REVIEW objects enter the clean label map
            if rec.qc_flag != "FAIL":
                clean_mask[obj_bin] = clean_label_idx
                rec.label_idx = clean_label_idx
                # Update object ID to match clean label index
                rec.object_id = f"{rec.pair_key}_{rec.timepoint}_obj{clean_label_idx}"
                clean_label_idx += 1

            records.append(rec)

        return clean_mask, records


def filter_and_measure_objects(
    image: np.ndarray,
    raw_label_mask: np.ndarray,
    image_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Union[PipelineConfig, QCConfig]] = None,
) -> Tuple[np.ndarray, List[SpheroidObjectRecord]]:
    """
    Standard interface contract function matching PROJECT.md:
    filter_and_measure_objects(image, raw_label_mask, image_metadata, config)
    -> Tuple[clean_label_mask, List[SpheroidObjectRecord]]
    """
    if config is None:
        qc_cfg = QCConfig()
    elif isinstance(config, PipelineConfig):
        qc_cfg = config.qc
    elif isinstance(config, QCConfig):
        qc_cfg = config
    else:
        qc_cfg = QCConfig()

    qcf = QCFilter(qc_cfg)
    meta = image_metadata or {}
    pixel_size = meta.get("pixel_size_um")
    return qcf.filter_and_measure(
        image=image,
        raw_label_mask=raw_label_mask,
        image_metadata=meta,
        pixel_size_um=pixel_size,
    )


def deduplicate_dataset_objects(
    objects: List[SpheroidObjectRecord],
    max_duplicate_distance_px: float = 35.0,
) -> Tuple[List[SpheroidObjectRecord], Dict[str, Any]]:
    """
    Cross-FOV Deduplication Pass within the same well and timepoint.
    Identifies spheroids imaged across redundant or multi-focal Z-planes:
    For any two objects O_A and O_B in the same (condition, replicate, timepoint)
    with FOV_A != FOV_B and centroid distance <= max_duplicate_distance_px:
      - Keeps the in-focus instance with higher annular contrast ratio (or higher circularity)
      - Marks the other duplicate as qc_flag = 'FAIL', qc_reasons = 'duplicate_focal_plane'
    """
    from collections import defaultdict
    well_groups = defaultdict(list)
    for idx, obj in enumerate(objects):
        if obj.qc_flag in ("PASS", "REVIEW"):
            well_groups[(obj.condition, obj.replicate, obj.timepoint)].append((idx, obj))

    resolved_losers = set()
    duplicate_count = 0

    for (cond, rep, tp), group_items in well_groups.items():
        if len(group_items) < 2:
            continue
        matched_pairs = []
        n = len(group_items)
        for i in range(n):
            idx_a, obj_a = group_items[i]
            for j in range(i + 1, n):
                idx_b, obj_b = group_items[j]
                if obj_a.fov != obj_b.fov:
                    dist = math.hypot(obj_a.centroid_x - obj_b.centroid_x, obj_a.centroid_y - obj_b.centroid_y)
                    if dist <= max_duplicate_distance_px:
                        matched_pairs.append((dist, idx_a, obj_a, idx_b, obj_b))

        # Sort matches by distance ascending
        matched_pairs.sort(key=lambda x: x[0])

        for dist, idx_a, obj_a, idx_b, obj_b in matched_pairs:
            if idx_a in resolved_losers or idx_b in resolved_losers:
                continue

            diff_contrast = obj_a.contrast_ratio - obj_b.contrast_ratio
            if abs(diff_contrast) > 0.02:
                winner_idx, winner_obj, loser_idx, loser_obj = (
                    (idx_a, obj_a, idx_b, obj_b) if diff_contrast > 0 else (idx_b, obj_b, idx_a, obj_a)
                )
            else:
                diff_circ = obj_a.circularity - obj_b.circularity
                winner_idx, winner_obj, loser_idx, loser_obj = (
                    (idx_a, obj_a, idx_b, obj_b) if diff_circ >= 0 else (idx_b, obj_b, idx_a, obj_a)
                )

            resolved_losers.add(loser_idx)
            duplicate_count += 1

            loser_obj.qc_flag = "FAIL"
            reason_str = f"duplicate_focal_plane (matched with {winner_obj.object_id} in fov {winner_obj.fov}, dist={dist:.1f}px)"
            if loser_obj.qc_reasons in ("None", "", None):
                loser_obj.qc_reasons = reason_str
            else:
                loser_obj.qc_reasons = f"{loser_obj.qc_reasons}; {reason_str}"

    diagnostics = {
        "total_objects": len(objects),
        "duplicates_flagged": duplicate_count,
        "remaining_pass_or_review": sum(1 for o in objects if o.qc_flag in ("PASS", "REVIEW")),
    }
    logger.info(f"Cross-FOV deduplication completed: {duplicate_count} duplicate focal-plane spheroids marked FAIL.")
    return objects, diagnostics
