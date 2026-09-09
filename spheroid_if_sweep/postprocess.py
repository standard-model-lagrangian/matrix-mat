"""
Post-processing module for spheroid IF imaging.
Applied to cpsam label masks before QC and feature extraction, in fixed order:
1. Deduplication (IoU threshold)
2. Size gating ([min_vol_um3, max_vol_um3])
3. Clump splitting (seeded watershed on candidate masks > split_factor * image_median_vol)
4. Border exclusion (clearance_um margin exclusion from density statistics)
All parameters operate in physical units (um, um3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max

from spheroid_if_sweep.qc import deduplicate_masks

logger = logging.getLogger("spheroid_if_sweep.postprocess")

# Data-driven reference nuclear volume derived from 001_A_baseline cohort median of material medians
REFERENCE_NUCLEAR_VOLUME_UM3 = 841.04
DEFAULT_MIN_VOL_UM3 = 210.0   # ~0.25x reference volume
DEFAULT_MAX_VOL_UM3 = 3364.0  # ~4.0x reference volume


@dataclass
class SplitEvent:
    parent_label: int
    parent_vol_um3: float
    n_children: int
    child_vols_um3: List[float]
    accepted: bool
    reversion_reason: str = ""


@dataclass
class PostprocessStats:
    image_id: str
    n_raw: int
    n_dedup_removed: int
    n_size_gate_removed_small: int
    n_size_gate_removed_large: int
    n_split_candidates: int
    n_split_accepted: int
    n_split_reverted: int
    n_children_created: int
    n_border_excluded: int
    n_final_total: int
    n_final_included: int
    removed_sizes_um3: List[float] = field(default_factory=list)
    split_events: List[Dict[str, Any]] = field(default_factory=list)


def compute_object_volume(area_px: int | float, pixel_size_um: float) -> Tuple[float, float]:
    """
    Compute equivalent 2D area (um2) and spherical equivalent volume (um3).
    Returns (area_um2, volume_um3).
    """
    area_um2 = float(area_px) * (pixel_size_um ** 2)
    eq_diam_um = 2.0 * np.sqrt(max(0.0, area_um2) / np.pi)
    vol_um3 = (np.pi / 6.0) * (eq_diam_um ** 3)
    return area_um2, vol_um3


def apply_size_gate(
    masks: np.ndarray,
    pixel_size_um: float,
    min_vol_um3: float,
    max_vol_um3: float,
) -> Tuple[np.ndarray, int, int, List[float]]:
    """
    Remove objects whose volume falls outside [min_vol_um3, max_vol_um3].
    Returns (clean_masks, n_removed_small, n_removed_large, removed_sizes_um3).
    """
    if masks.max() == 0:
        return masks.copy(), 0, 0, []

    props = measure.regionprops(masks)
    n_small = 0
    n_large = 0
    removed_sizes: List[float] = []
    labels_to_remove: Set[int] = set()

    for p in props:
        _, vol = compute_object_volume(p.area, pixel_size_um)
        if vol < min_vol_um3:
            labels_to_remove.add(p.label)
            n_small += 1
            removed_sizes.append(round(vol, 2))
        elif vol > max_vol_um3:
            labels_to_remove.add(p.label)
            n_large += 1
            removed_sizes.append(round(vol, 2))

    if not labels_to_remove:
        return masks.copy(), 0, 0, []

    clean = masks.copy()
    for lbl in labels_to_remove:
        clean[clean == lbl] = 0

    # Relabel compactly
    relabeled = np.zeros_like(clean)
    new_lbl = 1
    for p in measure.regionprops(clean):
        relabeled[clean == p.label] = new_lbl
        new_lbl += 1

    return relabeled, n_small, n_large, removed_sizes


def split_merged_nuclei(
    masks: np.ndarray,
    nuclear_img: np.ndarray,
    pixel_size_um: float,
    split_factor: float = 2.0,
    h_maxima: float = 10.0,
    min_distance_um: float = 4.0,
    min_vol_um3: float = DEFAULT_MIN_VOL_UM3,
    max_vol_um3: float = DEFAULT_MAX_VOL_UM3,
) -> Tuple[np.ndarray, int, int, int, int, List[SplitEvent]]:
    """
    Split clumped/merged nuclei candidates (volume > split_factor * image_median_vol).
    Uses seeded watershed initialized by intensity h-maxima (fallback: distance transform peaks).
    Accepts split only if ALL children pass the size gate; otherwise reverts to parent.
    Returns (result_masks, n_candidates, n_accepted, n_reverted, n_children_created, split_events).
    """
    if masks.max() == 0:
        return masks.copy(), 0, 0, 0, 0, []

    props = measure.regionprops(masks)
    if not props:
        return masks.copy(), 0, 0, 0, 0, []

    vols = {p.label: compute_object_volume(p.area, pixel_size_um)[1] for p in props}
    image_median_vol = float(np.median(list(vols.values()))) if vols else REFERENCE_NUCLEAR_VOLUME_UM3
    candidate_thresh = split_factor * image_median_vol

    candidates = [p for p in props if vols[p.label] > candidate_thresh]
    if not candidates:
        return masks.copy(), 0, 0, 0, 0, []

    result = masks.copy()
    current_max_label = int(result.max())
    n_candidates = len(candidates)
    n_accepted = 0
    n_reverted = 0
    n_children_created = 0
    split_events: List[SplitEvent] = []

    for p in candidates:
        minr, minc, maxr, maxc = p.bbox
        sub_mask = (result[minr:maxr, minc:maxc] == p.label)
        sub_img = nuclear_img[minr:maxr, minc:maxc]

        # 1. Seeds: intensity h-maxima
        smooth = ndimage.gaussian_filter(sub_img.astype(float), sigma=1.0)
        h_domes = morphology.h_maxima(smooth, h=h_maxima)
        seed_mask = (h_domes > 0) & sub_mask
        seed_labels, n_seeds = measure.label(seed_mask, return_num=True)

        # 2. Fallback: distance transform peaks
        if n_seeds <= 1:
            dist = ndimage.distance_transform_edt(sub_mask)
            min_dist_px = max(2, int(round(min_distance_um / max(1e-6, pixel_size_um))))
            coords = peak_local_max(dist, min_distance=min_dist_px, labels=sub_mask)
            if len(coords) > 1:
                seed_labels = np.zeros_like(sub_mask, dtype=np.int32)
                for idx, (r, c) in enumerate(coords, 1):
                    seed_labels[r, c] = idx
                n_seeds = len(coords)

        # 3. Watershed split if multiple seeds identified
        if n_seeds > 1:
            dist = ndimage.distance_transform_edt(sub_mask)
            sub_split = segmentation.watershed(-dist, seed_labels, mask=sub_mask)
            child_labels = [c for c in np.unique(sub_split) if c > 0]

            # Check child size criteria
            child_vols = [
                compute_object_volume(np.sum(sub_split == c), pixel_size_um)[1]
                for c in child_labels
            ]

            all_pass = (len(child_labels) > 1)
            reversion_reason = ""
            for c_vol in child_vols:
                if c_vol < min_vol_um3:
                    all_pass = False
                    reversion_reason = f"Child volume {c_vol:.1f} < min {min_vol_um3:.1f}"
                    break
                elif c_vol > max_vol_um3:
                    all_pass = False
                    reversion_reason = f"Child volume {c_vol:.1f} > max {max_vol_um3:.1f}"
                    break

            if all_pass:
                # Accept split
                n_accepted += 1
                n_children_created += len(child_labels)
                # Clear parent
                sub_area = result[minr:maxr, minc:maxc]
                sub_area[sub_mask] = 0

                # Assign first child the parent's label, rest new labels
                sub_area[sub_split == child_labels[0]] = p.label
                for extra_c in child_labels[1:]:
                    current_max_label += 1
                    sub_area[sub_split == extra_c] = current_max_label

                split_events.append(SplitEvent(
                    parent_label=p.label,
                    parent_vol_um3=round(vols[p.label], 2),
                    n_children=len(child_labels),
                    child_vols_um3=[round(v, 2) for v in child_vols],
                    accepted=True,
                ))
            else:
                n_reverted += 1
                split_events.append(SplitEvent(
                    parent_label=p.label,
                    parent_vol_um3=round(vols[p.label], 2),
                    n_children=len(child_labels),
                    child_vols_um3=[round(v, 2) for v in child_vols],
                    accepted=False,
                    reversion_reason=reversion_reason,
                ))
        else:
            n_reverted += 1
            split_events.append(SplitEvent(
                parent_label=p.label,
                parent_vol_um3=round(vols[p.label], 2),
                n_children=1,
                child_vols_um3=[round(vols[p.label], 2)],
                accepted=False,
                reversion_reason="Single seed found; no split boundary",
            ))

    # Relabel compactly
    relabeled = np.zeros_like(result)
    new_lbl = 1
    for p in measure.regionprops(result):
        relabeled[result == p.label] = new_lbl
        new_lbl += 1

    return relabeled, n_candidates, n_accepted, n_reverted, n_children_created, split_events


def identify_border_objects(
    masks: np.ndarray,
    pixel_size_um: float,
    clearance_um: float,
) -> Tuple[Set[int], int]:
    """
    Identify objects within clearance_um of the field border.
    Returns (border_labels_set, count_border_objects).
    """
    if masks.max() == 0 or clearance_um <= 0:
        return set(), 0

    h, w = masks.shape[:2]
    clearance_px = max(1, int(round(clearance_um / max(1e-6, pixel_size_um))))

    props = measure.regionprops(masks)
    border_labels: Set[int] = set()

    for p in props:
        minr, minc, maxr, maxc = p.bbox
        if minr <= clearance_px or minc <= clearance_px or maxr >= (h - clearance_px) or maxc >= (w - clearance_px):
            border_labels.add(p.label)

    return border_labels, len(border_labels)


def postprocess_field_masks(
    image_id: str,
    raw_masks: np.ndarray,
    nuclear_img: np.ndarray,
    pixel_size_um: float,
    postprocess_config: Dict[str, Any],
) -> Tuple[np.ndarray, Set[int], PostprocessStats]:
    """
    Full post-processing pipeline for a single image:
    1. Deduplicate
    2. Size gate
    3. Split merged nuclei
    4. Border exclusion
    Returns (final_masks, border_labels, stats).
    """
    enabled = bool(postprocess_config.get("enabled", True))
    n_raw = int(raw_masks.max())

    if not enabled:
        # Passthrough
        stats = PostprocessStats(
            image_id=image_id,
            n_raw=n_raw,
            n_dedup_removed=0,
            n_size_gate_removed_small=0,
            n_size_gate_removed_large=0,
            n_split_candidates=0,
            n_split_accepted=0,
            n_split_reverted=0,
            n_children_created=0,
            n_border_excluded=0,
            n_final_total=n_raw,
            n_final_included=n_raw,
        )
        return raw_masks.copy(), set(), stats

    # 1. Deduplicate
    dedup_cfg = postprocess_config.get("deduplicate", {})
    iou_thresh = float(dedup_cfg.get("iou_threshold", 0.50))
    masks_dedup, n_dups = deduplicate_masks(raw_masks, iou_threshold=iou_thresh)

    # 2. Size gate
    size_cfg = postprocess_config.get("size_gate", {})
    size_enabled = bool(size_cfg.get("enabled", True))
    min_vol = float(size_cfg.get("min_vol_um3", DEFAULT_MIN_VOL_UM3))
    max_vol = float(size_cfg.get("max_vol_um3", DEFAULT_MAX_VOL_UM3))

    if size_enabled:
        masks_gated, n_small, n_large, removed_sizes = apply_size_gate(
            masks_dedup, pixel_size_um, min_vol, max_vol
        )
    else:
        masks_gated = masks_dedup
        n_small, n_large, removed_sizes = 0, 0, []

    # 3. Split merged nuclei
    split_cfg = postprocess_config.get("splitting", {})
    split_enabled = bool(split_cfg.get("enabled", True))
    if split_enabled:
        split_factor = float(split_cfg.get("split_factor", 2.0))
        h_max = float(split_cfg.get("h_maxima", 10.0))
        min_dist_um = float(split_cfg.get("min_distance_um", 4.0))
        masks_split, n_cand, n_acc, n_rev, n_kids, split_evs = split_merged_nuclei(
            masks=masks_gated,
            nuclear_img=nuclear_img,
            pixel_size_um=pixel_size_um,
            split_factor=split_factor,
            h_maxima=h_max,
            min_distance_um=min_dist_um,
            min_vol_um3=min_vol,
            max_vol_um3=max_vol,
        )
    else:
        masks_split = masks_gated
        n_cand, n_acc, n_rev, n_kids = 0, 0, 0, 0
        split_evs = []

    # 4. Border exclusion
    border_cfg = postprocess_config.get("border_exclusion", {})
    border_enabled = bool(border_cfg.get("enabled", True))
    clearance_um = float(border_cfg.get("clearance_um", 5.85)) if border_enabled else 0.0

    border_labels, n_border = identify_border_objects(masks_split, pixel_size_um, clearance_um)

    n_final = int(masks_split.max())
    n_included = n_final - n_border

    stats = PostprocessStats(
        image_id=image_id,
        n_raw=n_raw,
        n_dedup_removed=n_dups,
        n_size_gate_removed_small=n_small,
        n_size_gate_removed_large=n_large,
        n_split_candidates=n_cand,
        n_split_accepted=n_acc,
        n_split_reverted=n_rev,
        n_children_created=n_kids,
        n_border_excluded=n_border,
        n_final_total=n_final,
        n_final_included=n_included,
        removed_sizes_um3=removed_sizes,
        split_events=[e.__dict__ for e in split_evs],
    )

    return masks_split, border_labels, stats
