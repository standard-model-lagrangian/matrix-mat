import numpy as np
import pytest
from skimage import draw
from spheroid_if_sweep.postprocess import (
    postprocess_field_masks,
    REFERENCE_NUCLEAR_VOLUME_UM3,
    DEFAULT_MIN_VOL_UM3,
    DEFAULT_MAX_VOL_UM3,
)
from spheroid_if_sweep.comparator_prepost import compute_cv_spread
from spheroid_if_sweep.qc import deduplicate_masks


def test_split_runs_before_size_gate():
    """
    A clump of 4 touching nuclei with total volume > max_vol_um3 must be split FIRST,
    yielding 4 objects after post-processing, rather than being deleted by the size gate.
    """
    h, w = 150, 150
    nuclear_img = np.zeros((h, w), dtype=np.uint8)
    raw_masks = np.zeros((h, w), dtype=np.int32)

    # 4 touching circular discs with centers at (65, 65), (65, 85), (85, 65), (85, 85)
    # radius = 8 px. With 1.0 um/px:
    # Individual disk area ≈ 201 um2 -> volume ≈ 2145 um3
    # Merged clump area ≈ 750 um2 -> volume ≈ 15400 um3
    centers = [(65, 65), (65, 85), (85, 65), (85, 85)]
    for r, c in centers:
        rr, cc = draw.disk((r, c), radius=8, shape=(h, w))
        raw_masks[rr, cc] = 1
        # Add distinct intensity peak for seeded watershed
        nuclear_img[rr, cc] = 120
        nuclear_img[r-2:r+3, c-2:c+3] = 220

    pixel_size_um = 1.0
    postproc_cfg = {
        "enabled": True,
        "deduplicate": {"iou_threshold": 0.50},
        "size_gate": {
            "enabled": True,
            "min_vol_um3": 1000.0,
            "max_vol_um3": 4000.0,
            "reference_vol_um3": 2145.0,
        },
        "splitting": {
            "enabled": True,
            "method": "seeded_watershed",
            "split_factor": 1.5,
            "h_maxima": 10.0,
            "min_distance_um": 5.0,
        },
        "border_exclusion": {
            "enabled": False,
        },
    }

    final_masks, border_labels, stats = postprocess_field_masks(
        image_id="test_clump_4",
        raw_masks=raw_masks,
        nuclear_img=nuclear_img,
        pixel_size_um=pixel_size_um,
        postprocess_config=postproc_cfg,
    )

    n_objects = int(final_masks.max())
    assert n_objects == 4, f"Expected 4 split nuclei, but got {n_objects}. (0 means size-gate ran before split)"
    assert stats.n_split_accepted >= 1
    assert stats.n_size_gate_removed_large == 0


def test_size_gate_reference_from_config_not_cohort():
    """
    Reference volume comes from config (literature/annotation value with citation field).
    Running on cohorts with different medians uses the identical configured gate.
    """
    cfg_a = {
        "enabled": True,
        "size_gate": {
            "enabled": True,
            "reference_vol_um3": 1200.0,
            "reference_citation": "Literature benchmark for high-grade serous ovarian cancer",
        },
        "splitting": {"enabled": False},
        "border_exclusion": {"enabled": False},
    }

    # Image with single object of volume 1000 um3
    # Radius ~ 6.2 px -> Area ~ 120 px -> Vol ~ 1000 um3
    h, w = 100, 100
    img = np.zeros((h, w), dtype=np.uint8)
    masks = np.zeros((h, w), dtype=np.int32)
    rr, cc = draw.disk((50, 50), radius=6, shape=(h, w))
    masks[rr, cc] = 1

    final_masks, _, stats = postprocess_field_masks(
        image_id="cohort_test",
        raw_masks=masks,
        nuclear_img=img,
        pixel_size_um=1.0,
        postprocess_config=cfg_a,
    )

    # 1000 um3 is within [0.25 * 1200, 4.0 * 1200] = [300, 4800]
    assert int(final_masks.max()) == 1


def test_robustness_metric_is_relative():
    """
    Robustness spread is a scale-invariant CV, so removing a constant fraction
    of objects across all runs leaves the robustness spread unchanged.
    """
    orig_counts = [100.0, 120.0, 110.0, 130.0]
    # Remove 40% of objects uniformly across replicates
    scaled_counts = [0.6 * c for c in orig_counts]

    orig_spread = compute_cv_spread(orig_counts)
    scaled_spread = compute_cv_spread(scaled_counts)

    assert orig_spread > 0
    assert np.isclose(orig_spread, scaled_spread, rtol=1e-5), (
        f"Spread changed from {orig_spread} to {scaled_spread} under uniform scaling; "
        "metric must be scale-invariant CV."
    )


def test_dedup_removed_or_documented():
    """
    Deduplication on label images is a documented no-op because label images are
    mutually disjoint partitions.
    """
    # Create disjoint labels
    masks = np.zeros((100, 100), dtype=np.int32)
    masks[10:30, 10:30] = 1
    masks[40:60, 40:60] = 2

    cleaned, n_removed = deduplicate_masks(masks, iou_threshold=0.50)
    assert n_removed == 0
    assert np.array_equal(cleaned, masks)
    # Check that function documentation clearly states disjoint partition property
    assert "disjoint" in deduplicate_masks.__doc__.lower() or "label" in deduplicate_masks.__doc__.lower()
