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


def _disk_masks(radii, pixel_size_um=1.0, shape=(200, 200)):
    masks = np.zeros(shape, dtype=np.int32)
    img = np.zeros(shape, dtype=np.uint8)
    for i, r in enumerate(radii, start=1):
        cy = 20 + ((i - 1) // 5) * 40
        cx = 20 + ((i - 1) % 5) * 40
        rr, cc = draw.disk((cy, cx), radius=r, shape=shape)
        masks[rr, cc] = i
        img[rr, cc] = 150
    return masks, img


def test_size_gate_requires_configured_reference_volume():
    """With size gating or splitting enabled, a missing reference volume is a configuration error."""
    masks, img = _disk_masks([6, 6, 6])
    for section in ("size_gate", "splitting"):
        cfg = {
            "enabled": True,
            "size_gate": {"enabled": section == "size_gate", "min_vol_um3": 50.0, "max_vol_um3": 5000.0},
            "splitting": {"enabled": section == "splitting"},
            "border_exclusion": {"enabled": False},
        }
        with pytest.raises(ValueError, match="reference_vol_um3"):
            postprocess_field_masks("no_ref", masks, img, 1.0, cfg)


def test_size_gate_reference_from_config_not_cohort():
    """
    Two cohorts with very different median nuclear sizes are gated and split against the same
    configured reference, so the decision for an identical object does not depend on its neighbours.
    """
    cfg = {
        "enabled": True,
        "size_gate": {
            "enabled": True,
            "reference_vol_um3": 1200.0,
            "reference_citation": "test reference",
        },
        "splitting": {"enabled": True, "split_factor": 2.0},
        "border_exclusion": {"enabled": False},
    }
    probe_radius = 8  # ~2145 um3 at 1 um/px: below 2 x 1200, so never a split candidate
    small_cohort, img_s = _disk_masks([5, 5, 5, 5, probe_radius])
    large_cohort, img_l = _disk_masks([12, 12, 12, 12, probe_radius])

    _, _, stats_s = postprocess_field_masks("small", small_cohort, img_s, 1.0, cfg)
    _, _, stats_l = postprocess_field_masks("large", large_cohort, img_l, 1.0, cfg)

    # Same configured bounds [0.25, 4] x 1200 um3 in both fields: the r=5 disks (~524 um3) pass,
    # the r=12 disks (~7238 um3) exceed 2 x 1200 and become split candidates.
    assert stats_s.n_split_candidates == 0
    assert stats_l.n_split_candidates == 4
    assert stats_s.n_size_gate_removed_small == 0


def test_shipped_if_configs_cite_reference_volume():
    import yaml
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for name in ("sweep.yaml", "prepost_ablation.yaml", "robustness_check.yaml"):
        data = yaml.safe_load((repo / "configs" / name).read_text())
        sections = [data.get("postprocessing") or {}] + [c.get("postprocessing") or {} for c in data["combos"]]
        for sec in sections:
            if not sec.get("enabled"):
                continue
            gate = sec.get("size_gate") or {}
            assert gate.get("reference_vol_um3"), f"{name}: enabled post-processing without reference_vol_um3"
            assert gate.get("reference_citation"), f"{name}: reference_vol_um3 without a citation"


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
