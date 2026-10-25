import numpy as np
import pytest
from spheroid_pipeline_v2.pair import SpheroidPairer
from spheroid_pipeline_v2.measure import SpheroidObjectRecord
from spheroid_pipeline_v2.config import PairingConfig, SegmentationConfig
from spheroid_pipeline_v2.segment import ClassicalMultiScaleSegmenter, SegmentationHierarchy


def test_vanished_spheroid_log2fc_is_nan_and_flagged():
    """
    A vanished spheroid (v7 = 0 or volume collapse) must produce log2_fc = NaN
    and a reason in qc_reasons, NOT log2_fc = 0.0 (which falsely indicates no change).
    """
    pairer = SpheroidPairer(PairingConfig())

    # Day 0 object with valid volume
    obj0 = SpheroidObjectRecord(
        object_id="obj_d0_1",
        image_id="img_d0",
        timepoint="t0",
        condition="Mat",
        replicate="gel1",
        fov="0001",
        pair_key="Mat_gel1_0001",
        label_idx=1,
        pixel_size_um=1.0,
        area_px=5000.0,
        perimeter_px=250.0,
        equivalent_diameter_px=50.0,
        centroid_y=100.0,
        centroid_x=100.0,
        area_um2=10000.0,
        equivalent_diameter_um=100.0,
        major_axis_um=110.0,
        minor_axis_um=90.0,
        aspect_ratio=1.2,
        volume_sphere_um3=50000.0,
        volume_ellipsoid_um3=52000.0,
        volume_discrepancy_ratio=1.04,
        circularity=0.9,
        solidity=0.95,
        eccentricity=0.2,
        mean_interior_intensity=0.2,
        mean_background_ring_intensity=0.8,
        contrast_ratio=0.25,
        qc_flag="PASS",
        qc_reasons="",
    )

    # Day 7 object with zero/vanished volume
    obj7 = SpheroidObjectRecord(
        object_id="obj_d7_2",
        image_id="img_d7",
        timepoint="t7",
        condition="Mat",
        replicate="gel1",
        fov="0001",
        pair_key="Mat_gel1_0001",
        label_idx=2,
        pixel_size_um=1.0,
        area_px=0.0,
        perimeter_px=0.0,
        equivalent_diameter_px=0.0,
        centroid_y=100.0,
        centroid_x=100.0,
        area_um2=0.0,
        equivalent_diameter_um=0.0,
        major_axis_um=0.0,
        minor_axis_um=0.0,
        aspect_ratio=1.0,
        volume_sphere_um3=0.0,
        volume_ellipsoid_um3=0.0,
        volume_discrepancy_ratio=1.0,
        circularity=0.0,
        solidity=0.0,
        eccentricity=0.0,
        mean_interior_intensity=0.0,
        mean_background_ring_intensity=0.0,
        contrast_ratio=0.0,
        qc_flag="FAIL",
        qc_reasons="t7_vanished",
    )

    pairs, un0, un7 = pairer.pair_fov_objects([obj0], [obj7], pair_key="Mat_gel1_0001")
    assert len(pairs) == 1
    p = pairs[0]
    assert np.isnan(p.log2_fold_change), f"Expected log2_fc to be NaN for vanished spheroid, but got {p.log2_fold_change}"
    assert "VANISHED" in p.qc_reasons or "FAIL" in p.qc_reasons or "t7_FAIL" in p.qc_reasons


def test_classical_tier_uses_effective_pixel_size():
    """
    When an image is downsampled by scale_factor=0.5, 1 downsampled pixel equals
    pixel_size_um / 0.5. The classical segmenter min-area filter in pixels must be
    computed using the effective downsampled pixel size so equivalent physical diameter
    (e.g. 40 um) matches correctly.
    """
    cfg = SegmentationConfig(primary_backend="classical")
    hierarchy = SegmentationHierarchy(cfg)

    # Synthetic image 400x400 with a dark blob of diameter 50 um
    # At pixel_size = 1.0 um/px, 50 um = 50 px.
    # At scale_factor = 0.5, downsampled image is 200x200, blob diameter is 25 px.
    # Physical min_d_um is 40 um.
    # With effective pixel size = 1.0 / 0.5 = 2.0 um/px, min_d_um=40 um -> min_d_px = 20 px -> min_area_px ≈ 314 px.
    # Since blob has diameter 25 px (area ≈ 490 px), it passes!
    # If unscaled pixel size (1.0) were used, min_d_px would be 40 px (area ≈ 1256 px), and the 490 px blob would be rejected!
    img = np.ones((400, 400), dtype=np.float32) * 0.8
    y, x = np.ogrid[:400, :400]
    dist_from_center = np.sqrt((y - 200)**2 + (x - 200)**2)
    img[dist_from_center <= 25] = 0.2  # diameter 50 px

    # Force scale_factor = 0.5 in preprocessor
    hierarchy.preprocessor.target_width = 200

    mask, meta = hierarchy.segment(img, image_id="test_effective_scale", pixel_size_um=1.0, force_recompute=True)
    assert np.max(mask) > 0, "Blob with physical diameter 50 um was falsely rejected because unscaled pixel size was used!"
