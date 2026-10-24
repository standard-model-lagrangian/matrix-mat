import cv2
import numpy as np
import pytest

from spheroid_pipeline_v2.config import PipelineConfig, SegmentationConfig
from spheroid_pipeline_v2.pair import pair_timepoints
from spheroid_pipeline_v2.qc_filter import filter_and_measure_objects
from spheroid_pipeline_v2.segment import ClassicalMultiScaleSegmenter, SegmentationHierarchy


def _make_disk_image(
    h: int,
    w: int,
    center: tuple[int, int],
    radius_px: float,
    bg_val: float,
    disk_val: float,
    border_val: float = None,
) -> np.ndarray:
    img = np.full((h, w), bg_val, dtype=np.float32)
    cv2.circle(img, center, int(round(radius_px)), disk_val, -1)
    if border_val is not None:
        cv2.circle(img, center, int(round(radius_px)), border_val, 1)
    return img


def test_known_diameter_recovered():
    pixel_size_um = 1.5
    target_d_um = 200.0
    radius_px = (target_d_um / pixel_size_um) / 2.0  # 66.6667 px

    img = _make_disk_image(
        h=512,
        w=512,
        center=(256, 256),
        radius_px=radius_px,
        bg_val=0.65,
        disk_val=0.20,
        border_val=0.15,
    )

    cfg = SegmentationConfig(primary_backend="classical")
    hier = SegmentationHierarchy(config=cfg)
    mask, _ = hier.segment(img, image_id="test_d200", pixel_size_um=pixel_size_um)

    meta = {
        "image_id": "test_d200",
        "timepoint": "t0",
        "condition": "Mat",
        "replicate": "gel1",
        "fov": "0001",
        "pair_key": "gel1_0001",
        "pixel_size_um": pixel_size_um,
        "mask_source": "classical",
        "is_calibrated": True,
        "scale_source": "tiff_tag",
    }
    pipe_cfg = PipelineConfig()
    clean_mask, obj_records = filter_and_measure_objects(img, mask, meta, pipe_cfg)

    pass_objs = [o for o in obj_records if o.qc_flag == "PASS"]
    assert len(pass_objs) == 1, f"Expected 1 PASS object, got {len(pass_objs)}"
    recovered_d = pass_objs[0].equivalent_diameter_um
    # Within +- 3 %
    assert abs(recovered_d - target_d_um) / target_d_um <= 0.03, (
        f"Recovered diameter {recovered_d:.2f} um not within 3% of target {target_d_um} um"
    )


def test_known_fold_change_recovered():
    pixel_size_um = 1.5
    d0_um = 100.0
    d7_um = 150.0
    expected_fc_vol = (d7_um / d0_um) ** 3  # 1.5^3 = 3.375

    r0_px = (d0_um / pixel_size_um) / 2.0
    r7_px = (d7_um / pixel_size_um) / 2.0

    img0 = _make_disk_image(512, 512, (256, 256), r0_px, bg_val=0.65, disk_val=0.20, border_val=0.15)
    img7 = _make_disk_image(512, 512, (256, 256), r7_px, bg_val=0.65, disk_val=0.20, border_val=0.15)

    cfg = SegmentationConfig(primary_backend="classical")
    hier = SegmentationHierarchy(config=cfg)
    mask0, _ = hier.segment(img0, image_id="d0_img", pixel_size_um=pixel_size_um)
    mask7, _ = hier.segment(img7, image_id="d7_img", pixel_size_um=pixel_size_um)

    pipe_cfg = PipelineConfig()
    meta0 = {
        "image_id": "d0_img", "timepoint": "t0", "condition": "Mat",
        "replicate": "gel1", "fov": "0001", "pair_key": "gel1_0001",
        "pixel_size_um": pixel_size_um, "mask_source": "classical",
    }
    meta7 = {
        "image_id": "d7_img", "timepoint": "t7", "condition": "Mat",
        "replicate": "gel1", "fov": "0001", "pair_key": "gel1_0001",
        "pixel_size_um": pixel_size_um, "mask_source": "classical",
    }

    _, objs0 = filter_and_measure_objects(img0, mask0, meta0, pipe_cfg)
    _, objs7 = filter_and_measure_objects(img7, mask7, meta7, pipe_cfg)

    pairs, _, _ = pair_timepoints(objs0, objs7, pipe_cfg.pairing)
    assert len(pairs) == 1, f"Expected 1 paired spheroid, got {len(pairs)}"
    fc = pairs[0].fold_change_volume
    # Within +- 5 %
    assert abs(fc - expected_fc_vol) / expected_fc_vol <= 0.05, (
        f"Recovered FC {fc:.3f} not within 5% of expected {expected_fc_vol:.3f}"
    )


def test_bright_blob_rejected():
    pixel_size_um = 1.5
    # High-intensity bright blob (e.g. bubble or glare)
    img = _make_disk_image(
        h=512,
        w=512,
        center=(256, 256),
        radius_px=60,
        bg_val=0.55,
        disk_val=0.95,
        border_val=0.40,
    )

    cfg = SegmentationConfig(primary_backend="classical")
    hier = SegmentationHierarchy(config=cfg)
    mask, _ = hier.segment(img, image_id="bright_blob", pixel_size_um=pixel_size_um)

    meta = {
        "image_id": "bright_blob", "timepoint": "t0", "condition": "Mat",
        "replicate": "gel1", "fov": "0001", "pair_key": "gel1_0001",
        "pixel_size_um": pixel_size_um, "mask_source": "classical",
    }
    pipe_cfg = PipelineConfig()
    clean_mask, obj_records = filter_and_measure_objects(img, mask, meta, pipe_cfg)
    pass_objs = [o for o in obj_records if o.qc_status == "PASS"]
    assert len(pass_objs) == 0, f"Expected 0 PASS objects for bright blob, got {len(pass_objs)}"
