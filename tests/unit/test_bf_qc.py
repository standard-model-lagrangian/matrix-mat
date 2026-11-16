"""Regression tests for brightfield QC fixes: object_id uniqueness, max-objects gate, cross-FOV dedup."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from matrix_mat.brightfield.config import PipelineConfig, QCConfig
from matrix_mat.brightfield.qc_filter import deduplicate_dataset_objects, filter_and_measure_objects

META = {"image_id": "img", "pair_key": "Mat_gel1_0001", "timepoint": "t0", "condition": "Mat",
        "replicate": "gel1", "fov": "0001", "pixel_size_um": 1.5, "is_calibrated": True}


def _scene():
    """Raw labels 1 and 2 = clean dark spheroids, raw label 3 = FAIL (touches border, processed last)."""
    img = np.full((400, 400), 0.8, dtype=np.float32)
    lab = np.zeros((400, 400), dtype=np.int32)
    cv2.circle(lab, (120, 120), 40, 1, -1)
    cv2.circle(lab, (280, 280), 40, 2, -1)
    cv2.circle(lab, (5, 200), 30, 3, -1)  # border-touching -> FAIL
    img[lab > 0] = 0.2
    return img, lab


def test_object_ids_unique_with_fail_objects():
    img, lab = _scene()
    _, recs = filter_and_measure_objects(img, lab, dict(META), QCConfig())
    ids = [r.object_id for r in recs]
    assert len(ids) == len(set(ids)), ids
    fails = [r for r in recs if r.qc_flag == "FAIL"]
    assert len(fails) == 1
    assert fails[0].object_id == "Mat_gel1_0001_t0_fail3"
    clean_ids = sorted(r.object_id for r in recs if r.qc_flag != "FAIL")
    assert clean_ids == ["Mat_gel1_0001_t0_obj1", "Mat_gel1_0001_t0_obj2"]


def test_max_objects_per_fov_never_promotes_fail():
    img, lab = _scene()
    _, recs = filter_and_measure_objects(img, lab, dict(META), QCConfig(max_objects_per_fov=1))
    by_raw = {r.object_id: r for r in recs}
    assert by_raw["Mat_gel1_0001_t0_fail3"].qc_flag == "FAIL"
    flags = sorted(r.qc_flag for r in recs if r.qc_flag != "FAIL")
    assert flags == ["PASS", "REVIEW"]
    demoted = [r for r in recs if r.qc_flag == "REVIEW"][0]
    assert "exceeds_max_objects_per_fov" in demoted.qc_reasons


def _two_fov_objects(dx_px, ps=1.518817):
    img, lab = _scene()
    _, recs = filter_and_measure_objects(img, lab, {**META, "pixel_size_um": ps}, QCConfig())
    a = [r for r in recs if r.qc_flag == "PASS"][0]
    b = replace(a, fov="0002", object_id=a.object_id.replace("0001", "0002"), centroid_x=a.centroid_x + dx_px,
                contrast_ratio=a.contrast_ratio - 0.1)
    return [a, b]


def test_dedup_can_be_disabled():
    objs = _two_fov_objects(5.0)
    _, diag = deduplicate_dataset_objects(objs, enabled=False)
    assert diag["duplicates_flagged"] == 0
    assert all(o.qc_flag == "PASS" for o in objs)


def test_dedup_threshold_in_micrometres_uses_pixel_size():
    # 30 px apart: at 1.5 um/px = 45 um (< 53.16 um default) -> duplicate
    objs = _two_fov_objects(30.0, ps=1.5)
    _, diag = deduplicate_dataset_objects(objs, max_duplicate_distance_um=QCConfig().dedup_max_distance_um)
    assert diag["duplicates_flagged"] == 1
    # 30 px apart at 3 um/px = 90 um -> not a duplicate, although 30 px < 35 px
    objs = _two_fov_objects(30.0, ps=3.0)
    _, diag = deduplicate_dataset_objects(objs, max_duplicate_distance_um=QCConfig().dedup_max_distance_um)
    assert diag["duplicates_flagged"] == 0


def test_dedup_default_equivalent_to_35px_at_default_scale():
    assert QCConfig().dedup_max_distance_um == pytest.approx(35.0 * 1.518817)
    cfg = PipelineConfig.from_yaml("configs/spheroid_brightfield.yaml")
    assert cfg.qc.dedup_cross_fov is True
    assert cfg.qc.dedup_max_distance_um == pytest.approx(35.0 * 1.518817)
