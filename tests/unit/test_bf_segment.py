"""Regression tests for brightfield segmentation fixes: cache key, fallback tiers, config plumbing."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from matrix_mat.brightfield.config import PreprocessConfig, SegmentationConfig
from matrix_mat.brightfield.decisions import DecisionsLogger
from matrix_mat.brightfield.preprocess import Preprocessor
from matrix_mat.brightfield.segment import (
    CellposeSAMSegmenter,
    ClassicalMultiScaleSegmenter,
    SegmentationHierarchy,
    get_optimal_device,
)


class _Stub:
    def __init__(self, mask_fn, backend):
        self.mask_fn, self.backend, self.calls = mask_fn, backend, 0

    def segment(self, img, pixel_size_um):
        self.calls += 1
        m = self.mask_fn(img)
        return m, {"backend": self.backend, "model_id": self.backend, "status": "success" if m.max() else "empty"}


def _blob_image():
    img = np.full((256, 256), 0.8, dtype=np.float32)
    cv2.circle(img, (128, 128), 40, 0.2, -1)
    return img


def _hier(tmp_path, seg_cfg=None, prep_cfg=None):
    seg_cfg = seg_cfg or SegmentationConfig(primary_backend="classical", cache_dir=str(tmp_path / "cache"))
    return SegmentationHierarchy(
        seg_cfg,
        decisions_logger=DecisionsLogger(tmp_path / "decisions.jsonl"),
        mask_cache=str(tmp_path / "cache"),
        preprocessor=Preprocessor(prep_cfg or PreprocessConfig()),
    )


def test_cache_key_includes_pixel_size_and_preprocess(tmp_path):
    img = _blob_image()
    h1 = _hier(tmp_path)
    _, m1 = h1.segment(img, "img", pixel_size_um=1.5)
    _, m2 = h1.segment(img, "img", pixel_size_um=1.5)
    assert m1["cached"] is False and m2["cached"] is True
    _, m3 = h1.segment(img, "img", pixel_size_um=0.5)
    assert m3["cached"] is False, "different pixel size must not hit the cache"
    h2 = _hier(tmp_path, prep_cfg=PreprocessConfig(percentile_low=5.0))
    _, m4 = h2.segment(img, "img", pixel_size_um=1.5)
    assert m4["cached"] is False, "different preprocessing must not hit the cache"
    # Output-only switches do not invalidate the cache
    h3 = _hier(tmp_path, prep_cfg=PreprocessConfig(save_corrected_images=False))
    _, m5 = h3.segment(img, "img", pixel_size_um=1.5)
    assert m5["cached"] is True


def test_per_image_last_resort_is_plausibility_checked_and_not_cached(tmp_path):
    cfg = SegmentationConfig(primary_backend="cpdino", fallback_policy="per_image", fallback_backend="cpsam",
                             last_resort_backend="classical", cache_dir=str(tmp_path / "cache"))
    h = _hier(tmp_path, seg_cfg=cfg)
    empty = _Stub(lambda im: np.zeros(im.shape[:2], np.int32), "cpdino")
    also_empty = _Stub(lambda im: np.zeros(im.shape[:2], np.int32), "cpsam")
    giant = _Stub(lambda im: np.ones(im.shape[:2], np.int32), "classical")
    h.cyto3_segmenter, h.cpsam_segmenter, h.classical_segmenter = empty, also_empty, giant
    img = np.zeros((128, 128), np.uint8)
    _, meta = h.segment(img, "bad", pixel_size_um=1.5)
    assert (empty.calls, also_empty.calls, giant.calls) == (1, 1, 1)
    assert meta["qc_flag"] == "SEG_FAIL"
    assert meta["backend"] == "classical"
    _, meta2 = h.segment(img, "bad", pixel_size_um=1.5)
    assert meta2["cached"] is False, "fallback result must not be cached under the primary key"
    lines = [json.loads(x) for x in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    finals = [x for x in lines if x["step"] == "Segmentation Hierarchy Result"]
    assert finals and all(x["outcome"].startswith("SEG_FAIL") for x in finals)


def test_fallback_tiers_follow_config(tmp_path):
    cfg = SegmentationConfig(primary_backend="cpsam", fallback_policy="per_image", fallback_backend="classical",
                             last_resort_backend="cpdino", cache_dir=str(tmp_path / "cache"))
    h = _hier(tmp_path, seg_cfg=cfg)

    def _one(im):
        m = np.zeros(im.shape[:2], np.int32)
        m[40:60, 40:60] = 1
        return m

    h.cpsam_segmenter = _Stub(lambda im: np.zeros(im.shape[:2], np.int32), "cpsam")
    h.classical_segmenter = _Stub(_one, "classical")
    h.cyto3_segmenter = _Stub(_one, "cpdino")
    _, meta = h.segment(np.zeros((128, 128), np.uint8), "x", pixel_size_um=1.5)
    assert meta["backend"] == "classical"
    assert h.cyto3_segmenter.calls == 0
    assert meta.get("qc_flag") != "SEG_FAIL"


def test_strict_seg_fail_logged_as_failure(tmp_path):
    cfg = SegmentationConfig(primary_backend="cpsam", cache_dir=str(tmp_path / "cache"))
    h = _hier(tmp_path, seg_cfg=cfg)
    h.cpsam_segmenter = _Stub(lambda im: np.zeros(im.shape[:2], np.int32), "cpsam")
    h.segment(np.zeros((64, 64), np.uint8), "empty", pixel_size_um=1.5)
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[-1])
    assert rec["outcome"].startswith("SEG_FAIL")


def test_classical_gates_read_from_config():
    img = _blob_image()
    m_default, _ = ClassicalMultiScaleSegmenter(SegmentationConfig()).segment(img, 1.5)
    assert m_default.max() == 1
    m_strict, _ = ClassicalMultiScaleSegmenter(SegmentationConfig(classical_min_contrast=0.95)).segment(img, 1.5)
    assert m_strict.max() == 0
    m_big, _ = ClassicalMultiScaleSegmenter(SegmentationConfig(classical_min_d_um=500.0)).segment(img, 1.5)
    assert m_big.max() == 0


def test_device_config_honoured():
    assert get_optimal_device("cpu") == (False, "cpu")
    seg = CellposeSAMSegmenter(SegmentationConfig(device="cpu"))
    seg._resolve_device()
    assert (seg.use_gpu, seg.device_str) == (False, "cpu")
    assert SegmentationConfig().device == "auto"
    with pytest.raises(RuntimeError, match="device"):
        get_optimal_device("tpu")
