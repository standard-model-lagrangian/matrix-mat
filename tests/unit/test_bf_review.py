"""Regression tests for manual-mask loading (binary vs label images, JPEG, shape checks)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from matrix_mat.brightfield.review import ReviewManager


def _antialiased_disc(shape=(200, 200)):
    big = np.zeros((shape[0] * 4, shape[1] * 4), np.uint8)
    cv2.circle(big, (shape[1] * 2, shape[0] * 2), 200, 255, -1)
    return cv2.resize(big, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)  # many grey levels at the edge


def test_antialiased_binary_png_is_one_object(tmp_path):
    arr = _antialiased_disc()
    assert len(np.unique(arr)) > 10
    p = tmp_path / "img.png"
    Image.fromarray(arr).save(p)
    m = ReviewManager(tmp_path).load_manual_mask(p)
    assert int(m.max()) == 1


def test_integer_label_image_kept(tmp_path):
    lab = np.zeros((100, 100), np.uint16)
    lab[10:30, 10:30] = 1
    lab[50:80, 50:80] = 2
    lab[10:20, 60:90] = 3
    p = tmp_path / "img.png"
    Image.fromarray(lab).save(p)
    m = ReviewManager(tmp_path).load_manual_mask(p)
    assert sorted(np.unique(m).tolist()) == [0, 1, 2, 3]


def test_binary_255_mask_components(tmp_path):
    arr = np.zeros((100, 100), np.uint8)
    arr[10:30, 10:30] = 255
    arr[60:90, 60:90] = 255
    p = tmp_path / "img.png"
    Image.fromarray(arr).save(p)
    assert int(ReviewManager(tmp_path).load_manual_mask(p).max()) == 2


def test_jpeg_masks_ignored(tmp_path):
    Image.fromarray(_antialiased_disc()).save(tmp_path / "img.jpg")
    rm = ReviewManager(tmp_path)
    assert rm.find_manual_mask("img") is None
    with pytest.raises(ValueError, match="JPEG"):
        rm.load_manual_mask(tmp_path / "img.jpg")


def test_shape_mismatch_raises(tmp_path):
    p = tmp_path / "img.png"
    Image.fromarray(np.zeros((50, 60), np.uint8)).save(p)
    with pytest.raises(ValueError, match="shape"):
        ReviewManager(tmp_path).load_manual_mask(p, expected_shape=(100, 100))
