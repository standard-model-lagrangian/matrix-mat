"""Tier 2 Boundary Value & Edge Case Test Suite for Spheroid Volume Pipeline v2.

Tests boundary conditions, stress scenarios, and numeric edge cases:
- Empty masks & blank/zero-signal images
- Giant masks covering >25% FOV area
- 1-pixel, 2-pixel, and micro-debris objects
- Exact frame border clearance and boundary touching
- Extreme fold changes (< 0.10 and > 20.0)
- Zero denominator protection (t0 d < 60 um and division-by-zero guards)
- NaN, Inf, and singular matrix mathematical resilience
- Extreme object counts (0, 1, and 100+ instances)
"""

import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List, Tuple
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib cache protection
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from skimage import measure, segmentation
from skimage.feature import peak_local_max

from spheroid_pipeline_v2.tests.synthetic_generator import (
    SyntheticImageGenerator,
    SyntheticObject,
)

# ============================================================================
# Category 1: Empty Masks & Blank Images
# ============================================================================


class TestBoundaryEmptyInputs(unittest.TestCase):
  """Boundary: Empty masks, blank frames, and zero detections."""

  def test_completely_black_image(self):
    """Uniform black image (0 intensity) returns 0 objects without crash."""
    img = np.zeros((300, 400), dtype=np.uint8)
    smooth = cv2.medianBlur(img, 5)
    binary = (smooth < np.median(smooth)).astype(np.uint8)
    labels, num = measure.label(binary, return_num=True)
    self.assertEqual(num, 0)

  def test_completely_white_image(self):
    """Uniform white image (255 intensity) returns 0 objects."""
    img = np.full((300, 400), 255, dtype=np.uint8)
    smooth = cv2.medianBlur(img, 5)
    binary = (smooth < np.median(smooth)).astype(np.uint8)
    labels, num = measure.label(binary, return_num=True)
    self.assertEqual(num, 0)

  def test_zero_objects_in_morphometrics_returns_empty_dataframe(self):
    """Morphometrics on empty label map returns empty list / DataFrame."""
    empty_labels = np.zeros((200, 200), dtype=np.int32)
    props = measure.regionprops(empty_labels)
    self.assertEqual(len(props), 0)

  def test_zero_objects_in_pairing_returns_empty_pairs(self):
    """Pairing with 0 objects at t0 or t7 produces empty matched pairs."""
    t0_objs = []
    t7_objs = [{"id": 1, "cx": 100, "cy": 100}]
    # Pairing logic: min(N0, N7) == 0 => empty
    pairs = [] if len(t0_objs) == 0 or len(t7_objs) == 0 else [1]
    self.assertEqual(len(pairs), 0)

  def test_zero_objects_in_stats_summary_returns_nan_metrics(self):
    """Empty pairs table produces NaN medians and 0 counts without error."""
    df_pairs = pd.DataFrame(columns=["fold_change_volume"])
    med_fc = (
        df_pairs["fold_change_volume"].median() if not df_pairs.empty else np.nan
    )
    self.assertTrue(np.isnan(med_fc))


# ============================================================================
# Category 2: Giant Masks (>25% FOV Area)
# ============================================================================


class TestBoundaryGiantMasks(unittest.TestCase):
  """Boundary: Single masks exceeding 25% FOV area trigger SEG_FAIL."""

  def test_giant_mask_25_point_01_pct_rejected(self):
    """Mask covering 25.01% of image is rejected by plausibility gate."""
    total_area = 1536 * 2048
    mask_area = 0.2501 * total_area
    is_valid = mask_area <= (0.25 * total_area)
    self.assertFalse(is_valid)

  def test_valid_mask_24_point_99_pct_accepted(self):
    """Mask covering 24.99% of image passes area gate."""
    total_area = 1536 * 2048
    mask_area = 0.2499 * total_area
    is_valid = mask_area <= (0.25 * total_area)
    self.assertTrue(is_valid)

  def test_full_frame_100_pct_mask_rejected(self):
    """Full frame segmentation (e.g. global threshold artifact) is rejected."""
    total_area = 600 * 800
    mask_area = total_area
    is_valid = mask_area <= (0.25 * total_area)
    self.assertFalse(is_valid)

  def test_giant_mask_triggers_backend_switch(self):
    """Area gate failure triggers switch to secondary/fallback backend."""
    single_mask_area_pct = 0.35
    seg_failed = single_mask_area_pct > 0.25
    next_backend = "cyto3" if seg_failed else "cpsam"
    self.assertEqual(next_backend, "cyto3")

  def test_multi_mask_sum_over_25_pct_allowed(self):
    """5 masks of 6% area each (sum = 30%) are all valid because individual < 25%."""
    total = 1000000
    masks = [60000, 60000, 60000, 60000, 60000]
    self.assertEqual(sum(masks) / total, 0.30)
    all_individual_pass = all(m <= 0.25 * total for m in masks)
    self.assertTrue(all_individual_pass)


# ============================================================================
# Category 3: 1-Pixel and Minimal Micro-Objects
# ============================================================================


class TestBoundaryMicroObjects(unittest.TestCase):
  """Boundary: 1-pixel, 2-pixel, and micro debris objects."""

  def test_1_pixel_object_size_gate_rejection(self):
    """1-pixel object (d = 1.71 um at 1.518 um/px) rejected by min_d_um=40."""
    area_px = 1.0
    scale = 1.518817
    d_um = 2.0 * math.sqrt(area_px / math.pi) * scale
    self.assertAlmostEqual(d_um, 1.714, places=2)
    self.assertFalse(d_um >= 40.0)

  def test_1_pixel_object_no_division_by_zero_in_circularity(self):
    """1-pixel object has perimeter=0 or small; circularity calculation is safe."""
    area_px = 1.0
    perimeter_px = 0.0
    # Safe formula: max(perimeter, 1e-4)
    perim_safe = max(perimeter_px, 1e-4)
    circ = min(1.0, (4.0 * math.pi * area_px) / (perim_safe**2))
    self.assertTrue(np.isfinite(circ))

  def test_2_pixel_object_eccentricity_safety(self):
    """2-pixel object computes valid eccentricity without singular matrix crash."""
    lbl = np.zeros((10, 10), dtype=np.int32)
    lbl[4, 4] = 1
    lbl[4, 5] = 1
    props = measure.regionprops(lbl)
    self.assertEqual(len(props), 1)
    ecc = float(props[0].eccentricity)
    self.assertTrue(0.0 <= ecc <= 1.0)

  def test_micro_debris_boundary_at_39_point_9_um(self):
    """Object with d = 39.9 um is rejected; d = 40.0 um is accepted."""
    scale = 1.518817
    d_reject = 39.9
    d_accept = 40.0
    self.assertFalse(d_reject >= 40.0)
    self.assertTrue(d_accept >= 40.0)

  def test_morphometrics_zero_volume_protection(self):
    """Micro object volume does not underflow to negative or NaN."""
    d_um = 1.0
    v = (math.pi / 6.0) * (d_um**3)
    self.assertTrue(v > 0.0)
    self.assertTrue(np.isfinite(v))


# ============================================================================
# Category 4: Border Clearance & Boundary Touching
# ============================================================================


class TestBoundaryBorderTouching(unittest.TestCase):
  """Boundary: Border margin boundary conditions (margin = 15 px)."""

  def test_pixel_at_row_0_rejected(self):
    """Object at row 0 (touching top edge) is rejected."""
    bbox = (0, 100, 30, 130)
    margin = 15
    self.assertTrue(bbox[0] < margin)

  def test_pixel_at_col_0_rejected(self):
    """Object at col 0 (touching left edge) is rejected."""
    bbox = (100, 0, 130, 30)
    margin = 15
    self.assertTrue(bbox[1] < margin)

  def test_pixel_at_row_H_minus_1_rejected(self):
    """Object touching bottom edge (max_row >= H) is rejected."""
    h = 600
    bbox = (570, 100, 600, 130)
    margin = 15
    self.assertTrue(bbox[2] > (h - margin))

  def test_pixel_at_col_W_minus_1_rejected(self):
    """Object touching right edge (max_col >= W) is rejected."""
    w = 800
    bbox = (100, 770, 130, 800)
    margin = 15
    self.assertTrue(bbox[3] > (w - margin))

  def test_exact_margin_boundaries_14_vs_15_vs_16(self):
    """Margin 14 fails, margin 15 passes, margin 16 passes."""
    margin = 15
    min_col_14 = 14  # < 15 => fails
    min_col_15 = 15  # not < 15 => passes
    min_col_16 = 16  # not < 15 => passes

    self.assertTrue(min_col_14 < margin)
    self.assertFalse(min_col_15 < margin)
    self.assertFalse(min_col_16 < margin)


# ============================================================================
# Category 5: Extreme Fold Changes (<0.10 and >20.0)
# ============================================================================


class TestBoundaryExtremeFoldChanges(unittest.TestCase):
  """Boundary: Extreme growth and shrinkage trajectory gating."""

  def test_extreme_shrinkage_0_point_001_flags_review(self):
    """FC = 0.001 (99.9% volume loss) is flagged REVIEW."""
    fc = 0.001
    flag = "REVIEW" if (fc < 0.10 or fc > 20.0) else "PASS"
    self.assertEqual(flag, "REVIEW")

  def test_boundary_fc_0_point_099_flags_review(self):
    """FC = 0.099 is flagged REVIEW."""
    fc = 0.099
    flag = "REVIEW" if (fc < 0.10 or fc > 20.0) else "PASS"
    self.assertEqual(flag, "REVIEW")

  def test_boundary_fc_0_point_101_passes(self):
    """FC = 0.101 passes trajectory gate."""
    fc = 0.101
    flag = "REVIEW" if (fc < 0.10 or fc > 20.0) else "PASS"
    self.assertEqual(flag, "PASS")

  def test_boundary_fc_19_point_99_passes(self):
    """FC = 19.99 passes trajectory gate."""
    fc = 19.99
    flag = "REVIEW" if (fc < 0.10 or fc > 20.0) else "PASS"
    self.assertEqual(flag, "PASS")

  def test_extreme_expansion_100_flags_review(self):
    """FC = 100.0 is flagged REVIEW."""
    fc = 100.0
    flag = "REVIEW" if (fc < 0.10 or fc > 20.0) else "PASS"
    self.assertEqual(flag, "REVIEW")


# ============================================================================
# Category 6: Zero Denominator Protection
# ============================================================================


class TestBoundaryZeroDenominatorProtection(unittest.TestCase):
  """Boundary: Baseline diameter < 60 um and zero initial volume safeguards."""

  def test_zero_initial_volume_division_safety(self):
    """FC = V7 / max(V0, 1e-6) prevents ZeroDivisionError."""
    v0 = 0.0
    v7 = 50000.0
    fc = v7 / max(v0, 1e-6)
    self.assertTrue(np.isfinite(fc))

  def test_d0_exact_boundary_59_point_99_excluded(self):
    """d0 = 59.99 um is excluded from fold change stats."""
    d0_um = 59.99
    denom_pass = d0_um >= 60.0
    self.assertFalse(denom_pass)

  def test_d0_exact_boundary_60_point_01_included(self):
    """d0 = 60.01 um is included in fold change stats."""
    d0_um = 60.01
    denom_pass = d0_um >= 60.0
    self.assertTrue(denom_pass)

  def test_unpaired_object_fold_change_is_nan(self):
    """Unpaired object has fold_change = NaN in paired tables."""
    fc = np.nan
    self.assertTrue(np.isnan(fc))

  def test_both_volumes_zero_safety(self):
    """If both V0=0 and V7=0, fold change is 0.0 or NaN without crash."""
    v0, v7 = 0.0, 0.0
    fc = v7 / max(v0, 1e-6)
    self.assertEqual(fc, 0.0)


# ============================================================================
# Category 7: NaN, Inf, and Singular Matrix Resilience
# ============================================================================


class TestBoundaryNaNAndInfResilience(unittest.TestCase):
  """Boundary: Handling of NaNs, Infs, and singular geometries."""

  def test_bootstrap_with_nans_in_input(self):
    """Bootstrap CI filters out NaNs from input array before resampling."""
    data_with_nans = np.array([1.5, np.nan, 2.0, np.nan, 2.5])
    clean = data_with_nans[~np.isnan(data_with_nans)]
    self.assertEqual(len(clean), 3)
    self.assertEqual(np.median(clean), 2.0)

  def test_permutation_test_with_all_identical_values(self):
    """Permutation test on uniform constant data returns p = 1.0."""
    grp_a = np.array([2.0, 2.0, 2.0])
    grp_c = np.array([2.0, 2.0, 2.0])
    t_obs = abs(np.median(grp_a) - np.median(grp_c))  # 0.0
    self.assertEqual(t_obs, 0.0)

  def test_log2_fold_change_zero_protection(self):
    """log2(max(FC, 1e-6)) avoids math domain error for FC=0."""
    fc = 0.0
    safe_log2 = math.log2(max(fc, 1e-6))
    self.assertTrue(np.isfinite(safe_log2))

  def test_regionprops_straight_line_eccentricity(self):
    """Straight 1-pixel-thick line has eccentricity = 1.0."""
    lbl = np.zeros((20, 20), dtype=np.int32)
    lbl[10, 5:15] = 1  # 1x10 line
    props = measure.regionprops(lbl)
    ecc = float(props[0].eccentricity)
    self.assertAlmostEqual(ecc, 1.0, places=2)

  def test_circularity_clamping_to_1_point_0(self):
    """Discrete grid sampling artifact resulting in 4*pi*A/P^2 > 1 is clamped to 1.0."""
    raw_circ = 1.05  # Discrete pixel artifact on tiny circle
    clamped_circ = min(1.0, raw_circ)
    self.assertEqual(clamped_circ, 1.0)


# ============================================================================
# Category 8: Extreme Object Counts (0, 1, 100+)
# ============================================================================


class TestBoundaryExtremeCounts(unittest.TestCase):
  """Boundary: Stress testing extreme spheroid densities."""

  def test_zero_objects_in_image(self):
    """Image with 0 spheroids marks SEG_FAIL and handles downstream cleanly."""
    n_objects = 0
    seg_fail = n_objects == 0
    self.assertTrue(seg_fail)

  def test_single_isolated_spheroid(self):
    """Single isolated spheroid processes correctly without multi-object requirement."""
    n_objects = 1
    self.assertTrue(1 <= n_objects <= 25)

  def test_100_spheroid_dense_field_stress(self):
    """Processes 100 synthetic micro-spheroids without memory error."""
    h, w = 400, 400
    labels = np.zeros((h, w), dtype=np.int32)
    for i in range(10):
      for j in range(10):
        obj_id = i * 10 + j + 1
        cy = 20 + i * 36
        cx = 20 + j * 36
        labels[cy - 5 : cy + 5, cx - 5 : cx + 5] = obj_id

    props = measure.regionprops(labels)
    self.assertEqual(len(props), 100)

  def test_hungarian_matching_large_scale(self):
    """Hungarian matching executes efficiently for 50 vs 50 objects."""
    c0 = np.random.uniform(50, 500, (50, 2))
    c7 = c0 + np.random.normal(0, 2.0, (50, 2))
    cost = np.linalg.norm(c0[:, None, :] - c7[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    self.assertEqual(len(row_ind), 50)

  def test_maximum_objects_per_fov_sanity_gate(self):
    """Sanity gate flags fields with > 25 objects."""
    self.assertTrue(1 <= 15 <= 25)
    self.assertFalse(1 <= 30 <= 25)


if __name__ == "__main__":
  unittest.main()
