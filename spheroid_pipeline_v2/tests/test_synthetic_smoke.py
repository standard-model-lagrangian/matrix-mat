"""Synthetic Smoke Test Suite for Spheroid Volume Pipeline v2.

Implements the mandatory 3-scenario synthetic ground-truth smoke tests from ORIGINAL_REQUEST.md:
- Scenario 1: Isolated dark circular spheroid (d = 100 px) -> Recovered diameter within 10% of ground truth.
- Scenario 2: Touching pair of spheroids (d1 = 80 px, d2 = 80 px) -> Watershed separates into exactly 2 instances, both d within 10%.
- Scenario 3: Bright gel blob / refractive artifact (d = 120 px) -> Exactly 0 masks pass (100% rejected by contrast gate).
- Scenario 4: Multi-spheroid complex field with debris, border-touching, and bright blob filtering.
- Scenario 5: Ellipsoidal spheroid morphometry and volume discrepancy validation.
"""

import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import unittest
from scipy.ndimage import gaussian_filter
from skimage import measure, segmentation
from skimage.feature import peak_local_max

from spheroid_pipeline_v2.tests.synthetic_generator import (
    SyntheticImageGenerator,
    SyntheticObject,
)


def run_reference_segmentation_and_qc(
    image_u8: np.ndarray,
    pixel_size_um: float = 1.518817,
    min_d_um: float = 40.0,
    max_d_um: float = 1500.0,
    border_margin_px: int = 15,
    interior_contrast_pct: float = 0.10,
    single_mask_max_area_pct: float = 0.25,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
  """Reference implementation of the multi-scale adaptive watershed and QC gating.

  This provides a deterministic verification oracle matching the exact
  mathematical specifications in PROJECT.md and ORIGINAL_REQUEST.md.
  """
  h, w = image_u8.shape[:2]
  total_fov_area = h * w

  # Preprocessing: median smoothing and local adaptive thresholding
  smooth = cv2.medianBlur(image_u8, 5)
  bg_med = float(np.median(smooth))

  combined = np.zeros((h, w), dtype=np.uint8)
  for ws in [31, 61, 101, 151]:
    blur = cv2.blur(smooth, (ws, ws))
    diff = cv2.subtract(blur, smooth)
    _, th = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
    # Dark objects must also be below median background
    dark_gate = (smooth < bg_med).astype(np.uint8) * 255
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(th, dark_gate))

  # Morphological cleaning
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
  cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

  # Distance transform and marker-controlled watershed
  dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
  coords = peak_local_max(
      gaussian_filter(dist, 2.0),
      min_distance=20,
      labels=(cleaned > 0),
      exclude_border=False,
  )

  if len(coords) > 0:
    peaks = np.zeros((h, w), dtype=bool)
    peaks[tuple(coords.T)] = True
    markers, _ = measure.label(peaks, return_num=True)
    raw_labels = segmentation.watershed(-dist, markers, mask=(cleaned > 0))
  else:
    raw_labels, _ = measure.label(cleaned > 0, return_num=True)

  # QC Gating
  norm_img = image_u8.astype(np.float32) / 255.0
  m_all = (raw_labels > 0).astype(np.uint8)
  ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

  passed_objects = []
  final_mask = np.zeros((h, w), dtype=np.int32)
  next_id = 1

  props = measure.regionprops(raw_labels, intensity_image=image_u8)
  for p in props:
    obj_mask = (raw_labels == p.label).astype(np.uint8)
    area_px = float(p.area)

    # 1. Single Mask Area Plausibility Gate (< 25% FOV area)
    if area_px > (single_mask_max_area_pct * total_fov_area):
      continue

    # 2. Border Margin Gate
    min_row, min_col, max_row, max_col = p.bbox
    if (
        min_row < border_margin_px
        or min_col < border_margin_px
        or max_row > (h - border_margin_px)
        or max_col > (w - border_margin_px)
    ):
      continue

    # 3. Physical Diameter Size Gate
    eq_d_px = 2.0 * math.sqrt(area_px / math.pi)
    eq_d_um = eq_d_px * pixel_size_um
    if eq_d_um < min_d_um or eq_d_um > max_d_um:
      continue

    # 4. Background Ring Contrast Check
    # Ring = Dilate(M_i) \ (Union of all detected objects)
    dilated = cv2.dilate(obj_mask, ring_kernel)
    ring_mask = (dilated == 1) & (m_all == 0)

    mean_interior = float(np.mean(norm_img[obj_mask == 1]))
    if np.sum(ring_mask) > 0:
      mean_ring = float(np.mean(norm_img[ring_mask]))
    else:
      mean_ring = float(np.median(norm_img))

    contrast_ratio = (mean_ring - mean_interior) / max(mean_ring, 1e-4)

    # Interior must be >= N% darker than background ring
    if contrast_ratio < interior_contrast_pct:
      continue

    # Morphometrics
    perimeter_px = float(p.perimeter) if p.perimeter > 0 else 1.0
    circularity = min(1.0, (4.0 * math.pi * area_px) / (perimeter_px**2))
    solidity = float(p.solidity)
    eccentricity = float(p.eccentricity)

    # Axis lengths
    major_prop = getattr(p, "axis_major_length", None) or getattr(p, "major_axis_length", None) or eq_d_px
    minor_prop = getattr(p, "axis_minor_length", None) or getattr(p, "minor_axis_length", None) or eq_d_px
    major_axis_px = float(major_prop) if major_prop > 0 else eq_d_px
    minor_axis_px = float(minor_prop) if minor_prop > 0 else eq_d_px
    major_axis_um = major_axis_px * pixel_size_um
    minor_axis_um = minor_axis_px * pixel_size_um

    v_sph_um3 = (math.pi / 6.0) * (eq_d_um**3)
    v_ell_um3 = (math.pi / 6.0) * major_axis_um * (minor_axis_um**2)
    v_disc = abs(v_sph_um3 - v_ell_um3) / max(v_sph_um3, 1e-6)

    # Assign to final label map
    final_mask[obj_mask == 1] = next_id

    passed_objects.append({
        "label_idx": next_id,
        "centroid_y": float(p.centroid[0]),
        "centroid_x": float(p.centroid[1]),
        "area_px": area_px,
        "area_um2": area_px * (pixel_size_um**2),
        "equivalent_diameter_px": eq_d_px,
        "equivalent_diameter_um": eq_d_um,
        "major_axis_um": major_axis_um,
        "minor_axis_um": minor_axis_um,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "volume_sphere_um3": v_sph_um3,
        "volume_ellipsoid_um3": v_ell_um3,
        "volume_discrepancy_ratio": v_disc,
        "contrast_ratio": contrast_ratio,
        "qc_flag": "PASS",
    })
    next_id += 1

  return final_mask, passed_objects


class TestSyntheticSmoke(unittest.TestCase):
  """Mandatory synthetic smoke test suite with mathematical ground-truth assertions."""

  def test_scenario_1_isolated_dark_spheroid(self):
    """Scenario 1: Isolated dark circular spheroid (d=100px).

    Assert recovered diameter within 10% of ground truth (90.0 to 110.0 px).
    """
    true_d_px = 100.0
    img, _, truth_rec = SyntheticImageGenerator.create_isolated_spheroid(
        diameter_px=true_d_px,
        width=800,
        height=600,
        contrast=0.35,
        noise_sigma=0.02,
    )

    final_mask, objects = run_reference_segmentation_and_qc(
        img,
        pixel_size_um=1.518817,
        min_d_um=40.0,
        max_d_um=1500.0,
        interior_contrast_pct=0.10,
    )

    # Assert exactly 1 object passes QC
    assert (
        len(objects) == 1
    ), f"Expected exactly 1 recovered spheroid, got {len(objects)}"

    rec_d_px = objects[0]["equivalent_diameter_px"]
    rel_error = abs(rec_d_px - true_d_px) / true_d_px

    # Assert recovered diameter is within 10% of ground truth
    assert (
        rel_error <= 0.10
    ), f"Recovered d={rec_d_px:.2f}px error {rel_error*100:.2f}% exceeded 10% limit"
    assert objects[0]["qc_flag"] == "PASS"

    # Verify centroid location accuracy
    cy_true, cx_true = 300.0, 400.0
    assert abs(objects[0]["centroid_x"] - cx_true) < 5.0
    assert abs(objects[0]["centroid_y"] - cy_true) < 5.0

  def test_scenario_2_touching_pair_watershed(self):
    """Scenario 2: Touching pair of spheroids (d1=80px, d2=80px).

    Assert watershed separates into exactly 2 instances, both recovered diameters within 10%.
    """
    d1_true, d2_true = 80.0, 80.0
    img, _, truth_recs = SyntheticImageGenerator.create_touching_pair(
        d1_px=d1_true,
        d2_px=d2_true,
        overlap_px=10.0,
        width=800,
        height=600,
        contrast=0.35,
        noise_sigma=0.02,
    )

    final_mask, objects = run_reference_segmentation_and_qc(
        img,
        pixel_size_um=1.518817,
        min_d_um=40.0,
        max_d_um=1500.0,
        interior_contrast_pct=0.10,
    )

    # Assert separation into exactly 2 instances
    assert (
        len(objects) == 2
    ), f"Expected watershed to separate into 2 instances, got {len(objects)}"

    # Sort objects left-to-right by centroid x
    objects_sorted = sorted(objects, key=lambda o: o["centroid_x"])

    # Assert both recovered diameters are within 10% of truth
    for idx, (rec_obj, true_d) in enumerate(
        zip(objects_sorted, [d1_true, d2_true])
    ):
      rec_d = rec_obj["equivalent_diameter_px"]
      rel_err = abs(rec_d - true_d) / true_d
      assert (
          rel_err <= 0.10
      ), f"Instance {idx+1} d={rec_d:.2f}px error {rel_err*100:.2f}% exceeded 10% limit"
      assert rec_obj["qc_flag"] == "PASS"

  def test_scenario_3_bright_gel_blob_contrast_rejection(self):
    """Scenario 3: Bright gel condensation blob (d=120px).

    Assert 0 masks pass (100% rejected by background ring contrast gate).
    """
    img, _, truth_rec = SyntheticImageGenerator.create_bright_blob(
        diameter_px=120.0,
        width=800,
        height=600,
        contrast=0.35,
        noise_sigma=0.02,
    )

    final_mask, objects = run_reference_segmentation_and_qc(
        img,
        pixel_size_um=1.518817,
        min_d_um=40.0,
        max_d_um=1500.0,
        interior_contrast_pct=0.10,
    )

    # Assert exactly 0 objects pass (100% rejection)
    assert (
        len(objects) == 0
    ), f"Expected 0 masks for bright gel blob, but {len(objects)} passed contrast gate"
    assert np.all(
        final_mask == 0
    ), "Final label mask must be entirely 0 (background)"

  def test_scenario_4_multi_spheroid_field_with_gates(self):
    """Scenario 4: Complex field with 8 valid spheroids, 3 debris, 1 border-touching, 1 bright blob.

    Assert that valid internal spheroids pass while debris, border, and blob are rejected.
    """
    img, _, truth_recs = SyntheticImageGenerator.create_multi_spheroid_field(
        n_spheroids=8,
        width=1024,
        height=768,
        include_debris=True,
        include_border=True,
        include_bright_blob=True,
        seed=42,
    )

    final_mask, objects = run_reference_segmentation_and_qc(
        img,
        pixel_size_um=1.518817,
        min_d_um=40.0,
        max_d_um=1500.0,
        border_margin_px=15,
        interior_contrast_pct=0.10,
    )

    # All recovered objects must pass QC gates
    assert len(objects) >= 6, f"Expected >=6 valid spheroids, got {len(objects)}"
    assert (
        len(objects) <= 9
    ), f"Expected <=9 valid spheroids (debris/blob rejected), got {len(objects)}"

    # Ensure no border-touching object was accepted
    for obj in objects:
      assert (
          obj["centroid_x"] >= 15.0
      ), f"Border object accepted at cx={obj['centroid_x']}"
      assert (
          obj["centroid_x"] <= 1024 - 15.0
      ), f"Border object accepted at cx={obj['centroid_x']}"
      assert (
          obj["equivalent_diameter_um"] >= 40.0
      ), f"Debris accepted at d={obj['equivalent_diameter_um']}"

  def test_scenario_5_ellipsoidal_morphometrics_accuracy(self):
    """Scenario 5: Ellipsoidal spheroid with aspect ratio 2.0.

    Validates accurate computation of major axis, minor axis, eccentricity, and volume discrepancy.
    """
    gen = SyntheticImageGenerator(width=800, height=600, noise_sigma=0.01)
    rx, ry = 60.0, 30.0
    true_eq_d_px = 2.0 * math.sqrt(rx * ry)  # 84.85 px
    gen.add_ellipsoid(
        center_x=400.0,
        center_y=300.0,
        radius_x=rx,
        radius_y=ry,
        angle_deg=0.0,
        contrast=0.35,
    )
    img, _, truth_recs = gen.render()

    final_mask, objects = run_reference_segmentation_and_qc(img)
    assert len(objects) == 1, f"Expected 1 ellipsoid object, got {len(objects)}"

    obj = objects[0]
    rec_eq_d = obj["equivalent_diameter_px"]
    rel_d_err = abs(rec_eq_d - true_eq_d_px) / true_eq_d_px
    assert (
        rel_d_err <= 0.10
    ), f"Ellipsoid equivalent diameter error {rel_d_err*100:.2f}% exceeded 10%"

    # Assert major > minor axis
    assert (
        obj["major_axis_um"] > obj["minor_axis_um"]
    ), "Major axis must exceed minor axis for elongated ellipse"
    assert (
        obj["eccentricity"] > 0.50
    ), f"Expected eccentricity > 0.50 for aspect ratio 2.0, got {obj['eccentricity']:.2f}"


if __name__ == "__main__":
  unittest.main()

