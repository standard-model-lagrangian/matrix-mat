"""
Unit and Integration Test Suite for Milestone 2:
Spheroid Filtering, Background Ring Contrast, Morphometrics, Volumetrics & QC Gating.
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
import pandas as pd
from skimage import draw, measure

from spheroid_pipeline_v2.config import PipelineConfig, QCConfig
from spheroid_pipeline_v2.measure import (
    OBJECTS_CSV_COLUMNS,
    SpheroidObjectRecord,
    compute_circularity,
    compute_ellipsoidal_volume,
    compute_spherical_volume,
    compute_volume_discrepancy_ratio,
    evaluate_morphological_qc,
    measure_single_object,
    save_objects_csv,
)
from spheroid_pipeline_v2.qc_filter import (
    QCFilter,
    check_image_plausibility,
    compute_annular_ring_contrast,
    filter_and_measure_objects,
    resolve_overlaps,
)


class TestOverlapResolution(unittest.TestCase):
    """Tests for candidate instance proposal overlap resolution."""

    def test_disjoint_masks_all_retained(self):
        """Completely disjoint candidate instances are both retained with unique labels."""
        m1 = np.zeros((100, 100), dtype=bool)
        m1[10:30, 10:30] = True
        m2 = np.zeros((100, 100), dtype=bool)
        m2[60:80, 60:80] = True

        master = resolve_overlaps([m1, m2])
        self.assertEqual(len(np.unique(master)), 3)  # 0, 1, 2
        self.assertEqual(np.sum(master == 1), 400)
        self.assertEqual(np.sum(master == 2), 400)

    def test_nested_duplicate_rejected(self):
        """Proposal with 100% overlap against a larger proposal is rejected as a duplicate."""
        m_large = np.zeros((100, 100), dtype=bool)
        m_large[20:60, 20:60] = True  # area 1600
        m_small = np.zeros((100, 100), dtype=bool)
        m_small[30:45, 30:45] = True  # area 225, entirely inside m_large

        master = resolve_overlaps([m_large, m_small])
        self.assertEqual(len(np.unique(master)), 2)  # 0, 1
        self.assertEqual(np.sum(master == 1), 1600)

    def test_partial_overlap_claims_unclaimed_pixels(self):
        """Proposal with <50% overlap claims only its unclaimed pixels."""
        m1 = np.zeros((100, 100), dtype=bool)
        m1[20:40, 20:40] = True  # 400 px
        m2 = np.zeros((100, 100), dtype=bool)
        m2[20:40, 35:55] = True  # 400 px, overlap is 20x5 = 100 px (25%)

        master = resolve_overlaps([m1, m2])
        self.assertEqual(len(np.unique(master)), 3)
        self.assertEqual(np.sum(master == 1), 400)
        self.assertEqual(np.sum(master == 2), 300)  # 400 - 100

    def test_50_percent_overlap_threshold(self):
        """Proposal with exactly >=50% overlap is rejected."""
        m1 = np.zeros((100, 100), dtype=bool)
        m1[20:40, 20:40] = True  # 400 px
        m2 = np.zeros((100, 100), dtype=bool)
        m2[20:40, 30:50] = True  # 400 px, overlap is 20x10 = 200 px (50%)

        master = resolve_overlaps([m1, m2], overlap_threshold=0.50)
        self.assertEqual(len(np.unique(master)), 2)  # Only m1 kept

    def test_sorting_by_scores(self):
        """Proposals with higher confidence score are assigned first even if smaller."""
        m1 = np.zeros((100, 100), dtype=bool)
        m1[20:50, 20:50] = True  # 900 px
        m2 = np.zeros((100, 100), dtype=bool)
        m2[20:40, 20:40] = True  # 400 px, inside m1

        # m2 has score 0.99, m1 has score 0.50
        master = resolve_overlaps([m1, m2], scores=[0.50, 0.99])
        # m2 is assigned first (400 px), m1 claims remaining (500 px) because m1 has 400/900 = 44.4% overlap (<50%)
        self.assertEqual(np.sum(master == 1), 400)
        self.assertEqual(np.sum(master == 2), 500)

    def test_2d_labeled_mask_input(self):
        """Accepts 2D labeled integer mask."""
        labeled = np.zeros((100, 100), dtype=np.int32)
        labeled[10:30, 10:30] = 1
        labeled[50:70, 50:70] = 2

        master = resolve_overlaps(labeled)
        self.assertEqual(len(np.unique(master)), 3)

    def test_empty_candidates_handling(self):
        """Handles empty candidate mask list gracefully."""
        master = resolve_overlaps([])
        self.assertEqual(master.shape, (100, 100))
        self.assertEqual(np.sum(master), 0)


class TestPhysicalSizeGate(unittest.TestCase):
    """Tests for physical equivalent diameter gating [min_d_um, max_d_um]."""

    def setUp(self):
        self.cfg = QCConfig(min_d_um=40.0, max_d_um=1500.0)
        self.pixel_size_um = 1.518817

    def test_sub_40um_debris_fails(self):
        """Object with equivalent diameter < 40 um fails with debris_undersized."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=35.0,
            bbox=(50, 50, 70, 70),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("debris_undersized" in r for r in reasons))

    def test_oversized_artifact_fails(self):
        """Object with equivalent diameter > 1500 um fails with oversized_artifact."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=1650.0,
            bbox=(50, 50, 450, 450),
            image_shape=(1000, 1000),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("oversized_artifact" in r for r in reasons))

    def test_valid_spheroid_passes(self):
        """Spheroid with d=250 um passes size gate."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=250.0,
            bbox=(50, 50, 200, 200),
            image_shape=(500, 500),
            mean_interior_intensity=0.35,
            mean_ring_intensity=0.75,
            contrast_ratio=0.53,
            circularity=0.88,
            solidity=0.95,
            eccentricity=0.35,
            config=self.cfg,
        )
        self.assertEqual(flag, "PASS")
        self.assertEqual(len(reasons), 0)

    def test_exact_boundary_conditions(self):
        """Objects at exact boundaries 40.0 um and 1500.0 um pass."""
        flag_min, _ = evaluate_morphological_qc(
            equivalent_diameter_um=40.0,
            bbox=(50, 50, 80, 80),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag_min, "PASS")

        flag_max, _ = evaluate_morphological_qc(
            equivalent_diameter_um=1500.0,
            bbox=(50, 50, 950, 950),
            image_shape=(2000, 2000),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag_max, "PASS")


class TestBorderMarginGate(unittest.TestCase):
    """Tests for rejecting objects touching or within border_margin_px of FOV boundaries."""

    def setUp(self):
        self.cfg = QCConfig(border_margin_px=15)

    def test_touching_left_border_fails(self):
        """Bounding box with min_c < 15 px fails with touches_image_border."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(100, 10, 150, 60),  # min_c = 10 < 15
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("touches_image_border" in r for r in reasons))

    def test_touching_right_border_fails(self):
        """Bounding box with max_c > 485 px (in 500 px wide image) fails."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(100, 440, 150, 490),  # max_c = 490 > (500 - 15)
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("touches_image_border" in r for r in reasons))

    def test_touching_top_border_fails(self):
        """Bounding box with min_r < 15 px fails."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(5, 100, 55, 150),  # min_r = 5 < 15
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("touches_image_border" in r for r in reasons))

    def test_touching_bottom_border_fails(self):
        """Bounding box with max_r > (500 - 15) fails."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(440, 100, 490, 150),  # max_r = 490 > 485
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("touches_image_border" in r for r in reasons))

    def test_safe_centered_object_passes(self):
        """Object with bbox >= 15 px from all borders passes."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(20, 20, 80, 80),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "PASS")


class TestBackgroundRingContrastGate(unittest.TestCase):
    """Tests for annular background ring contrast computation and dark contrast gating."""

    def setUp(self):
        self.cfg = QCConfig(interior_contrast_pct=0.10, background_ring_dilation_px=15)

    def test_annular_ring_excludes_neighboring_objects(self):
        """Annular background ring R_i = Dilate(M_i) \\ (bigcup M_j) excludes adjacent instances."""
        image = np.full((100, 100), 0.80, dtype=np.float32)
        # Spheroid 1: dark (0.30)
        image[30:50, 30:50] = 0.30
        # Spheroid 2: adjacent dark (0.30)
        image[30:50, 60:80] = 0.30

        m1 = np.zeros((100, 100), dtype=bool)
        m1[30:50, 30:50] = True
        all_m = np.zeros((100, 100), dtype=bool)
        all_m[30:50, 30:50] = True
        all_m[30:50, 60:80] = True

        mean_int, mean_ring, contrast = compute_annular_ring_contrast(
            image=image,
            obj_mask=m1,
            all_objects_mask=all_m,
            dilation_px=15,
        )
        self.assertAlmostEqual(mean_int, 0.30, places=2)
        self.assertAlmostEqual(mean_ring, 0.80, places=2)
        self.assertAlmostEqual(contrast, (0.80 - 0.30) / 0.80, places=2)

    def test_dark_spheroid_passes_contrast_gate(self):
        """Object with contrast >= 10% passes."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.45,
            mean_ring_intensity=0.75,
            contrast_ratio=(0.75 - 0.45) / 0.75,  # 40% contrast
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "PASS")

    def test_bright_blob_rejected_by_contrast_gate(self):
        """Bright artifact (e.g. gel flare) with negative or <10% contrast fails."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.85,
            mean_ring_intensity=0.60,
            contrast_ratio=(0.60 - 0.85) / 0.60,  # Negative contrast
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag, "FAIL")
        self.assertTrue(any("insufficient_dark_contrast" in r for r in reasons))

    def test_contrast_boundary_value(self):
        """Exactly 10% contrast passes; 9.0% fails."""
        # 10% contrast
        flag_pass, _ = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.63,
            mean_ring_intensity=0.70,
            contrast_ratio=0.10,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag_pass, "PASS")

        # 9% contrast
        flag_fail, reasons_fail = evaluate_morphological_qc(
            equivalent_diameter_um=100.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.637,
            mean_ring_intensity=0.70,
            contrast_ratio=0.09,
            circularity=0.90,
            solidity=0.95,
            eccentricity=0.30,
            config=self.cfg,
        )
        self.assertEqual(flag_fail, "FAIL")
        self.assertTrue(any("insufficient_dark_contrast" in r for r in reasons_fail))


class TestMorphologicalQCAndVolumetrics(unittest.TestCase):
    """Tests for morphometric calculations, volume models, and QC review flagging."""

    def setUp(self):
        self.cfg = QCConfig(
            min_circularity=0.65,
            min_solidity=0.85,
            max_eccentricity=0.80,
        )

    def test_spherical_volume_math(self):
        """V_sphere = (pi / 6) * d^3."""
        d = 200.0
        v = compute_spherical_volume(d)
        expected = (math.pi / 6.0) * (200.0 ** 3)
        self.assertAlmostEqual(v, expected, places=3)

    def test_ellipsoidal_volume_math(self):
        """V_ellip = (pi / 6) * a * b^2."""
        a, b = 250.0, 180.0
        v = compute_ellipsoidal_volume(a, b)
        expected = (math.pi / 6.0) * 250.0 * (180.0 ** 2)
        self.assertAlmostEqual(v, expected, places=3)

    def test_volume_discrepancy_ratio_math(self):
        """Discrepancy = |V_sph - V_ell| / V_sph."""
        v_sph = 1000000.0
        v_ell = 800000.0
        disc = compute_volume_discrepancy_ratio(v_sph, v_ell)
        self.assertAlmostEqual(disc, 0.20, places=4)

    def test_circularity_formula(self):
        """Circularity = 4 * pi * Area / Perimeter^2."""
        # True circle: r=50 -> A = pi*2500, P = 2*pi*50
        r = 50.0
        area = math.pi * (r ** 2)
        perim = 2.0 * math.pi * r
        circ = compute_circularity(area, perim)
        self.assertAlmostEqual(circ, 1.0, places=4)

    def test_review_flag_on_low_circularity(self):
        """Irregular shape with circularity 0.55 < 0.65 gets REVIEW flag."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=150.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.55,
            solidity=0.95,
            eccentricity=0.40,
            config=self.cfg,
        )
        self.assertEqual(flag, "REVIEW")
        self.assertTrue(any("low_circularity" in r for r in reasons))

    def test_review_flag_on_low_solidity(self):
        """Shape with solidity 0.78 < 0.85 gets REVIEW flag."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=150.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.85,
            solidity=0.78,
            eccentricity=0.40,
            config=self.cfg,
        )
        self.assertEqual(flag, "REVIEW")
        self.assertTrue(any("low_solidity" in r for r in reasons))

    def test_review_flag_on_high_eccentricity(self):
        """Elongated ellipse with eccentricity 0.88 > 0.80 gets REVIEW flag."""
        flag, reasons = evaluate_morphological_qc(
            equivalent_diameter_um=150.0,
            bbox=(50, 50, 150, 150),
            image_shape=(500, 500),
            mean_interior_intensity=0.30,
            mean_ring_intensity=0.70,
            contrast_ratio=0.57,
            circularity=0.85,
            solidity=0.95,
            eccentricity=0.88,
            config=self.cfg,
        )
        self.assertEqual(flag, "REVIEW")
        self.assertTrue(any("high_eccentricity" in r for r in reasons))


class TestPlausibilityAndAreaGate(unittest.TestCase):
    """Tests for single mask max area gate (25% FOV) and per-image plausibility check."""

    def test_single_mask_exceeding_25_percent_fov_is_rejected(self):
        """Single mask covering >25% FOV fails area gate."""
        h, w = 400, 400
        total_px = h * w
        mask_large = np.zeros((h, w), dtype=np.int32)
        mask_large[50:350, 50:350] = 1  # 300x300 = 90,000 px = 56.25% of 160,000 px
        image = np.full((h, w), 0.8, dtype=np.float32)
        image[mask_large > 0] = 0.2

        clean_mask, records = filter_and_measure_objects(image, mask_large)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].qc_flag, "FAIL")
        self.assertIn("exceeds_max_area_limit", records[0].qc_reasons)
        self.assertEqual(np.sum(clean_mask), 0)  # Excluded from clean mask

    def test_image_plausibility_flags_seg_fail_on_zero_objects(self):
        """Plausibility check returns False when 0 objects pass."""
        empty_clean = np.zeros((200, 200), dtype=np.int32)
        is_plausible, msg = check_image_plausibility(empty_clean)
        self.assertFalse(is_plausible)
        self.assertIn("SEG_FAIL", msg)


class TestObjectsCSVSchemaAndPersistence(unittest.TestCase):
    """Tests for 32-column objects.csv schema conformance and persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_csv = Path(self.test_dir) / "objects.csv"

    def test_exact_32_column_sequence(self):
        """SpheroidObjectRecord produces exactly 32 columns in mandated order."""
        rec = SpheroidObjectRecord(
            object_id="Mat_gel1_0001_t0_obj1",
            image_id="Day0 Mat gel1_0001_TRANS",
            timepoint="t0",
            condition="Mat",
            replicate="gel1",
            fov="0001",
            pair_key="Mat_gel1_0001",
            label_idx=1,
            pixel_size_um=1.518817,
            area_px=5000.0,
            perimeter_px=250.0,
            equivalent_diameter_px=79.79,
            centroid_y=250.5,
            centroid_x=300.2,
            area_um2=11533.9,
            equivalent_diameter_um=121.18,
            major_axis_um=130.0,
            minor_axis_um=115.0,
            aspect_ratio=1.13,
            volume_sphere_um3=931652.0,
            volume_ellipsoid_um3=899312.0,
            volume_discrepancy_ratio=0.035,
            circularity=0.98,
            solidity=0.99,
            eccentricity=0.45,
            mean_interior_intensity=0.35,
            mean_background_ring_intensity=0.75,
            contrast_ratio=0.533,
            qc_flag="PASS",
            qc_reasons="None",
            pair_id="Mat_gel1_0001_pair1",
            mask_source="automated",
        )
        d = rec.to_dict()
        self.assertEqual(list(d.keys()), OBJECTS_CSV_COLUMNS)
        self.assertEqual(len(d.keys()), 32)

    def test_save_and_reload_objects_csv(self):
        """save_objects_csv writes valid CSV readable by pandas with identical column headers."""
        rec = SpheroidObjectRecord(
            object_id="Mat_gel1_0001_t0_obj1",
            image_id="Day0 Mat gel1_0001_TRANS",
            timepoint="t0",
            condition="Mat",
            replicate="gel1",
            fov="0001",
            pair_key="Mat_gel1_0001",
            label_idx=1,
            pixel_size_um=1.518817,
            area_px=5000.0,
            perimeter_px=250.0,
            equivalent_diameter_px=79.79,
            centroid_y=250.5,
            centroid_x=300.2,
            area_um2=11533.9,
            equivalent_diameter_um=121.18,
            major_axis_um=130.0,
            minor_axis_um=115.0,
            aspect_ratio=1.13,
            volume_sphere_um3=931652.0,
            volume_ellipsoid_um3=899312.0,
            volume_discrepancy_ratio=0.035,
            circularity=0.98,
            solidity=0.99,
            eccentricity=0.45,
            mean_interior_intensity=0.35,
            mean_background_ring_intensity=0.75,
            contrast_ratio=0.533,
            qc_flag="PASS",
            qc_reasons="None",
            pair_id="UNPAIRED",
            mask_source="automated",
        )
        df_saved = save_objects_csv([rec], self.output_csv)
        self.assertTrue(self.output_csv.exists())

        df_loaded = pd.read_csv(self.output_csv)
        self.assertEqual(list(df_loaded.columns), OBJECTS_CSV_COLUMNS)
        self.assertEqual(len(df_loaded), 1)
        self.assertEqual(df_loaded.iloc[0]["object_id"], "Mat_gel1_0001_t0_obj1")
        self.assertEqual(df_loaded.iloc[0]["qc_flag"], "PASS")


class TestEndToEndFilterAndMeasureIntegration(unittest.TestCase):
    """End-to-end integration tests for filter_and_measure_objects on synthetic multi-spheroid fields."""

    def test_multi_spheroid_fov_filtering_and_clean_mask(self):
        """
        Creates synthetic FOV with 4 objects:
        1. Normal round dark spheroid (PASS)
        2. Irregular dark spheroid (REVIEW)
        3. Border-touching spheroid (FAIL - rejected from clean mask)
        4. Bright condensation blob (FAIL - rejected from clean mask)
        Asserts clean mask contains exactly 2 objects (renumbered 1 and 2),
        and records list contains all 4 objects with correct flags.
        """
        h, w = 500, 500
        image = np.full((h, w), 0.75, dtype=np.float32)
        raw_mask = np.zeros((h, w), dtype=np.int32)

        # 1. Normal round dark spheroid (center 150, 150, r=30) -> PASS
        rr1, cc1 = draw.disk((150, 150), 30, shape=(h, w))
        image[rr1, cc1] = 0.30
        raw_mask[rr1, cc1] = 1

        # 2. Irregular dark spheroid (center 350, 150, elongated ellipse) -> REVIEW (eccentricity > 0.80)
        rr2, cc2 = draw.ellipse(350, 150, 50, 15, shape=(h, w))
        image[rr2, cc2] = 0.30
        raw_mask[rr2, cc2] = 2

        # 3. Border-touching spheroid (center 10, 350, r=20) -> FAIL
        rr3, cc3 = draw.disk((10, 350), 20, shape=(h, w))
        image[rr3, cc3] = 0.30
        raw_mask[rr3, cc3] = 3

        # 4. Bright blob (center 350, 350, r=25, intensity 0.90) -> FAIL (contrast)
        rr4, cc4 = draw.disk((350, 350), 25, shape=(h, w))
        image[rr4, cc4] = 0.90
        raw_mask[rr4, cc4] = 4

        meta = {
            "image_id": "Day0 Mat gel1_0001_TRANS",
            "timepoint": "t0",
            "condition": "Mat",
            "replicate": "gel1",
            "fov": "0001",
            "pair_key": "Mat_gel1_0001",
            "pixel_size_um": 1.518817,
        }

        clean_mask, records = filter_and_measure_objects(image, raw_mask, image_metadata=meta)

        # Check records
        self.assertEqual(len(records), 4)
        flags = {r.object_id: r.qc_flag for r in records}
        self.assertEqual(records[0].qc_flag, "PASS")
        self.assertEqual(records[1].qc_flag, "REVIEW")
        self.assertEqual(records[2].qc_flag, "FAIL")
        self.assertEqual(records[3].qc_flag, "FAIL")

        # Clean mask must only contain PASS and REVIEW objects (labels 1 and 2)
        unique_clean = [int(x) for x in np.unique(clean_mask) if x > 0]
        self.assertEqual(unique_clean, [1, 2])
        self.assertEqual(int(np.max(clean_mask)), 2)

        # Border-touching and bright blob pixels must be 0 in clean_mask
        self.assertEqual(np.sum(clean_mask[rr3, cc3]), 0)
        self.assertEqual(np.sum(clean_mask[rr4, cc4]), 0)


if __name__ == "__main__":
    unittest.main()
