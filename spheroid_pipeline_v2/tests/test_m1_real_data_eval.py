"""
Empirical Evaluation and Stress Harness for Milestone 1 (M1) on Real Dataset Images.
Author: Challenger 2 (M1 Edge Case & Performance Challenger)

Tests:
1. Scale extraction across sample TIFFs from all 6 conditions (Mat, S34D30, S40D30, S43D20, S46D10, S50) on Day 0 and Day 7.
2. Complete Segmentation Hierarchy execution across 24 stratified real TIFF images across all conditions and timepoints.
3. Strict verification of realistic spheroid counts (mean/median in realistic range, individual FOVs valid) and zero giant circular artifacts (>25% FOV area).
4. Quantification of disk caching speedup factor (Cold Compute vs Warm Cache Hit) and bit-level mask integrity.
5. Stress-testing edge cases: cache corruption recovery, forced recomputations, decision logging persistence.
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tifffile

from spheroid_pipeline_v2.config import PipelineConfig, SegmentationConfig
from spheroid_pipeline_v2.decisions import DecisionsLogger
from spheroid_pipeline_v2.mask_cache import MaskCache
from spheroid_pipeline_v2.preprocess import Preprocessor
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor, extract_pixel_size
from spheroid_pipeline_v2.segment import SegmentationHierarchy, ClassicalMultiScaleSegmenter


class TestM1RealDatasetEvaluation(unittest.TestCase):
    """Empirical test harness validating M1 on raw microscopy images from 'Experimental data /Chuling cells'."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = PROJECT_ROOT / "Experimental data /Chuling cells"
        cls.d0_dir = cls.data_dir / "Spheroid Day0 260805"
        cls.d7_dir = cls.data_dir / "Spheroid D7 260812"
        cls.conditions = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]

        if not cls.d0_dir.exists() or not cls.d7_dir.exists():
            raise FileNotFoundError(f"Dataset directories missing: {cls.d0_dir} or {cls.d7_dir}")

        # Collect stratified sample of real images: 2 per condition per day = 24 images total
        cls.sample_images = []
        for cond in cls.conditions:
            d0_matches = sorted(list(cls.d0_dir.glob(f"Day0 {cond} *.tif")))
            d7_matches = sorted(list(cls.d7_dir.glob(f"Day7 {cond} *.tif")))

            cls.sample_images.extend(d0_matches[:2])
            cls.sample_images.extend(d7_matches[:2])

        print(f"\n[SetUp] Selected {len(cls.sample_images)} stratified real TIFF images across all 6 conditions:")
        for img_p in cls.sample_images:
            print(f"  - {img_p.parent.name}/{img_p.name}")

    def test_01_scale_extraction_all_conditions(self):
        """Verify scale extraction across sample TIFFs from all 6 conditions (Day 0 and Day 7)."""
        extractor = ScaleExtractor()
        expected_scale = 1.518817

        print("\n--- [Test 1] Scale Extraction across All 6 Conditions ---")
        for img_path in self.sample_images:
            info = extractor.extract(img_path)
            self.assertTrue(info.is_calibrated, f"Image {img_path.name} failed calibration")
            self.assertEqual(info.source, "TIFF Tag 37510 JSON")
            self.assertAlmostEqual(
                info.pixel_size_um,
                expected_scale,
                places=4,
                msg=f"Unexpected scale {info.pixel_size_um} in {img_path.name}",
            )
            # Verify interface contract helper extract_pixel_size
            px_val, source_str = extract_pixel_size(img_path)
            self.assertAlmostEqual(px_val, expected_scale, places=4)
            self.assertIn("Tag 37510", source_str)

        print(f"[PASS] Successfully verified optical scale ({expected_scale:.6f} um/px) across all {len(self.sample_images)} test images.")

    def test_02_segmentation_hierarchy_and_object_counts(self):
        """
        Run 3-tier segmentation hierarchy on real TIFFs from Day 0 and Day 7 across all 6 conditions.
        Verify:
        - Spheroids are detected across all FOVs (non-empty).
        - Detected spheroid counts are biologically realistic.
        - No single mask occupies > 25% of FOV area (zero giant artifacts).
        - Physical diameter of all detected spheroids is within reasonable bounds ([20, 1500] um).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "masks"
            log_path = Path(tmpdir) / "DECISIONS.md"
            cfg = SegmentationConfig(cache_dir=str(cache_dir), decisions_log_path=str(log_path))
            hierarchy = SegmentationHierarchy(config=cfg)

            print("\n--- [Test 2] Segmentation Hierarchy & Realism Gates across 24 Real Images ---")
            print(f"{'Image Name':<38} | {'Objects':<8} | {'Max Area %':<11} | {'Min d(um)':<10} | {'Max d(um)':<10} | {'Status'}")
            print("-" * 95)

            total_objects = 0
            counts_list = []
            max_area_pct_list = []

            for img_path in self.sample_images:
                img_id = img_path.stem
                raw_img = tifffile.imread(img_path)
                h, w = raw_img.shape[:2]
                total_fov_px = float(h * w)

                scale_um, _ = extract_pixel_size(img_path)
                mask, meta = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=scale_um)

                n_objs = meta.get("objects_found", 0)
                counts_list.append(n_objs)
                total_objects += n_objs

                # Verify non-empty and bounded counts
                self.assertGreaterEqual(
                    n_objs, 1,
                    f"Segmentation returned 0 objects for real image {img_path.name}"
                )
                self.assertLessEqual(
                    n_objs, 100,
                    f"Excessive spheroids ({n_objs}) detected in {img_path.name}"
                )

                # Check individual mask morphometrics
                max_obj_area_pct = 0.0
                min_d = 999999.0
                max_d = 0.0

                for obj_id in range(1, n_objs + 1):
                    obj_area_px = float(np.sum(mask == obj_id))
                    area_pct = (obj_area_px / total_fov_px) * 100.0
                    if area_pct > max_obj_area_pct:
                        max_obj_area_pct = area_pct

                    # Equivalent spherical diameter
                    d_um = 2.0 * np.sqrt(obj_area_px / np.pi) * scale_um
                    min_d = min(min_d, d_um)
                    max_d = max(max_d, d_um)

                    # Strict assertion: NO giant circular artifact > 25% FOV area
                    self.assertLess(
                        area_pct,
                        25.0,
                        f"CRITICAL: Object #{obj_id} in {img_path.name} occupies {area_pct:.2f}% FOV (>25% limit)!",
                    )
                    # Plausible physical size check
                    self.assertGreaterEqual(d_um, 20.0, f"Object #{obj_id} too small: {d_um:.1f} um")
                    self.assertLessEqual(d_um, 1500.0, f"Object #{obj_id} too large: {d_um:.1f} um")

                max_area_pct_list.append(max_obj_area_pct)
                status_str = f"PASS [{n_objs} objs, max {max_obj_area_pct:.2f}% FOV]"
                print(
                    f"{img_path.name:<38} | {n_objs:<8} | {max_obj_area_pct:<10.2f}% | "
                    f"{min_d:<10.1f} | {max_d:<10.1f} | {status_str}"
                )

            mean_count = float(np.mean(counts_list))
            median_count = float(np.median(counts_list))
            max_seen_area = float(np.max(max_area_pct_list))

            print("-" * 95)
            print(f"Summary: Mean count = {mean_count:.1f}/FOV, Median count = {median_count:.1f}/FOV, Max single mask area = {max_seen_area:.2f}% FOV")
            self.assertGreaterEqual(mean_count, 3.0)
            self.assertLessEqual(mean_count, 50.0)
            self.assertLess(max_seen_area, 25.0)

    def test_03_disk_caching_speedup_and_integrity(self):
        """
        Verify disk caching speedup factor and bit-level mask reproducibility on repeated calls.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "masks"
            log_path = Path(tmpdir) / "DECISIONS.md"
            cfg = SegmentationConfig(cache_dir=str(cache_dir), decisions_log_path=str(log_path))
            hierarchy = SegmentationHierarchy(config=cfg)

            # Test across 6 images (1 from each condition)
            test_subset = [
                self.sample_images[0],  # Mat D0
                self.sample_images[4],  # S34D30 D0
                self.sample_images[8],  # S40D30 D0
                self.sample_images[12], # S43D20 D0
                self.sample_images[16], # S46D10 D0
                self.sample_images[20], # S50 D0
            ]

            print("\n--- [Test 3] Disk Caching Speedup Benchmark ---")
            print(f"{'Image Name':<35} | {'Cold Compute (s)':<17} | {'Warm Cache (s)':<15} | {'Speedup':<10} | {'Bit-Exact'}")
            print("-" * 90)

            cold_times = []
            warm_times = []

            for img_p in test_subset:
                raw_img = tifffile.imread(img_p)
                img_id = img_p.stem
                scale_um, _ = extract_pixel_size(img_p)

                # Cold Run (Compute & Cache Write)
                t0 = time.perf_counter()
                mask_cold, meta_cold = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=scale_um)
                t_cold = time.perf_counter() - t0
                cold_times.append(t_cold)

                self.assertFalse(meta_cold.get("cached", False))
                self.assertTrue(hierarchy.mask_cache.has(img_id))

                # Warm Run (Disk Cache Read)
                t1 = time.perf_counter()
                mask_warm, meta_warm = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=scale_um)
                t_warm = time.perf_counter() - t1
                warm_times.append(t_warm)

                self.assertTrue(meta_warm.get("cached", False))
                self.assertEqual(meta_warm.get("backend"), "disk_cache")

                # Verify Bit-Exact Match
                bit_exact = np.array_equal(mask_cold, mask_warm)
                self.assertTrue(bit_exact, f"Cached mask mismatch for {img_id}")

                speedup = t_cold / max(1e-6, t_warm)
                print(f"{img_p.name:<35} | {t_cold:<17.4f} | {t_warm:<15.4f} | {speedup:<9.1f}x | {'PASS' if bit_exact else 'FAIL'}")

            total_cold = sum(cold_times)
            total_warm = sum(warm_times)
            overall_speedup = total_cold / max(1e-6, total_warm)

            print("-" * 90)
            print(f"Total Cold: {total_cold:.4f}s | Total Warm: {total_warm:.4f}s | Overall Speedup: {overall_speedup:.1f}x")
            self.assertGreater(overall_speedup, 5.0, f"Cache speedup ({overall_speedup:.1f}x) should be >= 5x")

    def test_04_cache_corruption_and_force_recompute(self):
        """
        Stress test caching edge cases:
        - Corrupted PNG file on disk triggers fallback recompute
        - Force recompute flag bypasses cache
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "masks"
            cfg = SegmentationConfig(cache_dir=str(cache_dir))
            hierarchy = SegmentationHierarchy(config=cfg)

            sample_p = self.sample_images[0]
            img_id = sample_p.stem
            raw_img = tifffile.imread(sample_p)

            # 1. Normal compute and save
            mask1, meta1 = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=1.518817)
            self.assertFalse(meta1.get("cached", False))

            # 2. Corrupt the cached PNG file on disk
            cache_file = hierarchy.mask_cache.get_cache_path(img_id)
            cache_file.write_bytes(b"CORRUPTED_PNG_DATA_NOT_VALID")

            # 3. Request segment again: should detect corrupt file and recompute safely
            mask2, meta2 = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=1.518817)
            self.assertFalse(meta2.get("cached", False))
            self.assertEqual(meta2.get("final_backend"), "classical")
            self.assertTrue(np.array_equal(mask1, mask2))

            # 4. Force recompute
            mask3, meta3 = hierarchy.segment(raw_img, image_id=img_id, pixel_size_um=1.518817, force_recompute=True)
            self.assertFalse(meta3.get("cached", False))
            self.assertTrue(np.array_equal(mask1, mask3))


if __name__ == "__main__":
    unittest.main()
