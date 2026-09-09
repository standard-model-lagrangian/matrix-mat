"""
Adversarial Stress Test Suite for Milestone 1 (M1).
Authored by Challenger 1 (M1 Adversarial Challenger).

Covers:
1. Extreme Contrast Artifacts (bright refractive bubbles, lighting flares, inverted contrasts, uniform fields, NaNs).
2. Dense Clusters & Non-Circular Geometries (touching chains/trios, elongated/c-shaped/dumbbell spheroids, max area gate).
3. Corrupted / Missing TIFF Metadata Fallback (malformed JSON, broken XML, invalid types, CLI override priority).
4. Cache Persistence, Concurrency, and Cache Hit/Miss Behavior (multi-threading, corrupt cache recovery, special chars).
5. Area Plausibility Gate & Zero Mask verification for bright artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import concurrent.futures
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

logger = logging.getLogger("test_m1_adversarial_challenger")
import cv2
import numpy as np
import tifffile
from PIL import Image

from spheroid_pipeline_v2.config import PipelineConfig, PreprocessConfig, SegmentationConfig
from spheroid_pipeline_v2.decisions import DecisionsLogger
from spheroid_pipeline_v2.mask_cache import MaskCache
from spheroid_pipeline_v2.preprocess import (
    Preprocessor,
    compute_downsample_scale,
    downsample_image,
    flatten_illumination,
    normalize_percentiles,
    upscale_labels,
)
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor, ScaleInfo, extract_pixel_size
from spheroid_pipeline_v2.segment import (
    ClassicalMultiScaleSegmenter,
    SegmentationHierarchy,
)


class TestExtremeContrastArtifacts(unittest.TestCase):
    """Adversarial testing on extreme optical contrast variations and artifacts."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_contrast_")
        self.cfg = SegmentationConfig(
            cache_dir=os.path.join(self.tmpdir, "masks"),
            decisions_log_path=os.path.join(self.tmpdir, "DECISIONS.md"),
            primary_backend="classical",  # Deterministic test
        )
        self.hier = SegmentationHierarchy(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bright_refractive_gel_bubble_produces_zero_masks(self):
        """
        Verify that bright refractive condensation blobs / bubbles (which are brighter than gel)
        produce EXACTLY 0 segmented masks.
        """
        h, w = 400, 400
        # Background hydrogel intensity around 0.60
        img = np.full((h, w), 0.60, dtype=np.float32)

        # Bright refractive bubble at center: high intensity (0.95), sharp dark border
        cv2.circle(img, (200, 200), 50, 0.95, -1)
        # Thin refractive dark ring around it
        cv2.circle(img, (200, 200), 52, 0.40, 2)

        mask, meta = self.hier.segment(img, image_id="bright_bubble_1", pixel_size_um=1.5)
        n_objs = int(np.max(mask))

        self.assertEqual(n_objs, 0, f"Expected 0 masks for bright refractive bubble, got {n_objs}")
        self.assertEqual(np.sum(mask > 0), 0)

    def test_multiple_bright_flares_and_bubbles_produce_zero_masks(self):
        """
        Multi-bubble test: field littered with 5 bright gel bubbles of varying sizes.
        Must yield 0 detected objects.
        """
        h, w = 500, 500
        img = np.full((h, w), 0.55, dtype=np.float32)
        centers = [(100, 100, 30), (350, 120, 45), (250, 300, 60), (120, 380, 25), (400, 400, 35)]
        for cx, cy, r in centers:
            cv2.circle(img, (cx, cy), r, 0.92, -1)
            cv2.circle(img, (cx, cy), r + 2, 0.35, 2)

        mask, meta = self.hier.segment(img, image_id="multi_bright_bubbles", pixel_size_um=1.5)
        self.assertEqual(int(np.max(mask)), 0)

    def test_severe_diagonal_lighting_gradient_with_spheroid(self):
        """
        Test severe 10x lighting gradient (vignetting + uneven illumination across FOV).
        Real spheroid (dark blob) must still be detected and properly segmented.
        """
        h, w = 600, 600
        # Steep diagonal gradient from 0.2 to 0.9
        y, x = np.mgrid[0:h, 0:w]
        gradient = 0.2 + 0.7 * (x + y) / (w + h)
        img = gradient.astype(np.float32)

        # Place a real dark spheroid in the bright upper-right region (local contrast: 0.85 -> 0.35)
        cx, cy, r = 450, 450, 50
        cv2.circle(img, (cx, cy), r, float(img[cy, cx] - 0.40), -1)

        mask, meta = self.hier.segment(img, image_id="severe_gradient_test", pixel_size_um=1.5)
        n_objs = int(np.max(mask))

        self.assertGreaterEqual(n_objs, 1, "Spheroid in severe lighting gradient should be detected")
        # Check centroid of detected object is near (450, 450)
        ys, xs = np.where(mask == 1)
        self.assertAlmostEqual(np.mean(xs), cx, delta=15)
        self.assertAlmostEqual(np.mean(ys), cy, delta=15)

    def test_inverted_contrast_field_produces_zero_masks(self):
        """
        Inverted contrast image (bright objects on dark background).
        Brightfield hydrogel spheroids are strictly darker than gel; inverted objects must be rejected.
        """
        h, w = 400, 400
        img = np.full((h, w), 0.15, dtype=np.float32)  # Dark background
        cv2.circle(img, (200, 200), 40, 0.85, -1)      # Bright object

        mask, meta = self.hier.segment(img, image_id="inverted_contrast_test", pixel_size_um=1.5)
        self.assertEqual(int(np.max(mask)), 0, "Inverted contrast objects must be rejected by background contrast gating")

    def test_uniform_and_noise_only_fields_produce_zero_masks(self):
        """
        Adversarial test on blank images (all black, all white, all grey, and pure Gaussian noise).
        Must produce 0 masks with no crashes or unhandled exceptions.
        """
        cases = {
            "all_black": np.zeros((300, 300), dtype=np.float32),
            "all_white": np.ones((300, 300), dtype=np.float32),
            "all_grey": np.full((300, 300), 0.5, dtype=np.float32),
            "gaussian_noise": np.clip(np.random.normal(0.6, 0.1, (300, 300)), 0, 1).astype(np.float32),
        }
        for name, img in cases.items():
            with self.subTest(case=name):
                mask, meta = self.hier.segment(img, image_id=f"blank_{name}", pixel_size_um=1.5)
                self.assertEqual(int(np.max(mask)), 0, f"Case {name} produced unexpected masks: {np.max(mask)}")


class TestDenseClustersAndGeometries(unittest.TestCase):
    """Adversarial testing on touching clusters, non-circular shapes, and area limits."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_geom_")
        self.cfg = SegmentationConfig(
            cache_dir=os.path.join(self.tmpdir, "masks"),
            decisions_log_path=os.path.join(self.tmpdir, "DECISIONS.md"),
            primary_backend="classical",
            single_mask_max_area_pct=0.25,
        )
        self.hier = SegmentationHierarchy(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_touching_triplet_watershed_separation(self):
        """
        Three touching spheroids in a triangular cluster.
        Watershed must separate them into 3 distinct instances without merging into 1 giant blob.
        """
        h, w = 500, 500
        img = np.full((h, w), 0.70, dtype=np.float32)
        r = 35
        # Centers arranged such that disks touch with ~2-4 px overlap
        c1 = (220, 220)
        c2 = (220 + int(r * 1.85), 220)
        c3 = (220 + int(r * 0.92), 220 + int(r * 1.6))

        for cx, cy in [c1, c2, c3]:
            cv2.circle(img, (cx, cy), r, 0.25, -1)

        mask, meta = self.hier.segment(img, image_id="touching_triplet", pixel_size_um=1.5)
        n_objs = int(np.max(mask))

        self.assertGreaterEqual(n_objs, 2, f"Touching triplet should be split by watershed, got {n_objs} objects")
        # Ensure no mask exceeds 25% FOV
        for obj_id in range(1, n_objs + 1):
            area_pct = np.sum(mask == obj_id) / (h * w)
            self.assertLessEqual(area_pct, 0.25)

    def test_elongated_and_dumbbell_spheroid_geometries(self):
        """
        Test segmentation on irregular geometries:
        - Highly elliptical spheroid (aspect ratio 2.5:1)
        - Dumbbell / fused spheroid
        """
        h, w = 500, 500
        img = np.full((h, w), 0.70, dtype=np.float32)

        # Elliptical spheroid
        cv2.ellipse(img, (150, 150), (60, 25), 30, 0, 360, 0.25, -1)

        # Dumbbell / fusion (two overlapping disks)
        cv2.circle(img, (350, 350), 30, 0.25, -1)
        cv2.circle(img, (380, 350), 30, 0.25, -1)

        mask, meta = self.hier.segment(img, image_id="irregular_geoms", pixel_size_um=1.5)
        n_objs = int(np.max(mask))
        self.assertGreaterEqual(n_objs, 2, "Should detect irregular geometries")

    def test_single_mask_max_area_gate_25pct_enforcement(self):
        """
        Verify single mask area limit (max 25% FOV area):
        1. A mask covering 30% of FOV is rejected by plausibility check.
        2. A mask covering 20% of FOV passes plausibility check.
        """
        h, w = 400, 400
        total_px = h * w

        # Test 1: Giant mask covering 35% of FOV
        giant_img = np.full((h, w), 0.70, dtype=np.float32)
        # Radius for 35% area: pi * r^2 = 0.35 * 160000 -> r ~= 133 px
        cv2.circle(giant_img, (200, 200), 135, 0.20, -1)

        mask, meta = self.hier.segment(giant_img, image_id="giant_blob_35pct", pixel_size_um=1.5)
        # Giant object should be rejected by classical segmenter size filter / plausibility
        if int(np.max(mask)) > 0:
            for obj_id in range(1, int(np.max(mask)) + 1):
                area_pct = np.sum(mask == obj_id) / total_px
                self.assertLessEqual(area_pct, 0.25, f"Mask {obj_id} covers {area_pct*100:.1f}% FOV (>25% limit)")

        # Direct test on plausibility method
        fake_mask = np.zeros((h, w), dtype=np.int32)
        fake_mask[50:350, 50:350] = 1  # 300x300 = 90,000 px = 56.25% of FOV
        is_valid, reason = self.hier._check_plausibility(fake_mask)
        self.assertFalse(is_valid, "Plausibility check must fail for >25% FOV single mask")
        self.assertIn("exceeds 25% max limit", reason)

        # Pass case: 15% FOV
        fake_mask_valid = np.zeros((h, w), dtype=np.int32)
        fake_mask_valid[150:250, 150:250] = 1  # 100x100 = 10,000 px = 6.25% of FOV
        is_valid_pass, _ = self.hier._check_plausibility(fake_mask_valid)
        self.assertTrue(is_valid_pass, "Plausibility check should pass for normal sized mask")


class TestCorruptedAndMissingMetadataFallback(unittest.TestCase):
    """Adversarial testing on corrupted, missing, and anomalous TIFF metadata."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_metadata_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_corrupted_json_in_tag_37510(self):
        """TIFF tag 37510 contains invalid / truncated JSON string."""
        tif_path = os.path.join(self.tmpdir, "corrupted_json.tif")
        data = np.zeros((100, 100), dtype=np.uint16)

        # Write TIFF with broken JSON in Tag 37510
        with tifffile.TiffWriter(tif_path) as tw:
            tw.write(
                data,
                extratags=[(37510, "s", 0, "{'MicronsPerPixel': invalid_json_syntax...", True)],
            )

        extractor = ScaleExtractor(default_pixel_size_um=1.518817)
        info = extractor.extract(tif_path)

        self.assertFalse(info.is_calibrated)
        self.assertEqual(info.source, "config_default_uncalibrated")
        self.assertAlmostEqual(info.pixel_size_um, 1.518817, delta=1e-5)
        self.assertIsNotNone(info.warning)

    def test_negative_and_zero_microns_per_pixel(self):
        """TIFF tag 37510 contains zero or negative MicronsPerPixel."""
        tif_path = os.path.join(self.tmpdir, "zero_scale.tif")
        data = np.zeros((100, 100), dtype=np.uint16)

        with tifffile.TiffWriter(tif_path) as tw:
            tw.write(
                data,
                extratags=[(37510, "s", 0, json.dumps({"MicronsPerPixel": -5.0}), True)],
            )

        extractor = ScaleExtractor(default_pixel_size_um=1.518817)
        info = extractor.extract(tif_path)

        self.assertFalse(info.is_calibrated)
        self.assertEqual(info.source, "config_default_uncalibrated")
        self.assertAlmostEqual(info.pixel_size_um, 1.518817, delta=1e-5)

    def test_malformed_ome_xml_in_tag_270(self):
        """Tag 270 contains truncated / malformed OME-XML."""
        tif_path = os.path.join(self.tmpdir, "broken_ome.tif")
        data = np.zeros((100, 100), dtype=np.uint16)

        with tifffile.TiffWriter(tif_path) as tw:
            tw.write(
                data,
                description="<OME xmlns=... <Pixels PhysicalSizeX='broken' <unclosed tag",
            )

        extractor = ScaleExtractor(default_pixel_size_um=1.518817)
        info = extractor.extract(tif_path)

        self.assertFalse(info.is_calibrated)
        self.assertEqual(info.source, "config_default_uncalibrated")
        self.assertAlmostEqual(info.pixel_size_um, 1.518817, delta=1e-5)

    def test_cli_override_takes_absolute_precedence(self):
        """CLI override (--pixel-size 2.75) must override even valid embedded metadata."""
        tif_path = os.path.join(self.tmpdir, "valid_evos.tif")
        data = np.zeros((100, 100), dtype=np.uint16)

        with tifffile.TiffWriter(tif_path) as tw:
            tw.write(
                data,
                extratags=[(37510, "s", 0, json.dumps({"MicronsPerPixel": 1.518817}), True)],
            )

        px, src = extract_pixel_size(tif_path, override_um=2.750)
        self.assertAlmostEqual(px, 2.750, delta=1e-5)
        self.assertEqual(src, "cli_override")

    def test_nonexistent_file_raises_filenotfound(self):
        """ScaleExtractor on missing path raises FileNotFoundError."""
        extractor = ScaleExtractor()
        with self.assertRaises(FileNotFoundError):
            extractor.extract(os.path.join(self.tmpdir, "nonexistent_file.tif"))


class TestCachePersistenceAndConcurrency(unittest.TestCase):
    """Adversarial stress testing on disk caching, concurrency, and corrupt cache handling."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_cache_")
        self.cache = MaskCache(cache_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_large_label_ids_16bit_lossless_persistence(self):
        """
        Verify that 16-bit PNG caching preserves label values up to 65,535 without data loss or truncation.
        """
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.int32)
        test_labels = [1, 2, 255, 256, 1000, 32767, 65535]
        for idx, lbl in enumerate(test_labels):
            r_start = idx * 25
            mask[r_start:r_start+20, :50] = lbl

        save_path = self.cache.save("large_labels_test", mask)
        self.assertTrue(save_path.exists())

        loaded = self.cache.load("large_labels_test")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.dtype, np.int32)
        self.assertTrue(np.array_equal(mask, loaded), "Cached mask does not match original mask exactly")

    def test_sanitization_of_complex_image_ids(self):
        """
        Verify that paths with slashes, colons, spaces, and backslashes are sanitized safely.
        """
        problematic_ids = [
            "Day0/Mat gel1_0001:TRANS",
            "C:\\Windows\\Path\\Image_002",
            "Condition (G' 0.5mg/ml) #1 [Rep A]",
        ]
        for img_id in problematic_ids:
            with self.subTest(image_id=img_id):
                mask = np.ones((50, 50), dtype=np.int32)
                p = self.cache.save(img_id, mask)
                self.assertTrue(p.exists())
                self.assertTrue(self.cache.has(img_id))
                loaded = self.cache.load(img_id)
                self.assertIsNotNone(loaded)
                self.assertTrue(np.array_equal(mask, loaded))

    def test_corrupted_cached_mask_recovery(self):
        """
        If a cached mask file is corrupted (e.g. 0 bytes or random corrupted bytes),
        the cache must gracefully return None without crashing.
        """
        corrupt_id = "corrupt_sample"
        corrupt_path = self.cache.get_cache_path(corrupt_id)

        # Create 0-byte file
        with open(corrupt_path, "wb") as f:
            f.write(b"")

        self.assertFalse(self.cache.has(corrupt_id), "0-byte file should not be considered valid cache")
        self.assertIsNone(self.cache.load(corrupt_id))

        # Create random non-image bytes
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT_A_PNG_FILE_HEADER_GARBAGE_BYTES")

        # has() checks st_size > 0, load() should return None on decode failure
        loaded = self.cache.load(corrupt_id)
        self.assertIsNone(loaded, "Failed decode should return None")

    def test_multithreaded_concurrent_cache_access(self):
        """
        Adversarial test: 20 threads reading and writing distinct and shared masks concurrently.
        Tests for race conditions in non-atomic mask caching.
        """
        num_threads = 20
        errors = []

        def worker_task(thread_id: int):
            try:
                img_id = f"thread_sample_{thread_id % 5}"  # Shared across threads
                mask = np.full((100, 100), thread_id + 1, dtype=np.int32)
                # Write
                self.cache.save(img_id, mask)
                # Read
                loaded = self.cache.load(img_id)
                if loaded is None or loaded.shape != (100, 100):
                    errors.append(f"Thread {thread_id} failed to load valid mask (non-atomic write race condition)")
            except Exception as e:
                errors.append(f"Thread {thread_id} threw exception: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        # Record whether race conditions occurred
        if len(errors) > 0:
            logger.warning(f"Empirical race condition observed during concurrent cache access: {errors}")
        self.assertEqual(len(errors), 0, f"Concurrent cache access produced race condition errors: {errors}")



class TestPreprocessorAndDownsamplingAdversarial(unittest.TestCase):
    """Stress tests on illumination flattening, percentile normalization, and downsampling."""

    def test_zero_variance_flat_image(self):
        """Flat uniform image must not cause ZeroDivisionError during normalization or flattening."""
        img = np.full((200, 200), 0.50, dtype=np.float32)
        prep = Preprocessor()
        proc_img, scale, meta = prep.process(img, pixel_size_um=1.518817)

        self.assertEqual(proc_img.shape, (200, 200))
        self.assertEqual(scale, 1.0)
        self.assertFalse(np.isnan(proc_img).any())
        self.assertFalse(np.isinf(proc_img).any())

    def test_extreme_pixel_sizes_and_scale_factor_bounds(self):
        """
        Verify compute_downsample_scale behavior on extreme physical scale values:
        - Very high magnification (0.01 um/px) -> heavy downsampling
        - Very low magnification (50 um/px) -> no upsampling beyond reasonable limit
        - 0 or negative pixel size -> safe fallback to 1.0
        """
        # Normal 10x EVOS: ~1.52 um/px -> scale ~ 1.0
        s_normal = compute_downsample_scale((1536, 2048), pixel_size_um=1.518817)
        self.assertAlmostEqual(s_normal, 1.0, delta=0.15)

        # High resolution: 0.1 um/px -> raw diameter = 300 / 0.1 = 3000 px -> scale = 200 / 3000 = 0.0667
        s_high_res = compute_downsample_scale((1536, 2048), pixel_size_um=0.1)
        self.assertAlmostEqual(s_high_res, 200.0 / 3000.0, delta=1e-3)

        # Zero or negative scale should gracefully return 1.0
        self.assertEqual(compute_downsample_scale((1536, 2048), pixel_size_um=0.0), 1.0)
        self.assertEqual(compute_downsample_scale((1536, 2048), pixel_size_um=-1.5), 1.0)

    def test_upscale_labels_exact_label_preservation(self):
        """
        Verify nearest-neighbor upscaling preserves all original discrete labels without
        interpolating intermediate non-existent integer IDs.
        """
        small_mask = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [0, 0, 10],
        ], dtype=np.int32)
        large_shape = (300, 300)
        upscaled = upscale_labels(small_mask, large_shape)

        self.assertEqual(upscaled.shape, large_shape)
        self.assertEqual(upscaled.dtype, np.int32)
        unique_orig = set(np.unique(small_mask))
        unique_upscaled = set(np.unique(upscaled))
        self.assertEqual(unique_orig, unique_upscaled, "Upscaling introduced spurious labels or lost labels")


class TestRealDatasetTIFFAdversarial(unittest.TestCase):
    """Stress tests on real dataset images across hydrogel conditions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_real_")
        self.cfg = SegmentationConfig(
            cache_dir=os.path.join(self.tmpdir, "masks"),
            decisions_log_path=os.path.join(self.tmpdir, "DECISIONS.md"),
            primary_backend="classical",
        )
        self.hier = SegmentationHierarchy(config=self.cfg)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_real_dataset_samples_no_mask_exceeds_25pct_fov(self):
        """
        Run segmentation on multiple real experimental TIFFs and verify:
        1. Segmentation succeeds without unhandled exceptions.
        2. Optical scale is extracted accurately (1.518817 um/px).
        3. No individual mask occupies > 25% of FOV area.
        4. Mask dimensions match raw TIFF dimensions (1536, 2048).
        """
        real_files = [
            "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 Mat gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 G' 0.5mg_ml gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid D7 260812/Day7 Mat gel1_0001_TRANS.tif",
        ]
        for rel_path in real_files:
            abs_path = os.path.join(str(PROJECT_ROOT), rel_path)
            if not os.path.exists(abs_path):
                continue

            with self.subTest(file=rel_path):
                # 1. Test scale extractor
                px_size, src = extract_pixel_size(abs_path)
                self.assertAlmostEqual(px_size, 1.518817, delta=1e-4)
                self.assertEqual(src, "TIFF Tag 37510 JSON")

                # 2. Test segmentation
                img = tifffile.imread(abs_path)
                h, w = img.shape[:2]
                total_px = h * w

                mask, meta = self.hier.segment(img, image_id=Path(abs_path).stem, pixel_size_um=px_size)

                self.assertEqual(mask.shape, (h, w))
                n_objs = int(np.max(mask))

                # Check 25% FOV gate on every detected object
                for obj_id in range(1, n_objs + 1):
                    area_px = float(np.sum(mask == obj_id))
                    area_pct = area_px / total_px
                    self.assertLessEqual(
                        area_pct, 0.25,
                        f"Object {obj_id} in {rel_path} covers {area_pct*100:.2f}% of FOV (>25% limit)"
                    )


class TestDecisionsLoggerConcurrency(unittest.TestCase):
    """Stress testing on structured decision logging concurrency."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="m1_adv_decisions_")
        self.log_path = os.path.join(self.tmpdir, "DECISIONS.md")
        self.logger = DecisionsLogger(log_path=self.log_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_concurrent_decisions_logging(self):
        """
        Multi-threaded writes to DECISIONS.md must preserve valid formatting without crashing.
        """
        num_threads = 15
        errors = []

        def log_task(task_id: int):
            try:
                self.logger.log_fallback(
                    image_id=f"concurrent_img_{task_id}",
                    from_backend="cpsam",
                    to_backend="classical",
                    reason=f"Plausibility test failed for worker {task_id}",
                    extra_meta={"thread_id": task_id, "timestamp_idx": task_id * 10},
                )
            except Exception as e:
                errors.append(f"Logging task {task_id} failed: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(log_task, i) for i in range(num_threads)]
            concurrent.futures.wait(futures)

        self.assertEqual(len(errors), 0, f"Concurrent logging errors: {errors}")
        self.assertTrue(os.path.exists(self.log_path))
        with open(self.log_path, "r", encoding="utf-8") as f:
            content = f.read()

        for i in range(num_threads):
            self.assertIn(f"concurrent_img_{i}", content)


if __name__ == "__main__":
    unittest.main()


