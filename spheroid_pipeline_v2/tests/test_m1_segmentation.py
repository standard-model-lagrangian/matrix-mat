"""
Unit and Integration Tests for Milestone 1 (M1: Foundation, Segmentation Hierarchy & Mask Caching).
Covers:
- Configuration dataclasses and YAML serialization/deserialization
- Optical scale extraction from TIFF Tag 37510 JSON, OME-XML, and CLI overrides
- Image preprocessing: flat-field illumination correction, percentile normalization, and proportional downsampling
- Lossless 16-bit PNG mask caching
- Structured DECISIONS.md logging
- 3-tier segmentation hierarchy with MPS/CPU fallback and plausibility gating
- Synthetic image verification (isolated disk <=10% err, touching pair watershed split, bright blob rejected)
- Real dataset TIFF segmentation
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tifffile

from spheroid_pipeline_v2.config import (
    ArtifactConfig,
    PairingConfig,
    PipelineConfig,
    PreprocessConfig,
    QCConfig,
    SegmentationConfig,
    StatsConfig,
)
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
from spheroid_pipeline_v2.scale_extractor import (
    ScaleExtractor,
    ScaleInfo,
    extract_pixel_size,
)
from spheroid_pipeline_v2.segment import (
    ClassicalMultiScaleSegmenter,
    SegmentationHierarchy,
    get_optimal_device,
)


class TestConfig(unittest.TestCase):
    """Test configuration dataclasses, YAML loading/saving, and CLI override merging."""

    def test_default_config_instantiation(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.output_dir, "output")
        self.assertAlmostEqual(cfg.default_pixel_size_um, 1.518817, places=5)
        self.assertEqual(cfg.preprocess.flat_field_sigma, 80.0)
        self.assertEqual(cfg.segmentation.primary_backend, "cpsam")
        self.assertEqual(cfg.segmentation.fallback_backend, "cyto3")
        self.assertEqual(cfg.segmentation.last_resort_backend, "classical")
        self.assertEqual(cfg.qc.min_d_um, 40.0)
        self.assertEqual(cfg.qc.max_d_um, 1500.0)
        self.assertEqual(cfg.qc.interior_contrast_pct, 0.10)
        self.assertEqual(cfg.pairing.min_t0_d_um_denominator, 60.0)
        self.assertEqual(cfg.stats.control_condition, "Mat")

    def test_yaml_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            cfg = PipelineConfig(
                output_dir="custom_output",
                default_pixel_size_um=2.0,
            )
            cfg.to_yaml(yaml_path)
            self.assertTrue(yaml_path.exists())

            loaded = PipelineConfig.from_yaml(yaml_path)
            self.assertEqual(loaded.output_dir, "custom_output")
            self.assertAlmostEqual(loaded.default_pixel_size_um, 2.0)
            self.assertEqual(loaded.preprocess.flat_field_sigma, 80.0)

    def test_cli_overrides(self):
        overrides = {
            "output_dir": "cli_output",
            "cli_pixel_size_um": 3.1415,
            "min_d_um": 55.0,
        }
        cfg = PipelineConfig.from_yaml("non_existent.yaml", cli_overrides=overrides)
        self.assertEqual(cfg.output_dir, "cli_output")
        self.assertAlmostEqual(cfg.cli_pixel_size_um, 3.1415)
        self.assertEqual(cfg.qc.min_d_um, 55.0)


class TestScaleExtractor(unittest.TestCase):
    """Test optical scale extraction across TIFF Tag 37510, CLI override, and fallbacks."""

    def setUp(self):
        self.real_tiff_path = Path("Experimental data /Chuling cells/Spheroid Day0 260805/Day0 Mat gel1_0001_TRANS.tif")

    def test_real_tiff_scale_extraction(self):
        if not self.real_tiff_path.exists():
            self.skipTest(f"Dataset TIFF not found at {self.real_tiff_path}")

        px_size, source = extract_pixel_size(self.real_tiff_path)
        self.assertAlmostEqual(px_size, 1.518817, places=4)
        self.assertIn("Tag 37510", source)

    def test_cli_override(self):
        if not self.real_tiff_path.exists():
            self.skipTest(f"Dataset TIFF not found at {self.real_tiff_path}")

        px_size, source = extract_pixel_size(self.real_tiff_path, override_um=2.718)
        self.assertAlmostEqual(px_size, 2.718)
        self.assertEqual(source, "cli_override")

    def test_uncalibrated_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_tiff = Path(tmpdir) / "dummy.tif"
            # Create a simple uncalibrated TIFF
            arr = np.zeros((100, 100), dtype=np.uint8)
            tifffile.imwrite(dummy_tiff, arr)

            extractor = ScaleExtractor(default_pixel_size_um=1.234)
            info = extractor.extract(dummy_tiff)
            self.assertAlmostEqual(info.pixel_size_um, 1.234)
            self.assertFalse(info.is_calibrated)
            self.assertIsNotNone(info.warning)


class TestPreprocessor(unittest.TestCase):
    """Test flat-field illumination flattening, percentile normalization, and downsampling."""

    def test_flatten_illumination(self):
        # Create an image with an intense linear illumination gradient
        h, w = 256, 256
        grad = np.linspace(50, 200, w, dtype=np.float32)
        img = np.tile(grad, (h, 1))

        flattened = flatten_illumination(img, sigma=40.0)
        # Background gradient should be significantly smoothed/flattened
        col_std_before = np.std(np.mean(img, axis=0))
        col_std_after = np.std(np.mean(flattened, axis=0))
        self.assertLess(col_std_after, col_std_before)

    def test_normalize_percentiles(self):
        img = np.array([[10.0, 20.0], [80.0, 90.0]], dtype=np.float32)
        norm, p_low, p_high = normalize_percentiles(img, p_low=0.0, p_high=100.0)
        self.assertAlmostEqual(norm.min(), 0.0)
        self.assertAlmostEqual(norm.max(), 1.0)
        self.assertEqual(norm.dtype, np.float32)

    def test_compute_downsample_scale(self):
        # If expected raw px is 200 px, s should snap to 1.0
        s1 = compute_downsample_scale((1000, 1000), pixel_size_um=1.5, expected_diameter_um=300.0, target_diameter_px=200.0)
        self.assertEqual(s1, 1.0)

        # If expected raw px is 600 px (at 0.5 um/px), s = 200 / 600 = 0.333
        s2 = compute_downsample_scale((1000, 1000), pixel_size_um=0.5, expected_diameter_um=300.0, target_diameter_px=200.0)
        self.assertAlmostEqual(s2, 0.333333, places=3)

    def test_downsample_and_upscale_labels(self):
        labels_small = np.zeros((100, 100), dtype=np.int32)
        labels_small[20:40, 20:40] = 1
        labels_small[60:80, 60:80] = 2

        upscaled = upscale_labels(labels_small, (200, 200))
        self.assertEqual(upscaled.shape, (200, 200))
        self.assertEqual(int(np.max(upscaled)), 2)
        self.assertTrue(np.all(np.isin(np.unique(upscaled), [0, 1, 2])))


class TestMaskCache(unittest.TestCase):
    """Test 16-bit PNG disk caching for segmentation masks."""

    def test_mask_cache_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = MaskCache(cache_dir=tmpdir)
            image_id = "Day0_Mat_gel1_0001_TRANS"

            self.assertFalse(cache.has(image_id))
            self.assertIsNone(cache.load(image_id))

            # Create multi-instance mask with large label IDs (up to 5000)
            mask = np.zeros((1536, 2048), dtype=np.int32)
            mask[100:200, 100:200] = 1
            mask[300:400, 300:400] = 5000

            saved_path = cache.save(image_id, mask)
            self.assertTrue(saved_path.exists())
            self.assertTrue(cache.has(image_id))

            loaded = cache.load(image_id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.dtype, np.int32)
            self.assertEqual(loaded.shape, (1536, 2048))
            self.assertTrue(np.array_equal(mask, loaded))

            deleted = cache.clear()
            self.assertEqual(deleted, 1)
            self.assertFalse(cache.has(image_id))


class TestDecisionsLogger(unittest.TestCase):
    """Test structured logging to DECISIONS.md."""

    def test_log_creation_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "DECISIONS.md"
            logger_inst = DecisionsLogger(log_path=log_file)
            self.assertTrue(log_file.exists())

            logger_inst.log_fallback(
                image_id="Test_Image_01",
                from_backend="cpsam",
                to_backend="cyto3",
                reason="Simulated deep learning failure",
            )

            content = log_file.read_text(encoding="utf-8")
            self.assertIn("Test_Image_01", content)
            self.assertIn("cpsam", content)
            self.assertIn("cyto3", content)
            self.assertIn("Simulated deep learning failure", content)


class TestSegmentation(unittest.TestCase):
    """Test classical segmentation, 3-tier hierarchy, and synthetic scenarios."""

    def test_optimal_device(self):
        use_gpu, dev_str = get_optimal_device()
        self.assertIsInstance(use_gpu, bool)
        self.assertIn(dev_str, ["mps", "cuda", "cpu"])

    def test_synthetic_scenario_1_single_disk(self):
        """Synthetic Scenario 1: Dark disk of known radius r=50 (d=100px) on uneven gradient."""
        h, w = 512, 512
        bg = np.tile(np.linspace(180, 230, w, dtype=np.float32), (h, 1))
        rr, cc = np.ogrid[:h, :w]
        disk_mask = (rr - 256) ** 2 + (cc - 256) ** 2 <= 50 ** 2
        bg[disk_mask] = 50.0
        img_u8 = cv2.GaussianBlur(bg, (9, 9), 2.0).astype(np.uint8)

        flat = flatten_illumination(img_u8, sigma=40.0)
        norm_img, _, _ = normalize_percentiles(flat)

        segmenter = ClassicalMultiScaleSegmenter()
        labels, meta = segmenter.segment(norm_img, pixel_size_um=1.0)

        n_objs = int(np.max(labels))
        self.assertEqual(n_objs, 1, f"Expected 1 object, found {n_objs}")

        area = float(np.sum(labels == 1))
        recovered_d = 2.0 * np.sqrt(area / np.pi)
        error_pct = abs(recovered_d - 100.0) / 100.0 * 100.0
        self.assertLessEqual(error_pct, 10.0, f"Recovered diameter {recovered_d:.2f}px error {error_pct:.2f}% exceeds 10% limit")

    def test_synthetic_scenario_2_touching_pair(self):
        """Synthetic Scenario 2: Two touching spheres (r=40px) separated by watershed."""
        h, w = 512, 512
        bg = np.tile(np.linspace(180, 230, w, dtype=np.float32), (h, 1))
        rr, cc = np.ogrid[:h, :w]
        # Centers at (256, 221) and (256, 291) -> distance 70 px, touching because r1+r2=80 px
        m1 = (rr - 256) ** 2 + (cc - 221) ** 2 <= 40 ** 2
        m2 = (rr - 256) ** 2 + (cc - 291) ** 2 <= 40 ** 2
        bg[m1 | m2] = 50.0
        img_u8 = cv2.GaussianBlur(bg, (9, 9), 2.0).astype(np.uint8)

        flat = flatten_illumination(img_u8, sigma=40.0)
        norm_img, _, _ = normalize_percentiles(flat)

        segmenter = ClassicalMultiScaleSegmenter()
        labels, meta = segmenter.segment(norm_img, pixel_size_um=1.0)

        n_objs = int(np.max(labels))
        self.assertEqual(n_objs, 2, f"Expected watershed to separate 2 touching spheroids, got {n_objs}")

    def test_synthetic_scenario_3_bright_gel_blob_rejected(self):
        """Synthetic Scenario 3: Bright gel blob must NOT be segmented (0 masks)."""
        h, w = 512, 512
        bg = np.tile(np.linspace(180, 230, w, dtype=np.float32), (h, 1))
        rr, cc = np.ogrid[:h, :w]
        # Bright blob at center
        blob_mask = (rr - 256) ** 2 + (cc - 256) ** 2 <= 50 ** 2
        bg[blob_mask] = 250.0
        img_u8 = cv2.GaussianBlur(bg, (9, 9), 2.0).astype(np.uint8)

        flat = flatten_illumination(img_u8, sigma=40.0)
        norm_img, _, _ = normalize_percentiles(flat)

        segmenter = ClassicalMultiScaleSegmenter()
        labels, meta = segmenter.segment(norm_img, pixel_size_um=1.0)

        n_objs = int(np.max(labels))
        self.assertEqual(n_objs, 0, f"Bright gel blob must be rejected, but got {n_objs} masks")



    def test_segmentation_hierarchy_end_to_end(self):
        """Test complete 3-tier hierarchy, caching, and fallback on real image."""
        real_tiff = Path("Experimental data /Chuling cells/Spheroid Day0 260805/Day0 Mat gel1_0001_TRANS.tif")
        if not real_tiff.exists():
            self.skipTest(f"Dataset TIFF not found at {real_tiff}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cfg = SegmentationConfig(
                cache_dir=str(tmp_path / "masks"),
                decisions_log_path=str(tmp_path / "DECISIONS.md"),
            )
            hierarchy = SegmentationHierarchy(config=cfg)

            img = tifffile.imread(real_tiff)
            image_id = "test_hierarchy_img"

            # Pass 1: Compute (should fallback to classical and save to disk)
            mask1, meta1 = hierarchy.segment(img, image_id=image_id, pixel_size_um=1.518817)
            self.assertFalse(meta1.get("cached", False))
            self.assertGreater(meta1.get("objects_found", 0), 0)
            self.assertEqual(mask1.shape, img.shape)
            self.assertTrue(Path(cfg.cache_dir, f"{image_id}.png").exists())

            # Pass 2: Cache Hit
            mask2, meta2 = hierarchy.segment(img, image_id=image_id, pixel_size_um=1.518817)
            self.assertTrue(meta2.get("cached", False))
            self.assertEqual(meta2.get("backend"), "disk_cache")
            self.assertTrue(np.array_equal(mask1, mask2))

            # Pass 3: Force Recompute
            mask3, meta3 = hierarchy.segment(img, image_id=image_id, pixel_size_um=1.518817, force_recompute=True)
            self.assertFalse(meta3.get("cached", False))
            self.assertEqual(meta3.get("final_backend"), "classical")
            self.assertTrue(np.array_equal(mask1, mask3))

    def test_plausibility_max_area_rejection(self):
        """Test that single mask covering >25% FOV is rejected by plausibility check."""
        cfg = SegmentationConfig(single_mask_max_area_pct=0.25)
        hierarchy = SegmentationHierarchy(config=cfg)

        # 512x512 image, total pixels = 262,144. 30% area = 78,643 px
        mask = np.zeros((512, 512), dtype=np.int32)
        mask[:300, :300] = 1  # 90,000 px = 34.3% FOV
        is_valid, reason = hierarchy._check_plausibility(mask)
        self.assertFalse(is_valid)
        self.assertIn("exceeds 25% max limit", reason)

    def test_extract_ome_xml_and_imagej_metadata(self):
        """Test ScaleExtractor on mock OME-XML and ImageJ metadata tags."""
        extractor = ScaleExtractor()

        # Mock OME-XML Tag 270
        ome_xml_sample = (
            '<?xml version="1.0"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image ID="Image:0">'
            '<Pixels DimensionOrder="XYZCT" ID="Pixels:0" PhysicalSizeX="2.345" PhysicalSizeXUnit="µm" '
            'PhysicalSizeY="2.345" PhysicalSizeYUnit="µm" SizeC="1" SizeT="1" SizeX="2048" SizeY="1536" SizeZ="1" Type="uint8">'
            '</Pixels>'
            '</Image>'
            '</OME>'
        )
        ome_tags = {270: ome_xml_sample}
        ome_info = extractor._extract_ome_xml(ome_tags, (2048, 1536))
        self.assertIsNotNone(ome_info)
        self.assertAlmostEqual(ome_info.pixel_size_um, 2.345)
        self.assertEqual(ome_info.source, "OME-XML Tag 270")

        # Mock ImageJ metadata Tag 270
        imagej_desc = "ImageJ=1.53c\nunit=um\nspacing=1.654"
        imagej_tags = {270: imagej_desc}
        imagej_info = extractor._extract_imagej(imagej_tags, {})
        self.assertIsNotNone(imagej_info)
        self.assertAlmostEqual(imagej_info.pixel_size_um, 1.654)
        self.assertEqual(imagej_info.source, "ImageJ metadata")

    def test_package_exports(self):
        """Test all top-level symbols exported in spheroid_pipeline_v2."""
        import spheroid_pipeline_v2 as sp2
        expected_exports = [
            "PipelineConfig", "PreprocessConfig", "SegmentationConfig", "QCConfig",
            "PairingConfig", "StatsConfig", "ArtifactConfig", "ScaleExtractor",
            "ScaleInfo", "extract_pixel_size", "Preprocessor", "flatten_illumination",
            "normalize_percentiles", "compute_downsample_scale", "downsample_image",
            "upscale_labels", "ClassicalMultiScaleSegmenter", "CellposeSAMSegmenter",
            "CellposeCyto3Segmenter", "SegmentationHierarchy", "MaskCache", "DecisionsLogger",
        ]
        for name in expected_exports:
            self.assertTrue(hasattr(sp2, name), f"Missing export: {name}")

    def test_multi_condition_real_images(self):
        """Verify classical segmentation succeeds on 5 diverse condition images."""
        sample_paths = [
            "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 Mat gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 S34D30 gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid Day0 260805/Day0 S40D30 gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid D7 260812/Day7 S46D10 gel1_0001_TRANS.tif",
            "Experimental data /Chuling cells/Spheroid D7 260812/Day7 S50 gel1_0001_TRANS.tif",
        ]

        prep = Preprocessor()
        segmenter = ClassicalMultiScaleSegmenter()

        for p_str in sample_paths:
            p = Path(p_str)
            if not p.exists():
                continue
            img = tifffile.imread(p)
            proc_img, _, _ = prep.process(img, pixel_size_um=1.518817)
            labels, meta = segmenter.segment(proc_img, pixel_size_um=1.518817)
            n_objs = meta.get("objects_found", 0)
            self.assertGreater(n_objs, 0, f"Segmentation yielded 0 objects on {p.name}")
            # Ensure no single mask exceeds 25% of FOV
            h, w = labels.shape[:2]
            for i in range(1, n_objs + 1):
                area = np.sum(labels == i)
                self.assertLessEqual(area, 0.25 * h * w, f"Object {i} in {p.name} exceeded 25% FOV")


if __name__ == "__main__":
    unittest.main()

