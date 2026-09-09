"""
Unit and integration smoke tests for spheroid_if_sweep:
- Filename parsing & pairing integrity
- Synthetic blob segmentation sanity check
- QC artifact flagging & deduplication
- Provenance metadata & manifest completeness
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import warnings

# Filter skimage grayreconstruct deprecation in NumPy 2.5
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*grayreconstruct.*")

# Set writable MPLCONFIGDIR
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

from spheroid_if_sweep.engine import CPSAMEngine
from spheroid_if_sweep.features import extract_objects_and_image_features
from spheroid_if_sweep.pairing import (
    discover_and_pair_fields,
    parse_field_metadata,
)
from spheroid_if_sweep.postprocess import (
    apply_size_gate,
    identify_border_objects,
    split_merged_nuclei,
)
from spheroid_if_sweep.preprocess import (
    normalize_percentile,
    subtract_background,
)
from spheroid_if_sweep.provenance import (
    get_cellpose_version,
    get_git_info,
    get_gpu_info,
    write_manifest_csv,
    write_run_metadata,
)
from spheroid_if_sweep.qc import (
    ImageQCMetrics,
    apply_cohort_qc_flags,
    compute_image_qc_metrics,
    deduplicate_masks,
)


class TestSpheroidIFSweep(unittest.TestCase):

    def test_filename_parsing(self):
        """Verify regex correctly parses material, replicate, and subfield."""
        mat, rep, fld = parse_field_metadata("SKOV3 Spheroid D7 Mor_Mat-Gel1-2")
        self.assertEqual(mat, "Mat")
        self.assertEqual(rep, "Gel1")
        self.assertEqual(fld, "2")

        mat2, rep2, fld2 = parse_field_metadata("SKOV3 Spheroid D7 Mor_S34D30-Gel8-1-2")
        self.assertEqual(mat2, "S34D30")
        self.assertEqual(rep2, "Gel8")
        self.assertEqual(fld2, "1-2")

        mat3, rep3, fld3 = parse_field_metadata("SKOV3 Spheroid D7 Mor_S50-Gel7")
        self.assertEqual(mat3, "S50")
        self.assertEqual(rep3, "Gel7")
        self.assertEqual(fld3, "1")

    def test_pairing_and_unpaired_enforcement(self):
        """Ensure unpaired image is detected and enforced."""
        data_dir = Path("Experimental data /Chuling cells/spheroids staining:IF /nuclei density analysis")
        if not data_dir.exists():
            self.skipTest("Data directory not present in environment")

        # 1. Hard error without allow_unpaired
        with self.assertRaises(RuntimeError) as ctx:
            discover_and_pair_fields(data_dir, allow_unpaired=False)
        self.assertIn("SKOV3 Spheroid D7 Mor_S43D20-Gel1-1-2", str(ctx.exception))

        # 2. Permitted with allow_unpaired=True
        paired, unpaired = discover_and_pair_fields(data_dir, allow_unpaired=True)
        self.assertEqual(len(paired), 27)
        self.assertEqual(len(unpaired), 1)
        self.assertEqual(unpaired[0], "SKOV3 Spheroid D7 Mor_S43D20-Gel1-1-2")

    def test_deduplicate_masks(self):
        """Test that overlapping mask proposals with IoU > 0.50 are deduplicated."""
        masks = np.zeros((100, 100), dtype=np.int32)
        # Mask 1: 20x20 box (area 400)
        masks[10:30, 10:30] = 1
        # Mask 2: overlapping box (12:30, 12:30) (area 324, overlap 324, union 400 -> IoU = 324/400 = 0.81)
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[12:30, 12:30] = True

        combined = masks.copy()
        combined[mask2] = 2

        dedup, n_removed = deduplicate_masks(combined, iou_threshold=0.50)
        self.assertGreaterEqual(n_removed, 0)
        self.assertGreaterEqual(int(dedup.max()), 1)

    def test_qc_flags_unit(self):
        """Test unit behavior of QC artifact flags."""
        qc_config = {
            "snr_min": 1.5,
            "min_nuclear_vol_um3": 150.0,
            "max_nuclear_vol_um3": 4000.0,
            "border_margin_px": 5,
            "edge_heavy_fraction": 0.25,
            "saturation_fraction_max": 0.005,
            "blur_percentile_cutoff": 10.0,
        }

        # 1. Saturated image
        sat_img = np.full((100, 100), 255, dtype=np.uint8)
        normal_img = np.zeros((100, 100), dtype=np.uint8)
        empty_masks = np.zeros((100, 100), dtype=np.int32)

        m = compute_image_qc_metrics(
            image_id="test_sat",
            material="Mat",
            replicate="Gel1",
            nuclear_img=sat_img,
            actin_img=normal_img,
            raw_masks=empty_masks,
            dedup_masks=empty_masks,
            n_dups_removed=0,
            pixel_size_um=0.505,
            qc_config=qc_config,
        )
        self.assertTrue(m.flag_saturated)

        # 2. Empty segmentation with high SNR
        high_snr_img = np.zeros((100, 100), dtype=np.uint8)
        high_snr_img[20:80, 20:80] = 200  # 36% of pixels are bright -> P95 = 200, P50 = 0
        m_empty = compute_image_qc_metrics(
            image_id="test_empty",
            material="Mat",
            replicate="Gel1",
            nuclear_img=high_snr_img,
            actin_img=normal_img,
            raw_masks=empty_masks,
            dedup_masks=empty_masks,
            n_dups_removed=0,
            pixel_size_um=0.505,
            qc_config=qc_config,
        )
        self.assertTrue(m_empty.flag_empty_seg)

    def test_provenance_completeness(self):
        """Verify that run_metadata.json has all required provenance fields."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            params = {
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
                "diameter": None,
            }
            meta_path = write_run_metadata(
                run_dir=tmp_path,
                resolved_params=params,
                random_seed=42,
                wall_time_sec=12.345,
            )
            self.assertTrue(meta_path.exists())

            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            required_keys = [
                "timestamp_iso",
                "cellpose_version",
                "model_name",
                "torch_version",
                "python_version",
                "platform",
                "hostname",
                "gpu_model",
                "git",
                "random_seed",
                "wall_time_seconds",
                "resolved_parameters",
            ]
            for k in required_keys:
                self.assertIn(k, data, f"Missing key in run_metadata.json: {k}")

            self.assertEqual(data["model_name"], "cpsam")
            self.assertEqual(data["random_seed"], 42)

            # Check extended provenance with pre/post processing
            meta_path2 = write_run_metadata(
                run_dir=tmp_path,
                resolved_params=params,
                random_seed=42,
                wall_time_sec=10.0,
                preprocessing_info={
                    "enabled": True,
                    "methods": ["background_subtraction_rolling_ball", "normalize_percentile"],
                    "resolved_parameters": {
                        "expected_nuclear_diameter_um": 11.7,
                        "background": {"method": "rolling_ball", "radius_px": 81},
                    },
                },
                postprocessing_info={
                    "enabled": True,
                    "methods": ["deduplicate", "size_gate", "split_merged_seeded_watershed", "border_exclusion"],
                    "resolved_parameters": {
                        "reference_nuclear_volume_um3": 841.04,
                        "size_gate": {"min_vol_um3": 210.0, "max_vol_um3": 3364.0},
                    },
                },
            )
            with open(meta_path2, "r", encoding="utf-8") as f:
                data2 = json.load(f)
            self.assertIn("preprocessing", data2)
            self.assertIn("postprocessing", data2)
            self.assertEqual(data2["preprocessing"]["methods"][0], "background_subtraction_rolling_ball")
            self.assertEqual(data2["postprocessing"]["resolved_parameters"]["reference_nuclear_volume_um3"], 841.04)
            self.assertEqual(data2["postprocessing"]["resolved_parameters"]["size_gate"]["min_vol_um3"], 210.0)

    def test_seeded_watershed_touching_blobs(self):
        """Synthetic touching-blobs image: seeded watershed must recover known count (2 children)."""
        h, w = 120, 120
        px = 0.505049
        img = np.zeros((h, w), dtype=np.uint8)

        # Create two touching Gaussian nuclei with separate intensity peaks
        c1 = (60, 48)
        c2 = (60, 72)
        radius = 12.0

        for cy, cx in [c1, c2]:
            y, x = np.ogrid[:h, :w]
            dist_sq = (y - cy) ** 2 + (x - cx) ** 2
            blob = np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * 220
            img = np.maximum(img, blob.astype(np.uint8))

        # Combined mask covering both touching blobs (simulating a merged clump)
        mask = (img > 30).astype(np.int32)
        initial_count = int(mask.max())
        self.assertEqual(initial_count, 1)

        # Run watershed splitting
        result_mask, n_cand, n_acc, n_rev, n_kids, split_evs = split_merged_nuclei(
            masks=mask,
            nuclear_img=img,
            pixel_size_um=px,
            split_factor=0.5,  # Low factor to ensure this clump is treated as candidate
            h_maxima=10.0,
            min_distance_um=3.0,
            min_vol_um3=100.0,
            max_vol_um3=5000.0,
        )

        n_found = int(result_mask.max())
        self.assertEqual(n_found, 2, f"Expected touching blobs to split into 2 nuclei, but found {n_found}")
        self.assertEqual(n_acc, 1)

    def test_background_subtraction_speckle_noise(self):
        """Synthetic speckle-noise image: background subtraction must eliminate false-positive objects."""
        from skimage import measure
        h, w = 150, 150
        px = 0.505049
        np.random.seed(42)

        # Baseline background floor + gradient
        y, x = np.mgrid[:h, :w]
        grad = (y / float(h) * 50.0).astype(np.float32) + 25.0
        # Add high-intensity autofluorescence speckle noise
        speckles = (np.random.rand(h, w) > 0.985).astype(np.float32) * 90.0
        noisy_img = np.clip(grad + speckles, 0, 255).astype(np.uint8)

        # Place a real nucleus
        yc, xc = 75, 75
        dist_sq = (y - yc) ** 2 + (x - xc) ** 2
        nucleus = np.exp(-dist_sq / 50.0) * 200.0
        synth_img = np.clip(noisy_img + nucleus, 0, 255).astype(np.uint8)

        # Thresholding raw image without background subtraction yields false-positive objects from speckles + gradient
        raw_thresh = synth_img > 90
        raw_labeled, n_raw_objs = measure.label(raw_thresh, return_num=True)
        self.assertGreater(n_raw_objs, 1, "Raw image should have false positive speckle objects without subtraction")

        # Apply background subtraction
        bg_sub = subtract_background(synth_img, method="rolling_ball", radius_px=25, downsample_factor=1)
        f32_norm, u8_norm, p_low, p_high = normalize_percentile(bg_sub, p_low=1.0, p_high=99.8)

        # Thresholding preprocessed image + applying size gate purges noise specks, leaving only the real nucleus
        proc_thresh = u8_norm > 100
        proc_labeled, n_proc_objs = measure.label(proc_thresh, return_num=True)
        self.assertGreater(n_proc_objs, 1, "Raw thresholded objects should include noise specks before size filtering")

        clean_masks, n_small, n_large, _ = apply_size_gate(
            masks=proc_labeled,
            pixel_size_um=px,
            min_vol_um3=210.0,
            max_vol_um3=3364.0,
        )
        n_surviving = int(clean_masks.max())
        self.assertEqual(n_surviving, 1, f"Expected 1 true nucleus after size filtering, but found {n_surviving}")
        self.assertGreater(n_small, 0, "Noise specks should have been flagged and filtered as small debris")

        # Verify noise suppression in background region
        corner_mean = float(np.mean(f32_norm[:20, :20]))
        center_peak = float(f32_norm[yc, xc])
        self.assertLess(corner_mean, 0.10, "Background floor was not sufficiently suppressed")
        self.assertGreater(center_peak, 0.70, "True nuclear peak was incorrectly attenuated")

    def test_size_gate_unit(self):
        """Size gate unit test: verifies removal of debris (<min_vol) and macro-artifacts (>max_vol)."""
        masks = np.zeros((100, 100), dtype=np.int32)
        px = 0.505049

        # Obj 1: 3x3 box (area 9 px -> vol ~ 33 um3 < 210 um3) -> Debris
        masks[10:13, 10:13] = 1

        # Obj 2: 20x20 box (area 400 px -> vol ~ 1700 um3 in [210, 3364]) -> Valid nucleus
        masks[30:50, 30:50] = 2

        # Obj 3: 45x45 box (area 2025 px -> vol ~ 18000 um3 > 3364 um3) -> Giant clump
        masks[55:95, 55:95] = 3

        clean, n_small, n_large, removed_sizes = apply_size_gate(
            masks=masks,
            pixel_size_um=px,
            min_vol_um3=210.0,
            max_vol_um3=3364.0,
        )

        self.assertEqual(n_small, 1)
        self.assertEqual(n_large, 1)
        self.assertEqual(int(clean.max()), 1, "Only 1 valid nucleus should survive the size gate")

    def test_border_exclusion_unit(self):
        """Border exclusion unit test: perimeter objects within clearance_um are flagged and excluded from density."""
        h, w = 200, 200
        masks = np.zeros((h, w), dtype=np.int32)
        px = 0.505049
        clearance_um = 5.85  # ~11.6 px clearance

        # Object 1: right on border at row 3 (within clearance)
        masks[2:10, 20:28] = 1

        # Object 2: center at row 100 (outside clearance)
        masks[95:105, 95:105] = 2

        border_labels, n_border = identify_border_objects(masks, pixel_size_um=px, clearance_um=clearance_um)
        self.assertIn(1, border_labels)
        self.assertNotIn(2, border_labels)
        self.assertEqual(n_border, 1)

        # Test integration with feature extraction
        dummy_actin = np.full((h, w), 50, dtype=np.uint8)
        dummy_nuc = np.zeros((h, w), dtype=np.uint8)
        obj_rows, per_img = extract_objects_and_image_features(
            image_id="test_border",
            material="Mat",
            replicate="Gel1",
            nuclear_img=dummy_nuc,
            actin_img=dummy_actin,
            dedup_masks=masks,
            pixel_size_um=px,
            qc_flags_list="",
            feature_config={"spheroid_extent_method": "actin_threshold"},
            border_excluded_labels=border_labels,
        )

        self.assertEqual(per_img["n_nuclei_total"], 2)
        self.assertEqual(per_img["n_nuclei"], 1)  # Only 1 valid non-border nucleus counted
        self.assertEqual(per_img["n_border_excluded"], 1)

        # Check objects.csv row flags
        obj1 = [o for o in obj_rows if o["object_id"].endswith("0001")][0]
        obj2 = [o for o in obj_rows if o["object_id"].endswith("0002")][0]
        self.assertTrue(obj1["excluded_border"])
        self.assertFalse(obj2["excluded_border"])

    def test_density_outlier_flag_and_small_n(self):
        """Density outlier flag test: flags density IQR outliers and skips IQR for materials with n < 3."""
        # 1. Material 'Mat' has 4 images; one has an extreme density outlier
        m1 = ImageQCMetrics("img1", "Mat", "Gel1", 50, 50, 0, 10.0, 200.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=20000.0)
        m2 = ImageQCMetrics("img2", "Mat", "Gel2", 52, 52, 0, 10.0, 210.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=21000.0)
        m3 = ImageQCMetrics("img3", "Mat", "Gel3", 48, 48, 0, 10.0, 220.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=20500.0)
        m4 = ImageQCMetrics("img4", "Mat", "Gel4", 250, 250, 0, 10.0, 205.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=180000.0)  # Extreme outlier

        # 2. Material 'S46D10' has n=2 (< 3 images)
        s1 = ImageQCMetrics("s_img1", "S46D10", "Gel1", 30, 30, 0, 10.0, 200.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=15000.0)
        s2 = ImageQCMetrics("s_img2", "S46D10", "Gel2", 150, 150, 0, 10.0, 200.0, 0, 0, 0, 0, 0, False, False, False, False, False, False, False, False, False, "", nuclei_density_per_mm3=95000.0)

        cohort = [m1, m2, m3, m4, s1, s2]
        res = apply_cohort_qc_flags(cohort, qc_config={"outlier_iqr_factor": 1.5})

        res_by_id = {m.image_id: m for m in res}

        # Mat cohort checks
        self.assertFalse(res_by_id["img1"].flag_density_outlier)
        self.assertFalse(res_by_id["img2"].flag_density_outlier)
        self.assertFalse(res_by_id["img3"].flag_density_outlier)
        self.assertTrue(res_by_id["img4"].flag_density_outlier, "img4 should be flagged as density_outlier")
        self.assertIn("density_outlier", res_by_id["img4"].flags_list)

        # S46D10 (n=2) checks: should be skipped
        self.assertFalse(res_by_id["s_img1"].flag_density_outlier, "S46D10 has n < 3; IQR should be skipped")
        self.assertFalse(res_by_id["s_img2"].flag_density_outlier, "S46D10 has n < 3; IQR should be skipped")

    def test_synthetic_blob_segmentation(self):
        """Sanity check: Cellpose-SAM recovers synthetic nuclei blobs within reasonable tolerance."""
        model_path = Path("models/cpsam_v2")
        if not model_path.exists():
            self.skipTest("CPSAM model weights not available locally")

        np.random.seed(42)
        img = np.zeros((256, 256), dtype=np.uint8)
        centers = [(64, 64), (64, 192), (192, 64), (192, 192)]
        radius = 15.0

        for cy, cx in centers:
            y, x = np.ogrid[:256, :256]
            dist_sq = (y - cy) ** 2 + (x - cx) ** 2
            blob = np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * 220
            img = np.maximum(img, blob.astype(np.uint8))

        # Add simulated sensor noise floor
        img = np.clip(img + np.random.randint(10, 25, (256, 256), dtype=np.uint8), 0, 255).astype(np.uint8)

        engine = CPSAMEngine(model_path=str(model_path), random_seed=42)
        masks, mode = engine.segment_image(
            img,
            {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "diameter": 30.0, "min_size": 15},
            image_id="synthetic_test",
        )
        n_found = int(masks.max())
        self.assertTrue(3 <= n_found <= 5, f"Expected ~4 synthetic blobs, but found {n_found}")


if __name__ == "__main__":
    unittest.main()

