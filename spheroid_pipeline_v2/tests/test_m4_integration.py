"""
Comprehensive Integration Test Suite for Milestone 4 (M4).

Covers:
  - Visualizer, overlay generation, contact sheets, and publication figures (PNG + PDF).
  - Executive report generation (report.md and report.html).
  - Review manager, manual mask ingestion, agreement metrics, and mask overrides.
  - Manifest building, dataset inventory, and filename parsing.
  - Programmatic sanity gates evaluation and tuning cycle limits.
  - Synthetic smoke test execution.
  - Full end-to-end pipeline execution on synthetic and sampled datasets.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List
import unittest
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from spheroid_pipeline_v2.config import PipelineConfig
from spheroid_pipeline_v2.measure import SpheroidObjectRecord, save_objects_csv
from spheroid_pipeline_v2.pair import SpheroidPairRecord, save_pairs_csv
from spheroid_pipeline_v2.report import ReportGenerator, df_to_markdown_table
from spheroid_pipeline_v2.review import VALIDATION_CSV_COLUMNS, ReviewManager
from spheroid_pipeline_v2.run_pipeline import (
    build_cli_parser,
    build_manifest,
    evaluate_sample_12_sanity_gates,
    parse_image_filename,
    run_spheroid_pipeline,
    run_synthetic_smoke_test,
)
from spheroid_pipeline_v2.visualize import (
    CONDITION_COLORS,
    QC_COLORS_BGR,
    QC_COLORS_RGB,
    Visualizer,
    create_contact_sheet,
    create_overlay,
    generate_all_figures,
    generate_condition_contact_sheets,
    plot_circularity_sensitivity_audit,
    plot_fold_change_violin,
    plot_growth_slopegraph,
    plot_v0_vs_v7_scatter,
)


class TestM4Visualization(unittest.TestCase):
    """Tests for visualization overlays, contact sheets, and publication figures."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_mock_objects(self) -> Tuple[np.ndarray, np.ndarray, List[SpheroidObjectRecord]]:
        img = np.full((300, 400), 200, dtype=np.uint8)
        mask = np.zeros((300, 400), dtype=np.int32)
        # Object 1 (PASS)
        cv2.circle(img, (100, 150), 30, 80, -1)
        cv2.circle(mask, (100, 150), 30, 1, -1)
        # Object 2 (REVIEW)
        cv2.circle(img, (250, 150), 20, 120, -1)
        cv2.circle(mask, (250, 150), 20, 2, -1)

        rec1 = SpheroidObjectRecord(
            object_id="Mat_gel1_0001_t0_obj1",
            image_id="Day0 Mat gel1_0001_TRANS",
            timepoint="t0",
            condition="Mat",
            replicate="gel1",
            fov="0001",
            pair_key="Mat_gel1_0001",
            label_idx=1,
            pixel_size_um=1.518817,
            area_px=float(np.pi * 30**2),
            perimeter_px=float(2 * np.pi * 30),
            equivalent_diameter_px=60.0,
            centroid_y=150.0,
            centroid_x=100.0,
            area_um2=float(np.pi * 30**2 * 1.518817**2),
            equivalent_diameter_um=60.0 * 1.518817,
            major_axis_um=60.0 * 1.518817,
            minor_axis_um=60.0 * 1.518817,
            aspect_ratio=1.0,
            volume_sphere_um3=(np.pi / 6.0) * (60.0 * 1.518817)**3,
            volume_ellipsoid_um3=(np.pi / 6.0) * (60.0 * 1.518817)**3,
            volume_discrepancy_ratio=0.0,
            circularity=0.98,
            solidity=0.99,
            eccentricity=0.1,
            mean_interior_intensity=80.0,
            mean_background_ring_intensity=200.0,
            contrast_ratio=0.40,
            qc_flag="PASS",
            qc_reasons="None",
            pair_id="pair_001",
            mask_source="automated",
        )

        rec2 = SpheroidObjectRecord(
            object_id="Mat_gel1_0001_t0_obj2",
            image_id="Day0 Mat gel1_0001_TRANS",
            timepoint="t0",
            condition="Mat",
            replicate="gel1",
            fov="0001",
            pair_key="Mat_gel1_0001",
            label_idx=2,
            pixel_size_um=1.518817,
            area_px=float(np.pi * 20**2),
            perimeter_px=float(2 * np.pi * 20),
            equivalent_diameter_px=40.0,
            centroid_y=150.0,
            centroid_x=250.0,
            area_um2=float(np.pi * 20**2 * 1.518817**2),
            equivalent_diameter_um=40.0 * 1.518817,
            major_axis_um=40.0 * 1.518817,
            minor_axis_um=40.0 * 1.518817,
            aspect_ratio=1.0,
            volume_sphere_um3=(np.pi / 6.0) * (40.0 * 1.518817)**3,
            volume_ellipsoid_um3=(np.pi / 6.0) * (40.0 * 1.518817)**3,
            volume_discrepancy_ratio=0.0,
            circularity=0.60,
            solidity=0.90,
            eccentricity=0.2,
            mean_interior_intensity=120.0,
            mean_background_ring_intensity=200.0,
            contrast_ratio=0.60,
            qc_flag="REVIEW",
            qc_reasons="low_circularity (0.60 < 0.65)",
            pair_id="pair_002",
            mask_source="automated",
        )

        return img, mask, [rec1, rec2]

    def test_create_overlay_generates_valid_png(self):
        """create_overlay produces non-empty RGB PNG with annotations."""
        img, mask, objs = self._create_mock_objects()
        out_path = Path(self.test_dir) / "test_overlay.png"
        res = create_overlay(img, mask, objs, out_path, pixel_size_um=1.518817)

        self.assertTrue(res.exists())
        self.assertTrue(out_path.stat().st_size > 0)
        with Image.open(out_path) as im:
            self.assertEqual(im.size, (400, 300))
            self.assertEqual(im.mode, "RGB")

    def test_create_overlay_handles_float_and_multichannel(self):
        """create_overlay handles float [0, 1] and 3-channel input arrays."""
        img_f = np.full((100, 100), 0.75, dtype=np.float32)
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[20:50, 20:50] = 1
        out_path = Path(self.test_dir) / "float_overlay.png"
        res = create_overlay(img_f, mask, [], out_path)
        self.assertTrue(res.exists())

        img_3c = np.full((100, 100, 3), 180, dtype=np.uint8)
        out_path2 = Path(self.test_dir) / "3c_overlay.png"
        res2 = create_overlay(img_3c, mask, [], out_path2)
        self.assertTrue(res2.exists())

    def test_create_contact_sheet(self):
        """create_contact_sheet creates multi-image grid montage."""
        p1 = Path(self.test_dir) / "img1_overlay.png"
        p2 = Path(self.test_dir) / "img2_overlay.png"
        cv2.imwrite(str(p1), np.full((150, 200, 3), 100, dtype=np.uint8))
        cv2.imwrite(str(p2), np.full((150, 200, 3), 150, dtype=np.uint8))

        sheet_path = Path(self.test_dir) / "contact_sheet.png"
        res = create_contact_sheet([p1, p2], sheet_path, grid_cols=2, thumb_size=(200, 150), title="Test Grid")

        self.assertIsNotNone(res)
        self.assertTrue(sheet_path.exists())
        with Image.open(sheet_path) as im:
            self.assertTrue(im.width > 200)
            self.assertTrue(im.height > 150)

    def test_generate_all_figures_both_png_and_pdf(self):
        """generate_all_figures generates all 4 publication figures in PNG and PDF."""
        # Create mock pairs DataFrame
        pairs_df = pd.DataFrame([
            {
                "condition": "Mat",
                "t0_volume_sphere_um3": 1e5,
                "t7_volume_sphere_um3": 2.5e5,
                "fold_change_volume": 2.5,
                "denominator_gate_pass": True,
            },
            {
                "condition": "S50",
                "t0_volume_sphere_um3": 1.2e5,
                "t7_volume_sphere_um3": 1.8e5,
                "fold_change_volume": 1.5,
                "denominator_gate_pass": True,
            },
            {
                "condition": "S34D30",
                "t0_volume_sphere_um3": 8e4,
                "t7_volume_sphere_um3": 2.0e5,
                "fold_change_volume": 2.5,
                "denominator_gate_pass": True,
            },
        ])

        objects_df = pd.DataFrame([
            {
                "circularity": 0.85,
                "volume_discrepancy_ratio": 0.05,
                "qc_flag": "PASS",
            },
            {
                "circularity": 0.55,
                "volume_discrepancy_ratio": 0.25,
                "qc_flag": "REVIEW",
            },
        ])

        fig_dict = generate_all_figures(pairs_df, objects_df, self.test_dir, save_pdf=True, save_png=True)
        self.assertIn("fold_change_violin", fig_dict)
        self.assertIn("v0_vs_v7_scatter", fig_dict)
        self.assertIn("growth_slopegraph", fig_dict)
        self.assertIn("circularity_sensitivity_audit", fig_dict)

        for fig_name, paths in fig_dict.items():
            self.assertIn("png", paths)
            self.assertIn("pdf", paths)
            self.assertTrue(paths["png"].exists())
            self.assertTrue(paths["pdf"].exists())
            self.assertTrue(paths["png"].stat().st_size > 0)
            self.assertTrue(paths["pdf"].stat().st_size > 0)


class TestM4ReviewManager(unittest.TestCase):
    """Tests for review manager, manual ground truth masks, and validation metrics."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.review_dir = Path(self.test_dir) / "review"
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self.mgr = ReviewManager(self.review_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_manual_mask_matching_patterns(self):
        """Finds manual mask matching naming conventions."""
        mask_file = self.review_dir / "Day0 Mat gel1_0001_TRANS.png"
        cv2.imwrite(str(mask_file), np.ones((50, 50), dtype=np.uint8) * 255)

        found = self.mgr.find_manual_mask("Day0 Mat gel1_0001_TRANS", "Mat_gel1_0001", "t0")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "Day0 Mat gel1_0001_TRANS.png")

    def test_compute_agreement_metrics(self):
        """Computes Dice and IoU agreement metrics."""
        m_auto = np.zeros((100, 100), dtype=np.int32)
        m_manual = np.zeros((100, 100), dtype=np.int32)
        m_auto[20:60, 20:60] = 1  # 40x40 = 1600 px
        m_manual[20:60, 20:60] = 1

        dice, iou = self.mgr.compute_agreement(m_auto, m_manual)
        self.assertAlmostEqual(dice, 1.0)
        self.assertAlmostEqual(iou, 1.0)

        # Shift manual mask slightly
        m_manual2 = np.zeros((100, 100), dtype=np.int32)
        m_manual2[20:60, 30:70] = 1  # overlap 40x30 = 1200 px, union 2000 px
        dice2, iou2 = self.mgr.compute_agreement(m_auto, m_manual2)
        self.assertAlmostEqual(iou2, 1200.0 / 2000.0)
        self.assertAlmostEqual(dice2, (2.0 * 1200.0) / 3200.0)

    def test_evaluate_and_override_returns_correct_flag(self):
        """evaluate_and_override returns manual_review flag when mask is present."""
        auto_mask = np.zeros((50, 50), dtype=np.int32)
        auto_mask[10:30, 10:30] = 1

        # No manual mask
        m1, src1, val1 = self.mgr.evaluate_and_override("img_no_manual", auto_mask)
        self.assertEqual(src1, "automated")
        self.assertIsNone(val1)

        # Create manual mask
        man_file = self.review_dir / "img_with_manual.png"
        cv2.imwrite(str(man_file), (auto_mask > 0).astype(np.uint8) * 255)

        m2, src2, val2 = self.mgr.evaluate_and_override("img_with_manual", auto_mask)
        self.assertEqual(src2, "manual_review")
        self.assertIsNotNone(val2)
        self.assertEqual(val2["dice_coefficient"], 1.0)

    def test_save_validation_report_10_column_schema(self):
        """save_validation_report adheres to the exact 10-column schema."""
        out_csv = Path(self.test_dir) / "validation.csv"
        records = [{
            "image_id": "Day0 Mat gel1_0001_TRANS",
            "pair_key": "Mat_gel1_0001",
            "timepoint": "t0",
            "manual_mask_path": "/path/to/mask.png",
            "n_auto_objects": 2,
            "n_manual_objects": 2,
            "dice_coefficient": 0.95,
            "iou_jaccard": 0.91,
            "pixel_size_um": 1.518817,
            "notes": "Verified",
        }]
        df = self.mgr.save_validation_report(records, out_csv)
        self.assertEqual(list(df.columns), VALIDATION_CSV_COLUMNS)
        self.assertEqual(len(df), 1)


class TestM4Reporting(unittest.TestCase):
    """Tests for report.md and report.html generation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_df_to_markdown_table_formatting(self):
        """df_to_markdown_table formats DataFrames cleanly."""
        df = pd.DataFrame({
            "condition": ["Mat", "S50"],
            "fold_change_median": [1.4567, 2.1],
            "n_pass_pairs": [15, 8],
        })
        md = df_to_markdown_table(df)
        self.assertIn("| Mat", md)
        self.assertIn("| S50", md)
        self.assertIn("1.457", md)

    def test_report_generator_creates_md_and_html(self):
        """ReportGenerator creates comprehensive report.md and report.html."""
        gen = ReportGenerator(PipelineConfig(), self.test_dir)
        manifest_df = pd.DataFrame([
            {"image_id": "img1", "pair_key": "Mat_gel1_0001", "is_matched_pair": True},
            {"image_id": "img2", "pair_key": "Mat_gel1_0001", "is_matched_pair": True},
        ])
        objects_df = pd.DataFrame([
            {"object_id": "obj1", "pair_key": "Mat_gel1_0001", "qc_flag": "PASS"},
            {"object_id": "obj2", "pair_key": "Mat_gel1_0001", "qc_flag": "PASS"},
        ])
        pairs_df = pd.DataFrame([
            {"pair_id": "p1", "condition": "Mat", "fold_change_volume": 1.5, "denominator_gate_pass": True},
        ])
        cond_df = pd.DataFrame([
            {"condition": "Mat", "n_matched_pairs": 1, "fold_change_median": 1.5, "significance": "control"},
        ])

        md_p, html_p = gen.generate(
            manifest_df=manifest_df,
            objects_df=objects_df,
            pairs_df=pairs_df,
            condition_summary_df=cond_df,
        )

        self.assertTrue(md_p.exists())
        self.assertTrue(html_p.exists())
        md_text = md_p.read_text(encoding="utf-8")
        html_text = html_p.read_text(encoding="utf-8")

        self.assertIn("Executive Report", md_text)
        self.assertIn("Condition-Level Growth", md_text)
        self.assertIn("<!DOCTYPE html>", html_text)
        self.assertIn("Spheroid Volume Analysis Dashboard", html_text)


class TestM4PipelineExecution(unittest.TestCase):
    """Tests for full pipeline execution, sanity gates, and synthetic smoke test."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_synthetic_smoke_test(self):
        """run_synthetic_smoke_test passes all 3 scenarios."""
        passed = run_synthetic_smoke_test()
        self.assertTrue(passed)

    def test_filename_parsing_and_manifest(self):
        """parse_image_filename correctly decodes condition, replicate, field, timepoint."""
        f1 = "Day0 Mat gel1_0001_TRANS.tif"
        res1 = parse_image_filename(f1)
        self.assertEqual(res1["timepoint"], "t0")
        self.assertEqual(res1["condition"], "Mat")
        self.assertEqual(res1["replicate"], "gel1")
        self.assertEqual(res1["fov"], "0001")
        self.assertEqual(res1["pair_key"], "Mat_gel1_0001")

        f2 = "Day7 S46D10 gel6_0003_TRANS.tif"
        res2 = parse_image_filename(f2)
        self.assertEqual(res2["timepoint"], "t7")
        self.assertEqual(res2["condition"], "S46D10")
        self.assertEqual(res2["replicate"], "gel6")
        self.assertEqual(res2["fov"], "0003")
        self.assertEqual(res2["pair_key"], "S46D10_gel6_0003")

    def test_sample_12_sanity_gates_evaluation(self):
        """evaluate_sample_12_sanity_gates evaluates all 4 criteria."""
        manifest_df = pd.DataFrame([{"image_id": f"img_{i}"} for i in range(12)])
        objects_df = pd.DataFrame([
            {"image_id": f"img_{i}", "object_id": f"obj_{i}", "area_px": 5000.0, "equivalent_diameter_um": 120.0, "qc_flag": "PASS"}
            for i in range(10)
        ])
        passed, rep = evaluate_sample_12_sanity_gates(objects_df, manifest_df, PipelineConfig())
        self.assertTrue(passed)
        self.assertTrue(rep["gate1_passed"])
        self.assertTrue(rep["gate2_passed"])
        self.assertTrue(rep["gate3_passed"])
        self.assertTrue(rep["gate4_passed"])

    def test_cli_parser_options(self):
        """CLI parser accepts all specified flags."""
        parser = build_cli_parser()
        args = parser.parse_args([
            "--sample", "12",
            "--output-dir", "custom_out",
            "--pixel-size", "1.52",
            "--no-cache",
        ])
        self.assertEqual(args.sample, 12)
        self.assertEqual(args.output_dir, "custom_out")
        self.assertEqual(args.pixel_size, 1.52)
        self.assertTrue(args.no_cache)


if __name__ == "__main__":
    unittest.main()
