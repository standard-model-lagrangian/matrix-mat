"""
Unit & Integration Tests for Preflight Validation, Provenance Tracking, and Biological Plots.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spheroid_if_sweep.biological_plots import (
    plot_clean_vs_all_density,
    plot_nuclear_volume_distributions,
    plot_nuclei_density_by_material,
)
from spheroid_if_sweep.preflight import audit_if_dataset
from spheroid_pipeline_v2.config import PipelineConfig
from spheroid_pipeline_v2.preflight import PreflightAuditor
from spheroid_pipeline_v2.presentation_plots import (
    compute_well_growth_comparison,
    generate_all_presentation_figures,
    stagger_log_positions,
)
from spheroid_pipeline_v2.provenance import (
    compute_file_sha256,
    export_complete_manifest,
    write_run_metadata,
)


class TestPreflightAndProvenance(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Create dummy TIFF files
        self.t0_dir = self.tmp_path / "t0"
        self.t7_dir = self.tmp_path / "t7"
        self.t0_dir.mkdir()
        self.t7_dir.mkdir()

        # Write dummy images matching naming convention
        img_arr = np.random.randint(50, 200, size=(100, 100), dtype=np.uint8)
        tifffile.imwrite(self.t0_dir / "Day0 Mat gel1_0001_TRANS.tif", img_arr)
        tifffile.imwrite(self.t7_dir / "Day7 Mat gel1_0001_TRANS.tif", img_arr)

        self.config = PipelineConfig(
            t0_dir=str(self.t0_dir),
            t7_dir=str(self.t7_dir),
            output_dir=str(self.tmp_path / "out"),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preflight_brightfield_valid(self):
        auditor = PreflightAuditor(self.config)
        report = auditor.run_preflight_check()
        self.assertTrue(report.passed)
        self.assertEqual(report.stats["t0_count"], 1)
        self.assertEqual(report.stats["t7_count"], 1)
        self.assertEqual(report.stats["matched_pairs_count"], 1)
        self.assertTrue(report.stats["control_found"])

    def test_preflight_brightfield_missing_dir(self):
        bad_config = PipelineConfig(
            t0_dir=str(self.tmp_path / "non_existent"),
            t7_dir=str(self.t7_dir),
        )
        auditor = PreflightAuditor(bad_config)
        report = auditor.run_preflight_check()
        self.assertFalse(report.passed)
        self.assertTrue(any("does not exist" in err for err in report.fatal_errors))

    def test_preflight_brightfield_missing_control(self):
        self.config.stats.control_condition = "NonExistentControl"
        auditor = PreflightAuditor(self.config)
        report = auditor.run_preflight_check()
        self.assertTrue(report.passed)
        self.assertTrue(any("Control condition 'NonExistentControl' not found" in w for w in report.warnings))

    def test_provenance_file_sha256(self):
        test_file = self.tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        sha = compute_file_sha256(test_file)
        self.assertEqual(sha, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")

    def test_write_run_metadata(self):
        out_dir = self.tmp_path / "meta_out"
        out_file = write_run_metadata(
            output_dir=out_dir,
            config=self.config,
            wall_time_sec=12.345,
            warnings=["Test warning"],
            extra_info={"objects": 10},
        )
        self.assertTrue(out_file.exists())
        with open(out_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["pipeline_name"], "spheroid_pipeline_v2")
        self.assertEqual(data["wall_time_seconds"], 12.345)
        self.assertEqual(data["warnings_count"], 1)
        self.assertIn("environment", data)
        self.assertIn("dependency_versions", data)

    def test_export_complete_manifest(self):
        manifest_df = pd.DataFrame([{
            "image_id": "Day0 Mat gel1_0001_TRANS",
            "timepoint": "t0",
            "condition": "Mat",
            "replicate": "gel1",
            "fov": "0001",
            "pair_key": "Mat_gel1_0001",
            "file_path": str(self.t0_dir / "Day0 Mat gel1_0001_TRANS.tif"),
            "pixel_size_um": 1.5188,
            "scale_source": "test",
            "is_matched_pair": True,
        }])
        out_dir = self.tmp_path / "manifest_out"
        manifest_csv = export_complete_manifest(manifest_df, out_dir, compute_checksums=True)
        self.assertTrue(manifest_csv.exists())
        df = pd.read_csv(manifest_csv)
        self.assertEqual(len(df), 1)
        self.assertIn("sha256", df.columns)
        self.assertEqual(len(df["sha256"].iloc[0]), 64)

    def test_if_preflight_valid(self):
        # Create dummy paired IF files
        if_dir = self.tmp_path / "if_data"
        if_dir.mkdir()
        tifffile.imwrite(if_dir / "Field1_ch00.tif", np.zeros((50, 50), dtype=np.uint8))
        tifffile.imwrite(if_dir / "Field1_ch02.tif", np.zeros((50, 50), dtype=np.uint8))

        report = audit_if_dataset(data_dir=if_dir, allow_unpaired=True)
        self.assertTrue(report.passed)
        self.assertEqual(report.stats["paired_fields"], 1)

    def test_if_preflight_missing_data_dir(self):
        report = audit_if_dataset(data_dir=self.tmp_path / "non_existent")
        self.assertFalse(report.passed)
        self.assertTrue(any("does not exist" in err for err in report.fatal_errors))

    def test_stagger_log_positions(self):
        y = [10.0, 10.5, 100.0, 1000.0]
        staggered = stagger_log_positions(y, min_log_gap=0.1)
        self.assertEqual(len(staggered), 4)
        self.assertTrue(np.all(staggered > 0))

    def test_biological_plots_generation(self):
        fig_dir = self.tmp_path / "bio_figs"
        per_img_df = pd.DataFrame([
            {"image_id": "img1", "material": "Mat", "nuclei_density_per_mm3": 120.0},
            {"image_id": "img2", "material": "Mat", "nuclei_density_per_mm3": 130.0},
            {"image_id": "img3", "material": "S34D30", "nuclei_density_per_mm3": 210.0},
            {"image_id": "img4", "material": "S34D30", "nuclei_density_per_mm3": 220.0},
        ])
        qc_df = pd.DataFrame([
            {"image_id": "img1", "flag_any": False},
            {"image_id": "img2", "flag_any": False},
            {"image_id": "img3", "flag_any": False},
            {"image_id": "img4", "flag_any": True},
        ])
        objs_df = pd.DataFrame([
            {"material": "Mat", "volume_um3": 850.0, "excluded_border": False},
            {"material": "Mat", "volume_um3": 920.0, "excluded_border": False},
            {"material": "S34D30", "volume_um3": 1100.0, "excluded_border": False},
        ])

        p1_png, p1_pdf = plot_nuclei_density_by_material(per_img_df, fig_dir, control_material="Mat")
        self.assertTrue(p1_png.exists())
        self.assertTrue(p1_pdf.exists())

        p2_png, p2_pdf = plot_clean_vs_all_density(per_img_df, qc_df, fig_dir)
        self.assertTrue(p2_png.exists())
        self.assertTrue(p2_pdf.exists())

        p3_png, p3_pdf = plot_nuclear_volume_distributions(objs_df, fig_dir)
        self.assertTrue(p3_png.exists())
        self.assertTrue(p3_pdf.exists())


if __name__ == "__main__":
    unittest.main()
