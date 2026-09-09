"""
End-to-end integration test verifying that the bundled synthetic TIFF dataset fixture
can be fully analyzed without requiring external private microscopy files.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from spheroid_pipeline_v2.config import PipelineConfig
from spheroid_pipeline_v2.run_pipeline import run_spheroid_pipeline


class TestSyntheticDatasetFixture(unittest.TestCase):
    """Verifies that the standalone synthetic dataset runs end-to-end in CI."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="spheroid_fixture_test_"))
        self.d0_dir = Path("tests/fixtures/synthetic_dataset/Day0")
        self.d7_dir = Path("tests/fixtures/synthetic_dataset/Day7")

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_fixture_runs_end_to_end(self):
        self.assertTrue(self.d0_dir.exists(), "Synthetic Day0 fixture missing")
        self.assertTrue(self.d7_dir.exists(), "Synthetic Day7 fixture missing")

        config = PipelineConfig(
            t0_dir=str(self.d0_dir),
            t7_dir=str(self.d7_dir),
            output_dir=str(self.temp_dir),
            random_seed=42,
        )
        # Use classical backend in headless CI to avoid GPU/DL model downloading
        config.segmentation.primary_backend = "classical"
        config.segmentation.force_recompute = True

        result = run_spheroid_pipeline(config)

        self.assertEqual(result.get("status"), "SUCCESS")
        self.assertTrue((self.temp_dir / "manifest.csv").exists())
        self.assertTrue((self.temp_dir / "objects.csv").exists())
        self.assertTrue((self.temp_dir / "pairs.csv").exists())
        self.assertTrue((self.temp_dir / "condition_summary.csv").exists())
        self.assertTrue((self.temp_dir / "report.md").exists())
        self.assertTrue((self.temp_dir / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
