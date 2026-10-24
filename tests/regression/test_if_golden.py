import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from skimage import measure

from matrix_mat.if_density.sweep import run_sweep


class StubIFEngine:
    """Deterministic CPU stub segmenter for IF synthetic testing."""
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def segment_image(self, image: np.ndarray, params: dict, image_id: str = ""):
        # Simple intensity thresholding producing clean labeled masks
        binary = (image > 100)
        labeled = measure.label(binary).astype(np.int32)
        return labeled, "2D"


def test_synthetic_if_matches_golden(tmp_path, pytestconfig):
    data_dir = Path("tests/fixtures/synthetic_if_dataset").resolve()
    golden_dir = Path("tests/regression/golden_if").resolve()
    golden_dir.mkdir(parents=True, exist_ok=True)

    config_content = f"""
data_dir: "{data_dir}"
runs_dir: "{tmp_path}/runs"
allow_unpaired: true
overwrite: true
random_seed: 42
default_pixel_size_um: 0.505049

model:
  pretrained_model: "models/cpsam_v2"
  model_type: "cpsam"
  use_gpu: false

combos:
  - slug: "001_A_baseline"
    name: "A_baseline"
    description: "Synthetic test baseline"
    flow_threshold: 0.4
    cellprob_threshold: 0.0
    diameter: null
    min_size: 5
    stitch_threshold: 0.0
    do_3D: false
    preprocessing:
      enabled: false
    postprocessing:
      enabled: true
      size_gate:
        enabled: true
        reference_vol_um3: 500.0
        min_vol_um3: 50.0
        max_vol_um3: 2000.0
      splitting:
        enabled: false
      border_exclusion:
        enabled: false

preprocessing:
  enabled: false

postprocessing:
  enabled: true
  size_gate:
    enabled: true
    reference_vol_um3: 500.0
    min_vol_um3: 50.0
    max_vol_um3: 2000.0
  splitting:
    enabled: false
  border_exclusion:
    enabled: false

qc:
  snr_min: 1.5
  min_nuclear_vol_um3: 50.0
  max_nuclear_vol_um3: 2000.0
  border_margin_px: 2
  edge_heavy_fraction: 0.25
  saturation_fraction_max: 0.005
  blur_percentile_cutoff: 10.0
  duplicate_iou_threshold: 0.50
  outlier_iqr_factor: 1.5

features:
  spheroid_extent_method: "actin_threshold"
  actin_background_percentile: 50.0
"""
    config_file = tmp_path / "if_test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")

    stub_engine = StubIFEngine()
    result = run_sweep(config_path=config_file, engine=stub_engine)
    assert result.get("status") == "SUCCESS"

    run_dir = tmp_path / "runs" / "001_A_baseline"
    target_files = {
        "qc_flags.csv": run_dir / "qc" / "qc_flags.csv",
        "objects.csv": run_dir / "features" / "objects.csv",
        "per_image.csv": run_dir / "features" / "per_image.csv",
        "summary_by_material.csv": run_dir / "summary" / "summary_by_material.csv",
    }

    update_golden = pytestconfig.getoption("--update-golden", default=False)

    for fname, actual_path in target_files.items():
        golden_path = golden_dir / fname
        assert actual_path.exists(), f"Output file missing: {actual_path}"

        if update_golden or not golden_path.exists():
            shutil.copyfile(actual_path, golden_path)

        df_actual = pd.read_csv(actual_path)
        df_golden = pd.read_csv(golden_path)

        assert list(df_actual.columns) == list(df_golden.columns), (
            f"Columns mismatch for {fname}:\n"
            f"Actual ({len(df_actual.columns)}): {list(df_actual.columns)}\n"
            f"Golden ({len(df_golden.columns)}): {list(df_golden.columns)}"
        )
        pd.testing.assert_frame_equal(
            df_actual,
            df_golden,
            check_exact=False,
            rtol=1e-6,
            atol=1e-8,
        )
