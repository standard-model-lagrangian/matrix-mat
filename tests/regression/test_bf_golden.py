import shutil
from pathlib import Path
import pandas as pd
import pytest

from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.brightfield.run_pipeline import run_spheroid_pipeline


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_fixture_matches_golden(tmp_path, pytestconfig):
    d0_dir = REPO_ROOT / "tests/fixtures/synthetic_dataset/Day0"
    d7_dir = REPO_ROOT / "tests/fixtures/synthetic_dataset/Day7"
    golden_dir = REPO_ROOT / "tests/regression/golden"

    # The fixture TIFFs carry no EVOS tag; an explicit (user-asserted) pixel size keeps objects
    # calibrated so pairs reach PASS and the statistics columns are actually exercised.
    config = PipelineConfig(
        t0_dir=str(d0_dir),
        t7_dir=str(d7_dir),
        output_dir=str(tmp_path),
        random_seed=42,
        cli_pixel_size_um=1.518817,
    )
    config.segmentation.primary_backend = "classical"
    config.segmentation.force_recompute = True

    result = run_spheroid_pipeline(config)
    assert result.get("status") == "SUCCESS"

    target_files = ["objects.csv", "pairs.csv", "condition_summary.csv"]
    update_golden = pytestconfig.getoption("--update-golden")

    for fname in target_files:
        actual_path = tmp_path / fname
        golden_path = golden_dir / fname
        assert actual_path.exists(), f"Output file missing: {fname}"

        if update_golden:
            golden_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual_path, golden_path)
        assert golden_path.exists(), f"Golden file missing: {golden_path} (regenerate with --update-golden)"

        df_actual = pd.read_csv(actual_path)
        df_golden = pd.read_csv(golden_path)

        # Full column equality - no hiding columns via intersection
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


def test_golden_condition_summary_exercises_statistics():
    """Guard against a golden file in which every statistics column is empty."""
    df = pd.read_csv(REPO_ROOT / "tests/regression/golden/condition_summary.csv")
    assert (df["n_spheroids"] > 0).all()
    assert df["fold_change_median"].notna().all()
    assert df["bootstrap_ci_95_low"].notna().all()
