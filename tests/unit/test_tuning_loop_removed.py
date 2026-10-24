import copy
from pathlib import Path
import pytest

from spheroid_pipeline_v2.config import PipelineConfig
from spheroid_pipeline_v2.run_pipeline import run_spheroid_pipeline


def test_sample_mode_does_not_mutate_config(tmp_path):
    """Running in sample mode must not mutate the shared PipelineConfig."""
    fixture_dir = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_dataset"
    out_dir = tmp_path / "sample_out"

    config = PipelineConfig(
        t0_dir=str(fixture_dir / "Day0"),
        t7_dir=str(fixture_dir / "Day7"),
        output_dir=str(out_dir),
    )
    config.segmentation.primary_backend = "classical"
    config.segmentation.force_recompute = True

    config_snapshot = copy.deepcopy(config)

    res = run_spheroid_pipeline(config, sample_n=4)
    assert res.get("status") == "SUCCESS"

    # Compare config after run with initial snapshot
    assert config == config_snapshot, "PipelineConfig was mutated during sample execution!"
