"""Runs every shipped IF config end to end on the synthetic fixture with a CPU stub segmenter."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from skimage import measure

import matrix_mat.if_density.sweep as sweep_mod
from matrix_mat.if_density.sweep import run_sweep

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_if_dataset"
SHIPPED = ["sweep.yaml", "prepost_ablation.yaml", "robustness_check.yaml"]


class StubIFEngine:
    def segment_image(self, image, params, image_id=""):
        return measure.label(np.asarray(image) > 100).astype(np.int32), "2D"


def _materialise(config_name: str, tmp_path: Path) -> Path:
    data = yaml.safe_load((REPO_ROOT / "configs" / config_name).read_text())
    data["data_dir"] = str(FIXTURE)
    data["runs_dir"] = str(tmp_path / "runs")
    # Keep the comparator dispatch (chosen from the file name) identical to the shipped config.
    out = tmp_path / config_name
    out.write_text(yaml.safe_dump(data))
    return out


@pytest.mark.parametrize("config_name", SHIPPED)
def test_shipped_if_config_runs(config_name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = _materialise(config_name, tmp_path)
    result = run_sweep(config_path=cfg_path, engine=StubIFEngine())
    assert result["status"] == "SUCCESS"
    combos = yaml.safe_load(cfg_path.read_text())["combos"]
    for combo in combos:
        assert (tmp_path / "runs" / combo["slug"] / "run_metadata.json").exists()


@pytest.mark.parametrize("config_name", SHIPPED)
def test_run_metadata_reflects_steps_that_actually_ran(config_name, tmp_path, monkeypatch):
    """run_metadata.json pre/post 'enabled' must equal what the pipeline functions actually did."""
    monkeypatch.chdir(tmp_path)
    executed = {}
    real_pre = sweep_mod.preprocess_field
    real_post = sweep_mod.postprocess_field_masks

    def spy_pre(*args, **kwargs):
        res = real_pre(*args, **kwargs)
        executed.setdefault(current["slug"], {})["pre"] = bool(res.resolved_config.get("enabled", True))
        return res

    def spy_post(*args, **kwargs):
        masks, border, stats = real_post(*args, **kwargs)
        executed.setdefault(current["slug"], {})["post"] = bool(stats.enabled)
        return masks, border, stats

    current = {}
    real_single = sweep_mod.run_single_combo

    def spy_single(*args, **kwargs):
        current["slug"] = kwargs["combo"]["slug"]
        return real_single(*args, **kwargs)

    monkeypatch.setattr(sweep_mod, "preprocess_field", spy_pre)
    monkeypatch.setattr(sweep_mod, "postprocess_field_masks", spy_post)
    monkeypatch.setattr(sweep_mod, "run_single_combo", spy_single)

    cfg_path = _materialise(config_name, tmp_path)
    run_sweep(config_path=cfg_path, engine=StubIFEngine())

    assert executed, "no combos executed"
    for slug, ran in executed.items():
        meta = json.loads((tmp_path / "runs" / slug / "run_metadata.json").read_text())
        assert meta["preprocessing"]["enabled"] == ran["pre"], slug
        assert meta["postprocessing"]["enabled"] == ran["post"], slug
