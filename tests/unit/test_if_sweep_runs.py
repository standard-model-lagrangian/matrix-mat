"""Sweep orchestration on the synthetic fixture with a CPU stub engine (no Cellpose)."""
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from skimage import measure

import matrix_mat.if_density.sweep as sweep_mod
from matrix_mat.if_density.engine import CPSAMEngine
from matrix_mat.if_density.sweep import resolve_run_slug, run_sweep, stratified_sample, subset_slug

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_if_dataset"


class CountingEngine:
    def __init__(self):
        self.calls = 0

    def segment_image(self, image, params, image_id=""):
        self.calls += 1
        return measure.label(np.asarray(image) > 100).astype(np.int32), "2D"


def _config(tmp_path: Path, name="prepost_ablation.yaml", data_dir=FIXTURE, out_name=None, **over) -> Path:
    data = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    data["data_dir"] = str(data_dir)
    data["runs_dir"] = str(tmp_path / "runs")
    data["allow_uncalibrated"] = True
    weights = tmp_path / "weights_cpsam_v2"
    weights.write_bytes(b"x")
    data["model"]["pretrained_model"] = str(weights)
    data.update(over)
    out = tmp_path / (out_name or name)
    out.write_text(yaml.safe_dump(data))
    return out


# --- item 9 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("101_PP00_none_none", "101_PP00_none_none_subset"),
    ("10_test", "10_test_subset"),
    ("abs_slug", "abs_slug_subset"),
    ("x_subset", "x_subset"),
])
def test_subset_slug_is_suffix_only(raw, expected):
    assert subset_slug(raw) == expected
    assert resolve_run_slug({"slug": raw}, 1, "full") == raw


# --- items 5, 11: cache key and resume --------------------------------------------------------------

def test_arms_with_same_segmentation_inputs_share_masks_and_runs_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    eng = CountingEngine()
    run_sweep(config_path=cfg, engine=eng)
    # PP00 and PP01 segment the same raw image with the same Cellpose params -> one segmentation each
    # field; PP10 and PP11 share the preprocessed image -> one more. 2 fields x 2 distinct inputs.
    assert eng.calls == 4
    for slug in ("101_PP00_none_none", "104_PP11_full"):
        assert (tmp_path / "runs" / slug / sweep_mod.COMPLETION_MARKER).exists()

    eng2 = CountingEngine()
    res = run_sweep(config_path=cfg, engine=eng2)
    assert eng2.calls == 0 and len(res["skipped_runs"]) == 4

    eng3 = CountingEngine()
    res = run_sweep(config_path=cfg, engine=eng3, overwrite_override=True)
    assert res["skipped_runs"] == [] and eng3.calls == 8  # overwrite also bypasses the mask cache


def test_cache_key_includes_seed_and_model_hash_but_not_names():
    combo = {"slug": "a", "name": "A", "description": "x", "flow_threshold": 0.4, "preprocessing": {"enabled": True}}
    other = dict(combo, slug="b", name="B", description="y", preprocessing=None)
    p = sweep_mod.segmentation_cache_params
    assert p(combo, 42) == p(other, 42)
    assert p(combo, 42) != p(combo, 7)

    class E:
        weights_sha256 = "abc"
    assert sweep_mod.engine_model_hash(E()) == "abc"
    assert sweep_mod.engine_model_hash(CountingEngine()).startswith("engine:")


def test_corrupt_cache_entry_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    run_sweep(config_path=cfg, engine=CountingEngine())
    for png in (tmp_path / "runs" / "cache" / "masks").glob("*.png"):
        png.write_bytes(b"not a png")
    for marker in (tmp_path / "runs").glob(f"*/{sweep_mod.COMPLETION_MARKER}"):
        marker.unlink()
    eng = CountingEngine()
    run_sweep(config_path=cfg, engine=eng)
    assert eng.calls == 4


# --- item 6: arm-independent denominator ------------------------------------------------------------

def test_spheroid_area_identical_across_preprocessing_arms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_sweep(config_path=_config(tmp_path), engine=CountingEngine())
    areas = {
        slug: pd.read_csv(tmp_path / "runs" / slug / "features" / "per_image.csv").set_index("image_id")["spheroid_area_um2"]
        for slug in ("101_PP00_none_none", "102_PP10_bg_only", "103_PP01_post_only", "104_PP11_full")
    }
    ref = areas["101_PP00_none_none"]
    for _slug, a in areas.items():
        pd.testing.assert_series_equal(a, ref, check_names=False)
    pi = pd.read_csv(tmp_path / "runs" / "104_PP11_full" / "features" / "per_image.csv")
    assert {"density_area_um2", "spheroid_extent_method", "spheroid_extent_hull_fallback", "scale_source"} <= set(pi.columns)
    # Border exclusion is on in PP11: the denominator excludes the border strip.
    assert (pi["density_area_um2"] <= pi["spheroid_area_um2"]).all()


# --- item 7: provenance lists only what ran -------------------------------------------------------

def test_provenance_methods_are_truthful(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_sweep(config_path=_config(tmp_path), engine=CountingEngine())
    m00 = json.loads((tmp_path / "runs" / "101_PP00_none_none" / "run_metadata.json").read_text())
    m11 = json.loads((tmp_path / "runs" / "104_PP11_full" / "run_metadata.json").read_text())
    assert m00["preprocessing"]["methods"] == [] and m00["postprocessing"]["methods"] == []
    assert m11["preprocessing"]["methods"] == ["background_subtraction_rolling_ball", "normalize_percentile"]
    post = m11["postprocessing"]
    assert "deduplicate" not in post["methods"]
    assert set(post["methods"]) == {"split_merged_seeded_watershed", "size_gate", "border_exclusion"}
    sg = post["resolved_parameters"]["size_gate"]
    assert sg["min_vol_um3"] == 210.0 and sg["min_vol_um3_source"] == "config"
    assert sg["vol_multiplier_min"] == pytest.approx(210.0 / 841.04, abs=1e-4)
    assert m11["progress"]["status"] == "complete" and m11["progress"]["fields_completed"] == 2
    assert "spheroid_extent" in m11["execution_stats"] and "pixel_scale" in m11["execution_stats"]


# --- item 10: comparator from the config key, not the file name -------------------------------------

def test_comparator_chosen_by_config_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    called = []
    monkeypatch.setattr(sweep_mod, "run_comparator_prepost", lambda **kw: called.append("prepost"))
    monkeypatch.setattr(sweep_mod, "generate_cross_run_comparison", lambda **kw: called.append(("generic", kw.get("run_names"))))
    cfg = _config(tmp_path, out_name="robustness_prepost_named.yaml", comparator="generic")
    run_sweep(config_path=cfg, engine=CountingEngine())
    assert called == [("generic", ["101_PP00_none_none", "102_PP10_bg_only", "103_PP01_post_only", "104_PP11_full"])]


# --- item 11: pairing, scope, sampling, exit codes ---------------------------------------------------

def _data_with_unpaired(tmp_path):
    d = tmp_path / "data"
    shutil.copytree(FIXTURE, d)
    shutil.copy(next(d.glob("*_ch00.tif")), d / "SKOV3 Spheroid D7 Mor_S50-Gel3-1_ch00.tif")
    return d


def test_strict_pairing_overrides_yaml_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path, data_dir=_data_with_unpaired(tmp_path))
    with pytest.raises(RuntimeError, match="unpaired"):
        run_sweep(config_path=cfg, engine=CountingEngine(), strict_pairing=True)
    assert run_sweep(config_path=cfg, engine=CountingEngine())["status"] == "SUCCESS"


def test_subset_scope_without_matching_fields_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep_mod, "SUBSET_FIELD_IDS", ["no such field"])
    with pytest.raises(ValueError, match="subset"):
        run_sweep(config_path=_config(tmp_path), engine=CountingEngine(), scope="subset")


def test_subset_runs_get_suffix_and_do_not_touch_full_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sweep_mod, "SUBSET_FIELD_IDS", ["SKOV3 Spheroid D7 Mor_Mat-Gel1-1"])
    run_sweep(config_path=_config(tmp_path), engine=CountingEngine(), scope="subset")
    runs = {p.name for p in (tmp_path / "runs").iterdir()}
    assert "101_PP00_none_none_subset" in runs and "101_PP00_none_none" not in runs
    assert "comparison_prepost_subset" in runs


def test_sample_is_stratified_by_material():
    class R:
        def __init__(self, fid, mat):
            self.field_id, self.material = fid, mat
    recs = [R(f"a{i}", "Mat") for i in range(5)] + [R("b0", "S34D30")]
    assert {r.material for r in stratified_sample(recs, 2)} == {"Mat", "S34D30"}


def test_main_returns_exit_codes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _config(tmp_path)
    assert sweep_mod.main(["sweep", "--config", str(tmp_path / "missing.yaml")]) == 1
    assert sweep_mod.main(["sweep", "--config", str(cfg), "--dry-run"]) == 0
    assert not (tmp_path / "runs").exists()  # dry run writes nothing


def test_dry_run_reports_missing_weights_and_does_not_import_torch(tmp_path):
    cfg = _config(tmp_path)
    data = yaml.safe_load(cfg.read_text())
    data["model"]["pretrained_model"] = str(tmp_path / "absent")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data))
    code = (
        "import sys; from matrix_mat.if_density.sweep import main; "
        f"ok = main(['sweep', '--config', {str(cfg)!r}, '--dry-run']); "
        f"bad = main(['sweep', '--config', {str(bad)!r}, '--dry-run']); "
        "print('RESULT', ok, bad, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
                         env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")})
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, out.stderr[-2000:]
    assert line[0] == "RESULT 0 1 False"
    assert "fetch_models.py" in out.stdout + out.stderr
    assert "101_PP00_none_none" in out.stdout


# --- items 16, 17, 18: logging, reports, progress, single read ---------------------------------------

def test_log_handlers_do_not_accumulate_and_lines_are_not_duplicated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pkg = logging.getLogger("matrix_mat")
    child = logging.getLogger("matrix_mat.if_density")
    before = (len(pkg.handlers), len(child.handlers))
    run_sweep(config_path=_config(tmp_path), engine=CountingEngine())
    assert (len(pkg.handlers), len(child.handlers)) == before
    log = (tmp_path / "runs" / "104_PP11_full" / "log.txt").read_text()
    starts = [ln for ln in log.splitlines() if "Starting Run: 104_PP11_full" in ln]
    assert len(starts) == 1
    assert "field 1/2" in log and "ETA" in log

    class Boom(CountingEngine):
        def segment_image(self, *a, **k):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError):
        run_sweep(config_path=_config(tmp_path), engine=Boom(), overwrite_override=True)
    assert (len(pkg.handlers), len(child.handlers)) == before


def test_each_field_read_once_per_combo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reads = []
    monkeypatch.setattr(sweep_mod, "run_comparator_prepost", lambda **kw: None)  # count combo reads only
    real = sweep_mod.tifffile.imread
    monkeypatch.setattr(sweep_mod.tifffile, "imread", lambda p, *a, **k: (reads.append(str(p)), real(p, *a, **k))[1])
    run_sweep(config_path=_config(tmp_path), engine=CountingEngine())
    raw = [r for r in reads if str(FIXTURE) in r]
    assert len(raw) == 4 * 2 * 2  # 4 combos x 2 fields x 2 channels


def test_flagged_panels_sorted_by_severity_and_figure_failure_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class M:
        def __init__(self, i, flags):
            self.image_id, self.flags_list, self.flag_any = i, flags, True
    seen = []
    monkeypatch.setattr(sweep_mod, "save_qc_overlay_panels", lambda imgs, d: seen.extend(x["image_id"] for x in imgs))
    monkeypatch.setattr(sweep_mod, "generate_all_if_biological_figures", lambda **k: (_ for _ in ()).throw(ValueError("no fig")))
    metrics = [M("a", "low_snr"), M("b", "blurry;saturated;low_snr"), M("c", "density_outlier;edge_heavy;saturated")]
    status = sweep_mod.render_reports(metrics, {m.image_id: tmp_path / "x.png" for m in metrics}, tmp_path, tmp_path, "s")
    assert seen == ["b", "c", "a"]
    assert status.startswith("failed")

    res = run_sweep(config_path=_config(tmp_path), engine=CountingEngine())
    assert set(res["figure_failures"]) == {"101_PP00_none_none", "102_PP10_bg_only", "103_PP01_post_only", "104_PP11_full"}
    meta = json.loads((tmp_path / "runs" / "101_PP00_none_none" / "run_metadata.json").read_text())
    assert meta["execution_stats"]["figures"].startswith("failed")
    marker = json.loads((tmp_path / "runs" / "101_PP00_none_none" / sweep_mod.COMPLETION_MARKER).read_text())
    assert marker["figures"].startswith("failed")


# --- items 7, 12, 18: engine --------------------------------------------------------------------------

class _FakeModel:
    def __init__(self):
        self.seen = None

    def eval(self, img, **kw):
        self.seen = (img.shape, img.dtype, kw)
        return np.zeros(img.shape[:2], np.int32), None, None


def test_engine_honours_use_gpu_and_model_type(caplog):
    with caplog.at_level(logging.INFO, logger="matrix_mat.if_density.engine"):
        eng = CPSAMEngine(model_path="x", use_gpu=False)
    assert (eng.use_gpu, eng.device_str) == (False, "cpu")
    assert "device=cpu" in caplog.text
    with pytest.raises(ValueError, match="model_type"):
        CPSAMEngine(model_path="x", use_gpu=False, model_type="cyto3")


def test_engine_converts_rgb_and_rejects_stacks_and_3d():
    eng = CPSAMEngine(model_path="x", use_gpu=False)
    eng._model = _FakeModel()
    rgb = np.zeros((64, 64, 3), np.uint8)
    eng.segment_image(rgb, {})
    assert eng._model.seen[0] == (64, 64) and eng._model.seen[2]["do_3D"] is False
    eng.segment_image(np.full((64, 64), 4000, np.uint16), {})
    assert eng._model.seen[1] == np.uint8
    with pytest.raises(ValueError):
        eng.segment_image(np.zeros((5, 64, 64), np.uint8), {})
    with pytest.raises(ValueError, match="do_3D"):
        eng.segment_image(np.zeros((64, 64), np.uint8), {"do_3D": True})
