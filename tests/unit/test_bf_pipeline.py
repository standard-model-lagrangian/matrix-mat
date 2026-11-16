"""Regression tests for the brightfield orchestrator / legacy CLI (run_pipeline.py)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from matrix_mat.brightfield import run_pipeline
from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.brightfield.run_pipeline import (
    build_cli_parser,
    build_manifest,
    main,
    resolve_input_dir,
    run_spheroid_pipeline,
    select_images,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/synthetic_dataset"


def _config(tmp_path, t0=None, t7=None, **kw) -> PipelineConfig:
    cfg = PipelineConfig(
        t0_dir=str(t0 or FIXTURE / "Day0"), t7_dir=str(t7 or FIXTURE / "Day7"),
        output_dir=str(tmp_path / "out"), review_dir=str(tmp_path / "review"),
        cli_pixel_size_um=1.518817, **kw,
    )
    cfg.segmentation.primary_backend = "classical"
    cfg.segmentation.cache_dir = str(tmp_path / "cache")
    cfg.preprocess.save_corrected_images = False
    return cfg


def _yaml(tmp_path, out_dir=None) -> Path:
    y = tmp_path / "cfg.yaml"
    y.write_text(
        f't0_dir: "{FIXTURE / "Day0"}"\n'
        f't7_dir: "{FIXTURE / "Day7"}"\n'
        f'output_dir: "{out_dir or tmp_path / "yaml_out"}"\n'
        f'review_dir: "{tmp_path / "review"}"\n'
        "cli_pixel_size_um: 1.518817\n"
        "preprocess:\n  save_corrected_images: false\n"
        f'segmentation:\n  primary_backend: "classical"\n  cache_dir: "{tmp_path / "cache"}"\n',
        encoding="utf-8",
    )
    return y


def _copy_fixture(tmp_path) -> tuple[Path, Path]:
    d0, d7 = tmp_path / "data" / "Day0", tmp_path / "data" / "Day7"
    shutil.copytree(FIXTURE / "Day0", d0)
    shutil.copytree(FIXTURE / "Day7", d7)
    return d0, d7


# --- Bug 3: legacy CLI -------------------------------------------------------------------------

def test_legacy_main_runs_and_uses_yaml_output_dir(tmp_path):
    """main() must not crash with a TypeError and must not override YAML output_dir/review_dir."""
    rc = main(["--config", str(_yaml(tmp_path)), "--quiet"])
    assert rc == 0
    assert (tmp_path / "yaml_out" / "objects.csv").exists()


def test_cli_defaults_do_not_override_yaml():
    args = build_cli_parser().parse_args([])
    assert args.output_dir is None and args.review_dir is None


def test_missing_config_file_errors(tmp_path):
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2


def test_input_dir_is_wired(tmp_path, capsys):
    d0, d7 = _copy_fixture(tmp_path)
    assert resolve_input_dir(tmp_path / "data") == (d0, d7)
    rc = main(["--config", str(_yaml(tmp_path)), "--input-dir", str(tmp_path / "data"), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(d0) in out and str(d7) in out


def test_dry_run_loads_no_models(tmp_path, capsys, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("dry run must not build the segmentation hierarchy")

    monkeypatch.setattr(run_pipeline, "SegmentationHierarchy", _boom)
    rc = main(["--config", str(_yaml(tmp_path)), "--dry-run", "--condition", "Mat"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 image(s)" in out and "S34D30" not in out
    assert not (tmp_path / "yaml_out").exists()


def test_quiet_smoke_test_prints_nothing_and_uses_temp_dirs(tmp_path, capsys, monkeypatch):
    out_root, cache_root = tmp_path / "out_root", tmp_path / "cache_root"
    monkeypatch.setenv("MATRIX_MAT_OUTPUT_ROOT", str(out_root))
    monkeypatch.setenv("MATRIX_MAT_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("MATRIX_MAT_DECISIONS_PATH", str(out_root / "decisions.jsonl"))
    assert main(["--smoke-test", "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert not out_root.exists() and not cache_root.exists()


def test_smoke_test_does_not_mutate_config():
    cfg = PipelineConfig()
    before = cfg.segmentation.primary_backend
    assert run_pipeline.run_synthetic_smoke_test(cfg) is True
    assert cfg.segmentation.primary_backend == before


def test_sample_is_stratified_across_conditions():
    rows = []
    for cond in ("A", "B", "C"):
        for fov in ("0001", "0002", "0003"):
            for tp in ("t0", "t7"):
                rows.append({"condition": cond, "pair_key": f"{cond}_gel1_{fov}", "timepoint": tp,
                             "image_id": f"{tp}_{cond}_{fov}", "is_matched_pair": True})
    df = pd.DataFrame(rows)
    sel = select_images(df, sample_n=6)
    assert len(sel) == 6
    assert sorted(sel["condition"].unique()) == ["A", "B", "C"]
    assert all(sel.groupby("pair_key")["timepoint"].nunique() == 2)
    assert len(select_images(df, sample_n=12)) == 12
    assert set(select_images(df, condition="B")["condition"]) == {"B"}


def test_build_manifest_case_insensitive_and_deduplicated(tmp_path):
    d0, d7 = _copy_fixture(tmp_path)
    src = d0 / "Day0 Mat gel1_0001_TRANS.tif"
    shutil.copy(src, d0 / "Day0 Mat gel1_0002_TRANS.TIFF")
    shutil.copy(src, d0 / "Day0 Mat gel1_0001_TRANS.tiff")  # duplicate key of an existing file
    df = build_manifest(d0, d7)
    assert "Mat_gel1_0002" in set(df["pair_key"])
    assert len(df[(df["pair_key"] == "Mat_gel1_0001") & (df["timepoint"] == "t0")]) == 1


# --- Bugs 7, 9, 10, 12, 17: orchestration ------------------------------------------------------

def test_report_step_failure_is_recorded_not_fatal(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("figure exploded")

    monkeypatch.setattr(run_pipeline, "audit_fail_class", _boom)
    res = run_spheroid_pipeline(_config(tmp_path))
    assert res["status"] == "COMPLETED_WITH_ERRORS"
    assert [f["step"] for f in res["failed_steps"]] == ["fail_class_audit"]
    assert res["md_report"] is not None and Path(res["md_report"]).exists()
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert "COMPLETED_WITH_ERRORS" in json.dumps(meta)


def test_main_nonzero_when_report_step_fails(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("figure exploded")

    monkeypatch.setattr(run_pipeline, "audit_session_quality", _boom)
    assert main(["--config", str(_yaml(tmp_path)), "--quiet"]) == 1


def test_unreadable_image_recorded_and_all_failed_run_fails(tmp_path, monkeypatch):
    # Preflight only spot-checks the first files; bypass it to exercise the per-image path.
    from matrix_mat.brightfield.preflight import PreflightReport
    monkeypatch.setattr(PreflightReport, "log_and_raise_if_fatal", lambda self: None)
    d0, d7 = tmp_path / "Day0", tmp_path / "Day7"
    d0.mkdir()
    d7.mkdir()
    shutil.copy(FIXTURE / "Day0" / "Day0 Mat gel1_0001_TRANS.tif", d0)
    shutil.copy(FIXTURE / "Day7" / "Day7 Mat gel1_0001_TRANS.tif", d7)
    (d0 / "Day0 Mat gel1_0002_TRANS.tif").write_bytes(b"not a tiff")
    res = run_spheroid_pipeline(_config(tmp_path, t0=d0, t7=d7))
    status = pd.read_csv(tmp_path / "out" / "images_status.csv")
    bad = status[status["image_id"] == "Day0 Mat gel1_0002_TRANS"]
    assert bad["status"].tolist() == ["READ_FAIL"]
    assert res["status"] == "SUCCESS"
    assert len(res["image_failures"]) == 1

    # Every image unreadable -> FAILED
    for f in list(d0.iterdir()) + list(d7.iterdir()):
        f.write_bytes(b"garbage")
    res2 = run_spheroid_pipeline(_config(tmp_path / "second", t0=d0, t7=d7))
    assert res2["status"] == "FAILED"


def test_seg_fail_image_recorded_and_counted(tmp_path):
    d0, d7 = _copy_fixture(tmp_path)
    tifffile.imwrite(d0 / "Day0 S50 gel1_0001_TRANS.tif", np.full((256, 256), 120, np.uint8))  # blank -> no objects
    tifffile.imwrite(d7 / "Day7 S50 gel1_0001_TRANS.tif", np.full((256, 256), 120, np.uint8))
    cfg = _config(tmp_path, t0=d0, t7=d7)
    run_state: dict = {}
    _, out_dir = None, Path(cfg.output_dir)
    out_dir.mkdir(parents=True)
    manifest = build_manifest(d0, d7)
    objs, _ = run_pipeline.segment_cohort(manifest, cfg, out_dir, run_state=run_state)
    st = pd.DataFrame(run_state["image_status"])
    assert set(st[st["condition"] == "S50"]["status"]) == {"SEG_FAIL"}
    _, _, _, _ = run_pipeline.pair_timepoints(objs, cfg, out_dir)
    _, excl = run_pipeline.compute_stats([], objs, cfg, out_dir, image_status=run_state["image_status"])
    e1 = excl[excl["Stage"] == "E1"]["Count"].iloc[0]
    e2 = excl[excl["Stage"] == "E2"]["Count"].iloc[0]
    assert e1 == "3 / 3 / 6"
    assert e2 == "2"
    decisions = [json.loads(x) for x in (out_dir / "decisions.jsonl").read_text().splitlines()]
    s50 = [d for d in decisions if "S50" in d["image_id"] and d["step"] == "Segmentation Hierarchy Result"]
    assert s50 and all(d["outcome"].startswith("SEG_FAIL") for d in s50)


def test_rgb_tiff_is_converted_to_grayscale(tmp_path):
    d0, d7 = _copy_fixture(tmp_path)
    for f in list(d0.iterdir()) + list(d7.iterdir()):
        g = tifffile.imread(f)
        tifffile.imwrite(f, np.stack([g, g, g], axis=-1), photometric="rgb")
    res = run_spheroid_pipeline(_config(tmp_path, t0=d0, t7=d7))
    assert res["status"] == "SUCCESS"
    assert res["pairs_count"] == 2


def test_decisions_log_is_per_run_and_overlays_are_this_runs(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    overlays = tmp_path / "out" / cfg.artifacts.overlay_dir
    overlays.mkdir(parents=True)
    (overlays / "AAA_stale_overlay.png").write_bytes(b"\x89PNG stale")
    captured = {}
    from matrix_mat.reporting import report as report_mod
    real = report_mod.ReportGenerator.generate

    def _spy(self, *a, **kw):
        captured["overlay_samples"] = kw.get("overlay_samples")
        return real(self, *a, **kw)

    monkeypatch.setattr(report_mod.ReportGenerator, "generate", _spy)
    run_spheroid_pipeline(cfg)
    run_spheroid_pipeline(cfg)
    names = [Path(p).name for p in captured["overlay_samples"]]
    assert "AAA_stale_overlay.png" not in names
    assert names == sorted(names) and len(names) == 4
    lines = (tmp_path / "out" / "decisions.jsonl").read_text().splitlines()
    finals = [x for x in lines if "Segmentation Hierarchy Result" in x]
    assert len(finals) == 4, "decisions.jsonl must only hold the latest run"


def test_objects_csv_written_before_pairing(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("pairing crashed")

    monkeypatch.setattr(run_pipeline, "pair_dataset", _boom)
    with pytest.raises(RuntimeError):
        run_spheroid_pipeline(_config(tmp_path))
    df = pd.read_csv(tmp_path / "out" / "objects.csv")
    assert len(df) == 4


def test_dedup_uses_config_and_overlay_redrawn(tmp_path, monkeypatch):
    d0, d7 = _copy_fixture(tmp_path)
    # Same field saved as a second FOV (z-plane duplicate) at both timepoints
    shutil.copy(d0 / "Day0 Mat gel1_0001_TRANS.tif", d0 / "Day0 Mat gel1_0002_TRANS.tif")
    calls = []
    import matrix_mat.reporting.visualize as vis
    real = vis.create_overlay

    def _spy(image, mask, objects, out, pixel_size_um=1.518817):
        calls.append((Path(out).name, sorted(o.qc_flag for o in objects)))
        return real(image, mask, objects, out, pixel_size_um=pixel_size_um)

    monkeypatch.setattr(vis, "create_overlay", _spy)
    cfg = _config(tmp_path, t0=d0, t7=d7)
    manifest = build_manifest(d0, d7)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True)
    objs, _ = run_pipeline.segment_cohort(manifest, cfg, out)
    fails = [o for o in objs if "duplicate_focal_plane" in o.qc_reasons]
    assert len(fails) == 1
    loser_overlay = f"{fails[0].image_id}_overlay.png"
    last = [c for c in calls if c[0] == loser_overlay][-1]
    assert last[1] == ["FAIL"], "overlay must be redrawn with the post-dedup flag"

    cfg.qc.dedup_cross_fov = False
    objs2, _ = run_pipeline.segment_cohort(manifest, cfg, out)
    assert not any("duplicate_focal_plane" in o.qc_reasons for o in objs2)


def test_corrected_images_saved_and_manual_mask_shape_checked(tmp_path):
    from PIL import Image

    cfg = _config(tmp_path)
    cfg.preprocess.save_corrected_images = True
    review = Path(cfg.review_dir)
    review.mkdir(parents=True)
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(review / "Day0 Mat gel1_0001_TRANS.png")  # wrong shape
    out = Path(cfg.output_dir)
    out.mkdir(parents=True)
    run_state: dict = {}
    objs, validation = run_pipeline.segment_cohort(build_manifest(cfg.t0_dir, cfg.t7_dir), cfg, out, run_state=run_state)
    assert sorted(p.name for p in (out / "corrected_images").iterdir()) == sorted(
        f"{p.stem}.png" for d in (FIXTURE / "Day0", FIXTURE / "Day7") for p in d.iterdir()
    )
    st = pd.DataFrame(run_state["image_status"]).set_index("image_id")
    row = st.loc["Day0 Mat gel1_0001_TRANS"]
    assert row["mask_source"] == "automated"
    assert "manual mask rejected" in row["reason"]
    assert validation == []
