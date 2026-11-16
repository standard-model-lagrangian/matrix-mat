"""Regression tests for brightfield audit figures and preflight checks."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from matrix_mat.brightfield.audit import audit_fail_class, audit_session_quality, ordered_conditions
from matrix_mat.brightfield.config import PipelineConfig, QCConfig
from matrix_mat.brightfield.preflight import PreflightAuditor

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/synthetic_dataset"


def _objects(conditions):
    rows = []
    for c in conditions:
        for flag in ("PASS", "FAIL"):
            rows.append({"condition": c, "qc_flag": flag, "equivalent_diameter_um": 100.0,
                         "contrast_ratio": 0.3, "solidity": 0.9})
    return pd.DataFrame(rows)


def test_ordered_conditions_known_first_then_unknown():
    assert ordered_conditions(["Zeta", "S50", "Mat", "Alpha"]) == ["Mat", "S50", "Alpha", "Zeta"]


def test_fail_class_audit_handles_unknown_conditions(tmp_path):
    res = audit_fail_class(_objects(["GelX"]), tmp_path, save_pdf=False, qc_config=QCConfig(min_d_um=55.0))
    assert res["png"].exists()
    res = audit_fail_class(_objects(["Mat", "NewGel"]), tmp_path, save_pdf=False)
    assert res["png"].exists()


def test_fail_class_gate_lines_follow_config(tmp_path, monkeypatch):
    import matplotlib.axes
    lines = []
    real = matplotlib.axes.Axes.axvline

    def _spy(self, x=0, *a, **kw):
        lines.append(x)
        return real(self, x, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "axvline", _spy)
    audit_fail_class(_objects(["Mat"]), tmp_path, save_pdf=False,
                     qc_config=QCConfig(min_d_um=55.0, ring_contrast_factor=0.8, min_solidity=0.7))
    assert lines == [55.0, 0.19999999999999996, 0.7] or np.allclose(lines, [55.0, 0.2, 0.7])


def test_session_audit_16bit_uppercase_and_selection(tmp_path):
    d0, d7 = tmp_path / "Day0", tmp_path / "Day7"
    d0.mkdir()
    d7.mkdir()
    img16 = (np.random.default_rng(0).integers(0, 4000, (64, 64))).astype(np.uint16)
    tifffile.imwrite(d0 / "Day0 NewGel gel1_0001_TRANS.TIF", img16)
    tifffile.imwrite(d7 / "Day7 NewGel gel1_0001_TRANS.tiff", img16)
    df, figs = audit_session_quality(d0, d7, tmp_path / "out", save_pdf=False)
    assert len(df) == 2
    assert set(df["condition"]) == {"NewGel"}
    assert df["mean_intensity"].iloc[0] > 255, "16-bit intensities must not be squashed to 8-bit"

    manifest = pd.DataFrame([{"file_path": str(d0 / "Day0 NewGel gel1_0001_TRANS.TIF"),
                              "timepoint": "t0", "condition": "NewGel"}])
    df_sel, _ = audit_session_quality(d0, d7, tmp_path / "out2", save_pdf=False, manifest_df=manifest)
    assert len(df_sel) == 1


def test_session_audit_empty_dirs(tmp_path):
    df, figs = audit_session_quality(tmp_path / "none0", tmp_path / "none7", tmp_path / "out", save_pdf=False)
    assert df.empty and figs["png"].exists()


def test_preflight_warns_on_duplicate_keys_and_missing_weights(tmp_path):
    d0, d7 = tmp_path / "Day0", tmp_path / "Day7"
    shutil.copytree(FIXTURE / "Day0", d0)
    shutil.copytree(FIXTURE / "Day7", d7)
    shutil.copy(d0 / "Day0 Mat gel1_0001_TRANS.tif", d0 / "Mat gel1_0001_TRANS.tif")  # same key via dir timepoint
    cfg = PipelineConfig(t0_dir=str(d0), t7_dir=str(d7), output_dir=str(tmp_path / "o"))
    cfg.segmentation.primary_backend = "cpsam"
    cfg.segmentation.cpsam_model_path = str(tmp_path / "missing_weights")
    rep = PreflightAuditor(cfg).run_preflight_check()
    text = " ".join(rep.warnings)
    assert "duplicate FOV key" in text
    assert "fetch_models" in text
    assert "--allow-uncalibrated" in text
