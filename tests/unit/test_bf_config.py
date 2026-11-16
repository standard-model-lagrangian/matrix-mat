"""Regression tests for brightfield config loading fixes."""

from __future__ import annotations

from matrix_mat.brightfield.config import PipelineConfig


def test_windows_root_in_double_quoted_yaml(tmp_path, monkeypatch):
    """Backslash roots must not be re-parsed as YAML escapes (placeholders expand after parsing)."""
    win_root = r"C:\Users\name\new\data"
    monkeypatch.setenv("MATRIX_MAT_DATA_ROOT", win_root)
    y = tmp_path / "c.yaml"
    y.write_text('t0_dir: "${DATA_ROOT}/brightfield/Day0"\nreview_dir: "${DATA_ROOT}/review"\n', encoding="utf-8")
    cfg = PipelineConfig.from_yaml(y)
    assert cfg.t0_dir == win_root + "/brightfield/Day0"
    assert cfg.review_dir == win_root + "/review"


def test_env_expansion_limited_to_string_values(tmp_path, monkeypatch):
    """$VARS expand inside string values only; comments and keys are never rewritten."""
    monkeypatch.setenv("MM_TEST_SUBDIR", "abc")
    monkeypatch.setenv("MM_BREAK", "x: [unclosed")
    y = tmp_path / "c.yaml"
    y.write_text("# $MM_BREAK in a comment must be harmless\nt7_dir: \"/data/$MM_TEST_SUBDIR/Day7\"\n", encoding="utf-8")
    cfg = PipelineConfig.from_yaml(y)
    assert cfg.t7_dir == "/data/abc/Day7"


def test_shipped_yaml_device_and_classical_keys():
    cfg = PipelineConfig.from_yaml("configs/spheroid_brightfield.yaml")
    assert cfg.segmentation.device == "auto"
    assert cfg.segmentation.classical_min_d_um == 40.0
    assert cfg.segmentation.classical_min_contrast == 0.10
    assert cfg.segmentation.classical_ring_width_px == 15
