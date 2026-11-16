"""Regression tests for brightfield stats fixes: seed plumbing, low-gel flag, NaN handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

import matrix_mat.brightfield.stats as bf_stats
from matrix_mat.brightfield.config import PipelineConfig, StatsConfig
from matrix_mat.brightfield.stats import CONDITION_SUMMARY_COLUMNS, compute_condition_statistics


def _pairs(n_gels_ctrl=4, n_gels_trt=4, with_nan=False):
    rng = np.random.default_rng(0)
    rows = []
    for cond, n_gels, shift in (("Mat", n_gels_ctrl, 0.0), ("S50", n_gels_trt, 0.5)):
        for g in range(n_gels):
            for k in range(5):
                fc = float(2.0 ** (1.0 + shift + rng.normal(0, 0.2)))
                rows.append({
                    "pair_id": f"{cond}_{g}_{k}", "condition": cond, "replicate": f"gel{g}", "fov": "0001",
                    "fold_change_volume": fc, "log2_fold_change": float(np.log2(fc)),
                    "t0_volume_sphere_um3": 1e6, "t7_volume_sphere_um3": 1e6 * fc,
                    "delta_volume_um3": 1e6 * (fc - 1), "combined_qc_flag": "PASS", "denominator_gate_pass": True,
                })
    df = pd.DataFrame(rows)
    if with_nan:
        df.loc[df.index[-1], "log2_fold_change"] = np.nan
    return df


def test_random_seed_from_pipeline_config_is_used(monkeypatch):
    seen = []
    real = bf_stats.hierarchical_bootstrap_ci

    def _spy(*a, **kw):
        seen.append(kw.get("seed"))
        return real(*a, **kw)

    monkeypatch.setattr(bf_stats, "hierarchical_bootstrap_ci", _spy)
    cfg = PipelineConfig(random_seed=1234)
    cfg.stats.bootstrap_iterations = 50
    cfg.stats.permutation_iterations = 50
    compute_condition_statistics(_pairs(), config=cfg)
    assert seen and set(seen) == {1234}


def test_compute_stats_stage_passes_seed(monkeypatch, tmp_path):
    from matrix_mat.brightfield import run_pipeline

    captured = {}

    def _fake(pairs, all_objects=None, config=None, random_seed=None):
        captured["seed"] = random_seed
        return pd.DataFrame(columns=CONDITION_SUMMARY_COLUMNS)

    monkeypatch.setattr(run_pipeline, "compute_condition_statistics", _fake)
    cfg = PipelineConfig(random_seed=777, output_dir=str(tmp_path))
    run_pipeline.compute_stats([], [], cfg, tmp_path)
    assert captured["seed"] == 777


def test_low_gel_n_flagged():
    cfg = StatsConfig(bootstrap_iterations=50, permutation_iterations=50)
    df = compute_condition_statistics(_pairs(n_gels_ctrl=1, n_gels_trt=4), config=cfg)
    assert "low_gel_n" in df.columns
    row_ctrl = df[df["condition"] == "Mat"].iloc[0]
    row_trt = df[df["condition"] == "S50"].iloc[0]
    assert bool(row_ctrl["low_gel_n"]) is True
    assert row_ctrl["ci_method"] != "hierarchical_bootstrap_log2"
    assert "low_gel_n" in row_ctrl["ci_method"]
    assert bool(row_trt["low_gel_n"]) is False
    assert row_trt["ci_method"] == "hierarchical_bootstrap_log2"


def test_permutation_drops_nan_log2(monkeypatch):
    seen = []
    real = bf_stats.cluster_permutation_test

    def _spy(df, value_col, **kw):
        seen.append(int(df[value_col].isna().sum()))
        return real(df, value_col, **kw)

    monkeypatch.setattr(bf_stats, "cluster_permutation_test", _spy)
    cfg = StatsConfig(bootstrap_iterations=50, permutation_iterations=50)
    out = compute_condition_statistics(_pairs(with_nan=True), config=cfg)
    assert seen == [0]
    assert np.isfinite(out[out["condition"] == "S50"]["p_raw"].iloc[0])
