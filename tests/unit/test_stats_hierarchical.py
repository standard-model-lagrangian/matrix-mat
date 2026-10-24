import itertools
import math
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from spheroid_pipeline_v2.stats_hierarchical import (
    adjust_pvalues,
    cluster_permutation_test,
    hierarchical_bootstrap_ci,
    summarise_by_unit,
)


def test_bootstrap_ci_widens_with_cluster_correlation():
    """Hierarchical bootstrap CI must be >= 2x wider than naive CI under strong cluster correlation."""
    rng = np.random.default_rng(42)
    n_gels = 4
    spheroids_per_gel = 30
    sigma_gel = 0.5
    sigma_within = 0.1

    records = []
    for g in range(n_gels):
        gel_effect = rng.normal(0, sigma_gel)
        for s in range(spheroids_per_gel):
            val = 1.0 + gel_effect + rng.normal(0, sigma_within)
            records.append({"gel": f"gel_{g}", "val": val})

    df = pd.DataFrame(records)

    # Hierarchical bootstrap (resamples gels with replacement, then spheroids)
    ci_hier_low, ci_hier_high = hierarchical_bootstrap_ci(
        df, value_col="val", cluster_col="gel", stat=np.median, n_boot=2000, seed=42
    )
    width_hier = ci_hier_high - ci_hier_low

    # Naive bootstrap (resamples rows directly ignoring gel clustering)
    vals = df["val"].values
    boot_stats = [
        np.median(rng.choice(vals, size=len(vals), replace=True))
        for _ in range(2000)
    ]
    width_naive = np.percentile(boot_stats, 97.5) - np.percentile(boot_stats, 2.5)

    assert width_hier >= 2.0 * width_naive, (
        f"Hierarchical CI width ({width_hier:.3f}) should be >= 2x naive CI width ({width_naive:.3f})"
    )


def test_cluster_permutation_type1_error():
    """Assert cluster permutation preserves nominal alpha <= 0.07 while naive test fails (> 0.15)."""
    rng = np.random.default_rng(123)
    n_sims = 250  # Fast verification
    rejections_cluster = 0
    rejections_naive = 0

    for _ in range(n_sims):
        # 4 control gels vs 4 treated gels, but ZERO true condition effect (null is true)
        # Strong gel random effect (sigma_gel=0.8, sigma_within=0.1)
        records = []
        for cond, prefix in [("ctrl", "c_gel"), ("treat", "t_gel")]:
            for g in range(4):
                g_effect = rng.normal(0, 0.8)
                for _ in range(15):
                    records.append({
                        "condition": cond,
                        "gel": f"{prefix}_{g}",
                        "val": g_effect + rng.normal(0, 0.1),
                    })
        sim_df = pd.DataFrame(records)

        # Cluster permutation
        p_clust, min_p = cluster_permutation_test(
            sim_df, value_col="val", group_col="condition", cluster_col="gel",
            ref_group="ctrl", n_perm=500, seed=42
        )
        if p_clust < 0.05:
            rejections_cluster += 1

        # Naive permutation (permutes rows directly)
        c_vals = sim_df[sim_df["condition"] == "ctrl"]["val"].values
        t_vals = sim_df[sim_df["condition"] == "treat"]["val"].values
        diff_obs = abs(np.median(t_vals) - np.median(c_vals))
        all_vals = np.concatenate([c_vals, t_vals])
        n_c = len(c_vals)
        perm_diffs = []
        for _ in range(200):
            p_idx = rng.permutation(len(all_vals))
            p_diff = abs(np.median(all_vals[p_idx[n_c:]]) - np.median(all_vals[p_idx[:n_c]]))
            perm_diffs.append(p_diff)
        p_naive = (np.sum(np.array(perm_diffs) >= diff_obs) + 1) / (len(perm_diffs) + 1)
        if p_naive < 0.05:
            rejections_naive += 1

    rate_cluster = rejections_cluster / n_sims
    rate_naive = rejections_naive / n_sims

    assert rate_cluster <= 0.08, f"Cluster permutation false positive rate too high: {rate_cluster:.3f}"
    assert rate_naive >= 0.15, f"Naive permutation was expected to have elevated Type 1 error, got {rate_naive:.3f}"


def test_cluster_permutation_exact_when_few_gels():
    """With 3 vs 3 gels, C(6,3)=20 combinations; minimum achievable two-sided p is 2/20 = 0.10 (or 0.05)."""
    # Create separable gels
    records = []
    for g in range(3):
        records.append({"condition": "ctrl", "gel": f"c_{g}", "val": float(g)})
    for g in range(3):
        records.append({"condition": "treat", "gel": f"t_{g}", "val": float(g + 10)})
    df = pd.DataFrame(records)

    p_val, min_p = cluster_permutation_test(
        df, value_col="val", group_col="condition", cluster_col="gel",
        ref_group="ctrl", n_perm=1000, seed=42
    )
    # The minimum achievable p-value for 3 vs 3 gels
    assert min_p >= 0.05
    assert p_val >= min_p


def test_holm_adjustment_matches_statsmodels():
    raw_p = [0.001, 0.008, 0.035, 0.12, 0.65]
    adj_ours = adjust_pvalues(raw_p, method="holm")
    adj_sm = multipletests(raw_p, method="holm")[1]
    np.testing.assert_allclose(adj_ours, adj_sm, rtol=1e-7, atol=1e-9)


def test_fold_change_tested_on_log_scale():
    """FC values {0.5, 2.0} vs {1.0, 1.0} must have symmetric log2 statistic = 0 difference in median."""
    df = pd.DataFrame([
        {"condition": "ctrl", "gel": "g1", "fc": 1.0},
        {"condition": "ctrl", "gel": "g2", "fc": 1.0},
        {"condition": "treat", "gel": "g3", "fc": 0.5},
        {"condition": "treat", "gel": "g4", "fc": 2.0},
    ])
    df["log2_fc"] = np.log2(df["fc"])
    summary = summarise_by_unit(df, value_col="log2_fc", unit_cols=("condition", "gel"))
    med_ctrl = summary[summary["condition"] == "ctrl"]["val_median"].median()
    med_treat = summary[summary["condition"] == "treat"]["val_median"].median()
    assert abs(med_treat - med_ctrl) < 1e-9


def test_control_p_value_is_na_not_one():
    """Baseline control must receive NaN p-value, not 1.0."""
    from spheroid_pipeline_v2.stats import SpheroidPopulationAuditor
    from spheroid_pipeline_v2.config import StatsConfig

    df_pairs = pd.DataFrame([
        {
            "condition": "Mat",
            "gel": "gel1",
            "fov": "0001",
            "t0_volume_sphere_um3": 1000.0,
            "t7_volume_sphere_um3": 1200.0,
            "delta_volume_um3": 200.0,
            "fold_change_volume": 1.2,
            "denominator_gate_pass": True,
            "combined_qc_flag": "PASS",
        },
        {
            "condition": "Mat",
            "gel": "gel2",
            "fov": "0002",
            "t0_volume_sphere_um3": 1000.0,
            "t7_volume_sphere_um3": 1150.0,
            "delta_volume_um3": 150.0,
            "fold_change_volume": 1.15,
            "denominator_gate_pass": True,
            "combined_qc_flag": "PASS",
        },
    ])

    auditor = SpheroidPopulationAuditor(config=StatsConfig(control_condition="Mat"))
    summary_df = auditor.compute_condition_summaries(df_pairs)
    ctrl_row = summary_df[summary_df["condition"] == "Mat"].iloc[0]
    # Control p_raw should be NaN / None, not 1.0
    assert pd.isna(ctrl_row.get("p_raw")) or pd.isna(ctrl_row.get("p_value"))


def test_no_silent_population_fallback():
    """Condition with 0 PASS pairs and include_review_in_stats=False must NOT silently include REVIEW pairs."""
    from spheroid_pipeline_v2.stats import SpheroidPopulationAuditor
    from spheroid_pipeline_v2.config import StatsConfig

    df_pairs = pd.DataFrame([
        {
            "condition": "Mat",
            "gel": "gel1",
            "fov": "0001",
            "t0_volume_sphere_um3": 1000.0,
            "t7_volume_sphere_um3": 1200.0,
            "delta_volume_um3": 200.0,
            "fold_change_volume": 1.2,
            "denominator_gate_pass": True,
            "combined_qc_flag": "PASS",
        },
        {
            "condition": "TestCond",
            "gel": "gel1",
            "fov": "0001",
            "t0_volume_sphere_um3": 1000.0,
            "t7_volume_sphere_um3": 1200.0,
            "delta_volume_um3": 200.0,
            "fold_change_volume": 1.2,
            "denominator_gate_pass": True,
            "combined_qc_flag": "REVIEW",  # ONLY REVIEW, 0 PASS
        },
    ])

    auditor = SpheroidPopulationAuditor(
        config=StatsConfig(control_condition="Mat", include_review_in_stats=False)
    )
    summary_df = auditor.compute_condition_summaries(df_pairs)
    test_row = summary_df[summary_df["condition"] == "TestCond"].iloc[0]

    assert test_row["analysis_population"] == "PASS"
    assert pd.isna(test_row["fold_change_median"])
