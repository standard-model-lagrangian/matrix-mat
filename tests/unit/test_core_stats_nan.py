"""NaN / small-n guards in core.stats (0.2.2)."""

import numpy as np
import pandas as pd

from matrix_mat.core.stats import (
    cluster_permutation_test,
    compute_mwu_pvalue,
    hierarchical_bootstrap_ci,
    paired_volume_change_test,
)


def _df(ref_vals, treat_vals):
    rows = []
    for cond, gels in (("Mat", ref_vals), ("S50", treat_vals)):
        for gi, vals in enumerate(gels):
            for v in vals:
                rows.append({"condition": cond, "gel": f"{cond}_g{gi}", "val": v})
    return pd.DataFrame(rows)


def test_all_nan_cluster_is_dropped_not_p_zero():
    # Without the fix the NaN cluster median makes every |diff| comparison False -> p = 0.
    df = _df([[1.0, 1.1], [1.2, 1.3], [np.nan, np.nan]], [[2.0, 2.1], [2.2, 2.3], [2.4, 2.5]])
    p, minp = cluster_permutation_test(df, "val", "condition", "gel", ref_group="Mat")
    assert np.isfinite(p) and p > 0
    clean = _df([[1.0, 1.1], [1.2, 1.3]], [[2.0, 2.1], [2.2, 2.3], [2.4, 2.5]])
    assert (p, minp) == cluster_permutation_test(clean, "val", "condition", "gel", ref_group="Mat")


def test_fewer_than_two_clusters_per_group_gives_nan():
    df = _df([[1.0], [np.nan]], [[2.0], [3.0], [4.0]])
    assert all(np.isnan(cluster_permutation_test(df, "val", "condition", "gel", ref_group="Mat")))
    df1 = _df([[1.0], [1.5]], [[2.0]])
    assert all(np.isnan(cluster_permutation_test(df1, "val", "condition", "gel", ref_group="Mat")))


def test_nan_and_inf_observations_ignored_within_cluster():
    a = _df([[1.0, np.nan], [1.2, 1.3]], [[2.0, np.inf], [2.2, 2.3]])
    b = _df([[1.0], [1.2, 1.3]], [[2.0], [2.2, 2.3]])
    assert cluster_permutation_test(a, "val", "condition", "gel") == cluster_permutation_test(b, "val", "condition", "gel")


def test_zero_variance_groups_give_p_one():
    df = _df([[1.0], [1.0]], [[1.0], [1.0]])
    p, _ = cluster_permutation_test(df, "val", "condition", "gel")
    assert p == 1.0


def test_mwu_drops_nan_and_handles_short_inputs():
    p, s = compute_mwu_pvalue(np.array([1.0, np.nan]), np.array([2.0, 3.0]))
    assert np.isnan(p) and s == "ns"
    p2, _ = compute_mwu_pvalue(np.array([1.0, 2.0, np.nan, 3.0]), np.array([10.0, 11.0, 12.0]))
    p3, _ = compute_mwu_pvalue(np.array([1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0]))
    assert p2 == p3


def test_paired_test_ignores_inf():
    v0 = [1.0, 2.0, 3.0, 4.0, np.inf]
    v7 = [2.0, 3.0, 4.0, 5.5, 1.0]
    assert paired_volume_change_test(v0, v7) == paired_volume_change_test(v0[:4], v7[:4])


def test_bootstrap_all_nan_returns_nan():
    df = pd.DataFrame({"gel": ["a", "b"], "v": [np.nan, np.nan]})
    assert all(np.isnan(hierarchical_bootstrap_ci(df, "v", "gel", n_boot=10)))
