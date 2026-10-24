import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from matrix_mat.core.stats import (
    adjust_pvalues,
    cluster_permutation_test,
    hierarchical_bootstrap_ci,
    paired_volume_change_test,
    summarise_by_unit,
)


def test_bh_adjustment_matches_statsmodels():
    raw = [0.001, 0.02, 0.03, 0.2, 0.5]
    np.testing.assert_allclose(adjust_pvalues(raw, method="bh"), multipletests(raw, method="fdr_bh")[1])


def test_adjustment_keeps_nan_positions_and_ignores_them():
    raw = [0.01, np.nan, 0.04]
    out = adjust_pvalues(raw, method="holm")
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], multipletests([0.01, 0.04], method="holm")[1])


def test_adjustment_unknown_method_raises():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 0.2], method="bonkers")


def test_monte_carlo_branch_for_many_gels_is_seeded_and_valid():
    rng = np.random.default_rng(0)
    rows = [{"condition": c, "gel": f"{c}{g}", "val": rng.normal(1.0 if c == "t" else 0.0, 0.3)}
            for c in ("c", "t") for g in range(9)]  # C(18,9)=48620 > 2000 -> Monte Carlo
    df = pd.DataFrame(rows)
    p1, minp = cluster_permutation_test(df, "val", "condition", "gel", ref_group="c", n_perm=2000, seed=1)
    p2, _ = cluster_permutation_test(df, "val", "condition", "gel", ref_group="c", n_perm=2000, seed=1)
    assert p1 == p2
    assert 0 < p1 < 0.01
    assert minp > 0


def test_permutation_missing_groups_return_nan():
    df = pd.DataFrame({"condition": ["c", "c"], "gel": ["a", "b"], "val": [1.0, 2.0]})
    assert all(np.isnan(cluster_permutation_test(df, "val", "condition", "gel", ref_group="x")))
    assert all(np.isnan(cluster_permutation_test(df, "val", "condition", "gel", ref_group="c")))


def test_bootstrap_edge_cases():
    empty = pd.DataFrame({"gel": [], "v": []})
    assert all(np.isnan(hierarchical_bootstrap_ci(empty, "v", "gel")))
    single = pd.DataFrame({"gel": ["a", "a", "a"], "v": [1.0, 1.0, 1.0]})
    assert hierarchical_bootstrap_ci(single, "v", "gel", n_boot=100) == (1.0, 1.0)


def test_summarise_by_unit_empty_and_mean():
    assert list(summarise_by_unit(pd.DataFrame(columns=["condition", "gel", "v"]), "v").columns) == ["condition", "gel", "val_median", "n_obs"]
    df = pd.DataFrame({"condition": ["a"] * 3, "gel": ["g"] * 3, "v": [1.0, 2.0, 6.0]})
    assert summarise_by_unit(df, "v", agg="mean")["val_median"].iloc[0] == pytest.approx(3.0)


def test_paired_test_edge_cases():
    assert np.isnan(paired_volume_change_test([1.0, 2.0], [1.0, 2.0]))
    assert paired_volume_change_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
