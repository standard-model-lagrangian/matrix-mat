import numpy as np
import pytest
from scipy import stats

from spheroid_pipeline_v2.stats import paired_volume_change_test


def test_paired_day0_day7_uses_signed_rank():
    """Documents power/correctness difference between paired Wilcoxon signed-rank vs unpaired Mann-Whitney."""
    rng = np.random.default_rng(42)
    # 10 spheroids with large variation in initial size (v0 from 1000 to 10000)
    v0 = rng.uniform(1000, 10000, size=10)
    # Every spheroid grows by ~10% (consistent paired growth)
    v7 = v0 * 1.10 + rng.normal(0, 5, size=10)

    # Unpaired MWU test: loses power due to large between-spheroid variance
    mwu_stat, mwu_p = stats.mannwhitneyu(v7, v0, alternative="two-sided")
    assert mwu_p > 0.30, f"Unpaired test should fail to detect paired growth, p={mwu_p:.3f}"

    # Paired signed-rank test
    p_val = paired_volume_change_test(v0, v7)
    assert p_val < 0.01, f"Paired signed-rank test should detect growth, p={p_val:.4f}"
