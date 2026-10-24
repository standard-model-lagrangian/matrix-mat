"""Hierarchical and clustered statistical methods for spheroid population analysis.

Resolves pseudo-replication by:
1. Treating biological unit (gel/well) as the clustering unit of analysis.
2. Two-stage hierarchical bootstrap resampling (cluster then observation).
3. Cluster-level label permutation testing with exact small-sample enumeration.
4. Multiplicity adjustment (Holm-Bonferroni / Benjamini-Hochberg).
5. Paired two-sided Wilcoxon signed-rank testing.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def summarise_by_unit(
    df: pd.DataFrame,
    value_col: str,
    unit_cols: Sequence[str] = ("condition", "gel"),
    agg: Union[str, Callable] = "median",
) -> pd.DataFrame:
    """Aggregates observation-level metrics to the cluster/unit level."""
    if df.empty:
        cols = list(unit_cols) + ["val_median", "n_obs"]
        return pd.DataFrame(columns=cols)

    agg_func = np.median if agg == "median" else (np.mean if agg == "mean" else agg)

    grouped = df.groupby(list(unit_cols))[value_col].agg(
        val_median=agg_func,
        n_obs="count",
    ).reset_index()
    return grouped


def hierarchical_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "gel",
    stat: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 2000,
    confidence_level: float = 0.95,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """Computes two-stage hierarchical bootstrap confidence interval.

    Stage 1: Resample clusters (gels) with replacement.
    Stage 2: Within each selected cluster, resample observations with replacement.
    """
    if df.empty or value_col not in df.columns:
        return np.nan, np.nan

    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    if n_clusters == 0:
        return np.nan, np.nan

    # Group data by cluster for fast lookup
    cluster_data = {c: df[df[cluster_col] == c][value_col].dropna().values for c in clusters}
    # Filter out empty clusters
    cluster_data = {c: vals for c, vals in cluster_data.items() if len(vals) > 0}
    active_clusters = list(cluster_data.keys())
    if not active_clusters:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boot_estimates = []

    for _ in range(n_boot):
        chosen_clusters = rng.choice(active_clusters, size=len(active_clusters), replace=True)
        resampled_vals = []
        for c in chosen_clusters:
            vals = cluster_data[c]
            n_obs = len(vals)
            obs_sample = rng.choice(vals, size=n_obs, replace=True)
            resampled_vals.extend(obs_sample)

        if resampled_vals:
            boot_estimates.append(float(stat(np.array(resampled_vals))))

    if not boot_estimates:
        return np.nan, np.nan

    alpha = 1.0 - confidence_level
    low_pct = 100.0 * (alpha / 2.0)
    high_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_low = float(np.percentile(boot_estimates, low_pct))
    ci_high = float(np.percentile(boot_estimates, high_pct))
    return ci_low, ci_high


def cluster_permutation_test(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    cluster_col: str = "gel",
    ref_group: str = "Mat",
    target_group: Optional[str] = None,
    n_perm: int = 5000,
    seed: Optional[int] = 42,
) -> Tuple[float, float]:
    """Permutes condition assignments at the cluster (gel) level.

    Returns:
        (p_value, min_achievable_p)
    """
    groups = df[group_col].dropna().unique()
    if ref_group not in groups:
        return np.nan, np.nan

    non_ref_groups = [g for g in groups if g != ref_group]
    if target_group is not None:
        if target_group not in non_ref_groups:
            return np.nan, np.nan
        chosen_target = target_group
    else:
        if len(non_ref_groups) == 0:
            return np.nan, np.nan
        chosen_target = non_ref_groups[0]

    # Aggregate to cluster-level medians first
    cluster_df = summarise_by_unit(df, value_col, unit_cols=(group_col, cluster_col), agg="median")
    ref_clusters = cluster_df[cluster_df[group_col] == ref_group]
    treat_clusters = cluster_df[cluster_df[group_col] == chosen_target]

    n_ref = len(ref_clusters)
    n_treat = len(treat_clusters)
    n_total = n_ref + n_treat

    if n_ref == 0 or n_treat == 0:
        return np.nan, np.nan

    obs_ref = np.median(ref_clusters["val_median"].values)
    obs_treat = np.median(treat_clusters["val_median"].values)
    obs_diff = abs(obs_treat - obs_ref)

    all_vals = np.concatenate([ref_clusters["val_median"].values, treat_clusters["val_median"].values])

    # Check if exact enumeration is small
    n_comb = math.comb(n_total, n_ref)
    min_achievable_p = 1.0 / n_comb

    if n_comb <= 2000:
        # Enumerate exact splits
        indices = range(n_total)
        diffs = []
        for ref_idx in itertools.combinations(indices, n_ref):
            ref_idx_set = set(ref_idx)
            treat_idx = [i for i in indices if i not in ref_idx_set]
            diff = abs(np.median(all_vals[treat_idx]) - np.median(all_vals[list(ref_idx)]))
            diffs.append(diff)
        diffs_arr = np.array(diffs)
        # Exact two-sided p-value
        p_val = float(np.mean(diffs_arr >= obs_diff - 1e-12))
        return max(p_val, min_achievable_p), min_achievable_p
    else:
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(n_perm):
            shuffled = rng.permutation(all_vals)
            diff = abs(np.median(shuffled[n_ref:]) - np.median(shuffled[:n_ref]))
            diffs.append(diff)
        diffs_arr = np.array(diffs)
        p_val = float((np.sum(diffs_arr >= obs_diff - 1e-12) + 1) / (n_perm + 1))
        return max(p_val, min_achievable_p), min_achievable_p


def adjust_pvalues(
    p_values: Sequence[float],
    method: str = "holm",
) -> np.ndarray:
    """Adjusts p-values for multiple comparisons (Holm-Bonferroni or Benjamini-Hochberg)."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m <= 1:
        return p_arr

    nan_mask = np.isnan(p_arr)
    valid_idx = np.where(~nan_mask)[0]
    m_valid = len(valid_idx)
    if m_valid <= 1:
        return p_arr

    valid_p = p_arr[valid_idx]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    adjusted = np.zeros(m_valid, dtype=float)
    if method.lower() == "holm":
        # Holm-Bonferroni step-down: p_adj_i = min(1, max_{k <= i} (m - k + 1) * p_k)
        running_max = 0.0
        for i, p in enumerate(sorted_p):
            rank = i + 1
            cand = (m_valid - rank + 1) * p
            running_max = max(running_max, cand)
            adjusted[i] = min(running_max, 1.0)
    elif method.lower() in ("bh", "fdr_bh"):
        # Benjamini-Hochberg step-up
        running_min = 1.0
        for i in range(m_valid - 1, -1, -1):
            rank = i + 1
            cand = (m_valid / rank) * sorted_p[i]
            running_min = min(running_min, cand)
            adjusted[i] = min(running_min, 1.0)
    else:
        raise ValueError(f"Unknown adjustment method: {method}")

    # Revert to original order
    reverted_valid = np.zeros(m_valid, dtype=float)
    reverted_valid[order] = adjusted

    out = np.full(m, np.nan, dtype=float)
    out[valid_idx] = reverted_valid
    return out


def paired_volume_change_test(
    v0: Union[np.ndarray, Sequence[float]],
    v7: Union[np.ndarray, Sequence[float]],
) -> float:
    """Performs paired two-sided Wilcoxon signed-rank test on Day 0 vs Day 7 paired volumes."""
    from scipy import stats
    v0_arr = np.asarray(v0, dtype=float)
    v7_arr = np.asarray(v7, dtype=float)
    valid = (~np.isnan(v0_arr)) & (~np.isnan(v7_arr))
    v0_arr, v7_arr = v0_arr[valid], v7_arr[valid]
    if len(v0_arr) < 3:
        return np.nan
    diff = v7_arr - v0_arr
    if np.all(diff == 0):
        return 1.0
    try:
        res = stats.wilcoxon(v7_arr, v0_arr, alternative="two-sided", zero_method="wilcox")
        return float(res.pvalue)
    except Exception:
        return np.nan


def compute_mwu_pvalue(vals: np.ndarray, ctrl_data: np.ndarray) -> Tuple[float, str]:
    """Computes Mann-Whitney U test p-value and significance stars vs control."""
    from scipy import stats
    if len(vals) < 2 or len(ctrl_data) < 2:
        return float("nan"), "ns"
    try:
        _, p_val = stats.mannwhitneyu(vals, ctrl_data, alternative="two-sided")
        p_val_flt = float(p_val)
        sig_str = "***" if p_val_flt < 0.001 else ("**" if p_val_flt < 0.01 else ("*" if p_val_flt < 0.05 else "ns"))
        return p_val_flt, sig_str
    except Exception:
        return float("nan"), "ns"

