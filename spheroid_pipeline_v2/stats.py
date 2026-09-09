"""
Statistical Analysis & Multilevel Auditing Module for Spheroid Population Data.

Provides:
  - Robust condition-level summary statistics:
      * Sample sizes: n_images_t0, n_images_t7, n_objects_total, n_matched_pairs,
        n_pass_pairs, n_denominator_excluded, n_review_pairs
      * Central tendency & dispersion: median/IQR for V0, V7, delta_V, fold_change
      * Parametric estimators: mean and sample standard deviation of fold change
  - Denominator gating (d_0 >= 60 um) exclusion from fold change population statistics.
  - Non-parametric bootstrap 95% Confidence Intervals for median fold change (B=2000).
  - Two-sided Monte Carlo exact permutation tests against control baseline (M=10000).
  - Multilevel 10-step exclusion auditing (E1 to E10).
  - Exact 21-column condition_summary.csv schema compliance and persistence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from spheroid_pipeline_v2.config import PipelineConfig, StatsConfig
from spheroid_pipeline_v2.measure import SpheroidObjectRecord
from spheroid_pipeline_v2.pair import SpheroidPairRecord

logger = logging.getLogger("spheroid_pipeline_v2.stats")

# Exact 21-column header sequence mandated by PROJECT.md and spec report
CONDITION_SUMMARY_COLUMNS: List[str] = [
    "condition",
    "n_images_t0",
    "n_images_t7",
    "n_objects_total",
    "n_matched_pairs",
    "n_pass_pairs",
    "n_denominator_excluded",
    "n_review_pairs",
    "v0_median_um3",
    "v0_iqr_um3",
    "v7_median_um3",
    "v7_iqr_um3",
    "delta_v_median_um3",
    "fold_change_median",
    "fold_change_iqr",
    "fold_change_mean",
    "fold_change_std",
    "bootstrap_ci_95_low",
    "bootstrap_ci_95_high",
    "permutation_p_vs_control",
    "significance",
]


def bootstrap_ci_median(
    data: Union[np.ndarray, List[float]],
    n_boot: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Computes non-parametric bootstrap confidence interval for the sample median.

    Args:
        data: 1D array-like sample of numerical values
        n_boot: Number of bootstrap resamples (default 2000)
        confidence_level: Confidence level in range (0, 1) (default 0.95)
        random_seed: Random seed for deterministic reproducibility

    Returns:
        (ci_low, ci_high) percentiles or (NaN, NaN) if empty
    """
    arr = np.asarray(data, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(arr[0]), float(arr[0])

    rng = np.random.RandomState(random_seed)
    boot_medians = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        resample = rng.choice(arr, size=n, replace=True)
        boot_medians[i] = np.median(resample)

    alpha = 1.0 - confidence_level
    p_low = (alpha / 2.0) * 100.0
    p_high = (1.0 - alpha / 2.0) * 100.0

    ci_low = float(np.percentile(boot_medians, p_low))
    ci_high = float(np.percentile(boot_medians, p_high))
    return ci_low, ci_high


def permutation_test_median(
    group: Union[np.ndarray, List[float]],
    control: Union[np.ndarray, List[float]],
    n_perm: int = 10000,
    random_seed: int = 42,
) -> float:
    """
    Two-sample exact Monte Carlo permutation test comparing group median to control median.

    Hypothesis:
        H0: Median(group) == Median(control)
        H1: Median(group) != Median(control)

    Test Statistic:
        T_obs = |median(group) - median(control)|

    Exact p-value formula:
        p = (sum(T_perm >= T_obs) + 1) / (n_perm + 1)
    """
    grp = np.asarray(group, dtype=np.float64)
    grp = grp[~np.isnan(grp)]
    ctrl = np.asarray(control, dtype=np.float64)
    ctrl = ctrl[~np.isnan(ctrl)]

    n_grp = len(grp)
    n_ctrl = len(ctrl)

    if n_grp == 0 or n_ctrl == 0:
        return np.nan

    t_obs = abs(float(np.median(grp)) - float(np.median(ctrl)))
    pooled = np.concatenate([grp, ctrl])
    total_n = len(pooled)

    rng = np.random.RandomState(random_seed)
    perm_diffs = np.empty(n_perm, dtype=np.float64)

    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        perm_grp = shuffled[:n_grp]
        perm_ctrl = shuffled[n_grp:]
        perm_diffs[i] = abs(np.median(perm_grp) - np.median(perm_ctrl))

    count_ge = np.sum(perm_diffs >= (t_obs - 1e-12))
    p_value = (float(count_ge) + 1.0) / (float(n_perm) + 1.0)
    return float(p_value)


def get_significance_stars(p_value: float) -> str:
    """
    Converts p-value to scientific asterisk notation:
      p < 0.0001: '****'
      p < 0.001:  '***'
      p < 0.01:   '**'
      p < 0.05:   '*'
      p >= 0.05:  'ns'
      NaN:        'n/a'
    """
    if p_value is None or np.isnan(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


class PopulationStatsCalculator:
    """
    High-level statistical aggregation and hypothesis testing engine.
    """

    def __init__(self, config: Optional[Union[StatsConfig, PipelineConfig]] = None):
        if isinstance(config, PipelineConfig):
            self.config = config.stats
        elif isinstance(config, StatsConfig):
            self.config = config
        else:
            self.config = StatsConfig()

    def compute_condition_statistics(
        self,
        pairs: Union[List[SpheroidPairRecord], pd.DataFrame],
        all_objects: Optional[Union[List[SpheroidObjectRecord], pd.DataFrame]] = None,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Computes full per-condition summary statistics matching CONDITION_SUMMARY_COLUMNS.
        """
        if isinstance(pairs, list):
            if len(pairs) == 0:
                return pd.DataFrame(columns=CONDITION_SUMMARY_COLUMNS)
            df_pairs = pd.DataFrame([p.to_dict() for p in pairs])
        else:
            df_pairs = pairs.copy()

        if df_pairs.empty:
            return pd.DataFrame(columns=CONDITION_SUMMARY_COLUMNS)

        # Prepare objects dataframe if provided
        df_objects = None
        if all_objects is not None:
            if isinstance(all_objects, list):
                if len(all_objects) > 0:
                    df_objects = pd.DataFrame([o.to_dict() for o in all_objects])
            else:
                df_objects = all_objects.copy()

        ctrl_name = self.config.control_condition

        # Group pairs by condition
        conditions = sorted(df_pairs["condition"].unique())
        if ctrl_name in conditions:
            # Place control condition first
            conditions = [ctrl_name] + [c for c in conditions if c != ctrl_name]

        # Extract baseline control valid fold change distribution
        ctrl_df = df_pairs[df_pairs["condition"] == ctrl_name]
        ctrl_fc = self._extract_valid_fc(ctrl_df)

        rows = []
        for cond in conditions:
            grp_pairs = df_pairs[df_pairs["condition"] == cond]

            # 1. Sample Size Metrics
            if df_objects is not None and not df_objects.empty and "condition" in df_objects.columns:
                cond_objs = df_objects[df_objects["condition"] == cond]
                t0_objs = cond_objs[cond_objs["timepoint"].str.lower().str.contains("0", na=False)]
                t7_objs = cond_objs[~cond_objs["timepoint"].str.lower().str.contains("0", na=False)]
                n_img_t0 = t0_objs["image_id"].nunique() if not t0_objs.empty else grp_pairs["fov"].nunique()
                n_img_t7 = t7_objs["image_id"].nunique() if not t7_objs.empty else grp_pairs["fov"].nunique()
                n_objs_total = len(cond_objs)
            else:
                n_img_t0 = grp_pairs["fov"].nunique()
                n_img_t7 = grp_pairs["fov"].nunique()
                n_objs_total = len(grp_pairs) * 2

            n_matched = len(grp_pairs)
            n_pass = int((grp_pairs["combined_qc_flag"] == "PASS").sum())
            n_denom_excl = int((~grp_pairs["denominator_gate_pass"].astype(bool)).sum())
            n_review = int((grp_pairs["combined_qc_flag"] == "REVIEW").sum())

            # 2. Population Filtering for Quantitative Growth Statistics
            # Strictly enforce Denominator Gate (d0 >= 60 um)
            valid_grp = grp_pairs[grp_pairs["denominator_gate_pass"].astype(bool)]

            if not self.config.include_review_in_stats:
                pass_grp = valid_grp[valid_grp["combined_qc_flag"] == "PASS"]
                if len(pass_grp) > 0:
                    stat_grp = pass_grp
                else:
                    # Fallback to PASS+REVIEW if no strictly PASS pairs available
                    stat_grp = valid_grp[valid_grp["combined_qc_flag"].isin(["PASS", "REVIEW"])]
            else:
                stat_grp = valid_grp[valid_grp["combined_qc_flag"].isin(["PASS", "REVIEW"])]

            # Extract metrics
            v0_vals = stat_grp["t0_volume_sphere_um3"].dropna().values.astype(np.float64)
            v7_vals = stat_grp["t7_volume_sphere_um3"].dropna().values.astype(np.float64)
            dv_vals = stat_grp["delta_volume_um3"].dropna().values.astype(np.float64)
            fc_vals = stat_grp["fold_change_volume"].dropna().values.astype(np.float64)

            n_stat = len(fc_vals)
            if n_stat > 0:
                v0_med = float(np.median(v0_vals))
                v0_iqr = float(np.percentile(v0_vals, 75) - np.percentile(v0_vals, 25)) if len(v0_vals) > 1 else 0.0

                v7_med = float(np.median(v7_vals))
                v7_iqr = float(np.percentile(v7_vals, 75) - np.percentile(v7_vals, 25)) if len(v7_vals) > 1 else 0.0

                dv_med = float(np.median(dv_vals))

                fc_med = float(np.median(fc_vals))
                fc_iqr = float(np.percentile(fc_vals, 75) - np.percentile(fc_vals, 25)) if n_stat > 1 else 0.0
                fc_mean = float(np.mean(fc_vals))
                fc_std = float(np.std(fc_vals, ddof=1)) if n_stat > 1 else 0.0

                # Bootstrap 95% CI on median FC
                ci_low, ci_high = bootstrap_ci_median(
                    fc_vals,
                    n_boot=self.config.bootstrap_iterations,
                    confidence_level=self.config.confidence_level,
                    random_seed=random_seed,
                )
            else:
                v0_med, v0_iqr = np.nan, np.nan
                v7_med, v7_iqr = np.nan, np.nan
                dv_med = np.nan
                fc_med, fc_iqr = np.nan, np.nan
                fc_mean, fc_std = np.nan, np.nan
                ci_low, ci_high = np.nan, np.nan

            # 3. Permutation Test vs Control
            if cond == ctrl_name:
                p_val = 1.0
                signif = "control"
            elif len(ctrl_fc) > 0 and n_stat > 0:
                p_val = permutation_test_median(
                    fc_vals,
                    ctrl_fc,
                    n_perm=self.config.permutation_iterations,
                    random_seed=random_seed,
                )
                signif = get_significance_stars(p_val)
            else:
                p_val = np.nan
                signif = "n/a"

            row = {
                "condition": cond,
                "n_images_t0": int(n_img_t0),
                "n_images_t7": int(n_img_t7),
                "n_objects_total": int(n_objs_total),
                "n_matched_pairs": int(n_matched),
                "n_pass_pairs": int(n_pass),
                "n_denominator_excluded": int(n_denom_excl),
                "n_review_pairs": int(n_review),
                "v0_median_um3": v0_med,
                "v0_iqr_um3": v0_iqr,
                "v7_median_um3": v7_med,
                "v7_iqr_um3": v7_iqr,
                "delta_v_median_um3": dv_med,
                "fold_change_median": fc_med,
                "fold_change_iqr": fc_iqr,
                "fold_change_mean": fc_mean,
                "fold_change_std": fc_std,
                "bootstrap_ci_95_low": ci_low,
                "bootstrap_ci_95_high": ci_high,
                "permutation_p_vs_control": p_val,
                "significance": signif,
            }
            rows.append(row)

        summary_df = pd.DataFrame(rows, columns=CONDITION_SUMMARY_COLUMNS)
        return summary_df

    def _extract_valid_fc(self, df: pd.DataFrame) -> np.ndarray:
        """Extracts valid fold change array for statistical testing."""
        if df.empty:
            return np.array([], dtype=np.float64)

        valid = df[df["denominator_gate_pass"].astype(bool)]
        if not self.config.include_review_in_stats:
            pass_df = valid[valid["combined_qc_flag"] == "PASS"]
            if len(pass_df) > 0:
                return pass_df["fold_change_volume"].dropna().values.astype(np.float64)
        
        return valid[valid["combined_qc_flag"].isin(["PASS", "REVIEW"])]["fold_change_volume"].dropna().values.astype(np.float64)


def compute_condition_statistics(
    pairs: Union[List[SpheroidPairRecord], pd.DataFrame],
    all_objects: Optional[Union[List[SpheroidObjectRecord], pd.DataFrame]] = None,
    config: Optional[Union[StatsConfig, PipelineConfig]] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    High-level functional interface to calculate per-condition summary statistics.
    """
    calc = PopulationStatsCalculator(config=config)
    return calc.compute_condition_statistics(pairs, all_objects=all_objects, random_seed=random_seed)


def compute_exclusion_audit(
    all_objects: Optional[Union[List[SpheroidObjectRecord], pd.DataFrame]] = None,
    all_pairs: Optional[Union[List[SpheroidPairRecord], pd.DataFrame]] = None,
    t0_images_count: int = 0,
    t7_images_count: int = 0,
    plausibility_fallbacks: int = 0,
) -> Dict[str, Any]:
    """
    Computes exact multilevel exclusion tracking metrics across stages E1 to E10.
    """
    # Objects breakdown
    if all_objects is not None:
        if isinstance(all_objects, list):
            df_objs = pd.DataFrame([o.to_dict() for o in all_objects]) if len(all_objects) > 0 else pd.DataFrame()
        else:
            df_objs = all_objects.copy()
    else:
        df_objs = pd.DataFrame()

    # Pairs breakdown
    if all_pairs is not None:
        if isinstance(all_pairs, list):
            df_pairs = pd.DataFrame([p.to_dict() for p in all_pairs]) if len(all_pairs) > 0 else pd.DataFrame()
        else:
            df_pairs = all_pairs.copy()
    else:
        df_pairs = pd.DataFrame()

    total_t0_imgs = t0_images_count
    total_t7_imgs = t7_images_count
    if total_t0_imgs == 0 and not df_objs.empty and "timepoint" in df_objs.columns:
        t0_objs = df_objs[df_objs["timepoint"].str.lower().str.contains("0", na=False)]
        total_t0_imgs = t0_objs["image_id"].nunique()
    if total_t7_imgs == 0 and not df_objs.empty and "timepoint" in df_objs.columns:
        t7_objs = df_objs[~df_objs["timepoint"].str.lower().str.contains("0", na=False)]
        total_t7_imgs = t7_objs["image_id"].nunique()

    # E3-E6 exclusion reasons from objects table
    debris_count = 0
    oversized_count = 0
    border_count = 0
    contrast_count = 0
    unpaired_t0 = 0
    unpaired_t7 = 0

    if not df_objs.empty and "qc_reasons" in df_objs.columns:
        for _, r in df_objs.iterrows():
            reason = str(r.get("qc_reasons", ""))
            flag = str(r.get("qc_flag", ""))
            if "debris" in reason:
                debris_count += 1
            if "oversized" in reason:
                oversized_count += 1
            if "border" in reason:
                border_count += 1
            if "contrast" in reason or "insufficient_dark" in reason:
                contrast_count += 1

            pair_id = str(r.get("pair_id", "UNPAIRED"))
            if pair_id == "UNPAIRED" and flag != "FAIL":
                tp = str(r.get("timepoint", "")).lower()
                if "0" in tp:
                    unpaired_t0 += 1
                else:
                    unpaired_t7 += 1

    # E8-E10 from pairs table
    denom_excl = 0
    extreme_review = 0
    final_pass = 0

    if not df_pairs.empty:
        denom_excl = int((~df_pairs["denominator_gate_pass"].astype(bool)).sum())
        for _, p in df_pairs.iterrows():
            reasons = str(p.get("qc_reasons", ""))
            if "extreme_shrinkage" in reasons or "extreme_expansion" in reasons:
                extreme_review += 1
        final_pass = int((df_pairs["combined_qc_flag"] == "PASS").sum())

    return {
        "E1_ingestion_images_t0": total_t0_imgs,
        "E1_ingestion_images_t7": total_t7_imgs,
        "E1_ingestion_images_total": total_t0_imgs + total_t7_imgs,
        "E2_plausibility_fallbacks": plausibility_fallbacks,
        "E3_debris_undersized": debris_count,
        "E4_oversized_artifact": oversized_count,
        "E5_border_touching": border_count,
        "E6_insufficient_contrast": contrast_count,
        "E7_unpaired_t0": unpaired_t0,
        "E7_unpaired_t7": unpaired_t7,
        "E7_unpaired_total": unpaired_t0 + unpaired_t7,
        "E8_denominator_excluded": denom_excl,
        "E9_extreme_trajectory_review": extreme_review,
        "E10_final_pass_pairs": final_pass,
    }


def generate_exclusion_audit_table(audit_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Generates a structured 10-step exclusion auditing DataFrame.
    """
    stages = [
        {"Stage": "E1", "Filter Description": "Raw Image FOVs Ingested (t0 / t7 / Total)", "Count": f"{audit_metrics.get('E1_ingestion_images_t0', 0)} / {audit_metrics.get('E1_ingestion_images_t7', 0)} / {audit_metrics.get('E1_ingestion_images_total', 0)}", "Impact": "Baseline Scope"},
        {"Stage": "E2", "Filter Description": "FOV Plausibility Fallbacks (>25% FOV / 0 masks)", "Count": str(audit_metrics.get('E2_plausibility_fallbacks', 0)), "Impact": "Backend Switching"},
        {"Stage": "E3", "Filter Description": "Size Gate: Undersized Cellular Debris (d < 40 um)", "Count": str(audit_metrics.get('E3_debris_undersized', 0)), "Impact": "Object Discard"},
        {"Stage": "E4", "Filter Description": "Size Gate: Oversized Gel Artifacts (d > 1500 um)", "Count": str(audit_metrics.get('E4_oversized_artifact', 0)), "Impact": "Object Discard"},
        {"Stage": "E5", "Filter Description": "Border Clearance: Boundary-Touching Instances (margin < 15px)", "Count": str(audit_metrics.get('E5_border_touching', 0)), "Impact": "Object Discard"},
        {"Stage": "E6", "Filter Description": "Contrast Gate: Low-Contrast / Refractive Bubbles (contrast < 10%)", "Count": str(audit_metrics.get('E6_insufficient_contrast', 0)), "Impact": "Object Discard"},
        {"Stage": "E7", "Filter Description": "Tracking Displacement: Unpaired Objects (t0 / t7 / Total)", "Count": f"{audit_metrics.get('E7_unpaired_t0', 0)} / {audit_metrics.get('E7_unpaired_t7', 0)} / {audit_metrics.get('E7_unpaired_total', 0)}", "Impact": "Track Severed"},
        {"Stage": "E8", "Filter Description": "Denominator Gate: Excluded from Fold Change (d_0 < 60 um)", "Count": str(audit_metrics.get('E8_denominator_excluded', 0)), "Impact": "Excluded from Stats"},
        {"Stage": "E9", "Filter Description": "Trajectory QC: Extreme Growth/Shrinkage Flagged REVIEW (FC not in [0.1, 20])", "Count": str(audit_metrics.get('E9_extreme_trajectory_review', 0)), "Impact": "Flagged for Audit"},
        {"Stage": "E10", "Filter Description": "Final Primary Population: Valid PASS Pairs for Hypothesis Testing", "Count": str(audit_metrics.get('E10_final_pass_pairs', 0)), "Impact": "Hypothesis Testing"},
    ]
    return pd.DataFrame(stages)


def generate_exclusion_audit_markdown(audit_df: pd.DataFrame) -> str:
    """
    Formats exclusion audit DataFrame into GitHub-Flavored Markdown table.
    """
    headers = "| " + " | ".join(audit_df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(audit_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in audit_df.itertuples(index=False)
    ]
    return "\n".join([headers, divider] + rows)


def format_condition_summary_table(summary_df: pd.DataFrame, control_condition: str = "Mat") -> str:
    """
    Formats population statistics summary into a human-readable console string.
    """
    lines = []
    lines.append("=" * 105)
    lines.append("SPHEROID VOLUME POPULATION GROWTH SUMMARY (Day 0 -> Day 7)")
    lines.append(f"Baseline Control Condition: {control_condition}")
    lines.append("=" * 105)
    header = (
        f"{'Condition':<12} | {'Pairs (Pass)':<14} | {'Med V0 (um3)':<14} | "
        f"{'Med V7 (um3)':<14} | {'Med FC (V7/V0)':<16} | {'95% CI':<16} | {'p-val vs Ctrl':<14}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for _, r in summary_df.iterrows():
        cond = str(r["condition"])
        n_str = f"{r['n_matched_pairs']} ({r['n_pass_pairs']})"
        v0_str = f"{r['v0_median_um3']:.1f}" if not np.isnan(r['v0_median_um3']) else "N/A"
        v7_str = f"{r['v7_median_um3']:.1f}" if not np.isnan(r['v7_median_um3']) else "N/A"
        fc_str = f"{r['fold_change_median']:.3f}" if not np.isnan(r['fold_change_median']) else "N/A"
        
        if not np.isnan(r['bootstrap_ci_95_low']) and not np.isnan(r['bootstrap_ci_95_high']):
            ci_str = f"[{r['bootstrap_ci_95_low']:.2f}, {r['bootstrap_ci_95_high']:.2f}]"
        else:
            ci_str = "N/A"

        if not np.isnan(r['permutation_p_vs_control']):
            p_val = r['permutation_p_vs_control']
            sig = r['significance']
            if sig == "control":
                p_str = "Control (1.00)"
            else:
                p_str = f"{p_val:.4f} ({sig})"
        else:
            p_str = "N/A"

        line = (
            f"{cond:<12} | {n_str:<14} | {v0_str:<14} | "
            f"{v7_str:<14} | {fc_str:<16} | {ci_str:<16} | {p_str:<14}"
        )
        lines.append(line)

    lines.append("=" * 105)
    return "\n".join(lines)


def save_condition_summary_csv(
    summary_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Saves condition summary DataFrame to CSV adhering to the exact 21-column schema.
    Returns the created DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all 21 columns exist
    for col in CONDITION_SUMMARY_COLUMNS:
        if col not in summary_df.columns:
            summary_df[col] = np.nan

    ordered_df = summary_df[CONDITION_SUMMARY_COLUMNS]
    ordered_df.to_csv(output_path, index=False)
    logger.debug(f"Saved condition summary ({len(ordered_df)} conditions) to {output_path}")
    return ordered_df
