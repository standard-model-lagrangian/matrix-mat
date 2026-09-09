"""
Comprehensive Milestone 3 Test Suite: Pairing, Denominator Gating & Statistical Hygiene.

Tests:
  1. Hungarian Bipartite Matching (pair.py):
     - Optimal centroid distance assignment
     - Unequal object counts (N0 != N7)
     - Radius-scaled displacement tolerance rejection
     - Empty input handling and boundary robustness
     - Multi-well dataset pairing and pair_id assignment
  2. Trajectory Metrics & Denominator Gating:
     - Delta V, Fold change V, log2(FC), Delta d, Fold change d
     - Denominator gate boundary at 60.0 um
     - Combined QC flag logic (FAIL, REVIEW, PASS, extreme FC)
  3. Condition-Level Statistics (stats.py):
     - 21-column condition_summary.csv schema compliance
     - Denominator gate exclusion from population fold change stats
     - Median, IQR, Mean, and Sample Std calculations
     - Small sample size (N=0, N=1, N=2) edge case resilience
  4. Non-Parametric Bootstrap 95% Confidence Intervals:
     - Empirical percentile method coverage and deterministic seeding
     - Empty array and single sample edge cases
  5. Monte Carlo Exact Permutation Hypothesis Testing:
     - Identical distributions (p > 0.05, 'ns')
     - Different distributions (p < 0.01, '**' or '***')
     - Control condition vs itself (p = 1.0, 'control')
     - Missing / empty group handling (NaN, 'n/a')
     - Asterisk significance mapping
  6. Multilevel Exclusion Auditing:
     - 10-step stage tracking (E1 to E10)
     - Markdown exclusion table formatting
  7. CSV Persistence:
     - pairs.csv exact 30-column export
     - condition_summary.csv exact 21-column export
"""

import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import List
import unittest
import numpy as np
import pandas as pd

from spheroid_pipeline_v2.config import PairingConfig, PipelineConfig, StatsConfig
from spheroid_pipeline_v2.measure import SpheroidObjectRecord
from spheroid_pipeline_v2.pair import (
    PAIRS_CSV_COLUMNS,
    SpheroidPairRecord,
    SpheroidPairer,
    pair_dataset,
    pair_fov_objects,
    pair_timepoints,
    save_pairs_csv,
    update_objects_with_pair_ids,
)
from spheroid_pipeline_v2.stats import (
    CONDITION_SUMMARY_COLUMNS,
    PopulationStatsCalculator,
    bootstrap_ci_median,
    compute_condition_statistics,
    compute_exclusion_audit,
    format_condition_summary_table,
    generate_exclusion_audit_markdown,
    generate_exclusion_audit_table,
    get_significance_stars,
    permutation_test_median,
    save_condition_summary_csv,
)


def _make_dummy_object(
    object_id: str,
    image_id: str = "Mat_gel1_0001_t0",
    timepoint: str = "t0",
    condition: str = "Mat",
    replicate: str = "gel1",
    fov: str = "0001",
    cx: float = 100.0,
    cy: float = 100.0,
    diameter_um: float = 100.0,
    pixel_size_um: float = 1.5,
    qc_flag: str = "PASS",
    qc_reasons: str = "None",
    pair_id: str = "UNPAIRED",
) -> SpheroidObjectRecord:
    """Helper to instantiate a valid SpheroidObjectRecord for testing."""
    d_px = diameter_um / pixel_size_um
    area_px = math.pi * ((d_px / 2.0) ** 2)
    area_um2 = area_px * (pixel_size_um ** 2)
    vol_sphere = (math.pi / 6.0) * (diameter_um ** 3)

    return SpheroidObjectRecord(
        object_id=object_id,
        image_id=image_id,
        timepoint=timepoint,
        condition=condition,
        replicate=replicate,
        fov=fov,
        pair_key=f"{condition}_{replicate}_{fov}",
        label_idx=1,
        pixel_size_um=pixel_size_um,
        area_px=area_px,
        perimeter_px=math.pi * d_px,
        equivalent_diameter_px=d_px,
        centroid_y=cy,
        centroid_x=cx,
        area_um2=area_um2,
        equivalent_diameter_um=diameter_um,
        major_axis_um=diameter_um,
        minor_axis_um=diameter_um,
        aspect_ratio=1.0,
        volume_sphere_um3=vol_sphere,
        volume_ellipsoid_um3=vol_sphere,
        volume_discrepancy_ratio=0.0,
        circularity=0.95,
        solidity=0.98,
        eccentricity=0.1,
        mean_interior_intensity=0.4,
        mean_background_ring_intensity=0.8,
        contrast_ratio=0.5,
        qc_flag=qc_flag,
        qc_reasons=qc_reasons,
        pair_id=pair_id,
        mask_source="automated",
    )


class TestHungarianPairing(unittest.TestCase):
    """Unit and functional tests for bipartite Hungarian centroid matching."""

    def test_optimal_bipartite_matching_multiple_objects(self):
        """Matches 3 spheroids with known minimum Euclidean displacements."""
        t0_objs = [
            _make_dummy_object("obj0_1", cx=100.0, cy=100.0, diameter_um=150.0),
            _make_dummy_object("obj0_2", cx=300.0, cy=300.0, diameter_um=150.0),
            _make_dummy_object("obj0_3", cx=500.0, cy=500.0, diameter_um=150.0),
        ]
        t7_objs = [
            _make_dummy_object("obj7_2", timepoint="t7", cx=305.0, cy=298.0, diameter_um=180.0),
            _make_dummy_object("obj7_3", timepoint="t7", cx=498.0, cy=502.0, diameter_um=180.0),
            _make_dummy_object("obj7_1", timepoint="t7", cx=102.0, cy=104.0, diameter_um=180.0),
        ]

        pairs, un0, un7 = pair_timepoints(t0_objs, t7_objs)

        self.assertEqual(len(pairs), 3)
        self.assertEqual(len(un0), 0)
        self.assertEqual(len(un7), 0)

        # Check matched associations
        matched_map = {p.t0_object_id: p.t7_object_id for p in pairs}
        self.assertEqual(matched_map["obj0_1"], "obj7_1")
        self.assertEqual(matched_map["obj0_2"], "obj7_2")
        self.assertEqual(matched_map["obj0_3"], "obj7_3")

    def test_unequal_counts_t0_greater_than_t7(self):
        """When N0 > N7, matches N7 objects and leaves remaining N0 - N7 objects unpaired."""
        t0_objs = [
            _make_dummy_object("obj0_1", cx=100.0, cy=100.0),
            _make_dummy_object("obj0_2", cx=200.0, cy=200.0),
            _make_dummy_object("obj0_3", cx=300.0, cy=300.0),
        ]
        t7_objs = [
            _make_dummy_object("obj7_1", timepoint="t7", cx=102.0, cy=102.0),
        ]

        pairs, un0, un7 = pair_timepoints(t0_objs, t7_objs)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(un0), 2)
        self.assertEqual(len(un7), 0)
        self.assertEqual(pairs[0].t0_object_id, "obj0_1")
        self.assertEqual(pairs[0].t7_object_id, "obj7_1")
        self.assertEqual({o.object_id for o in un0}, {"obj0_2", "obj0_3"})

    def test_unequal_counts_t7_greater_than_t0(self):
        """When N7 > N0, matches N0 objects and leaves newly appeared t7 objects unpaired."""
        t0_objs = [
            _make_dummy_object("obj0_1", cx=100.0, cy=100.0),
        ]
        t7_objs = [
            _make_dummy_object("obj7_1", timepoint="t7", cx=101.0, cy=101.0),
            _make_dummy_object("obj7_2", timepoint="t7", cx=400.0, cy=400.0),
            _make_dummy_object("obj7_3", timepoint="t7", cx=600.0, cy=600.0),
        ]

        pairs, un0, un7 = pair_timepoints(t0_objs, t7_objs)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(un0), 0)
        self.assertEqual(len(un7), 2)
        self.assertEqual({o.object_id for o in un7}, {"obj7_2", "obj7_3"})

    def test_displacement_threshold_rejection(self):
        """Objects whose centroid distance exceeds max displacement tolerance remain unpaired."""
        # Mean radius = 20 px => max_dist = max(50, 1.5 * 20) = 50 px. Distance = 150 px.
        t0_objs = [_make_dummy_object("obj0_1", cx=100.0, cy=100.0, diameter_um=60.0, pixel_size_um=1.5)]
        t7_objs = [_make_dummy_object("obj7_1", timepoint="t7", cx=250.0, cy=100.0, diameter_um=60.0, pixel_size_um=1.5)]

        cfg = PairingConfig(max_displacement_px_floor=50.0, max_centroid_shift_fraction=1.5)
        pairs, un0, un7 = pair_timepoints(t0_objs, t7_objs, config=cfg)

        self.assertEqual(len(pairs), 0)
        self.assertEqual(len(un0), 1)
        self.assertEqual(len(un7), 1)

    def test_radius_scaled_displacement_tolerance(self):
        """Large spheroids have scaled displacement tolerance > 50 px floor."""
        # Diameter = 300 um (200 px) => mean_r = 100 px => max_dist = 1.5 * 100 = 150 px.
        # Shift = 120 px (< 150 px) should be accepted.
        t0_objs = [_make_dummy_object("obj0_1", cx=200.0, cy=200.0, diameter_um=300.0, pixel_size_um=1.5)]
        t7_objs = [_make_dummy_object("obj7_1", timepoint="t7", cx=320.0, cy=200.0, diameter_um=300.0, pixel_size_um=1.5)]

        cfg = PairingConfig(max_displacement_px_floor=50.0, max_centroid_shift_fraction=1.5)
        pairs, un0, un7 = pair_timepoints(t0_objs, t7_objs, config=cfg)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].centroid_distance_px, 120.0)

    def test_empty_inputs_handling(self):
        """Empty t0 or t7 lists return empty pairs list without raising error."""
        t0_objs = [_make_dummy_object("obj0_1")]
        pairs1, un0_1, un7_1 = pair_timepoints([], t0_objs)
        self.assertEqual(pairs1, [])
        self.assertEqual(len(un7_1), 1)

        pairs2, un0_2, un7_2 = pair_timepoints(t0_objs, [])
        self.assertEqual(pairs2, [])
        self.assertEqual(len(un0_2), 1)

        pairs3, un0_3, un7_3 = pair_timepoints([], [])
        self.assertEqual(pairs3, [])
        self.assertEqual(un0_3, [])
        self.assertEqual(un7_3, [])

    def test_dataset_level_pairing_and_id_assignment(self):
        """Pairing across an entire dataset groups by pair_key and updates pair_id in object records."""
        all_objs = [
            # Well 1 (Mat_gel1_0001)
            _make_dummy_object("Mat_1_t0_1", fov="0001", timepoint="t0", cx=100.0, cy=100.0),
            _make_dummy_object("Mat_1_t7_1", fov="0001", timepoint="t7", cx=102.0, cy=102.0),
            # Well 2 (Mat_gel1_0002)
            _make_dummy_object("Mat_2_t0_1", fov="0002", timepoint="t0", cx=200.0, cy=200.0),
            _make_dummy_object("Mat_2_t7_1", fov="0002", timepoint="t7", cx=204.0, cy=204.0),
            # Unpaired singleton in Well 2
            _make_dummy_object("Mat_2_t0_2", fov="0002", timepoint="t0", cx=800.0, cy=800.0),
        ]

        pairs, updated_objs = pair_dataset(all_objs)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(updated_objs), 5)

        obj_map = {o.object_id: o.pair_id for o in updated_objs}
        self.assertNotEqual(obj_map["Mat_1_t0_1"], "UNPAIRED")
        self.assertEqual(obj_map["Mat_1_t0_1"], obj_map["Mat_1_t7_1"])
        self.assertNotEqual(obj_map["Mat_2_t0_1"], "UNPAIRED")
        self.assertEqual(obj_map["Mat_2_t0_1"], obj_map["Mat_2_t7_1"])
        self.assertEqual(obj_map["Mat_2_t0_2"], "UNPAIRED")


class TestTrajectoryAndDenominatorGating(unittest.TestCase):
    """Tests for growth trajectories, denominator gating, and composite QC flags."""

    def test_trajectory_metrics_accuracy(self):
        """Computes delta_V, fold_change, log2_fold_change, delta_d, fold_change_d accurately."""
        # Initial d0 = 100 um => V0 = pi/6 * 100^3 = 523598.7756 um3
        # Final d7 = 200 um   => V7 = pi/6 * 200^3 = 4188790.2048 um3 (8x volume)
        t0 = _make_dummy_object("t0", cx=100.0, cy=100.0, diameter_um=100.0)
        t7 = _make_dummy_object("t7", timepoint="t7", cx=105.0, cy=105.0, diameter_um=200.0)

        pairs, _, _ = pair_timepoints([t0], [t7])
        self.assertEqual(len(pairs), 1)
        p = pairs[0]

        self.assertAlmostEqual(p.fold_change_volume, 8.0, places=4)
        self.assertAlmostEqual(p.log2_fold_change, 3.0, places=4)
        self.assertAlmostEqual(p.delta_volume_um3, t7.volume_sphere_um3 - t0.volume_sphere_um3, places=2)
        self.assertAlmostEqual(p.delta_diameter_um, 100.0, places=4)
        self.assertAlmostEqual(p.fold_change_diameter, 2.0, places=4)

    def test_denominator_gate_sub_60um_exclusion(self):
        """Pair with initial diameter < 60.0 um has denominator_gate_pass = False."""
        t0 = _make_dummy_object("t0", diameter_um=45.0)
        t7 = _make_dummy_object("t7", timepoint="t7", diameter_um=90.0)

        pairs, _, _ = pair_timepoints([t0], [t7])
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0].denominator_gate_pass)
        self.assertIn("sub_denominator_baseline", pairs[0].qc_reasons)

    def test_denominator_gate_exact_60um_inclusion(self):
        """Pair with initial diameter >= 60.0 um has denominator_gate_pass = True."""
        t0_exact = _make_dummy_object("t0_60", diameter_um=60.0)
        t7_exact = _make_dummy_object("t7_60", timepoint="t7", diameter_um=120.0)

        pairs, _, _ = pair_timepoints([t0_exact], [t7_exact])
        self.assertEqual(len(pairs), 1)
        self.assertTrue(pairs[0].denominator_gate_pass)

    def test_combined_qc_flag_propagation(self):
        """Combined QC flag correctly cascades FAIL and REVIEW flags."""
        # 1. PASS + PASS => PASS
        t0_pass = _make_dummy_object("t0_p", diameter_um=100.0, qc_flag="PASS")
        t7_pass = _make_dummy_object("t7_p", timepoint="t7", diameter_um=120.0, qc_flag="PASS")
        p_pass = pair_timepoints([t0_pass], [t7_pass])[0][0]
        self.assertEqual(p_pass.combined_qc_flag, "PASS")

        # 2. FAIL + PASS => FAIL
        t0_fail = _make_dummy_object("t0_f", diameter_um=100.0, qc_flag="FAIL", qc_reasons="debris")
        p_fail = pair_timepoints([t0_fail], [t7_pass])[0][0]
        self.assertEqual(p_fail.combined_qc_flag, "FAIL")
        self.assertIn("t0_FAIL", p_fail.qc_reasons)

        # 3. REVIEW + PASS => REVIEW
        t0_rev = _make_dummy_object("t0_r", diameter_um=100.0, qc_flag="REVIEW", qc_reasons="low_circularity")
        p_rev = pair_timepoints([t0_rev], [t7_pass])[0][0]
        self.assertEqual(p_rev.combined_qc_flag, "REVIEW")
        self.assertIn("t0_REVIEW", p_rev.qc_reasons)

    def test_extreme_fold_change_trajectory_gating(self):
        """Fold changes outside [0.10, 20.0] are flagged as REVIEW."""
        # Extreme shrinkage: d0=200um -> d7=50um => FC = (50/200)^3 = 0.0156 (< 0.10)
        t0_lg = _make_dummy_object("t0_lg", diameter_um=200.0)
        t7_sm = _make_dummy_object("t7_sm", timepoint="t7", diameter_um=50.0)
        p_shrink = pair_timepoints([t0_lg], [t7_sm])[0][0]
        self.assertEqual(p_shrink.combined_qc_flag, "REVIEW")
        self.assertIn("extreme_shrinkage", p_shrink.qc_reasons)

        # Extreme expansion: d0=70um -> d7=300um => FC = (300/70)^3 = 78.7 (> 20.0)
        t0_sm = _make_dummy_object("t0_sm", diameter_um=70.0)
        t7_lg = _make_dummy_object("t7_lg", timepoint="t7", diameter_um=300.0)
        p_expand = pair_timepoints([t0_sm], [t7_lg])[0][0]
        self.assertEqual(p_expand.combined_qc_flag, "REVIEW")
        self.assertIn("extreme_expansion", p_expand.qc_reasons)


class TestPopulationStatisticsAndHygiene(unittest.TestCase):
    """Unit and functional tests for condition statistics and denominator gating hygiene."""

    def test_condition_summary_schema_compliance(self):
        """Validates all 21 mandated columns in condition_summary.csv."""
        t0 = _make_dummy_object("t0", condition="Mat", diameter_um=100.0)
        t7 = _make_dummy_object("t7", condition="Mat", timepoint="t7", diameter_um=150.0)
        pairs, _, _ = pair_timepoints([t0], [t7])

        summary_df = compute_condition_statistics(pairs)
        self.assertEqual(list(summary_df.columns), CONDITION_SUMMARY_COLUMNS)
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(summary_df.iloc[0]["condition"], "Mat")

    def test_denominator_gate_excludes_pairs_from_fold_change_stats(self):
        """Pairs with d0 < 60 um are tracked in n_denominator_excluded and excluded from FC calculations."""
        # 1 valid pair: d0=100, d7=150 (FC = 3.375)
        # 1 excluded pair: d0=30, d7=300 (FC = 1000x artificial blowup)
        t0_valid = _make_dummy_object("t0_v", cx=100, cy=100, diameter_um=100.0)
        t7_valid = _make_dummy_object("t7_v", timepoint="t7", cx=100, cy=100, diameter_um=150.0)

        t0_excl = _make_dummy_object("t0_e", cx=400, cy=400, diameter_um=30.0)
        t7_excl = _make_dummy_object("t7_e", timepoint="t7", cx=400, cy=400, diameter_um=300.0)

        pairs, _, _ = pair_timepoints([t0_valid, t0_excl], [t7_valid, t7_excl])
        self.assertEqual(len(pairs), 2)

        summary_df = compute_condition_statistics(pairs)
        row = summary_df.iloc[0]

        self.assertEqual(row["n_matched_pairs"], 2)
        self.assertEqual(row["n_denominator_excluded"], 1)
        # Median FC must be 3.375, NOT skewed by 1000x debris explosion
        self.assertAlmostEqual(row["fold_change_median"], 3.375, places=3)

    def test_median_iqr_mean_std_calculations(self):
        """Verifies accurate calculation of median, IQR, mean, and sample std on known sample."""
        # Create 5 pairs with known fold changes: [1.0, 2.0, 3.0, 4.0, 5.0]
        # median = 3.0, IQR = 4.0 - 2.0 = 2.0, mean = 3.0, std (ddof=1) = sqrt(2.5) ~ 1.5811
        pairs: List[SpheroidPairRecord] = []
        for i, target_fc in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            d0 = 100.0
            d7 = d0 * (target_fc ** (1.0 / 3.0))
            t0 = _make_dummy_object(f"t0_{i}", condition="Mat", diameter_um=d0)
            t7 = _make_dummy_object(f"t7_{i}", condition="Mat", timepoint="t7", diameter_um=d7)
            p = pair_timepoints([t0], [t7])[0][0]
            pairs.append(p)

        summary_df = compute_condition_statistics(pairs)
        row = summary_df.iloc[0]

        self.assertAlmostEqual(row["fold_change_median"], 3.0, places=4)
        self.assertAlmostEqual(row["fold_change_iqr"], 2.0, places=4)
        self.assertAlmostEqual(row["fold_change_mean"], 3.0, places=4)
        self.assertAlmostEqual(row["fold_change_std"], math.sqrt(2.5), places=4)

    def test_single_sample_and_empty_edge_cases(self):
        """Single sample yields IQR=0, std=0; empty input yields NaNs without error."""
        # Single sample
        t0 = _make_dummy_object("t0", condition="Mat", diameter_um=100.0)
        t7 = _make_dummy_object("t7", condition="Mat", timepoint="t7", diameter_um=100.0)
        p = pair_timepoints([t0], [t7])[0][0]
        df_single = compute_condition_statistics([p])
        r_single = df_single.iloc[0]

        self.assertEqual(r_single["n_matched_pairs"], 1)
        self.assertEqual(r_single["fold_change_iqr"], 0.0)
        self.assertEqual(r_single["fold_change_std"], 0.0)
        self.assertEqual(r_single["bootstrap_ci_95_low"], r_single["bootstrap_ci_95_high"])

        # Empty
        df_empty = compute_condition_statistics([])
        self.assertTrue(df_empty.empty)
        self.assertEqual(list(df_empty.columns), CONDITION_SUMMARY_COLUMNS)


class TestBootstrapAndPermutationTesting(unittest.TestCase):
    """Tests for non-parametric bootstrap CIs and Monte Carlo permutation testing."""

    def test_bootstrap_ci_percentile_coverage(self):
        """Bootstrap 95% CI produces valid confidence interval around known distribution median."""
        rng = np.random.RandomState(42)
        sample = rng.normal(loc=3.0, scale=0.5, size=50)
        ci_low, ci_high = bootstrap_ci_median(sample, n_boot=2000, confidence_level=0.95, random_seed=42)

        sample_median = float(np.median(sample))
        self.assertTrue(ci_low <= sample_median <= ci_high)
        self.assertTrue(ci_high - ci_low > 0.0)

    def test_bootstrap_ci_deterministic_reproducibility(self):
        """Identical random seeds yield identical confidence intervals."""
        data = [1.2, 1.8, 2.5, 3.1, 4.0]
        ci1 = bootstrap_ci_median(data, n_boot=1000, random_seed=123)
        ci2 = bootstrap_ci_median(data, n_boot=1000, random_seed=123)
        self.assertEqual(ci1, ci2)

    def test_permutation_test_identical_distributions(self):
        """Identical distributions yield non-significant p-value (p > 0.05)."""
        rng = np.random.RandomState(42)
        grp = rng.normal(2.0, 0.4, 25)
        ctrl = rng.normal(2.0, 0.4, 25)

        p_val = permutation_test_median(grp, ctrl, n_perm=5000, random_seed=42)
        self.assertTrue(p_val > 0.05)
        self.assertEqual(get_significance_stars(p_val), "ns")

    def test_permutation_test_distinct_distributions(self):
        """Distinct distributions (e.g. 5.0 vs 1.0) yield statistically significant p-value (p < 0.01)."""
        rng = np.random.RandomState(42)
        grp = rng.normal(5.0, 0.3, 20)
        ctrl = rng.normal(1.0, 0.3, 20)

        p_val = permutation_test_median(grp, ctrl, n_perm=5000, random_seed=42)
        self.assertTrue(p_val < 0.01)
        self.assertIn(get_significance_stars(p_val), ["**", "***", "****"])

    def test_permutation_test_exact_formula_bound(self):
        """Permutation p-value is strictly bounded in (0.0, 1.0]."""
        grp = np.array([100.0, 100.0])
        ctrl = np.array([1.0, 1.0])
        p_val = permutation_test_median(grp, ctrl, n_perm=1000)
        self.assertGreater(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)

    def test_significance_stars_all_thresholds(self):
        """Tests scientific asterisk mapping across all significance thresholds."""
        self.assertEqual(get_significance_stars(0.00005), "****")
        self.assertEqual(get_significance_stars(0.0005), "***")
        self.assertEqual(get_significance_stars(0.005), "**")
        self.assertEqual(get_significance_stars(0.03), "*")
        self.assertEqual(get_significance_stars(0.12), "ns")
        self.assertEqual(get_significance_stars(np.nan), "n/a")


class TestExclusionAuditingAndCSVPersistence(unittest.TestCase):
    """Tests for multilevel exclusion auditing tables and CSV persistence."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_multilevel_exclusion_audit_metrics_and_markdown(self):
        """Computes 10-step exclusion metrics and formats as valid Markdown table."""
        objs = [
            _make_dummy_object("o1", timepoint="t0", qc_flag="FAIL", qc_reasons="debris_undersized"),
            _make_dummy_object("o2", timepoint="t0", qc_flag="FAIL", qc_reasons="touches_image_border"),
            _make_dummy_object("o3", timepoint="t0", qc_flag="FAIL", qc_reasons="insufficient_dark_contrast"),
            _make_dummy_object("o4", timepoint="t0", qc_flag="PASS", pair_id="UNPAIRED"),
            _make_dummy_object("o5", timepoint="t7", qc_flag="PASS", pair_id="UNPAIRED"),
            _make_dummy_object("o6", timepoint="t0", qc_flag="PASS", pair_id="pair_1"),
            _make_dummy_object("o7", timepoint="t7", qc_flag="PASS", pair_id="pair_1"),
        ]
        pairs = [
            SpheroidPairRecord(
                pair_id="pair_1",
                pair_key="Mat_gel1_0001",
                condition="Mat",
                replicate="gel1",
                fov="0001",
                t0_object_id="o6",
                t0_centroid_x=100.0,
                t0_centroid_y=100.0,
                t0_eq_diameter_um=80.0,
                t0_volume_sphere_um3=268082.0,
                t0_volume_ellipsoid_um3=268082.0,
                t0_circularity=0.95,
                t0_qc_flag="PASS",
                t7_object_id="o7",
                t7_centroid_x=102.0,
                t7_centroid_y=102.0,
                t7_eq_diameter_um=120.0,
                t7_volume_sphere_um3=904778.0,
                t7_volume_ellipsoid_um3=904778.0,
                t7_circularity=0.95,
                t7_qc_flag="PASS",
                centroid_distance_px=2.83,
                delta_volume_um3=636696.0,
                fold_change_volume=3.375,
                log2_fold_change=1.755,
                delta_diameter_um=40.0,
                fold_change_diameter=1.5,
                denominator_gate_pass=True,
                combined_qc_flag="PASS",
                qc_reasons="None",
            )
        ]

        audit = compute_exclusion_audit(all_objects=objs, all_pairs=pairs, plausibility_fallbacks=1)

        self.assertEqual(audit["E2_plausibility_fallbacks"], 1)
        self.assertEqual(audit["E3_debris_undersized"], 1)
        self.assertEqual(audit["E5_border_touching"], 1)
        self.assertEqual(audit["E6_insufficient_contrast"], 1)
        self.assertEqual(audit["E7_unpaired_total"], 2)
        self.assertEqual(audit["E10_final_pass_pairs"], 1)

        audit_table = generate_exclusion_audit_table(audit)
        self.assertEqual(len(audit_table), 10)
        self.assertIn("Stage", audit_table.columns)

        md = generate_exclusion_audit_markdown(audit_table)
        self.assertIn("| Stage | Filter Description |", md)
        self.assertIn("E10", md)

    def test_save_pairs_csv_export(self):
        """Saves pairs list to CSV matching exact 30-column schema."""
        t0 = _make_dummy_object("t0", diameter_um=100.0)
        t7 = _make_dummy_object("t7", timepoint="t7", diameter_um=150.0)
        pairs, _, _ = pair_timepoints([t0], [t7])

        out_path = Path(self.test_dir) / "pairs.csv"
        df = save_pairs_csv(pairs, out_path)

        self.assertTrue(out_path.exists())
        self.assertEqual(list(df.columns), PAIRS_CSV_COLUMNS)
        self.assertEqual(len(df), 1)

        # Read back with pandas
        df_read = pd.read_csv(out_path)
        self.assertEqual(list(df_read.columns), PAIRS_CSV_COLUMNS)

    def test_save_condition_summary_csv_export(self):
        """Saves condition summary to CSV matching exact 21-column schema."""
        t0 = _make_dummy_object("t0", condition="Mat", diameter_um=100.0)
        t7 = _make_dummy_object("t7", condition="Mat", timepoint="t7", diameter_um=150.0)
        pairs, _, _ = pair_timepoints([t0], [t7])

        summary_df = compute_condition_statistics(pairs)
        out_path = Path(self.test_dir) / "condition_summary.csv"
        df = save_condition_summary_csv(summary_df, out_path)

        self.assertTrue(out_path.exists())
        self.assertEqual(list(df.columns), CONDITION_SUMMARY_COLUMNS)
        self.assertEqual(len(df), 1)

        # Read back with pandas
        df_read = pd.read_csv(out_path)
        self.assertEqual(list(df_read.columns), CONDITION_SUMMARY_COLUMNS)

    def test_format_condition_summary_table_output(self):
        """Formats stdout summary string with header, border, and condition rows."""
        t0 = _make_dummy_object("t0", condition="Mat", diameter_um=100.0)
        t7 = _make_dummy_object("t7", condition="Mat", timepoint="t7", diameter_um=150.0)
        pairs, _, _ = pair_timepoints([t0], [t7])

        summary_df = compute_condition_statistics(pairs)
        table_str = format_condition_summary_table(summary_df, control_condition="Mat")

        self.assertIn("SPHEROID VOLUME POPULATION GROWTH SUMMARY", table_str)
        self.assertIn("Mat", table_str)
        self.assertIn("Control (1.00)", table_str)


if __name__ == "__main__":
    unittest.main()
