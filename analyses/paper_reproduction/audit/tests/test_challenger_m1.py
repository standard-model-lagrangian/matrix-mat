"""
Adversarial Stress Test Suite: Milestone 1 Canonical Data Invariants
Executed by: challenger_1_m1
Archetype: EMPIRICAL CHALLENGER (critic, specialist)

Stress-tests:
- points.parquet (168 rows, 0 nulls, exact zero difference against hydrogel_analysis.csv,
  batch accounting 100 closed-loop + 68 pareto = 168, parameter boundaries, continuous vs grid)
- prior.parquet (43 rows, index 2 absent, exact zero difference against hydrogel_bo.csv,
  LAP feature collapse, exposure time mode clustering)
- optima.parquet (7 rows, exact window pass, boundary margin checks, paper text contradictions)
- Total accounting identity (43 + 168 = 211) and cryptographic SHA-256 manifest integrity
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

AUDIT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = AUDIT_ROOT / "artifacts" / "canonical"
UPSTREAM_DIR = Path("/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO")


class TestAdversarialPointsParquet:
    """Adversarial challenge of points.parquet."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.points_path = CANONICAL_DIR / "points.parquet"
        self.raw_analysis_path = UPSTREAM_DIR / "hydrogel_analysis.csv"
        self.raw_bo_path = UPSTREAM_DIR / "hydrogel_bo.csv"
        assert self.points_path.is_file(), f"Missing {self.points_path}"
        assert self.raw_analysis_path.is_file(), f"Missing {self.raw_analysis_path}"
        self.df = pd.read_parquet(self.points_path)
        self.df_raw = pd.read_csv(self.raw_analysis_path, index_col=0)

    def test_points_row_count_strict(self):
        """Assert points.parquet has exactly 168 rows."""
        assert len(self.df) == 168, f"Expected exactly 168 rows, got {len(self.df)}"

    def test_points_zero_nulls_adversarial(self):
        """Assert zero null/NaN/None values across all 12 columns in points.parquet."""
        null_counts = self.df.isnull().sum()
        total_nulls = null_counts.sum()
        assert total_nulls == 0, f"Found {total_nulls} nulls in points.parquet: {null_counts[null_counts > 0].to_dict()}"

    def test_points_exact_zero_difference_against_raw_csv(self):
        """Assert exact zero difference (|diff| == 0.0) across all 7 numerical columns against hydrogel_analysis.csv."""
        col_map = {
            "Exposure Time": "exposure_time",
            "GelMA-L": "gelma_l",
            "GelMA-H": "gelma_h",
            "aCNC": "acnc",
            "LAP": "lap",
            "Modulus": "modulus",
            "Stress Relaxation": "stress_relaxation",
        }
        for raw_col, can_col in col_map.items():
            diff = np.abs(self.df[can_col].values - self.df_raw[raw_col].values)
            max_diff = float(np.max(diff))
            assert max_diff == 0.0, f"Divergence in {can_col}: max diff is {max_diff} (> 0.0)"

    def test_points_source_index_row_alignment_and_campaign_blocks(self):
        """
        Adversarially verify source_index ordering and completeness:
        1. source_index exactly matches hydrogel_analysis.csv index row-for-row.
        2. The set of source_index is exactly all integers from 44 to 211 (168 unique points).
        3. Document the 4 non-monotonic campaign execution blocks present in the upstream CSV:
           - Block 1: 44..95 (HealthySkin + ScarredSkin: 52 points)
           - Block 2: 136..183 (SoftTumor + StiffTumor: 48 points)
           - Block 3: 96..135 (Pareto_Utopian01 + Pareto_Utopian02: 40 points)
           - Block 4: 184..211 (Pareto_Utopian03 + Pareto_HV: 28 points)
        """
        actual_indices = self.df["source_index"].values
        raw_indices = self.df_raw.index.values
        assert (actual_indices == raw_indices).all(), "source_index does not strictly match hydrogel_analysis.csv index"
        assert set(actual_indices) == set(range(44, 212)), "Missing or out-of-range source indices in points.parquet"

        # Verify exact block boundaries
        b1 = actual_indices[0:52]
        b2 = actual_indices[52:100]
        b3 = actual_indices[100:140]
        b4 = actual_indices[140:168]

        assert (b1 == np.arange(44, 96)).all()
        assert (b2 == np.arange(136, 184)).all()
        assert (b3 == np.arange(96, 136)).all()
        assert (b4 == np.arange(184, 212)).all()


    def test_points_point_id_strictly_monotonic_0_to_167(self):
        """Assert point_id is strictly 0..167 without skips or permutations."""
        expected_ids = np.arange(168)
        assert (self.df["point_id"].values == expected_ids).all(), "point_id must be strictly 0..167"

    def test_points_batch_accounting_closed_loop_and_pareto(self):
        """
        Adversarially verify batch structure:
        - Exactly 42 batches (IDs 1..42)
        - Exactly 4 points per batch across ALL 42 batches
        - Batches 1..25 are closed_loop (100 rows)
        - Batches 26..42 are pareto_mapping (68 rows)
        - Total: 100 + 68 = 168 rows
        """
        assert self.df["batch_id"].nunique() == 42
        assert set(self.df["batch_id"].unique()) == set(range(1, 43))

        pts_per_batch = self.df.groupby("batch_id").size()
        assert (pts_per_batch == 4).all(), f"Batches without exactly 4 points: {pts_per_batch[pts_per_batch != 4].to_dict()}"

        closed_loop = self.df[self.df["batch_type"] == "closed_loop"]
        pareto = self.df[self.df["batch_type"] == "pareto_mapping"]
        assert len(closed_loop) == 100
        assert len(pareto) == 68
        assert (closed_loop["batch_id"] <= 25).all()
        assert (pareto["batch_id"] >= 26).all()

        # Detailed campaign breakdown
        expected_campaign_counts = {
            "HealthySkin": 20,
            "ScarredSkin": 32,
            "SoftTumor": 28,
            "StiffTumor": 20,
            "Pareto_Utopian01": 28,
            "Pareto_Utopian02": 12,
            "Pareto_Utopian03": 12,
            "Pareto_HV": 16,
        }
        for camp, count in expected_campaign_counts.items():
            actual = len(self.df[self.df["campaign"] == camp])
            assert actual == count, f"Campaign {camp}: expected {count}, got {actual}"

    def test_points_physical_boundaries_adversarial(self):
        """Adversarially assert all 5 formulation inputs and 2 output properties remain strictly in plausible bounds."""
        assert (self.df["exposure_time"] >= 0.0).all() and (self.df["exposure_time"] <= 600.0).all()
        assert (self.df["gelma_l"] >= 0.0).all() and (self.df["gelma_l"] <= 6.0).all()
        assert (self.df["gelma_h"] >= 0.0).all() and (self.df["gelma_h"] <= 6.0).all()
        assert (self.df["acnc"] >= 0.0).all() and (self.df["acnc"] <= 6.0).all()
        assert (self.df["lap"] >= 0.0).all() and (self.df["lap"] <= 0.20).all()

        # Target properties
        assert (self.df["modulus"] > 0.0).all()
        assert (self.df["modulus"] < 50.0).all()
        assert (self.df["stress_relaxation"] >= 0.0).all() and (self.df["stress_relaxation"] <= 100.0).all()

    def test_points_no_duplicate_formulations(self):
        """Assert zero exact duplicate formulations in the 168 experimental points."""
        features = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]
        dup_count = self.df.duplicated(subset=features).sum()
        assert dup_count == 0, f"Found {dup_count} duplicate formulations in points.parquet"

    def test_points_continuous_vs_discrete_grid_contradiction(self):
        """
        Adversarial finding:
        The paper claims a discrete 5D grid space (18,865 true or 6,230 claimed),
        but empirical active learning proposals were CONTINUOUS:
        < 15% of points lie on the discrete grid levels for any feature except LAP.
        """
        grid_levels = {
            "exposure_time": np.linspace(0, 600, 11),
            "gelma_l": np.linspace(0, 6, 7),
            "gelma_h": np.linspace(0, 6, 7),
            "acnc": np.linspace(0, 6, 7),
            "lap": np.linspace(0, 0.2, 5),
        }
        for param in ["exposure_time", "gelma_l", "gelma_h", "acnc"]:
            on_grid = self.df[param].apply(lambda v: any(np.isclose(v, grid_levels[param], atol=1e-3)))
            pct_on_grid = (on_grid.sum() / len(self.df)) * 100.0
            assert pct_on_grid < 15.0, f"Parameter {param} unexpectedly adheres to grid: {pct_on_grid:.1f}%"


class TestAdversarialPriorParquet:
    """Adversarial challenge of prior.parquet."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.prior_path = CANONICAL_DIR / "prior.parquet"
        self.raw_bo_path = UPSTREAM_DIR / "hydrogel_bo.csv"
        assert self.prior_path.is_file(), f"Missing {self.prior_path}"
        assert self.raw_bo_path.is_file(), f"Missing {self.raw_bo_path}"
        self.df = pd.read_parquet(self.prior_path)
        self.df_raw_bo = pd.read_csv(self.raw_bo_path, index_col=0)

    def test_prior_row_count_strictly_43(self):
        """Assert prior.parquet has exactly 43 rows, directly refuting the paper claim of 44."""
        assert len(self.df) == 43, f"Expected 43 rows in prior.parquet, got {len(self.df)}"

    def test_prior_absence_of_index_2(self):
        """Adversarially verify that source_index == 2 is strictly absent from prior.parquet and hydrogel_bo.csv."""
        assert 2 not in self.df["source_index"].values, "Critical bug: index 2 should NOT be in prior.parquet"
        assert 2 not in self.df_raw_bo.index.values, "Index 2 should NOT be in raw hydrogel_bo.csv"
        assert set(self.df["source_index"].values) == (set(range(44)) - {2})

    def test_prior_exact_zero_difference_against_upstream_oracle(self):
        """Assert exact zero difference between prior.parquet and first 43 rows of hydrogel_bo.csv."""
        oracle_prior = self.df_raw_bo.iloc[:43]
        col_map = {
            "Exposure Time": "exposure_time",
            "GelMA-L": "gelma_l",
            "GelMA-H": "gelma_h",
            "aCNC": "acnc",
            "LAP": "lap",
            "Modulus": "imputed_modulus",
            "Stress Relaxation": "stress_relaxation",
        }
        for raw_col, can_col in col_map.items():
            diff = np.abs(self.df[can_col].values - oracle_prior[raw_col].values)
            max_diff = float(np.max(diff))
            assert max_diff == 0.0, f"Prior mismatch in {can_col}: max diff {max_diff}"

        assert (self.df["source_index"].values == oracle_prior.index.values).all()

    def test_prior_zero_nulls(self):
        """Assert 0 nulls across all columns in prior.parquet."""
        assert self.df.isnull().sum().sum() == 0, "Null values found in prior.parquet"

    def test_prior_lap_zero_variance_collapse(self):
        """
        Adversarially confirm severe feature collapse in prior data:
        35 out of 43 rows (81.4%) have LAP fixed at exactly 0.10 wt%.
        """
        lap_counts = self.df["lap"].value_counts()
        assert 0.10 in lap_counts
        assert lap_counts[0.10] == 35
        pct = (35 / 43) * 100.0
        assert pct > 81.0, f"Expected >81% LAP=0.10, got {pct:.1f}%"

    def test_prior_exposure_time_mode_clustering(self):
        """Confirm 67.4% of prior conditions have exposure time clustered at exactly 30s (15 rows) or 90s (14 rows)."""
        counts = self.df["exposure_time"].value_counts()
        clustered = counts.get(30.0, 0) + counts.get(90.0, 0)
        assert clustered == 29
        assert (29 / 43) * 100.0 > 67.0


class TestAdversarialOptimaParquet:
    """Adversarial challenge of optima.parquet."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optima_path = CANONICAL_DIR / "optima.parquet"
        self.points_path = CANONICAL_DIR / "points.parquet"
        self.df_opt = pd.read_parquet(self.optima_path)
        self.df_pts = pd.read_parquet(self.points_path)

    def test_optima_row_count_and_candidate_set(self):
        """Assert exactly 7 optimal candidates corresponding to Table 1."""
        assert len(self.df_opt) == 7
        expected_candidates = {"I4-2", "I4-4", "I6-4", "I7-3", "I3-3", "I4-1", "I1-4"}
        assert set(self.df_opt["candidate_id"]) == expected_candidates

    def test_optima_match_points_parquet_identically(self):
        """Assert all 7 optima formulations and measured properties match points.parquet identically."""
        cols = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap", "modulus", "stress_relaxation"]
        for _, row in self.df_opt.iterrows():
            s_idx = row["source_index"]
            pt = self.df_pts[self.df_pts["source_index"] == s_idx].iloc[0]
            for c in ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]:
                assert row[c] == pt[c], f"Mismatch in {c} for {row['candidate_id']}"
            assert row["measured_modulus"] == pt["modulus"]
            assert row["measured_stress_relaxation"] == pt["stress_relaxation"]

    def test_optima_window_pass_evaluations(self):
        """Assert all 7 candidates evaluate window_pass == True against Table 1."""
        assert self.df_opt["window_pass"].all(), "Not all candidates pass window"

        # Explicitly verify each candidate
        for _, row in self.df_opt.iterrows():
            # Modulus check
            e_min = row["target_modulus"] - row["tolerance_modulus"]
            e_max = row["target_modulus"] + row["tolerance_modulus"]
            assert e_min <= row["measured_modulus"] <= e_max, (
                f"{row['candidate_id']} measured modulus {row['measured_modulus']} out of window [{e_min}, {e_max}]"
            )

            # Relaxation check (if constrained)
            if not np.isnan(row["target_stress_relaxation"]):
                sr_min = row["target_stress_relaxation"] - row["tolerance_stress_relaxation"]
                sr_max = row["target_stress_relaxation"] + row["tolerance_stress_relaxation"]
                assert sr_min <= row["measured_stress_relaxation"] <= sr_max, (
                    f"{row['candidate_id']} measured SR {row['measured_stress_relaxation']} out of window [{sr_min}, {sr_max}]"
                )

    def test_optima_soft_tumor_i4_1_extreme_upper_boundary(self):
        """
        Adversarially verify the extreme upper boundary condition of Soft Tumor candidate I4-1:
        Measured E = 1.428 kPa vs upper window bound 1.30 + 0.13 = 1.430 kPa.
        The candidate passed by an extremely thin margin of 0.002 kPa (2 Pa).
        """
        i4_1 = self.df_opt[self.df_opt["candidate_id"] == "I4-1"].iloc[0]
        upper = i4_1["target_modulus"] + i4_1["tolerance_modulus"]
        diff = upper - i4_1["measured_modulus"]
        assert 0.0 < diff < 0.003
        assert np.isclose(diff, 0.002, atol=1e-5)

    def test_optima_scarred_skin_text_contradiction(self):
        """
        Verify mathematical contradiction in paper narrative:
        Section 2.2 claims scarred skin relaxation is '22.5 +/- 1.5%'.
        Upper bound of Table 1 is 20.0 + 2.0 = 22.0%.
        Therefore the narrative claim of 22.5% strictly violates Table 1 target window!
        """
        scarred = self.df_opt[self.df_opt["target_name"] == "ScarredSkin"]
        assert set(scarred["measured_stress_relaxation"]) == {21.0, 20.5}
        mean_sr = scarred["measured_stress_relaxation"].mean()
        assert mean_sr == 20.75
        # Contrast with paper text 22.5
        assert 22.5 > 22.0, "22.5% strictly exceeds Table 1 upper bound of 22.0%"


class TestAdversarialTotalAccountingAndManifest:
    """Adversarial verification of total accounting and cryptographic integrity."""

    def test_total_accounting_identity_211_points(self):
        """
        Assert exact dataset accounting identity:
        43 prior points + 100 closed-loop points + 68 Pareto mapping points = 211 total points in hydrogel_bo.csv.
        """
        df_prior = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        df_pts = pd.read_parquet(CANONICAL_DIR / "points.parquet")
        df_bo = pd.read_csv(UPSTREAM_DIR / "hydrogel_bo.csv", index_col=0)

        assert len(df_prior) == 43
        assert len(df_pts) == 168
        assert len(df_prior) + len(df_pts) == 211
        assert len(df_bo) == 211

    def test_upstream_hydrogel_analysis_equals_bo_csv_tail(self):
        """Assert hydrogel_analysis.csv is identical to hydrogel_bo.csv rows 43..210."""
        df_ana = pd.read_csv(UPSTREAM_DIR / "hydrogel_analysis.csv", index_col=0)
        df_bo = pd.read_csv(UPSTREAM_DIR / "hydrogel_bo.csv", index_col=0)
        diff = np.abs(df_ana.values - df_bo.iloc[43:].values)
        assert np.max(diff) == 0.0

    def test_manifest_sha256_cryptographic_verification(self):
        """Adversarially re-compute SHA-256 digests and compare against manifest.sha256."""
        manifest_path = CANONICAL_DIR / "manifest.sha256"
        assert manifest_path.is_file()

        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = [line.strip().split(maxsplit=1) for line in f if line.strip() and not line.startswith("#")]

        assert len(entries) == 4, f"Expected 4 canonical files in manifest, got {len(entries)}"

        for exp_hash, rel_path in entries:
            target_path = AUDIT_ROOT / rel_path
            assert target_path.is_file(), f"Target file not found: {target_path}"
            with open(target_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == exp_hash, f"SHA-256 mismatch for {rel_path}: actual {actual_hash} != expected {exp_hash}"
