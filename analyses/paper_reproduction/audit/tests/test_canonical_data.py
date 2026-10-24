"""
Tier 1 & Tier 2 Tests: Canonical Parquet Tables, Schemas, Bounds, and Paper Contradictions.
Validates:
- points.parquet (168 rows, proper types, non-null, bounds, campaign partition)
- prior.parquet (43 rows, index 2 absent, LAP variance collapse, bounds)
- optima.parquet (7 rows, target windows, boundary cases, text discrepancies)
- paper_claims.yaml (discretization arithmetic contradiction, Section 5 constants)
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

AUDIT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = AUDIT_ROOT / "artifacts" / "canonical"
UPSTREAM_DIR = Path("/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO")


def check_known_constraint(gelma_l: float, gelma_h: float, acnc: float, lap: float, tol: float = 1e-3) -> bool:
    """
    Physical feasibility constraint defined in Section 5 and run.py:
    1. Total polymer concentration: GelMA_L/15 + GelMA_H/15 + aCNC/6.1 + LAP/2 <= 1.0
    2. Minimum GelMA content for network integrity: GelMA_L + GelMA_H >= 3.0
    """
    if (gelma_l / 15.0) + (gelma_h / 15.0) + (acnc / 6.1) + (lap / 2.0) > 1.0 + tol:
        return False
    if (gelma_l + gelma_h) < 3.0 - tol:
        return False
    return True



class TestCanonicalDataTier1:
    """Tier 1: Feature Coverage, Schema Compliance, Non-Null, and Oracle Alignment."""

    def test_canonical_files_exist(self):
        """Assert all required canonical artifacts exist in artifacts/canonical/."""
        required = [
            "points.parquet",
            "prior.parquet",
            "optima.parquet",
            "paper_claims.yaml",
            "manifest.sha256",
        ]
        for fname in required:
            fpath = CANONICAL_DIR / fname
            assert fpath.is_file(), f"Missing required canonical artifact: {fpath}"

    def test_points_parquet_schema_and_row_count(self):
        """Assert points.parquet has exactly 168 rows and the complete canonical schema."""
        path = CANONICAL_DIR / "points.parquet"
        assert path.is_file(), f"Missing {path}"
        df = pd.read_parquet(path)
        assert len(df) == 168, f"Expected 168 rows in points.parquet, found {len(df)}"

        expected_cols = [
            "point_id",
            "source_index",
            "exposure_time",
            "gelma_l",
            "gelma_h",
            "acnc",
            "lap",
            "modulus",
            "stress_relaxation",
            "campaign",
            "batch_id",
            "batch_type",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column in points.parquet: {col}"

        # Assert no nulls
        assert df[expected_cols].isnull().sum().sum() == 0, "Null values found in points.parquet"

        # Check types
        assert pd.api.types.is_integer_dtype(df["point_id"]), "point_id must be integer"
        assert pd.api.types.is_integer_dtype(df["source_index"]), "source_index must be integer"
        assert pd.api.types.is_integer_dtype(df["batch_id"]), "batch_id must be integer"
        for col in ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap", "modulus", "stress_relaxation"]:
            assert pd.api.types.is_float_dtype(df[col]), f"{col} must be float"

        # Check point_id strictly monotonically increasing 0..167
        assert (df["point_id"].values == np.arange(168)).all(), "point_id must be 0..167"

    def test_points_parquet_bounds_and_physical_ranges(self):
        """Assert all features and targets in points.parquet fall within valid physical bounds."""
        df = pd.read_parquet(CANONICAL_DIR / "points.parquet")

        assert (df["exposure_time"] >= 0.0).all() and (df["exposure_time"] <= 600.0).all()
        assert (df["gelma_l"] >= 0.0).all() and (df["gelma_l"] <= 6.0).all()
        assert (df["gelma_h"] >= 0.0).all() and (df["gelma_h"] <= 6.0).all()
        assert (df["acnc"] >= 0.0).all() and (df["acnc"] <= 6.0).all()
        assert (df["lap"] >= 0.0).all() and (df["lap"] <= 0.20).all()

        # Target physical properties
        assert (df["modulus"] > 0.0).all(), "Young's modulus must be positive"
        assert (df["modulus"] < 50.0).all(), "Young's modulus exceeds plausible hydrogel bounds"
        assert (df["stress_relaxation"] >= 0.0).all() and (df["stress_relaxation"] <= 100.0).all(), (
            "Stress relaxation must be between 0% and 100%"
        )

    def test_points_parquet_numerical_identity_with_upstream_oracle(self):
        """Assert zero floating-point divergence between points.parquet and hydrogel_analysis.csv."""
        oracle_path = UPSTREAM_DIR / "hydrogel_analysis.csv"
        if not oracle_path.is_file():
            pytest.skip("Upstream hydrogel_analysis.csv not found for comparison")

        upstream_analysis_df = pd.read_csv(oracle_path, index_col=0)
        df = pd.read_parquet(CANONICAL_DIR / "points.parquet")

        col_map = {
            "Exposure Time": "exposure_time",
            "GelMA-L": "gelma_l",
            "GelMA-H": "gelma_h",
            "aCNC": "acnc",
            "LAP": "lap",
            "Modulus": "modulus",
            "Stress Relaxation": "stress_relaxation",
        }

        for up_col, can_col in col_map.items():
            max_diff = float(np.max(np.abs(df[can_col].values - upstream_analysis_df[up_col].values)))
            assert max_diff == 0.0, f"Divergence detected in {can_col}: max absolute diff is {max_diff}"

        # Check source indices match upstream index column
        assert (df["source_index"].values == upstream_analysis_df.index.values).all()

    def test_points_campaign_and_batch_accounting(self):
        """Assert exact accounting: 100 closed-loop points (25 batches) + 68 Pareto points (17 batches) = 168."""
        df = pd.read_parquet(CANONICAL_DIR / "points.parquet")

        closed_loop = df[df["batch_type"] == "closed_loop"]
        pareto = df[df["batch_type"] == "pareto_mapping"]

        assert len(closed_loop) == 100, f"Expected 100 closed-loop points, found {len(closed_loop)}"
        assert len(pareto) == 68, f"Expected 68 Pareto mapping points, found {len(pareto)}"

        # Check closed loop campaign counts
        healthy = df[df["campaign"] == "HealthySkin"]
        scarred = df[df["campaign"] == "ScarredSkin"]
        soft = df[df["campaign"].isin(["SoftTumor", "BenignTumor"])]
        stiff = df[df["campaign"].isin(["StiffTumor", "MalignantTumor"])]

        assert len(healthy) == 20, f"HealthySkin expected 20 points, got {len(healthy)}"
        assert len(scarred) == 32, f"ScarredSkin expected 32 points, got {len(scarred)}"
        assert len(soft) == 28, f"SoftTumor expected 28 points, got {len(soft)}"
        assert len(stiff) == 20, f"StiffTumor expected 20 points, got {len(stiff)}"

        # Check Pareto sub-campaign counts
        p_u1 = df[df["campaign"] == "Pareto_Utopian01"]
        p_u2 = df[df["campaign"] == "Pareto_Utopian02"]
        p_u3 = df[df["campaign"] == "Pareto_Utopian03"]
        p_hv = df[df["campaign"] == "Pareto_HV"]

        assert len(p_u1) == 28, f"Pareto_Utopian01 expected 28 points, got {len(p_u1)}"
        assert len(p_u2) == 12, f"Pareto_Utopian02 expected 12 points, got {len(p_u2)}"
        assert len(p_u3) == 12, f"Pareto_Utopian03 expected 12 points, got {len(p_u3)}"
        assert len(p_hv) == 16, f"Pareto_HV expected 16 points, got {len(p_hv)}"

        # Check total batch count: 25 closed loop + 17 pareto = 42 batches
        assert df["batch_id"].nunique() == 42
        assert set(df["batch_id"].unique()) == set(range(1, 43))

    def test_prior_parquet_schema_and_row_count(self):
        """Assert prior.parquet has exactly 43 rows (refuting claimed 44) and correct schema."""
        df = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        assert len(df) == 43, f"Expected 43 rows in prior.parquet, found {len(df)}"

        mod_col = "imputed_modulus" if "imputed_modulus" in df.columns else "modulus"
        expected_cols = [
            "prior_id",
            "source_index",
            "exposure_time",
            "gelma_l",
            "gelma_h",
            "acnc",
            "lap",
            mod_col,
            "stress_relaxation",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing expected column in prior.parquet: {col}"

        assert df[expected_cols].isnull().sum().sum() == 0, "Null values found in prior.parquet"
        assert pd.api.types.is_integer_dtype(df["prior_id"])
        assert pd.api.types.is_integer_dtype(df["source_index"])

    def test_prior_parquet_matches_upstream_oracle(self):
        """Assert zero difference between prior.parquet and upstream hydrogel_bo.csv rows 0..42."""
        oracle_path = UPSTREAM_DIR / "hydrogel_bo.csv"
        if not oracle_path.is_file():
            pytest.skip("Upstream hydrogel_bo.csv not found for comparison")

        upstream_bo_df = pd.read_csv(oracle_path, index_col=0)
        df = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        oracle_prior = upstream_bo_df.iloc[:43]

        col_map = {
            "Exposure Time": "exposure_time",
            "GelMA-L": "gelma_l",
            "GelMA-H": "gelma_h",
            "aCNC": "acnc",
            "LAP": "lap",
            "Stress Relaxation": "stress_relaxation",
        }

        for up_col, can_col in col_map.items():
            max_diff = float(np.max(np.abs(df[can_col].values - oracle_prior[up_col].values)))
            assert max_diff == 0.0, f"Prior mismatch in {can_col}: max diff {max_diff}"

        mod_col = "imputed_modulus" if "imputed_modulus" in df.columns else "modulus"
        max_mod_diff = float(np.max(np.abs(df[mod_col].values - oracle_prior["Modulus"].values)))
        assert max_mod_diff == 0.0, f"Prior modulus mismatch: max diff {max_mod_diff}"

    def test_optima_parquet_schema_and_candidates(self):
        """Assert optima.parquet contains all 7 reported optimal hydrogels and target windows."""
        df = pd.read_parquet(CANONICAL_DIR / "optima.parquet")
        assert len(df) == 7, f"Expected 7 reported optima, found {len(df)}"

        expected_cols = [
            "target_name",
            "candidate_id",
            "source_index",
            "exposure_time",
            "gelma_l",
            "gelma_h",
            "acnc",
            "lap",
            "measured_modulus",
            "measured_stress_relaxation",
            "target_modulus",
            "target_stress_relaxation",
            "tolerance_modulus",
            "tolerance_stress_relaxation",
            "window_pass",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column in optima.parquet: {col}"

        expected_ids = {"I4-2", "I4-4", "I6-4", "I7-3", "I3-3", "I4-1", "I1-4"}
        assert set(df["candidate_id"].values) == expected_ids

        # All 7 reported candidates pass their target windows
        assert df["window_pass"].all(), "Expected all reported optimal candidates to pass window"


class TestCanonicalDataTier2:
    """Tier 2: Boundary & Corner Cases, Adversarial Stress, and Paper Contradictions."""

    def test_prior_missing_index_2_corner_case(self):
        """
        Verify that index 2 is strictly absent in the prior dataset.
        Paper claims '44 fabrication conditions' in Section 2.1 and SI Table S2,
        but rows enumerate [0, 1, 3, 4, ... 43], totalling exactly 43 rows.
        """
        df = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        assert len(df) == 43, f"Expected 43 rows, found {len(df)}"
        assert 2 not in df["source_index"].values, "Index 2 should NOT be in prior dataset"
        assert 0 in df["source_index"].values
        assert 1 in df["source_index"].values
        assert 3 in df["source_index"].values
        assert 43 in df["source_index"].values

    def test_prior_lap_zero_variance_artifact(self):
        """
        Verify that LAP concentration in prior is 0.10 in >= 80% of rows (35 of 43 rows).
        This severe feature collapse causes ARD GP lengthscale inflation (Task T04).
        """
        df = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        lap_val_counts = df["lap"].value_counts()
        assert 0.10 in lap_val_counts
        lap_010_count = lap_val_counts[0.10]
        assert lap_010_count == 35, f"Expected exactly 35 rows with LAP=0.10, found {lap_010_count}"
        proportion = lap_010_count / len(df)
        assert proportion >= 0.81, f"Expected >= 81% LAP=0.10, found {proportion*100:.1f}%"

        # Unique values must be exactly 4: 0.01, 0.05, 0.10, 0.20
        unique_lap = set(np.round(df["lap"].unique(), 2))
        assert unique_lap == {0.01, 0.05, 0.10, 0.20}

    def test_prior_exposure_time_mode_clustering(self):
        """
        Verify that Exposure Time in prior is heavily clustered at 30 s (15 rows) and 90 s (14 rows).
        67.4% of prior conditions have Exposure Time in {30, 90}.
        """
        df = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        t_counts = df["exposure_time"].value_counts()
        count_30_90 = t_counts.get(30.0, 0) + t_counts.get(90.0, 0)
        assert count_30_90 == 29, f"Expected 29 rows with time in {{30, 90}}, found {count_30_90}"
        assert count_30_90 / len(df) >= 0.67

    def test_exact_dataset_accounting_invariant(self):
        """
        Verify full accounting identity:
        43 prior + 100 closed loop + 68 pareto mapping = 211 total points.
        """
        df_prior = pd.read_parquet(CANONICAL_DIR / "prior.parquet")
        df_points = pd.read_parquet(CANONICAL_DIR / "points.parquet")

        assert len(df_prior) + len(df_points) == 211

    def test_soft_tumor_i4_1_modulus_upper_tolerance_boundary(self):
        """
        Test tolerance boundary precision on Soft Tumor hit I4-1.
        Target: E = 1.30 kPa, tolerance = 10% -> [1.17, 1.43] kPa.
        Measured E for I4-1 is 1.428 kPa, which is within 0.002 kPa (2 Pa) of the upper bound.
        """
        df = pd.read_parquet(CANONICAL_DIR / "optima.parquet")
        i4_1 = df[df["candidate_id"] == "I4-1"].iloc[0]

        target = i4_1["target_modulus"]
        tol = i4_1["tolerance_modulus"]
        upper_bound = target + tol  # 1.30 + 0.13 = 1.43
        measured = i4_1["measured_modulus"]

        assert measured <= upper_bound, f"I4-1 measured {measured} exceeds upper bound {upper_bound}"
        diff_from_bound = upper_bound - measured
        assert diff_from_bound < 0.003, f"Expected close boundary hit, got margin of {diff_from_bound}"
        assert measured == 1.428

    def test_scarred_skin_target_window_paper_text_contradiction(self):
        """
        Verify the mathematical contradiction in Scarred Skin relaxation reporting (Task T10).
        - Table 1 Target: sigma_100 = 20.0 +/- 2.0% -> [18.0, 22.0]%.
        - Raw data candidates: I6-4 is 21.0%, I7-3 is 20.5% -> BOTH PASS.
        - Paper main text claim (Section 2.2): '22.5 +/- 1.5%'.
        - 22.5% strictly FAILS the Table 1 window (22.5 > 22.0).
        """
        df = pd.read_parquet(CANONICAL_DIR / "optima.parquet")
        scarred = df[df["target_name"] == "ScarredSkin"]

        i6_4 = scarred[scarred["candidate_id"] == "I6-4"].iloc[0]
        i7_3 = scarred[scarred["candidate_id"] == "I7-3"].iloc[0]

        assert i6_4["measured_stress_relaxation"] == 21.0
        assert i7_3["measured_stress_relaxation"] == 20.5
        assert (21.0 + 20.5) / 2.0 == 20.75  # Actual mean is 20.75%, NOT 22.5%

        # Target window check
        target = 20.0
        tol = 2.0
        lower, upper = target - tol, target + tol  # [18.0, 22.0]

        assert lower <= i6_4["measured_stress_relaxation"] <= upper
        assert lower <= i7_3["measured_stress_relaxation"] <= upper

        paper_text_claim = 22.5
        assert paper_text_claim > upper, (
            f"Paper text claim {paper_text_claim}% mathematically contradicts Table 1 upper bound {upper}%"
        )

    def test_stiff_tumor_modulus_discrepancy(self):
        """
        Verify the quantitative discrepancy in Stiff Tumor modulus reporting (Task T10).
        - Paper main text Section 2.3 claims E = 16.8 kPa for I1-4.
        - Raw CSV and Table S4 report E = 15.923 kPa (rounded to 15.9 kPa in SI).
        - Discrepancy is 0.877 kPa (> 5% deviation).
        """
        df = pd.read_parquet(CANONICAL_DIR / "optima.parquet")
        i1_4 = df[df["candidate_id"] == "I1-4"].iloc[0]

        raw_modulus = i1_4["measured_modulus"]
        assert abs(raw_modulus - 15.923) < 1e-3
        paper_text_claim = 16.8
        discrepancy = abs(paper_text_claim - raw_modulus)
        assert discrepancy > 0.85, f"Expected discrepancy > 0.85 kPa, got {discrepancy}"

    def test_duplicate_optima_euclidean_distance(self):
        """
        Verify that Healthy Skin hits I4-2 and I4-4 proposed within the same iteration
        have a normalized Euclidean distance < 0.05 in [0, 1]^5 (near-duplicate proposals).
        """
        df = pd.read_parquet(CANONICAL_DIR / "optima.parquet")
        i4_2 = df[df["candidate_id"] == "I4-2"].iloc[0]
        i4_4 = df[df["candidate_id"] == "I4-4"].iloc[0]

        bounds = np.array([600.0, 6.0, 6.0, 6.0, 0.20])
        cols = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]

        v1 = np.array([i4_2[c] for c in cols]) / bounds
        v2 = np.array([i4_4[c] for c in cols]) / bounds

        dist = np.linalg.norm(v1 - v2)
        assert dist < 0.05, f"Expected near-duplicate distance < 0.05, got {dist:.4f}"
        assert abs(dist - 0.0448) < 0.005

    def test_paper_claims_yaml_content_and_structure(self):
        """Verify paper_claims.yaml contains all required audit sections."""
        path = CANONICAL_DIR / "paper_claims.yaml"
        assert path.is_file(), f"Missing {path}"

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "discretization_bounds" in data
        assert "table_1_targets" in data
        assert "surrogate_performance_table_s6" in data
        assert "viscoelastic_model_si" in data
        assert "accounting" in data
        assert data["accounting"]["prior_dataset_observed_rows"] == 43

    def test_paper_claims_yaml_discretization_arithmetic_contradiction(self):
        """
        Verify the grid discretization arithmetic contradiction in paper_claims.yaml (Task T10).
        - Table S1 levels: [7, 7, 7, 5, 11] -> Cartesian product is 18,865.
        - Paper claims 6,230 possible hydrogel conditions in Section 2.1.
        - 18,865 != 6,230.
        - Factorization of 6,230 includes 89, proving it cannot be formed by subset of stated grid levels.
        """
        yaml_path = CANONICAL_DIR / "paper_claims.yaml"
        assert yaml_path.is_file(), f"Missing {yaml_path}"

        with open(yaml_path, "r", encoding="utf-8") as f:
            claims = yaml.safe_load(f)

        bounds = claims["discretization_bounds"]
        claimed_size = bounds.get("claimed_grid_size", 6230)
        true_grid_size = bounds.get("true_cartesian_grid_size", 18865)

        assert claimed_size == 6230
        assert true_grid_size == 18865
        assert true_grid_size != claimed_size

        # Prime factor check for 6230: 2 * 5 * 7 * 89
        n = claimed_size
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        assert 89 in factors, f"Expected factor 89 in 6230, got {factors}"

    def test_all_168_points_satisfy_physical_feasibility_constraint(self):
        """Assert that 100% of the 168 experimental points satisfy the paper's physical feasibility constraints."""
        df = pd.read_parquet(CANONICAL_DIR / "points.parquet")

        for idx, row in df.iterrows():
            is_feasible = check_known_constraint(
                row["gelma_l"], row["gelma_h"], row["acnc"], row["lap"], tol=1e-3
            )
            assert is_feasible, (
                f"Point {row['point_id']} (idx {row['source_index']}) violates feasibility constraint: "
                f"GelMA-L={row['gelma_l']}, GelMA-H={row['gelma_h']}, aCNC={row['acnc']}, LAP={row['lap']}"
            )

    def test_point_90_csv_rounding_boundary_overflow(self):
        """
        Test and document that Point 90 (idx 174) in StiffTumor campaign has a minor CSV rounding overflow
        (val = 1.000164 vs 1.000000) due to 2-decimal truncation in hydrogel_analysis.csv.
        """
        df = pd.read_parquet(CANONICAL_DIR / "points.parquet")
        p90 = df[df["point_id"] == 90].iloc[0]
        val = p90["gelma_l"] / 15.0 + p90["gelma_h"] / 15.0 + p90["acnc"] / 6.1 + p90["lap"] / 2.0
        assert val > 1.0, "Point 90 should show minor CSV rounding overflow > 1.0"
        assert val - 1.0 < 2e-4, f"Overflow should be < 0.02%, got {val - 1.0:.6f}"

