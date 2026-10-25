"""
Unit and Integration Test Suite for Milestone 2: Tasks T01 & T04.
Chen et al. (Adv. Funct. Mater. 2026, 36:e75639) Reproduction & Adversarial Audit.

Validates:
1. GPSurrogate (BoTorch / GPyTorch SingleTaskGP wrapper across 8 configurations).
2. Task T01: Prospective One-Step-Ahead Sequential Surrogate Evaluation (t=1..25),
   soft-regime RMSE vs target tolerance, coverage, standardized residuals, and card validation.
3. Task T04: Prior Imputation Fingerprint, LAP constant-value pathology (81.4%),
   exposure time restriction (67.4%), ARD lengthscale degeneracy, propagated error ratio,
   and card validation.
4. Headline figure publication artifact integrity: report/figures/fig_soft_regime.pdf.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from models.gp_surrogate import GPSurrogate, DEFAULT_BOUNDS
from audit.prospective import run_sequential_evaluation, run_t01_audit, FEATURES
from audit.imputation import run_t04_audit
from utils.seeds import get_task_seed
from utils.card_validator import load_validator, validate_card_file
from utils.hashing import compute_sha256

AUDIT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = AUDIT_ROOT / "artifacts" / "canonical"
CARDS_DIR = AUDIT_ROOT / "artifacts" / "cards"
FIGURES_DIR = AUDIT_ROOT / "report" / "figures"
SCHEMA_PATH = AUDIT_ROOT / "src" / "utils" / "card_schema.json"


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def points_df() -> pd.DataFrame:
    p = CANONICAL_DIR / "points.parquet"
    assert p.is_file(), f"Missing {p}"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def prior_df() -> pd.DataFrame:
    p = CANONICAL_DIR / "prior.parquet"
    assert p.is_file(), f"Missing {p}"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def card_validator():
    return load_validator(SCHEMA_PATH)


# ==============================================================================
# 1. GPSurrogate Architecture & Invariants Tests
# ==============================================================================

class TestGPSurrogateArchitecture:
    """Verify BoTorch/GPyTorch wrapper behaviors, configurations, and edge cases."""

    @pytest.mark.parametrize("kernel", ["matern52", "rbf"])
    @pytest.mark.parametrize("ard", [True, False])
    @pytest.mark.parametrize("noise_model", ["learned", "replicate"])
    def test_all_8_configurations_instantiate_and_fit(self, kernel, ard, noise_model):
        """All 8 configurations must instantiate, fit, and predict without error."""
        np.random.seed(42)
        X = np.random.uniform(0.0, 1.0, size=(25, 5))
        y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + np.random.normal(0, 0.05, size=25)

        gp = GPSurrogate(
            kernel=kernel,
            ard=ard,
            noise_model=noise_model,
            fixed_noise_std=0.1 if noise_model == "replicate" else None,
            random_seed=42,
        )
        gp.fit(X, y)

        mu, sigma = gp.predict(X[:5])
        assert mu.shape == (5,)
        assert sigma.shape == (5,)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0.0)

        ls = gp.get_lengthscales()
        expected_ls_len = 5 if ard else 1
        assert len(ls) == expected_ls_len
        assert np.all(ls > 0.0)

        outscale = gp.get_outputscale()
        assert np.isfinite(outscale) and outscale > 0.0

        nvar = gp.get_noise_variance()
        assert np.isfinite(nvar) and nvar >= 0.0

    def test_invalid_configurations_rejected(self):
        """Invalid kernel or noise model choices must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel_type"):
            GPSurrogate(kernel="linear")

        with pytest.raises(ValueError, match="Unknown noise_model"):
            GPSurrogate(noise_model="heteroskedastic")

    def test_unfitted_model_calls_raise_runtime_error(self):
        """Calling predict or inspection methods prior to fit must raise RuntimeError."""
        gp = GPSurrogate()
        with pytest.raises(RuntimeError, match="Model is not fitted"):
            gp.predict(np.zeros((2, 5)))

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            gp.get_lengthscales()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            gp.get_outputscale()

        with pytest.raises(RuntimeError, match="Model is not fitted"):
            gp.get_noise_variance()

    def test_normalization_and_physical_scale_invariants(self):
        """Normalization maps input bounds to [0, 1] and predictions are unstandardized."""
        bounds = np.array([
            [10.0, 100.0],
            [0.0, 5.0],
            [1.0, 9.0],
            [0.0, 2.0],
            [0.0, 1.0],
        ])
        gp = GPSurrogate(bounds=bounds)
        test_x = np.array([[10.0, 0.0, 1.0, 0.0, 0.0], [100.0, 5.0, 9.0, 2.0, 1.0]])
        norm_x = gp.normalize_x(test_x)
        np.testing.assert_allclose(norm_x[0], [0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(norm_x[1], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_deterministic_seed_reproducibility(self):
        """Fitting twice with identical random seed must yield identical predictions."""
        np.random.seed(123)
        X = np.random.uniform(0.0, 1.0, size=(20, 5))
        y = X[:, 0] * 2.0 + np.random.normal(0, 0.1, size=20)

        gp1 = GPSurrogate(random_seed=9999)
        gp1.fit(X, y)
        mu1, sig1 = gp1.predict(X)

        gp2 = GPSurrogate(random_seed=9999)
        gp2.fit(X, y)
        mu2, sig2 = gp2.predict(X)

        np.testing.assert_allclose(mu1, mu2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sig1, sig2, rtol=1e-6, atol=1e-6)


# ==============================================================================
# 2. Task T01: Prospective Surrogate Evaluation Tests
# ==============================================================================

class TestT01ProspectiveEvaluation:
    """Verify prospective sequential GP evaluation methodology and quantitative findings."""

    def test_closed_loop_batch_chronology(self, points_df):
        """Closed-loop points must consist of 25 batches of 4 points partitioned chronologically."""
        cl = points_df[points_df["batch_type"] == "closed_loop"]
        assert len(cl) == 100
        batch_ids = sorted(cl["batch_id"].unique())
        assert batch_ids == list(range(1, 26))

        for b_id, grp in cl.groupby("batch_id"):
            assert len(grp) == 4

        # Chronological campaign order
        camps_by_batch = [cl[cl["batch_id"] == b]["campaign"].iloc[0] for b in range(1, 26)]
        assert camps_by_batch[:5] == ["HealthySkin"] * 5
        assert camps_by_batch[5:13] == ["ScarredSkin"] * 8
        assert camps_by_batch[13:20] == ["SoftTumor"] * 7
        assert camps_by_batch[20:25] == ["StiffTumor"] * 5

    def test_soft_regime_subpopulation(self, points_df):
        """Soft regime (E < 3.0 kPa) must contain exactly 48 closed-loop formulations."""
        cl = points_df[points_df["batch_type"] == "closed_loop"]
        soft_pts = cl[cl["modulus"] < 3.0]
        assert len(soft_pts) == 48

        counts = soft_pts["campaign"].value_counts().to_dict()
        assert counts.get("SoftTumor") == 28
        assert counts.get("HealthySkin") == 19
        assert counts.get("ScarredSkin") == 1

    def test_prospective_evaluation_sensitivity_sweep(self, points_df, prior_df):
        """Test sequential prospective evaluation on baseline config and check soft RMSE excess."""
        seed = get_task_seed("T01")
        res = run_sequential_evaluation(
            points_df=points_df,
            prior_df=prior_df,
            target_col="modulus",
            kernel="matern52",
            ard=True,
            noise_model="learned",
            seed=seed,
        )

        assert res["n_soft"] == 48
        assert len(res["y_true"]) == 100
        assert len(res["mu_pred"]) == 100
        assert len(res["sigma_pred"]) == 100

        # Primary audit claim: Soft RMSE (~1.90 kPa) is >10x the target tolerance (0.15 kPa)
        soft_rmse = res["rmse_soft"]
        assert 1.5 <= soft_rmse <= 2.5
        ratio_to_tol = soft_rmse / 0.15
        assert ratio_to_tol > 10.0, f"Expected RMSE/tol > 10, got {ratio_to_tol}"

        # Uncertainty calibration: empirical 95% interval coverage in soft regime
        cov_95 = res["coverage_95_soft"]
        assert 0.80 <= cov_95 <= 1.0, f"Expected 95% coverage between 80% and 100%, got {cov_95}"

    def test_t01_card_schema_and_provenance(self, card_validator):
        """T01 card must validate against Section 9 Draft-07 schema and disk file hashes."""
        card_file = CARDS_DIR / "T01" / "card.json"
        assert card_file.is_file(), f"Missing {card_file}"

        valid, errors = validate_card_file(
            card_file,
            card_validator,
            verify_hashes=True,
            base_dir=AUDIT_ROOT,
        )
        assert valid, f"Card T01 validation failed: {errors}"

        with open(card_file, "r", encoding="utf-8") as f:
            card = json.load(f)

        assert card["task_id"] == "T01"
        assert card["verdict"] == "WEAKENED"
        assert card["bias_direction"] == "conservative"

        # Check sensitivity grid contains all 8 configurations
        sens = card["metrics"]["sensitivity_grid"]
        assert len(sens) == 8
        cfg_names = [s["config"] for s in sens]
        assert "matern52_ardTrue_learned" in cfg_names
        assert "rbf_ardFalse_replicate" in cfg_names

        # Soft RMSE / tolerance ratio across all 8 configurations must exceed 10x
        for s in sens:
            assert s["rmse_div_tol_soft"] >= 10.0

    def test_headline_figure_artifacts_exist(self):
        """Headline publication figure fig_soft_regime.pdf and .png must exist and be non-empty."""
        pdf_path = FIGURES_DIR / "fig_soft_regime.pdf"
        png_path = FIGURES_DIR / "fig_soft_regime.png"

        assert pdf_path.is_file(), f"Missing {pdf_path}"
        assert png_path.is_file(), f"Missing {png_path}"

        assert pdf_path.stat().st_size > 5000, "PDF figure file is suspiciously small"
        assert png_path.stat().st_size > 10000, "PNG figure file is suspiciously small"

        # Check valid PDF magic header
        with open(pdf_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF", "Not a valid PDF file"


# ==============================================================================
# 3. Task T04: Imputation Fingerprint & ARD Degeneracy Tests
# ==============================================================================

class TestT04ImputationFingerprint:
    """Verify prior dataset imputation artifacts, LAP degeneracy, and ARD divergence."""

    def test_prior_dataset_invariants(self, prior_df):
        """Prior dataset has 43 rows, index 2 missing, with heavy LAP concentration."""
        assert len(prior_df) == 43
        assert 2 not in prior_df["source_index"].values

        # LAP identically 0.10 wt% in 35/43 rows (81.4%)
        lap_vc = prior_df["lap"].value_counts()
        assert lap_vc.loc[0.10] == 35
        pct_010 = 35 / 43 * 100.0
        assert abs(pct_010 - 81.395) < 0.01

        # Exposure time restricted to {30, 90} s in 29/43 rows (67.4%)
        exp_restricted = prior_df["exposure_time"].isin([30.0, 90.0]).sum()
        assert exp_restricted == 29
        pct_exp = 29 / 43 * 100.0
        assert abs(pct_exp - 67.44) < 0.1

    def test_propagated_imputation_error_scale(self, prior_df):
        """Nominal R^2 = 0.80 on imputed modulus propagates ~1.43 kPa RMSE (~9.5x target tolerance)."""
        std_e = float(np.std(prior_df["imputed_modulus"].values))
        prop_rmse = std_e * np.sqrt(1.0 - 0.80)
        assert 1.30 <= prop_rmse <= 1.55

        soft_tol = 0.15
        ratio = prop_rmse / soft_tol
        assert 8.5 <= ratio <= 10.5, f"Expected ratio ~9.5x, got {ratio:.2f}"

    def test_ard_lengthscale_degeneracy_contrast(self, prior_df, points_df):
        """ARD lengthscales on prior exhibit severe divergence/pegging compared to closed loop."""
        seed = get_task_seed("T04")

        # Fit ARD on Prior
        gp_prior = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=seed)
        gp_prior.fit(prior_df[FEATURES].values, prior_df["imputed_modulus"].values)
        ls_prior = gp_prior.get_lengthscales()

        # Fit ARD on Closed Loop
        cl = points_df[points_df["batch_type"] == "closed_loop"]
        gp_cl = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=seed)
        gp_cl.fit(cl[FEATURES].values, cl["modulus"].values)
        ls_cl = gp_cl.get_lengthscales()

        # Closed-loop lengthscales are well-conditioned (0.05 to 3.0 in unit cube)
        assert np.all(ls_cl > 0.05)
        assert np.all(ls_cl < 3.0)

        # Prior lengthscale on GelMA-L pegs to upper boundary (> 10.0 in unit cube)
        gelma_l_idx = FEATURES.index("gelma_l")
        assert ls_prior[gelma_l_idx] > 10.0, f"Expected GelMA-L lengthscale > 10, got {ls_prior[gelma_l_idx]}"

        # Prior lengthscale on aCNC collapses pathologically (< 1e-4)
        acnc_idx = FEATURES.index("acnc")
        assert ls_prior[acnc_idx] < 1e-4, f"Expected aCNC lengthscale < 1e-4, got {ls_prior[acnc_idx]}"

    def test_t04_card_schema_and_provenance(self, card_validator):
        """T04 card must validate against Section 9 Draft-07 schema and disk file hashes."""
        card_file = CARDS_DIR / "T04" / "card.json"
        assert card_file.is_file(), f"Missing {card_file}"

        valid, errors = validate_card_file(
            card_file,
            card_validator,
            verify_hashes=True,
            base_dir=AUDIT_ROOT,
        )
        assert valid, f"Card T04 validation failed: {errors}"

        with open(card_file, "r", encoding="utf-8") as f:
            card = json.load(f)

        assert card["task_id"] == "T04"
        assert card["verdict"] == "WEAKENED"
        assert card["bias_direction"] == "conservative"

        metrics = card["metrics"]
        assert metrics["prior_row_count"] == 43
        assert metrics["missing_prior_index"] == 2
        assert metrics["lap_constant_pct"] == 81.4
        assert metrics["exposure_time_restricted_pct"] == 67.4
        assert metrics["error_to_tolerance_ratio"] >= 8.0
        assert "gelma_l" in metrics["divergent_prior_features"]
