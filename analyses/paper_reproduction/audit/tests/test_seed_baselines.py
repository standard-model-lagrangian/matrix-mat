"""
Tests for Task T13: AL/BO Seed Dataset Collection Methods Benchmark ("Dumb Baselines").
Validates:
- Card schema compliance (Section 9 Draft-07)
- Provenance cryptographic hashes for inputs, outputs, and script
- Publication figure existence (PDF and PNG)
- Deterministic seed derivation (base seed 559357057)
- Sampler functionality, output dimensions, and physical feasibility
- Invariant metrics comparing dumb baselines to Chen et al. active learning
"""

from pathlib import Path
import json
import numpy as np
import pytest

from utils.seeds import get_task_seed, derive_seed, MASTER_SEED_DEFAULT
from utils.hashing import compute_sha256
from utils.card_validator import load_validator, validate_card_file

AUDIT_ROOT = Path(__file__).resolve().parent.parent
CARD_PATH = AUDIT_ROOT / "artifacts" / "cards" / "T13" / "card.json"
SCHEMA_PATH = AUDIT_ROOT / "src" / "utils" / "card_schema.json"
FIG_PDF = AUDIT_ROOT / "report" / "figures" / "fig_seed_baselines.pdf"
FIG_PNG = AUDIT_ROOT / "report" / "figures" / "fig_seed_baselines.png"


def test_t13_seed_derivation():
    """Verify Task T13 base seed matches the deterministic derivation rule."""
    expected_seed = derive_seed(MASTER_SEED_DEFAULT, "T13")
    assert expected_seed == 559357057
    actual_seed = get_task_seed("T13", AUDIT_ROOT / "env" / "seeds.yaml")
    assert actual_seed == expected_seed


def test_t13_card_schema_and_hashes():
    """Verify Section 9 Draft-07 schema compliance and provenance SHA-256 hashes."""
    assert CARD_PATH.is_file(), f"Card not found at {CARD_PATH}"
    validator = load_validator(SCHEMA_PATH)
    valid, errors = validate_card_file(CARD_PATH, validator, verify_hashes=True, base_dir=AUDIT_ROOT)
    assert valid, f"Card validation errors: {errors}"


def test_t13_figures_exist():
    """Verify publication figures exist and are non-empty."""
    assert FIG_PDF.is_file(), f"PDF figure missing at {FIG_PDF}"
    assert FIG_PNG.is_file(), f"PNG figure missing at {FIG_PNG}"
    assert FIG_PDF.stat().st_size > 1000
    assert FIG_PNG.stat().st_size > 1000


def test_t13_samplers_feasibility_and_dimensions():
    """Test that all 7 sampling methods return feasible points and correct dimensions."""
    from audit.seed_baselines import (
        sample_uniform_random,
        sample_lhs,
        sample_sobol,
        sample_halton,
        sample_d_optimal,
        sample_i_optimal,
        sample_ccd,
        is_feasible,
    )

    rng = np.random.default_rng(42)
    test_n = 20

    samples = {
        "Random": sample_uniform_random(test_n, rng),
        "LHS": sample_lhs(test_n, 42),
        "Sobol": sample_sobol(test_n, 42),
        "Halton": sample_halton(test_n, 42),
        "D-Optimal": sample_d_optimal(test_n, 42, pool_size=500),
        "I-Optimal": sample_i_optimal(test_n, 42, pool_size=500),
        "CCD": sample_ccd(test_n, 42),
    }

    for name, s in samples.items():
        assert s.shape == (test_n, 5), f"{name} returned shape {s.shape}, expected ({test_n}, 5)"
        feas = is_feasible(s)
        assert np.all(feas), f"{name} returned infeasible formulations: {np.sum(~feas)} points violated constraints"


def test_t13_benchmark_invariants():
    """Verify core audit invariants comparing passive dumb baselines to active learning."""
    with open(CARD_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]
    chen_closed = metrics["chen_datasets"]["Chen_ClosedLoop_N100"]
    chen_prior = metrics["chen_datasets"]["Chen_Prior_N43"]
    summary_n100 = metrics["benchmark_summary_n100"]

    # 1. Chen Closed-Loop hits vs Prior hits
    assert chen_closed["hits_eps"]["hits_15"] == 34, "Chen Closed-Loop must have 34 hits at d <= 0.15"
    assert chen_closed["hits_eps"]["hits_05"] == 8, "Chen Closed-Loop must have 8 hits at d <= 0.05"
    assert chen_prior["hits_eps"]["hits_15"] == 0, "Chen Prior (N=43) must have 0 hits at d <= 0.15"

    # 2. Dumb baselines at N=100 achieve < 2.0 hits on average at d <= 0.15 (< 0.5 for space-filling, 1.25 for CCD)
    for method_name, vals in summary_n100.items():
        assert vals["mean_hits_015"] < 2.0, f"{method_name} achieved unexpectedly high hits: {vals['mean_hits_015']}"
        assert vals["mean_hits_005"] < 0.05, f"{method_name} achieved unexpectedly high tight hits: {vals['mean_hits_005']}"
        # Mean minimum distance should be well above 0.10
        assert vals["mean_min_dist"] >= 0.10, f"{method_name} mean min dist too small: {vals['mean_min_dist']}"

    # 3. Digital twin property hits reflect mechanical response degeneracy (between 2 and 8 per 100)
    for method_name, vals in summary_n100.items():
        assert 1.0 <= vals["mean_property_hits"] <= 9.0, (
            f"{method_name} property hits outside expected degeneracy range [1.0, 9.0]: {vals['mean_property_hits']}"
        )
