"""
Tier 4 Tests: Deterministic Seed Derivation and Randomness Protocol.
Validates:
- env/seeds.yaml structure and master seed 20260919
- Deterministic derivation formula for all 17 tasks (T00-T12)
- Positive 31-bit integers and pairwise uniqueness
- Child seed sequence spawning with np.random.SeedSequence (>= 200 seeds)
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest
import yaml

from utils.seeds import derive_seed, get_task_seed, verify_seeds_config, MASTER_SEED_DEFAULT

AUDIT_ROOT = Path(__file__).resolve().parent.parent
SEEDS_CONFIG_PATH = AUDIT_ROOT / "env" / "seeds.yaml"

EXPECTED_DERIVATION_MAP = {
    "T00": 847815772,
    "T01": 1442961952,
    "T02a": 1002994493,
    "T02b": 39582884,
    "T03": 289516097,
    "T04": 1379629476,
    "T05": 1909588537,
    "T06": 1862010923,
    "T07a": 141364964,
    "T07b": 1573974761,
    "T07c": 209432717,
    "T07d": 705083881,
    "T08": 668166719,
    "T09": 2122623603,
    "T10": 1268654566,
    "T11": 1530637490,
    "T12": 1026428776,
    "T13": 559357057,
}


def test_seeds_config_verification():
    """Verify env/seeds.yaml passes deterministic derivation checks for all tasks."""
    assert SEEDS_CONFIG_PATH.is_file(), f"Missing {SEEDS_CONFIG_PATH}"
    result = verify_seeds_config(SEEDS_CONFIG_PATH)
    assert result is True


def test_seed_derivation_deterministic():
    """Verify derive_seed is pure, deterministic, and returns a 31-bit positive int."""
    s1 = derive_seed(MASTER_SEED_DEFAULT, "T00")
    s2 = derive_seed(MASTER_SEED_DEFAULT, "T00")
    assert s1 == s2
    assert 0 <= s1 < (2**31 - 1)
    assert s1 == 847815772

    # Different tasks must yield different seeds
    s_t01 = derive_seed(MASTER_SEED_DEFAULT, "T01")
    assert s_t01 != s1


def test_all_18_task_seeds_match_oracle_derivation():
    """Verify each of the 18 tasks matches the explicit oracle derivation map."""
    for task_id, expected_val in EXPECTED_DERIVATION_MAP.items():
        actual = derive_seed(MASTER_SEED_DEFAULT, task_id)
        assert actual == expected_val, f"Seed mismatch for {task_id}: expected {expected_val}, got {actual}"


def test_task_seeds_uniqueness():
    """Assert all 18 derived task seeds are pairwise distinct with zero collision."""
    seeds = [derive_seed(MASTER_SEED_DEFAULT, t) for t in EXPECTED_DERIVATION_MAP]
    assert len(seeds) == len(set(seeds)), "Duplicate task seed detected!"


def test_multiseed_child_sequence_spawning():
    """
    Verify that multi-seed sweeps (e.g. T02b scarred-skin benchmark >= 200 seeds)
    spawn deterministic, unique child seeds using np.random.SeedSequence.
    """
    base_seed = derive_seed(MASTER_SEED_DEFAULT, "T02b")
    ss1 = np.random.SeedSequence(base_seed)
    children1 = ss1.spawn(200)

    child_seeds_1 = [int(c.generate_state(1)[0]) for c in children1]
    assert len(child_seeds_1) == 200
    assert len(set(child_seeds_1)) == 200, "Spawned child seeds must be mutually distinct"

    # Strict repeatability check
    ss2 = np.random.SeedSequence(base_seed)
    children2 = ss2.spawn(200)
    child_seeds_2 = [int(c.generate_state(1)[0]) for c in children2]
    assert child_seeds_1 == child_seeds_2, "Child seed generation must be 100% deterministic"
