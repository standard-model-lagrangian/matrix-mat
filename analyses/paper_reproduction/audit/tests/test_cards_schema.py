"""
Tests for Section 9 card schema and validation utility.
Tier 3: Schema verification, fastjsonschema compilation, adversarial rejection, and directory audit.
"""

from pathlib import Path
import json
import pytest

from utils.card_validator import load_validator, validate_card_file, validate_cards_directory

AUDIT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = AUDIT_ROOT / "src" / "utils" / "card_schema.json"
CARDS_DIR = AUDIT_ROOT / "artifacts" / "cards"


@pytest.fixture(scope="module")
def valid_mock_card_data() -> dict:
    """Return a compliant mock card dictionary."""
    return {
        "task_id": "T00",
        "claim_reference": "Section 5.1 & Table S2: Ingestion of 168 experimental conditions and 43 prior conditions",
        "verdict": "SUPPORTED",
        "preregistration": {
            "hypothesis": "Canonical tables match upstream datasets with zero floating-point divergence",
            "primary_metric": "max_absolute_difference",
            "decision_rule": "max_absolute_difference == 0.0 across all 7 properties",
            "parameters": {"tolerance": 0.0},
        },
        "sensitivity_ranges": {
            "float_tolerance": [0.0, 1e-6],
        },
        "bias_direction": "neutral",
        "provenance_hashes": {
            "inputs": [
                {
                    "path": "Hydrogel_BO/hydrogel_analysis.csv",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                }
            ],
            "outputs": [
                {
                    "path": "artifacts/canonical/points.parquet",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                }
            ],
            "script": {
                "path": "src/data/ingest.py",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            },
        },
        "execution_metadata": {
            "execution_time_seconds": 0.42,
            "timestamp_utc": "2026-09-19T08:50:00Z",
            "seed": 847815772,
            "hardware": "Apple M4 (10 cores, 16 GB RAM)",
            "python_version": "3.13.14",
        },
        "metrics": {
            "points_count": 168,
            "prior_count": 43,
            "max_abs_diff": 0.0,
        },
        "summary": "Canonical data ingestion complete and verified with zero discrepancy.",
    }


def test_card_schema_compilation():
    """Verify Draft-07 card schema compiles successfully with fastjsonschema."""
    validator = load_validator(SCHEMA_PATH)
    assert callable(validator)


def test_t00_card_validation():
    """Verify artifacts/cards/T00/card.json validates against schema and provenance hashes."""
    validator = load_validator(SCHEMA_PATH)
    t00_path = CARDS_DIR / "T00" / "card.json"
    assert t00_path.is_file(), f"Missing {t00_path}"

    valid, errors = validate_card_file(
        t00_path,
        validator,
        verify_hashes=True,
        base_dir=AUDIT_ROOT,
    )
    assert valid, f"Card validation failed: {errors}"


def test_cards_directory_validation():
    """Verify validate_cards_directory passes on all generated cards."""
    passed, failed, errors = validate_cards_directory(
        CARDS_DIR,
        schema_path=SCHEMA_PATH,
        verify_hashes=True,
        base_dir=AUDIT_ROOT,
    )
    assert failed == 0, f"Card directory validation failed: {errors}"
    assert passed >= 1


def test_valid_mock_card_passes_validator(valid_mock_card_data: dict):
    """Verify valid mock card passes fastjsonschema compilation."""
    validator = load_validator(SCHEMA_PATH)
    validator(valid_mock_card_data)


@pytest.mark.parametrize(
    "missing_field",
    [
        "task_id",
        "claim_reference",
        "verdict",
        "preregistration",
        "sensitivity_ranges",
        "bias_direction",
        "provenance_hashes",
        "execution_metadata",
        "metrics",
        "summary",
    ],
)
def test_adversarial_missing_required_fields(valid_mock_card_data: dict, missing_field: str):
    """Verify omitting any top-level required field triggers validation error."""
    validator = load_validator(SCHEMA_PATH)
    mutated = dict(valid_mock_card_data)
    del mutated[missing_field]
    with pytest.raises(Exception):
        validator(mutated)


@pytest.mark.parametrize(
    "bad_task_id",
    ["Task00", "00", "T0", "T1000", "T00_extra", "t00", "T001a"],
)
def test_adversarial_invalid_task_id_pattern(valid_mock_card_data: dict, bad_task_id: str):
    """Verify task_id pattern ^T[0-9]{2}[a-z]?$ rejects invalid formats."""
    validator = load_validator(SCHEMA_PATH)
    mutated = dict(valid_mock_card_data)
    mutated["task_id"] = bad_task_id
    with pytest.raises(Exception):
        validator(mutated)


@pytest.mark.parametrize(
    "bad_verdict",
    ["PLAUSIBLE", "CONFIRMED", "TRUE", "FALSE", "PASSED", "FAILED"],
)
def test_adversarial_invalid_verdict_enum(valid_mock_card_data: dict, bad_verdict: str):
    """Verify verdict outside the 4 allowed values is rejected."""
    validator = load_validator(SCHEMA_PATH)
    mutated = dict(valid_mock_card_data)
    mutated["verdict"] = bad_verdict
    with pytest.raises(Exception):
        validator(mutated)


@pytest.mark.parametrize(
    "bad_bias",
    ["positive", "negative", "neutral_biased", "author_friendly"],
)
def test_adversarial_invalid_bias_direction_enum(valid_mock_card_data: dict, bad_bias: str):
    """Verify bias_direction outside the 5 allowed values is rejected."""
    validator = load_validator(SCHEMA_PATH)
    mutated = dict(valid_mock_card_data)
    mutated["bias_direction"] = bad_bias
    with pytest.raises(Exception):
        validator(mutated)


@pytest.mark.parametrize(
    "bad_sha256",
    [
        "short_hash",
        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b8555",
        "z3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ],
)
def test_adversarial_malformed_sha256(valid_mock_card_data: dict, bad_sha256: str):
    """Verify malformed SHA-256 digests in provenance_hashes are rejected."""
    validator = load_validator(SCHEMA_PATH)
    mutated = json.loads(json.dumps(valid_mock_card_data))
    mutated["provenance_hashes"]["inputs"][0]["sha256"] = bad_sha256
    with pytest.raises(Exception):
        validator(mutated)
