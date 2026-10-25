"""
Tier 5 Adversarial Negative Testing Suite for Verification Toolchain.
Challenger 2 (Milestone 1: Verification Toolchain & Negative Testing).

Proves empirically that:
1. Gate 2 (hashing.py / make verify-canonical): fails when manifest hash is corrupted, file is missing, manifest is empty or malformed.
2. Gate 3 (card_validator.py / make verify-cards): fails when required fields are missing, enums are invalid, types/regex mismatch, or JSON syntax is corrupted.
3. Gate 4 (card_validator.py --verify-hashes / make verify-hashes): fails when input, output, or script SHA-256 digests mismatch or files are missing.
4. Gate 5 (seeds.py / make verify-seeds): fails when task seeds diverge, master seed changes, or task_seeds config is invalid.
5. End-to-end cascade (make verify): terminates with non-zero exit code upon any upstream failure and passes with code 0 when clean.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import pytest
import yaml

from utils.card_validator import load_validator, validate_card_file, validate_cards_directory
from utils.hashing import compute_sha256, verify_manifest
from utils.seeds import derive_seed, verify_seeds_config, MASTER_SEED_DEFAULT

AUDIT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = AUDIT_ROOT / "artifacts" / "canonical" / "manifest.sha256"
CARDS_DIR = AUDIT_ROOT / "artifacts" / "cards"
T00_CARD_PATH = CARDS_DIR / "T00" / "card.json"
SEEDS_CONFIG_PATH = AUDIT_ROOT / "env" / "seeds.yaml"
SCHEMA_PATH = AUDIT_ROOT / "src" / "utils" / "card_schema.json"
CLEAN_MANIFEST = (
    "6660f21606d371feb5a8e39b0e4d8a8926611efd90b095b901bd6dc5f46804cc  artifacts/canonical/points.parquet\n"
    "ddcf4910feac2607b95528cf0053a707d862190f8308d848cd2f76d9e8fe20d1  artifacts/canonical/prior.parquet\n"
    "5e6cf98cafdff8e94ab8066f87dc6737e266419c10acbe3064adb1e60d6696ab  artifacts/canonical/optima.parquet\n"
    "b233786ea18f91dc282c2729557fb15da33a607b6a1772e6f5658d730aefdb58  artifacts/canonical/paper_claims.yaml\n"
)


# ==============================================================================
# 1. Negative Tests for Gate 2 (Manifest & Cryptographic Hashing)
# ==============================================================================

class TestNegativeManifestVerification:
    """Adversarial negative tests for hashing.py and manifest verification."""

    def test_corrupted_hash_raises_value_error(self, tmp_path: Path):
        """Assert that modifying a single character in a hash triggers ValueError mismatch."""
        orig_content = MANIFEST_PATH.read_text(encoding="utf-8")
        lines = orig_content.strip().splitlines()
        first_hash, first_path = lines[0].split(maxsplit=1)
        
        # Corrupt the first character of the hash
        corrupted_hash = ("0" if first_hash[0] != "0" else "1") + first_hash[1:]
        corrupted_line = f"{corrupted_hash}  {first_path}"
        corrupted_content = "\n".join([corrupted_line] + lines[1:]) + "\n"
        
        corrupted_manifest = tmp_path / "corrupted_manifest.sha256"
        corrupted_manifest.write_text(corrupted_content, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            verify_manifest(corrupted_manifest, base_dir=AUDIT_ROOT)
        
        assert "Manifest verification FAILED" in str(exc_info.value)
        assert "MISMATCH:" in str(exc_info.value)
        assert first_path in str(exc_info.value)

    def test_missing_file_in_manifest_raises_value_error(self, tmp_path: Path):
        """Assert that an entry for a non-existent file triggers a MISSING error."""
        dummy_line = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  artifacts/canonical/non_existent_file.parquet\n"
        corrupted_manifest = tmp_path / "missing_file_manifest.sha256"
        corrupted_manifest.write_text(dummy_line, encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            verify_manifest(corrupted_manifest, base_dir=AUDIT_ROOT)
        
        assert "Manifest verification FAILED" in str(exc_info.value)
        assert "MISSING:" in str(exc_info.value)
        assert "non_existent_file.parquet" in str(exc_info.value)

    def test_empty_manifest_raises_value_error(self, tmp_path: Path):
        """Assert that an empty manifest raises ValueError."""
        empty_manifest = tmp_path / "empty.sha256"
        empty_manifest.write_text("", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            verify_manifest(empty_manifest, base_dir=AUDIT_ROOT)
        assert "Manifest file is empty" in str(exc_info.value)

    def test_malformed_manifest_line_raises_value_error(self, tmp_path: Path):
        """Assert that a line with missing path component raises ValueError."""
        bad_manifest = tmp_path / "malformed.sha256"
        bad_manifest.write_text("6660f21606d371feb5a8e39b0e4d8a8926611efd90b095b901bd6dc5f46804cc\n", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            verify_manifest(bad_manifest, base_dir=AUDIT_ROOT)
        assert "Malformed manifest line" in str(exc_info.value)

    def test_nonexistent_manifest_file_raises_file_not_found(self, tmp_path: Path):
        """Assert that verifying a nonexistent manifest raises FileNotFoundError."""
        fake_manifest = tmp_path / "ghost.sha256"
        with pytest.raises(FileNotFoundError):
            verify_manifest(fake_manifest, base_dir=AUDIT_ROOT)


# ==============================================================================
# 2. Negative Tests for Gate 3 (Card Schema Validation)
# ==============================================================================

class TestNegativeCardSchemaValidation:
    """Adversarial negative tests for Section 9 card schema compliance."""

    @pytest.fixture
    def valid_card_data(self) -> dict:
        """Load the real T00 card as base for mutation."""
        with open(T00_CARD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def validator(self):
        return load_validator(SCHEMA_PATH)

    @pytest.mark.parametrize(
        "field",
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
    def test_missing_required_top_level_fields_fail(self, valid_card_data: dict, validator, tmp_path: Path, field: str):
        """Assert that omitting any required field fails validate_card_file."""
        mutated = dict(valid_card_data)
        del mutated[field]
        card_file = tmp_path / f"missing_{field}.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator)
        assert is_valid is False
        assert len(errors) > 0
        assert field in errors[0]

    @pytest.mark.parametrize(
        "bad_verdict",
        ["CONFIRMED", "PLAUSIBLE", "VALIDATED", "PASSED", "REJECTED", "UNKNOWN", ""],
    )
    def test_invalid_verdict_values_rejected(self, valid_card_data: dict, validator, tmp_path: Path, bad_verdict: str):
        """Assert that non-standard verdicts outside 4 allowed enums fail schema validation."""
        mutated = dict(valid_card_data)
        mutated["verdict"] = bad_verdict
        card_file = tmp_path / "bad_verdict.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator)
        assert is_valid is False
        assert any("verdict" in e for e in errors)

    @pytest.mark.parametrize(
        "bad_bias",
        ["positive", "negative", "pro-author", "biased", "left", ""],
    )
    def test_invalid_bias_direction_values_rejected(self, valid_card_data: dict, validator, tmp_path: Path, bad_bias: str):
        """Assert that invalid bias directions are rejected by schema."""
        mutated = dict(valid_card_data)
        mutated["bias_direction"] = bad_bias
        card_file = tmp_path / "bad_bias.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator)
        assert is_valid is False
        assert any("bias_direction" in e for e in errors)

    @pytest.mark.parametrize(
        "bad_task_id",
        ["T0", "T000", "task_00", "T00_run", "t00", "T999", ""],
    )
    def test_invalid_task_id_regex_rejected(self, valid_card_data: dict, validator, tmp_path: Path, bad_task_id: str):
        """Assert that task_id not matching ^T[0-9]{2}[a-z]?$ regex is rejected."""
        mutated = dict(valid_card_data)
        mutated["task_id"] = bad_task_id
        card_file = tmp_path / "bad_task_id.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator)
        assert is_valid is False

    def test_malformed_json_syntax_fails(self, tmp_path: Path, validator):
        """Assert that corrupted JSON syntax triggers decoding error."""
        bad_file = tmp_path / "corrupted_syntax.json"
        bad_file.write_text('{"task_id": "T00", unclosed: ', encoding="utf-8")

        is_valid, errors = validate_card_file(bad_file, validator)
        assert is_valid is False
        assert any("Invalid JSON syntax" in e for e in errors)

    def test_negative_execution_time_rejected(self, valid_card_data: dict, validator, tmp_path: Path):
        """Assert that negative execution time violates minimum 0 constraint."""
        mutated = json.loads(json.dumps(valid_card_data))
        mutated["execution_metadata"]["execution_time_seconds"] = -1.5
        card_file = tmp_path / "negative_time.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator)
        assert is_valid is False
        assert any("execution_time_seconds" in e for e in errors)


# ==============================================================================
# 3. Negative Tests for Gate 4 (Card Provenance Hashes)
# ==============================================================================

class TestNegativeCardProvenanceHashes:
    """Adversarial negative tests for card provenance hash verification."""

    @pytest.fixture
    def valid_card_data(self) -> dict:
        with open(T00_CARD_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def validator(self):
        return load_validator(SCHEMA_PATH)

    def test_corrupted_output_hash_fails_verification(self, valid_card_data: dict, validator, tmp_path: Path):
        """Assert that tamper in output sha256 is detected during hash verification."""
        mutated = json.loads(json.dumps(valid_card_data))
        # Modify points.parquet hash
        mutated["provenance_hashes"]["outputs"][0]["sha256"] = "0" * 64
        card_file = tmp_path / "card_corrupt_output_hash.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator, verify_hashes=True, base_dir=AUDIT_ROOT)
        assert is_valid is False
        assert any("Output hash mismatch" in e for e in errors)

    def test_corrupted_input_hash_fails_verification(self, valid_card_data: dict, validator, tmp_path: Path):
        """Assert that tamper in input sha256 is detected."""
        mutated = json.loads(json.dumps(valid_card_data))
        mutated["provenance_hashes"]["inputs"][0]["sha256"] = "1" * 64
        card_file = tmp_path / "card_corrupt_input_hash.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator, verify_hashes=True, base_dir=AUDIT_ROOT)
        assert is_valid is False
        assert any("Input hash mismatch" in e for e in errors)

    def test_corrupted_script_hash_fails_verification(self, valid_card_data: dict, validator, tmp_path: Path):
        """Assert that tamper in script sha256 is detected."""
        mutated = json.loads(json.dumps(valid_card_data))
        mutated["provenance_hashes"]["script"]["sha256"] = "2" * 64
        card_file = tmp_path / "card_corrupt_script_hash.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator, verify_hashes=True, base_dir=AUDIT_ROOT)
        assert is_valid is False
        assert any("Script hash mismatch" in e for e in errors)

    def test_missing_provenance_file_fails_verification(self, valid_card_data: dict, validator, tmp_path: Path):
        """Assert that pointing provenance to missing file triggers missing error."""
        mutated = json.loads(json.dumps(valid_card_data))
        mutated["provenance_hashes"]["outputs"][0]["path"] = "artifacts/canonical/non_existent.parquet"
        card_file = tmp_path / "card_missing_file.json"
        card_file.write_text(json.dumps(mutated), encoding="utf-8")

        is_valid, errors = validate_card_file(card_file, validator, verify_hashes=True, base_dir=AUDIT_ROOT)
        assert is_valid is False
        assert any("Provenance output file missing" in e for e in errors)


# ==============================================================================
# 4. Negative Tests for Gate 5 (Deterministic Seed Derivation)
# ==============================================================================

class TestNegativeSeedDerivation:
    """Adversarial negative tests for seeds.yaml derivation verification."""

    @pytest.fixture
    def valid_seeds_data(self) -> dict:
        with open(SEEDS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_tampered_task_seed_value_fails(self, valid_seeds_data: dict, tmp_path: Path):
        """Assert that changing any task seed triggers verification failure."""
        mutated = dict(valid_seeds_data)
        mutated["task_seeds"] = dict(valid_seeds_data["task_seeds"])
        mutated["task_seeds"]["T00"] = 999999999
        config_path = tmp_path / "corrupted_seeds.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(mutated, f)

        with pytest.raises(ValueError) as exc_info:
            verify_seeds_config(config_path)
        
        assert "Seed verification FAILED" in str(exc_info.value)
        assert "Task T00: stored 999999999 != derived 847815772" in str(exc_info.value)

    def test_tampered_master_seed_fails(self, valid_seeds_data: dict, tmp_path: Path):
        """Assert that changing master_seed invalidates all derived task seeds."""
        mutated = dict(valid_seeds_data)
        mutated["master_seed"] = 20260920  # changed master seed
        config_path = tmp_path / "corrupted_master_seed.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(mutated, f)

        with pytest.raises(ValueError) as exc_info:
            verify_seeds_config(config_path)
        
        assert "Seed verification FAILED" in str(exc_info.value)
        # Multiple tasks should mismatch
        assert "Task T00:" in str(exc_info.value)
        assert "Task T01:" in str(exc_info.value)

    def test_empty_task_seeds_fails(self, valid_seeds_data: dict, tmp_path: Path):
        """Assert that empty task_seeds dictionary raises ValueError."""
        mutated = dict(valid_seeds_data)
        mutated["task_seeds"] = {}
        config_path = tmp_path / "empty_task_seeds.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(mutated, f)

        with pytest.raises(ValueError) as exc_info:
            verify_seeds_config(config_path)
        assert "No task_seeds defined" in str(exc_info.value)

    def test_missing_seed_file_raises_file_not_found(self, tmp_path: Path):
        """Assert that nonexistent seeds config raises FileNotFoundError."""
        ghost_path = tmp_path / "ghost_seeds.yaml"
        with pytest.raises(FileNotFoundError):
            verify_seeds_config(ghost_path)


# ==============================================================================
# 5. Live E2E Subprocess Tests with Atomic Restores
# ==============================================================================

class TestLiveSubprocessRejectionAndRestoration:
    """
    Adversarially test the Makefile targets directly via subprocess.
    Each test:
    1. Backs up the target file
    2. Mutates the real file in place
    3. Executes the make target, asserting non-zero exit code
    4. Restores original file in try...finally block
    5. Re-executes the make target to confirm restoration passes cleanly
    """

    def test_live_manifest_corruption_fails_verify_canonical(self):
        """Corrupt artifacts/canonical/manifest.sha256 -> assert make verify-canonical fails."""
        backup = MANIFEST_PATH.read_text(encoding="utf-8")
        try:
            # Corrupt the manifest
            corrupted = backup.replace("6660f21606d371fe", "0000000000000000")
            MANIFEST_PATH.write_text(corrupted, encoding="utf-8")

            res = subprocess.run(["make", "verify-canonical"], cwd=AUDIT_ROOT, capture_output=True, text=True)
            assert res.returncode != 0, f"Expected failure, got returncode 0: {res.stdout}"
            assert "MISMATCH: artifacts/canonical/points.parquet" in (res.stderr + res.stdout)
        finally:
            MANIFEST_PATH.write_text(CLEAN_MANIFEST, encoding="utf-8")

        # Verify restoration passes cleanly
        clean_res = subprocess.run(["make", "verify-canonical"], cwd=AUDIT_ROOT, capture_output=True, text=True)
        assert clean_res.returncode == 0, f"Restoration failed: {clean_res.stderr}"

    def test_live_card_schema_corruption_fails_verify_cards(self):
        """Corrupt artifacts/cards/T00/card.json -> assert make verify-cards fails."""
        backup = T00_CARD_PATH.read_text(encoding="utf-8")
        try:
            # Delete verdict property
            data = json.loads(backup)
            del data["verdict"]
            T00_CARD_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

            res = subprocess.run(["make", "verify-cards"], cwd=AUDIT_ROOT, capture_output=True, text=True)
            assert res.returncode != 0, f"Expected failure, got returncode 0: {res.stdout}"
            assert "Card validation FAILED" in (res.stderr + res.stdout)
            assert "verdict" in (res.stderr + res.stdout)
        finally:
            T00_CARD_PATH.write_text(backup, encoding="utf-8")

        # Verify restoration passes cleanly
        clean_res = subprocess.run(["make", "verify-cards"], cwd=AUDIT_ROOT, capture_output=True, text=True)
        assert clean_res.returncode == 0, f"Restoration failed: {clean_res.stderr}"

    def test_live_card_provenance_hash_corruption_fails_verify_hashes(self):
        """Corrupt output hash in T00/card.json -> assert make verify-hashes fails."""
        backup = T00_CARD_PATH.read_text(encoding="utf-8")
        try:
            data = json.loads(backup)
            data["provenance_hashes"]["outputs"][0]["sha256"] = "0" * 64
            T00_CARD_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

            res = subprocess.run(["make", "verify-hashes"], cwd=AUDIT_ROOT, capture_output=True, text=True)
            assert res.returncode != 0, f"Expected failure, got returncode 0: {res.stdout}"
            assert "Card validation FAILED" in (res.stderr + res.stdout)
            assert "Output hash mismatch" in (res.stderr + res.stdout)
        finally:
            T00_CARD_PATH.write_text(backup, encoding="utf-8")

        # Verify restoration passes cleanly
        clean_res = subprocess.run(["make", "verify-hashes"], cwd=AUDIT_ROOT, capture_output=True, text=True)
        assert clean_res.returncode == 0, f"Restoration failed: {clean_res.stderr}"

    def test_live_seed_corruption_fails_verify_seeds(self):
        """Corrupt T00 seed in env/seeds.yaml -> assert make verify-seeds fails."""
        backup = SEEDS_CONFIG_PATH.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(backup)
            data["task_seeds"]["T00"] = 111111111
            with open(SEEDS_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(data, f)

            res = subprocess.run(["make", "verify-seeds"], cwd=AUDIT_ROOT, capture_output=True, text=True)
            assert res.returncode != 0, f"Expected failure, got returncode 0: {res.stdout}"
            assert "Seed verification FAILED" in (res.stderr + res.stdout)
            assert "stored 111111111 != derived 847815772" in (res.stderr + res.stdout)
        finally:
            SEEDS_CONFIG_PATH.write_text(backup, encoding="utf-8")

        # Verify restoration passes cleanly
        clean_res = subprocess.run(["make", "verify-seeds"], cwd=AUDIT_ROOT, capture_output=True, text=True)
        assert clean_res.returncode == 0, f"Restoration failed: {clean_res.stderr}"

    def test_live_corruption_aborts_make_verify_cascade(self):
        """Assert that make verify aborts at Gate 2 when manifest is corrupted."""
        backup = MANIFEST_PATH.read_text(encoding="utf-8")
        try:
            corrupted = backup.replace("6660f21606d371fe", "deadbeefdeadbeef")
            MANIFEST_PATH.write_text(corrupted, encoding="utf-8")

            res = subprocess.run(["make", "verify"], cwd=AUDIT_ROOT, capture_output=True, text=True)
            assert res.returncode != 0, f"make verify should have failed, got 0: {res.stdout}"
            assert "ALL GATE 1 INGESTION" not in res.stdout
            assert "MISMATCH: artifacts/canonical/points.parquet" in (res.stderr + res.stdout)
        finally:
            MANIFEST_PATH.write_text(CLEAN_MANIFEST, encoding="utf-8")
