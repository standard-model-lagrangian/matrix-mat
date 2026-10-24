"""
Tier 3 Tests: Cryptographic Hashing and Manifest Verification.
Validates:
- artifacts/canonical/manifest.sha256 format and full cryptographic integrity
- All required canonical tables are covered in manifest
- Cross-validation of Section 9 card provenance hashes against files on disk
"""

import json
from pathlib import Path
import pytest

from utils.hashing import compute_sha256, verify_manifest

AUDIT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = AUDIT_ROOT / "artifacts" / "canonical"
MANIFEST_PATH = CANONICAL_DIR / "manifest.sha256"
CARDS_DIR = AUDIT_ROOT / "artifacts" / "cards"


def test_manifest_verification():
    """Verify artifacts/canonical/manifest.sha256 passes full integrity verification."""
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    result = verify_manifest(MANIFEST_PATH, base_dir=AUDIT_ROOT)
    assert result is True


def test_manifest_file_hashes():
    """Verify each file listed in manifest matches its independent sha256."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    verified_count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(maxsplit=1)
        target = (AUDIT_ROOT / rel_path).resolve()
        assert target.is_file(), f"File in manifest does not exist: {target}"
        actual_hash = compute_sha256(target)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"
        verified_count += 1

    assert verified_count >= 4, f"Expected at least 4 canonical files verified, got {verified_count}"


def test_manifest_covers_all_required_canonical_tables():
    """Assert manifest covers points, prior, optima, and paper_claims."""
    content = MANIFEST_PATH.read_text(encoding="utf-8")
    required = ["points.parquet", "prior.parquet", "optima.parquet", "paper_claims.yaml"]
    for fname in required:
        assert fname in content, f"Required canonical file '{fname}' missing from manifest.sha256"


def test_card_provenance_hashes_match_disk():
    """Verify all existing card.json files have provenance hashes matching disk files."""
    if not CARDS_DIR.is_dir():
        pytest.skip("artifacts/cards directory not found")

    card_files = list(CARDS_DIR.glob("**/card.json"))
    if not card_files:
        pytest.skip("No card.json files found")

    verified_provenance_items = 0
    for card_path in card_files:
        with open(card_path, "r", encoding="utf-8") as f:
            card_data = json.load(f)

        prov = card_data.get("provenance_hashes", {})

        # 1. Script
        script_info = prov.get("script", {})
        if script_info.get("path") and script_info.get("sha256"):
            p = AUDIT_ROOT / script_info["path"]
            if p.is_file():
                assert compute_sha256(p) == script_info["sha256"], (
                    f"Script hash mismatch in {card_data['task_id']} for {script_info['path']}"
                )
                verified_provenance_items += 1

        # 2. Inputs
        for inp in prov.get("inputs", []):
            p_str = inp.get("path")
            exp_h = inp.get("sha256")
            if p_str and exp_h:
                p = AUDIT_ROOT / p_str
                if not p.is_file():
                    p = AUDIT_ROOT.parent / p_str
                if p.is_file():
                    assert compute_sha256(p) == exp_h, (
                        f"Input hash mismatch in {card_data['task_id']} for {p_str}"
                    )
                    verified_provenance_items += 1

        # 3. Outputs
        for out in prov.get("outputs", []):
            p_str = out.get("path")
            exp_h = out.get("sha256")
            if p_str and exp_h:
                p = AUDIT_ROOT / p_str
                if p.is_file():
                    assert compute_sha256(p) == exp_h, (
                        f"Output hash mismatch in {card_data['task_id']} for {p_str}"
                    )
                    verified_provenance_items += 1

    assert verified_provenance_items >= 1
