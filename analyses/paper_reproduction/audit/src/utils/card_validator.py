"""Section 9 Draft-07 task card validation CLI using fastjsonschema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import fastjsonschema

from utils.hashing import compute_sha256

DEFAULT_SCHEMA_PATH = Path(__file__).parent / "card_schema.json"


def load_validator(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> Any:
    """Compile and return fastjsonschema validator from JSON schema file."""
    path = Path(schema_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Card schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        schema_dict = json.load(f)

    return fastjsonschema.compile(schema_dict)


def validate_card_file(
    card_path: str | Path,
    validator: Any,
    verify_hashes: bool = False,
    base_dir: str | Path | None = None,
) -> tuple[bool, list[str]]:
    """Validate a single card.json file against compiled validator.
    
    Optionally verifies provenance hashes against actual files on disk.
    Returns (is_valid, list_of_error_messages).
    """
    card_file = Path(card_path).resolve()
    if not card_file.is_file():
        return False, [f"Card file does not exist: {card_file}"]

    try:
        with open(card_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON syntax in {card_file}: {e}"]

    errors: list[str] = []
    try:
        validator(data)
    except fastjsonschema.JsonSchemaException as e:
        errors.append(f"Schema validation error in {card_file.name}: {e.message} (path: {e.path})")

    if verify_hashes and not errors:
        prov = data.get("provenance_hashes", {})
        base = Path(base_dir).resolve() if base_dir else Path.cwd().resolve()

        # Verify inputs
        for inp in prov.get("inputs", []):
            fpath = inp.get("path")
            expected_hash = inp.get("sha256")
            p = Path(fpath)
            if not p.is_absolute():
                p = (base / p).resolve()
            if not p.is_file():
                errors.append(f"Provenance input file missing: {fpath} (resolved {p})")
            else:
                actual_hash = compute_sha256(p)
                if actual_hash.lower() != expected_hash.lower():
                    errors.append(
                        f"Input hash mismatch for {fpath}: expected {expected_hash}, got {actual_hash}"
                    )

        # Verify outputs
        for out in prov.get("outputs", []):
            fpath = out.get("path")
            expected_hash = out.get("sha256")
            p = Path(fpath)
            if not p.is_absolute():
                p = (base / p).resolve()
            if not p.is_file():
                errors.append(f"Provenance output file missing: {fpath} (resolved {p})")
            else:
                actual_hash = compute_sha256(p)
                if actual_hash.lower() != expected_hash.lower():
                    errors.append(
                        f"Output hash mismatch for {fpath}: expected {expected_hash}, got {actual_hash}"
                    )

        # Verify script
        script = prov.get("script", {})
        if script:
            fpath = script.get("path")
            expected_hash = script.get("sha256")
            p = Path(fpath)
            if not p.is_absolute():
                p = (base / p).resolve()
            if not p.is_file():
                errors.append(f"Provenance script file missing: {fpath} (resolved {p})")
            else:
                actual_hash = compute_sha256(p)
                if actual_hash.lower() != expected_hash.lower():
                    errors.append(
                        f"Script hash mismatch for {fpath}: expected {expected_hash}, got {actual_hash}"
                    )

    return (len(errors) == 0), errors


def validate_cards_directory(
    cards_dir: str | Path,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    verify_hashes: bool = False,
    base_dir: str | Path | None = None,
) -> tuple[int, int, list[str]]:
    """Scan and validate all *.json cards in cards_dir recursively.
    
    Returns (num_passed, num_failed, list_of_all_errors).
    """
    directory = Path(cards_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Cards directory does not exist: {directory}")

    card_files = sorted(directory.glob("**/card.json"))
    if not card_files:
        # Also check for any *.json files
        card_files = sorted(directory.glob("**/*.json"))

    if not card_files:
        return 0, 0, [f"No card JSON files found in {directory}"]

    validator = load_validator(schema_path)

    passed = 0
    failed = 0
    all_errors: list[str] = []

    for card_file in card_files:
        ok, errs = validate_card_file(
            card_file,
            validator,
            verify_hashes=verify_hashes,
            base_dir=base_dir,
        )
        if ok:
            passed += 1
        else:
            failed += 1
            all_errors.extend(errs)

    return passed, failed, all_errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit task card validator CLI")
    parser.add_argument("--card", type=str, default=None, help="Path to single card.json to validate")
    parser.add_argument("--cards-dir", type=str, default="artifacts/cards", help="Directory containing task cards")
    parser.add_argument("--schema", type=str, default=str(DEFAULT_SCHEMA_PATH), help="Path to Draft-07 card schema")
    parser.add_argument("--verify-hashes", action="store_true", help="Verify input/output/script provenance SHA-256 hashes")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory for relative paths in hashes")
    args = parser.parse_args()

    try:
        if args.card:
            validator = load_validator(args.schema)
            ok, errs = validate_card_file(
                args.card,
                validator,
                verify_hashes=args.verify_hashes,
                base_dir=args.base_dir,
            )
            if ok:
                mode = " (with hash verification)" if args.verify_hashes else ""
                print(f"✅ Card valid: {args.card}{mode}")
                return 0
            else:
                print(f"❌ Card validation FAILED: {args.card}")
                for e in errs:
                    print(f"  - {e}")
                return 1
        else:
            passed, failed, errs = validate_cards_directory(
                args.cards_dir,
                schema_path=args.schema,
                verify_hashes=args.verify_hashes,
                base_dir=args.base_dir,
            )
            mode = " (with hash verification)" if args.verify_hashes else ""
            if failed == 0 and passed > 0:
                print(f"✅ All {passed} task card(s) validated successfully{mode} in {args.cards_dir}")
                return 0
            else:
                print(f"❌ Card validation FAILED: {failed} failed, {passed} passed in {args.cards_dir}")
                for e in errs:
                    print(f"  - {e}")
                return 1
    except Exception as e:
        print(f"❌ Error during validation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
