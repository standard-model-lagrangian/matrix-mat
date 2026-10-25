"""Cryptographic hashing utilities for artifact provenance and manifest verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence


def compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hexadecimal digest for a file.
    
    Reads in 64KB chunks for memory efficiency on large parquet or model files.
    """
    path = Path(filepath).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_manifest(
    file_paths: Sequence[str | Path],
    output_manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, str]:
    """Compute SHA-256 digests for a sequence of files and write a standard manifest file.
    
    Format matches standard sha256sum:
    <sha256_hex>  <path_relative_to_base_dir>
    """
    out_path = Path(output_manifest_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = Path(base_dir).resolve() if base_dir else out_path.parent

    manifest_entries: dict[str, str] = {}
    lines: list[str] = []

    for file_path in file_paths:
        p = Path(file_path).resolve()
        digest = compute_sha256(p)
        try:
            rel_path = p.relative_to(base)
        except ValueError:
            rel_path = p
        
        manifest_entries[str(rel_path)] = digest
        lines.append(f"{digest}  {rel_path}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return manifest_entries


def verify_manifest(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> bool:
    """Verify all entries in a sha256 manifest against actual files on disk.
    
    Raises ValueError or FileNotFoundError if verification fails.
    Returns True on success.
    """
    man_path = Path(manifest_path).resolve()
    if not man_path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {man_path}")

    base = Path(base_dir).resolve() if base_dir else man_path.parent

    with open(man_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Manifest file is empty: {man_path}")

    mismatches: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line: '{line}' in {man_path}")

        expected_hash, file_rel = parts[0].strip(), parts[1].strip()
        # Handle leading * if binary mode was used
        if file_rel.startswith("*"):
            file_rel = file_rel[1:]

        target_file = (base / file_rel).resolve()
        if not target_file.is_file():
            mismatches.append(f"MISSING: {file_rel} (expected at {target_file})")
            continue

        actual_hash = compute_sha256(target_file)
        if actual_hash.lower() != expected_hash.lower():
            mismatches.append(
                f"MISMATCH: {file_rel} (expected {expected_hash[:16]}..., got {actual_hash[:16]}...)"
            )

    if mismatches:
        raise ValueError(
            f"Manifest verification FAILED for {man_path}:\n" + "\n".join(mismatches)
        )

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHA-256 manifest utility")
    parser.add_argument("--verify", type=str, help="Path to manifest file to verify")
    parser.add_argument("--base-dir", type=str, default=None, help="Base directory for relative paths")
    args = parser.parse_args()

    if args.verify:
        verify_manifest(args.verify, base_dir=args.base_dir)
        print(f"✅ Manifest verified successfully: {args.verify}")
