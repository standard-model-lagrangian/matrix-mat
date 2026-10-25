#!/usr/bin/env python3
"""
Model weights fetching and integrity verification tool for matrix-mat.
Downloads pinned model weights from models.lock.json and validates SHA-256 hashes.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from typing import Dict, Any, Optional



REPO_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_lock_file(lock_path: Path) -> Dict[str, Any]:
    """Load models.lock.json."""
    if not lock_path.exists():
        raise FileNotFoundError(f"Model lockfile not found: {lock_path}")
    with open(lock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_model(model_name: str, model_info: Dict[str, Any], models_dir: Path) -> bool:
    """Verify that a model exists and matches its pinned SHA-256."""
    target_path = models_dir / model_name
    if not target_path.exists():
        print(f"[-] {model_name}: MISSING at {target_path}")
        return False

    expected_sha = model_info.get("sha256")
    actual_sha = compute_sha256(target_path)
    if expected_sha and actual_sha.lower() != expected_sha.lower():
        print(f"[-] {model_name}: SHA-256 MISMATCH! Expected {expected_sha}, got {actual_sha}")
        return False

    print(f"[+] {model_name}: VALID (SHA-256: {actual_sha[:16]}...)")
    return True


def fetch_model(model_name: str, model_info: Dict[str, Any], models_dir: Path) -> bool:
    """Fetch model from URL if missing and verify SHA-256."""
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / model_name

    expected_sha = model_info.get("sha256", "")
    if target_path.exists():
        actual_sha = compute_sha256(target_path)
        if expected_sha and actual_sha.lower() == expected_sha.lower():
            print(f"[+] {model_name}: Already exists and verified.")
            return True
        else:
            print(f"[!] {model_name}: Exists but hash mismatch, re-downloading...")
            target_path.unlink()

    url = model_info.get("url")
    if not url:
        print(f"[-] {model_name}: No download URL in lockfile.")
        return False

    print(f"[*] Downloading {model_name} from {url} to {target_path}...")
    try:
        # Stream download with progress
        req = urllib.request.Request(url, headers={"User-Agent": "matrix-mat"})
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as out_f:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    print(f"\rDownloading: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="", flush=True)
        print()
    except Exception as e:
        print(f"[-] Download failed: {e}")
        if target_path.exists():
            target_path.unlink()
        return False

    actual_sha = compute_sha256(target_path)
    if expected_sha and actual_sha.lower() != expected_sha.lower():
        print(f"[-] SHA-256 verification failed! Expected {expected_sha}, got {actual_sha}")
        target_path.unlink()
        return False

    print(f"[+] {model_name} successfully downloaded and verified.")
    return True


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify model weights for matrix-mat.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory to store model weights (default: $MATRIX_MAT_MODELS_ROOT, else <repo>/models).",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=REPO_ROOT / "models.lock.json",
        help="Path to models.lock.json (default: <repo>/models.lock.json).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing models without downloading.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to fetch or verify (default: all).",
    )
    args = parser.parse_args(argv)
    if args.models_dir is None:
        env_root = os.environ.get("MATRIX_MAT_MODELS_ROOT")
        args.models_dir = Path(env_root) if env_root else REPO_ROOT / "models"

    try:
        lock_data = load_lock_file(args.lock_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    models_dict = lock_data.get("models", {})
    if args.model:
        if args.model not in models_dict:
            print(f"Error: Model '{args.model}' not found in {args.lock_file}", file=sys.stderr)
            return 1
        targets = {args.model: models_dict[args.model]}
    else:
        targets = models_dict

    all_ok = True
    for name, info in targets.items():
        if args.verify_only:
            ok = verify_model(name, info, args.models_dir)
        else:
            ok = fetch_model(name, info, args.models_dir)
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
