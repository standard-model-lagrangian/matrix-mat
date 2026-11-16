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
import tempfile
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


DOWNLOAD_TIMEOUT_S = 60.0


def _check_pin(model_name: str, model_info: Dict[str, Any]) -> Optional[str]:
    """Return the pinned sha256, or None (after printing INVALID) when the entry has none."""
    expected_sha = (model_info or {}).get("sha256")
    if not expected_sha:
        print(f"[-] {model_name}: INVALID lock entry (no sha256 pinned); refusing to trust or download it.")
        return None
    return str(expected_sha)


def _size_ok(model_name: str, model_info: Dict[str, Any], path: Path) -> bool:
    size = model_info.get("size_bytes")
    if size is None:
        return True
    actual = path.stat().st_size
    if int(size) != actual:
        print(f"[-] {model_name}: INVALID size {actual} bytes, expected {int(size)} bytes")
        return False
    return True


def verify_model(model_name: str, model_info: Dict[str, Any], models_dir: Path) -> bool:
    """Verify that a model exists and matches its pinned SHA-256 (and size, if pinned). Fails closed."""
    target_path = models_dir / model_name
    expected_sha = _check_pin(model_name, model_info)
    if expected_sha is None:
        return False
    if not target_path.exists():
        print(f"[-] {model_name}: INVALID (missing at {target_path})")
        return False
    if not _size_ok(model_name, model_info, target_path):
        return False

    actual_sha = compute_sha256(target_path)
    if actual_sha.lower() != expected_sha.lower():
        print(f"[-] {model_name}: INVALID SHA-256 mismatch! Expected {expected_sha}, got {actual_sha}")
        return False

    print(f"[+] {model_name}: VALID (SHA-256: {actual_sha[:16]}...)")
    return True


def fetch_model(
    model_name: str,
    model_info: Dict[str, Any],
    models_dir: Path,
    timeout: float = DOWNLOAD_TIMEOUT_S,
) -> bool:
    """Fetch a pinned model if missing/invalid and verify it.

    The download goes to a temporary file in ``models_dir`` that is only moved into place
    (``os.replace``) after the size and SHA-256 checks pass; any failure or interruption
    (including Ctrl-C) removes the partial file.
    """
    expected_sha = _check_pin(model_name, model_info)
    if expected_sha is None:
        return False

    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / model_name

    if target_path.exists():
        if compute_sha256(target_path).lower() == expected_sha.lower():
            print(f"[+] {model_name}: Already exists and verified.")
            return True
        print(f"[!] {model_name}: Exists but hash mismatch, re-downloading...")

    url = model_info.get("url")
    if not url:
        print(f"[-] {model_name}: No download URL in lockfile.")
        return False

    fd, tmp_name = tempfile.mkstemp(dir=models_dir, prefix=f".{model_name}.", suffix=".part")
    tmp_path = Path(tmp_name)
    print(f"[*] Downloading {model_name} from {url} to {target_path}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "matrix-mat"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, os.fdopen(fd, "wb") as out_f:
            total_size = int(resp.headers.get("content-length", 0) or 0)
            downloaded = 0
            while chunk := resp.read(1024 * 1024):
                out_f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    print(f"\rDownloading: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB)", end="", flush=True)
        print()

        if not _size_ok(model_name, model_info, tmp_path):
            tmp_path.unlink(missing_ok=True)
            return False
        actual_sha = compute_sha256(tmp_path)
        if actual_sha.lower() != expected_sha.lower():
            print(f"[-] SHA-256 verification failed! Expected {expected_sha}, got {actual_sha}")
            tmp_path.unlink(missing_ok=True)
            return False
        os.replace(tmp_path, target_path)
    except Exception as e:
        print(f"[-] Download failed: {e}")
        tmp_path.unlink(missing_ok=True)
        return False
    except BaseException:
        # KeyboardInterrupt / SystemExit: never leave a partial file behind.
        tmp_path.unlink(missing_ok=True)
        raise

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
