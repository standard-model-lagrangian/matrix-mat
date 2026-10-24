"""
Reproducibility, determinism, and pinned model weights verification for matrix-mat.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_model_weights_info(
    model_name: str = "cpsam_v2",
    lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Retrieve pinned metadata from models.lock.json."""
    if lock_path is None:
        # Search repo root or relative to module
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        lock_path = repo_root / "models.lock.json"
        if not lock_path.exists():
            lock_path = Path("models.lock.json")

    if not lock_path.exists():
        raise FileNotFoundError(f"models.lock.json not found at {lock_path}")

    with open(lock_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_map = data.get("models", {})
    if model_name not in models_map:
        raise KeyError(f"Model '{model_name}' not pinned in {lock_path}")

    return models_map[model_name]


def verify_model_weights(
    model_path: Path,
    expected_sha256: Optional[str] = None,
    model_name: str = "cpsam_v2",
) -> str:
    """
    Verify model weights existence and SHA-256 checksum.
    Raises FileNotFoundError with actionable download hint if missing.
    Raises ValueError on checksum mismatch.
    Returns the verified SHA-256 hash.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model weights for '{model_name}' not found at '{p}'. "
            f"Run 'python tools/fetch_models.py' to download and verify pinned weights."
        )

    actual_sha = compute_sha256(p)

    if expected_sha256 is None:
        try:
            info = get_model_weights_info(model_name)
            expected_sha256 = info.get("sha256")
        except Exception:
            expected_sha256 = None

    if expected_sha256:
        if actual_sha.lower() != expected_sha256.lower():
            raise ValueError(
                f"Model weights SHA-256 checksum mismatch for '{p}'. "
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha}\n"
                f"Run 'python tools/fetch_models.py' to re-fetch."
            )

    return actual_sha


def set_determinism(seed: int = 42, include_torch: Optional[bool] = None) -> np.random.Generator:
    """
    Enforce strict determinism across Python, NumPy, and PyTorch (CPU / CUDA / MPS).
    Returns an independent numpy Generator instance.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng = np.random.default_rng(seed)

    should_seed_torch = include_torch if include_torch is not None else ("torch" in sys.modules)
    if should_seed_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
        except ImportError:
            pass

    return rng
