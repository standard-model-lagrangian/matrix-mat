"""
Central path resolution for matrix-mat.
Strict precedence: CLI arguments > Environment variables > Defaults.
No library code writes to cwd directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def resolve_data_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for datasets."""
    if cli_val is not None:
        return Path(cli_val)
    if "MATRIX_MAT_DATA_ROOT" in os.environ:
        return Path(os.environ["MATRIX_MAT_DATA_ROOT"])
    return Path("data")


def resolve_output_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for pipeline output artifacts and figures."""
    if cli_val is not None:
        return Path(cli_val)
    if "MATRIX_MAT_OUTPUT_ROOT" in os.environ:
        return Path(os.environ["MATRIX_MAT_OUTPUT_ROOT"])
    return Path("output")


def resolve_models_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for model weights."""
    if cli_val is not None:
        return Path(cli_val)
    if "MATRIX_MAT_MODELS_ROOT" in os.environ:
        return Path(os.environ["MATRIX_MAT_MODELS_ROOT"])
    return Path("models")


def resolve_cache_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for content-addressed mask cache."""
    if cli_val is not None:
        return Path(cli_val)
    if "MATRIX_MAT_CACHE_DIR" in os.environ:
        return Path(os.environ["MATRIX_MAT_CACHE_DIR"])
    output_root = resolve_output_root()
    return output_root / "cache" / "masks"
