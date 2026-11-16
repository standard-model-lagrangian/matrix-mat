"""
Central path resolution for matrix-mat.
Strict precedence: CLI arguments > Environment variables > Defaults.
No library code writes to cwd directly.

``~`` is expanded and ``$VAR`` / ``${VAR}`` environment references are substituted in every
CLI / environment value; an unresolved ``${VAR}`` placeholder raises ``ValueError``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _unresolved(text: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(text)))


def _clean(val: Union[str, Path], origin: str) -> Path:
    text = os.path.expandvars(str(val))
    missing = _unresolved(text)
    if missing:
        raise ValueError(
            f"Unresolved placeholder(s) {', '.join('${' + m + '}' for m in missing)} in {origin} value {str(val)!r}; "
            "set the environment variable(s) or use an explicit path."
        )
    return Path(os.path.expanduser(text))


def _resolve(cli_val: Optional[Union[str, Path]], env: str, default: Path) -> Path:
    if cli_val is not None:
        return _clean(cli_val, "CLI")
    if env in os.environ:
        return _clean(os.environ[env], env)
    return default


def resolve_data_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for datasets."""
    return _resolve(cli_val, "MATRIX_MAT_DATA_ROOT", Path("data"))


def resolve_output_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for pipeline output artifacts and figures."""
    return _resolve(cli_val, "MATRIX_MAT_OUTPUT_ROOT", Path("output"))


def resolve_models_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for model weights."""
    return _resolve(cli_val, "MATRIX_MAT_MODELS_ROOT", Path("models"))


def resolve_cache_root(cli_val: Optional[Union[str, Path]] = None) -> Path:
    """Resolve base directory for content-addressed mask cache."""
    if cli_val is not None or "MATRIX_MAT_CACHE_DIR" in os.environ:
        return _resolve(cli_val, "MATRIX_MAT_CACHE_DIR", Path("unused"))
    return resolve_output_root() / "cache" / "masks"


def expand_root_placeholders(text: str) -> str:
    """Substitute ${DATA_ROOT}, ${OUTPUT_ROOT}, ${MODELS_ROOT}, ${CACHE_ROOT} and environment variables.

    Raises ``ValueError`` naming any ``${VAR}`` left unresolved outside ``#`` comment lines.
    """
    roots = {
        "DATA_ROOT": resolve_data_root(),
        "OUTPUT_ROOT": resolve_output_root(),
        "MODELS_ROOT": resolve_models_root(),
        "CACHE_ROOT": resolve_cache_root(),
    }
    for name, value in roots.items():
        text = text.replace("${" + name + "}", str(value))
    text = os.path.expandvars(text)
    missing = sorted({
        m for line in text.splitlines() if not line.lstrip().startswith("#")
        for m in _unresolved(line)
    })
    if missing:
        raise ValueError(
            f"Unresolved placeholder(s) {', '.join('${' + m + '}' for m in missing)} in config text; "
            "known placeholders are ${DATA_ROOT}, ${OUTPUT_ROOT}, ${MODELS_ROOT}, ${CACHE_ROOT} "
            "or any set environment variable."
        )
    return text
