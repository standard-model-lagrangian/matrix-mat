"""expanduser and unresolved-placeholder handling in core.paths (0.2.2)."""

from pathlib import Path

import pytest

from matrix_mat.core.paths import (
    expand_root_placeholders,
    resolve_cache_root,
    resolve_data_root,
    resolve_models_root,
    resolve_output_root,
)


def test_env_tilde_is_expanded(monkeypatch):
    monkeypatch.setenv("MATRIX_MAT_OUTPUT_ROOT", "~/out")
    monkeypatch.delenv("MATRIX_MAT_CACHE_DIR", raising=False)
    assert resolve_output_root() == Path.home() / "out"
    assert resolve_cache_root() == Path.home() / "out" / "cache" / "masks"


def test_cli_tilde_is_expanded():
    for fn in (resolve_data_root, resolve_output_root, resolve_models_root, resolve_cache_root):
        assert fn("~/x") == Path.home() / "x"


def test_env_value_with_env_var_is_expanded(monkeypatch):
    monkeypatch.setenv("MY_BASE", "/base")
    monkeypatch.setenv("MATRIX_MAT_DATA_ROOT", "${MY_BASE}/data")
    assert resolve_data_root() == Path("/base/data")


def test_unresolved_placeholder_in_env_raises(monkeypatch):
    monkeypatch.delenv("NOT_SET_ANYWHERE_123", raising=False)
    monkeypatch.setenv("MATRIX_MAT_DATA_ROOT", "${NOT_SET_ANYWHERE_123}/data")
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE_123"):
        resolve_data_root()


def test_expand_root_placeholders_known_and_unknown(monkeypatch):
    monkeypatch.setenv("MATRIX_MAT_DATA_ROOT", "/d")
    monkeypatch.delenv("NOT_SET_ANYWHERE_123", raising=False)
    assert expand_root_placeholders('data_dir: "${DATA_ROOT}/if"') == 'data_dir: "/d/if"'
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE_123"):
        expand_root_placeholders('x: "${NOT_SET_ANYWHERE_123}/a"')
    # Placeholders inside YAML comment lines are not an error.
    assert "${NOT_SET" in expand_root_placeholders("# see ${NOT_SET_ANYWHERE_123}\nx: 1")
