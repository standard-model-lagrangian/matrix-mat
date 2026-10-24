import os
import re
from pathlib import Path
import pytest

from spheroid_pipeline_v2.paths import (
    resolve_data_root,
    resolve_output_root,
    resolve_models_root,
    resolve_cache_root,
)


def test_paths_resolution_cli_env_default(monkeypatch):
    """Path resolution follows strict precedence: CLI > ENV > Default."""
    # 1. Default when neither CLI nor ENV is set
    monkeypatch.delenv("MATRIX_MAT_DATA_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_MODELS_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_CACHE_DIR", raising=False)

    assert resolve_data_root() == Path("data")
    assert resolve_output_root() == Path("output")
    assert resolve_models_root() == Path("models")
    assert resolve_cache_root() == Path("output/cache/masks")

    # 2. Env variable takes precedence over default
    monkeypatch.setenv("MATRIX_MAT_DATA_ROOT", "/custom/env/data")
    monkeypatch.setenv("MATRIX_MAT_OUTPUT_ROOT", "/custom/env/out")
    monkeypatch.setenv("MATRIX_MAT_MODELS_ROOT", "/custom/env/models")
    monkeypatch.setenv("MATRIX_MAT_CACHE_DIR", "/custom/env/cache")

    assert resolve_data_root() == Path("/custom/env/data")
    assert resolve_output_root() == Path("/custom/env/out")
    assert resolve_models_root() == Path("/custom/env/models")
    assert resolve_cache_root() == Path("/custom/env/cache")

    # 3. CLI argument takes precedence over ENV
    assert resolve_data_root("/cli/data") == Path("/cli/data")
    assert resolve_output_root("/cli/out") == Path("/cli/out")
    assert resolve_models_root("/cli/models") == Path("/cli/models")
    assert resolve_cache_root("/cli/cache") == Path("/cli/cache")


def test_decisions_log_is_per_run(tmp_path):
    """Decisions log must be written to <run_dir>/decisions.jsonl, not to repo root DECISIONS.md."""
    from spheroid_pipeline_v2.decisions import DecisionsLogger

    out_dir = tmp_path / "my_run"
    out_dir.mkdir()
    logger = DecisionsLogger(output_dir=out_dir)
    logger.log_decision(
        image_id="img_001",
        step="segmentation",
        from_backend="cpsam",
        to_backend="classical",
        reason="low_contrast",
    )

    jsonl_path = out_dir / "decisions.jsonl"
    assert jsonl_path.exists(), f"Expected decisions.jsonl in {out_dir}, but not found"
    content = jsonl_path.read_text(encoding="utf-8")
    assert "img_001" in content
    assert "low_contrast" in content


def test_no_hardcoded_data_paths_in_package():
    """Package source code must not hardcode personal/external user data paths."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    forbidden_patterns = [
        re.compile(r"Experimental data\s*/"),
        re.compile(r"Chuling"),
        re.compile(r"/Users/"),
    ]

    scan_dirs = [
        repo_root / "spheroid_pipeline_v2",
        repo_root / "spheroid_if_sweep",
    ]
    if (repo_root / "src" / "matrix_mat").exists():
        scan_dirs.append(repo_root / "src" / "matrix_mat")

    violations = []
    for sdir in scan_dirs:
        if not sdir.exists():
            continue
        for py_file in sdir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "tests" in str(py_file):
                continue
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                for pat in forbidden_patterns:
                    if pat.search(clean_line):
                        violations.append(f"{py_file.relative_to(repo_root)}:{line_no}: {clean_line}")

    assert not violations, (
        f"Found hardcoded data paths in production code:\n" + "\n".join(violations)
    )


def test_library_never_writes_to_cwd(tmp_path, monkeypatch):
    """Running from a temp cwd asserts the cwd stays completely empty."""
    temp_cwd = tmp_path / "clean_cwd"
    temp_cwd.mkdir()
    monkeypatch.chdir(temp_cwd)

    monkeypatch.delenv("MATRIX_MAT_DATA_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_MODELS_ROOT", raising=False)
    monkeypatch.delenv("MATRIX_MAT_CACHE_DIR", raising=False)

    # Calling path resolvers and importing modules must never create files in cwd
    from matrix_mat.core.paths import resolve_data_root, resolve_output_root, resolve_models_root, resolve_cache_root
    assert resolve_data_root() == Path("data")
    assert resolve_output_root() == Path("output")
    assert resolve_models_root() == Path("models")
    assert resolve_cache_root() == Path("output/cache/masks")

    # Verify no files or directories were created in cwd
    cwd_items = list(temp_cwd.iterdir())
    assert cwd_items == [], f"Library polluted cwd with: {cwd_items}"

