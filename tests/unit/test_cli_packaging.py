"""Packaging / tooling consistency (0.2.2)."""

import re
from pathlib import Path

import pytest

import matrix_mat  # noqa: F401  (production import for test_meta)

REPO = Path(__file__).resolve().parents[2]


def _names(reqs):
    return {re.split(r"[<>=\[ ]", r.strip(), maxsplit=1)[0].lower() for r in reqs if r.strip() and not r.strip().startswith("#")}


def test_requirements_match_pyproject_core_deps():
    tomllib = pytest.importorskip("tomllib")
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    req = (REPO / "requirements.txt").read_text().splitlines()
    assert _names(req) == _names(deps)
    assert "torch" not in _names(req) and "cellpose" not in _names(req)


def test_precommit_ruff_pinned_to_lock_version():
    lock = (REPO / "requirements-lock.txt").read_text()
    ver = re.search(r"^ruff==([\d.]+)", lock, re.M).group(1)
    assert f"rev: v{ver}" in (REPO / ".pre-commit-config.yaml").read_text()


def test_ruff_config_and_package_data():
    tomllib = pytest.importorskip("tomllib")
    py = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert py["tool"]["ruff"]["line-length"] == 120
    assert set(py["tool"]["ruff"]["lint"]["select"]) == {"E", "F", "B"}
    assert "models.lock.json" in py["tool"]["setuptools"]["package-data"]["matrix_mat"]


def test_ci_has_scheduled_slow_job_and_macos():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "workflow_dispatch" in ci and "cron:" in ci
    assert "macos-latest" in ci
    assert '-m "slow' in ci
