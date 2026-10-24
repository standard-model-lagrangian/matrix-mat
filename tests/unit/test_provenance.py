import hashlib
import json
import re
from pathlib import Path

import pytest

from matrix_mat.core import provenance
from matrix_mat.core.provenance import get_git_info, write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(not (REPO_ROOT / ".git").exists(), reason="not a git checkout")
def test_git_info_is_read_from_package_repo_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    info = get_git_info()
    assert re.fullmatch(r"[0-9a-f]{40}", info["commit"]), info


def test_run_metadata_records_dependency_lock_hash_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_run_metadata(output_dir=tmp_path / "run", backend="classical")
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    expected = hashlib.sha256((REPO_ROOT / "requirements-lock.txt").read_bytes()).hexdigest()
    assert meta["lockfile_sha256"] == expected
    expected_models = hashlib.sha256((REPO_ROOT / "models.lock.json").read_bytes()).hexdigest()
    assert meta["models_lock_sha256"] == expected_models


def test_classical_backend_records_no_weights_hash(tmp_path):
    write_run_metadata(output_dir=tmp_path, backend="classical")
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    assert meta["weights_sha256"] is None


def test_cpsam_backend_records_pinned_weights_hash(tmp_path):
    write_run_metadata(output_dir=tmp_path, backend="cpsam")
    meta = json.loads((tmp_path / "run_metadata.json").read_text())
    pinned = json.loads((REPO_ROOT / "models.lock.json").read_text())["models"]["cpsam_v2"]["sha256"]
    assert meta["weights_sha256"] == pinned
