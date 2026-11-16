"""Fail-closed weight verification, packaged models.lock.json and fetch_models safety (0.2.2)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from matrix_mat.core import provenance, repro
from matrix_mat.core.repro import default_models_lock_path, get_model_weights_info, verify_model_weights

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fetch_models():
    spec = importlib.util.spec_from_file_location("fetch_models_under_test", REPO_ROOT / "tools" / "fetch_models.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_models_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_packaged_lock_matches_repo_root_lock():
    packaged = REPO_ROOT / "src" / "matrix_mat" / "models.lock.json"
    assert packaged.read_bytes() == (REPO_ROOT / "models.lock.json").read_bytes()


def test_default_lock_found_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = default_models_lock_path()
    assert p.exists() and p.name == "models.lock.json"
    assert get_model_weights_info("cpsam_v2")["sha256"]
    assert provenance.get_models_lock_sha256() == hashlib.sha256(p.read_bytes()).hexdigest()
    assert provenance.get_default_weights_sha256() == get_model_weights_info("cpsam_v2")["sha256"]


def test_model_not_in_lock_fails_closed(tmp_path, monkeypatch):
    f = tmp_path / "w"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="not pinned"):
        verify_model_weights(f, model_name="no_such_model_xyz")


def test_lock_entry_without_sha_fails_closed(tmp_path, monkeypatch):
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps({"models": {"m": {"url": "file://x"}}}))
    monkeypatch.setattr(repro, "default_models_lock_path", lambda: lock)
    f = tmp_path / "m"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="sha256"):
        verify_model_weights(f, model_name="m")


# --- tools/fetch_models.py ------------------------------------------------------------

def _lock(tmp_path, entry):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"models": {"m": entry}}))
    return lock


def test_verify_only_without_sha_is_invalid(tmp_path, capsys):
    fm = _load_fetch_models()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m").write_bytes(b"abc")
    lock = _lock(tmp_path, {"url": "file://x"})
    rc = fm.main(["--verify-only", "--lock-file", str(lock), "--models-dir", str(tmp_path / "models")])
    assert rc != 0
    assert "INVALID" in capsys.readouterr().out


def test_verify_only_size_mismatch_is_invalid(tmp_path, capsys):
    fm = _load_fetch_models()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m").write_bytes(b"abc")
    lock = _lock(tmp_path, {"sha256": hashlib.sha256(b"abc").hexdigest(), "size_bytes": 99})
    rc = fm.main(["--verify-only", "--lock-file", str(lock), "--models-dir", str(tmp_path / "models")])
    assert rc != 0


def test_verify_only_valid(tmp_path, capsys):
    fm = _load_fetch_models()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m").write_bytes(b"abc")
    lock = _lock(tmp_path, {"sha256": hashlib.sha256(b"abc").hexdigest(), "size_bytes": 3})
    assert fm.main(["--verify-only", "--lock-file", str(lock), "--models-dir", str(tmp_path / "models")]) == 0


class _FakeResp:
    def __init__(self, chunks, interrupt=False):
        self._chunks = list(chunks)
        self._interrupt = interrupt
        self.headers = {"content-length": str(sum(len(c) for c in chunks))}

    def read(self, n):
        if not self._chunks:
            if self._interrupt:
                raise KeyboardInterrupt
            return b""
        return self._chunks.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_writes_atomically_and_verifies(tmp_path, monkeypatch):
    fm = _load_fetch_models()
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        return _FakeResp([b"ab", b"c"])

    monkeypatch.setattr(fm.urllib.request, "urlopen", fake_urlopen)
    info = {"sha256": hashlib.sha256(b"abc").hexdigest(), "size_bytes": 3, "url": "https://example.invalid/m"}
    assert fm.fetch_model("m", info, tmp_path) is True
    assert (tmp_path / "m").read_bytes() == b"abc"
    assert seen["timeout"] and seen["timeout"] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m"]


def test_fetch_bad_hash_leaves_no_file(tmp_path, monkeypatch):
    fm = _load_fetch_models()
    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp([b"xyz"]))
    info = {"sha256": hashlib.sha256(b"abc").hexdigest(), "url": "https://example.invalid/m"}
    assert fm.fetch_model("m", info, tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_fetch_interrupt_removes_partial(tmp_path, monkeypatch):
    fm = _load_fetch_models()
    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp([b"ab"], interrupt=True))
    info = {"sha256": hashlib.sha256(b"abc").hexdigest(), "url": "https://example.invalid/m"}
    with pytest.raises(KeyboardInterrupt):
        fm.fetch_model("m", info, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_refuses_entry_without_sha(tmp_path, monkeypatch):
    fm = _load_fetch_models()
    monkeypatch.setattr(fm.urllib.request, "urlopen", lambda req, timeout=None: _FakeResp([b"abc"]))
    assert fm.fetch_model("m", {"url": "https://example.invalid/m"}, tmp_path) is False
    assert list(tmp_path.iterdir()) == []
