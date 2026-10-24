import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from matrix_mat.core.cache import MaskCache
from matrix_mat.core.repro import compute_sha256, get_model_weights_info, set_determinism, verify_model_weights


def _lock(tmp_path, sha):
    lock = tmp_path / "models.lock.json"
    lock.write_text(json.dumps({"models": {"tiny": {"sha256": sha, "url": "file://x"}}}))
    return lock


def test_compute_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights" * 1000)
    assert compute_sha256(f) == hashlib.sha256(b"weights" * 1000).hexdigest()


def test_verify_weights_accepts_matching_hash(tmp_path):
    f = tmp_path / "tiny"
    f.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    assert verify_model_weights(f, expected_sha256=sha, model_name="tiny") == sha


def test_verify_weights_rejects_mismatch(tmp_path):
    f = tmp_path / "tiny"
    f.write_bytes(b"abc")
    with pytest.raises(ValueError, match="mismatch"):
        verify_model_weights(f, expected_sha256="0" * 64, model_name="tiny")


def test_verify_weights_missing_file_has_fetch_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        verify_model_weights(tmp_path / "absent", expected_sha256="0" * 64)


def test_get_model_weights_info_reads_lock(tmp_path):
    lock = _lock(tmp_path, "ab" * 32)
    assert get_model_weights_info("tiny", lock_path=lock)["sha256"] == "ab" * 32
    with pytest.raises(KeyError):
        get_model_weights_info("other", lock_path=lock)
    with pytest.raises(FileNotFoundError):
        get_model_weights_info("tiny", lock_path=tmp_path / "none.json")


def test_repo_lock_pins_cpsam_v2_on_hugging_face():
    info = get_model_weights_info("cpsam_v2")
    assert info["url"] == "https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam_v2"
    assert len(info["sha256"]) == 64


def test_set_determinism_returns_reproducible_generator():
    a = set_determinism(7, include_torch=False).random(5)
    b = set_determinism(7, include_torch=False).random(5)
    np.testing.assert_array_equal(a, b)


def test_mask_cache_roundtrip_with_metadata(tmp_path):
    cache = MaskCache(tmp_path / "c")
    mask = np.zeros((20, 30), dtype=np.int32)
    mask[2:5, 2:5] = 1
    mask[10:15, 10:20] = 300
    cache.save("k1", mask, metadata={"backend": "cpsam", "qc_flag": "SEG_FAIL"})
    assert cache.has("k1")
    np.testing.assert_array_equal(cache.load("k1"), mask)
    assert cache.load_meta("k1") == {"backend": "cpsam", "qc_flag": "SEG_FAIL"}
    assert cache.clear() == 1
    assert not cache.has("k1")
    assert cache.load("k1") is None
    assert cache.load_meta("k1") == {}


def test_disabled_cache_never_hits(tmp_path):
    cache = MaskCache(tmp_path / "c", enabled=False)
    cache.save("k", np.ones((4, 4), dtype=np.int32))
    assert not cache.has("k")
    assert cache.load("k") is None


def test_corrupt_cache_file_is_a_miss(tmp_path):
    cache = MaskCache(tmp_path)
    cache.get_cache_path("bad").write_bytes(b"not a png")
    assert cache.load("bad") is None
