"""Regression tests for core.cache bug fixes (0.2.2)."""

from dataclasses import dataclass

import numpy as np
import pytest

from matrix_mat.core.cache import MaskCache, compute_cache_key


@dataclass
class _Seg:
    diameter: float = 30.0
    cache_dir: str = "/tmp/x"


def test_transposed_shapes_do_not_collide():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = np.arange(12, dtype=np.uint8).reshape(4, 3)
    assert a.tobytes() == b.tobytes()
    assert compute_cache_key(a, {}) != compute_cache_key(b, {})


def test_dtype_is_part_of_key():
    a = np.zeros(8, dtype=np.uint8)
    b = np.zeros(4, dtype=np.uint16)
    assert compute_cache_key(a, {}) != compute_cache_key(b, {})


def test_combined_config_dict_is_deterministic_and_sensitive():
    img = np.zeros((4, 4), dtype=np.uint8)
    c1 = {"seg": _Seg(), "preprocess": {"clahe": True}, "pixel_size_um": 1.5188}
    c2 = {"pixel_size_um": 1.5188, "preprocess": {"clahe": True}, "seg": _Seg()}
    assert compute_cache_key(img, c1) == compute_cache_key(img, c2)
    c3 = dict(c1, pixel_size_um=1.6)
    assert compute_cache_key(img, c1) != compute_cache_key(img, c3)
    c4 = dict(c1, seg=_Seg(diameter=31.0))
    assert compute_cache_key(img, c1) != compute_cache_key(img, c4)
    # Non-algorithmic fields of nested dataclasses are ignored.
    c5 = dict(c1, seg=_Seg(cache_dir="/elsewhere"))
    assert compute_cache_key(img, c1) == compute_cache_key(img, c5)


def test_tuple_and_list_configs_are_hashed():
    img = np.zeros((4, 4), dtype=np.uint8)
    assert compute_cache_key(img, (_Seg(), 1.0)) != compute_cache_key(img, (_Seg(), 2.0))
    assert compute_cache_key(img, [_Seg(), 1.0]) == compute_cache_key(img, [_Seg(), 1.0])


def test_pydantic_model_in_combined_config():
    pydantic = pytest.importorskip("pydantic")

    class P(pydantic.BaseModel):
        sigma: float = 1.0

    img = np.zeros((4, 4), dtype=np.uint8)
    assert compute_cache_key(img, {"p": P()}) != compute_cache_key(img, {"p": P(sigma=2.0)})


def test_resave_without_metadata_removes_stale_sidecar(tmp_path):
    cache = MaskCache(tmp_path)
    m = np.ones((5, 5), dtype=np.int32)
    cache.save("k", m, metadata={"qc_flag": "SEG_FAIL"})
    assert cache.load_meta("k")["qc_flag"] == "SEG_FAIL"
    cache.save("k", m)
    assert cache.load_meta("k") == {}


def test_save_leaves_no_temp_files(tmp_path):
    cache = MaskCache(tmp_path)
    cache.save("k", np.ones((5, 5), dtype=np.int32), metadata={"a": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["k.meta.json", "k.png"]


def test_large_labels_are_preserved(tmp_path):
    cache = MaskCache(tmp_path)
    m = np.zeros((4, 4), dtype=np.int32)
    m[0, 0] = 70000
    m[1, 1] = 3
    cache.save("big", m)
    assert cache.has("big")
    out = cache.load("big")
    np.testing.assert_array_equal(out, m)
    # Re-saving with small labels switches back to PNG and removes the npz.
    cache.save("big", np.ones((4, 4), dtype=np.int32))
    np.testing.assert_array_equal(cache.load("big"), np.ones((4, 4), dtype=np.int32))
    assert not (tmp_path / "big.npz").exists()
    assert cache.clear() == 1


def test_negative_labels_rejected(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        MaskCache(tmp_path).save("neg", -np.ones((3, 3), dtype=np.int32))


def test_3d_mask_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        MaskCache(tmp_path).save("vol", np.ones((2, 3, 3), dtype=np.int32))


def test_corrupt_entry_is_a_consistent_miss(tmp_path):
    cache = MaskCache(tmp_path)
    cache.save("c", np.ones((4, 4), dtype=np.int32), metadata={"qc_flag": "PASS"})
    (tmp_path / "c.png").write_bytes(b"not a png at all")
    assert cache.has("c") is False
    assert cache.load("c") is None
    # Truncated PNG (valid signature, broken body): load() drops the entry and has() agrees.
    cache.save("d", np.ones((4, 4), dtype=np.int32))
    data = (tmp_path / "d.png").read_bytes()
    (tmp_path / "d.png").write_bytes(data[:20])
    assert cache.load("d") is None
    assert cache.has("d") is False
    assert not (tmp_path / "d.png").exists()
