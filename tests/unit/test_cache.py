import ast
from dataclasses import fields, replace
from pathlib import Path
import numpy as np
import pytest
from spheroid_pipeline_v2.config import SegmentationConfig
from spheroid_pipeline_v2.mask_cache import MaskCache, compute_cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]


# Non-algorithmic or path-only metadata fields that don't alter the mathematical segmentation output
NON_ALGORITHMIC_FIELDS = {
    "cache_dir",
    "decisions_log_path",
    "force_recompute",
    "enable_mps_fallback",
    "backend",  # Alias for primary_backend
    "classical_adaptive_windows",  # Alias for adaptive_window_sizes
}

SEG_FIELDS = [
    f.name for f in fields(SegmentationConfig)
    if f.name not in NON_ALGORITHMIC_FIELDS
]


@pytest.mark.parametrize("field_name", SEG_FIELDS)
def test_cache_miss_when_any_seg_param_changes(field_name):
    """
    Changing any algorithmic parameter in SegmentationConfig must produce a different
    cache key, guaranteeing no stale cache hits across parameter sweeps.
    """
    base_cfg = SegmentationConfig()
    img = np.zeros((100, 100), dtype=np.uint8)

    orig_val = getattr(base_cfg, field_name)
    # Enumerated fields must be mutated to another valid choice (the schema rejects unknown values)
    enum_choices = {
        "primary_backend": ("cpsam", "cpdino", "classical"),
        "fallback_backend": ("cpsam", "cpdino", "classical"),
        "last_resort_backend": ("cpsam", "cpdino", "classical"),
        "fallback_policy": ("strict", "per_image"),
    }
    # Mutate value based on type
    if field_name in enum_choices:
        new_val = next(v for v in enum_choices[field_name] if v != orig_val)
    elif isinstance(orig_val, bool):
        new_val = not orig_val
    elif isinstance(orig_val, int):
        new_val = orig_val + 5
    elif isinstance(orig_val, float):
        new_val = orig_val + 0.15
    elif isinstance(orig_val, str):
        new_val = f"{orig_val}_mutated"
    elif isinstance(orig_val, list):
        new_val = orig_val + [999]
    else:
        new_val = "mutated_val"

    mutated_cfg = replace(base_cfg, **{field_name: new_val})

    key_orig = compute_cache_key(img, base_cfg)
    key_mutated = compute_cache_key(img, mutated_cfg)

    assert key_orig != key_mutated, f"Cache key did not change when field '{field_name}' changed from {orig_val} to {new_val}"


def test_cache_miss_when_model_hash_changes():
    """Cache key must incorporate model weights hash."""
    cfg = SegmentationConfig()
    img = np.zeros((100, 100), dtype=np.uint8)

    key1 = compute_cache_key(img, cfg, model_hash="weights_sha_v1")
    key2 = compute_cache_key(img, cfg, model_hash="weights_sha_v2")

    assert key1 != key2


def test_same_filename_different_pixels_do_not_collide(tmp_path):
    """
    Two images with the identical filename/stem but different pixel contents
    must NOT collide in the cache.
    """
    cache = MaskCache(cache_dir=tmp_path)
    cfg = SegmentationConfig()

    img1 = np.zeros((50, 50), dtype=np.uint8)
    img2 = np.ones((50, 50), dtype=np.uint8) * 128

    key1 = cache.get_key(img1, cfg, image_id="sample_field")
    key2 = cache.get_key(img2, cfg, image_id="sample_field")

    assert key1 != key2

    mask1 = np.ones((50, 50), dtype=np.int32)
    mask2 = np.ones((50, 50), dtype=np.int32) * 2

    cache.save(key1, mask1, metadata={"backend": "cpsam"})
    cache.save(key2, mask2, metadata={"backend": "classical"})

    loaded1 = cache.load(key1)
    loaded2 = cache.load(key2)

    assert loaded1[0, 0] == 1
    assert loaded2[0, 0] == 2


def test_cache_records_original_backend(tmp_path):
    """Cache sidecar metadata must record the original backend, model_id, and diameter used."""
    cache = MaskCache(cache_dir=tmp_path)
    cfg = SegmentationConfig()
    img = np.zeros((40, 40), dtype=np.uint8)
    key = cache.get_key(img, cfg)

    mask = np.ones((40, 40), dtype=np.int32)
    cache.save(key, mask, metadata={
        "backend": "cpdino",
        "model_id": "cellpose_cpdino_2024",
        "diameter_used": 175.0,
    })

    meta = cache.load_meta(key)
    assert meta["backend"] == "cpdino"
    assert meta["model_id"] == "cellpose_cpdino_2024"
    assert meta["diameter_used"] == 175.0


def test_if_baseline_shortcut_removed():
    """Verify that src/matrix_mat/if_density/sweep.py does not read 001_A_baseline/masks."""
    sweep_file = (REPO_ROOT / "src/matrix_mat/if_density/sweep.py")
    assert sweep_file.exists()

    content = sweep_file.read_text(encoding="utf-8")
    assert "001_A_baseline" not in content, (
        "src/matrix_mat/if_density/sweep.py still contains shortcut reading 001_A_baseline/masks!"
    )



def test_cache_key_tracks_installed_package_version(monkeypatch):
    """Cached masks from another release must not be reused (cached metadata semantics can change)."""
    import matrix_mat

    img = np.zeros((8, 8), dtype=np.uint8)
    cfg = SegmentationConfig()
    k_now = compute_cache_key(img, cfg)
    monkeypatch.setattr(matrix_mat, "__version__", "999.0.0")
    assert compute_cache_key(img, cfg) != k_now


def test_package_version_matches_pyproject():
    tomllib = pytest.importorskip("tomllib")  # Python >= 3.11
    import matrix_mat

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert matrix_mat.__version__ == pyproject["project"]["version"]
