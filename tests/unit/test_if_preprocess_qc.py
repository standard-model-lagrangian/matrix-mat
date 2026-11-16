"""Bit depth handling, RGB/3D rejection, QC saturation, split filling, extent/denominator, provenance flags."""
import numpy as np
import pytest
from skimage import draw

from matrix_mat.if_density.features import (
    compute_spheroid_extent_details,
    extract_objects_and_image_features,
)
from matrix_mat.if_density.postprocess import (
    border_strip_mask,
    identify_border_objects,
    postprocess_field_masks,
    resolve_size_gate_bounds,
    split_merged_nuclei,
)
from matrix_mat.if_density.preprocess import ensure_2d_grayscale, preprocess_field, to_uint8
from matrix_mat.if_density.qc import compute_image_qc_metrics, deduplicate_masks


# --- item 2: >8-bit data are scaled, never wrapped -------------------------------------------------

def test_passthrough_scales_16bit_instead_of_wrapping():
    nuc = np.zeros((32, 32), np.uint16)
    nuc[8:16, 8:16] = 4095  # bright 12-bit nucleus
    nuc[0, 0] = 256        # would wrap to 0 with astype(uint8)
    res = preprocess_field("x", nuc, nuc.copy(), 0.5, {"enabled": False})
    u8 = res.nuclear.preprocessed_u8
    assert u8.dtype == np.uint8
    assert u8[10, 10] == 255
    assert u8[0, 0] > 0
    assert np.all(np.diff(np.sort(np.unique(u8)).astype(int)) > 0)


def test_passthrough_keeps_8bit_unchanged():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    res = preprocess_field("x", img, img, 0.5, {"enabled": False})
    assert np.array_equal(res.nuclear.preprocessed_u8, img)


def test_normalize_disabled_rescales_not_clips():
    img = np.zeros((32, 32), np.uint16)
    img[4:8, 4:8] = 1000
    img[20:24, 20:24] = 3000
    cfg = {"enabled": True, "background": {"method": "none"}, "normalize": {"enabled": False}}
    u8 = preprocess_field("x", img, img, 0.5, cfg).nuclear.preprocessed_u8
    assert u8[22, 22] == 255 and 0 < u8[6, 6] < 255  # both would be clipped to 255 before


def test_qc_saturation_uses_dtype_max():
    img16 = np.full((20, 20), 300, np.uint16)  # far below 65535: not saturated
    m = np.zeros((20, 20), np.int32)
    q = compute_image_qc_metrics("x", "Mat", "Gel1", img16, img16, m, m, 0, 0.5, {})
    assert q.nuclear_saturation_frac == 0.0 and not q.flag_saturated
    img16[:10] = 65535
    q = compute_image_qc_metrics("x", "Mat", "Gel1", img16, img16, m, m, 0, 0.5, {})
    assert q.flag_saturated


# --- item 7 (denoise) -------------------------------------------------------------------------------

def test_gaussian_denoise_runs_and_is_reported():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    cfg = {"enabled": True, "background": {"method": "none"}, "normalize": {"enabled": True},
           "denoise": {"method": "gaussian", "sigma_um": 1.0}}
    res = preprocess_field("x", img, img, 0.5, cfg)
    assert "denoise_gaussian" in res.nuclear.methods_applied
    assert res.nuclear.preprocessed_f32.std() < (img / 255.0).std()


def test_blank_gel_fallback_not_reported_as_applied(tmp_path):
    img = np.full((32, 32), 50, np.uint8)
    cfg = {"enabled": True, "background": {"method": "blank_gel_median", "blank_glob": str(tmp_path / "none*.tif")}}
    res = preprocess_field("x", img, img, 0.5, cfg)
    assert not any(m.startswith("background_subtraction") for m in res.nuclear.methods_applied)


# --- item 12: RGB and 3D --------------------------------------------------------------------------

def test_rgb_converted_to_grayscale_and_stack_rejected():
    rgb = np.zeros((40, 50, 3), np.uint8)
    rgb[..., 1] = 200
    g = ensure_2d_grayscale(rgb)
    assert g.shape == (40, 50) and g.dtype == np.uint8 and 110 < g[0, 0] < 125
    assert ensure_2d_grayscale(np.zeros((1, 40, 50), np.uint8)).shape == (40, 50)
    with pytest.raises(ValueError, match="2D only"):
        ensure_2d_grayscale(np.zeros((5, 40, 50), np.uint8))


def test_dedup_rejects_3d_arrays_and_label_images_are_noop():
    with pytest.raises(ValueError, match="3D"):
        deduplicate_masks(np.zeros((3, 10, 10), np.int32))
    m = np.zeros((20, 20), np.int32)
    m[2:5, 2:5] = 1
    m[10:15, 10:15] = 2
    _, _, stats = postprocess_field_masks(
        "x", m, np.zeros_like(m, dtype=np.uint8), 1.0,
        {"enabled": True, "size_gate": {"enabled": False}, "splitting": {"enabled": False},
         "border_exclusion": {"enabled": False}},
    )
    assert "deduplicate" not in stats.methods_applied


# --- item 7: size gate bounds derive from the reference when unset ---------------------------------

def test_size_gate_bounds_default_from_reference_volume():
    b = resolve_size_gate_bounds({"size_gate": {"reference_vol_um3": 800.0, "min_vol_um3": None, "max_vol_um3": None}})
    assert b["min_vol_um3"] == pytest.approx(200.0) and b["max_vol_um3"] == pytest.approx(3200.0)
    assert b["min_vol_um3_source"].startswith("derived")
    b2 = resolve_size_gate_bounds({"size_gate": {"reference_vol_um3": 800.0, "min_vol_um3": 100.0}})
    assert b2["min_vol_um3"] == 100.0 and b2["min_vol_um3_source"] == "config"
    assert b2["min_multiplier_of_reference"] == pytest.approx(0.125)


# --- item 17: accepted split keeps every parent pixel ---------------------------------------------

def test_accepted_split_keeps_unreached_parent_pixels():
    h, w = 80, 120
    mask = np.zeros((h, w), np.int32)
    img = np.zeros((h, w), np.uint8)
    for cx in (30, 60):
        rr, cc = draw.disk((40, cx), 14, shape=(h, w))
        mask[rr, cc] = 1
        img[rr, cc] = 120
        img[38:43, cx - 2:cx + 3] = 230
    # a seedless appendage connected through a 1-px bridge
    mask[40, 74:80] = 1
    mask[35:46, 80:92] = 1
    parent_area = int((mask > 0).sum())
    out, n_cand, n_acc, *_ = split_merged_nuclei(
        mask, img, 1.0, split_factor=0.5, h_maxima=10.0, min_distance_um=5.0,
        min_vol_um3=1.0, max_vol_um3=1e9, reference_vol_um3=1000.0,
    )
    assert n_acc == 1 and int(out.max()) >= 2
    assert int((out > 0).sum()) == parent_area


# --- item 6: arm-independent extent, hull fallback flag, border strip in the denominator -----------

def _field(size=200):
    actin = np.full((size, size), 10, np.uint8)
    rr, cc = draw.disk((size // 2, size // 2), 70, shape=actin.shape)
    actin[rr, cc] = 180
    labels = np.zeros_like(actin, dtype=np.int32)
    k = 0
    for y in range(40, 161, 20):
        for x in range(40, 161, 20):
            k += 1
            r2, c2 = draw.disk((y, x), 4, shape=actin.shape)
            labels[r2, c2] = k
    return actin, labels


def test_extent_from_raw_actin_is_same_for_every_arm():
    actin, labels = _field()
    a = compute_spheroid_extent_details(actin, labels, 1.0)
    b = compute_spheroid_extent_details(actin, labels[:, ::-1].copy(), 1.0)  # different masks (another arm)
    assert a.area_um2 == b.area_um2 and a.method_used == "actin_threshold" and not a.hull_fallback


def test_tiny_actin_mask_falls_back_to_hull_with_flag():
    actin = np.zeros((100, 100), np.uint8)
    actin[50, 50] = 255
    labels = np.zeros((100, 100), np.int32)
    labels[20:30, 20:30] = 1
    labels[60:70, 60:70] = 2
    e = compute_spheroid_extent_details(actin, labels, 1.0)
    assert e.hull_fallback and e.method_used == "nuclei_convex_hull_fallback"
    _, row = extract_objects_and_image_features("x", "Mat", "Gel1", actin, actin, labels, 1.0, "", {}, spheroid_extent=e)
    assert row["spheroid_extent_hull_fallback"] is True


def test_border_strip_removed_from_density_area():
    actin = np.full((100, 100), 200, np.uint8)  # spheroid fills the field
    actin[:2, :2] = 0
    labels = np.zeros((100, 100), np.int32)
    labels[1:6, 40:45] = 1    # touches border strip -> excluded
    labels[50:55, 50:55] = 2
    border, _ = identify_border_objects(labels, 1.0, 5.0)
    e = compute_spheroid_extent_details(actin, labels, 1.0)
    _, with_strip = extract_objects_and_image_features(
        "x", "Mat", "Gel1", actin, actin, labels, 1.0, "", {}, border_excluded_labels=border,
        spheroid_extent=e, border_clearance_px=5,
    )
    strip = border_strip_mask((100, 100), 5)
    assert with_strip["n_nuclei"] == 1
    assert with_strip["density_area_um2"] == pytest.approx(float((e.mask & ~strip).sum()), abs=0.1)
    assert with_strip["density_area_um2"] < e.area_um2 - 1000
    assert with_strip["nuclei_per_mm2"] == pytest.approx(1 / (with_strip["density_area_um2"] * 1e-6), rel=1e-3)
    assert with_strip["spheroid_area_um2"] == pytest.approx(e.area_um2, abs=0.1)  # column meaning unchanged


def test_radial_distance_nan_when_radius_nan():
    labels = np.zeros((50, 50), np.int32)
    labels[20:25, 20:25] = 1
    img = np.zeros((50, 50), np.uint8)
    objs, _ = extract_objects_and_image_features(
        "x", "Mat", "Gel1", img, img, labels, 1.0, "", {},
        spheroid_extent=(float("nan"), 25.0, 25.0, float("nan")),
    )
    assert np.isnan(objs[0]["normalized_radial_distance"])


def test_to_uint8_float_and_bool():
    assert to_uint8(np.array([[0.0, 1.0]], np.float32)).tolist() == [[0, 255]]
    assert to_uint8(np.array([[0.0, 1000.0]], np.float32)).tolist() == [[0, 255]]
