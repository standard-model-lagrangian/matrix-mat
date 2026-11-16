"""Regression tests for brightfield pairing fixes (FAIL exclusion, pre-assignment gate, config sources)."""

from __future__ import annotations

import pytest

from matrix_mat.brightfield.config import PairingConfig, PipelineConfig, StatsConfig
from matrix_mat.brightfield.measure import SpheroidObjectRecord
from matrix_mat.brightfield.pair import SpheroidPairer, pair_dataset, update_objects_with_pair_ids


def _obj(object_id, x, y=100.0, tp="t0", flag="PASS", d_px=20.0, d_um=100.0, vol=1.0e6):
    return SpheroidObjectRecord(
        object_id=object_id, image_id=f"img_{tp}", timepoint=tp, condition="Mat", replicate="gel1",
        fov="0001", pair_key="Mat_gel1_0001", label_idx=1, pixel_size_um=1.0, area_px=314.0,
        perimeter_px=63.0, equivalent_diameter_px=d_px, centroid_y=y, centroid_x=x, area_um2=314.0,
        equivalent_diameter_um=d_um, major_axis_um=d_um, minor_axis_um=d_um, aspect_ratio=1.0,
        volume_sphere_um3=vol, volume_ellipsoid_um3=vol, volume_discrepancy_ratio=0.0,
        circularity=0.95, solidity=0.98, eccentricity=0.1, mean_interior_intensity=0.2,
        mean_background_ring_intensity=0.8, contrast_ratio=0.7, qc_flag=flag, qc_reasons="None",
    )


def test_fail_debris_excluded_from_assignment():
    """A FAIL t0 debris object closer to the t7 spheroid must not steal its partner."""
    t0_pass = _obj("t0_obj1", 100.0)
    t0_fail = _obj("t0_fail2", 130.0, flag="FAIL")
    t7_pass = _obj("t7_obj1", 125.0, tp="t7")
    pairs, un0, un7 = SpheroidPairer(PairingConfig()).pair_fov_objects([t0_pass, t0_fail], [t7_pass])
    assert len(pairs) == 1
    assert pairs[0].t0_object_id == "t0_obj1"
    assert pairs[0].t7_object_id == "t7_obj1"
    assert un0 == [t0_fail]
    assert un7 == []


def test_distance_gate_applied_before_assignment():
    """Over-limit edges must not distort the optimal assignment of admissible pairs."""
    cfg = PairingConfig(min_displacement_px=50.0, max_centroid_shift_fraction=0.0)
    t0 = [_obj("a0", 0.0), _obj("b0", 100.0)]
    t7 = [_obj("a7", 90.0, tp="t7"), _obj("b7", 300.0, tp="t7")]
    pairs, un0, un7 = SpheroidPairer(cfg).pair_fov_objects(t0, t7)
    assert [(p.t0_object_id, p.t7_object_id) for p in pairs] == [("b0", "a7")]
    assert pairs[0].centroid_distance_px == pytest.approx(10.0)
    assert [o.object_id for o in un0] == ["a0"]
    assert [o.object_id for o in un7] == ["b7"]


def test_greedy_fallback_respects_gate(monkeypatch):
    import matrix_mat.brightfield.pair as pair_mod

    def _boom(_cost):
        raise ValueError("degenerate")

    monkeypatch.setattr(pair_mod, "linear_sum_assignment", _boom)
    cfg = PairingConfig(min_displacement_px=50.0, max_centroid_shift_fraction=0.0)
    t0 = [_obj("a0", 0.0), _obj("b0", 100.0)]
    t7 = [_obj("a7", 90.0, tp="t7"), _obj("b7", 300.0, tp="t7")]
    pairs, _, _ = SpheroidPairer(cfg).pair_fov_objects(t0, t7)
    assert [(p.t0_object_id, p.t7_object_id) for p in pairs] == [("b0", "a7")]


def test_fail_objects_never_inherit_pair_id():
    """Objects are matched to pair ids only through non-FAIL records."""
    t0_pass = _obj("Mat_gel1_0001_t0_obj1", 100.0)
    t0_fail = _obj("Mat_gel1_0001_t0_fail1", 400.0, flag="FAIL")
    t7_pass = _obj("Mat_gel1_0001_t7_obj1", 105.0, tp="t7")
    pairs, updated = pair_dataset([t0_pass, t0_fail, t7_pass], PairingConfig())
    by_id = {o.object_id: o.pair_id for o in updated}
    assert by_id["Mat_gel1_0001_t0_obj1"] == pairs[0].pair_id
    assert by_id["Mat_gel1_0001_t0_fail1"] == "UNPAIRED"


def test_duplicate_object_ids_rejected():
    a = _obj("dup", 100.0)
    b = _obj("dup", 300.0)
    with pytest.raises(ValueError, match="[Dd]uplicate object_id"):
        update_objects_with_pair_ids([a, b], [])


def test_stats_section_gates_reach_pairing():
    """YAML stats.* gate values must actually drive pairing (single source of truth)."""
    cfg = PipelineConfig(stats=StatsConfig(min_t0_d_um_denominator=120.0, min_fold_change=0.5, max_fold_change=3.0))
    assert cfg.pairing.min_t0_d_um_denominator == 120.0
    assert cfg.pairing.min_fold_change == 0.5
    assert cfg.pairing.max_fold_change == 3.0
    t0 = _obj("a0", 100.0, d_um=100.0, vol=1.0e6)
    t7 = _obj("a7", 101.0, tp="t7", vol=4.0e6)
    pairs, _ = pair_dataset([t0, t7], cfg)
    p = pairs[0]
    assert p.denominator_gate_pass is False  # 100 um < 120 um
    assert p.combined_qc_flag == "REVIEW"  # FC 4 > 3
    assert "extreme_expansion" in p.qc_reasons


def test_conflicting_stats_and_pairing_gates_raise():
    with pytest.raises(ValueError, match="[Cc]onflicting"):
        PipelineConfig(stats=StatsConfig(min_fold_change=0.5), pairing=PairingConfig(min_fold_change=0.3))


def test_stats_gate_from_yaml(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("stats:\n  max_fold_change: 7.5\n")
    cfg = PipelineConfig.from_yaml(y)
    assert cfg.pairing.max_fold_change == 7.5


def test_pairing_reads_canonical_names_after_mutation():
    """Mutating a canonical PairingConfig field after construction must take effect."""
    cfg = PairingConfig()
    cfg.min_displacement_px = 5.0
    cfg.max_centroid_shift_fraction = 0.0
    cfg.max_fold_change = 2.0
    t0 = _obj("a0", 100.0)
    t7_far = _obj("a7", 120.0, tp="t7")  # 20 px away: beyond the mutated 5 px floor
    pairs, _, _ = SpheroidPairer(cfg).pair_fov_objects([t0], [t7_far])
    assert pairs == []
    t7_near = _obj("b7", 102.0, tp="t7", vol=3.0e6)
    pairs, _, _ = SpheroidPairer(cfg).pair_fov_objects([t0], [t7_near])
    assert pairs[0].combined_qc_flag == "REVIEW"
