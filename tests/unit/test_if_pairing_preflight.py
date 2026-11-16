"""Filename parsing, shared discovery, calibration policy, preflight dtype/shape/weights checks."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from matrix_mat.if_density.pairing import (
    discover_and_pair_fields_detailed,
    discover_channel_files,
    parse_field_metadata,
    parse_field_name,
)
from matrix_mat.if_density.preflight import audit_if_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_if_dataset"


@pytest.mark.parametrize("name,expected", [
    ("SKOV3 Spheroid D7 Mor_S34D30-Gel8-1-2", ("SKOV3", "7", "S34D30", "Gel8", "1-2")),
    ("SKOV3 Spheroid D7 Mor_Mat-Gel1-3", ("SKOV3", "7", "Mat", "Gel1", "3")),
    ("OVCAR3 Spheroid D14 Mor_S4.5D30-Gel2-1", ("OVCAR3", "14", "S4.5D30", "Gel2", "1")),
    ("SKOV3 Spheroid D7 Mor_S40_D30-Gel1", ("SKOV3", "7", "S40_D30", "Gel1", "1")),
    ("HeLa Spheroid D3_S50-Gel7-2", ("HeLa", "3", "S50", "Gel7", "2")),
])
def test_generalised_filename_pattern(name, expected):
    info = parse_field_name(name)
    assert (info["cell_line"], info["day"], info["material"], info["replicate"], info["subfield"]) == expected


def test_unparseable_name_raises_instead_of_unknown():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_field_metadata("Field1")


def _write_pair(d: Path, base: str, shape=(40, 40), dtype=np.uint8, ext00=".tif", ext02=".tif", res=False):
    # 0.5 um/px written like the real microscope files: pixels per centimetre.
    kw = {"resolution": (2e4, 2e4), "resolutionunit": "CENTIMETER"} if res else {}
    tifffile.imwrite(d / f"{base}_ch00{ext00}", np.zeros(shape, dtype), **kw)
    tifffile.imwrite(d / f"{base}_ch02{ext02}", np.zeros(shape, dtype), **kw)


def test_unparseable_fields_are_skipped_not_pooled(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "d")
    _write_pair(tmp_path / "d", "random_field")
    res = discover_and_pair_fields_detailed(tmp_path / "d", allow_unpaired=True, allow_uncalibrated=True)
    assert res.unparsed_bases == ["random_field"]
    assert "Unknown" not in {r.material for r in res.records}
    assert len(res.records) == 2


def test_multiple_cell_lines_are_not_pooled(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "d")
    _write_pair(tmp_path / "d", "OVCAR3 Spheroid D7 Mor_Mat-Gel1-1")
    res = discover_and_pair_fields_detailed(tmp_path / "d", allow_uncalibrated=True)
    mats = {r.material for r in res.records}
    assert "SKOV3_D7_Mat" in mats and "OVCAR3_D7_Mat" in mats


def test_shared_discovery_is_case_insensitive_and_exact_suffix(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _write_pair(d, "SKOV3 Spheroid D7 Mor_Mat-Gel1-1", ext00=".TIF", ext02=".tiff")
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel2-1_ch00_extra.tif", np.zeros((4, 4), np.uint8))
    disc = discover_channel_files(d)
    assert set(disc.nuclear) == {"SKOV3 Spheroid D7 Mor_Mat-Gel1-1"}
    assert set(disc.actin) == {"SKOV3 Spheroid D7 Mor_Mat-Gel1-1"}
    assert [p.name for p in disc.unrecognized] == ["SKOV3 Spheroid D7 Mor_Mat-Gel2-1_ch00_extra.tif"]
    report = audit_if_dataset(d, model_path=tmp_path / "none", check_weights=False)
    assert report.stats["paired_fields"] == 1


def test_tif_tiff_collision_detected(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    _write_pair(d, "SKOV3 Spheroid D7 Mor_Mat-Gel1-1")
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel1-1_ch00.tiff", np.zeros((40, 40), np.uint8))
    assert discover_channel_files(d).collisions
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_and_pair_fields_detailed(d)
    report = audit_if_dataset(d, check_weights=False)
    assert not report.passed and any("Ambiguous" in e for e in report.fatal_errors)


def test_uncalibrated_fields_fail_unless_allowed(tmp_path):
    with pytest.raises(ValueError, match="allow_uncalibrated"):
        discover_and_pair_fields_detailed(FIXTURE, allow_uncalibrated=False)
    recs = discover_and_pair_fields_detailed(FIXTURE, allow_uncalibrated=True).records
    assert all(r.scale_source.startswith("config_default_uncalibrated") and not r.is_calibrated for r in recs)
    d = tmp_path / "cal"
    d.mkdir()
    _write_pair(d, "SKOV3 Spheroid D7 Mor_Mat-Gel1-1", res=True)
    rec = discover_and_pair_fields_detailed(d, allow_uncalibrated=False).records[0]
    assert rec.is_calibrated and rec.pixel_size_um == pytest.approx(0.5)


def test_preflight_checks_shape_mismatch_and_stacks(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel1-1_ch00.tif", np.zeros((40, 40), np.uint8))
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel1-1_ch02.tif", np.zeros((40, 50), np.uint8))
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel2-1_ch00.tif", np.zeros((5, 40, 40), np.uint8))
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel2-1_ch02.tif", np.zeros((5, 40, 40), np.uint8))
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel3-1_ch00.tif", np.zeros((40, 40), np.uint16))
    tifffile.imwrite(d / "SKOV3 Spheroid D7 Mor_Mat-Gel3-1_ch02.tif", np.zeros((40, 40), np.uint16))
    report = audit_if_dataset(d, check_weights=False)
    errs = " ".join(report.fatal_errors)
    assert "does not match" in errs and "multi-plane" in errs
    assert any("uint16" in w for w in report.warnings)
    assert report.stats["dtypes"]["uint16"] == 2


def test_preflight_fails_on_missing_weights_with_fetch_command(tmp_path):
    report = audit_if_dataset(FIXTURE, model_path=tmp_path / "missing_weights")
    assert not report.passed
    assert any("python tools/fetch_models.py" in e for e in report.fatal_errors)
    (tmp_path / "w").write_bytes(b"x")
    ok = audit_if_dataset(FIXTURE, model_path=tmp_path / "w")
    assert ok.passed  # present but wrong size: a warning (full SHA-256 is checked at model load)
    assert any("models.lock.json" in w for w in ok.warnings)


def test_preflight_uncalibrated_policy(tmp_path):
    (tmp_path / "w").write_bytes(b"x")
    assert audit_if_dataset(FIXTURE, model_path=tmp_path / "w").passed
    strict = audit_if_dataset(FIXTURE, model_path=tmp_path / "w", allow_uncalibrated=False)
    assert not strict.passed and any("pixel scale" in e for e in strict.fatal_errors)
