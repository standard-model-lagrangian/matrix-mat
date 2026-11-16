"""Anchored, normalising filename parser (0.2.2) + snapshot of real-data names."""

import pytest

from matrix_mat.core.naming import parse_image_filename

# Snapshot of real EVOS TRANS names (Chuling cells, Day0 260805 / D7 260812) and their parse
# result before the 0.2.2 parser rewrite. These must keep parsing identically.
REAL_SNAPSHOT = {
    "Day0 Mat gel1_0001_TRANS.tif": ("t0", "Mat", "gel1", "0001", "Mat_gel1_0001"),
    "Day0 S34D30 gel1_0001_TRANS.tif": ("t0", "S34D30", "gel1", "0001", "S34D30_gel1_0001"),
    "Day0 S40D30 gel1_0001_TRANS.tif": ("t0", "S40D30", "gel1", "0001", "S40D30_gel1_0001"),
    "Day0 S43D20 gel1_0001_TRANS.tif": ("t0", "S43D20", "gel1", "0001", "S43D20_gel1_0001"),
    "Day0 S46D10 gel1_0001_TRANS.tif": ("t0", "S46D10", "gel1", "0001", "S46D10_gel1_0001"),
    "Day0 S50 gel1_0001_TRANS.tif": ("t0", "S50", "gel1", "0001", "S50_gel1_0001"),
    "Day7 Mat gel8_0002_TRANS.tif": ("t7", "Mat", "gel8", "0002", "Mat_gel8_0002"),
    "Day7 S50 gel5_0004_TRANS.tif": ("t7", "S50", "gel5", "0004", "S50_gel5_0004"),
    "Day7 S34D30 gel1_0001_TRANS.tif": ("t7", "S34D30", "gel1", "0001", "S34D30_gel1_0001"),
    "Day7 S46D10 gel1_0001_TRANS.tif": ("t7", "S46D10", "gel1", "0001", "S46D10_gel1_0001"),
    "Day7 S43D20 gel1_0001_TRANS.tif": ("t7", "S43D20", "gel1", "0001", "S43D20_gel1_0001"),
}


@pytest.mark.parametrize("name,expected", sorted(REAL_SNAPSHOT.items()))
def test_real_names_parse_identically(name, expected):
    r = parse_image_filename(name)
    assert (r["timepoint"], r["condition"], r["replicate"], r["fov"], r["pair_key"]) == expected
    assert r["image_id"] == name[:-4]
    assert r["flag_unmatched"] is False


@pytest.mark.parametrize("name,cond", [
    ("Day0 S34 D30 gel1_0001_TRANS.tif", "S34D30"),
    ("Day0_Mat_gel1_0001_TRANS.tif", "Mat"),
    ("day0 mat GEL1_0001_TRANS.tif", "Mat"),
    ("Day0 s34d30 gel1_0001_trans.tif", "S34D30"),
])
def test_separators_and_case_normalised(name, cond):
    r = parse_image_filename(name)
    assert r["condition"] == cond
    assert r["pair_key"] == f"{cond}_gel1_0001"


def test_zero_padding_normalised():
    a = parse_image_filename("Day0 Mat gel01_001_TRANS.tif")
    b = parse_image_filename("Day0 Mat gel1_0001_TRANS.tif")
    assert a["pair_key"] == b["pair_key"] == "Mat_gel1_0001"


def test_source_dir_timepoint_underscore_name():
    r = parse_image_filename("Mat_gel2_0005_TRANS.tif", source_dir_timepoint="Day7")
    assert (r["condition"], r["timepoint"], r["pair_key"]) == ("Mat", "t7", "Mat_gel2_0005")


@pytest.mark.parametrize("bad", [
    "Day0 Mat gel1_0001_TRANS (1).tif",
    "Day0 Mat gel1_0001_TRANS copy.tif",
    "Day0 Mat+drug gel1_0001_TRANS.tif",
    "Day0 Mat extra gel1_0001_TRANS.tif",
    "prefix Day0 Mat gel1_0001_TRANS.tif",
    "Day0 gel1_0001_TRANS.tif",
])
def test_ambiguous_or_copied_names_raise(bad):
    with pytest.raises(ValueError):
        parse_image_filename(bad)
