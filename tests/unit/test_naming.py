import pytest
from spheroid_pipeline_v2.naming import parse_timepoint, parse_image_filename


@pytest.mark.parametrize("inp, expected", [
    ("Day0", 0),
    ("Day7", 7),
    ("D7", 7),
    ("Day10", 10),
    ("t0", 0),
    ("t7", 7),
    ("d0", 0),
    ("Day 0", 0),
    ("Day 7", 7),
    ("0", 0),
    ("7", 7),
    ("10", 10),
])
def test_parse_timepoint(inp, expected):
    """parse_timepoint accurately parses diverse timepoint strings into integer day values."""
    assert parse_timepoint(inp) == expected


@pytest.mark.parametrize("bad_inp", [
    "DayX",
    "invalid",
    "foo_bar",
    "Day",
    "t_unknown",
    "",
])
def test_parse_timepoint_raises_on_unknown(bad_inp):
    """parse_timepoint raises ValueError on unknown or malformed timepoint expressions."""
    with pytest.raises(ValueError):
        parse_timepoint(bad_inp)


def test_unmatched_filename_raises_or_flags():
    """Unmatched filename must raise ValueError or return flagged dict, never default to gel1/0001/t7."""
    bad_filename = "unstructured_microscope_capture_001.tif"
    # Should either raise ValueError or return flagged record with pair_key=None
    try:
        res = parse_image_filename(bad_filename)
        assert res.get("flag_unmatched", False) is True or res.get("pair_key") is None
        assert res.get("replicate") != "gel1" or res.get("fov") != "0001" or res.get("timepoint") != "t7", (
            "Must not silently default to gel1/0001/t7"
        )
    except ValueError:
        # Raising ValueError on unmatched filename is also valid per spec
        pass


def test_timepoint_taken_from_source_directory_when_configured():
    """When configured, timepoint can be inferred from the source directory name."""
    filename = "Mat_gel2_0005_TRANS.tif"
    res0 = parse_image_filename(filename, source_dir_timepoint="Day0")
    assert res0["timepoint"] == "t0"

    res7 = parse_image_filename(filename, source_dir_timepoint="Day7")
    assert res7["timepoint"] == "t7"
