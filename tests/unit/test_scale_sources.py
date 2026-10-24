"""End-to-end scale extraction from real TIFF files written with each metadata convention."""
import json

import numpy as np
import pytest
import tifffile

from matrix_mat.core.scale import ScaleExtractor, extract_pixel_size

IMG = np.zeros((32, 48), dtype=np.uint8)


def _write(path, **kw):
    tifffile.imwrite(path, IMG, **kw)
    return path


def test_evos_tag_37510_json(tmp_path):
    payload = json.dumps({"Target": {"PixelSize": 1.518817}, "ObjectiveSettings": {"Magnification": 4, "NumericalAperture": 0.13}})
    p = _write(tmp_path / "evos.tif", extratags=[(37510, "s", 0, "header " + payload, True)])
    info = ScaleExtractor().extract(p)
    assert info.source == "evos_tag_37510"
    assert info.is_calibrated
    assert info.pixel_size_um == pytest.approx(1.518817)
    assert info.magnification == 4


def test_evos_implausible_value_is_not_calibrated(tmp_path):
    p = _write(tmp_path / "evos_bad.tif", extratags=[(37510, "s", 0, json.dumps({"MicronsPerPixel": 350.0}), True)])
    info = ScaleExtractor().extract(p)
    assert info.is_calibrated is False


@pytest.mark.parametrize("value,unit,expected", [
    ("0.65", "µm", 0.65),
    ("650", "nm", 0.65),
    ("0.00065", "mm", 0.65),
    ("0.65", None, 0.65),
])
def test_ome_xml_physical_size_respects_unit(tmp_path, value, unit, expected):
    unit_attr = f' PhysicalSizeXUnit="{unit}"' if unit else ""
    ome = (
        '<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" SizeX="48" SizeY="32" PhysicalSizeX="{value}"{unit_attr} '
        'Type="uint8" DimensionOrder="XYZCT" SizeZ="1" SizeC="1" SizeT="1"/></Image></OME>'
    )
    p = _write(tmp_path / "ome.tif", description=ome.encode("utf-8"), metadata=None)
    info = ScaleExtractor().extract(p)
    assert info.source == "OME-XML Tag 270"
    assert info.pixel_size_um == pytest.approx(expected)
    assert info.is_calibrated


def test_ome_xml_implausible_size_not_calibrated(tmp_path):
    ome = (
        '<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image ID="Image:0"><Pixels ID="Pixels:0" SizeX="48" SizeY="32" PhysicalSizeX="500" PhysicalSizeXUnit="µm" '
        'Type="uint8" DimensionOrder="XYZCT" SizeZ="1" SizeC="1" SizeT="1"/></Image></OME>'
    )
    p = _write(tmp_path / "ome_bad.tif", description=ome.encode("utf-8"), metadata=None)
    info = ScaleExtractor().extract(p)
    assert info.is_calibrated is False


def test_resolution_tags_in_centimetres(tmp_path):
    p = _write(tmp_path / "res.tif", resolution=(10000 / 0.5, 10000 / 0.5), resolutionunit="CENTIMETER", metadata=None)
    info = ScaleExtractor().extract(p)
    assert info.pixel_size_um == pytest.approx(0.5)
    assert info.is_calibrated


def test_no_metadata_falls_back_to_uncalibrated_default(tmp_path):
    p = _write(tmp_path / "plain.tif", metadata=None)
    info = ScaleExtractor(default_pixel_size_um=1.25).extract(p)
    assert info.pixel_size_um == pytest.approx(1.25)
    assert info.is_calibrated is False


def test_cli_override_wins_and_is_calibrated(tmp_path):
    p = _write(tmp_path / "plain.tif", metadata=None)
    info = ScaleExtractor(cli_override_um=0.9).extract(p)
    assert (info.pixel_size_um, info.source, info.is_calibrated) == (0.9, "cli_override", True)
    assert extract_pixel_size(p, override_um=0.9) == (0.9, "cli_override")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ScaleExtractor().extract(tmp_path / "nope.tif")
