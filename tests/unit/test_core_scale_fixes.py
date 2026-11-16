"""Regression tests for core.scale bug fixes (0.2.2), using synthetic TIFFs written by tifffile."""

import json

import numpy as np
import pytest
import tifffile

from matrix_mat.core.scale import ScaleExtractor, ScaleInfo, extract_pixel_size, extract_scale_info

IMG = np.zeros((32, 48), dtype=np.uint8)
DEFAULT = 1.25


def _ex():
    return ScaleExtractor(default_pixel_size_um=DEFAULT)


# --- (a) ImageJ / Fiji calibration in XResolution ---------------------------------------

@pytest.mark.parametrize("unit,px,expected_um", [
    ("um", 0.65, 0.65),
    ("micron", 0.65, 0.65),
    ("\\u00B5m", 0.65, 0.65),
    ("nm", 650.0, 0.65),
    ("mm", 0.00065, 0.65),
    ("cm", 0.000065, 0.65),
    ("inch", 0.65 / 25400.0, 0.65),
])
def test_imagej_xresolution_with_unit(tmp_path, unit, px, expected_um):
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, IMG, imagej=True, resolution=(1.0 / px, 1.0 / px), metadata={"unit": unit})
    info = _ex().extract(p)
    assert info.source == "ImageJ metadata"
    assert info.is_calibrated is True
    assert info.pixel_size_um == pytest.approx(expected_um, rel=1e-5)


def test_imagej_utf8_micro_sign_with_resolution_unit_none(tmp_path):
    p = tmp_path / "ij_mu.tif"
    desc = "ImageJ=1.54f\nunit=µm\n".encode("utf-8")
    tifffile.imwrite(p, IMG, description=desc, resolution=(1 / 0.65, 1 / 0.65), resolutionunit="NONE", metadata=None)
    info = _ex().extract(p)
    assert (info.source, info.is_calibrated) == ("ImageJ metadata", True)
    assert info.pixel_size_um == pytest.approx(0.65, rel=1e-5)


def test_imagej_pixel_unit_is_uncalibrated(tmp_path):
    p = tmp_path / "ij_px.tif"
    tifffile.imwrite(p, IMG, imagej=True, resolution=(1.0, 1.0), metadata={"unit": "pixel"})
    info = _ex().extract(p)
    assert info.is_calibrated is False
    assert info.pixel_size_um == DEFAULT


def test_imagej_pixelwidth_unknown_unit_is_not_assumed_um():
    info = ScaleExtractor()._extract_imagej({270: "ImageJ=1.53\nunit=furlong\npixelWidth=0.5\n"}, {})
    assert info is None or info.is_calibrated is False


def test_imagej_pixelwidth_inch_converted():
    info = ScaleExtractor()._extract_imagej({270: "ImageJ=1.53\nunit=inch\npixelWidth=0.0000256\n"}, {})
    assert info is not None and info.pixel_size_um == pytest.approx(0.65024)


# --- (b) rejected calibrations fall back to the configured default ----------------------

def test_72_dpi_returns_default_not_nonsense(tmp_path):
    p = tmp_path / "dpi72.tif"
    tifffile.imwrite(p, IMG, resolution=(72, 72), resolutionunit="INCH", metadata=None)
    info = _ex().extract(p)
    assert info.is_calibrated is False
    assert info.pixel_size_um == DEFAULT
    assert "72" in info.source and "default" in info.source
    assert info.warning


def test_96_dpi_returns_default(tmp_path):
    p = tmp_path / "dpi96.tif"
    tifffile.imwrite(p, IMG, resolution=(96, 96), resolutionunit="INCH", metadata=None)
    info = _ex().extract(p)
    assert info.is_calibrated is False
    assert info.pixel_size_um == DEFAULT


def test_implausible_evos_returns_default(tmp_path):
    p = tmp_path / "evos_bad.tif"
    tifffile.imwrite(p, IMG, extratags=[(37510, "s", 0, json.dumps({"MicronsPerPixel": 350.0}), True)])
    info = _ex().extract(p)
    assert info.is_calibrated is False
    assert info.pixel_size_um == DEFAULT
    assert "implausible" in info.source


def test_extract_pixel_size_backcompat_and_info(tmp_path):
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, IMG, metadata=None)
    assert extract_pixel_size(p, default_um=DEFAULT) == (DEFAULT, "config_default_uncalibrated")
    info = extract_pixel_size(p, default_um=DEFAULT, return_info=True)
    assert isinstance(info, ScaleInfo) and info.is_calibrated is False
    assert extract_scale_info(p, default_um=DEFAULT).is_calibrated is False


# --- (c) OME-XML robustness ---------------------------------------------------------------

def _ome(ns_year="2016-06", prefix="", value="0.65"):
    return (
        f'{prefix}<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/{ns_year}">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" SizeX="48" SizeY="32" PhysicalSizeX="{value}" '
        'PhysicalSizeXUnit="µm" Type="uint8" DimensionOrder="XYZCT" SizeZ="1" SizeC="1" SizeT="1"/></Image></OME>'
    )


@pytest.mark.parametrize("ns_year", ["2016-06", "2015-01", "2013-06"])
def test_ome_any_schema_year(tmp_path, ns_year):
    p = tmp_path / "ome.tif"
    tifffile.imwrite(p, IMG, description=_ome(ns_year).encode(), metadata=None)
    info = _ex().extract(p)
    assert info.source == "OME-XML Tag 270"
    assert info.pixel_size_um == pytest.approx(0.65)


def test_ome_with_leading_bioformats_comment(tmp_path):
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?><!-- Warning: this comment is an OME-XML metadata block -->'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
        '<Pixels ID="Pixels:0" SizeX="48" SizeY="32" PhysicalSizeX="0.5" Type="uint8" DimensionOrder="XYZCT" '
        'SizeZ="1" SizeC="1" SizeT="1"/></Image></OME>'
    )
    p = tmp_path / "bf.tif"
    tifffile.imwrite(p, IMG, description=xml.encode(), metadata=None)
    info = _ex().extract(p)
    assert info.source == "OME-XML Tag 270"
    assert info.pixel_size_um == pytest.approx(0.5)


def test_ome_prefixed_namespace():
    xml = (
        '<ome:OME xmlns:ome="http://www.openmicroscopy.org/Schemas/OME/2016-06"><ome:Image ID="Image:0">'
        '<ome:Pixels ID="Pixels:0" PhysicalSizeX="0.7"/></ome:Image></ome:OME>'
    )
    info = ScaleExtractor()._extract_ome_xml({270: xml}, (48, 32))
    assert info is not None and info.pixel_size_um == pytest.approx(0.7)


# --- (d) lenient EVOS magnification ----------------------------------------------------------

def test_evos_magnification_with_x_suffix(tmp_path):
    payload = json.dumps({"MicronsPerPixel": "1.5188172690164", "Magnification": "4x", "Objective NA": "0.13"})
    p = tmp_path / "evos.tif"
    tifffile.imwrite(p, IMG, extratags=[(37510, "s", 0, "ASCIIEvos2 1.8: " + payload, True)])
    info = _ex().extract(p)
    assert info.source == "evos_tag_37510"
    assert info.is_calibrated
    assert info.pixel_size_um == pytest.approx(1.5188172690164)
    assert info.magnification == 4.0
    assert info.numerical_aperture == pytest.approx(0.13)


def test_evos_unparseable_magnification_kept_as_none(tmp_path):
    payload = json.dumps({"MicronsPerPixel": "1.5", "Magnification": "high"})
    p = tmp_path / "evos2.tif"
    tifffile.imwrite(p, IMG, extratags=[(37510, "s", 0, payload, True)])
    info = _ex().extract(p)
    assert info.is_calibrated and info.magnification is None and info.pixel_size_um == 1.5


# --- regression: real-data conventions parse identically --------------------------------------

def test_regression_real_evos_trans_format(tmp_path):
    """Mirrors the real EVOS TRANS files (285 images at 1.5188 um/px)."""
    obj = json.dumps({"NumericalAperture": 0.13, "Magnification": 4.0})
    payload = json.dumps({
        "Gamma": "1", "Objective NA": "0.13", "ObjectiveSettings": obj,
        "MicronsPerPixel": "1.5188172690164", "ChannelSettings": json.dumps({"Name": "TRANS"}),
    })
    desc = '<?xml version="1.0" encoding="utf-16"?>\r\n<OME xmlns:xsd="http://www.w3.org/2001/XMLSchema"></OME>'
    p = tmp_path / "Day0 Mat gel1_0001_TRANS.tif"
    tifffile.imwrite(
        p, np.zeros((32, 48), np.uint8), description=desc.encode(), metadata=None,
        extratags=[(37510, "s", 0, "ASCIIEvos2 1.8.3076.262-c3dfdd: " + payload, True)],
    )
    info = ScaleExtractor().extract(p)
    assert (round(info.pixel_size_um, 9), info.source, info.is_calibrated) == (1.518817269, "evos_tag_37510", True)
    assert info.magnification == 4.0
    assert info.numerical_aperture == pytest.approx(0.13)


@pytest.mark.parametrize("px", [0.505049247, 1.136362553, 0.454545952])
def test_regression_real_if_resolution_tags_cm(tmp_path, px):
    """Mirrors the real IF files (TIFF resolution tags, unit CENTIMETER, no ImageDescription)."""
    p = tmp_path / "SKOV3 Spheroid D7 Mor_Mat-Gel1-2_ch00.tif"
    tifffile.imwrite(p, IMG, resolution=(1e4 / px, 1e4 / px), resolutionunit="CENTIMETER", metadata=None)
    info = ScaleExtractor().extract(p)
    assert info.source == "TIFF Resolution Tags"
    assert info.is_calibrated
    assert info.pixel_size_um == pytest.approx(px, rel=1e-6)


def test_regression_resolution_tags_inch_calibrated(tmp_path):
    p = tmp_path / "inch.tif"
    tifffile.imwrite(p, IMG, resolution=(25400 / 0.65, 25400 / 0.65), resolutionunit="INCH", metadata=None)
    info = ScaleExtractor().extract(p)
    assert info.source == "TIFF Resolution Tags" and info.is_calibrated
    assert info.pixel_size_um == pytest.approx(0.65, rel=1e-6)
