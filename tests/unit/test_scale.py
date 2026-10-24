import numpy as np
import pytest
from pathlib import Path
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor, ScaleInfo


def test_72dpi_resolution_tag_not_trusted():
    """
    72 DPI resolution tag (standard uncalibrated screen default) must NOT be trusted
    as a calibrated microscope scale. Must return None or is_calibrated=False.
    """
    extractor = ScaleExtractor()
    # Mock tags with 72 DPI (unit 2 = inches, res = 72)
    tags = {282: (72, 1), 296: 2}
    info_dict = {"resolution_unit": 2, "dpi": (72, 72)}

    scale_info = extractor._extract_tiff_resolution(tags, info_dict)
    assert scale_info is None or scale_info.is_calibrated is False


def test_imagej_spacing_is_not_xy():
    """
    ImageJ metadata tag 'spacing' specifies Z-step size, NOT XY pixel size.
    If spacing=2.0 and XY resolution is 0.65 um/px, pixel_size_um must be 0.65, not 2.0.
    """
    extractor = ScaleExtractor()
    tags = {270: "ImageJ=1.53c\nunit=um\nspacing=2.0\npixelWidth=0.65\npixelHeight=0.65"}
    info_dict = {"description": tags[270]}

    scale_info = extractor._extract_imagej(tags, info_dict)
    assert scale_info is not None
    assert np.isclose(scale_info.pixel_size_um, 0.65), (
        f"Expected XY scale 0.65 um/px, but got {scale_info.pixel_size_um} (spacing=2.0 was improperly used)"
    )


def test_scale_plausibility_range():
    """
    Microscope pixel sizes outside the plausible range [0.05, 20.0] um/px must be rejected.
    """
    extractor = ScaleExtractor()

    # Case 1: Wildly small pixel size (0.001 um/px)
    tags_tiny = {282: (25400000, 1), 296: 2}
    info_tiny = {"resolution_unit": 2, "dpi": (25400000, 1)}
    res_tiny = extractor._extract_tiff_resolution(tags_tiny, info_tiny)
    assert res_tiny is None or res_tiny.is_calibrated is False

    # Case 2: Wildly large pixel size (350 um/px)
    tags_large = {282: (72, 1), 296: 2}
    info_large = {"resolution_unit": 2, "dpi": (72, 1)}
    res_large = extractor._extract_tiff_resolution(tags_large, info_large)
    assert res_large is None or res_large.is_calibrated is False

    # Case 3: Valid plausible scale (1.5 um/px)
    # 25400 / 1.5 ≈ 16933.33 dpi
    res_1p5 = 25400.0 / 1.5
    tags_valid = {282: (int(res_1p5), 1), 296: 2}
    info_valid = {"resolution_unit": 2, "dpi": (int(res_1p5), 1)}
    res_valid = extractor._extract_tiff_resolution(tags_valid, info_valid)
    assert res_valid is not None
    assert res_valid.is_calibrated is True
    assert np.isclose(res_valid.pixel_size_um, 1.5, atol=0.01)
