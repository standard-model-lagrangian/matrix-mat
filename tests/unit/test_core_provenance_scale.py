"""manifest.csv records the scale the run actually used (0.2.2)."""

import numpy as np
import pandas as pd
import tifffile

from matrix_mat.core.provenance import export_complete_manifest
from matrix_mat.core.scale import ScaleExtractor


def _manifest(tmp_path, **cols):
    p = tmp_path / "Day0 Mat gel1_0001_TRANS.tif"
    tifffile.imwrite(p, np.zeros((8, 8), np.uint8), metadata=None)
    row = {"file_path": str(p), "image_id": p.stem}
    row.update(cols)
    return pd.DataFrame([row])


def test_override_is_recorded(tmp_path):
    df = _manifest(tmp_path)
    out = export_complete_manifest(df, tmp_path / "o", compute_checksums=False, pixel_size_override_um=0.9)
    m = pd.read_csv(out)
    assert m["pixel_size_um"].iloc[0] == 0.9
    assert m["scale_source"].iloc[0] == "cli_override"
    assert bool(m["is_calibrated"].iloc[0]) is True


def test_config_default_is_recorded(tmp_path):
    df = _manifest(tmp_path)
    out = export_complete_manifest(df, tmp_path / "o", compute_checksums=False, default_pixel_size_um=2.5)
    assert pd.read_csv(out)["pixel_size_um"].iloc[0] == 2.5


def test_scale_extractor_instance_is_used(tmp_path):
    df = _manifest(tmp_path)
    out = export_complete_manifest(df, tmp_path / "o", compute_checksums=False,
                                   scale_extractor=ScaleExtractor(default_pixel_size_um=3.0))
    assert pd.read_csv(out)["pixel_size_um"].iloc[0] == 3.0


def test_nan_columns_are_treated_as_missing(tmp_path):
    df = _manifest(tmp_path, pixel_size_um=np.nan, scale_source=np.nan, is_calibrated=np.nan)
    out = export_complete_manifest(df, tmp_path / "o", compute_checksums=False, default_pixel_size_um=2.5)
    m = pd.read_csv(out)
    assert bool(m["is_calibrated"].iloc[0]) is False  # bool(nan) would have been True
    assert m["pixel_size_um"].iloc[0] == 2.5
    assert "uncalibrated" in m["scale_source"].iloc[0]


def test_nan_is_calibrated_with_known_source_is_false(tmp_path):
    df = _manifest(tmp_path, pixel_size_um=1.0, scale_source="x", is_calibrated=np.nan)
    m = pd.read_csv(export_complete_manifest(df, tmp_path / "o", compute_checksums=False))
    assert bool(m["is_calibrated"].iloc[0]) is False


def test_if_manifest_has_scale_columns(tmp_path):
    from types import SimpleNamespace

    from matrix_mat.core.provenance import write_manifest_csv

    recs = [SimpleNamespace(field_id="f1", pixel_size_um=0.5, scale_source="TIFF Resolution Tags", is_calibrated=True),
            SimpleNamespace(field_id="f2", pixel_size_um=1.0, scale_source=None, is_calibrated=np.nan)]
    m = pd.read_csv(write_manifest_csv(tmp_path, recs))
    assert list(m["scale_source"]) == ["TIFF Resolution Tags", "unknown"]
    assert list(m["is_calibrated"]) == [True, False]
