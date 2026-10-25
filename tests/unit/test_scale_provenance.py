"""
Unit tests for per-image optical scale provenance, calibration gating, and preflight inspection.
Validates WS2.3 (R4):
- objects.csv records scale_source and is_calibrated (non-null on every row)
- manifest.csv reflects per-image scale sources and contains real sha256 checksums (not 'skipped')
- uncalibrated images flag REVIEW by default unless --allow-uncalibrated is set
- preflight checks all files for scale tags, not just the first file
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile
from skimage import measure as skim_measure

from spheroid_pipeline_v2.config import PipelineConfig, QCConfig
from spheroid_pipeline_v2.measure import (
    OBJECTS_CSV_COLUMNS,
    SpheroidObjectRecord,
    measure_single_object,
    save_objects_csv,
)
from spheroid_pipeline_v2.preflight import PreflightAuditor
from spheroid_pipeline_v2.provenance import export_complete_manifest
from spheroid_pipeline_v2.run_pipeline import build_manifest, run_spheroid_pipeline
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor


def _create_evos_tiff(path: Path, data: np.ndarray, pixel_size_um: float = 1.518817):
    """Create a synthetic TIFF image with embedded EVOS Tag 37510 JSON metadata."""
    metadata = {
        "Target": {"PixelSize": float(pixel_size_um)},
        "Objective": "4x Plan Fluor",
        "Instrument": "EVOS M5000",
    }
    json_bytes = json.dumps(metadata).encode("utf-8")
    extratags = [(37510, "s", len(json_bytes), json_bytes, True)]
    tifffile.imwrite(str(path), data, extratags=extratags)


def _create_plain_tiff(path: Path, data: np.ndarray):
    """Create a synthetic TIFF without embedded scale calibration tags."""
    tifffile.imwrite(str(path), data)


def test_objects_csv_has_scale_source_and_calibrated(tmp_path):
    """
    Every object row in objects.csv must have non-null 'scale_source' and 'is_calibrated'.
    """
    assert "scale_source" in OBJECTS_CSV_COLUMNS
    assert "is_calibrated" in OBJECTS_CSV_COLUMNS

    # Create dummy mask and region property
    img = np.zeros((100, 100), dtype=np.uint8)
    lbl = np.zeros((100, 100), dtype=np.int32)
    lbl[40:60, 40:60] = 1
    props = skim_measure.regionprops(lbl)
    assert len(props) == 1

    rec = measure_single_object(
        prop=props[0],
        image_shape=(100, 100),
        pixel_size_um=1.518817,
        mean_interior_intensity=50.0,
        mean_ring_intensity=100.0,
        contrast_ratio=0.5,
        image_metadata={
            "image_id": "test_img_01",
            "scale_source": "evos_tag_37510",
            "is_calibrated": True,
        },
        config=QCConfig(),
    )

    csv_path = tmp_path / "objects.csv"
    df = save_objects_csv([rec], csv_path)

    assert "scale_source" in df.columns
    assert "is_calibrated" in df.columns
    assert df["scale_source"].iloc[0] == "evos_tag_37510"
    assert bool(df["is_calibrated"].iloc[0]) is True
    assert df["scale_source"].notnull().all()
    assert df["is_calibrated"].notnull().all()


def test_manifest_reflects_per_image_scale(tmp_path):
    """
    With two images in the dataset—one with EVOS Tag 37510 and one plain—
    manifest.csv must record different scale_source values reflecting their respective sources.
    """
    t0_dir = tmp_path / "Day0"
    t7_dir = tmp_path / "Day7"
    t0_dir.mkdir()
    t7_dir.mkdir()

    img1 = np.ones((50, 50), dtype=np.uint8) * 100
    img2 = np.ones((50, 50), dtype=np.uint8) * 120

    # Pair 1: Calibrated with Tag 37510
    _create_evos_tiff(t0_dir / "Day0 Mat gel1_0001_TRANS.tif", img1, pixel_size_um=1.234)
    _create_evos_tiff(t7_dir / "Day7 Mat gel1_0001_TRANS.tif", img1, pixel_size_um=1.234)

    # Pair 2: Plain uncalibrated
    _create_plain_tiff(t0_dir / "Day0 Mat gel1_0002_TRANS.tif", img2)
    _create_plain_tiff(t7_dir / "Day7 Mat gel1_0002_TRANS.tif", img2)

    manifest_df = build_manifest(t0_dir, t7_dir)
    assert not manifest_df.empty

    manifest_csv = export_complete_manifest(manifest_df, tmp_path / "out", compute_checksums=True)
    df_manifest = pd.read_csv(manifest_csv)

    sources = df_manifest["scale_source"].unique()
    assert len(sources) > 1, f"Expected multiple scale sources in manifest, got: {sources}"
    assert "evos_tag_37510" in sources
    assert any("uncalibrated" in s or "fallback" in s for s in sources)


def test_uncalibrated_image_flags_review_by_default():
    """
    An uncalibrated image must have its objects flagged 'REVIEW' with 'uncalibrated_optical_scale'
    by default. When allow_uncalibrated=True, this flag is suppressed.
    """
    lbl = np.zeros((100, 100), dtype=np.int32)
    lbl[20:60, 20:60] = 1
    props = skim_measure.regionprops(lbl)

    # 1. Default (allow_uncalibrated=False): uncalibrated image -> REVIEW
    rec_default = measure_single_object(
        prop=props[0],
        image_shape=(100, 100),
        pixel_size_um=1.518817,
        mean_interior_intensity=50.0,
        mean_ring_intensity=100.0,
        contrast_ratio=0.5,
        image_metadata={
            "image_id": "test_img_uncal",
            "scale_source": "config_default_uncalibrated",
            "is_calibrated": False,
            "allow_uncalibrated": False,
        },
        config=QCConfig(),
    )
    assert rec_default.qc_flag == "REVIEW"
    assert "uncalibrated_optical_scale" in rec_default.qc_reasons
    assert rec_default.is_calibrated is False

    # 2. Permissive (allow_uncalibrated=True): uncalibrated image allowed -> PASS (if other gates pass)
    rec_allowed = measure_single_object(
        prop=props[0],
        image_shape=(100, 100),
        pixel_size_um=1.518817,
        mean_interior_intensity=50.0,
        mean_ring_intensity=100.0,
        contrast_ratio=0.5,
        image_metadata={
            "image_id": "test_img_uncal",
            "scale_source": "config_default_uncalibrated",
            "is_calibrated": False,
            "allow_uncalibrated": True,
        },
        config=QCConfig(),
    )
    assert rec_allowed.qc_flag == "PASS"
    assert "uncalibrated_optical_scale" not in rec_allowed.qc_reasons


def test_manifest_contains_sha256(tmp_path):
    """
    Every row of manifest.csv must contain a valid 64-character SHA-256 hexadecimal hash,
    and must not be 'skipped'.
    """
    t0_dir = tmp_path / "Day0"
    t7_dir = tmp_path / "Day7"
    t0_dir.mkdir()
    t7_dir.mkdir()

    img = np.zeros((50, 50), dtype=np.uint8)
    _create_plain_tiff(t0_dir / "Day0 Mat gel1_0001_TRANS.tif", img)
    _create_plain_tiff(t7_dir / "Day7 Mat gel1_0001_TRANS.tif", img)

    manifest_df = build_manifest(t0_dir, t7_dir)
    manifest_csv = export_complete_manifest(manifest_df, tmp_path / "out", compute_checksums=True)

    df = pd.read_csv(manifest_csv)
    assert "sha256" in df.columns
    for sha in df["sha256"]:
        assert sha != "skipped", "SHA-256 hash was marked as skipped!"
        assert isinstance(sha, str)
        assert len(sha) == 64
        int(sha, 16)  # Must be valid hex


def test_preflight_checks_all_files_for_scale_tag(tmp_path):
    """
    Preflight must inspect all files across t0 and t7, not just t0_files[0].
    If file 1 has Tag 37510 but files 2 and 3 do not, preflight must identify
    the uncalibrated files and report them in warnings / stats.
    """
    t0_dir = tmp_path / "Day0"
    t7_dir = tmp_path / "Day7"
    t0_dir.mkdir()
    t7_dir.mkdir()

    img = np.ones((50, 50), dtype=np.uint8) * 100

    # t0_files[0] has tag 37510
    _create_evos_tiff(t0_dir / "Day0 Mat gel1_0001_TRANS.tif", img, 1.518817)
    # t0_files[1] lacks tag 37510
    _create_plain_tiff(t0_dir / "Day0 Mat gel1_0002_TRANS.tif", img)
    # t7_files[0] lacks tag 37510
    _create_plain_tiff(t7_dir / "Day7 Mat gel1_0001_TRANS.tif", img)

    config = PipelineConfig(
        t0_dir=str(t0_dir),
        t7_dir=str(t7_dir),
        output_dir=str(tmp_path / "out"),
    )

    auditor = PreflightAuditor(config)
    report = auditor.run_preflight_check()

    # Preflight must record accurate total counts and uncalibrated file counts
    assert "uncalibrated_files_count" in report.stats
    assert report.stats["uncalibrated_files_count"] == 2
    assert report.stats["calibrated_files_count"] == 1

    # Warnings must mention uncalibrated files
    warning_text = " ".join(report.warnings)
    assert "Day0 Mat gel1_0002_TRANS.tif" in warning_text or "2 uncalibrated" in warning_text
