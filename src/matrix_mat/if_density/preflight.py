"""
Preflight Validation for Spheroid Immunofluorescence (IF) Pipeline.

Validates input channel pairing (nuclear ch00 vs actin ch02),
model weights presence, image bit depth / dimensionality / channel shape agreement, and pixel calibration.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile

logger = logging.getLogger("matrix_mat.if_density.preflight")


@dataclasses.dataclass
class IFPreflightReport:
    """Structured report produced by IF preflight validation."""
    passed: bool
    fatal_errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    stats: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def log_and_raise_if_fatal(self) -> None:
        """Logs warnings and raises RuntimeError if preflight failed."""
        for w in self.warnings:
            logger.warning(f"[IF PREFLIGHT WARNING] {w}")
        if not self.passed:
            err_msg = (
                "\n" + "=" * 80 + "\n"
                "  IF PIPELINE PREFLIGHT FAILED — HALTED TO PREVENT INVALID RUN\n"
                + "=" * 80 + "\n"
                + "\n".join(f"  * FATAL: {err}" for err in self.fatal_errors)
                + "\n" + "=" * 80
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    def format_summary(self) -> str:
        """Returns formatted summary string."""
        lines = [
            "=" * 70,
            "         IF SWEEP PIPELINE — PREFLIGHT INTEGRITY AUDIT",
            "=" * 70,
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Total TIFF Files Found: {self.stats.get('total_files', 0)}",
            f"Paired Fields (ch00 + ch02): {self.stats.get('paired_fields', 0)}",
            f"Unpaired Fields:        {self.stats.get('unpaired_fields', 0)}",
            f"Materials Detected:     {', '.join(sorted(self.stats.get('materials', [])))}",
            f"Model Path:             {self.stats.get('model_path', 'N/A')} (Found: {self.stats.get('model_exists', False)})",
            f"Hardware Accelerator:   {self.stats.get('hardware', 'CPU')}",
        ]
        if self.warnings:
            lines.append("-" * 70)
            lines.append("Active Warnings:")
            for w in self.warnings:
                lines.append(f"  [!] {w}")
        if self.fatal_errors:
            lines.append("-" * 70)
            lines.append("Fatal Errors:")
            for err in self.fatal_errors:
                lines.append(f"  [X] {err}")
        lines.append("=" * 70)
        return "\n".join(lines)


FETCH_MODELS_HINT = "python tools/fetch_models.py"


def check_model_weights(model_path: Union[str, Path]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Check that the Cellpose-SAM weights file exists (fatal otherwise: the engine refuses to download)
    and that its size matches models.lock.json (warning; the full SHA-256 is verified at model load).
    Returns (fatal_errors, warnings, stats).
    """
    fatal: List[str] = []
    warns: List[str] = []
    m_path = Path(model_path)
    stats: Dict[str, Any] = {"model_path": str(m_path), "model_exists": m_path.is_file()}
    if not m_path.is_file():
        fatal.append(
            f"Cellpose-SAM weights not found at '{m_path.resolve()}'. The pipeline does not download weights "
            f"at run time; fetch and verify the pinned weights with: {FETCH_MODELS_HINT}"
        )
        return fatal, warns, stats
    try:
        from matrix_mat.core.repro import get_model_weights_info
        expected_size = get_model_weights_info("cpsam_v2").get("size_bytes")
    except Exception:
        expected_size = None
    actual_size = m_path.stat().st_size
    stats["model_size_bytes"] = actual_size
    if expected_size and int(expected_size) != actual_size:
        warns.append(
            f"Weights at '{m_path}' are {actual_size} bytes but models.lock.json pins {expected_size}; "
            f"the SHA-256 check at model load will fail. Re-fetch with: {FETCH_MODELS_HINT}"
        )
    return fatal, warns, stats


def audit_if_dataset(
    data_dir: Union[str, Path],
    model_path: Union[str, Path] = "models/cpsam_v2",
    allow_unpaired: bool = True,
    allow_uncalibrated: Optional[bool] = None,
    check_weights: bool = True,
) -> IFPreflightReport:
    """
    Audits IF dataset, channel pairings, image dtype/dimensionality, pixel calibration and model weights.

    Uses the same file discovery as the sweep (pairing.discover_channel_files). Fatal: missing data dir,
    no pairs, unpaired fields (unless allowed), .tif/.tiff collisions, z-stacks, ch00/ch02 shape mismatch,
    missing weights. allow_uncalibrated=False makes uncalibrated fields fatal (None: warning only).
    """
    from matrix_mat.if_density.pairing import (
        classify_shape,
        discover_channel_files,
        parse_field_metadata,
        spatial_shape,
    )

    data_path = Path(data_dir)
    fatal_errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, Any] = {}

    if not data_path.exists():
        fatal_errors.append(
            f"Data directory does not exist: '{data_path.resolve()}'. "
            "Verify path in YAML config under 'data_dir'."
        )
        return IFPreflightReport(passed=False, fatal_errors=fatal_errors, warnings=warnings, stats=stats)

    disc = discover_channel_files(data_path)
    stats["total_files"] = len(disc.files)

    if not disc.files:
        fatal_errors.append(
            f"No .tif/.tiff files found in '{data_path.resolve()}'."
        )
        return IFPreflightReport(passed=False, fatal_errors=fatal_errors, warnings=warnings, stats=stats)

    for c in disc.collisions:
        fatal_errors.append(f"Ambiguous channel files (same field and channel twice): {c}")
    if disc.unrecognized:
        warnings.append(
            f"{len(disc.unrecognized)} TIFF file(s) without an exact _chNN suffix are ignored: "
            + ", ".join(p.name for p in disc.unrecognized[:10])
        )

    # Channel pairing audit
    ch00_files = disc.nuclear
    ch02_files = disc.actin
    paired = set(ch00_files.keys()) & set(ch02_files.keys())
    unpaired_00 = set(ch00_files.keys()) - set(ch02_files.keys())
    unpaired_02 = set(ch02_files.keys()) - set(ch00_files.keys())
    total_unpaired = len(unpaired_00) + len(unpaired_02)

    stats["paired_fields"] = len(paired)
    stats["unpaired_fields"] = total_unpaired

    materials = set()
    for base in sorted(paired):
        try:
            materials.add(parse_field_metadata(base)[0])
        except ValueError:
            warnings.append(f"Unrecognised IF filename (no material/gel), will be skipped: {base}")
    stats["materials"] = sorted(materials)

    if len(paired) == 0:
        fatal_errors.append(
            "Found 0 paired fields with both nuclear channel (_ch00) and actin channel (_ch02). "
            "IF analysis requires both channels for nuclear density and spheroid extent quantification."
        )

    if total_unpaired > 0:
        unpaired_list = sorted(unpaired_00 | unpaired_02)
        if not allow_unpaired:
            fatal_errors.append(
                f"Found {total_unpaired} unpaired fields: {', '.join(unpaired_list)}. "
                "Set allow_unpaired=True in config or pass --allow-unpaired to skip them and proceed."
            )
        else:
            warnings.append(
                f"Skipping {total_unpaired} unpaired field(s) without complete ch00/ch02 pairs: {', '.join(unpaired_list)}."
            )

    # Image dtype / dimensionality / shape-match / calibration audit (header reads only).
    from matrix_mat.core.scale import ScaleExtractor
    scale_extractor = ScaleExtractor()
    dtypes: Dict[str, int] = {}
    uncalibrated: List[str] = []
    for base in sorted(paired):
        shapes = {}
        for ch, path in (("ch00", ch00_files[base]), ("ch02", ch02_files[base])):
            try:
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    shape, dtype = tuple(series.shape), np.dtype(series.dtype)
            except Exception as e:
                fatal_errors.append(f"{path.name}: unreadable TIFF ({e})")
                continue
            shapes[ch] = shape
            dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1
            kind = classify_shape(shape)
            if kind == "3D":
                fatal_errors.append(f"{path.name}: multi-plane stack {shape}; the IF pipeline supports 2D images only")
            elif kind == "RGB":
                warnings.append(f"{path.name}: RGB(A) image {shape} will be converted to grayscale")
            if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
                fatal_errors.append(f"{path.name}: unsupported pixel type {dtype}")
            elif dtype != np.uint8:
                warnings.append(f"{path.name}: {dtype} data will be scaled to 8 bit by bit depth (never wrapped)")
        if len(shapes) == 2 and spatial_shape(shapes["ch00"]) != spatial_shape(shapes["ch02"]):
            fatal_errors.append(f"{base}: ch00 shape {shapes['ch00']} does not match ch02 shape {shapes['ch02']}")
        try:
            if not scale_extractor.extract(ch00_files[base]).is_calibrated:
                uncalibrated.append(base)
        except Exception as e:
            warnings.append(f"{base}: could not read pixel scale ({e})")
    stats["dtypes"] = dtypes
    stats["uncalibrated_fields"] = len(uncalibrated)
    if uncalibrated:
        msg = f"{len(uncalibrated)} field(s) have no embedded pixel scale: {', '.join(uncalibrated)}"
        if allow_uncalibrated is False:
            fatal_errors.append(msg + ". Set allow_uncalibrated: true to use default_pixel_size_um.")
        else:
            warnings.append(msg + " (the sweep refuses them unless allow_uncalibrated: true).")

    # Model file audit
    if check_weights:
        w_fatal, w_warn, w_stats = check_model_weights(model_path)
        fatal_errors.extend(w_fatal)
        warnings.extend(w_warn)
        stats.update(w_stats)
    else:
        stats["model_path"] = str(model_path)
        stats["model_exists"] = Path(model_path).is_file()

    # Hardware detection (avoids heavy torch import during fast preflight audits unless torch is already loaded)
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            if torch.cuda.is_available():
                stats["hardware"] = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                stats["hardware"] = "Apple Silicon MPS"
            else:
                stats["hardware"] = "CPU"
                warnings.append("No GPU accelerator detected. Inference will run on CPU, which may be slow.")
        except Exception:
            stats["hardware"] = "CPU"
            warnings.append("No GPU accelerator detected. Inference will run on CPU, which may be slow.")
    else:
        stats["hardware"] = "CPU (torch deferred)"

    passed = len(fatal_errors) == 0
    return IFPreflightReport(
        passed=passed,
        fatal_errors=fatal_errors,
        warnings=warnings,
        stats=stats,
    )


def run_if_preflight(
    data_dir: Optional[Union[str, Path]] = None,
    model_path: Union[str, Path] = "models/cpsam_v2",
    allow_unpaired: bool = True,
    allow_uncalibrated: Optional[bool] = None,
) -> IFPreflightReport:
    """Executes immunofluorescence preflight audit."""
    if data_dir is None:
        from matrix_mat.core.paths import resolve_data_root
        data_dir = resolve_data_root() / "if"
    return audit_if_dataset(
        data_dir=data_dir, model_path=model_path, allow_unpaired=allow_unpaired,
        allow_uncalibrated=allow_uncalibrated,
    )


__all__ = [
    "IFPreflightReport",
    "check_model_weights",
    "audit_if_dataset",
    "run_if_preflight",
]
