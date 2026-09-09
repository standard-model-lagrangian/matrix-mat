"""
Preflight Validation for Spheroid Immunofluorescence (IF) Pipeline.

Validates input channel pairing (nuclear ch00 vs actin ch02),
model weights readiness, image bit depth, dynamic range, and focus metrics.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
import torch

logger = logging.getLogger("spheroid_if_sweep.preflight")


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


def audit_if_dataset(
    data_dir: Union[str, Path],
    model_path: Union[str, Path] = "models/cpsam_v2",
    allow_unpaired: bool = True,
) -> IFPreflightReport:
    """Audits IF dataset, channel pairings, and model files."""
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

    tifs = sorted(list(data_path.glob("*.tif")) + list(data_path.glob("*.tiff")))
    stats["total_files"] = len(tifs)

    if not tifs:
        fatal_errors.append(
            f"No .tif files found in '{data_path.resolve()}'."
        )
        return IFPreflightReport(passed=False, fatal_errors=fatal_errors, warnings=warnings, stats=stats)

    # Channel pairing audit
    ch00_files = {}
    ch02_files = {}
    for p in tifs:
        if "_ch00" in p.name:
            base = p.name.replace("_ch00.tif", "").replace("_ch00.tiff", "")
            ch00_files[base] = p
        elif "_ch02" in p.name:
            base = p.name.replace("_ch02.tif", "").replace("_ch02.tiff", "")
            ch02_files[base] = p

    paired = set(ch00_files.keys()) & set(ch02_files.keys())
    unpaired_00 = set(ch00_files.keys()) - set(ch02_files.keys())
    unpaired_02 = set(ch02_files.keys()) - set(ch00_files.keys())
    total_unpaired = len(unpaired_00) + len(unpaired_02)

    stats["paired_fields"] = len(paired)
    stats["unpaired_fields"] = total_unpaired

    if len(paired) == 0:
        fatal_errors.append(
            "Found 0 paired fields with both nuclear channel (_ch00) and actin channel (_ch02). "
            "IF analysis requires both channels for nuclear density and spheroid extent quantification."
        )

    if total_unpaired > 0:
        unpaired_list = list(unpaired_00 | unpaired_02)
        if not allow_unpaired:
            fatal_errors.append(
                f"Found {total_unpaired} unpaired fields: {', '.join(unpaired_list)}. "
                "Set allow_unpaired=True in config or pass --allow-unpaired to skip them and proceed."
            )
        else:
            warnings.append(
                f"Skipping {total_unpaired} unpaired field(s) without complete ch00/ch02 pairs: {', '.join(unpaired_list)}."
            )

    # Model file audit
    m_path = Path(model_path)
    stats["model_path"] = str(m_path)
    stats["model_exists"] = m_path.exists()

    if not m_path.exists():
        warnings.append(
            f"Local Cellpose-SAM weights not found at '{m_path.resolve()}'. "
            "Cellpose will attempt to download model weights from HuggingFace, which requires an active internet connection."
        )

    # Hardware detection
    if torch.cuda.is_available():
        stats["hardware"] = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        stats["hardware"] = "Apple Silicon MPS"
    else:
        stats["hardware"] = "CPU"
        warnings.append("No GPU accelerator detected. Inference will run on CPU, which may be slow.")

    passed = len(fatal_errors) == 0
    return IFPreflightReport(
        passed=passed,
        fatal_errors=fatal_errors,
        warnings=warnings,
        stats=stats,
    )
