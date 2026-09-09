"""
Preflight Validation & Data Integrity Auditor for Spheroid Pipeline v2.

Performs proactive validation of input directories, TIFF file integrity,
optical metadata calibration, pairing feasibility, baseline control presence,
and image contrast before execution. Emits actionable, specific diagnostics
to prevent unreproducible or invalid runs.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile

from spheroid_pipeline_v2.config import PipelineConfig

logger = logging.getLogger("spheroid_pipeline_v2.preflight")


@dataclasses.dataclass
class PreflightReport:
    """Structured report produced by preflight validation."""
    passed: bool
    fatal_errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    stats: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def log_and_raise_if_fatal(self) -> None:
        """Logs all warnings and raises RuntimeError if any fatal error occurred."""
        for w in self.warnings:
            logger.warning(f"[PREFLIGHT WARNING] {w}")
        if not self.passed:
            err_msg = (
                "\n" + "=" * 80 + "\n"
                "  PREFLIGHT AUDIT FAILED — PIPELINE HALTED TO PREVENT INVALID RUN\n"
                + "=" * 80 + "\n"
                + "\n".join(f"  * FATAL: {err}" for err in self.fatal_errors)
                + "\n" + "=" * 80
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

    def format_summary(self) -> str:
        """Returns human-readable formatted summary string."""
        lines = [
            "=" * 70,
            "         SPHEROID PIPELINE v2 — PREFLIGHT INTEGRITY AUDIT",
            "=" * 70,
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            f"Total t0 Images Found: {self.stats.get('t0_count', 0)}",
            f"Total t7 Images Found: {self.stats.get('t7_count', 0)}",
            f"Matched Pair FOVs:     {self.stats.get('matched_pairs_count', 0)}",
            f"Singleton FOVs (t0):   {self.stats.get('singleton_t0_count', 0)}",
            f"Singleton FOVs (t7):   {self.stats.get('singleton_t7_count', 0)}",
            f"Conditions Detected:   {', '.join(sorted(self.stats.get('conditions', [])))}",
            f"Control Condition:     '{self.stats.get('control_condition', 'Mat')}' (Found: {self.stats.get('control_found', False)})",
            f"Optical Scale Source:  {self.stats.get('scale_source_summary', 'Unknown')}",
            f"Hardware Accelerator:  {self.stats.get('hardware', 'CPU')}",
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


class PreflightAuditor:
    """Audits inputs, optical scales, and environment before pipeline execution."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def run_preflight_check(self) -> PreflightReport:
        """Executes full suite of preflight sanity checks."""
        fatal_errors: List[str] = []
        warnings: List[str] = []
        stats: Dict[str, Any] = {}

        t0_dir = Path(self.config.t0_dir)
        t7_dir = Path(self.config.t7_dir)

        # 1. Directory Existence and Non-Emptiness
        t0_files: List[Path] = []
        if not t0_dir.exists():
            fatal_errors.append(
                f"Day 0 directory does not exist: '{t0_dir.resolve()}'. "
                "Verify path in config.yaml under 't0_dir'."
            )
        else:
            t0_files = sorted(list(t0_dir.glob("*.tif")) + list(t0_dir.glob("*.tiff")))
            if not t0_files:
                fatal_errors.append(
                    f"Day 0 directory '{t0_dir.resolve()}' contains zero .tif/.tiff images."
                )

        t7_files: List[Path] = []
        if not t7_dir.exists():
            fatal_errors.append(
                f"Day 7 directory does not exist: '{t7_dir.resolve()}'. "
                "Verify path in config.yaml under 't7_dir'."
            )
        else:
            t7_files = sorted(list(t7_dir.glob("*.tif")) + list(t7_dir.glob("*.tiff")))
            if not t7_files:
                fatal_errors.append(
                    f"Day 7 directory '{t7_dir.resolve()}' contains zero .tif/.tiff images."
                )

        stats["t0_count"] = len(t0_files)
        stats["t7_count"] = len(t7_files)

        if fatal_errors:
            return PreflightReport(passed=False, fatal_errors=fatal_errors, warnings=warnings, stats=stats)

        # 2. File Readability & Integrity Spot Check
        corrupted_files: List[str] = []
        blank_images: List[str] = []
        sample_check_files = t0_files[:5] + t7_files[:5]

        for p in sample_check_files:
            try:
                img = tifffile.imread(str(p))
                if img is None or img.size == 0:
                    corrupted_files.append(p.name)
                    continue
                # Contrast & saturation check
                std_dev = float(np.std(img))
                if std_dev < 1e-3:
                    blank_images.append(f"{p.name} (zero variance / blank)")
                elif np.mean(img == np.max(img)) > 0.95:
                    blank_images.append(f"{p.name} (>95% saturated)")
            except Exception as e:
                corrupted_files.append(f"{p.name} ({e})")

        if corrupted_files:
            fatal_errors.append(
                f"Detected unreadable or corrupted TIFF files: {', '.join(corrupted_files)}"
            )

        if blank_images:
            warnings.append(
                f"Degraded image contrast detected in sample: {', '.join(blank_images)}. "
                "These FOVs may fail segmentation."
            )

        # 3. Pair Mapping & Condition Inventory
        import re
        stem_pattern = re.compile(r"Day([07])\s+([A-Za-z0-9]+)\s+(gel\d+)_(\d+)_TRANS")
        t0_keys: Dict[str, Path] = {}
        t7_keys: Dict[str, Path] = {}
        conditions_found = set()

        for p in t0_files:
            m = stem_pattern.search(p.name)
            if m:
                cond, rep, fov = m.group(2), m.group(3), m.group(4)
                t0_keys[f"{cond}_{rep}_{fov}"] = p
                conditions_found.add(cond)
            else:
                t0_keys[p.stem] = p

        for p in t7_files:
            m = stem_pattern.search(p.name)
            if m:
                cond, rep, fov = m.group(2), m.group(3), m.group(4)
                t7_keys[f"{cond}_{rep}_{fov}"] = p
                conditions_found.add(cond)
            else:
                t7_keys[p.stem] = p

        matched_keys = set(t0_keys.keys()) & set(t7_keys.keys())
        singleton_t0 = set(t0_keys.keys()) - set(t7_keys.keys())
        singleton_t7 = set(t7_keys.keys()) - set(t0_keys.keys())

        stats["matched_pairs_count"] = len(matched_keys)
        stats["singleton_t0_count"] = len(singleton_t0)
        stats["singleton_t7_count"] = len(singleton_t7)
        stats["conditions"] = list(conditions_found)

        if len(matched_keys) == 0:
            fatal_errors.append(
                "Zero matched FOV pairs found between Day 0 and Day 7 directories. "
                "Pairing naming pattern requires 'Day0 <Cond> gel<N>_<FOV>_TRANS.tif' "
                "and 'Day7 <Cond> gel<N>_<FOV>_TRANS.tif'."
            )

        if singleton_t0 or singleton_t7:
            warnings.append(
                f"Found {len(singleton_t0)} unpaired Day 0 and {len(singleton_t7)} unpaired Day 7 images. "
                "These will be included in cross-sectional population stats but excluded from paired tracking."
            )

        # 4. Control Condition Check
        control_cond = self.config.stats.control_condition
        stats["control_condition"] = control_cond
        stats["control_found"] = control_cond in conditions_found

        if control_cond not in conditions_found:
            warnings.append(
                f"Control condition '{control_cond}' not found in dataset conditions {sorted(conditions_found)}. "
                "Permutation hypothesis testing will be skipped or require re-configuring stats.control_condition."
            )

        # 5. Optical Calibration Inspection
        sample_img_path = t0_files[0]
        has_tag_37510 = False
        try:
            with tifffile.TiffFile(str(sample_img_path)) as tif:
                for page in tif.pages:
                    for tag in page.tags:
                        if tag.code == 37510:
                            has_tag_37510 = True
                            break
        except Exception:
            pass

        if self.config.cli_pixel_size_um is not None:
            stats["scale_source_summary"] = f"CLI Override ({self.config.cli_pixel_size_um:.6f} um/px)"
        elif has_tag_37510:
            stats["scale_source_summary"] = "TIFF Tag 37510 (EVOS Calibration Metadata)"
        else:
            stats["scale_source_summary"] = f"Uncalibrated Fallback ({self.config.default_pixel_size_um:.6f} um/px)"
            warnings.append(
                f"TIFF optical scale tag (37510) not detected in sample '{sample_img_path.name}'. "
                f"Pipeline will assume default {self.config.default_pixel_size_um:.6f} um/pixel. "
                "To ensure physical accuracy, verify calibration or override with --pixel-size <val>."
            )

        # 6. Hardware Acceleration
        import torch
        if torch.cuda.is_available():
            stats["hardware"] = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            stats["hardware"] = "Apple Silicon MPS"
        else:
            stats["hardware"] = "CPU"

        passed = len(fatal_errors) == 0
        return PreflightReport(
            passed=passed,
            fatal_errors=fatal_errors,
            warnings=warnings,
            stats=stats,
        )
