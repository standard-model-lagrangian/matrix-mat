"""
Provenance & Run Metadata Tracking for Spheroid Pipeline v2.

Records environment specifications, git commit hashes, dirty status,
package versions, random seeds, hardware accelerators, warnings,
and full image manifests with SHA-256 checksums.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tifffile
import torch

from spheroid_pipeline_v2.config import PipelineConfig

logger = logging.getLogger("spheroid_pipeline_v2.provenance")


def compute_file_sha256(filepath: Union[str, Path], block_size: int = 65536) -> str:
    """Computes SHA-256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info() -> Dict[str, Any]:
    """Retrieves current git commit hash, branch, and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        is_dirty = len(status) > 0
        return {
            "commit": commit,
            "branch": branch,
            "is_dirty": is_dirty,
            "status_summary": "clean" if not is_dirty else "dirty",
        }
    except Exception as e:
        logger.debug(f"Git execution not available: {e}")
        return {
            "commit": "unknown",
            "branch": "unknown",
            "is_dirty": False,
            "status_summary": "no_git_or_sandbox_isolated",
        }


def get_gpu_info() -> str:
    """Detects available hardware accelerator."""
    if torch.cuda.is_available():
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return f"Apple Silicon MPS ({platform.processor()} / {platform.machine()})"
    return "CPU"


def get_package_version(package_name: str) -> str:
    """Safely retrieves installed package version."""
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def write_run_metadata(
    output_dir: Union[str, Path],
    config: PipelineConfig,
    wall_time_sec: float,
    warnings: Optional[List[str]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes comprehensive run_metadata.json into the output directory.
    Captures complete reproducibility metadata: environment, hardware,
    git commit, package versions, configuration snapshot, and logged warnings.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "pipeline_name": "spheroid_pipeline_v2",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wall_time_seconds": round(wall_time_sec, 3),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "hardware_accelerator": get_gpu_info(),
            "git": get_git_info(),
        },
        "dependency_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tifffile": get_package_version("tifffile"),
            "cellpose": get_package_version("cellpose"),
            "scipy": get_package_version("scipy"),
            "skimage": get_package_version("scikit-image"),
            "opencv": get_package_version("opencv-python-headless"),
            "matplotlib": get_package_version("matplotlib"),
        },
        "random_seed": config.random_seed,
        "resolved_parameters": {
            "t0_dir": str(config.t0_dir),
            "t7_dir": str(config.t7_dir),
            "output_dir": str(config.output_dir),
            "review_dir": str(config.review_dir),
            "default_pixel_size_um": getattr(config, "default_pixel_size_um", 1.518817),
            "cli_pixel_size_um": getattr(config, "cli_pixel_size_um", None),
            "segmentation_backend": getattr(config.segmentation, "primary_backend", getattr(config.segmentation, "backend", "cpsam")),
            "qc_min_d_um": getattr(config.qc, "min_d_um", 40.0),
            "qc_max_d_um": getattr(config.qc, "max_d_um", 1500.0),
            "qc_border_margin_px": getattr(config.qc, "border_margin_px", 15),
            "qc_ring_contrast_factor": getattr(config.qc, "ring_contrast_factor", getattr(config.qc, "interior_contrast_pct", 0.90)),
            "stats_control_condition": getattr(config.stats, "control_condition", "Mat"),
            "stats_min_denominator_d0_um": getattr(config.stats, "min_denominator_d0_um", 60.0),
            "stats_bootstrap_iterations": getattr(config.stats, "bootstrap_iterations", 2000),
            "stats_permutation_iterations": getattr(config.stats, "permutation_iterations", 10000),
        },
        "warnings_logged": warnings or [],
        "warnings_count": len(warnings or []),
    }

    if extra_info:
        metadata["execution_stats"] = extra_info

    out_file = out_dir / "run_metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Recorded run metadata to {out_file}")
    return out_file


def export_complete_manifest(
    manifest_df: pd.DataFrame,
    output_dir: Union[str, Path],
    compute_checksums: bool = True,
) -> Path:
    """
    Exports a comprehensive manifest.csv documenting every ingested image,
    file size, SHA-256 hash, dimensions, optical pixel size, and pair status.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in manifest_df.iterrows():
        p = Path(r["file_path"])
        f_size = p.stat().st_size if p.exists() else 0
        sha = compute_file_sha256(p) if (compute_checksums and p.exists()) else "skipped"

        # Image shape and bit depth inspection
        shape_str = "unknown"
        dtype_str = "unknown"
        if p.exists():
            try:
                with tifffile.TiffFile(str(p)) as tif:
                    if len(tif.pages) > 0:
                        page = tif.pages[0]
                        shape_str = f"{page.shape[0]}x{page.shape[1]}"
                        dtype_str = str(page.dtype)
            except Exception:
                pass

        rows.append({
            "image_id": r.get("image_id", p.stem),
            "timepoint": r.get("timepoint", ""),
            "condition": r.get("condition", ""),
            "replicate": r.get("replicate", ""),
            "fov": r.get("fov", ""),
            "pair_key": r.get("pair_key", ""),
            "file_path": str(p),
            "file_size_bytes": f_size,
            "sha256": sha,
            "dimensions": shape_str,
            "bit_depth": dtype_str,
            "pixel_size_um": r.get("pixel_size_um", 1.518817),
            "scale_source": r.get("scale_source", "default_fallback"),
            "is_matched_pair": r.get("is_matched_pair", False),
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "manifest.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Exported complete data manifest ({len(df)} images) to {out_csv}")
    return out_csv
