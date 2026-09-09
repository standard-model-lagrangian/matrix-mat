"""
Provenance tracking: gather environment metadata, git commit/diff, write run_metadata.json and manifest.csv.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import logging
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from spheroid_if_sweep.pairing import FieldRecord

logger = logging.getLogger("spheroid_if_sweep.provenance")


def get_git_info() -> Dict[str, str]:
    """Retrieve git commit hash or dirty state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        is_dirty = len(status) > 0
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "status_summary": "clean" if not is_dirty else "dirty",
        }
    except Exception as e:
        logger.debug(f"Git execution not available: {e}")
        return {
            "commit": "unknown",
            "is_dirty": False,
            "status_summary": "no_git_or_sandbox_isolated",
        }


def get_gpu_info() -> str:
    """Determine available GPU/MPS device model."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return f"Apple Silicon MPS ({platform.processor()} / {platform.machine()})"
    return "CPU"


def get_cellpose_version() -> str:
    """Retrieve installed cellpose version."""
    try:
        return importlib.metadata.version("cellpose")
    except Exception:
        import cellpose
        return getattr(cellpose, "__version__", "unknown")


def write_run_metadata(
    run_dir: Path,
    resolved_params: Dict[str, Any],
    random_seed: int,
    wall_time_sec: float,
    extra_info: Dict[str, Any] | None = None,
    preprocessing_info: Dict[str, Any] | None = None,
    postprocessing_info: Dict[str, Any] | None = None,
) -> Path:
    """Write run_metadata.json into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    git_info = get_git_info()

    metadata = {
        "timestamp_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cellpose_version": get_cellpose_version(),
        "model_name": "cpsam",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "gpu_model": get_gpu_info(),
        "git": git_info,
        "random_seed": random_seed,
        "wall_time_seconds": round(wall_time_sec, 3),
        "resolved_parameters": resolved_params,
    }
    if preprocessing_info:
        metadata["preprocessing"] = preprocessing_info
    if postprocessing_info:
        metadata["postprocessing"] = postprocessing_info
    if extra_info:
        metadata["extra"] = extra_info

    out_path = run_dir / "run_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=False)

    logger.info(f"Wrote run metadata to {out_path}")
    return out_path


def write_manifest_csv(
    run_dir: Path,
    field_records: List[FieldRecord],
) -> Path:
    """Write manifest.csv containing file paths, checksums, and metadata."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in field_records:
        rows.append({
            "field_id": rec.field_id,
            "material": rec.material,
            "replicate": rec.replicate,
            "subfield": rec.subfield,
            "actin_path": str(rec.actin_path),
            "nuclear_path": str(rec.nuclear_path),
            "actin_sha256": rec.actin_sha256,
            "nuclear_sha256": rec.nuclear_sha256,
            "image_shape": "x".join(str(s) for s in rec.image_shape),
            "voxel_size_um": rec.pixel_size_um,
            "segmentation_mode": rec.segmentation_mode,
        })

    df = pd.DataFrame(rows)
    out_path = run_dir / "manifest.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote manifest ({len(df)} records) to {out_path}")
    return out_path
