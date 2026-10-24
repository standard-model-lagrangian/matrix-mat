"""
Provenance & Run Metadata Tracking for matrix-mat.

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

from matrix_mat.core.scale import ScaleExtractor

logger = logging.getLogger("matrix_mat.core.provenance")


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
    """Detects available hardware accelerator without unconditionally importing torch."""
    import sys
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        if torch_mod.cuda.is_available():
            return f"CUDA: {torch_mod.cuda.get_device_name(0)}"
        elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
            return f"Apple Silicon MPS ({platform.processor()} / {platform.machine()})"
    return "CPU"


def get_package_version(package_name: str) -> str:
    """Safely retrieves installed package version."""
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return "unknown"


def get_cellpose_version() -> str:
    """Safely retrieves installed cellpose package version."""
    return get_package_version("cellpose")


def get_lockfile_sha256() -> Optional[str]:
    """Computes SHA-256 of models.lock.json or dependencies lockfile if available."""
    for lock_name in ("models.lock.json", "uv.lock", "poetry.lock"):
        p = Path(lock_name)
        if p.exists() and p.is_file():
            try:
                return compute_file_sha256(p)
            except Exception:
                pass
    return None


def get_default_weights_sha256(model_ident: str = "cpsam_v2") -> Optional[str]:
    """Retrieves expected or computed weights SHA-256 from models.lock.json or disk."""
    lock_path = Path("models.lock.json")
    if lock_path.exists():
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_info = data.get("models", {}).get(model_ident)
            if model_info and "sha256" in model_info:
                return str(model_info["sha256"])
        except Exception:
            pass

    for candidate in [Path("models/cpsam_v2"), Path(f"models/{model_ident}")]:
        if candidate.exists() and candidate.is_file():
            try:
                return compute_file_sha256(candidate)
            except Exception:
                pass
    return None


def write_run_metadata(
    output_dir: Optional[Union[str, Path]] = None,
    config: Any = None,
    wall_time_sec: float = 0.0,
    warnings: Optional[List[str]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    weights_sha256: Optional[str] = None,
    lockfile_sha256: Optional[str] = None,
    backend: Optional[str] = None,
    pipeline_name: str = "matrix-mat",
    resolved_params: Optional[Dict[str, Any]] = None,
    preprocessing_info: Optional[Dict[str, Any]] = None,
    postprocessing_info: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    run_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Writes comprehensive run_metadata.json into the output directory.
    Captures complete reproducibility metadata: environment, hardware,
    git commit, package versions, configuration snapshot, lockfile hash,
    weights hash, and logged warnings.
    """
    target_dir = Path(output_dir or run_dir or ".")
    target_dir.mkdir(parents=True, exist_ok=True)

    final_lockfile_sha256 = lockfile_sha256 or get_lockfile_sha256()
    final_weights_sha256 = weights_sha256 or get_default_weights_sha256()
    active_backend = backend or (
        getattr(getattr(config, "segmentation", None), "primary_backend", None)
        if config else None
    ) or (resolved_params.get("model_name") if resolved_params else None) or "cpsam"

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_info = get_git_info()
    metadata = {
        "pipeline_name": pipeline_name,
        "timestamp_utc": now_iso,
        "timestamp_iso": now_iso,
        "backend": active_backend,
        "model_name": active_backend,
        "cellpose_version": get_cellpose_version(),
        "torch_version": get_package_version("torch"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "gpu_model": get_gpu_info(),
        "git": git_info,
        "weights_sha256": final_weights_sha256,
        "lockfile_sha256": final_lockfile_sha256,
        "wall_time_seconds": round(wall_time_sec, 3),
        "random_seed": random_seed if random_seed is not None else getattr(config, "random_seed", 42),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "hardware_accelerator": get_gpu_info(),
            "git": git_info,
        },
        "dependency_versions": {
            "torch": get_package_version("torch"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tifffile": get_package_version("tifffile"),
            "cellpose": get_package_version("cellpose"),
            "scipy": get_package_version("scipy"),
            "skimage": get_package_version("scikit-image"),
            "opencv": get_package_version("opencv-python-headless"),
            "matplotlib": get_package_version("matplotlib"),
        },
    }

    if config is not None:
        metadata["resolved_config"] = config.model_dump() if hasattr(config, "model_dump") else {}
        metadata["config"] = config.model_dump() if hasattr(config, "model_dump") else {}
        metadata["resolved_parameters"] = {
            "t0_dir": str(getattr(config, "t0_dir", "")),
            "t7_dir": str(getattr(config, "t7_dir", "")),
            "output_dir": str(getattr(config, "output_dir", "")),
            "review_dir": str(getattr(config, "review_dir", "")),
            "default_pixel_size_um": getattr(config, "default_pixel_size_um", 1.518817),
            "cli_pixel_size_um": getattr(config, "cli_pixel_size_um", None),
            "segmentation_backend": getattr(getattr(config, "segmentation", None), "primary_backend", "cpsam"),
            "qc_min_d_um": getattr(getattr(config, "qc", None), "min_d_um", 40.0),
            "qc_max_d_um": getattr(getattr(config, "qc", None), "max_d_um", 1500.0),
            "qc_border_margin_px": getattr(getattr(config, "qc", None), "border_margin_px", 15),
            "qc_ring_contrast_factor": getattr(getattr(config, "qc", None), "ring_contrast_factor", 0.90),
            "stats_control_condition": getattr(getattr(config, "stats", None), "control_condition", "Mat"),
            "stats_min_denominator_d0_um": getattr(getattr(config, "stats", None), "min_denominator_d0_um", 60.0),
            "stats_bootstrap_iterations": getattr(getattr(config, "stats", None), "bootstrap_iterations", 2000),
            "stats_permutation_iterations": getattr(getattr(config, "stats", None), "permutation_iterations", 10000),
        }
    elif resolved_params:
        metadata["resolved_parameters"] = resolved_params

    if preprocessing_info:
        metadata["preprocessing"] = preprocessing_info
    if postprocessing_info:
        metadata["postprocessing"] = postprocessing_info
    if extra_info:
        metadata["execution_stats"] = extra_info
        metadata["extra"] = extra_info

    metadata["warnings_logged"] = warnings or []
    metadata["warnings_count"] = len(warnings or [])

    out_file = target_dir / "run_metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Recorded run metadata to {out_file}")
    return out_file


def write_manifest_csv(
    run_dir: Path,
    field_records: Any,
) -> Path:
    """Write manifest.csv containing file paths, checksums, and metadata for IF fields."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in field_records:
        rows.append({
            "field_id": getattr(rec, "field_id", ""),
            "material": getattr(rec, "material", ""),
            "replicate": getattr(rec, "replicate", ""),
            "subfield": getattr(rec, "subfield", ""),
            "actin_path": str(getattr(rec, "actin_path", "")),
            "nuclear_path": str(getattr(rec, "nuclear_path", "")),
            "actin_sha256": getattr(rec, "actin_sha256", ""),
            "nuclear_sha256": getattr(rec, "nuclear_sha256", ""),
            "pixel_size_um": getattr(rec, "pixel_size_um", 0.505049),
            "image_shape": "x".join(str(s) for s in rec.image_shape) if hasattr(rec, "image_shape") and rec.image_shape else "",
            "voxel_size_um": getattr(rec, "pixel_size_um", 0.505049),
            "segmentation_mode": getattr(rec, "segmentation_mode", "2D"),
        })
    df = pd.DataFrame(rows)
    out_path = run_dir / "manifest.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved IF manifest with {len(df)} entries to {out_path}")
    return out_path


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
        # Scale metadata
        pixel_size = r.get("pixel_size_um")
        scale_src = r.get("scale_source")
        is_cal = r.get("is_calibrated")

        if (scale_src is None or is_cal is None) and p.exists():
            s_info = ScaleExtractor().extract(p)
            if pixel_size is None:
                pixel_size = s_info.pixel_size_um
            if scale_src is None:
                scale_src = s_info.source
            if is_cal is None:
                is_cal = s_info.is_calibrated

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
            "pixel_size_um": pixel_size if pixel_size is not None else 1.518817,
            "scale_source": scale_src if scale_src is not None else "default_fallback",
            "is_calibrated": bool(is_cal) if is_cal is not None else False,
            "is_matched_pair": r.get("is_matched_pair", False),
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "manifest.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Exported complete data manifest ({len(df)} images) to {out_csv}")
    return out_csv
