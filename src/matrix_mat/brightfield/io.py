"""
Image discovery and loading helpers shared by the brightfield pipeline, preflight and audits.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import tifffile

TIFF_SUFFIXES = (".tif", ".tiff")


def list_tiff_files(directory: Union[str, Path]) -> List[Path]:
    """Sorted .tif/.tiff files (case-insensitive extension) directly inside ``directory``."""
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """
    Reduce an image array to a single 2-D plane, preserving dtype and bit depth.

    Accepts (H, W), (H, W, 1|3|4) and channel-first (1|3|4, H, W). RGB(A) is converted with
    luminance weights; other extra axes raise ValueError.
    """
    a = np.asarray(arr)
    a = np.squeeze(a) if a.ndim > 2 else a
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
            a = np.moveaxis(a, 0, -1)
        if a.shape[-1] in (3, 4):
            rgb = a[..., :3]
            if rgb.dtype in (np.uint8, np.uint16, np.float32):
                return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
            gray = rgb.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
            return gray.astype(rgb.dtype) if np.issubdtype(rgb.dtype, np.floating) else np.round(gray).astype(rgb.dtype)
    raise ValueError(f"Unsupported image shape {np.asarray(arr).shape}; expected a 2-D grayscale or RGB(A) image")


def load_grayscale_image(path: Union[str, Path]) -> np.ndarray:
    """Read a TIFF with tifffile (keeps 16-bit depth) and return a single 2-D plane."""
    return to_grayscale(tifffile.imread(str(path)))


def to_uint8_display(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-stretch any-bit-depth grayscale image to uint8 for display."""
    a = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(a, (p_low, p_high))
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
