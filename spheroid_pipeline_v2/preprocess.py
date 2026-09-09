"""
Preprocessing & Downsampling Module for Brightfield Microscopy Images.
Performs hierarchical flat-field illumination flattening, robust percentile normalization (1%-99%),
and proportional downsampling to match deep learning receptive fields (150-250 px diameter),
with nearest-neighbor label mask upscaling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from spheroid_pipeline_v2.config import PreprocessConfig

logger = logging.getLogger("spheroid_pipeline_v2.preprocess")


def flatten_illumination(image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Flat-field illumination correction using multi-scale Gaussian background estimation.
    Estimates low-frequency background B_hat and flattens uneven lighting/vignetting.
    """
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if sigma <= 0:
        return img.copy()

    # Downsample 4x for fast large-kernel Gaussian blur
    small_w = max(32, w // 4)
    small_h = max(32, h // 4)
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)

    small_sigma = max(1.0, sigma / 4.0)
    ksize = int(2 * np.ceil(2 * small_sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1

    small_bg = cv2.GaussianBlur(small_img, (ksize, ksize), small_sigma)
    bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg_safe = np.maximum(bg, 1e-4)

    mean_bg = float(np.mean(bg))
    corrected = (img / bg_safe) * mean_bg
    return corrected.astype(np.float32)


def normalize_percentiles(
    image: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize image contrast to [0.0, 1.0] using 1st and 99th intensity percentiles.
    Returns (normalized_image, val_p_low, val_p_high).
    """
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    sample = img[::4, ::4] if (h >= 32 and w >= 32) else img
    val_low = float(np.percentile(sample, p_low))
    val_high = float(np.percentile(sample, p_high))

    if val_high > val_low:
        norm = (img - val_low) / (val_high - val_low)
        norm = np.clip(norm, 0.0, 1.0)
    else:
        norm = np.zeros_like(img, dtype=np.float32)

    return norm.astype(np.float32), val_low, val_high



def compute_downsample_scale(
    image_shape: Tuple[int, int],
    pixel_size_um: float,
    expected_diameter_um: float = 300.0,
    target_diameter_px: float = 200.0,
) -> float:
    """
    Calculate proportional downsampling scale factor s = W' / W.
    Aligns expected physical diameter to target receptive field [150, 250] px.
    If 0.85 <= s <= 1.15, s is snapped to 1.0 (no scaling).
    """
    if pixel_size_um <= 0:
        return 1.0

    raw_expected_diameter_px = expected_diameter_um / pixel_size_um
    if raw_expected_diameter_px <= 0:
        return 1.0

    s = float(target_diameter_px / raw_expected_diameter_px)
    if 0.85 <= s <= 1.15:
        return 1.0
    return s


def downsample_image(image: np.ndarray, scale_factor: float) -> Tuple[np.ndarray, float]:
    """
    Downsamples image by scale_factor s using cv2.INTER_AREA.
    Returns (downsampled_image, scale_factor).
    """
    if abs(scale_factor - 1.0) < 1e-4:
        return image.copy(), 1.0

    h, w = image.shape[:2]
    new_w = max(32, int(round(w * scale_factor)))
    new_h = max(32, int(round(h * scale_factor)))

    downsampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    actual_scale = float(new_w / w)
    return downsampled, actual_scale


def upscale_labels(labels: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upscales downsampled integer label mask back to full resolution (target_shape: (H, W))
    using nearest-neighbor interpolation (cv2.INTER_NEAREST) to preserve exact integer label IDs.
    """
    target_h, target_w = target_shape[:2]
    cur_h, cur_w = labels.shape[:2]

    if (cur_h, cur_w) == (target_h, target_w):
        return labels.astype(np.int32)

    upscaled = cv2.resize(labels, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return upscaled.astype(np.int32)


class Preprocessor:
    """End-to-end preprocessing pipeline for brightfield microscopy images."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config if config is not None else PreprocessConfig()

    def process(
        self,
        image: np.ndarray,
        pixel_size_um: float = 1.518817,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Executes illumination flattening, percentile normalization, and proportional downsampling.
        Returns:
            downsampled_image: float32 array in [0, 1]
            scale_factor: s = W_downsampled / W_original
            metadata: diagnostics dictionary
        """
        raw_h, raw_w = image.shape[:2]

        # 1. Illumination flattening
        flattened = flatten_illumination(image, sigma=self.config.flat_field_sigma)

        # 2. Percentile normalization
        norm_img, p_low, p_high = normalize_percentiles(
            flattened,
            p_low=self.config.percentile_low,
            p_high=self.config.percentile_high,
        )

        # 3. Proportional downsampling
        target_d = float((self.config.target_diameter_min_px + self.config.target_diameter_max_px) / 2.0)
        s_target = compute_downsample_scale(
            image_shape=(raw_h, raw_w),
            pixel_size_um=pixel_size_um,
            expected_diameter_um=self.config.expected_spheroid_diameter_um,
            target_diameter_px=target_d,
        )

        downsampled_img, actual_s = downsample_image(norm_img, s_target)

        prep_meta = {
            "raw_shape": (raw_h, raw_w),
            "processed_shape": downsampled_img.shape[:2],
            "scale_factor": actual_s,
            "p_low": p_low,
            "p_high": p_high,
            "mean_intensity": float(np.mean(downsampled_img)),
            "std_intensity": float(np.std(downsampled_img)),
        }
        if meta is not None:
            prep_meta.update(meta)

        return downsampled_img, actual_s, prep_meta

    @staticmethod
    def save_image(img: np.ndarray, output_path: str | Path) -> None:
        """Save normalized float32 image [0, 1] as 8-bit PNG."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), arr_uint8)
