"""
Preprocessing module for spheroid IF imaging.
Config-driven pipeline applied to a copy of each channel in fixed order:
1. Background subtraction (rolling_ball, tophat, blank_gel_median, none)
2. Percentile normalization (scaled to float32 [0, 1] and uint8 [0, 255])
3. Denoising (gaussian, none)
All spatial parameters are specified in physical units (um) and resolved to pixels
using image-specific pixel_size_um metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import skimage.restoration as ski_restoration
import tifffile

logger = logging.getLogger("spheroid_if_sweep.preprocess")

_BLANK_MEDIAN_CACHE: Dict[str, np.ndarray] = {}


@dataclass
class ChannelPreprocessResult:
    """Holds preprocessed channel outputs and provenance metadata."""
    raw_img: np.ndarray
    preprocessed_f32: np.ndarray  # float32 in [0, 1]
    preprocessed_u8: np.ndarray   # uint8 in [0, 255]
    p_low_val: float
    p_high_val: float
    bg_radius_px: int
    bg_radius_um: float


@dataclass
class FieldPreprocessResult:
    """Holds results for paired nuclear and actin channels."""
    image_id: str
    nuclear: ChannelPreprocessResult
    actin: ChannelPreprocessResult
    resolved_config: Dict[str, Any]
    nuclear_tif_path: Optional[Path] = None
    actin_tif_path: Optional[Path] = None
    montage_path: Optional[Path] = None


def resolve_radius_px(
    expected_nuclear_diameter_um: float,
    radius_multiplier: float,
    pixel_size_um: float,
) -> Tuple[float, int]:
    """Derive background subtraction radius in um and pixels."""
    radius_um = expected_nuclear_diameter_um * radius_multiplier
    radius_px = max(3, int(round(radius_um / max(1e-6, pixel_size_um))))
    return radius_um, radius_px


def compute_blank_gel_median(blank_glob: str) -> Optional[np.ndarray]:
    """Compute and cache median projection of blank gel reference images."""
    if not blank_glob:
        return None
    if blank_glob in _BLANK_MEDIAN_CACHE:
        return _BLANK_MEDIAN_CACHE[blank_glob]

    import glob
    blank_files = glob.glob(blank_glob)
    if not blank_files:
        logger.warning(f"No blank gel images found matching glob: '{blank_glob}'")
        return None

    logger.info(f"Building median blank gel projection from {len(blank_files)} files.")
    imgs = [tifffile.imread(f).astype(np.float32) for f in blank_files]
    median_proj = np.median(np.stack(imgs, axis=0), axis=0)
    _BLANK_MEDIAN_CACHE[blank_glob] = median_proj
    return median_proj


def subtract_background(
    img: np.ndarray,
    method: str,
    radius_px: int,
    blank_glob: str = "",
    downsample_factor: int = 2,
) -> np.ndarray:
    """
    Apply background subtraction in-place on a float32 copy.
    Methods: 'rolling_ball', 'tophat', 'blank_gel_median', 'none'.
    """
    img_f32 = img.astype(np.float32, copy=True)
    method_lower = (method or "none").lower()

    if method_lower == "none":
        return img_f32

    elif method_lower == "rolling_ball":
        # Fast rolling ball via 2x downsample for large kernels if requested
        h, w = img_f32.shape[:2]
        if downsample_factor > 1 and radius_px > 20:
            target_w = max(32, w // downsample_factor)
            target_h = max(32, h // downsample_factor)
            small = cv2.resize(img_f32, (target_w, target_h), interpolation=cv2.INTER_AREA)
            r_small = max(2, radius_px // downsample_factor)
            bg_small = ski_restoration.rolling_ball(small, radius=r_small)
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bg = ski_restoration.rolling_ball(img_f32, radius=radius_px)
        return np.clip(img_f32 - bg, 0.0, None)

    elif method_lower == "tophat":
        ksize = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if img_f32.max() <= 255.0:
            th = cv2.morphologyEx(np.clip(img_f32, 0, 255).astype(np.uint8), cv2.MORPH_TOPHAT, kernel)
            return th.astype(np.float32)
        else:
            opened = ndimage.grey_opening(img_f32, footprint=kernel)
            return np.clip(img_f32 - opened, 0.0, None)

    elif method_lower == "blank_gel_median":
        blank = compute_blank_gel_median(blank_glob)
        if blank is not None and blank.shape == img_f32.shape:
            return np.clip(img_f32 - blank, 0.0, None)
        else:
            logger.warning("blank_gel_median unavailable; falling back to passthrough.")
            return img_f32

    else:
        logger.warning(f"Unknown background subtraction method '{method}', skipping.")
        return img_f32


def normalize_percentile(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.8,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Clip to [p_low, p_high] percentiles and scale to float32 [0, 1] and uint8 [0, 255].
    Returns (f32_norm, u8_norm, p_low_val, p_high_val).
    """
    p_low_val = float(np.percentile(img, p_low))
    p_high_val = float(np.percentile(img, p_high))
    denom = max(1e-6, p_high_val - p_low_val)

    clipped = np.clip(img, p_low_val, p_high_val)
    f32_norm = ((clipped - p_low_val) / denom).astype(np.float32)
    u8_norm = np.clip(f32_norm * 255.0, 0.0, 255.0).astype(np.uint8)

    return f32_norm, u8_norm, p_low_val, p_high_val


def denoise_image(
    img: np.ndarray,
    method: str,
    sigma_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Apply denoising filter if enabled."""
    method_lower = (method or "none").lower()
    if method_lower == "none" or sigma_um <= 0:
        return img

    if method_lower == "gaussian":
        sigma_px = sigma_um / max(1e-6, pixel_size_um)
        denoised = ndimage.gaussian_filter(img, sigma=sigma_px)
        return denoised
    else:
        logger.warning(f"Unknown denoise method '{method}', skipping.")
        return img


def preprocess_single_channel(
    channel_img: np.ndarray,
    pixel_size_um: float,
    config: Dict[str, Any],
) -> ChannelPreprocessResult:
    """Run full 3-stage preprocessing pipeline on a single 2D channel."""
    # Never mutate raw input
    img_work = channel_img.astype(np.float32, copy=True)

    expected_diam = float(config.get("expected_nuclear_diameter_um", 11.7))
    bg_cfg = config.get("background", {})
    bg_method = bg_cfg.get("method", "none")
    radius_mult = float(bg_cfg.get("radius_multiplier", 3.5))
    blank_glob = bg_cfg.get("blank_glob", "")
    downsample_factor = int(bg_cfg.get("downsample_factor", 2))

    radius_um, radius_px = resolve_radius_px(expected_diam, radius_mult, pixel_size_um)

    # Stage 1: Background subtraction
    if bg_method and bg_method.lower() != "none":
        img_work = subtract_background(
            img=img_work,
            method=bg_method,
            radius_px=radius_px,
            blank_glob=blank_glob,
            downsample_factor=downsample_factor,
        )

    # Stage 2: Normalization
    norm_cfg = config.get("normalize", {})
    norm_enabled = norm_cfg.get("enabled", True)
    if norm_enabled:
        p_low = float(norm_cfg.get("p_low", 1.0))
        p_high = float(norm_cfg.get("p_high", 99.8))
        f32_norm, u8_norm, p_low_val, p_high_val = normalize_percentile(img_work, p_low, p_high)
    else:
        p_low_val = float(img_work.min())
        p_high_val = float(img_work.max())
        f32_norm = np.clip(img_work / max(1e-6, p_high_val), 0.0, 1.0).astype(np.float32)
        u8_norm = np.clip(img_work, 0.0, 255.0).astype(np.uint8)

    # Stage 3: Denoise
    denoise_cfg = config.get("denoise", {})
    denoise_method = denoise_cfg.get("method", "none")
    sigma_um = float(denoise_cfg.get("sigma_um", 0.0))
    if denoise_method and denoise_method.lower() != "none" and sigma_um > 0:
        f32_norm = denoise_image(f32_norm, denoise_method, sigma_um, pixel_size_um)
        f32_norm = np.clip(f32_norm, 0.0, 1.0).astype(np.float32)
        u8_norm = np.clip(f32_norm * 255.0, 0.0, 255.0).astype(np.uint8)

    return ChannelPreprocessResult(
        raw_img=channel_img,
        preprocessed_f32=f32_norm,
        preprocessed_u8=u8_norm,
        p_low_val=p_low_val,
        p_high_val=p_high_val,
        bg_radius_px=radius_px,
        bg_radius_um=radius_um,
    )


def save_preprocess_montage(
    image_id: str,
    raw_nuc: np.ndarray,
    proc_nuc: np.ndarray,
    output_path: Path,
    p_low: float,
    p_high: float,
    bg_method: str,
) -> Path:
    """Generate a before/after inspection panel showing raw vs preprocessed channel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("black")

    im0 = axes[0].imshow(raw_nuc, cmap="magma")
    axes[0].set_title(f"Raw Nuclear Channel\nmin={raw_nuc.min()}, max={raw_nuc.max()}", color="white", fontsize=12)
    axes[0].axis("off")
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar0.ax.axes, "yticklabels"), color="white")

    im1 = axes[1].imshow(proc_nuc, cmap="magma")
    axes[1].set_title(
        f"Preprocessed (method={bg_method})\nP1.0={p_low:.1f}, P99.8={p_high:.1f}",
        color="white",
        fontsize=12,
    )
    axes[1].axis("off")
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar1.ax.axes, "yticklabels"), color="white")

    plt.suptitle(f"Preprocessing Montage: {image_id}", color="white", fontsize=14, y=0.96)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return output_path


def preprocess_field(
    image_id: str,
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    pixel_size_um: float,
    preprocess_config: Dict[str, Any],
    output_dir: Optional[Path] = None,
    save_montage: bool = False,
) -> FieldPreprocessResult:
    """
    Execute preprocessing pipeline on nuclear and actin channels.
    Optionally saves intermediate TIFFs and inspection montage into output_dir.
    """
    enabled = bool(preprocess_config.get("enabled", True))
    if not enabled:
        # Passthrough mode
        raw_nuc_f32 = np.clip(nuclear_img.astype(np.float32) / 255.0, 0.0, 1.0)
        raw_act_f32 = np.clip(actin_img.astype(np.float32) / 255.0, 0.0, 1.0)
        nuc_res = ChannelPreprocessResult(
            raw_img=nuclear_img,
            preprocessed_f32=raw_nuc_f32,
            preprocessed_u8=nuclear_img.astype(np.uint8),
            p_low_val=float(nuclear_img.min()),
            p_high_val=float(nuclear_img.max()),
            bg_radius_px=0,
            bg_radius_um=0.0,
        )
        act_res = ChannelPreprocessResult(
            raw_img=actin_img,
            preprocessed_f32=raw_act_f32,
            preprocessed_u8=actin_img.astype(np.uint8),
            p_low_val=float(actin_img.min()),
            p_high_val=float(actin_img.max()),
            bg_radius_px=0,
            bg_radius_um=0.0,
        )
        return FieldPreprocessResult(
            image_id=image_id,
            nuclear=nuc_res,
            actin=act_res,
            resolved_config={"enabled": False},
        )

    # Process channels independently
    nuc_res = preprocess_single_channel(nuclear_img, pixel_size_um, preprocess_config)
    act_res = preprocess_single_channel(actin_img, pixel_size_um, preprocess_config)

    nuc_tif_path = None
    act_tif_path = None
    montage_path = None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        nuc_tif_path = output_dir / f"{image_id}_nuclear.tif"
        act_tif_path = output_dir / f"{image_id}_actin.tif"

        tifffile.imwrite(nuc_tif_path, nuc_res.preprocessed_f32)
        tifffile.imwrite(act_tif_path, act_res.preprocessed_f32)

        if save_montage:
            bg_method = preprocess_config.get("background", {}).get("method", "none")
            montage_path = output_dir / f"{image_id}_montage.png"
            save_preprocess_montage(
                image_id=image_id,
                raw_nuc=nuclear_img,
                proc_nuc=nuc_res.preprocessed_u8,
                output_path=montage_path,
                p_low=nuc_res.p_low_val,
                p_high=nuc_res.p_high_val,
                bg_method=bg_method,
            )

    bg_cfg = preprocess_config.get("background", {})
    resolved_info = {
        "enabled": True,
        "expected_nuclear_diameter_um": preprocess_config.get("expected_nuclear_diameter_um", 11.7),
        "background": {
            "method": bg_cfg.get("method", "none"),
            "radius_multiplier": bg_cfg.get("radius_multiplier", 3.5),
            "radius_um": nuc_res.bg_radius_um,
            "radius_px": nuc_res.bg_radius_px,
        },
        "normalize": preprocess_config.get("normalize", {}),
        "denoise": preprocess_config.get("denoise", {}),
    }

    return FieldPreprocessResult(
        image_id=image_id,
        nuclear=nuc_res,
        actin=act_res,
        resolved_config=resolved_info,
        nuclear_tif_path=nuc_tif_path,
        actin_tif_path=act_tif_path,
        montage_path=montage_path,
    )
