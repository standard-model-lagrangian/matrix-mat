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

logger = logging.getLogger("matrix_mat.if_density.preprocess")

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
    # Steps that actually changed the data (provenance lists only these).
    methods_applied: List[str] = field(default_factory=list)


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


def ensure_2d_grayscale(img: np.ndarray, image_id: str = "") -> np.ndarray:
    """
    Return a 2D single-channel image. RGB(A) (H, W, 3|4) is converted to grayscale explicitly
    (ITU-R BT.601 luma, dtype preserved); singleton leading axes are squeezed; z-stacks raise ValueError
    because the IF pipeline is 2D only.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 4:
        logger.warning(f"[{image_id}] RGB(A) input {arr.shape} converted to grayscale (BT.601 luma)")
        luma = arr[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            luma = np.clip(np.rint(luma), info.min, info.max)
        return luma.astype(arr.dtype)
    squeezed = arr.reshape(arr.shape[-2:]) if arr.ndim > 2 and all(s == 1 for s in arr.shape[:-2]) else None
    if squeezed is not None:
        return squeezed
    raise ValueError(
        f"[{image_id}] multi-plane input {arr.shape} is not supported: the IF pipeline is 2D only"
    )


def to_unit_float(img: np.ndarray) -> np.ndarray:
    """
    Scale an image to float32 [0, 1] without wrapping:
    - uint8 (and any integer image whose values fit 0..255): divide by 255 (8-bit values are kept as is);
    - wider integer data: divide by the actual maximum (a 12-bit image in a uint16 container is not
      squashed into the bottom 1/16th of the range);
    - float data already in [0, 1] is kept; other float data is divided by its maximum.
    Negative values are clipped to 0.
    """
    arr = np.asarray(img)
    f = arr.astype(np.float32, copy=True)
    if arr.dtype == np.bool_:
        return f
    if np.issubdtype(arr.dtype, np.integer):
        vmax = float(arr.max()) if arr.size else 0.0
        denom = 255.0 if vmax <= 255.0 else vmax
    else:
        vmax = float(np.nanmax(f)) if f.size else 0.0
        denom = 1.0 if vmax <= 1.0 else vmax
    return np.clip(f / max(denom, 1e-6), 0.0, 1.0).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Scale an image to uint8 by bit depth (see to_unit_float); 8-bit input is returned unchanged."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr.copy()
    return np.clip(np.rint(to_unit_float(arr) * 255.0), 0, 255).astype(np.uint8)


def dtype_max_value(img: np.ndarray) -> float:
    """Largest representable value of an image's dtype (1.0 for float images in [0, 1], inf otherwise)."""
    arr = np.asarray(img)
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    if arr.dtype == np.bool_:
        return 1.0
    return 1.0 if (arr.size and float(np.nanmax(arr)) <= 1.0) else float("inf")


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
    Apply background subtraction on a float32 copy.
    Methods: 'rolling_ball', 'tophat', 'blank_gel_median', 'none'.
    """
    return _subtract_background(img, method, radius_px, blank_glob, downsample_factor)[0]


def _subtract_background(
    img: np.ndarray,
    method: str,
    radius_px: int,
    blank_glob: str = "",
    downsample_factor: int = 2,
) -> Tuple[np.ndarray, bool]:
    """Background subtraction; returns (result, applied) so provenance can tell a fallback from a run."""
    img_f32 = img.astype(np.float32, copy=True)
    method_lower = (method or "none").lower()

    if method_lower == "none":
        return img_f32, False

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
        return np.clip(img_f32 - bg, 0.0, None), True

    elif method_lower == "tophat":
        ksize = 2 * radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if img_f32.max() <= 255.0:
            th = cv2.morphologyEx(np.clip(img_f32, 0, 255).astype(np.uint8), cv2.MORPH_TOPHAT, kernel)
            return th.astype(np.float32), True
        else:
            opened = ndimage.grey_opening(img_f32, footprint=kernel)
            return np.clip(img_f32 - opened, 0.0, None), True

    elif method_lower == "blank_gel_median":
        blank = compute_blank_gel_median(blank_glob)
        if blank is not None and blank.shape == img_f32.shape:
            return np.clip(img_f32 - blank, 0.0, None), True
        else:
            logger.warning("blank_gel_median unavailable (no blank images or shape mismatch); not applied.")
            return img_f32, False

    else:
        raise ValueError(f"Unknown background subtraction method '{method}'")


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
        raise ValueError(f"Unknown denoise method '{method}'")


def preprocess_single_channel(
    channel_img: np.ndarray,
    pixel_size_um: float,
    config: Dict[str, Any],
) -> ChannelPreprocessResult:
    """Run full 3-stage preprocessing pipeline on a single 2D channel."""
    # Never mutate raw input
    img_work = channel_img.astype(np.float32, copy=True)
    applied: List[str] = []

    expected_diam = float(config.get("expected_nuclear_diameter_um", 11.7))
    bg_cfg = config.get("background", {})
    bg_method = bg_cfg.get("method", "none")
    radius_mult = float(bg_cfg.get("radius_multiplier", 3.5))
    blank_glob = bg_cfg.get("blank_glob") or ""
    downsample_factor = int(bg_cfg.get("downsample_factor", 2))

    radius_um, radius_px = resolve_radius_px(expected_diam, radius_mult, pixel_size_um)

    # Stage 1: Background subtraction
    if bg_method and bg_method.lower() != "none":
        img_work, bg_applied = _subtract_background(
            img=img_work,
            method=bg_method,
            radius_px=radius_px,
            blank_glob=blank_glob,
            downsample_factor=downsample_factor,
        )
        if bg_applied:
            applied.append(f"background_subtraction_{bg_method}")

    # Stage 2: Normalization
    norm_cfg = config.get("normalize", {})
    norm_enabled = norm_cfg.get("enabled", True)
    if norm_enabled:
        p_low = float(norm_cfg.get("p_low", 1.0))
        p_high = float(norm_cfg.get("p_high", 99.8))
        f32_norm, u8_norm, p_low_val, p_high_val = normalize_percentile(img_work, p_low, p_high)
        applied.append("normalize_percentile")
    else:
        # No percentile normalisation: rescale by bit depth (8-bit stays on its 0..255 scale, wider data
        # by its maximum). Never clip or wrap values above 255.
        p_low_val = float(img_work.min())
        p_high_val = float(img_work.max())
        if np.issubdtype(channel_img.dtype, np.integer) and p_high_val <= 255.0:
            denom = 255.0
        else:
            denom = max(1e-6, p_high_val)
        f32_norm = np.clip(img_work / denom, 0.0, 1.0).astype(np.float32)
        u8_norm = np.clip(np.rint(f32_norm * 255.0), 0.0, 255.0).astype(np.uint8)

    # Stage 3: Denoise
    denoise_cfg = config.get("denoise", {})
    denoise_method = denoise_cfg.get("method", "none")
    sigma_um = float(denoise_cfg.get("sigma_um", 0.0))
    if denoise_method and denoise_method.lower() != "none" and sigma_um > 0:
        f32_norm = denoise_image(f32_norm, denoise_method, sigma_um, pixel_size_um)
        f32_norm = np.clip(f32_norm, 0.0, 1.0).astype(np.float32)
        u8_norm = np.clip(f32_norm * 255.0, 0.0, 255.0).astype(np.uint8)
        applied.append(f"denoise_{denoise_method}")

    return ChannelPreprocessResult(
        raw_img=channel_img,
        preprocessed_f32=f32_norm,
        preprocessed_u8=u8_norm,
        p_low_val=p_low_val,
        p_high_val=p_high_val,
        bg_radius_px=radius_px,
        bg_radius_um=radius_um,
        methods_applied=applied,
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
    # Default matches the config schema: preprocessing runs only when explicitly enabled.
    enabled = bool(preprocess_config.get("enabled", False))
    if not enabled:
        # Passthrough mode: bit-depth scaling to uint8 only (8-bit data are unchanged; >8-bit data are
        # scaled, never wrapped by a bare astype).
        nuc_res = ChannelPreprocessResult(
            raw_img=nuclear_img,
            preprocessed_f32=to_unit_float(nuclear_img),
            preprocessed_u8=to_uint8(nuclear_img),
            p_low_val=float(nuclear_img.min()),
            p_high_val=float(nuclear_img.max()),
            bg_radius_px=0,
            bg_radius_um=0.0,
        )
        act_res = ChannelPreprocessResult(
            raw_img=actin_img,
            preprocessed_f32=to_unit_float(actin_img),
            preprocessed_u8=to_uint8(actin_img),
            p_low_val=float(actin_img.min()),
            p_high_val=float(actin_img.max()),
            bg_radius_px=0,
            bg_radius_um=0.0,
        )
        return FieldPreprocessResult(
            image_id=image_id,
            nuclear=nuc_res,
            actin=act_res,
            resolved_config={"enabled": False, "methods_applied": []},
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
        "methods_applied": list(nuc_res.methods_applied),
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
