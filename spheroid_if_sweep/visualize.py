"""
Visualization utilities: 2-channel merge overlays with segmentation outlines and QC inspection panels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tifffile

logger = logging.getLogger("spheroid_if_sweep.visualize")


def create_dual_channel_overlay(
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    masks: np.ndarray,
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Create an RGB overlay:
    - Red channel: Actin (phalloidin)
    - Blue/Cyan channel: Nuclear (DAPI)
    - Green contours: Segmented nuclei boundaries
    """
    # Normalize channels to 8-bit [0, 255]
    nuc_u8 = cv2.normalize(nuclear_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    act_u8 = cv2.normalize(actin_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Compose RGB base: Red=Actin, Green=dim actin+nuc, Blue=Nuclear
    # BGR format for OpenCV: B = Nuclear, G = 0, R = Actin
    b = nuc_u8
    g = np.zeros_like(nuc_u8)
    r = act_u8
    bgr = cv2.merge([b, g, r])

    # Draw contours in high-visibility bright green (0, 255, 0)
    n_objs = int(masks.max())
    if n_objs > 0:
        # Find boundaries
        for lbl in range(1, n_objs + 1):
            m = (masks == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, contours, -1, (0, 255, 128), 1, cv2.LINE_AA)

    # Title banner
    if title:
        cv2.putText(
            bgr, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            bgr, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA
        )

    return bgr


def save_mask_and_overlay(
    image_id: str,
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    masks: np.ndarray,
    masks_dir: Path,
    run_name: str = "",
) -> Tuple[Path, Path]:
    """Save masks/<image_id>_mask.tif and masks/<image_id>_overlay.png."""
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Save 16-bit label mask
    mask_tif_path = masks_dir / f"{image_id}_mask.tif"
    tifffile.imwrite(mask_tif_path, masks.astype(np.uint16))

    # Save overlay PNG
    title = f"{run_name} | {image_id} | {masks.max()} nuclei"
    overlay = create_dual_channel_overlay(nuclear_img, actin_img, masks, title=title)
    overlay_path = masks_dir / f"{image_id}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return mask_tif_path, overlay_path


def save_qc_overlay_panels(
    worst_images: List[dict],
    qc_dir: Path,
) -> None:
    """Save QC overlay PNGs for the worst-flagged images."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    for info in worst_images:
        img_id = info["image_id"]
        flags = info["flags"]
        overlay_src = info["overlay_path"]
        if overlay_src.exists():
            img = cv2.imread(str(overlay_src))
            if img is not None:
                # Add banner with active flags
                banner_text = f"QC FLAGS: {flags}"
                cv2.putText(img, banner_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                out_path = qc_dir / f"QC_FLAGGED_{img_id}.png"
                cv2.imwrite(str(out_path), img)
