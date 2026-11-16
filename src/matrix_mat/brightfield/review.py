"""
Review & Validation Module for Human Supervised Masks.
Monitors the `review/` directory for manual ground-truth masks, computes Dice and IoU
agreement metrics against automated segmentations, outputs `validation.csv`, and seamlessly
overrides automated masks when manual reviews are provided.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure

logger = logging.getLogger("matrix_mat.brightfield.review")

# Lossless formats accepted for manual masks (JPEG is deliberately excluded).
MASK_EXTENSIONS = (".png", ".tif", ".tiff")
# Largest number of labels for which an integer mask is interpreted as an instance label image.
MAX_MANUAL_LABELS = 1000


def _looks_like_label_image(arr: np.ndarray, nonzero_labels: np.ndarray) -> bool:
    """True only for integer masks whose labels are exactly 1..K (K >= 2, small)."""
    if not np.issubdtype(arr.dtype, np.integer):
        return False
    k = len(nonzero_labels)
    if k < 2 or k > MAX_MANUAL_LABELS:
        return False
    if arr.dtype == np.uint8 and int(nonzero_labels.max()) == 255:
        return False
    return bool(np.array_equal(nonzero_labels, np.arange(1, k + 1)))

# Exact 10-column schema mandated by specification
VALIDATION_CSV_COLUMNS: List[str] = [
    "image_id",
    "pair_key",
    "timepoint",
    "manual_mask_path",
    "n_auto_objects",
    "n_manual_objects",
    "dice_coefficient",
    "iou_jaccard",
    "pixel_size_um",
    "notes",
]


class ReviewManager:
    """Manages manual ground-truth mask ingestion, agreement validation, and mask overriding."""

    def __init__(self, review_dir: Union[str, Path] = "review"):
        # Read-only input folder: never created here, so runs leave no stray directories behind.
        self.review_dir = Path(review_dir)

    def find_manual_mask(
        self,
        image_stem: str,
        pair_key: Optional[str] = None,
        timepoint: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Looks for matching manual ground truth mask file in review/ folder.

        Checks multiple standard file naming conventions:
          - <image_stem>.png / .tif / .tiff
          - <image_stem>_mask.png / .tif / .tiff
          - <pair_key>_<timepoint>.png / .tif / .tiff
          - <pair_key>_<timepoint>_mask.png / .tif / .tiff
        """
        candidates: List[Path] = []
        for ext in MASK_EXTENSIONS:
            candidates.append(self.review_dir / f"{image_stem}{ext}")
            candidates.append(self.review_dir / f"{image_stem}_mask{ext}")
            if pair_key and timepoint:
                candidates.append(self.review_dir / f"{pair_key}_{timepoint}{ext}")
                candidates.append(self.review_dir / f"{pair_key}_{timepoint}_mask{ext}")

        for cand in candidates:
            if cand.exists():
                logger.info(f"Found manual ground truth mask: {cand.name}")
                return cand

        # JPEG is lossy (compression ringing turns a binary mask into many grey levels); never used.
        for ext in (".jpg", ".jpeg"):
            for stem in filter(None, [image_stem, f"{image_stem}_mask",
                                      f"{pair_key}_{timepoint}" if pair_key and timepoint else None]):
                cand = self.review_dir / f"{stem}{ext}"
                if cand.exists():
                    logger.warning(
                        f"Ignoring manual mask {cand.name}: JPEG masks are lossy and not supported; "
                        "save the mask as PNG or TIFF."
                    )
        return None

    def load_manual_mask(self, mask_path: Union[str, Path], expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Loads a manual ground truth mask and returns a 2D int32 labeled mask.

        Masks are treated as BINARY (foreground = connected components) unless they are a genuine
        integer label image: integer dtype, more than one label, labels exactly 1..K with
        K <= MAX_MANUAL_LABELS, and not an 8-bit image that reaches 255 (an anti-aliased binary
        drawing). Grey-level binary drawings are thresholded at half their maximum.

        Raises ValueError if ``expected_shape`` is given and the mask's (H, W) differs.
        """
        mask_path = Path(mask_path)
        if mask_path.suffix.lower() in (".jpg", ".jpeg"):
            raise ValueError(f"JPEG manual masks are not supported (lossy): {mask_path}")
        with Image.open(mask_path) as img:
            arr = np.array(img)

        # Convert RGB/RGBA to single channel (alpha ignored; any coloured channel counts)
        if arr.ndim == 3:
            arr = arr[:, :, :3].max(axis=2) if arr.shape[2] >= 3 else arr[:, :, 0]

        if expected_shape is not None and tuple(arr.shape[:2]) != tuple(expected_shape[:2]):
            raise ValueError(
                f"Manual mask {mask_path.name} has shape {arr.shape[:2]} but the image is {tuple(expected_shape[:2])}"
            )

        if arr.dtype == bool:
            labeled, _ = measure.label(arr, return_num=True)
            return labeled.astype(np.int32)

        unique_vals = np.unique(arr)
        nonzero = unique_vals[unique_vals > 0]
        if _looks_like_label_image(arr, nonzero):
            return arr.astype(np.int32)

        if len(unique_vals) <= 2:
            fg = arr > 0
        else:
            # Anti-aliased / grey-level binary drawing: threshold at half the maximum.
            fg = arr > (float(unique_vals.max()) / 2.0)
            logger.info(
                f"Manual mask {mask_path.name}: {len(unique_vals)} grey levels treated as a binary drawing "
                "(thresholded at 50% of max) rather than a label image."
            )
        labeled, _ = measure.label(fg, return_num=True)
        return labeled.astype(np.int32)

    @staticmethod
    def compute_agreement(auto_mask: np.ndarray, manual_mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculates Dice coefficient and IoU Jaccard index between automated and manual masks.

        Dice = 2 * |A ∩ B| / (|A| + |B|)
        IoU = |A ∩ B| / |A ∪ B|
        """
        auto_bin = (auto_mask > 0).astype(bool)
        man_bin = (manual_mask > 0).astype(bool)

        intersection = float(np.logical_and(auto_bin, man_bin).sum())
        union = float(np.logical_or(auto_bin, man_bin).sum())
        total = float(auto_bin.sum() + man_bin.sum())

        if total == 0:
            # Both masks empty -> perfect agreement
            dice = 1.0
            iou = 1.0
        else:
            dice = float((2.0 * intersection) / total)
            iou = float(intersection / union) if union > 0 else 0.0

        return dice, iou

    def evaluate_and_override(
        self,
        image_id: str,
        auto_mask: np.ndarray,
        pair_key: str = "",
        timepoint: str = "",
        pixel_size_um: float = 1.518817,
    ) -> Tuple[np.ndarray, str, Optional[Dict[str, Any]]]:
        """
        Checks if manual mask exists for the image. If found:
          - Computes Dice & IoU validation metrics.
          - Returns (manual_mask, 'manual_review', validation_record).
        If not found:
          - Returns (auto_mask, 'automated', None).
        """
        manual_path = self.find_manual_mask(image_id, pair_key=pair_key, timepoint=timepoint)
        if manual_path is None:
            return auto_mask, "automated", None

        try:
            try:
                manual_mask = self.load_manual_mask(manual_path, expected_shape=auto_mask.shape[:2])
            except ValueError as e:
                logger.warning(f"{e} for {image_id}. Keeping auto mask.")
                return auto_mask, "automated", None

            dice, iou = self.compute_agreement(auto_mask, manual_mask)
            n_auto = int(np.max(auto_mask))
            n_manual = int(np.max(manual_mask))

            validation_record = {
                "image_id": image_id,
                "pair_key": pair_key,
                "timepoint": timepoint,
                "manual_mask_path": str(manual_path),
                "n_auto_objects": n_auto,
                "n_manual_objects": n_manual,
                "dice_coefficient": float(round(dice, 4)),
                "iou_jaccard": float(round(iou, 4)),
                "pixel_size_um": float(pixel_size_um),
                "notes": f"Manual override applied. Dice={dice:.3f}, IoU={iou:.3f}",
            }

            logger.info(f"Manual mask override applied for {image_id}: Dice={dice:.4f}, IoU={iou:.4f}")
            return manual_mask, "manual_review", validation_record

        except Exception as e:
            logger.error(f"Error loading manual mask {manual_path}: {e}")
            return auto_mask, "automated", None

    @staticmethod
    def save_validation_report(
        records: List[Dict[str, Any]],
        output_path: Union[str, Path] = "validation.csv",
    ) -> pd.DataFrame:
        """
        Saves validation agreement records to validation.csv adhering to the 10-column schema.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not records:
            df = pd.DataFrame(columns=VALIDATION_CSV_COLUMNS)
        else:
            df = pd.DataFrame(records)
            for col in VALIDATION_CSV_COLUMNS:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[VALIDATION_CSV_COLUMNS]

        df.to_csv(output_path, index=False)
        logger.info(f"Saved validation metrics table to {output_path} ({len(df)} records)")
        return df
