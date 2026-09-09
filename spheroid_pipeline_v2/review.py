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

logger = logging.getLogger("spheroid_pipeline_v2.review")

# Exact 10-column schema mandated by PROJECT.md / survey specification
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
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)

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
        for ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
            candidates.append(self.review_dir / f"{image_stem}{ext}")
            candidates.append(self.review_dir / f"{image_stem}_mask{ext}")
            if pair_key and timepoint:
                candidates.append(self.review_dir / f"{pair_key}_{timepoint}{ext}")
                candidates.append(self.review_dir / f"{pair_key}_{timepoint}_mask{ext}")

        for cand in candidates:
            if cand.exists():
                logger.info(f"Found manual ground truth mask: {cand.name}")
                return cand

        return None

    def load_manual_mask(self, mask_path: Union[str, Path]) -> np.ndarray:
        """
        Loads manual ground truth mask image and returns a 2D int32 labeled mask.
        Handles binary masks (converting connected components to distinct labels)
        and existing multi-label integer masks.
        """
        mask_path = Path(mask_path)
        with Image.open(mask_path) as img:
            arr = np.array(img)

        # Convert RGB/RGBA to single channel
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        # Case 1: Boolean mask
        if arr.dtype == bool:
            labeled, _ = measure.label(arr, return_num=True)
            return labeled.astype(np.int32)

        # Case 2: Binary mask (0 and 255)
        unique_vals = np.unique(arr)
        if len(unique_vals) <= 2 and (unique_vals.max() > 1):
            labeled, _ = measure.label(arr > 0, return_num=True)
            return labeled.astype(np.int32)
        elif len(unique_vals) <= 2 and (unique_vals.max() == 1):
            labeled, _ = measure.label(arr > 0, return_num=True)
            return labeled.astype(np.int32)
        else:
            # Multi-label integer mask
            return arr.astype(np.int32)

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
            manual_mask = self.load_manual_mask(manual_path)
            # Ensure shape matches
            if manual_mask.shape[:2] != auto_mask.shape[:2]:
                logger.warning(
                    f"Manual mask shape {manual_mask.shape} does not match auto mask {auto_mask.shape} for {image_id}. Keeping auto mask."
                )
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
