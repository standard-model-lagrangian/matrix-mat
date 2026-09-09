"""
Mask Disk Caching Module for Spheroid Segmentation.
Provides lossless 16-bit PNG caching in `masks/<image_id>.png` to eliminate redundant inference.
Supports up to 65,535 distinct instance IDs per field of view.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from typing import Optional, Union
import cv2
import numpy as np

logger = logging.getLogger("spheroid_pipeline_v2.mask_cache")


class MaskCache:
    """Persistent lossless 16-bit PNG mask cache."""

    def __init__(self, cache_dir: Union[str, Path] = "masks", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = bool(enabled)
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_image_id(self, image_id: str) -> str:
        """Sanitize image ID for safe filesystem filename while preserving identity."""
        # Replace characters that are invalid in file paths across OS
        clean = image_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return clean

    def get_cache_path(self, image_id: str) -> Path:
        """Return the absolute path to the cached mask file for a given image ID."""
        clean_id = self._sanitize_image_id(image_id)
        if not clean_id.endswith(".png"):
            clean_id = f"{clean_id}.png"
        return self.cache_dir / clean_id

    def has(self, image_id: str) -> bool:
        """Check whether a valid cached mask exists for the image ID."""
        if not self.enabled:
            return False
        cache_path = self.get_cache_path(image_id)
        return cache_path.exists() and cache_path.is_file() and cache_path.stat().st_size > 0

    def load(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load cached 16-bit integer label mask from disk.
        Returns int32 numpy array or None if cache miss / corrupt.
        """
        if not self.has(image_id):
            return None

        cache_path = self.get_cache_path(image_id)
        try:
            mask = cv2.imread(str(cache_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                logger.warning(f"Failed to decode cached mask image at {cache_path}")
                return None
            return mask.astype(np.int32)
        except Exception as e:
            logger.warning(f"Error loading cached mask from {cache_path}: {e}")
            return None

    def save(self, image_id: str, mask: np.ndarray) -> Path:
        """
        Save labeled segmentation mask to disk as lossless 16-bit PNG using an atomic write pattern.
        Returns the saved file path.
        """
        if not self.enabled:
            return self.get_cache_path(image_id)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.get_cache_path(image_id)

        # Convert to uint16 (handles 0..65535 labels)
        mask_u16 = np.clip(mask, 0, 65535).astype(np.uint16)

        # Write to temporary file in the same directory, then atomically rename via os.replace
        with tempfile.NamedTemporaryFile(
            dir=self.cache_dir,
            prefix=f".tmp_{cache_path.stem}_",
            suffix=".png",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            success = cv2.imwrite(str(tmp_path), mask_u16)
            if not success:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise IOError(f"cv2.imwrite failed to save mask at temporary path {tmp_path}")
            os.replace(tmp_path, cache_path)
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

        logger.debug(f"Saved mask cache to {cache_path} (shape: {mask.shape}, max_label: {int(np.max(mask))})")
        return cache_path

    def clear(self) -> int:
        """Remove all cached PNG masks in cache directory. Returns count of deleted files."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for p in self.cache_dir.glob("*.png"):
            try:
                p.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Could not delete {p}: {e}")
        return count
