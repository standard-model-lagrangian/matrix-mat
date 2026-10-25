"""
Mask Disk Caching Module for Spheroid Segmentation.
Provides lossless 16-bit PNG caching to eliminate redundant inference.
Supports up to 65,535 distinct instance IDs per field of view.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np

logger = logging.getLogger("matrix_mat.core.cache")

NON_ALGORITHMIC_FIELDS = {
    "cache_dir",
    "decisions_log_path",
    "force_recompute",
    "enable_mps_fallback",
    "backend",
}


def compute_cache_key(
    image: np.ndarray | bytes,
    config: Any,
    model_hash: Optional[str] = None,
    package_version: Optional[str] = None,
) -> str:
    """
    Compute unique content-addressed cache key:
    sha256(image bytes) || sha256(canonical JSON of segmentation params) || model weights sha256 || package version
    """
    # 1. Image hash
    if isinstance(image, bytes):
        img_bytes = image
    elif isinstance(image, np.ndarray):
        img_bytes = image.tobytes()
    else:
        raise TypeError(f"Expected numpy array or bytes, got {type(image)}")
    img_sha = hashlib.sha256(img_bytes).hexdigest()

    # 2. Canonical JSON of segmentation parameters
    if hasattr(config, "model_dump"):
        cfg_dict = config.model_dump()
    elif is_dataclass(config) and not isinstance(config, type):
        cfg_dict = asdict(config)
    elif isinstance(config, dict):
        cfg_dict = dict(config)
    elif hasattr(config, "__dict__"):
        cfg_dict = dict(config.__dict__)
    else:
        cfg_dict = {}

    filtered_params = {
        k: v for k, v in cfg_dict.items()
        if k not in NON_ALGORITHMIC_FIELDS
    }

    def _canon(obj):
        if isinstance(obj, dict):
            return {k: _canon(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [_canon(x) for x in obj]
        elif isinstance(obj, float):
            return round(obj, 6)
        return obj

    canon_dict = _canon(filtered_params)
    param_json = json.dumps(canon_dict, sort_keys=True, separators=(",", ":"))
    param_sha = hashlib.sha256(param_json.encode("utf-8")).hexdigest()

    # 3. Model weights hash
    model_sha = hashlib.sha256(str(model_hash or "none").encode("utf-8")).hexdigest()

    # 4. Package version hash (cached metadata semantics change between releases)
    if package_version is None:
        import matrix_mat
        package_version = matrix_mat.__version__
    pkg_sha = hashlib.sha256(str(package_version).encode("utf-8")).hexdigest()

    # Combined composite SHA256 key
    combined = f"img:{img_sha}|params:{param_sha}|model:{model_sha}|pkg:{pkg_sha}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class MaskCache:
    """Persistent lossless 16-bit PNG mask cache."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, enabled: bool = True):
        from matrix_mat.core.paths import resolve_cache_root
        if cache_dir is None:
            self.cache_dir = resolve_cache_root()
        else:
            self.cache_dir = Path(cache_dir)
        self.enabled = bool(enabled)

    def get_key(
        self,
        image: np.ndarray | bytes,
        config: Any,
        model_hash: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> str:
        """Compute the content-addressed cache key for an image and configuration."""
        return compute_cache_key(image, config, model_hash=model_hash)

    def _sanitize_image_id(self, image_id: str) -> str:
        """Sanitize image ID for safe filesystem filename while preserving identity."""
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

    def load_meta(self, image_id: str) -> Dict[str, Any]:
        """Load cached metadata JSON from disk if available."""
        if not self.enabled:
            return {}
        cache_path = self.get_cache_path(image_id)
        meta_path = cache_path.with_suffix(".meta.json")
        if meta_path.exists() and cache_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache metadata from {meta_path}: {e}")
        return {}

    def save(
        self,
        image_id: str,
        mask: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save labeled segmentation mask to disk as lossless 16-bit PNG using an atomic write pattern.
        Optionally persists accompanying metadata dictionary as sidecar JSON.
        Returns the saved file path.
        """
        if not self.enabled:
            return self.get_cache_path(image_id)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.get_cache_path(image_id)

        # Convert to uint16 (handles 0..65535 labels)
        mask_u16 = np.clip(mask, 0, 65535).astype(np.uint16)

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
            if metadata:
                meta_path = cache_path.with_suffix(".meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, default=str)
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
            # Remove the metadata sidecar too, so no stale flags outlive their mask.
            p.with_suffix(".meta.json").unlink(missing_ok=True)
        return count
