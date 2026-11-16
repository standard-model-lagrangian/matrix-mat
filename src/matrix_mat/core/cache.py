"""
Mask Disk Caching Module for Spheroid Segmentation.

Masks are stored losslessly as 16-bit PNG (labels 0..65,535). Masks whose largest
label exceeds 65,535 are stored as compressed ``.npz`` (int32/int64) instead, so
no label is ever clipped. All writes (mask and metadata sidecar) are atomic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
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

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_NPZ_MAGIC = b"PK\x03\x04"
_PNG_MAX_LABEL = 65535


def _canonicalize(obj: Any, _depth: int = 0) -> Any:
    """Convert a (possibly nested) config into plain JSON-serialisable data, deterministically.

    Handles pydantic models, dataclasses, dicts, lists/tuples/sets, numpy scalars/arrays,
    paths and floats (rounded to 6 d.p.; NaN/inf encoded as strings). Non-algorithmic
    fields are dropped from every dataclass / model / mapping level.
    """
    if _depth > 50:
        raise ValueError("Cache config nesting too deep (possible reference cycle)")
    d = _depth + 1

    def _mapping(m: Dict[Any, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in m.items():
            ks = str(k)
            if ks in NON_ALGORITHMIC_FIELDS:
                continue
            out[ks] = _canonicalize(v, d)
        return dict(sorted(out.items()))

    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.generic):
        return _canonicalize(obj.item(), d)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return round(obj, 6)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump") and not isinstance(obj, type):
        return {"__type__": type(obj).__name__, **_mapping(obj.model_dump())}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": type(obj).__name__,
            **_mapping({f.name: getattr(obj, f.name) for f in fields(obj)}),
        }
    if isinstance(obj, dict):
        return _mapping(obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x, d) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonicalize(x, d) for x in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": [list(obj.shape), str(obj.dtype)], "data": _canonicalize(obj.tolist(), d)}
    if hasattr(obj, "__dict__"):
        return _mapping({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    return repr(obj)


def compute_cache_key(
    image: np.ndarray | bytes,
    config: Any,
    model_hash: Optional[str] = None,
    package_version: Optional[str] = None,
) -> str:
    """
    Compute unique content-addressed cache key:
    sha256(image shape/dtype + bytes) || sha256(canonical JSON of config) || model weights sha256 || package version

    ``config`` may be a dataclass, a pydantic model, a dict, or a list/tuple combining any of
    these with plain values (e.g. ``{"seg": seg_cfg, "preprocess": pp_cfg, "pixel_size_um": 1.5}``).
    """
    # 1. Image hash (shape and dtype included so 3x4 and 4x3 arrays never collide)
    h = hashlib.sha256()
    if isinstance(image, bytes):
        h.update(b"bytes:")
        h.update(image)
    elif isinstance(image, np.ndarray):
        h.update(f"ndarray:{tuple(image.shape)}:{image.dtype.str}:".encode("utf-8"))
        h.update(np.ascontiguousarray(image).tobytes())
    else:
        raise TypeError(f"Expected numpy array or bytes, got {type(image)}")
    img_sha = h.hexdigest()

    # 2. Canonical JSON of algorithmic parameters
    canon = _canonicalize(config) if config is not None else {}
    param_json = json.dumps(canon, sort_keys=True, separators=(",", ":"))
    param_sha = hashlib.sha256(param_json.encode("utf-8")).hexdigest()

    # 3. Model weights hash
    model_sha = hashlib.sha256(str(model_hash or "none").encode("utf-8")).hexdigest()

    # 4. Package version hash (cached metadata semantics change between releases)
    if package_version is None:
        import matrix_mat
        package_version = matrix_mat.__version__
    pkg_sha = hashlib.sha256(str(package_version).encode("utf-8")).hexdigest()

    combined = f"img:{img_sha}|params:{param_sha}|model:{model_sha}|pkg:{pkg_sha}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".tmp_{target.name}_")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class MaskCache:
    """Persistent lossless label-mask cache (16-bit PNG, ``.npz`` for >65,535 labels)."""

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
        """Return the path of the PNG cache entry for a given image ID."""
        clean_id = self._sanitize_image_id(image_id)
        if not clean_id.endswith(".png"):
            clean_id = f"{clean_id}.png"
        return self.cache_dir / clean_id

    def _npz_path(self, image_id: str) -> Path:
        return self.get_cache_path(image_id).with_suffix(".npz")

    def _meta_path(self, image_id: str) -> Path:
        return self.get_cache_path(image_id).with_suffix(".meta.json")

    @staticmethod
    def _has_magic(path: Path, magic: bytes) -> bool:
        try:
            if not path.is_file() or path.stat().st_size <= len(magic):
                return False
            with open(path, "rb") as f:
                return f.read(len(magic)) == magic
        except OSError:
            return False

    def _existing_entry(self, image_id: str) -> Optional[Path]:
        png = self.get_cache_path(image_id)
        if self._has_magic(png, _PNG_MAGIC):
            return png
        npz = self._npz_path(image_id)
        if self._has_magic(npz, _NPZ_MAGIC):
            return npz
        return None

    def _drop(self, image_id: str) -> None:
        for p in (self.get_cache_path(image_id), self._npz_path(image_id), self._meta_path(image_id)):
            try:
                p.unlink(missing_ok=True)
            except OSError as e:  # pragma: no cover - permissions
                logger.warning(f"Could not delete corrupt cache file {p}: {e}")

    def has(self, image_id: str) -> bool:
        """Whether a cache entry with a valid file signature exists for the image ID.

        A corrupt body that still passes the signature check is detected by :meth:`load`,
        which deletes the entry and returns ``None``; ``has()`` is then False.
        """
        if not self.enabled:
            return False
        return self._existing_entry(image_id) is not None

    def load(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load a cached integer label mask. Returns an int32 array (int64 if labels need it),
        or None on a miss. Corrupt entries are deleted and reported as a miss.
        """
        if not self.enabled:
            return None
        path = self._existing_entry(image_id)
        if path is None:
            # A file without a valid signature is corrupt: remove it so has()/load() agree.
            if self.get_cache_path(image_id).exists() or self._npz_path(image_id).exists():
                logger.warning(f"Discarding corrupt cache entry for {image_id}")
                self._drop(image_id)
            return None
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as z:
                    mask = np.asarray(z["mask"])
                if mask.ndim != 2:
                    raise ValueError(f"cached mask has {mask.ndim} dims")
                return mask.astype(np.int64 if mask.max(initial=0) > np.iinfo(np.int32).max else np.int32)
            png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if png is None:
                raise ValueError("cv2 could not decode PNG")
            return png.astype(np.int32)
        except Exception as e:
            logger.warning(f"Discarding corrupt cached mask at {path}: {e}")
            self._drop(image_id)
            return None

    def load_meta(self, image_id: str) -> Dict[str, Any]:
        """Load cached metadata JSON from disk if available."""
        if not self.enabled:
            return {}
        meta_path = self._meta_path(image_id)
        if meta_path.exists() and self._existing_entry(image_id) is not None:
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
        Atomically save a 2-D integer label mask (16-bit PNG, or ``.npz`` when labels exceed
        65,535). The metadata sidecar is written atomically; when ``metadata`` is empty any
        stale sidecar from an earlier save is removed. Returns the saved file path.
        """
        if not self.enabled:
            return self.get_cache_path(image_id)

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(
                f"MaskCache only stores 2-D label masks; got array with shape {mask.shape} (ndim={mask.ndim})"
            )
        if mask.size and not np.issubdtype(mask.dtype, np.integer) and not np.issubdtype(mask.dtype, np.bool_):
            if not np.all(np.isfinite(mask)) or not np.all(np.equal(np.mod(mask, 1), 0)):
                raise ValueError("MaskCache labels must be finite integers")
        min_label = int(np.min(mask)) if mask.size else 0
        max_label = int(np.max(mask)) if mask.size else 0
        if min_label < 0:
            raise ValueError(f"MaskCache labels must be non-negative; got minimum label {min_label}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.get_cache_path(image_id)
        npz_path = self._npz_path(image_id)
        meta_path = self._meta_path(image_id)

        # Remove the old sidecar first so a crash can never pair new pixels with stale flags.
        meta_path.unlink(missing_ok=True)

        if max_label <= _PNG_MAX_LABEL:
            ok, buf = cv2.imencode(".png", mask.astype(np.uint16))
            if not ok:
                raise IOError(f"cv2.imencode failed to encode mask for {png_path}")
            _atomic_write_bytes(png_path, buf.tobytes())
            npz_path.unlink(missing_ok=True)
            saved = png_path
        else:
            import io
            dtype = np.int32 if max_label <= np.iinfo(np.int32).max else np.int64
            bio = io.BytesIO()
            np.savez_compressed(bio, mask=mask.astype(dtype))
            _atomic_write_bytes(npz_path, bio.getvalue())
            png_path.unlink(missing_ok=True)
            saved = npz_path
            logger.info(f"Mask {image_id} has {max_label} labels (>65,535): cached as .npz")

        if metadata:
            payload = json.dumps(metadata, indent=2, default=str).encode("utf-8")
            _atomic_write_bytes(meta_path, payload)

        logger.debug(f"Saved mask cache to {saved} (shape: {mask.shape}, max_label: {max_label})")
        return saved

    def clear(self) -> int:
        """Remove all cached masks (PNG and NPZ) and sidecars. Returns count of deleted masks."""
        if not self.cache_dir.exists():
            return 0
        count = 0
        for p in list(self.cache_dir.glob("*.png")) + list(self.cache_dir.glob("*.npz")):
            try:
                p.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Could not delete {p}: {e}")
            # Remove the metadata sidecar too, so no stale flags outlive their mask.
            p.with_suffix(".meta.json").unlink(missing_ok=True)
        return count
