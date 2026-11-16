"""
Segmentation Module for Brightfield Spheroids.
Provides 3-tier segmentation hierarchy:
  1. Primary: Cellpose-SAM ('cpsam') with MPS (Apple M4) / CUDA / CPU fallback
  2. Fallback: Cellpose ('cpdino' with diameter sweep)
  3. Last Resort / Self-Contained Engine: Classical multi-scale adaptive watershed + annular contrast gating
With persistent disk caching, decision logging, and nearest-neighbor mask upscaling.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure, segmentation
from skimage.feature import peak_local_max

from matrix_mat.core.cache import NON_ALGORITHMIC_FIELDS, MaskCache
from matrix_mat.core.repro import verify_model_weights
from matrix_mat.brightfield.config import SegmentationConfig
from matrix_mat.brightfield.decisions import DecisionsLogger
from matrix_mat.brightfield.preprocess import Preprocessor, upscale_labels

logger = logging.getLogger("matrix_mat.brightfield.segment")


def get_optimal_device(preferred: str = "auto") -> Tuple[bool, str]:
    """
    Resolve the PyTorch computing device.

    ``preferred='auto'`` (default) detects the best device:
    - NVIDIA CUDA
    - Apple Silicon MPS (macOS M-series) with PYTORCH_ENABLE_MPS_FALLBACK=1
    - CPU fallback
    ``'cpu'`` forces the CPU. ``'cuda'``, ``'cuda:N'`` or ``'mps'`` force that accelerator and
    raise RuntimeError if it is not available (no silent fallback).
    Returns (use_gpu_bool, device_name_str).
    """
    pref = str(preferred or "auto").strip().lower()
    if pref == "cpu":
        return False, "cpu"
    if pref != "auto":
        try:
            import torch
        except Exception as e:  # pragma: no cover - torch missing
            raise RuntimeError(f"segmentation.device={pref!r} requested but PyTorch is not importable: {e}") from e
        if pref.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"segmentation.device={pref!r} requested but CUDA is not available")
            return True, pref
        if pref == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("segmentation.device='mps' requested but Apple MPS is not available")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return True, "mps"
        raise RuntimeError(f"Unsupported segmentation.device={pref!r}")

    try:
        import torch

        if torch.cuda.is_available():
            return True, "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return True, "mps"
    except Exception as e:
        logger.debug(f"PyTorch device detection encounter: {e}")

    return False, "cpu"


def _cellpose_device_kwargs(device_str: str, requested: str) -> Dict[str, Any]:
    """Extra CellposeModel kwargs pinning an explicitly configured device."""
    if str(requested or "auto").lower() == "auto":
        return {}
    try:
        import torch
        return {"device": torch.device(device_str)}
    except Exception:  # pragma: no cover
        return {}


class BaseSegmenter(ABC):
    """Abstract base class for segmentation backends."""

    @abstractmethod
    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segment a normalized 2D float32 image in [0, 1].
        Returns:
            labeled_mask: (H, W) int32 array where 0=background, 1..N=instances
            metadata: dictionary containing diagnostics
        """
        pass


class ClassicalMultiScaleSegmenter(BaseSegmenter):
    """
    Self-contained, deterministic classical computer vision segmentation engine.
    Uses multi-window adaptive thresholding, morphological closing/opening, hole filling,
    distance transform + marker-controlled watershed, and annular background contrast gating.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config if config is not None else SegmentationConfig()

    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        img_f32 = image.astype(np.float32)

        # Convert to 8-bit for OpenCV morphology
        img_u8 = np.clip(img_f32 * 255.0, 0, 255).astype(np.uint8)
        smooth = cv2.medianBlur(img_u8, 5)

        # 1. Multi-window local adaptive difference for varying blob diameters
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        window_sizes = self.config.adaptive_window_sizes or self.config.classical_adaptive_windows
        thresh_diff = self.config.classical_threshold_diff

        for w_size in window_sizes:
            if w_size >= min(h, w):
                continue
            k = int(w_size) if int(w_size) % 2 == 1 else int(w_size) + 1
            blur = cv2.blur(smooth, (k, k))
            diff = cv2.subtract(blur, smooth)
            _, th_dark = cv2.threshold(diff, thresh_diff, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_or(combined_mask, th_dark)

        # 2. Global Otsu threshold on inverted image as fallback / supplement
        inv = 255 - smooth
        _, th_otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_candidates = cv2.bitwise_and(combined_mask, th_otsu)

        # 3. Morphological closing to bridge fragmented spheroid edges
        close_r = max(2, int(self.config.morph_close_radius))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
        closed = cv2.morphologyEx(dark_candidates, cv2.MORPH_CLOSE, kernel_close)

        # 4. SciPy binary hole filling
        filled = ndimage.binary_fill_holes(closed > 0)

        # 5. Morphological opening to eliminate fine background noise
        open_r = max(1, int(self.config.morph_open_radius))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_r * 2 + 1, open_r * 2 + 1))
        opened = cv2.morphologyEx((filled * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel_open)

        # 6. Connected components size and border margin filtering
        min_d_um = float(getattr(self.config, "classical_min_d_um", 40.0))
        min_area_px = np.pi * ((min_d_um / (2.0 * max(1e-4, pixel_size_um))) ** 2)
        max_area_px = float(self.config.single_mask_max_area_pct * (h * w))
        margin = max(1, int(self.config.border_margin_px))

        num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(opened)
        filtered = np.zeros((h, w), dtype=np.uint8)
        valid_cc = 0
        for idx in range(1, num_cc):
            bx = stats[idx, cv2.CC_STAT_LEFT]
            by = stats[idx, cv2.CC_STAT_TOP]
            bw = stats[idx, cv2.CC_STAT_WIDTH]
            bh = stats[idx, cv2.CC_STAT_HEIGHT]
            area = stats[idx, cv2.CC_STAT_AREA]

            # Reject objects touching or within border margin
            if bx < margin or by < margin or (bx + bw) > (w - margin) or (by + bh) > (h - margin):
                continue
            if min_area_px <= area <= max_area_px:
                filtered[labels_cc == idx] = 255
                valid_cc += 1

        if valid_cc == 0:
            return np.zeros((h, w), dtype=np.int32), {
                "backend": "classical",
                "objects_found": 0,
                "status": "empty",
            }

        # 7. Distance transform & marker-controlled watershed
        dist = cv2.distanceTransform(filtered, cv2.DIST_L2, 5)
        dist_smooth = gaussian_filter(dist, sigma=2.0)

        min_dist_px = max(10, int(round(self.config.watershed_min_distance_px / max(1.0, pixel_size_um))))
        coords = peak_local_max(dist_smooth, min_distance=min_dist_px, labels=(filtered > 0), threshold_abs=3)

        if len(coords) > 0:
            mask_peaks = np.zeros((h, w), dtype=bool)
            mask_peaks[tuple(coords.T)] = True
            markers, _ = measure.label(mask_peaks, return_num=True)
            raw_labels = segmentation.watershed(-dist_smooth, markers, mask=(filtered > 0))
        else:
            raw_labels, _ = measure.label(filtered > 0, return_num=True)

        # 8. Annular background ring contrast verification (>= 10% darker than local ring)
        n_raw = int(np.max(raw_labels))
        clean_labels = np.zeros((h, w), dtype=np.int32)
        ring_r = max(5, int(getattr(self.config, "classical_ring_width_px", 15)))
        min_contrast = float(getattr(self.config, "classical_min_contrast", 0.10))
        kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_r * 2 + 1, ring_r * 2 + 1))
        all_mask = (raw_labels > 0).astype(np.uint8)

        kept_count = 0
        for obj_id in range(1, n_raw + 1):
            obj_mask = (raw_labels == obj_id).astype(np.uint8)
            ys, xs = np.where(obj_mask > 0)
            if len(ys) == 0:
                continue

            # Double-check border margin after watershed
            if ys.min() < margin or ys.max() > (h - margin) or xs.min() < margin or xs.max() > (w - margin):
                continue

            dilated = cv2.dilate(obj_mask, kernel_ring)
            ring_mask = (dilated > 0) & (all_mask == 0)

            int_mean = float(np.mean(img_f32[obj_mask > 0])) if np.sum(obj_mask) > 0 else 1.0
            if np.sum(ring_mask) > 0:
                ring_mean = float(np.mean(img_f32[ring_mask]))
            else:
                ring_mean = float(np.median(img_f32))

            contrast = (ring_mean - int_mean) / max(ring_mean, 1e-4)

            # Keep only if at least classical_min_contrast (default 10%) darker than background
            if contrast >= min_contrast:
                kept_count += 1
                clean_labels[obj_mask > 0] = kept_count

        meta = {
            "backend": "classical",
            "model_id": "classical",
            "objects_found": kept_count,
            "raw_candidates": n_raw,
            "status": "success" if kept_count > 0 else "empty",
        }
        return clean_labels.astype(np.int32), meta


class CellposeSAMSegmenter(BaseSegmenter):
    """
    Primary deep learning segmentation backend using Cellpose-SAM ('cpsam_v2' / 'cpsam').
    Supports MPS (Apple Silicon M-series), CUDA, and CPU fallback.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config if config is not None else SegmentationConfig()
        # Resolved lazily in _init_model so an unavailable forced device only matters when used.
        self.requested_device = str(getattr(self.config, "device", "auto") or "auto")
        self.use_gpu, self.device_str = False, self.requested_device
        self._model = None

    def _resolve_device(self) -> None:
        self.use_gpu, self.device_str = get_optimal_device(self.requested_device)

    def _init_model(self) -> Any:
        if self._model is None:
            from pathlib import Path
            model_path_str = getattr(self.config, "cpsam_model_path", "models/cpsam_v2")
            local_model = Path(model_path_str).resolve()
            # Verify pinned weights before touching cellpose so a missing/mismatched file aborts the run.
            self.model_sha256 = verify_model_weights(local_model, model_name="cpsam_v2")
            self._resolve_device()
            from cellpose import models
            models.MODEL_DIR = local_model.parent
            self._model = models.CellposeModel(
                pretrained_model=str(local_model),
                gpu=self.use_gpu,
                **_cellpose_device_kwargs(self.device_str, self.requested_device),
            )
        return self._model

    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        # Model initialisation failures (missing weights, checksum mismatch, missing cellpose)
        # are run-level errors and must propagate; only per-image inference errors are caught.
        model = self._init_model()
        try:
            # Ensure uint8 [0, 255] for Cellpose
            if image.dtype != np.uint8:
                img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = image

            d_val = float(self.config.cpsam_expected_diameter_px)
            masks, flows, styles = model.eval(
                img_u8,
                diameter=d_val,
                channels=[0, 0],
                flow_threshold=float(self.config.cpsam_flow_threshold),
                cellprob_threshold=float(self.config.cpsam_cellprob_threshold),
                min_size=int(self.config.cpsam_min_size_px),
            )
            labels = np.asarray(masks, dtype=np.int32)
            n_objs = int(np.max(labels))
            return labels, {
                "backend": "cpsam",
                "model_id": "cpsam_v2",
                "device": self.device_str,
                "objects_found": n_objs,
                "diameter_used": d_val,
                "status": "success" if n_objs > 0 else "empty",
            }
        except Exception as e:
            logger.warning(f"Cellpose-SAM inference failed: {e}")
            return np.zeros((h, w), dtype=np.int32), {
                "backend": "cpsam",
                "model_id": "cpsam_v2",
                "device": self.device_str,
                "error": str(e),
                "status": "failed",
                "objects_found": 0,
            }


class CellposeFoundationSegmenter(BaseSegmenter):
    """
    Secondary deep learning fallback backend using Cellpose foundation vision transformer ('cpdino').
    Performs multi-diameter scale sweep ([150, 200, 250, 300] px) if base scale yields 0 objects.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config if config is not None else SegmentationConfig()
        # Resolved lazily in _init_model so an unavailable forced device only matters when used.
        self.requested_device = str(getattr(self.config, "device", "auto") or "auto")
        self.use_gpu, self.device_str = False, self.requested_device
        self._model = None

    def _resolve_device(self) -> None:
        self.use_gpu, self.device_str = get_optimal_device(self.requested_device)

    def _init_model(self) -> Any:
        if self._model is None:
            self._resolve_device()
            from cellpose import models
            self._model = models.CellposeModel(
                pretrained_model="cpdino",
                gpu=self.use_gpu,
                **_cellpose_device_kwargs(self.device_str, self.requested_device),
            )
            loaded_path = str(getattr(self._model, "pretrained_model", ""))
            if "cpdino" not in loaded_path:
                raise RuntimeError(
                    f"Requested model 'cpdino' but Cellpose initialized with '{loaded_path}'. "
                    "Silent fallback to cpsam is disallowed."
                )
        return self._model

    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        # Model initialisation failures are run-level errors; see CellposeSAMSegmenter.segment.
        model = self._init_model()
        try:
            if image.dtype != np.uint8:
                img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = image

            # Initial evaluation with target expected diameter
            d_val = float(self.config.cpsam_expected_diameter_px)
            masks, flows, styles = model.eval(
                img_u8,
                diameter=d_val,
                channels=[0, 0],
                flow_threshold=float(self.config.cpsam_flow_threshold),
                cellprob_threshold=float(self.config.cpsam_cellprob_threshold),
                min_size=int(self.config.cpsam_min_size_px),
            )
            labels = np.asarray(masks, dtype=np.int32)
            n_objs = int(np.max(labels))

            # Multi-diameter sweep fallback if initial scale finds 0 objects
            sweep_diameters = self.config.cellpose_diameter_sweep or [150, 200, 250, 300]
            if n_objs == 0 and sweep_diameters:
                for sweep_d in sweep_diameters:
                    if abs(sweep_d - d_val) < 5.0:
                        continue
                    m_sw, _, _ = model.eval(
                        img_u8,
                        diameter=float(sweep_d),
                        channels=[0, 0],
                        flow_threshold=float(self.config.cpsam_flow_threshold),
                        cellprob_threshold=float(self.config.cpsam_cellprob_threshold),
                        min_size=int(self.config.cpsam_min_size_px),
                    )
                    n_sw = int(np.max(m_sw))
                    if n_sw > 0:
                        labels = np.asarray(m_sw, dtype=np.int32)
                        n_objs = n_sw
                        d_val = float(sweep_d)
                        break

            return labels, {
                "backend": "cpdino",
                "model_id": "cpdino",
                "device": self.device_str,
                "objects_found": n_objs,
                "diameter_used": d_val,
                "status": "success" if n_objs > 0 else "empty",
            }
        except Exception as e:
            logger.warning(f"Cellpose cpdino inference failed: {e}")
            return np.zeros((h, w), dtype=np.int32), {
                "backend": "cpdino",
                "model_id": "cpdino",
                "device": self.device_str,
                "error": str(e),
                "status": "failed",
                "objects_found": 0,
            }


# Backwards compatibility alias
CellposeCyto3Segmenter = CellposeFoundationSegmenter  # Deprecated name; the model is cpdino, not cyto3.

SUPPORTED_BACKENDS = ("cpsam", "cpdino", "classical")


class SegmentationHierarchy:
    """
    Coordinates the 3-tier segmentation hierarchy with disk caching, proportional downsampling,
    plausibility gating, decisions logging, and nearest-neighbor label mask upscaling.
    """

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
        decisions_logger: Optional[Union[DecisionsLogger, str, Path]] = None,
        mask_cache: Optional[Union[MaskCache, str, Path]] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.config = config if config is not None else SegmentationConfig()
        if isinstance(decisions_logger, (str, Path)):
            self.decisions_logger = DecisionsLogger(decisions_logger)
        elif decisions_logger is not None:
            self.decisions_logger = decisions_logger
        else:
            self.decisions_logger = DecisionsLogger(self.config.decisions_log_path)

        if isinstance(mask_cache, (str, Path)):
            self.mask_cache = MaskCache(mask_cache)
        elif mask_cache is not None:
            self.mask_cache = mask_cache
        else:
            self.mask_cache = MaskCache(self.config.cache_dir)

        self.preprocessor = preprocessor or Preprocessor()

        # Instantiate backends
        self.cpsam_segmenter = CellposeSAMSegmenter(self.config)
        self.cyto3_segmenter = CellposeCyto3Segmenter(self.config)
        self.classical_segmenter = ClassicalMultiScaleSegmenter(self.config)

    def _segmenter_for(self, backend: str) -> BaseSegmenter:
        """Return the (possibly test-substituted) segmenter instance for a backend name."""
        if backend == "cpsam":
            return self.cpsam_segmenter
        if backend == "cpdino":
            return self.cyto3_segmenter
        if backend == "classical":
            return self.classical_segmenter
        raise ValueError(f"Unknown segmentation backend {backend!r}; expected one of {SUPPORTED_BACKENDS}")

    def cache_key_payload(self, pixel_size_um: float) -> Dict[str, Any]:
        """Algorithmic parameters that determine the mask: segmentation + preprocessing + scale."""
        seg = {k: v for k, v in self.config.model_dump().items() if k not in NON_ALGORITHMIC_FIELDS}
        prep_cfg = getattr(self.preprocessor, "config", None)
        prep = prep_cfg.model_dump() if hasattr(prep_cfg, "model_dump") else {}
        # Output-only preprocessing switches do not change the mask.
        prep.pop("save_corrected_images", None)
        return {
            "segmentation": seg,
            "preprocess": prep,
            "pixel_size_um": round(float(pixel_size_um), 9),
        }

    def _check_plausibility(self, mask: np.ndarray) -> Tuple[bool, str]:
        """
        Validate whether segmentation output is plausible:
        1. Must contain at least 1 object.
        2. No single mask may cover > 25% of total field of view area.
        """
        h, w = mask.shape[:2]
        total_px = float(h * w)
        n_objs = int(np.max(mask))

        if n_objs == 0:
            return False, "0 objects found"

        max_allowed_px = self.config.single_mask_max_area_pct * total_px
        for obj_id in range(1, n_objs + 1):
            area = float(np.sum(mask == obj_id))
            if area > max_allowed_px:
                pct = (area / total_px) * 100.0
                return False, f"Single mask #{obj_id} occupies {pct:.1f}% FOV (exceeds {self.config.single_mask_max_area_pct * 100:.0f}% max limit)"

        return True, "Passed plausibility"

    def segment(
        self,
        image: np.ndarray,
        image_id: str,
        pixel_size_um: float = 1.518817,
        force_recompute: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main entry point for segmenting an image.
        Checks disk cache, downsamples, routes through 3-tier hierarchy, upscales,
        saves to cache, and logs decisions.
        """
        raw_h, raw_w = image.shape[:2]
        should_force = force_recompute if force_recompute is not None else self.config.force_recompute
        active_backend = self.config.primary_backend
        if active_backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown segmentation backend {active_backend!r}; expected one of {SUPPORTED_BACKENDS}")
        if active_backend == "cpsam":
            if isinstance(self.cpsam_segmenter, CellposeSAMSegmenter):
                # Loads and checksums the pinned weights once per run; raises if they are missing.
                self.cpsam_segmenter._init_model()
            model_ident = getattr(self.cpsam_segmenter, "model_sha256", None) or getattr(self.config, "cpsam_model_path", "cpsam_v2")
        elif active_backend == "cpdino":
            model_ident = "cpdino"
        else:
            model_ident = active_backend

        # 1. Content-addressed disk cache check. The key covers everything that changes the mask:
        # segmentation parameters, preprocessing parameters and the pixel size (which sets the
        # downsampling factor and the classical size gate).
        cache_key = self.mask_cache.get_key(
            image, self.cache_key_payload(pixel_size_um), model_hash=str(model_ident), image_id=image_id
        )
        if not should_force and self.mask_cache.has(cache_key):
            cached_mask = self.mask_cache.load(cache_key)
            if cached_mask is not None and cached_mask.shape[:2] == (raw_h, raw_w):
                n_objs = int(np.max(cached_mask))
                cached_meta = self.mask_cache.load_meta(cache_key)
                logger.debug(f"Cache hit for {image_id}: {n_objs} objects loaded from disk")
                hit_meta = {
                    "backend": cached_meta.get("backend", active_backend),
                    "model_id": cached_meta.get("model_id", active_backend),
                    "diameter_used": cached_meta.get("diameter_used", self.config.cpsam_expected_diameter_px),
                    "cached": True,
                    "objects_found": n_objs,
                    "image_id": image_id,
                }
                # Segmentation QC outcome is part of the cached result, so re-runs flag the same images.
                for k in ("plausibility_status", "qc_flag", "qc_reasons"):
                    if cached_meta.get(k) is not None:
                        hit_meta[k] = cached_meta[k]
                if hit_meta.get("qc_flag") == "SEG_FAIL":
                    hit_outcome = f"SEG_FAIL: {hit_meta.get('qc_reasons', 'implausible mask')} ({n_objs} objects)"
                else:
                    hit_outcome = f"SUCCESS: {n_objs} objects identified"
                self.decisions_logger.log(
                    image_id=image_id,
                    stage="Segmentation Hierarchy Result",
                    trigger_event="Cache hit",
                    action_taken=f"Loaded cached `{hit_meta['backend']}` mask",
                    outcome=hit_outcome,
                    backend_attempted=hit_meta["backend"],
                    details={"pixel_size_um": pixel_size_um, "objects_found": n_objs, "cached": True},
                )
                return cached_mask, hit_meta

        # 2. Preprocess & downsample
        downsampled_img, scale_factor, prep_meta = self.preprocessor.process(
            image, pixel_size_um=pixel_size_um,
        )
        effective_pixel_size = pixel_size_um / max(1e-6, float(scale_factor))

        active_backend = self.config.primary_backend
        selected_mask: Optional[np.ndarray] = None
        selected_meta: Dict[str, Any] = {}

        # 3. Primary Backend Execution
        mask_t1, meta_t1 = self._segmenter_for(active_backend).segment(downsampled_img, effective_pixel_size)

        if meta_t1.get("status") == "failed":
            # Per-image inference error: flag the image and never cache the empty placeholder mask.
            reason = f"inference error: {meta_t1.get('error', 'unknown')}"
            meta_t1 = {**meta_t1, "plausibility_status": "FAIL", "qc_flag": "SEG_FAIL", "qc_reasons": f"SEG_FAIL: {reason}"}
            full_res_mask = np.zeros((raw_h, raw_w), dtype=np.int32)
            self.decisions_logger.log(
                image_id=image_id,
                stage="Segmentation Hierarchy Result",
                trigger_event="Inference error",
                action_taken=f"Image flagged SEG_FAIL with `{active_backend}` backend",
                outcome=f"FAILED: {reason}",
                backend_attempted=active_backend,
                details={"pixel_size_um": pixel_size_um},
            )
            return full_res_mask, {**prep_meta, **meta_t1, "backend": active_backend, "final_backend": active_backend,
                                   "objects_found": 0, "image_id": image_id, "cached": False}

        is_valid, reason = self._check_plausibility(mask_t1)
        cacheable = True
        if is_valid:
            selected_mask = mask_t1
            selected_meta = meta_t1
        else:
            if self.config.fallback_policy == "per_image":
                logger.warning(
                    f"Image {image_id}: {active_backend} plausibility check failed ({reason}); attempting per-image fallback"
                )
                attempts = [f"{active_backend}: {reason}"]
                for tier_name, tier_backend in (
                    ("fallback", self.config.fallback_backend),
                    ("last_resort", self.config.last_resort_backend),
                ):
                    if tier_backend == active_backend or any(a.startswith(f"{tier_backend}:") for a in attempts):
                        continue
                    try:
                        mask_t, meta_t = self._segmenter_for(tier_backend).segment(downsampled_img, effective_pixel_size)
                    except Exception as e:
                        logger.warning(f"{tier_name} backend {tier_backend} failed for {image_id}: {e}")
                        attempts.append(f"{tier_backend}: error {e}")
                        continue
                    meta_t = {**meta_t, "backend": meta_t.get("backend", tier_backend)}
                    if meta_t.get("status") == "failed":
                        attempts.append(f"{tier_backend}: inference error {meta_t.get('error', 'unknown')}")
                        selected_mask, selected_meta = mask_t, meta_t
                        continue
                    ok_t, reason_t = self._check_plausibility(mask_t)
                    self.decisions_logger.log_fallback(
                        image_id=image_id, from_backend=active_backend, to_backend=tier_backend,
                        reason=reason, outcome="plausible" if ok_t else f"implausible: {reason_t}",
                    )
                    selected_mask, selected_meta = mask_t, meta_t
                    if ok_t:
                        break
                    attempts.append(f"{tier_backend}: {reason_t}")
                else:
                    # No tier produced a plausible mask: keep the last attempt but flag the image.
                    if selected_mask is None:
                        selected_mask, selected_meta = mask_t1, meta_t1
                    selected_meta = {
                        **selected_meta,
                        "plausibility_status": "FAIL",
                        "qc_flag": "SEG_FAIL",
                        "qc_reasons": "SEG_FAIL: no plausible mask from any tier (" + "; ".join(attempts) + ")",
                    }
                # Only primary-backend results are cached under the primary backend's key.
                cacheable = selected_meta.get("backend", active_backend) == active_backend
                if selected_meta.get("status") == "failed":
                    cacheable = False
            else:
                # Strict policy: one backend per run. Flag image, do NOT switch backend!
                selected_mask = mask_t1
                selected_meta = meta_t1
                selected_meta["plausibility_status"] = "FAIL"
                selected_meta["qc_flag"] = "SEG_FAIL"
                selected_meta["qc_reasons"] = f"SEG_FAIL: {reason}"

        if selected_mask is None:
            selected_mask = mask_t1
            selected_meta = meta_t1

        # 6. Upscale label mask to full raw resolution using nearest-neighbor interpolation
        full_res_mask = upscale_labels(selected_mask, (raw_h, raw_w))
        n_final = int(np.max(full_res_mask))

        # 7. Save to persistent 16-bit PNG disk cache
        try:
            cache_meta = {
                "backend": selected_meta.get("backend", active_backend),
                "model_id": selected_meta.get("model_id", active_backend),
                "diameter_used": selected_meta.get("diameter_used", self.config.cpsam_expected_diameter_px),
                "image_id": image_id,
                "cache_key": cache_key,
                "plausibility_status": selected_meta.get("plausibility_status"),
                "qc_flag": selected_meta.get("qc_flag"),
                "qc_reasons": selected_meta.get("qc_reasons"),
            }
            if cacheable:
                self.mask_cache.save(cache_key, full_res_mask, metadata=cache_meta)
            else:
                logger.debug(f"Not caching non-primary/failed result for {image_id}")
        except Exception as e:
            logger.warning(f"Could not cache mask for {image_id}: {e}")

        # Record final decision outcome
        final_backend = selected_meta.get("backend", active_backend)
        if selected_meta.get("qc_flag") == "SEG_FAIL":
            outcome = f"SEG_FAIL: {selected_meta.get('qc_reasons', 'implausible mask')} ({n_final} objects)"
        else:
            outcome = f"SUCCESS: {n_final} objects identified"
        self.decisions_logger.log(
            image_id=image_id,
            stage="Segmentation Hierarchy Result",
            trigger_event="Segmentation completed",
            action_taken=f"Segmented with `{final_backend}` backend",
            outcome=outcome,
            backend_attempted=final_backend,
            details={
                "pixel_size_um": pixel_size_um,
                "scale_factor": scale_factor,
                "raw_dimensions": f"{raw_w}x{raw_h}",
                "objects_found": n_final,
            },
        )

        final_meta = {
            **prep_meta,
            **selected_meta,
            "backend": final_backend,
            "final_backend": final_backend,
            "objects_found": n_final,
            "image_id": image_id,
            "cached": False,
        }

        return full_res_mask, final_meta
