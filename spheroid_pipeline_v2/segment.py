"""
Segmentation Module for Spheroid Volume Pipeline v2 (spheroid_pipeline_v2).
Provides 3-tier segmentation hierarchy:
  1. Primary: Cellpose-SAM ('cpsam') with MPS (Apple M4) / CUDA / CPU fallback
  2. Fallback: Cellpose ('cyto3' with diameter sweep)
  3. Last Resort / Self-Contained Engine: Classical multi-scale adaptive watershed + annular contrast gating
With persistent disk caching (masks/<image_id>.png), decision logging, and nearest-neighbor mask upscaling.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure, segmentation
from skimage.feature import peak_local_max

from spheroid_pipeline_v2.config import SegmentationConfig
from spheroid_pipeline_v2.decisions import DecisionsLogger
from spheroid_pipeline_v2.mask_cache import MaskCache
from spheroid_pipeline_v2.preprocess import Preprocessor, upscale_labels

logger = logging.getLogger("spheroid_pipeline_v2.segment")


def get_optimal_device() -> Tuple[bool, str]:
    """
    Detect optimal PyTorch computing device:
    - Apple Silicon MPS (macOS M-series) with PYTORCH_ENABLE_MPS_FALLBACK=1
    - NVIDIA CUDA
    - CPU fallback
    Returns (use_gpu_bool, device_name_str).
    """
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
        window_sizes = self.config.classical_adaptive_windows
        thresh_diff = self.config.classical_threshold_diff

        for w_size in window_sizes:
            if w_size >= min(h, w):
                continue
            k = int(w_size) if int(w_size) % 2 == 1 else int(w_size) + 1
            blur = cv2.blur(smooth, (k, k))
            diff = cv2.subtract(blur, smooth)
            _, th_dark = cv2.threshold(diff, thresh_diff, 255, cv2.THRESH_BINARY)
            combined_mask = cv2.bitwise_or(combined_mask, th_dark)

        # 2. Relative local background constraint (spheroids are strictly darker than median)
        med_val = np.median(smooth)
        combined_mask[smooth > int(med_val * 0.98)] = 0

        # 3. Morphological closing to seal internal pores
        close_r = max(2, int(self.config.morph_close_radius))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2 + 1, close_r * 2 + 1))
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

        # 4. SciPy binary hole filling
        filled = ndimage.binary_fill_holes(closed > 0)

        # 5. Morphological opening to eliminate fine background noise
        open_r = max(1, int(self.config.morph_open_radius))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_r * 2 + 1, open_r * 2 + 1))
        opened = cv2.morphologyEx((filled * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel_open)

        # 6. Connected components size and border margin filtering
        min_d_um = 40.0
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
        all_mask = (raw_labels > 0).astype(np.uint8)
        kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

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

            # Keep only if at least 10% darker than background
            if contrast >= 0.10:
                kept_count += 1
                clean_labels[obj_mask > 0] = kept_count

        meta = {
            "backend": "classical",
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
        self.use_gpu, self.device_str = get_optimal_device()
        self._model = None

    def _init_model(self) -> Any:
        if self._model is None:
            from cellpose import models
            from pathlib import Path
            local_model = Path("models/cpsam_v2").resolve()
            if local_model.exists():
                models.MODEL_DIR = local_model.parent
                self._model = models.CellposeModel(
                    pretrained_model=str(local_model),
                    gpu=self.use_gpu,
                )
            else:
                self._model = models.CellposeModel(
                    model_type="cpsam",
                    gpu=self.use_gpu,
                )
        return self._model

    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        try:
            model = self._init_model()
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
                "device": self.device_str,
                "objects_found": n_objs,
                "diameter_used": d_val,
                "status": "success" if n_objs > 0 else "empty",
            }
        except Exception as e:
            logger.warning(f"Cellpose-SAM inference failed: {e}")
            return np.zeros((h, w), dtype=np.int32), {
                "backend": "cpsam",
                "device": self.device_str,
                "error": str(e),
                "status": "failed",
                "objects_found": 0,
            }


class CellposeFoundationSegmenter(BaseSegmenter):
    """
    Secondary deep learning fallback backend using Cellpose foundation vision transformer ('cpdino' / 'cpsam').
    Performs multi-diameter scale sweep ([150, 200, 250, 300] px) if base scale yields 0 objects.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config if config is not None else SegmentationConfig()
        self.use_gpu, self.device_str = get_optimal_device()
        self._model = None

    def _init_model(self) -> Any:
        if self._model is None:
            from cellpose import models
            self._model = models.CellposeModel(
                model_type="cpdino",
                gpu=self.use_gpu,
            )
        return self._model

    def segment(self, image: np.ndarray, pixel_size_um: float = 1.518817) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = image.shape[:2]
        try:
            model = self._init_model()
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
                "device": self.device_str,
                "objects_found": n_objs,
                "diameter_used": d_val,
                "status": "success" if n_objs > 0 else "empty",
            }
        except Exception as e:
            logger.warning(f"Cellpose cpdino inference failed: {e}")
            return np.zeros((h, w), dtype=np.int32), {
                "backend": "cpdino",
                "device": self.device_str,
                "error": str(e),
                "status": "failed",
                "objects_found": 0,
            }


# Backwards compatibility alias
CellposeCyto3Segmenter = CellposeFoundationSegmenter


class SegmentationHierarchy:
    """
    Coordinates the 3-tier segmentation hierarchy with disk caching, proportional downsampling,
    plausibility gating, decisions logging, and nearest-neighbor label mask upscaling.
    """

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
        decisions_logger: Optional[DecisionsLogger] = None,
        mask_cache: Optional[MaskCache] = None,
        preprocessor: Optional[Preprocessor] = None,
    ):
        self.config = config if config is not None else SegmentationConfig()
        self.decisions_logger = decisions_logger or DecisionsLogger(self.config.decisions_log_path)
        self.mask_cache = mask_cache or MaskCache(self.config.cache_dir)
        self.preprocessor = preprocessor or Preprocessor()

        # Instantiate backends
        self.cpsam_segmenter = CellposeSAMSegmenter(self.config)
        self.cyto3_segmenter = CellposeCyto3Segmenter(self.config)
        self.classical_segmenter = ClassicalMultiScaleSegmenter(self.config)

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

        # 1. Disk cache check
        if not should_force and self.mask_cache.has(image_id):
            cached_mask = self.mask_cache.load(image_id)
            if cached_mask is not None and cached_mask.shape[:2] == (raw_h, raw_w):
                n_objs = int(np.max(cached_mask))
                logger.debug(f"Cache hit for {image_id}: {n_objs} objects loaded from disk")
                return cached_mask, {
                    "backend": "disk_cache",
                    "cached": True,
                    "objects_found": n_objs,
                    "image_id": image_id,
                }

        # 2. Preprocess & downsample
        downsampled_img, scale_factor, prep_meta = self.preprocessor.process(
            image, pixel_size_um=pixel_size_um,
        )

        active_backend = self.config.primary_backend
        selected_mask: Optional[np.ndarray] = None
        selected_meta: Dict[str, Any] = {}

        # 3. Tier 1 Execution (Primary: Cellpose-SAM, cpdino, or classical)
        if active_backend == "classical":
            mask_t1, meta_t1 = self.classical_segmenter.segment(downsampled_img, pixel_size_um)
            selected_mask = mask_t1
            selected_meta = meta_t1
        elif active_backend == "cpdino":
            try:
                mask_t2, meta_t2 = self.cyto3_segmenter.segment(downsampled_img, pixel_size_um)
                is_valid, reason = self._check_plausibility(mask_t2)
                if is_valid:
                    selected_mask = mask_t2
                    selected_meta = meta_t2
                else:
                    self.decisions_logger.log_fallback(
                        image_id=image_id,
                        from_backend="cpdino",
                        to_backend=self.config.last_resort_backend,
                        reason=f"Cellpose cpdino plausibility check failed: {reason}",
                    )
            except Exception as e:
                self.decisions_logger.log_fallback(
                    image_id=image_id,
                    from_backend="cpdino",
                    to_backend=self.config.last_resort_backend,
                    reason=f"Cellpose cpdino exception: {e}",
                )
        else:
            try:
                mask_t1, meta_t1 = self.cpsam_segmenter.segment(downsampled_img, pixel_size_um)
                is_valid, reason = self._check_plausibility(mask_t1)
                if is_valid:
                    selected_mask = mask_t1
                    selected_meta = meta_t1
                elif meta_t1.get("status") == "empty":
                    # CPSAM evaluated successfully and found 0 objects; check Tier 2
                    logger.debug(f"{image_id}: CPSAM found 0 objects, checking Tier 2")
                else:
                    self.decisions_logger.log_fallback(
                        image_id=image_id,
                        from_backend="cpsam",
                        to_backend=self.config.fallback_backend,
                        reason=f"Cellpose-SAM plausibility check failed: {reason}",
                    )
            except Exception as e:
                self.decisions_logger.log_fallback(
                    image_id=image_id,
                    from_backend="cpsam",
                    to_backend=self.config.fallback_backend,
                    reason=f"Cellpose-SAM exception: {e}",
                )

            # 4. Tier 2 Execution (Fallback: Cellpose cpdino)
            if selected_mask is None:
                try:
                    mask_t2, meta_t2 = self.cyto3_segmenter.segment(downsampled_img, pixel_size_um)
                    is_valid, reason = self._check_plausibility(mask_t2)
                    if is_valid:
                        selected_mask = mask_t2
                        selected_meta = meta_t2
                    elif meta_t2.get("status") == "empty":
                        # Both deep learning models found 0 objects; fall through to Classical Tier 3
                        logger.debug(f"{image_id}: Both DL backends returned 0 objects; falling through to classical Tier 3")
                        self.decisions_logger.log_fallback(
                            image_id=image_id,
                            from_backend="cpdino",
                            to_backend=self.config.last_resort_backend,
                            reason="Both DL backends returned 0 objects; attempting classical adaptive watershed",
                        )
                    else:
                        self.decisions_logger.log_fallback(
                            image_id=image_id,
                            from_backend="cpdino",
                            to_backend=self.config.last_resort_backend,
                            reason=f"Cellpose cpdino plausibility check failed: {reason}",
                        )
                except Exception as e:
                    self.decisions_logger.log_fallback(
                        image_id=image_id,
                        from_backend="cpdino",
                        to_backend=self.config.last_resort_backend,
                        reason=f"Cellpose cpdino exception: {e}",
                    )

        # 5. Tier 3 Execution (Last Resort: Classical Adaptive Watershed)
        if selected_mask is None:
            mask_t3, meta_t3 = self.classical_segmenter.segment(downsampled_img, pixel_size_um)
            selected_mask = mask_t3
            selected_meta = meta_t3

        # 6. Upscale label mask to full raw resolution using nearest-neighbor interpolation
        full_res_mask = upscale_labels(selected_mask, (raw_h, raw_w))
        n_final = int(np.max(full_res_mask))

        # 7. Save to persistent 16-bit PNG disk cache
        try:
            self.mask_cache.save(image_id, full_res_mask)
        except Exception as e:
            logger.warning(f"Could not cache mask for {image_id}: {e}")

        # Record final decision outcome
        final_backend = selected_meta.get("backend", "unknown")
        self.decisions_logger.log(
            image_id=image_id,
            stage="Segmentation Hierarchy Result",
            trigger_event="Segmentation completed",
            action_taken=f"Segmented with `{final_backend}` backend",
            outcome=f"SUCCESS: {n_final} objects identified",
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
            "final_backend": final_backend,
            "objects_found": n_final,
            "image_id": image_id,
            "cached": False,
        }

        return full_res_mask, final_meta
