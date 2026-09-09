"""
Cellpose-SAM (cpsam) execution engine with hardware acceleration, determinism, and 2D/3D mode detection.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tifffile
import torch
import yaml

from cellpose import models
from spheroid_if_sweep.pairing import FieldRecord

logger = logging.getLogger("spheroid_if_sweep.engine")


def setup_run_logger(run_dir: Path) -> logging.FileHandler:
    """Attach a per-run FileHandler logging to run_dir/log.txt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.getLogger("spheroid_if_sweep").addHandler(file_handler)
    return file_handler


def get_torch_device() -> Tuple[bool, str]:
    """Detect PyTorch accelerator device (CUDA, MPS, CPU)."""
    if torch.cuda.is_available():
        return True, "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return True, "mps"
    return False, "cpu"


class CPSAMEngine:
    """Cellpose-SAM inference manager."""

    def __init__(
        self,
        model_path: str = "models/cpsam_v2",
        random_seed: int = 42,
    ):
        self.model_path = Path(model_path).resolve()
        self.random_seed = random_seed
        self.use_gpu, self.device_str = get_torch_device()
        self._model: Optional[models.CellposeModel] = None

    def load_model(self) -> models.CellposeModel:
        """Instantiate and cache the Cellpose-SAM model."""
        if self._model is None:
            logger.info(
                f"Loading Cellpose-SAM from '{self.model_path}' on device '{self.device_str}' (use_gpu={self.use_gpu})"
            )
            if self.model_path.exists():
                models.MODEL_DIR = self.model_path.parent
                self._model = models.CellposeModel(
                    pretrained_model=str(self.model_path),
                    gpu=self.use_gpu,
                )
            else:
                logger.warning(
                    f"Model path {self.model_path} not found. Attempting model_type='cpsam' download..."
                )
                self._model = models.CellposeModel(
                    model_type="cpsam",
                    gpu=self.use_gpu,
                )
        return self._model

    def segment_image(
        self,
        image: np.ndarray,
        params: Dict[str, Any],
        image_id: str = "",
    ) -> Tuple[np.ndarray, str]:
        """
        Run segmentation with the given hyperparameter dictionary.
        Returns (masks, mode_used).
        """
        # Set seeds for deterministic flow simulation
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        model = self.load_model()

        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0 and image.dtype in (np.float32, np.float64):
                img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_u8 = image

        is_3d = (img_u8.ndim >= 3 and img_u8.shape[0] > 1)
        do_3d = bool(params.get("do_3D", False))
        stitch_threshold = float(params.get("stitch_threshold", 0.0))

        if not is_3d:
            if do_3d or stitch_threshold > 0:
                mode_used = "2D (3D/stitch requested but input is single-plane 2D; skipped volumetric per spec)"
                logger.info(f"[{image_id}] Image is single-plane 2D ({img_u8.shape}); skipping 3D volumetric mode and stitching per spec.")
            else:
                mode_used = "2D"
            eval_do_3d = False
            eval_stitch = 0.0
        else:
            mode_used = "3D Volumetric" if do_3d else "2D-per-plane stitched"
            eval_do_3d = do_3d
            eval_stitch = stitch_threshold

        flow_threshold = float(params.get("flow_threshold", 0.4))
        cellprob_threshold = float(params.get("cellprob_threshold", 0.0))
        diameter = params.get("diameter", None)
        if diameter is not None:
            diameter = float(diameter)
        min_size = int(params.get("min_size", 15))

        masks, flows, styles = model.eval(
            img_u8,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            stitch_threshold=eval_stitch,
            do_3D=eval_do_3d,
        )

        labels = np.asarray(masks, dtype=np.int32)
        return labels, mode_used
