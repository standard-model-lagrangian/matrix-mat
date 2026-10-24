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
import yaml

from matrix_mat.if_density.pairing import FieldRecord

logger = logging.getLogger("matrix_mat.if_density.engine")


def setup_run_logger(run_dir: Path) -> logging.FileHandler:
    """Attach a per-run FileHandler logging to run_dir/log.txt."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logging.getLogger("matrix_mat.if_density").addHandler(file_handler)
    return file_handler


def get_torch_device() -> Tuple[bool, str]:
    """Detect PyTorch accelerator device (CUDA, MPS, CPU). Lazy to avoid eager torch loading."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            return True, "mps"
    except ImportError:
        pass
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
        self._model: Optional[Any] = None

    def load_model(self) -> Any:
        """Instantiate and cache the Cellpose-SAM model."""
        if self._model is None:
            logger.info(
                f"Loading Cellpose-SAM from '{self.model_path}' on device '{self.device_str}' (use_gpu={self.use_gpu})"
            )
            if self.model_path.exists():
                from matrix_mat.core.repro import verify_model_weights
                from cellpose import models
                verify_model_weights(self.model_path, model_name="cpsam_v2")
                models.MODEL_DIR = self.model_path.parent
                self._model = models.CellposeModel(
                    pretrained_model=str(self.model_path),
                    gpu=self.use_gpu,
                )
            else:
                raise FileNotFoundError(
                    f"Model weights for 'cpsam_v2' not found at '{self.model_path}'. "
                    f"Run 'python tools/fetch_models.py' to download and verify pinned weights."
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
        from matrix_mat.core.repro import set_determinism
        set_determinism(self.random_seed)

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


__all__ = [
    "setup_run_logger",
    "get_torch_device",
    "CPSAMEngine",
]
