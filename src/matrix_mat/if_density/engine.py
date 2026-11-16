"""
Cellpose-SAM (cpsam) execution engine with hardware acceleration, determinism, and 2D/3D mode detection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


logger = logging.getLogger("matrix_mat.if_density.engine")

# Per-run log files capture everything the package logs during the run (if_density, core, reporting).
RUN_LOG_LOGGER = "matrix_mat"


def setup_run_logger(run_dir: Path) -> logging.FileHandler:
    """
    Attach a per-run FileHandler (run_dir/log.txt) to the package logger only. Child loggers propagate
    to it, so each record is written once. Pair with teardown_run_logger in a finally block.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    pkg_logger = logging.getLogger(RUN_LOG_LOGGER)
    # The run log records INFO even when the console is quieter; the previous level is restored on teardown.
    file_handler._previous_level = pkg_logger.level  # type: ignore[attr-defined]
    if pkg_logger.getEffectiveLevel() > logging.INFO:
        pkg_logger.setLevel(logging.INFO)
    pkg_logger.addHandler(file_handler)
    return file_handler


def teardown_run_logger(handler: logging.Handler) -> None:
    """Detach a handler created by setup_run_logger from the logger it was attached to and close it."""
    pkg_logger = logging.getLogger(RUN_LOG_LOGGER)
    pkg_logger.removeHandler(handler)
    prev = getattr(handler, "_previous_level", None)
    if prev is not None:
        pkg_logger.setLevel(prev)
    handler.close()


def get_torch_device(use_gpu: Optional[bool] = None) -> Tuple[bool, str]:
    """
    Detect PyTorch accelerator device (CUDA, MPS, CPU). Lazy to avoid eager torch loading.
    use_gpu=False forces CPU without importing torch.
    """
    if use_gpu is False:
        return False, "cpu"
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
        use_gpu: Optional[bool] = None,
        model_type: str = "cpsam",
    ):
        if model_type != "cpsam":
            raise ValueError(f"model_type '{model_type}' is not supported; only 'cpsam' (Cellpose-SAM) is")
        self.model_path = Path(model_path).resolve()
        self.model_type = model_type
        self.random_seed = random_seed
        self.requested_gpu = use_gpu
        self.use_gpu, self.device_str = get_torch_device(use_gpu)
        if use_gpu and not self.use_gpu:
            logger.warning("model.use_gpu=true but no CUDA/MPS accelerator is available; running on CPU")
        logger.info(
            f"Cellpose-SAM engine: device={self.device_str} (use_gpu requested={use_gpu}, model_type={model_type})"
        )
        self._model: Optional[Any] = None
        self._weights_sha256: Optional[str] = None

    @property
    def weights_sha256(self) -> str:
        """SHA-256 of the weights file this engine loads (verified against models.lock.json once)."""
        if self._weights_sha256 is None:
            from matrix_mat.core.repro import verify_model_weights
            self._weights_sha256 = verify_model_weights(self.model_path, model_name="cpsam_v2")
        return self._weights_sha256

    def load_model(self) -> Any:
        """Instantiate and cache the Cellpose-SAM model."""
        if self._model is None:
            logger.info(
                f"Loading Cellpose-SAM from '{self.model_path}' on device '{self.device_str}' (use_gpu={self.use_gpu})"
            )
            if self.model_path.exists():
                from cellpose import models
                _ = self.weights_sha256  # verifies the checksum once
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

        if bool(params.get("do_3D", False)):
            raise ValueError("do_3D=true is not supported: the IF pipeline is 2D only")
        # RGB(A) is converted to grayscale explicitly (never read as a 3-plane z-stack); stacks are refused.
        from matrix_mat.if_density.preprocess import ensure_2d_grayscale, to_uint8
        img_u8 = to_uint8(ensure_2d_grayscale(image, image_id))

        stitch_threshold = float(params.get("stitch_threshold", 0.0))
        if stitch_threshold > 0:
            mode_used = "2D (stitch_threshold ignored: single-plane input)"
        else:
            mode_used = "2D"
        eval_do_3d = False
        eval_stitch = 0.0

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
    "teardown_run_logger",
    "get_torch_device",
    "CPSAMEngine",
]
