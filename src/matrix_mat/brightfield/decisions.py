"""
Architectural & Runtime Decision Logging Module for matrix-mat brightfield.
Records backend attempts, successes, fallbacks, and parameter decisions to `decisions.jsonl`
in structured JSON Lines format per run.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from matrix_mat.core.paths import resolve_output_root

logger = logging.getLogger("matrix_mat.brightfield.decisions")


class DecisionsLogger:
    """Logs runtime architectural and fallback decisions to a per-run decisions.jsonl file."""

    def __init__(
        self,
        log_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        if output_dir is not None:
            self.jsonl_path = Path(output_dir) / "decisions.jsonl"
            self.log_path = self.jsonl_path
        elif log_path is not None:
            self.log_path = Path(log_path)
            if self.log_path.suffix == ".jsonl":
                self.jsonl_path = self.log_path
            else:
                self.jsonl_path = self.log_path.parent / "decisions.jsonl"
        elif "MATRIX_MAT_DECISIONS_PATH" in os.environ:
            env_path = Path(os.environ["MATRIX_MAT_DECISIONS_PATH"])
            self.log_path = env_path
            self.jsonl_path = env_path.parent / "decisions.jsonl" if env_path.suffix != ".jsonl" else env_path
        else:
            out_root = resolve_output_root()
            self.jsonl_path = out_root / "decisions.jsonl"
            self.log_path = self.jsonl_path

    def log_decision(
        self,
        image_id: str,
        step: str,
        from_backend: Optional[str] = None,
        to_backend: Optional[str] = None,
        reason: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[Union[Dict[str, Any], str]] = None,
    ) -> None:
        """Append a structured JSON line decision record to decisions.jsonl."""
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record = {
            "timestamp": now_str,
            "image_id": image_id,
            "step": step,
            "from_backend": from_backend,
            "to_backend": to_backend,
            "reason": reason,
            "outcome": outcome,
            "details": details,
        }
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            logger.debug(f"Decision logged for {image_id}: {step} -> {to_backend}")
        except Exception as e:
            logger.warning(f"Failed to append to decisions.jsonl at {self.jsonl_path}: {e}")

    def log(
        self,
        image_id: str,
        stage: str,
        trigger_event: str,
        action_taken: str,
        outcome: str,
        backend_attempted: Optional[str] = None,
        details: Optional[Union[Dict[str, Any], str]] = None,
    ) -> None:
        """Append decision record matching legacy API."""
        self.log_decision(
            image_id=image_id,
            step=stage,
            from_backend=backend_attempted,
            to_backend=action_taken,
            reason=trigger_event,
            outcome=outcome,
            details=details,
        )

    def log_fallback(
        self,
        image_id: str,
        from_backend: str,
        to_backend: str,
        reason: str,
        outcome: str = "Fallback backend engaged",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience wrapper for logging segmentation backend fallback events."""
        self.log_decision(
            image_id=image_id,
            step="segmentation_fallback",
            from_backend=from_backend,
            to_backend=to_backend,
            reason=reason,
            outcome=outcome,
            details=extra_meta,
        )
