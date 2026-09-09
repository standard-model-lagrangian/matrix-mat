"""
Architectural & Runtime Decision Logging Module.
Records backend attempts, successes, fallbacks, and parameter decisions to `DECISIONS.md`
in structured GitHub-Flavored Markdown format.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("spheroid_pipeline_v2.decisions")


class DecisionsLogger:
    """Logs runtime architectural and fallback decisions to a persistent markdown file."""

    def __init__(self, log_path: Union[str, Path] = "DECISIONS.md"):
        self.log_path = Path(log_path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Initialize DECISIONS.md with title and explanatory metadata if it does not exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            header_content = (
                "# Pipeline Architectural & Fallback Decisions Log\n\n"
                "This document records all runtime segmentation backend attempts, "
                "plausibility rejections, fallbacks, and parameter tuning events.\n\n"
                "---\n\n"
            )
            try:
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.write(header_content)
            except Exception as e:
                logger.warning(f"Could not create DECISIONS.md header: {e}")

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
        """
        Append a structured decision record to DECISIONS.md.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

        lines = [
            f"### [{now_str}] {stage} — `{image_id}`\n",
            f"- **Image ID**: `{image_id}`\n",
        ]
        if backend_attempted:
            lines.append(f"- **Backend Attempted**: `{backend_attempted}`\n")
        lines.append(f"- **Trigger / Event**: {trigger_event}\n")
        lines.append(f"- **Action Taken**: {action_taken}\n")
        lines.append(f"- **Outcome**: {outcome}\n")

        if details:
            if isinstance(details, dict):
                lines.append("- **Details**:\n")
                for k, v in details.items():
                    lines.append(f"  - `{k}`: {v}\n")
            else:
                lines.append(f"- **Details**: {details}\n")

        lines.append("\n---\n\n")

        entry = "".join(lines)

        try:
            self._ensure_header()
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.debug(f"Decision logged for {image_id}: {action_taken}")
        except Exception as e:
            logger.warning(f"Failed to append to DECISIONS.md at {self.log_path}: {e}")

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
        self.log(
            image_id=image_id,
            stage="Segmentation Hierarchy Fallback",
            backend_attempted=from_backend,
            trigger_event=reason,
            action_taken=f"Switched from `{from_backend}` to `{to_backend}`",
            outcome=outcome,
            details=extra_meta,
        )
