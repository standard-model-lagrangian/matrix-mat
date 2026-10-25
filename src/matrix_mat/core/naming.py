"""
Standardized naming, timepoint parsing, and filename ingestion routines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


def parse_timepoint(s: Any) -> int:
    """
    Parse a timepoint identifier into an integer day number.
    
    Accepts:
        'Day0', 'Day 0', 'D0', 'd0', 't0', 'T0', '0' -> 0
        'Day7', 'Day 7', 'D7', 'd7', 't7', 'T7', '7' -> 7
        'Day10', 'D10', 't10', '10'                  -> 10
        
    Raises:
        ValueError if the format is unrecognized, ambiguous, or malformed.
    """
    if s is None:
        raise ValueError("Timepoint cannot be None")
    if isinstance(s, (int,)):
        return int(s)
    
    text = str(s).strip()
    # Match Day<N>, D<N>, t<N>, or bare integer <N>
    m = re.fullmatch(r"(?:(?:day|d|t)\s*(\d+)|(\d+))", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized timepoint format: '{s}'")
    
    day_str = m.group(1) if m.group(1) is not None else m.group(2)
    return int(day_str)


def is_t0(s: Any) -> bool:
    """Return True if timepoint represents Day 0 / t0."""
    return parse_timepoint(s) == 0


def parse_image_filename(filename: str | Path, source_dir_timepoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses microscopy filename into structured metadata:
    Canonical Pattern: Day<0|7|10> <Condition> gel<N>_<Field:04d>_TRANS.tif
    Alternative Pattern: <Condition>_gel<N>_<Field>_TRANS.tif (with source_dir_timepoint)
    
    Raises:
        ValueError if filename does not match any recognized experimental scheme.
    """
    stem = Path(filename).stem
    
    # 1. Full standard pattern with embedded timepoint
    p_full = re.compile(
        r"(?:Day|D)\s*(?P<day>\d+)\s+(?P<condition>[A-Za-z0-9_-]+)\s+gel(?P<gel>\d+)_(?P<field>\d+)_TRANS",
        re.IGNORECASE,
    )
    m = p_full.search(stem)
    if m:
        day_num = int(m.group("day"))
        condition = m.group("condition")
        replicate = f"gel{m.group('gel')}"
        fov = m.group("field")
        return {
            "image_id": stem,
            "timepoint": f"t{day_num}",
            "timepoint_day": day_num,
            "condition": condition,
            "replicate": replicate,
            "fov": fov,
            "pair_key": f"{condition}_{replicate}_{fov}",
            "flag_unmatched": False,
        }
    
    # 2. Pattern where timepoint is omitted from filename but supplied by source directory
    if source_dir_timepoint is not None:
        p_dir = re.compile(
            r"(?P<condition>[A-Za-z0-9_-]+)[_\s]+gel(?P<gel>\d+)_(?P<field>\d+)_TRANS",
            re.IGNORECASE,
        )
        m_dir = p_dir.search(stem)
        if m_dir:
            day_num = parse_timepoint(source_dir_timepoint)
            condition = m_dir.group("condition")
            replicate = f"gel{m_dir.group('gel')}"
            fov = m_dir.group("field")
            return {
                "image_id": stem,
                "timepoint": f"t{day_num}",
                "timepoint_day": day_num,
                "condition": condition,
                "replicate": replicate,
                "fov": fov,
                "pair_key": f"{condition}_{replicate}_{fov}",
                "flag_unmatched": False,
            }

    # If completely unmatched, raise ValueError per WS1.7 specification
    raise ValueError(f"Unmatched filename scheme for: '{filename}'")
