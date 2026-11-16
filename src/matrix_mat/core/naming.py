"""
Standardized naming, timepoint parsing, and filename ingestion routines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


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


# Separator between filename tokens: one or more spaces and/or underscores.
_SEP = r"[ _]+"
_TAIL = r"gel[ _]?(?P<gel>\d+)[ _](?P<field>\d+)[ _]TRANS"
_FULL_RE = re.compile(rf"^(?:day|d)[ _]?(?P<day>\d+){_SEP}(?P<cond>.+?){_SEP}{_TAIL}$", re.IGNORECASE)
_DIR_RE = re.compile(rf"^(?P<cond>.+?){_SEP}{_TAIL}$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9-]+$")
_SD_RE = re.compile(r"^S(?P<s>\d+)(?:D(?P<d>\d+))?$", re.IGNORECASE)
# Copy suffixes added by file managers: "name (1)", "name copy", "name - Copy"
_COPY_RE = re.compile(r"(?:\s*\(\d+\)|[ _-]+copy(?:[ _]*\d+)?)$", re.IGNORECASE)

# Known control condition spelling (canonical case).
CONTROL_CONDITION = "Mat"


def normalise_condition(raw: str) -> str:
    """Return the canonical spelling of a condition token sequence or raise ValueError.

    ``Mat``/``mat``/``MAT`` -> ``Mat``; ``s34d30``/``S34 D30`` -> ``S34D30``; ``s50`` -> ``S50``.
    Any other single token (letters, digits, '-') is kept as-is; multi-token conditions that
    do not resolve to the ``S<n>D<n>`` pattern are ambiguous and rejected.
    """
    tokens = [t for t in re.split(r"[ _]+", raw.strip()) if t]
    if not tokens:
        raise ValueError("empty condition")
    joined = "".join(tokens)
    if len(tokens) == 1 and joined.lower() == CONTROL_CONDITION.lower():
        return CONTROL_CONDITION
    m = _SD_RE.match(joined)
    if m:
        return f"S{m.group('s')}" + (f"D{m.group('d')}" if m.group("d") is not None else "")
    if len(tokens) == 1 and _TOKEN_RE.match(joined):
        return joined
    raise ValueError(f"ambiguous or invalid condition {raw!r}")


def _record(stem: str, day_num: int, cond: str, gel: str, field: str) -> Dict[str, Any]:
    replicate = f"gel{int(gel)}"
    fov = f"{int(field):04d}"
    return {
        "image_id": stem,
        "timepoint": f"t{day_num}",
        "timepoint_day": day_num,
        "condition": cond,
        "replicate": replicate,
        "fov": fov,
        "pair_key": f"{cond}_{replicate}_{fov}",
        "flag_unmatched": False,
    }


def parse_image_filename(filename: str | Path, source_dir_timepoint: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses microscopy filename into structured metadata.

    Canonical pattern (whole stem must match; spaces or underscores separate tokens):
        Day<N> <Condition> gel<N>_<Field>_TRANS.tif
    Alternative pattern (timepoint supplied by ``source_dir_timepoint``):
        <Condition>_gel<N>_<Field>_TRANS.tif

    Normalisation: the control condition is spelled ``Mat`` and ``S<n>D<n>`` conditions are
    upper-cased (``S34 D30`` -> ``S34D30``); ``gel01`` -> ``gel1``; fields are zero-padded to
    four digits (``001`` -> ``0001``).

    Raises:
        ValueError if the filename does not match, carries a file-manager copy suffix such as
        ``(1)``, or its condition is ambiguous (e.g. ``Mat+drug``, ``Mat extra``).
    """
    stem = Path(filename).stem.strip()

    if _COPY_RE.search(stem):
        raise ValueError(
            f"Filename looks like a duplicated copy (e.g. ' (1)' / ' copy' suffix): '{filename}'. "
            "Rename or remove the duplicate so it cannot be paired twice."
        )

    m = _FULL_RE.match(stem)
    day_num: Optional[int] = None
    if m:
        day_num = int(m.group("day"))
    elif source_dir_timepoint is not None:
        m = _DIR_RE.match(stem)
        if m:
            day_num = parse_timepoint(source_dir_timepoint)

    if m is None or day_num is None:
        raise ValueError(f"Unmatched filename scheme for: '{filename}'")

    try:
        cond = normalise_condition(m.group("cond"))
    except ValueError as e:
        raise ValueError(f"Cannot parse condition unambiguously from '{filename}': {e}") from None
    return _record(stem, day_num, cond, m.group("gel"), m.group("field"))
