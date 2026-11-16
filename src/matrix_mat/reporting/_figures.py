"""Shared figure helpers for matrix_mat.reporting."""

from __future__ import annotations

import functools
from typing import Any, Callable, List, Optional, Sequence, TypeVar, cast

import matplotlib.pyplot as plt

F = TypeVar("F", bound=Callable[..., Any])


def closes_new_figures(func: F) -> F:
    """Close every matplotlib figure the wrapped function opened, even when it raises."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        before = set(plt.get_fignums())
        try:
            return func(*args, **kwargs)
        finally:
            for num in set(plt.get_fignums()) - before:
                plt.close(num)

    return cast(F, wrapper)


# Known condition / timepoint display order; anything else found in the data follows, sorted.
KNOWN_CONDITION_ORDER: Sequence[str] = ("Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50")


def ordered_values(values: Any, known: Optional[Sequence[str]] = None) -> List[Any]:
    """Unique non-null values of ``values``: known order first, then the rest sorted."""
    known = list(known if known is not None else KNOWN_CONDITION_ORDER)
    present: List[Any] = []
    for v in values:
        if v is None or (isinstance(v, float) and v != v) or v in present:
            continue
        present.append(v)
    head = [k for k in known if k in present]
    tail = sorted((v for v in present if v not in head), key=lambda x: str(x))
    return head + tail


def timepoint_order(values: Any) -> List[Any]:
    """Unique timepoints ordered by their day number (t0, t7, t10, ...), unparseable last."""
    from matrix_mat.core.naming import parse_timepoint

    uniq = ordered_values(values, known=[])

    def key(v: Any) -> Any:
        try:
            return (0, parse_timepoint(v), str(v))
        except ValueError:
            return (1, 0, str(v))

    return sorted(uniq, key=key)
