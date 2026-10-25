"""
matrix_mat.core: Pure, dependency-light foundational modules.
Includes physical units, scale extraction, naming parsing, hierarchical statistics,
content-addressed cache, provenance, paths resolution, and determinism.
"""

from matrix_mat.core import units
from matrix_mat.core import scale
from matrix_mat.core import naming
from matrix_mat.core import stats
from matrix_mat.core import cache
from matrix_mat.core import provenance
from matrix_mat.core import paths
from matrix_mat.core import repro

__all__ = [
    "units",
    "scale",
    "naming",
    "stats",
    "cache",
    "provenance",
    "paths",
    "repro",
]
