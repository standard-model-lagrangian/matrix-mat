"""Deprecated shim for spheroid_if_sweep.engine. Use matrix_mat.if_density.engine instead."""
import warnings
from matrix_mat.if_density.engine import *

warnings.warn(
    "spheroid_if_sweep.engine is deprecated, use matrix_mat.if_density.engine instead",
    DeprecationWarning,
    stacklevel=2,
)
