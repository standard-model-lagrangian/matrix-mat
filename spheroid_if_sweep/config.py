"""Deprecated shim for spheroid_if_sweep.config. Use matrix_mat.if_density.config instead."""
import warnings
from matrix_mat.if_density.config import *

warnings.warn(
    "spheroid_if_sweep.config is deprecated, use matrix_mat.if_density.config instead",
    DeprecationWarning,
    stacklevel=2,
)
