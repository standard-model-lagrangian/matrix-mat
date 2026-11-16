"""Deprecated shim for spheroid_if_sweep.sweep. Use matrix_mat.if_density.sweep instead."""
import warnings
from matrix_mat.if_density.sweep import *

warnings.warn(
    "spheroid_if_sweep.sweep is deprecated, use matrix_mat.if_density.sweep instead",
    DeprecationWarning,
    stacklevel=2,
)
