"""Deprecated shim for spheroid_if_sweep.cli. Use matrix_mat.if_density.sweep instead."""
import warnings
from matrix_mat.if_density.sweep import *

warnings.warn(
    "spheroid_if_sweep.cli is deprecated, use matrix_mat.if_density.sweep instead",
    DeprecationWarning,
    stacklevel=2,
)
