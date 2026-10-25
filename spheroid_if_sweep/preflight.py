"""Deprecated shim for spheroid_if_sweep.preflight. Use matrix_mat.if_density.preflight instead."""
import warnings
from matrix_mat.if_density.preflight import *

warnings.warn(
    "spheroid_if_sweep.preflight is deprecated, use matrix_mat.if_density.preflight instead",
    DeprecationWarning,
    stacklevel=2,
)
