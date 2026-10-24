"""Deprecated shim for spheroid_if_sweep.features. Use matrix_mat.if_density.features instead."""
import warnings
from matrix_mat.if_density.features import *

warnings.warn(
    "spheroid_if_sweep.features is deprecated, use matrix_mat.if_density.features instead",
    DeprecationWarning,
    stacklevel=2,
)
