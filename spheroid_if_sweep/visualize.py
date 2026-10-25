"""Deprecated shim for spheroid_if_sweep.visualize. Use matrix_mat.if_density.visualize instead."""
import warnings
from matrix_mat.if_density.visualize import *

warnings.warn(
    "spheroid_if_sweep.visualize is deprecated, use matrix_mat.if_density.visualize instead",
    DeprecationWarning,
    stacklevel=2,
)
