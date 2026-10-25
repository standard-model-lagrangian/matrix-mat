"""Deprecated shim for spheroid_if_sweep.postprocess. Use matrix_mat.if_density.postprocess instead."""
import warnings
from matrix_mat.if_density.postprocess import *

warnings.warn(
    "spheroid_if_sweep.postprocess is deprecated, use matrix_mat.if_density.postprocess instead",
    DeprecationWarning,
    stacklevel=2,
)
