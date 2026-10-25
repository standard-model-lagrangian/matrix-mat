"""Deprecated shim for spheroid_if_sweep.preprocess. Use matrix_mat.if_density.preprocess instead."""
import warnings
from matrix_mat.if_density.preprocess import *

warnings.warn(
    "spheroid_if_sweep.preprocess is deprecated, use matrix_mat.if_density.preprocess instead",
    DeprecationWarning,
    stacklevel=2,
)
