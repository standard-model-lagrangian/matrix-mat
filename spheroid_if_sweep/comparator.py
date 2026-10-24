"""Deprecated shim for spheroid_if_sweep.comparator. Use matrix_mat.if_density.comparator instead."""
import warnings
from matrix_mat.if_density.comparator import *

warnings.warn(
    "spheroid_if_sweep.comparator is deprecated, use matrix_mat.if_density.comparator instead",
    DeprecationWarning,
    stacklevel=2,
)
