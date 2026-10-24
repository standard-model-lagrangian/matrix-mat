"""Deprecated shim for spheroid_if_sweep.comparator_prepost. Use matrix_mat.if_density.comparator_prepost instead."""
import warnings
from matrix_mat.if_density.comparator_prepost import *

warnings.warn(
    "spheroid_if_sweep.comparator_prepost is deprecated, use matrix_mat.if_density.comparator_prepost instead",
    DeprecationWarning,
    stacklevel=2,
)
