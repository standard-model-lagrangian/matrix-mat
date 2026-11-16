"""Deprecated shim for spheroid_if_sweep.comparator_biological. Use matrix_mat.if_density.comparator_biological instead."""
import warnings
from matrix_mat.if_density.comparator_biological import *

warnings.warn(
    "spheroid_if_sweep.comparator_biological is deprecated, use matrix_mat.if_density.comparator_biological instead",
    DeprecationWarning,
    stacklevel=2,
)
