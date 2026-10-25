"""Deprecated shim for spheroid_if_sweep.qc. Use matrix_mat.if_density.qc instead."""
import warnings
from matrix_mat.if_density.qc import *

warnings.warn(
    "spheroid_if_sweep.qc is deprecated, use matrix_mat.if_density.qc instead",
    DeprecationWarning,
    stacklevel=2,
)
