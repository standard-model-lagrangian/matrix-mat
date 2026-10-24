"""Deprecated shim for spheroid_if_sweep.provenance. Use matrix_mat.if_density.provenance instead."""
import warnings
from matrix_mat.if_density.provenance import *

warnings.warn(
    "spheroid_if_sweep.provenance is deprecated, use matrix_mat.if_density.provenance instead",
    DeprecationWarning,
    stacklevel=2,
)
