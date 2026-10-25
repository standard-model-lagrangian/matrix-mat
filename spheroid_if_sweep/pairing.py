"""Deprecated shim for spheroid_if_sweep.pairing. Use matrix_mat.if_density.pairing instead."""
import warnings
from matrix_mat.if_density.pairing import *

warnings.warn(
    "spheroid_if_sweep.pairing is deprecated, use matrix_mat.if_density.pairing instead",
    DeprecationWarning,
    stacklevel=2,
)
