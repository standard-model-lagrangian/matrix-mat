"""Deprecated shim for spheroid_pipeline_v2.units. Use matrix_mat.core.units instead."""
import warnings
from matrix_mat.core.units import *

warnings.warn(
    "spheroid_pipeline_v2.units is deprecated, use matrix_mat.core.units instead",
    DeprecationWarning,
    stacklevel=2,
)
