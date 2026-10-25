"""Deprecated shim for spheroid_pipeline_v2.measure. Use matrix_mat.brightfield.measure instead."""
import warnings
from matrix_mat.brightfield.measure import *

warnings.warn(
    "spheroid_pipeline_v2.measure is deprecated, use matrix_mat.brightfield.measure instead",
    DeprecationWarning,
    stacklevel=2,
)
