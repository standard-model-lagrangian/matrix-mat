"""Deprecated shim for spheroid_pipeline_v2.preflight. Use matrix_mat.brightfield.preflight instead."""
import warnings
from matrix_mat.brightfield.preflight import *

warnings.warn(
    "spheroid_pipeline_v2.preflight is deprecated, use matrix_mat.brightfield.preflight instead",
    DeprecationWarning,
    stacklevel=2,
)
