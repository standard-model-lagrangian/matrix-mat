"""Deprecated shim for spheroid_pipeline_v2.decisions. Use matrix_mat.brightfield.decisions instead."""
import warnings
from matrix_mat.brightfield.decisions import *

warnings.warn(
    "spheroid_pipeline_v2.decisions is deprecated, use matrix_mat.brightfield.decisions instead",
    DeprecationWarning,
    stacklevel=2,
)
