"""Deprecated shim for spheroid_pipeline_v2.segment. Use matrix_mat.brightfield.segment instead."""
import warnings
from matrix_mat.brightfield.segment import *

warnings.warn(
    "spheroid_pipeline_v2.segment is deprecated, use matrix_mat.brightfield.segment instead",
    DeprecationWarning,
    stacklevel=2,
)
