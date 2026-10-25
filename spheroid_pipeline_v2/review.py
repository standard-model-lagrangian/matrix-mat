"""Deprecated shim for spheroid_pipeline_v2.review. Use matrix_mat.brightfield.review instead."""
import warnings
from matrix_mat.brightfield.review import *

warnings.warn(
    "spheroid_pipeline_v2.review is deprecated, use matrix_mat.brightfield.review instead",
    DeprecationWarning,
    stacklevel=2,
)
