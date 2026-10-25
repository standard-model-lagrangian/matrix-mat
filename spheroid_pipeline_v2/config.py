"""Deprecated shim for spheroid_pipeline_v2.config. Use matrix_mat.brightfield.config instead."""
import warnings
from matrix_mat.brightfield.config import *

warnings.warn(
    "spheroid_pipeline_v2.config is deprecated, use matrix_mat.brightfield.config instead",
    DeprecationWarning,
    stacklevel=2,
)
