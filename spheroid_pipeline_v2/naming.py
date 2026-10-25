"""Deprecated shim for spheroid_pipeline_v2.naming. Use matrix_mat.core.naming instead."""
import warnings
from matrix_mat.core.naming import *

warnings.warn(
    "spheroid_pipeline_v2.naming is deprecated, use matrix_mat.core.naming instead",
    DeprecationWarning,
    stacklevel=2,
)
