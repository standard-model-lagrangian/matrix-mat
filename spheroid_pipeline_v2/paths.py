"""Deprecated shim for spheroid_pipeline_v2.paths. Use matrix_mat.core.paths instead."""
import warnings
from matrix_mat.core.paths import *

warnings.warn(
    "spheroid_pipeline_v2.paths is deprecated, use matrix_mat.core.paths instead",
    DeprecationWarning,
    stacklevel=2,
)
