"""Deprecated shim for spheroid_pipeline_v2.stats. Use matrix_mat.brightfield.stats instead."""
import warnings
from matrix_mat.brightfield.stats import *

warnings.warn(
    "spheroid_pipeline_v2.stats is deprecated, use matrix_mat.brightfield.stats instead",
    DeprecationWarning,
    stacklevel=2,
)
