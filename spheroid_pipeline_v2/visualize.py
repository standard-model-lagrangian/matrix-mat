"""Deprecated shim for spheroid_pipeline_v2.visualize. Use matrix_mat.reporting.visualize instead."""
import warnings
from matrix_mat.reporting.visualize import *

warnings.warn(
    "spheroid_pipeline_v2.visualize is deprecated, use matrix_mat.reporting.visualize instead",
    DeprecationWarning,
    stacklevel=2,
)
