"""Deprecated shim for spheroid_pipeline_v2.report. Use matrix_mat.reporting.report instead."""
import warnings
from matrix_mat.reporting.report import *

warnings.warn(
    "spheroid_pipeline_v2.report is deprecated, use matrix_mat.reporting.report instead",
    DeprecationWarning,
    stacklevel=2,
)
