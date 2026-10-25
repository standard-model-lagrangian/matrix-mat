"""Deprecated shim for spheroid_pipeline_v2.presentation_plots. Use matrix_mat.reporting.presentation_plots instead."""
import warnings
from matrix_mat.reporting.presentation_plots import *

warnings.warn(
    "spheroid_pipeline_v2.presentation_plots is deprecated, use matrix_mat.reporting.presentation_plots instead",
    DeprecationWarning,
    stacklevel=2,
)
