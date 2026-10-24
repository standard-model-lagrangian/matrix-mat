"""Deprecated shim for spheroid_if_sweep.summaries. Use matrix_mat.reporting.summaries instead."""
import warnings
from matrix_mat.reporting.summaries import *

warnings.warn(
    "spheroid_if_sweep.summaries is deprecated, use matrix_mat.reporting.summaries instead",
    DeprecationWarning,
    stacklevel=2,
)
