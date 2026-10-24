"""Deprecated shim for spheroid_if_sweep.biological_plots. Use matrix_mat.reporting.biological_plots instead."""
import warnings
from matrix_mat.reporting.biological_plots import *

warnings.warn(
    "spheroid_if_sweep.biological_plots is deprecated, use matrix_mat.reporting.biological_plots instead",
    DeprecationWarning,
    stacklevel=2,
)
