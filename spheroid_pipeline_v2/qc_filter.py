"""Deprecated shim for spheroid_pipeline_v2.qc_filter. Use matrix_mat.brightfield.qc_filter instead."""
import warnings
from matrix_mat.brightfield.qc_filter import *

warnings.warn(
    "spheroid_pipeline_v2.qc_filter is deprecated, use matrix_mat.brightfield.qc_filter instead",
    DeprecationWarning,
    stacklevel=2,
)
