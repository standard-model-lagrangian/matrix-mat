"""Deprecated shim for spheroid_pipeline_v2.run_pipeline. Use matrix_mat.brightfield.run_pipeline instead."""
import warnings
from matrix_mat.brightfield.run_pipeline import *

warnings.warn(
    "spheroid_pipeline_v2.run_pipeline is deprecated, use matrix_mat.brightfield.run_pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)
