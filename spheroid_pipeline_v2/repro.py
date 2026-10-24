"""Deprecated shim for spheroid_pipeline_v2.repro. Use matrix_mat.core.repro instead."""
import warnings
from matrix_mat.core.repro import *

warnings.warn(
    "spheroid_pipeline_v2.repro is deprecated, use matrix_mat.core.repro instead",
    DeprecationWarning,
    stacklevel=2,
)
