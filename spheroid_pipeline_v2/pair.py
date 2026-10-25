"""Deprecated shim for spheroid_pipeline_v2.pair. Use matrix_mat.brightfield.pair instead."""
import warnings
from matrix_mat.brightfield.pair import *

warnings.warn(
    "spheroid_pipeline_v2.pair is deprecated, use matrix_mat.brightfield.pair instead",
    DeprecationWarning,
    stacklevel=2,
)
