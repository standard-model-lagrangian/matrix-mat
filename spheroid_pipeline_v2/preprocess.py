"""Deprecated shim for spheroid_pipeline_v2.preprocess. Use matrix_mat.brightfield.preprocess instead."""
import warnings
from matrix_mat.brightfield.preprocess import *

warnings.warn(
    "spheroid_pipeline_v2.preprocess is deprecated, use matrix_mat.brightfield.preprocess instead",
    DeprecationWarning,
    stacklevel=2,
)
