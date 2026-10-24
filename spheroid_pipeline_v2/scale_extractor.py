"""Deprecated shim for spheroid_pipeline_v2.scale_extractor. Use matrix_mat.core.scale instead."""
import warnings
from matrix_mat.core.scale import *

warnings.warn(
    "spheroid_pipeline_v2.scale_extractor is deprecated, use matrix_mat.core.scale instead",
    DeprecationWarning,
    stacklevel=2,
)
