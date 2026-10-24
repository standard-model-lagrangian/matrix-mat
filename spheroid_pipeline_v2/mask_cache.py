"""Deprecated shim for spheroid_pipeline_v2.mask_cache. Use matrix_mat.core.cache instead."""
import warnings
from matrix_mat.core.cache import *

warnings.warn(
    "spheroid_pipeline_v2.mask_cache is deprecated, use matrix_mat.core.cache instead",
    DeprecationWarning,
    stacklevel=2,
)
