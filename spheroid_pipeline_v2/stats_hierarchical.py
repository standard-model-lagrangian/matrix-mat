"""Deprecated shim for spheroid_pipeline_v2.stats_hierarchical. Use matrix_mat.core.stats instead."""
import warnings
from matrix_mat.core.stats import *

warnings.warn(
    "spheroid_pipeline_v2.stats_hierarchical is deprecated, use matrix_mat.core.stats instead",
    DeprecationWarning,
    stacklevel=2,
)
