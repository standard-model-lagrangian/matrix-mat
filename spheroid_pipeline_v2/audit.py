"""Deprecated shim for spheroid_pipeline_v2.audit. Use matrix_mat.brightfield.audit instead."""
import warnings
from matrix_mat.brightfield.audit import *

warnings.warn(
    "spheroid_pipeline_v2.audit is deprecated, use matrix_mat.brightfield.audit instead",
    DeprecationWarning,
    stacklevel=2,
)
