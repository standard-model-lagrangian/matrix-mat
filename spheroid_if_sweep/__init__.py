"""
Legacy spheroid_if_sweep package shim.
All production functionality has been migrated to matrix_mat.
"""
import warnings

warnings.warn(
    "spheroid_if_sweep is deprecated, use matrix_mat instead",
    DeprecationWarning,
    stacklevel=2,
)

from matrix_mat.if_density.sweep import *
from matrix_mat.if_density.engine import *
from matrix_mat.if_density.pairing import *
from matrix_mat.if_density.features import *
from matrix_mat.if_density.postprocess import *
from matrix_mat.if_density.preprocess import *
from matrix_mat.if_density.qc import *
from matrix_mat.if_density.comparator import *
from matrix_mat.if_density.comparator_prepost import *
from matrix_mat.if_density.config import *
from matrix_mat.if_density.preflight import *
from matrix_mat.if_density.provenance import *
from matrix_mat.if_density.visualize import *
from matrix_mat.reporting.biological_plots import *
from matrix_mat.reporting.summaries import *
