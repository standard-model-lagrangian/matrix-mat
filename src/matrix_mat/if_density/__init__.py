"""
matrix_mat.if_density: Immunofluorescence nuclear density pipeline.
Includes preprocessing, Cellpose-SAM inference engine, postprocessing (size gating,
touching split, border exclusion), feature extraction (areal density), QC gating,
sweep evaluation, and cross-run comparator analysis.
"""

from matrix_mat.if_density import (
    comparator,
    comparator_biological,
    comparator_prepost,
    config,
    engine,
    features,
    pairing,
    postprocess,
    preprocess,
    qc,
    sweep,
)

__all__ = [
    "comparator",
    "comparator_biological",
    "comparator_prepost",
    "config",
    "engine",
    "features",
    "pairing",
    "postprocess",
    "preprocess",
    "qc",
    "sweep",
]
