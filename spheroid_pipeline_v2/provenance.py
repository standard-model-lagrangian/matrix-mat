"""Deprecated shim for spheroid_pipeline_v2.provenance. Use matrix_mat.core.provenance instead."""
import warnings
from matrix_mat.core.provenance import *

warnings.warn(
    "spheroid_pipeline_v2.provenance is deprecated, use matrix_mat.core.provenance instead",
    DeprecationWarning,
    stacklevel=2,
)

import matrix_mat.core.provenance as _core_prov


def write_run_metadata(
    output_dir,
    config,
    wall_time_sec=0.0,
    warnings=None,
    extra_info=None,
    weights_sha256=None,
    backend=None,
    pipeline_name="spheroid_pipeline_v2",
):
    return _core_prov.write_run_metadata(
        output_dir=output_dir,
        config=config,
        wall_time_sec=wall_time_sec,
        warnings=warnings,
        extra_info=extra_info,
        weights_sha256=weights_sha256,
        backend=backend,
        pipeline_name=pipeline_name,
    )

