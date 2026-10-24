"""
matrix_mat.reporting: Visualization, figures, presentation plots, and markdown reporting.
Modules must only format and display pre-computed statistical outputs and never compute
statistical tests directly or import scipy.stats.
"""

from matrix_mat.reporting import (
    biological_plots,
    presentation_plots,
    report,
    summaries,
    visualize,
)

__all__ = [
    "biological_plots",
    "presentation_plots",
    "report",
    "summaries",
    "visualize",
]
