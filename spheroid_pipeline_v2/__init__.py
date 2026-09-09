"""
Spheroid Volume Pipeline v2 (spheroid_pipeline_v2)
End-to-end multi-spheroid instance segmentation, robust background contrast gating,
denominator gating, temporal tracking, statistical hypothesis testing, and self-verification
pipeline for brightfield hydrogel spheroid cultures.
"""

from spheroid_pipeline_v2.config import (
    ArtifactConfig,
    PairingConfig,
    PipelineConfig,
    PreprocessConfig,
    QCConfig,
    QCGatingConfig,
    SegmentationConfig,
    StatsConfig,
)
from spheroid_pipeline_v2.decisions import DecisionsLogger
from spheroid_pipeline_v2.mask_cache import MaskCache
from spheroid_pipeline_v2.measure import (
    OBJECTS_CSV_COLUMNS,
    SpheroidObjectRecord,
    compute_circularity,
    compute_ellipsoidal_volume,
    compute_spherical_volume,
    compute_volume_discrepancy_ratio,
    evaluate_morphological_qc,
    measure_single_object,
    save_objects_csv,
)
from spheroid_pipeline_v2.pair import (
    PAIRS_CSV_COLUMNS,
    SpheroidPairRecord,
    SpheroidPairer,
    pair_dataset,
    pair_fov_objects,
    pair_timepoints,
    save_pairs_csv,
    update_objects_with_pair_ids,
)
from spheroid_pipeline_v2.preprocess import (
    Preprocessor,
    compute_downsample_scale,
    downsample_image,
    flatten_illumination,
    normalize_percentiles,
    upscale_labels,
)
from spheroid_pipeline_v2.qc_filter import (
    QCFilter,
    check_image_plausibility,
    compute_annular_ring_contrast,
    filter_and_measure_objects,
    resolve_overlaps,
)
from spheroid_pipeline_v2.report import (
    ReportGenerator,
    df_to_markdown_table,
)
from spheroid_pipeline_v2.review import (
    VALIDATION_CSV_COLUMNS,
    ReviewManager,
)
from spheroid_pipeline_v2.run_pipeline import (
    build_manifest,
    evaluate_sample_12_sanity_gates,
    parse_image_filename,
    run_spheroid_pipeline,
    run_synthetic_smoke_test,
)
from spheroid_pipeline_v2.scale_extractor import (
    ScaleExtractor,
    ScaleInfo,
    extract_pixel_size,
)
from spheroid_pipeline_v2.segment import (
    BaseSegmenter,
    CellposeCyto3Segmenter,
    CellposeSAMSegmenter,
    ClassicalMultiScaleSegmenter,
    SegmentationHierarchy,
    get_optimal_device,
)
from spheroid_pipeline_v2.stats import (
    CONDITION_SUMMARY_COLUMNS,
    PopulationStatsCalculator,
    bootstrap_ci_median,
    compute_condition_statistics,
    compute_exclusion_audit,
    format_condition_summary_table,
    generate_exclusion_audit_markdown,
    generate_exclusion_audit_table,
    get_significance_stars,
    permutation_test_median,
    save_condition_summary_csv,
)
from spheroid_pipeline_v2.visualize import (
    CONDITION_COLORS,
    QC_COLORS_BGR,
    QC_COLORS_RGB,
    Visualizer,
    create_contact_sheet,
    create_overlay,
    generate_all_figures,
    generate_condition_contact_sheets,
    plot_circularity_sensitivity_audit,
    plot_fold_change_violin,
    plot_growth_slopegraph,
    plot_v0_vs_v7_scatter,
)

__version__ = "2.0.0"

__all__ = [
    "__version__",
    # Config
    "PipelineConfig",
    "PreprocessConfig",
    "SegmentationConfig",
    "QCConfig",
    "QCGatingConfig",
    "PairingConfig",
    "StatsConfig",
    "ArtifactConfig",
    # Scale Extractor
    "ScaleExtractor",
    "ScaleInfo",
    "extract_pixel_size",
    # Preprocessor
    "Preprocessor",
    "flatten_illumination",
    "normalize_percentiles",
    "compute_downsample_scale",
    "downsample_image",
    "upscale_labels",
    # Segmentation
    "BaseSegmenter",
    "ClassicalMultiScaleSegmenter",
    "CellposeSAMSegmenter",
    "CellposeCyto3Segmenter",
    "SegmentationHierarchy",
    "get_optimal_device",
    # Mask Cache
    "MaskCache",
    # Decisions Logger
    "DecisionsLogger",
    # QC & Filtering
    "QCFilter",
    "filter_and_measure_objects",
    "resolve_overlaps",
    "compute_annular_ring_contrast",
    "check_image_plausibility",
    # Measure & Morphometrics
    "SpheroidObjectRecord",
    "OBJECTS_CSV_COLUMNS",
    "compute_spherical_volume",
    "compute_ellipsoidal_volume",
    "compute_volume_discrepancy_ratio",
    "compute_circularity",
    "evaluate_morphological_qc",
    "measure_single_object",
    "save_objects_csv",
    # Pairing & Tracking
    "SpheroidPairRecord",
    "PAIRS_CSV_COLUMNS",
    "SpheroidPairer",
    "pair_timepoints",
    "pair_fov_objects",
    "pair_dataset",
    "update_objects_with_pair_ids",
    "save_pairs_csv",
    # Stats & Hypothesis Testing
    "CONDITION_SUMMARY_COLUMNS",
    "PopulationStatsCalculator",
    "bootstrap_ci_median",
    "permutation_test_median",
    "get_significance_stars",
    "compute_condition_statistics",
    "compute_exclusion_audit",
    "generate_exclusion_audit_table",
    "generate_exclusion_audit_markdown",
    "format_condition_summary_table",
    "save_condition_summary_csv",
    # Visualization & Artifacts
    "Visualizer",
    "create_overlay",
    "create_contact_sheet",
    "generate_condition_contact_sheets",
    "generate_all_figures",
    "plot_fold_change_violin",
    "plot_v0_vs_v7_scatter",
    "plot_growth_slopegraph",
    "plot_circularity_sensitivity_audit",
    "CONDITION_COLORS",
    "QC_COLORS_BGR",
    "QC_COLORS_RGB",
    # Reporting
    "ReportGenerator",
    "df_to_markdown_table",
    # Review & Validation
    "ReviewManager",
    "VALIDATION_CSV_COLUMNS",
    # Pipeline Orchestration & CLI
    "run_spheroid_pipeline",
    "run_synthetic_smoke_test",
    "evaluate_sample_12_sanity_gates",
    "build_manifest",
    "parse_image_filename",
]
