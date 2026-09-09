"""
Configuration module for Spheroid Volume Pipeline v2 (spheroid_pipeline_v2).
Provides validated dataclasses, YAML loader, CLI override merging, and parameter defaults.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("spheroid_pipeline_v2.config")


@dataclass
class PreprocessConfig:
    """Configuration for illumination flattening, normalization, and downsampling."""
    flat_field_sigma: float = 0.0
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    target_diameter_px: float = 200.0
    target_diameter_min_px: int = 150
    target_diameter_max_px: int = 250
    expected_spheroid_diameter_um: float = 300.0
    save_corrected_images: bool = True


@dataclass
class SegmentationConfig:
    """Configuration for 3-tier segmentation hierarchy, devices, and caching."""
    primary_backend: str = "cpsam"  # 'cpsam' | 'cpdino' | 'classical'
    backend: str = "cpsam"          # YAML alias for primary_backend
    fallback_backend: str = "cpdino"
    last_resort_backend: str = "classical"
    cpsam_model_path: str = "models/cpsam_v2"
    cpsam_expected_diameter_px: float = 160.0
    cpsam_flow_threshold: float = 0.4
    cpsam_cellprob_threshold: float = 0.0
    cpsam_min_size_px: int = 40
    adaptive_window_sizes: List[int] = field(default_factory=lambda: [31, 61, 121, 241])
    classical_adaptive_windows: List[int] = field(default_factory=lambda: [31, 61, 121, 241])
    classical_threshold_diff: int = 8
    cellpose_diameter_sweep: List[int] = field(default_factory=lambda: [150, 200, 250, 300])
    cache_dir: str = "masks"
    enable_mps_fallback: bool = True
    force_recompute: bool = False
    morph_open_radius: int = 2
    morph_close_radius: int = 4
    watershed_min_distance_px: int = 25
    single_mask_max_area_pct: float = 0.25
    border_margin_px: int = 15
    decisions_log_path: str = "DECISIONS.md"
    hough_fallback_enabled: bool = True
    hough_min_radius_um: float = 25.0
    hough_max_radius_um: float = 600.0

    def __post_init__(self):
        if self.backend and self.backend != self.primary_backend:
            self.primary_backend = self.backend
        if self.adaptive_window_sizes and not self.classical_adaptive_windows:
            self.classical_adaptive_windows = self.adaptive_window_sizes
        elif self.classical_adaptive_windows and not self.adaptive_window_sizes:
            self.adaptive_window_sizes = self.classical_adaptive_windows


@dataclass
class QCConfig:
    """Configuration for morphological, contrast, and size quality control gating."""
    min_d_um: float = 40.0
    max_d_um: float = 1500.0
    min_pass_diameter_um: float = 40.0
    max_pass_diameter_um: float = 1200.0
    border_margin_px: int = 15
    ring_width_px: int = 15
    ring_contrast_factor: float = 0.90
    interior_contrast_pct: float = 0.10  # 1.0 - ring_contrast_factor
    background_ring_dilation_px: int = 15
    single_mask_max_area_pct: float = 0.25
    max_single_mask_area_fraction: float = 0.25  # YAML alias
    min_circularity: float = 0.65
    min_solidity: float = 0.85
    max_eccentricity: float = 0.82
    max_objects_per_fov: int = 0

    def __post_init__(self):
        if self.ring_contrast_factor and not self.interior_contrast_pct:
            self.interior_contrast_pct = 1.0 - self.ring_contrast_factor
        if self.max_single_mask_area_fraction:
            self.single_mask_max_area_pct = self.max_single_mask_area_fraction


# Backward compatibility alias
QCGatingConfig = QCConfig


@dataclass
class PairingConfig:
    """Configuration for Hungarian bipartite tracking and denominator gating."""
    max_centroid_shift_fraction: float = 1.5
    min_displacement_px: float = 50.0
    max_displacement_px_floor: float = 50.0  # Alias for min_displacement_px
    min_t0_d_um_denominator: float = 60.0
    min_denominator_d0_um: float = 60.0      # Alias
    min_fold_change: float = 0.10
    max_fold_change: float = 20.0
    fold_change_min_review: float = 0.10     # Alias
    fold_change_max_review: float = 20.0     # Alias
    fallback_greedy: bool = True

    def __post_init__(self):
        if self.min_displacement_px:
            self.max_displacement_px_floor = self.min_displacement_px
        if self.min_denominator_d0_um:
            self.min_t0_d_um_denominator = self.min_denominator_d0_um
        if self.min_fold_change:
            self.fold_change_min_review = self.min_fold_change
        if self.max_fold_change:
            self.fold_change_max_review = self.max_fold_change


@dataclass
class StatsConfig:
    """Configuration for statistical aggregation, bootstrapping, and hypothesis testing."""
    control_condition: str = "Mat"
    min_denominator_d0_um: float = 60.0
    min_fold_change: float = 0.10
    max_fold_change: float = 20.0
    bootstrap_iterations: int = 2000
    permutation_iterations: int = 10000
    confidence_level: float = 0.95
    include_review_in_stats: bool = False


@dataclass
class ArtifactConfig:
    """Configuration for visual and reporting output artifacts."""
    overlay_dir: str = "overlays"
    contact_sheets_dir: str = "contact_sheets"
    figures_dir: str = "figures"
    reports_dir: str = "reports"
    save_pdf: bool = True
    save_png: bool = True
    contact_sheet_cols: int = 4
    contact_sheet_dpi: int = 150
    figure_dpi: int = 300


def _deep_update_config(target: Any, overrides: Dict[str, Any]) -> None:
    """Recursively updates dataclass fields from nested or flat dictionaries."""
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, dict) and hasattr(target, k):
            sub_target = getattr(target, k)
            if hasattr(sub_target, "__annotations__"):
                _deep_update_config(sub_target, v)
            elif isinstance(sub_target, dict):
                sub_target.update(v)
            else:
                setattr(target, k, v)
        elif hasattr(target, k):
            setattr(target, k, v)
        else:
            # Check nested sub-configs
            matched = False
            for attr_name in ("preprocess", "segmentation", "qc", "pairing", "stats", "artifacts"):
                if hasattr(target, attr_name):
                    sub = getattr(target, attr_name)
                    if hasattr(sub, k):
                        setattr(sub, k, v)
                        matched = True
                        break
            if not matched:
                logger.warning(f"Unrecognized configuration override: {k}={v}")


@dataclass
class PipelineConfig:
    """Master pipeline configuration schema."""
    t0_dir: str = "Experimental data /Chuling cells/Spheroid Day0 260805"
    t7_dir: str = "Experimental data /Chuling cells/Spheroid D7 260812"
    output_dir: str = "output"
    review_dir: str = "review"
    default_pixel_size_um: float = 1.518817
    cli_pixel_size_um: Optional[float] = None
    random_seed: int = 42
    sample_n: Optional[int] = None
    condition: Optional[str] = None

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, cli_overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """Load configuration from YAML file, filling defaults and applying CLI overrides."""
        yaml_path = Path(yaml_path)
        data: Dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Config file {yaml_path} not found. Using default parameters.")

        preprocess_data = data.get("preprocess", {})
        seg_data = data.get("segmentation", {})
        qc_data = data.get("qc", data.get("qc_gating", {}))
        pairing_data = data.get("pairing", {})
        stats_data = data.get("stats", {})
        artifacts_data = data.get("artifacts", {})

        cfg = cls(
            t0_dir=data.get("t0_dir", cls.t0_dir),
            t7_dir=data.get("t7_dir", cls.t7_dir),
            output_dir=data.get("output_dir", cls.output_dir),
            review_dir=data.get("review_dir", cls.review_dir),
            default_pixel_size_um=float(data.get("default_pixel_size_um", cls.default_pixel_size_um)),
            cli_pixel_size_um=float(data["cli_pixel_size_um"]) if data.get("cli_pixel_size_um") is not None else None,
            random_seed=int(data.get("random_seed", cls.random_seed)),
            sample_n=int(data["sample_n"]) if data.get("sample_n") is not None else None,
            condition=data.get("condition"),
            preprocess=PreprocessConfig(**{k: v for k, v in preprocess_data.items() if k in PreprocessConfig.__annotations__}),
            segmentation=SegmentationConfig(**{k: v for k, v in seg_data.items() if k in SegmentationConfig.__annotations__}),
            qc=QCConfig(**{k: v for k, v in qc_data.items() if k in QCConfig.__annotations__}),
            pairing=PairingConfig(**{k: v for k, v in pairing_data.items() if k in PairingConfig.__annotations__}),
            stats=StatsConfig(**{k: v for k, v in stats_data.items() if k in StatsConfig.__annotations__}),
            artifacts=ArtifactConfig(**{k: v for k, v in artifacts_data.items() if k in ArtifactConfig.__annotations__}),
        )

        if cli_overrides:
            _deep_update_config(cfg, cli_overrides)

        return cfg

    def to_yaml(self, output_path: str | Path) -> None:
        """Serialize configuration to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
