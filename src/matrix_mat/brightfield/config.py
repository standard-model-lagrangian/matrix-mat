"""
Configuration module for Brightfield Spheroid Volume Analysis Pipeline (matrix_mat.brightfield).
Provides validated Pydantic dataclasses with extra="forbid", YAML loader, CLI override merging,
parameter defaults, and strict alias deprecation & conflict checking.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, model_validator

from matrix_mat.core.paths import (
    resolve_data_root,
    resolve_output_root,
    resolve_models_root,
    resolve_cache_root,
)

logger = logging.getLogger("matrix_mat.brightfield.config")


@dataclass(config=ConfigDict(extra="forbid"))
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

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
class SegmentationConfig:
    """Configuration for 3-tier segmentation hierarchy, devices, and caching."""
    primary_backend: str = "cpsam"  # 'cpsam' | 'cpdino' | 'classical'
    backend: Optional[str] = field(default=None, repr=False)  # Deprecated YAML alias
    fallback_policy: str = "strict"        # 'strict' (one backend per run) | 'per_image' (opt-in exploratory)
    fallback_backend: str = "cpdino"
    last_resort_backend: str = "classical"
    cpsam_model_path: str = field(default_factory=lambda: str(resolve_models_root() / "cpsam_v2"))
    cpsam_expected_diameter_px: float = 160.0
    cpsam_flow_threshold: float = 0.4
    cpsam_cellprob_threshold: float = 0.0
    cpsam_min_size_px: int = 40
    adaptive_window_sizes: List[int] = field(default_factory=lambda: [31, 61, 121, 241])
    classical_adaptive_windows: Optional[List[int]] = field(default=None, repr=False)  # Deprecated alias
    classical_threshold_diff: int = 8
    cellpose_diameter_sweep: List[int] = field(default_factory=lambda: [150, 200, 250, 300])
    cache_dir: str = field(default_factory=lambda: str(resolve_cache_root()))
    enable_mps_fallback: bool = True
    force_recompute: bool = False
    morph_open_radius: int = 2
    morph_close_radius: int = 4
    watershed_min_distance_px: int = 25
    single_mask_max_area_pct: float = 0.25
    border_margin_px: int = 15
    device: str = "cpu"
    decisions_log_path: str = field(default_factory=lambda: str(resolve_output_root() / "decisions.jsonl"))

    @model_validator(mode="before")
    @classmethod
    def _validate_segmentation_aliases(cls, data: Any) -> Any:
        kwargs = data.kwargs if hasattr(data, "kwargs") and data.kwargs is not None else (data if isinstance(data, dict) else {})
        # Alias: backend -> primary_backend
        if "backend" in kwargs and kwargs["backend"] is not None:
            warnings.warn(
                "'backend' is deprecated; use 'primary_backend' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "primary_backend" in kwargs and kwargs["primary_backend"] is not None:
                if kwargs["primary_backend"] != kwargs["backend"]:
                    raise ValueError(
                        f"Conflicting values for canonical 'primary_backend' ({kwargs['primary_backend']!r}) "
                        f"and alias 'backend' ({kwargs['backend']!r})"
                    )
            else:
                kwargs["primary_backend"] = kwargs["backend"]

        # Alias: classical_adaptive_windows -> adaptive_window_sizes
        if "classical_adaptive_windows" in kwargs and kwargs["classical_adaptive_windows"] is not None:
            warnings.warn(
                "'classical_adaptive_windows' is deprecated; use 'adaptive_window_sizes' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "adaptive_window_sizes" in kwargs and kwargs["adaptive_window_sizes"] is not None:
                if kwargs["adaptive_window_sizes"] != kwargs["classical_adaptive_windows"]:
                    raise ValueError(
                        "Conflicting values for canonical 'adaptive_window_sizes' and alias 'classical_adaptive_windows'"
                    )
            else:
                kwargs["adaptive_window_sizes"] = kwargs["classical_adaptive_windows"]

        return data

    def __post_init__(self):
        # Env variable overrides
        if os.environ.get("MATRIX_MAT_CACHE_DIR") and self.cache_dir in ("masks", str(resolve_cache_root())):
            self.cache_dir = os.environ["MATRIX_MAT_CACHE_DIR"]
        if os.environ.get("MATRIX_MAT_DECISIONS_PATH") and self.decisions_log_path in ("DECISIONS.md", str(resolve_output_root() / "decisions.jsonl")):
            self.decisions_log_path = os.environ["MATRIX_MAT_DECISIONS_PATH"]

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
class QCConfig:
    """Configuration for morphological, contrast, and size quality control gating."""
    min_d_um: float = 40.0
    max_d_um: float = 1500.0
    min_pass_diameter_um: float = 40.0
    max_pass_diameter_um: float = 1200.0
    border_margin_px: int = 15
    ring_width_px: Optional[int] = field(default=None, repr=False)
    background_ring_dilation_px: int = 15
    ring_contrast_factor: float = 0.90
    interior_contrast_pct: Optional[float] = field(default=None, repr=False)
    single_mask_max_area_pct: float = 0.25
    max_single_mask_area_fraction: Optional[float] = field(default=None, repr=False)
    min_circularity: float = 0.65
    min_solidity: float = 0.85
    max_eccentricity: float = 0.82
    max_objects_per_fov: int = 0
    require_calibration: bool = True

    @model_validator(mode="before")
    @classmethod
    def _validate_qc_aliases(cls, data: Any) -> Any:
        kwargs = data.kwargs if hasattr(data, "kwargs") and data.kwargs is not None else (data if isinstance(data, dict) else {})
        # max_single_mask_area_fraction -> single_mask_max_area_pct
        if "max_single_mask_area_fraction" in kwargs and kwargs["max_single_mask_area_fraction"] is not None:
            warnings.warn(
                "'max_single_mask_area_fraction' is deprecated; use 'single_mask_max_area_pct' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "single_mask_max_area_pct" in kwargs and kwargs["single_mask_max_area_pct"] is not None:
                if kwargs["single_mask_max_area_pct"] != kwargs["max_single_mask_area_fraction"]:
                    raise ValueError(
                        f"Conflicting values for canonical 'single_mask_max_area_pct' ({kwargs['single_mask_max_area_pct']}) "
                        f"and alias 'max_single_mask_area_fraction' ({kwargs['max_single_mask_area_fraction']})"
                    )
            else:
                kwargs["single_mask_max_area_pct"] = kwargs["max_single_mask_area_fraction"]

        # ring_width_px -> background_ring_dilation_px
        if "ring_width_px" in kwargs and kwargs["ring_width_px"] is not None:
            warnings.warn(
                "'ring_width_px' is deprecated; use 'background_ring_dilation_px' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "background_ring_dilation_px" in kwargs and kwargs["background_ring_dilation_px"] is not None:
                if kwargs["background_ring_dilation_px"] != kwargs["ring_width_px"]:
                    raise ValueError(
                        "Conflicting values for canonical 'background_ring_dilation_px' and alias 'ring_width_px'"
                    )
            else:
                kwargs["background_ring_dilation_px"] = kwargs["ring_width_px"]

        # interior_contrast_pct -> ring_contrast_factor
        if "interior_contrast_pct" in kwargs and kwargs["interior_contrast_pct"] is not None:
            warnings.warn(
                "'interior_contrast_pct' is deprecated; use 'ring_contrast_factor' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            expected_ring = 1.0 - float(kwargs["interior_contrast_pct"])
            if "ring_contrast_factor" in kwargs and kwargs["ring_contrast_factor"] is not None:
                if abs(float(kwargs["ring_contrast_factor"]) - expected_ring) > 1e-4:
                    raise ValueError(
                        f"Conflicting values for 'ring_contrast_factor' ({kwargs['ring_contrast_factor']}) "
                        f"and 'interior_contrast_pct' ({kwargs['interior_contrast_pct']})"
                    )
            else:
                kwargs["ring_contrast_factor"] = expected_ring

        return data

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


# Backward compatibility alias
QCGatingConfig = QCConfig


@dataclass(config=ConfigDict(extra="forbid"))
class PairingConfig:
    """Configuration for Hungarian bipartite tracking and denominator gating."""
    max_centroid_shift_fraction: float = 1.5
    min_displacement_px: float = 50.0
    max_displacement_px_floor: Optional[float] = field(default=None, repr=False)
    min_t0_d_um_denominator: float = 60.0
    min_denominator_d0_um: Optional[float] = field(default=None, repr=False)
    min_fold_change: float = 0.10
    fold_change_min_review: Optional[float] = field(default=None, repr=False)
    max_fold_change: float = 20.0
    fold_change_max_review: Optional[float] = field(default=None, repr=False)
    fallback_greedy: bool = True

    @model_validator(mode="before")
    @classmethod
    def _validate_pairing_aliases(cls, data: Any) -> Any:
        kwargs = data.kwargs if hasattr(data, "kwargs") and data.kwargs is not None else (data if isinstance(data, dict) else {})
        # min_displacement_px vs max_displacement_px_floor
        if "max_displacement_px_floor" in kwargs and kwargs["max_displacement_px_floor"] is not None:
            warnings.warn(
                "'max_displacement_px_floor' is deprecated; use 'min_displacement_px' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "min_displacement_px" in kwargs and kwargs["min_displacement_px"] is not None:
                if kwargs["min_displacement_px"] != kwargs["max_displacement_px_floor"]:
                    raise ValueError(
                        "Conflicting values for 'min_displacement_px' and 'max_displacement_px_floor'"
                    )
            else:
                kwargs["min_displacement_px"] = kwargs["max_displacement_px_floor"]

        # min_t0_d_um_denominator vs min_denominator_d0_um
        if "min_denominator_d0_um" in kwargs and kwargs["min_denominator_d0_um"] is not None:
            warnings.warn(
                "'min_denominator_d0_um' is deprecated; use 'min_t0_d_um_denominator' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "min_t0_d_um_denominator" in kwargs and kwargs["min_t0_d_um_denominator"] is not None:
                if kwargs["min_t0_d_um_denominator"] != kwargs["min_denominator_d0_um"]:
                    raise ValueError(
                        "Conflicting values for 'min_t0_d_um_denominator' and 'min_denominator_d0_um'"
                    )
            else:
                kwargs["min_t0_d_um_denominator"] = kwargs["min_denominator_d0_um"]

        # min_fold_change vs fold_change_min_review
        if "fold_change_min_review" in kwargs and kwargs["fold_change_min_review"] is not None:
            warnings.warn(
                "'fold_change_min_review' is deprecated; use 'min_fold_change' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "min_fold_change" in kwargs and kwargs["min_fold_change"] is not None:
                if kwargs["min_fold_change"] != kwargs["fold_change_min_review"]:
                    raise ValueError("Conflicting values for 'min_fold_change' and 'fold_change_min_review'")
            else:
                kwargs["min_fold_change"] = kwargs["fold_change_min_review"]

        # max_fold_change vs fold_change_max_review
        if "fold_change_max_review" in kwargs and kwargs["fold_change_max_review"] is not None:
            warnings.warn(
                "'fold_change_max_review' is deprecated; use 'max_fold_change' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if "max_fold_change" in kwargs and kwargs["max_fold_change"] is not None:
                if kwargs["max_fold_change"] != kwargs["fold_change_max_review"]:
                    raise ValueError("Conflicting values for 'max_fold_change' and 'fold_change_max_review'")
            else:
                kwargs["max_fold_change"] = kwargs["fold_change_max_review"]

        return data

    def __post_init__(self):
        if self.max_displacement_px_floor is None:
            self.max_displacement_px_floor = self.min_displacement_px
        if self.min_denominator_d0_um is None:
            self.min_denominator_d0_um = self.min_t0_d_um_denominator
        if self.fold_change_min_review is None:
            self.fold_change_min_review = self.min_fold_change
        if self.fold_change_max_review is None:
            self.fold_change_max_review = self.max_fold_change

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
class StatsConfig:
    """Configuration for statistical aggregation, bootstrapping, and hypothesis testing."""
    control_condition: str = "Mat"
    min_t0_d_um_denominator: float = 60.0
    min_denominator_d0_um: Optional[float] = field(default=None, repr=False)
    min_fold_change: float = 0.10
    max_fold_change: float = 20.0
    bootstrap_iterations: int = 2000
    permutation_iterations: int = 10000
    confidence_level: float = 0.95
    include_review_in_stats: bool = False

    @model_validator(mode="before")
    @classmethod
    def _validate_stats_aliases(cls, data: Any) -> Any:
        kwargs = data.kwargs if hasattr(data, "kwargs") and data.kwargs is not None else (data if isinstance(data, dict) else {})
        if "min_denominator_d0_um" in kwargs and kwargs["min_denominator_d0_um"] is not None:
            if "min_t0_d_um_denominator" in kwargs and kwargs["min_t0_d_um_denominator"] is not None:
                if kwargs["min_t0_d_um_denominator"] != kwargs["min_denominator_d0_um"]:
                    raise ValueError(
                        "Conflicting values for 'min_t0_d_um_denominator' and 'min_denominator_d0_um'"
                    )
            else:
                kwargs["min_t0_d_um_denominator"] = kwargs["min_denominator_d0_um"]
        return data

    def __post_init__(self):
        if self.min_denominator_d0_um is None:
            self.min_denominator_d0_um = self.min_t0_d_um_denominator
        if self.min_t0_d_um_denominator is None:
            self.min_t0_d_um_denominator = self.min_denominator_d0_um

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(config=ConfigDict(extra="forbid"))
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

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update_config_dict(target: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Recursively updates configuration dictionary from nested overrides."""
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, dict) and k in target and isinstance(target[k], dict):
            _deep_update_config_dict(target[k], v)
        else:
            target[k] = v


@dataclass(config=ConfigDict(extra="forbid"))
class PipelineConfig:
    """Master pipeline configuration schema."""
    t0_dir: str = "data/brightfield/Day0"
    t7_dir: str = "data/brightfield/Day7"
    output_dir: str = field(default_factory=lambda: str(resolve_output_root()))
    review_dir: str = "review"
    default_pixel_size_um: float = 1.518817
    cli_pixel_size_um: Optional[float] = None
    random_seed: int = 42
    sample_n: Optional[int] = None
    condition: Optional[str] = None
    allow_uncalibrated: bool = False

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def __post_init__(self):
        if os.environ.get("MATRIX_MAT_OUTPUT_ROOT") and self.output_dir in ("output", str(resolve_output_root())):
            self.output_dir = os.environ["MATRIX_MAT_OUTPUT_ROOT"]
        if os.environ.get("MATRIX_MAT_DATA_ROOT"):
            data_root = Path(os.environ["MATRIX_MAT_DATA_ROOT"])
            if self.t0_dir == "data/brightfield/Day0":
                self.t0_dir = str(data_root / "brightfield" / "Day0")
            if self.t7_dir == "data/brightfield/Day7":
                self.t7_dir = str(data_root / "brightfield" / "Day7")

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, cli_overrides: Optional[Dict[str, Any]] = None) -> PipelineConfig:
        """Load configuration from YAML file, filling defaults and applying CLI overrides."""
        yaml_path = Path(yaml_path)
        data: Dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
            content = os.path.expandvars(content)
            data_root = os.environ.get("MATRIX_MAT_DATA_ROOT", "data")
            content = content.replace("${DATA_ROOT}", data_root)
            data = yaml.safe_load(content) or {}
        else:
            logger.warning(f"Config file {yaml_path} not found. Using default parameters.")

        # Alias for qc_gating section
        if "qc_gating" in data and "qc" not in data:
            data["qc"] = data.pop("qc_gating")

        if cli_overrides:
            _deep_update_config_dict(data, cli_overrides)
            # Remove conflicting legacy aliases if their canonical counterpart is overridden by CLI
            if "segmentation" in data and isinstance(data["segmentation"], dict):
                seg_ov = cli_overrides.get("segmentation", {})
                if "primary_backend" in seg_ov:
                    data["segmentation"].pop("backend", None)
                if "adaptive_window_sizes" in seg_ov:
                    data["segmentation"].pop("classical_adaptive_windows", None)
                if "single_mask_max_area_pct" in seg_ov:
                    data["segmentation"].pop("max_single_mask_area_fraction", None)
            if "qc" in data and isinstance(data["qc"], dict):
                qc_ov = cli_overrides.get("qc", {})
                if "background_ring_dilation_px" in qc_ov:
                    data["qc"].pop("ring_width_px", None)
                if "single_mask_max_area_pct" in qc_ov:
                    data["qc"].pop("max_single_mask_area_fraction", None)
            if "pairing" in data and isinstance(data["pairing"], dict):
                pair_ov = cli_overrides.get("pairing", {})
                if "min_t0_d_um_denominator" in pair_ov:
                    data["pairing"].pop("min_denominator_d0_um", None)
                if "min_fold_change" in pair_ov:
                    data["pairing"].pop("fold_change_min_review", None)
                if "max_fold_change" in pair_ov:
                    data["pairing"].pop("fold_change_max_review", None)
            if "stats" in data and isinstance(data["stats"], dict):
                stats_ov = cli_overrides.get("stats", {})
                if "min_t0_d_um_denominator" in stats_ov:
                    data["stats"].pop("min_denominator_d0_um", None)
                if "min_fold_change" in stats_ov:
                    data["stats"].pop("fold_change_min_review", None)
                if "max_fold_change" in stats_ov:
                    data["stats"].pop("fold_change_max_review", None)

        preprocess_data = data.pop("preprocess", {})
        seg_data = data.pop("segmentation", {})
        qc_data = data.pop("qc", {})
        pairing_data = data.pop("pairing", {})
        stats_data = data.pop("stats", {})
        artifacts_data = data.pop("artifacts", {})

        preprocess = PreprocessConfig(**preprocess_data) if isinstance(preprocess_data, dict) else preprocess_data
        segmentation = SegmentationConfig(**seg_data) if isinstance(seg_data, dict) else seg_data
        qc = QCConfig(**qc_data) if isinstance(qc_data, dict) else qc_data
        pairing = PairingConfig(**pairing_data) if isinstance(pairing_data, dict) else pairing_data
        stats = StatsConfig(**stats_data) if isinstance(stats_data, dict) else stats_data
        artifacts = ArtifactConfig(**artifacts_data) if isinstance(artifacts_data, dict) else artifacts_data

        return cls(
            preprocess=preprocess,
            segmentation=segmentation,
            qc=qc,
            pairing=pairing,
            stats=stats,
            artifacts=artifacts,
            **data,
        )

    def to_yaml(self, output_path: str | Path) -> None:
        """Serialize configuration to a YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)
