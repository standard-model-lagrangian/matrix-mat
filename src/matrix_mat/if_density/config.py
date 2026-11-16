"""
Configuration schema for Spheroid IF Sweep pipeline.
Provides Pydantic models with extra="forbid" for strictly validated configs.

Method selectors are Literal enums so a typo fails validation instead of silently falling back to a
default. Settings that only make sense at run level (reference volume when post-processing is on,
the blur threshold, unique slugs) are validated here, before any image is read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("matrix_mat.if_density.config")

BackgroundMethod = Literal["rolling_ball", "tophat", "blank_gel_median", "none"]
DenoiseMethod = Literal["none", "gaussian"]
SpheroidExtentMethod = Literal["actin_threshold", "nuclei_convex_hull"]
ComparatorKind = Literal["prepost", "generic"]


class IFBackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: BackgroundMethod = "rolling_ball"
    radius_multiplier: float = 3.5
    downsample_factor: int = 2
    # Glob of blank-gel reference images; required by (and only used by) method=blank_gel_median.
    blank_glob: Optional[str] = None

    @model_validator(mode="after")
    def _blank_glob_required(self) -> "IFBackgroundConfig":
        if self.method == "blank_gel_median" and not self.blank_glob:
            raise ValueError("background.method=blank_gel_median requires background.blank_glob")
        return self


class IFNormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    p_low: float = 1.0
    p_high: float = 99.8


class IFDenoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: DenoiseMethod = "none"
    # Gaussian sigma in micrometres (resolved to pixels per image); required > 0 for method=gaussian.
    sigma_um: float = 0.0

    @model_validator(mode="after")
    def _sigma_required(self) -> "IFDenoiseConfig":
        if self.method == "gaussian" and self.sigma_um <= 0:
            raise ValueError("denoise.method=gaussian requires denoise.sigma_um > 0")
        return self


class IFPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    expected_nuclear_diameter_um: float = 11.7
    background: IFBackgroundConfig = Field(default_factory=IFBackgroundConfig)
    normalize: IFNormalizeConfig = Field(default_factory=IFNormalizeConfig)
    denoise: IFDenoiseConfig = Field(default_factory=IFDenoiseConfig)


class IFDeduplicateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Only meaningful for overlapping candidate proposals; on a 2D label image (the only input the
    # pipeline accepts) deduplication cannot remove anything and is recorded as not applied.
    iou_threshold: float = 0.50


class IFSizeGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # None: derive from reference_vol_um3 (0.25x and 4.0x). Provenance records the values used.
    min_vol_um3: Optional[float] = None
    max_vol_um3: Optional[float] = None
    reference_vol_um3: Optional[float] = None
    reference_citation: Optional[str] = None


class IFSplittingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Seeded watershed is the only implemented splitter, so there is no method selector.
    enabled: bool = True
    split_factor: float = 2.0
    h_maxima: float = 10.0
    min_distance_um: float = 4.0


class IFBorderExclusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clearance_um: float = 5.85
    enabled: bool = True


class IFPostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    reference_vol_um3: Optional[float] = None
    deduplicate: IFDeduplicateConfig = Field(default_factory=IFDeduplicateConfig)
    size_gate: IFSizeGateConfig = Field(default_factory=IFSizeGateConfig)
    splitting: IFSplittingConfig = Field(default_factory=IFSplittingConfig)
    border_exclusion: IFBorderExclusionConfig = Field(default_factory=IFBorderExclusionConfig)

    @model_validator(mode="after")
    def _reference_volume_required(self) -> "IFPostprocessConfig":
        if self.enabled and (self.size_gate.enabled or self.splitting.enabled):
            if not (self.size_gate.reference_vol_um3 or self.reference_vol_um3):
                raise ValueError(
                    "post-processing size gate/splitting is enabled but size_gate.reference_vol_um3 "
                    "is not configured (set it, with reference_citation, in the YAML)"
                )
        return self


class IFModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pretrained_model: str = "models/cpsam_v2"
    # Only Cellpose-SAM is supported; any other value fails validation instead of being ignored.
    model_type: Literal["cpsam"] = "cpsam"
    # false forces CPU; true uses CUDA/MPS when available (falls back to CPU with a warning).
    use_gpu: bool = True


class ComboConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    name: str
    description: str = ""
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    diameter: Optional[float] = None
    min_size: int = 15
    stitch_threshold: float = 0.0
    do_3D: bool = False
    preprocessing: Optional[IFPreprocessConfig] = None
    postprocessing: Optional[IFPostprocessConfig] = None

    @model_validator(mode="after")
    def _reject_3d(self) -> "ComboConfig":
        if self.do_3D:
            raise ValueError(
                f"combo '{self.slug}': do_3D=true is not supported. The IF pipeline is 2D-only end to end "
                "(single-plane inputs, planar areal density); set do_3D: false."
            )
        if self.stitch_threshold > 0:
            logger.warning(
                f"combo '{self.slug}': stitch_threshold={self.stitch_threshold} only affects z-stacks; on the "
                "2D single-plane inputs this pipeline accepts it has no effect."
            )
        return self


class IFQCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snr_min: float = 1.5
    min_nuclear_vol_um3: float = 150.0
    max_nuclear_vol_um3: float = 4000.0
    border_margin_px: int = 5
    edge_heavy_fraction: float = 0.25
    saturation_fraction_max: float = 0.005
    # Blur flag: an absolute Laplacian-variance threshold on the raw nuclear channel. The cohort-percentile
    # mode flags the worst N% of fields by construction and must be chosen explicitly.
    blur_mode: Literal["absolute", "cohort_percentile"] = "absolute"
    blur_abs_threshold: Optional[float] = None
    blur_percentile_cutoff: float = 10.0
    outlier_iqr_factor: float = 1.5


class IFFeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # actin_threshold: Otsu on the raw actin channel (bit-depth scaled, arm independent), falling back to
    # the nuclei convex hull (flagged per field) when the actin mask is tiny. nuclei_convex_hull: always
    # the hull of the final nuclei masks.
    spheroid_extent_method: SpheroidExtentMethod = "actin_threshold"


class IFSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str = "data/if"
    runs_dir: str = "runs"
    allow_unpaired: bool = True
    # Fields whose TIFFs carry no physical scale fall back to default_pixel_size_um; that is refused
    # unless explicitly allowed here, because every density depends on the pixel size.
    allow_uncalibrated: bool = False
    overwrite: bool = False
    random_seed: int = 42
    default_pixel_size_um: float = 0.5050492474122553
    # Which cross-run comparison to run after the sweep. None: 'prepost' when any pre/post-processing
    # is configured, otherwise 'generic'.
    comparator: Optional[ComparatorKind] = None
    model: IFModelConfig = Field(default_factory=IFModelConfig)
    combos: List[ComboConfig]
    preprocessing: IFPreprocessConfig = Field(default_factory=IFPreprocessConfig)
    postprocessing: IFPostprocessConfig = Field(default_factory=IFPostprocessConfig)
    qc: IFQCConfig = Field(default_factory=IFQCConfig)
    features: IFFeaturesConfig = Field(default_factory=IFFeaturesConfig)

    @model_validator(mode="after")
    def _run_level_checks(self) -> "IFSweepConfig":
        slugs = [c.slug for c in self.combos]
        dups = sorted({s for s in slugs if slugs.count(s) > 1})
        if dups:
            raise ValueError(f"duplicate combo slugs: {dups}; each combo writes to runs/<slug>")
        if self.qc.blur_mode == "absolute" and self.qc.blur_abs_threshold is None:
            raise ValueError(
                "qc.blur_abs_threshold must be set (absolute Laplacian variance of the raw nuclear channel), "
                "or choose qc.blur_mode: cohort_percentile explicitly"
            )
        return self

    def resolved_comparator(self) -> str:
        """Explicit `comparator`, or inferred from whether any pre/post-processing is configured."""
        if self.comparator:
            return self.comparator
        has_prepost = self.preprocessing.enabled or self.postprocessing.enabled or any(
            c.preprocessing is not None or c.postprocessing is not None for c in self.combos
        )
        return "prepost" if has_prepost else "generic"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> IFSweepConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        from matrix_mat.core.paths import expand_root_placeholders
        content = expand_root_placeholders(content)
        data = yaml.safe_load(content) or {}
        return cls(**data)
