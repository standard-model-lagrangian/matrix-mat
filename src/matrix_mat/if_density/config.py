"""
Configuration schema for Spheroid IF Sweep pipeline.
Provides Pydantic models with extra="forbid" for strictly validated configs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("matrix_mat.if_density.config")


class IFBackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "rolling_ball"
    radius_multiplier: float = 3.5
    downsample_factor: int = 2


class IFNormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    p_low: float = 1.0
    p_high: float = 99.8


class IFDenoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: str = "none"


class IFPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    expected_nuclear_diameter_um: float = 11.7
    background: IFBackgroundConfig = Field(default_factory=IFBackgroundConfig)
    normalize: IFNormalizeConfig = Field(default_factory=IFNormalizeConfig)
    denoise: IFDenoiseConfig = Field(default_factory=IFDenoiseConfig)


class IFDeduplicateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    iou_threshold: float = 0.50


class IFSizeGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    min_vol_um3: float = 210.0
    max_vol_um3: float = 3364.0
    reference_vol_um3: Optional[float] = None
    reference_citation: Optional[str] = None


class IFSplittingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    method: str = "seeded_watershed"
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


class IFModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pretrained_model: str = "models/cpsam_v2"
    model_type: str = "cpsam"
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


class IFQCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snr_min: float = 1.5
    min_nuclear_vol_um3: float = 150.0
    max_nuclear_vol_um3: float = 4000.0
    border_margin_px: int = 5
    edge_heavy_fraction: float = 0.25
    saturation_fraction_max: float = 0.005
    blur_percentile_cutoff: float = 10.0
    duplicate_iou_threshold: float = 0.50
    outlier_iqr_factor: float = 1.5


class IFFeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spheroid_extent_method: str = "actin_threshold"
    actin_background_percentile: float = 50.0


class IFSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: str = "data/if"
    runs_dir: str = "runs"
    allow_unpaired: bool = True
    overwrite: bool = False
    random_seed: int = 42
    default_pixel_size_um: float = 0.5050492474122553
    model: IFModelConfig = Field(default_factory=IFModelConfig)
    combos: List[ComboConfig]
    preprocessing: IFPreprocessConfig = Field(default_factory=IFPreprocessConfig)
    postprocessing: IFPostprocessConfig = Field(default_factory=IFPostprocessConfig)
    qc: IFQCConfig = Field(default_factory=IFQCConfig)
    features: IFFeaturesConfig = Field(default_factory=IFFeaturesConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> IFSweepConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        import os
        content = os.path.expandvars(content)
        data_root = os.environ.get("MATRIX_MAT_DATA_ROOT", "data")
        content = content.replace("${DATA_ROOT}", data_root)
        data = yaml.safe_load(content) or {}
        return cls(**data)
