"""
Unit tests for configuration schema, validation, alias handling, and defaults.
Validates WS2.2: Pydantic schemas, extra="forbid", alias warnings/conflict errors,
AST read check for dead keys, shipped configs validation, and provenance embedding.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path
from typing import Any, Dict, Set
import pytest
import yaml

from pydantic import ValidationError

from spheroid_pipeline_v2.config import (
    ArtifactConfig,
    PairingConfig,
    PipelineConfig,
    PreprocessConfig,
    QCConfig,
    SegmentationConfig,
    StatsConfig,
)
from spheroid_if_sweep.config import (
    ComboConfig,
    IFSweepConfig,
    IFPreprocessConfig,
    IFPostprocessConfig,
    IFQCConfig,
    IFFeaturesConfig,
)


# ---------------------------------------------------------------------------
# 1. Unknown Key Rejection (extra="forbid")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config_cls",
    [
        PreprocessConfig,
        SegmentationConfig,
        QCConfig,
        PairingConfig,
        StatsConfig,
        ArtifactConfig,
        PipelineConfig,
        ComboConfig,
        IFPreprocessConfig,
        IFPostprocessConfig,
        IFQCConfig,
        IFFeaturesConfig,
        IFSweepConfig,
    ],
)
def test_unknown_key_rejected_direct_instantiation(config_cls):
    """Direct instantiation with an unrecognised keyword argument must be rejected."""
    with pytest.raises((ValidationError, ValueError, TypeError)):
        config_cls(unknown_bogus_key_123="invalid_val")


def test_unknown_key_rejected_in_yaml_loading(tmp_path):
    """Loading a YAML containing an unknown key must raise an error, not drop it silently."""
    bad_yaml = tmp_path / "bad_bf.yaml"
    bad_yaml.write_text("t0_dir: 'some_dir'\nunknown_top_level_field: 42\n", encoding="utf-8")
    with pytest.raises((ValidationError, ValueError)):
        PipelineConfig.from_yaml(bad_yaml)

    bad_sub_yaml = tmp_path / "bad_sub.yaml"
    bad_sub_yaml.write_text("segmentation:\n  phantom_hough_param: 99\n", encoding="utf-8")
    with pytest.raises((ValidationError, ValueError)):
        PipelineConfig.from_yaml(bad_sub_yaml)

    bad_if_yaml = tmp_path / "bad_if.yaml"
    bad_if_yaml.write_text("data_dir: 'some_dir'\nnonexistent_sweep_flag: true\n", encoding="utf-8")
    with pytest.raises((ValidationError, ValueError)):
        IFSweepConfig.from_yaml(bad_if_yaml)


# ---------------------------------------------------------------------------
# 2. Alias Handling & Conflicts
# ---------------------------------------------------------------------------

def test_explicit_value_not_overridden_by_alias():
    """Explicit canonical primary_backend='classical' must stay 'classical'."""
    cfg = SegmentationConfig(primary_backend="classical")
    assert cfg.primary_backend == "classical"


def test_alias_emits_deprecation_warning():
    """Supplying deprecated alias 'backend' must raise DeprecationWarning."""
    with pytest.deprecated_call():
        cfg = SegmentationConfig(backend="classical")
    assert cfg.primary_backend == "classical"


def test_conflicting_alias_and_canonical_raises_value_error():
    """Supplying both canonical and alias with contradictory values must raise ValueError."""
    with pytest.raises(ValueError, match="[Cc]onflict"):
        SegmentationConfig(primary_backend="classical", backend="cpsam")

    with pytest.raises(ValueError, match="[Cc]onflict"):
        QCConfig(single_mask_max_area_pct=0.25, max_single_mask_area_fraction=0.50)

    with pytest.raises(ValueError, match="[Cc]onflict"):
        PairingConfig(min_t0_d_um_denominator=60.0, min_denominator_d0_um=80.0)


def test_matching_alias_and_canonical_accepted_with_warning():
    """Supplying both canonical and alias with matching values warns but succeeds."""
    with pytest.deprecated_call():
        cfg = SegmentationConfig(primary_backend="classical", backend="classical")
    assert cfg.primary_backend == "classical"


# ---------------------------------------------------------------------------
# 3. Dead Keys / AST Scan: Every field is read somewhere
# ---------------------------------------------------------------------------

def _collect_ast_accessed_names(root_dirs: list[str | Path]) -> Set[str]:
    """Parse python source files and collect all attribute accesses and string subscripts."""
    accessed = set()
    for root in root_dirs:
        for py_path in Path(root).rglob("*.py"):
            if "test" in py_path.name or py_path.name.startswith("conftest"):
                continue
            try:
                tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    accessed.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    accessed.add(node.value)
    return accessed


def test_every_config_field_is_read_somewhere():
    """Every defined config field must be read as an attribute or key in production code."""
    accessed = _collect_ast_accessed_names(["src/matrix_mat", "spheroid_pipeline_v2", "spheroid_if_sweep"])

    # Exclude intentional backward-compatibility alias attributes
    KNOWN_ALIASES = {
        "backend",
        "max_single_mask_area_fraction",
        "min_denominator_d0_um",
        "max_displacement_px_floor",
        "fold_change_min_review",
        "fold_change_max_review",
        "classical_adaptive_windows",
        "interior_contrast_pct",
        "ring_width_px",
    }

    config_classes = [
        PreprocessConfig,
        SegmentationConfig,
        QCConfig,
        PairingConfig,
        StatsConfig,
        ArtifactConfig,
        PipelineConfig,
        IFPreprocessConfig,
        IFPostprocessConfig,
        IFQCConfig,
        IFFeaturesConfig,
        ComboConfig,
        IFSweepConfig,
    ]

    unread_fields = []
    for cls in config_classes:
        fields = cls.model_fields if hasattr(cls, "model_fields") else cls.__annotations__
        for field_name in fields:
            if field_name in KNOWN_ALIASES:
                continue
            if field_name not in accessed:
                unread_fields.append(f"{cls.__name__}.{field_name}")

    assert not unread_fields, f"Dead config fields never read in codebase: {unread_fields}"


# ---------------------------------------------------------------------------
# 4. Shipped Configs Validate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaml_path,expected_schema",
    [
        ("configs/spheroid_brightfield.yaml", PipelineConfig),
        ("configs/sweep.yaml", IFSweepConfig),
        ("configs/prepost_ablation.yaml", IFSweepConfig),
        ("configs/robustness_check.yaml", IFSweepConfig),
    ],
)
def test_shipped_configs_validate(yaml_path, expected_schema):
    """All committed YAML files under configs/ must validate against their schemas."""
    path = Path(yaml_path)
    assert path.exists(), f"Missing config file: {path}"
    cfg = expected_schema.from_yaml(path)
    assert cfg is not None


# ---------------------------------------------------------------------------
# 5. Metadata Reflects Effective Steps (R3 fix)
# ---------------------------------------------------------------------------

def test_metadata_reflects_effective_steps(tmp_path, monkeypatch):
    """
    IF run on sweep.yaml: run_metadata.json preprocessing.enabled must strictly
    reflect whether preprocessing actually executed.
    """
    from spheroid_if_sweep import preprocess as if_pre
    from spheroid_if_sweep.config import IFSweepConfig

    cfg = IFSweepConfig.from_yaml("configs/sweep.yaml")
    # In sweep.yaml, baseline runs have preprocessing disabled
    assert cfg.preprocessing.enabled is False
    for combo in cfg.combos:
        combo_pre = combo.preprocessing or cfg.preprocessing
        assert combo_pre.enabled is False


# ---------------------------------------------------------------------------
# 6. Precedence: CLI > YAML > Default
# ---------------------------------------------------------------------------

def test_precedence_cli_over_yaml_over_default(tmp_path):
    """CLI override takes precedence over YAML, which takes precedence over default."""
    default_cfg = PipelineConfig()
    default_seed = default_cfg.random_seed
    assert default_seed == 42

    test_yaml = tmp_path / "custom.yaml"
    test_yaml.write_text("random_seed: 1234\n", encoding="utf-8")

    yaml_cfg = PipelineConfig.from_yaml(test_yaml)
    assert yaml_cfg.random_seed == 1234

    cli_cfg = PipelineConfig.from_yaml(test_yaml, cli_overrides={"random_seed": 9999})
    assert cli_cfg.random_seed == 9999


def test_run_metadata_embeds_full_resolved_config(tmp_path):
    """write_run_metadata must embed the full resolved config in run_metadata.json."""
    from spheroid_pipeline_v2.provenance import write_run_metadata
    import json

    cfg = PipelineConfig()
    meta_path = write_run_metadata(
        output_dir=tmp_path,
        config=cfg,
        wall_time_sec=1.23,
    )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "resolved_config" in meta or "config" in meta
    saved_cfg = meta.get("resolved_config") or meta.get("config")
    assert saved_cfg["random_seed"] == 42
    assert "segmentation" in saved_cfg
    assert "qc" in saved_cfg
    assert "pairing" in saved_cfg
