"""IF config schema: enums, required companions, dead keys removed, run-level validators, comparator key."""
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from matrix_mat.if_density.config import (
    ComboConfig,
    IFBackgroundConfig,
    IFDenoiseConfig,
    IFFeaturesConfig,
    IFModelConfig,
    IFPostprocessConfig,
    IFSweepConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ("sweep.yaml", "prepost_ablation.yaml", "robustness_check.yaml")


def _base(**over):
    d = {"combos": [{"slug": "a", "name": "a"}], "qc": {"blur_abs_threshold": 100.0}}
    d.update(over)
    return d


@pytest.mark.parametrize("cls,kwargs", [
    (IFBackgroundConfig, {"method": "rolling_bal"}),
    (IFDenoiseConfig, {"method": "gausian", "sigma_um": 1.0}),
    (IFFeaturesConfig, {"spheroid_extent_method": "actin"}),
    (IFModelConfig, {"model_type": "cyto3"}),
])
def test_method_typos_fail_validation(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs)


def test_companion_settings_required():
    with pytest.raises(ValidationError, match="blank_glob"):
        IFBackgroundConfig(method="blank_gel_median")
    assert IFBackgroundConfig(method="blank_gel_median", blank_glob="b/*.tif").blank_glob == "b/*.tif"
    with pytest.raises(ValidationError, match="sigma_um"):
        IFDenoiseConfig(method="gaussian")
    assert IFDenoiseConfig(method="gaussian", sigma_um=0.5).sigma_um == 0.5


@pytest.mark.parametrize("section,key", [
    ("qc", "duplicate_iou_threshold"),
    ("features", "actin_background_percentile"),
])
def test_dead_keys_removed(section, key):
    data = _base()
    data.setdefault(section, {})[key] = 1.0
    with pytest.raises(ValidationError):
        IFSweepConfig(**data)
    with pytest.raises(ValidationError):
        IFPostprocessConfig(splitting={"method": "seeded_watershed"})


def test_run_level_validation_happens_before_images():
    with pytest.raises(ValidationError, match="reference_vol_um3"):
        IFPostprocessConfig(enabled=True)
    with pytest.raises(ValidationError, match="blur_abs_threshold"):
        IFSweepConfig(combos=[{"slug": "a", "name": "a"}])
    with pytest.raises(ValidationError, match="duplicate combo slugs"):
        IFSweepConfig(**_base(combos=[{"slug": "a", "name": "a"}, {"slug": "a", "name": "b"}]))


def test_size_gate_bounds_default_to_none():
    cfg = IFPostprocessConfig(enabled=True, size_gate={"reference_vol_um3": 800.0})
    assert cfg.size_gate.min_vol_um3 is None and cfg.size_gate.max_vol_um3 is None


def test_do_3d_rejected():
    with pytest.raises(ValidationError, match="do_3D"):
        ComboConfig(slug="x", name="x", do_3D=True)


def test_comparator_explicit_or_inferred():
    assert IFSweepConfig(**_base()).resolved_comparator() == "generic"
    assert IFSweepConfig(**_base(combos=[{"slug": "a", "name": "a", "preprocessing": {"enabled": False}}])).resolved_comparator() == "prepost"
    assert IFSweepConfig(**_base(comparator="prepost")).resolved_comparator() == "prepost"
    with pytest.raises(ValidationError):
        IFSweepConfig(**_base(comparator="robustness"))


@pytest.mark.parametrize("name,expected", [("sweep.yaml", "generic"), ("prepost_ablation.yaml", "prepost"),
                                           ("robustness_check.yaml", "prepost")])
def test_shipped_configs_declare_comparator_and_no_3d(name, expected):
    raw = yaml.safe_load((REPO_ROOT / "configs" / name).read_text())
    assert raw["comparator"] == expected
    assert raw["allow_uncalibrated"] is False
    assert not any(c.get("do_3D") for c in raw["combos"])
    cfg = IFSweepConfig.from_yaml(REPO_ROOT / "configs" / name)
    assert cfg.resolved_comparator() == expected


def test_shipped_slugs_have_no_subset_suffix_and_are_unique():
    for name in SHIPPED:
        slugs = [c["slug"] for c in yaml.safe_load((REPO_ROOT / "configs" / name).read_text())["combos"]]
        assert len(set(slugs)) == len(slugs)
        assert not any(s.endswith("_subset") for s in slugs)
