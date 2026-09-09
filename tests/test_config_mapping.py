"""
Automated unit tests asserting that 100% of keys in configuration YAML files
map directly onto dataclass attributes, preventing silent config drop regressions.
"""

import unittest
from pathlib import Path
import yaml

from spheroid_pipeline_v2.config import (
    PipelineConfig,
    PreprocessConfig,
    SegmentationConfig,
    QCConfig,
    PairingConfig,
    StatsConfig,
    ArtifactConfig,
)


class TestConfigYAMLDataclassMapping(unittest.TestCase):
    """Asserts that YAML config keys are strictly mapped to dataclasses without silent drops."""

    def test_spheroid_brightfield_yaml_keys_map_to_dataclass(self):
        yaml_path = Path("configs/spheroid_brightfield.yaml")
        self.assertTrue(yaml_path.exists(), f"Missing config file: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cfg = PipelineConfig.from_yaml(yaml_path)

        unmapped = []
        for k, v in data.items():
            if isinstance(v, dict):
                if not hasattr(cfg, k):
                    unmapped.append(f"PipelineConfig missing section: '{k}'")
                else:
                    section_obj = getattr(cfg, k)
                    for sub_k in v.keys():
                        if not hasattr(section_obj, sub_k):
                            unmapped.append(
                                f"Section '{k}' ({type(section_obj).__name__}) missing field: '{sub_k}'"
                            )
            else:
                if not hasattr(cfg, k):
                    unmapped.append(f"PipelineConfig missing top-level field: '{k}'")

        self.assertEqual(
            unmapped,
            [],
            f"Found unmapped YAML keys in {yaml_path}:\n" + "\n".join(unmapped),
        )

    def test_cli_overrides_retain_dataclass_types(self):
        """Asserts that nested cli_overrides do not replace dataclass instances with dicts."""
        cfg = PipelineConfig.from_yaml(
            "configs/spheroid_brightfield.yaml",
            cli_overrides={
                "segmentation": {"force_recompute": True},
                "output_dir": "custom_output",
                "condition": "S34D30",
            },
        )
        self.assertIsInstance(cfg.segmentation, SegmentationConfig)
        self.assertTrue(cfg.segmentation.force_recompute)
        self.assertEqual(cfg.output_dir, "custom_output")
        self.assertEqual(cfg.condition, "S34D30")

    def test_if_sweep_yaml_loads_without_errors(self):
        """Asserts that configs/sweep.yaml is valid YAML with required structure."""
        yaml_path = Path("configs/sweep.yaml")
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.assertIsInstance(data, dict)
            self.assertIn("data_dir", data)
            self.assertIn("combos", data)


if __name__ == "__main__":
    unittest.main()
