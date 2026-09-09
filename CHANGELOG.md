# Changelog

All notable changes to the **Matrix-Mat** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-09

### Added
- **Packaging & Installation**: Added `pyproject.toml` with console script entry points (`spheroid-brightfield`, `spheroid-if`) and editable install support (`pip install -e .`).
- **Standard Packaging**: Added MIT `LICENSE`, `CITATION.cff`, and `requirements-lock.txt`.
- **CI Matrix**: Added GitHub Actions workflow (`.github/workflows/ci.yml`) supporting Python 3.10 through 3.13 on Linux and macOS.
- **Config Key Mapping Test**: Added `tests/test_config_mapping.py` asserting 100% concordance between YAML config files and Python dataclass schemas.
- **Module Entry Point**: Added `spheroid_pipeline_v2/__main__.py` enabling `python -m spheroid_pipeline_v2`.
- **CLI Flags**: Added `--condition` hydrogel filter and `-o`/`--out` alias in `run_pipeline.py`.
- **Multi-Diameter Sweep**: Implemented Cellpose vision transformer diameter sweep fallback (`[150, 200, 250, 300]`).

### Fixed
- **CLI Override Collision**: Fixed `--no-cache` / nested `cli_overrides` replacing dataclass instances with plain dictionaries in `config.py`.
- **YAML Config Alignment**: Added all previously unmapped configuration keys to dataclasses (`adaptive_window_sizes`, `cpsam_*`, `hough_*`, `ring_contrast_factor`, `min_displacement_px`, etc.).
- **DL Parameter Wiring**: Wired configured hyperparameters (`expected_diameter`, `flow_threshold`, `cellprob_threshold`, `min_size`) into `model.eval()` calls.
- **Deep Learning Fallback Routing**: Fixed Tier 2 fallback logic so empty deep learning detections route down to Tier 3 Classical Adaptive Watershed rather than prematurely declaring verified empty.
- **Audit Crop Grid Generation**: Fixed image path extension in `create_fail_audit_crops_grid` so FAIL visual supervision grids are generated reliably.
- **FOV Metric Double-Counting**: Corrected `matched_pairs_fov` in `report.py` to use `.nunique()` on `pair_key`, fixing ~2x reporting inflation and derived call rates.
- **Dependency Floor**: Bumped Cellpose floor to `>=4.0.0` in `requirements.txt` to guarantee Cellpose-SAM (`cpsam_v2`) and `cpdino` availability.
- **Plotting Robustness**: Guarded `max(fcs)` on empty arrays in Figure 1 Panel A and dynamically computed y-axis limits in Figure 6.
- **CI Resilience**: Converted missing TIFF dataset errors in `test_m1_real_data_eval.py` to `unittest.SkipTest` so CI passes without private image files.

### Removed
- **Runtime Residue**: Deleted committed runtime output `DECISIONS.md` and added it to `.gitignore`.
- **Duplicate Scripts**: Removed redundant root script wrappers (`generate_presentation_plots.py`, `generate_raw_volumes_plot.py`, `generate_well_growth_analysis.py`, `compare_runs.py`, `compare_prepost.py`, `config.yaml`).
- **Internal Agent Prompt**: Removed legacy agent briefing prompt (`ORIGINAL_REQUEST.md`) and point-in-time document (`TEST_READY.md`).
- **Ad-Hoc Experiments**: Moved `test_cpsam_batch.py` and `test_cpsam_overlays.py` to `scripts/`.
