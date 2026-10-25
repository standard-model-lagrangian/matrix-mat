# Changelog

All notable changes to the **Matrix-Mat** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-24

Verification of 0.2.0 against `docs/IMPROVEMENT_SPEC.md`, including a full real-data run, found the problems below.
Every fix has a test that failed first.

### Fixed — would change or silently corrupt results
- **IF: shipped `sweep.yaml` and `robustness_check.yaml` crashed** (`'NoneType' object has no attribute 'get'`): combos without their own pre/post sections now inherit the top-level ones. All three shipped IF configs are run end to end in CI on the synthetic fixture.
- **Brightfield: missing model weights produced 0 objects and exit code 0.** Weight verification now runs before Cellpose is imported and aborts the run, and `matrix-mat bf run` returns non-zero on failure (it previously read a `success` key the pipeline never set, so it always returned 0).
- **Brightfield: confidence-interval columns were on the log₂ scale** next to ratio-scale medians. `bootstrap_ci_95_*` are now on the fold-change scale (back-transformed); `log2fc_ci_95_*` hold the log₂ interval.
- **Brightfield: `n_gels` counted every gel with any pair**, not the gels in the analysed population.
- **Brightfield: presentation figures 1–6 failed on every real run** (`KeyError` from string vs int FOV keys); the error was logged as a warning.
- **Segmentation: SEG_FAIL flags were lost on mask-cache hits**, and failed inference was cached as an empty mask. Flags are now stored with the cached mask, failed inference is never cached, and the cache key includes the real package version (it was hard-coded to 0.1.0).
- **Segmentation: `--backend cyto3` / `sam` silently ran cpsam.** Unknown backends are rejected by the CLI and the config schema.
- **IF: size-gate/split reference volume fell back to each image's own median** whenever the config omitted it, which all shipped configs did. A configured `reference_vol_um3` with `reference_citation` is now required.
- **IF: no gel-level hypothesis test existed**, although the figure title said "*p<0.05 vs Mat". `summary_by_material.csv` now has gel-level permutation p-values (`p_raw`, Holm `p_adj`, `min_achievable_p`). Empty subsets report NaN instead of 0.0. Mislabelled `mean_nuclei_density_*` aliases were removed.
- **IF: the pseudo-volumetric density was still written** to `per_image.csv` and `qc_flags.csv`, and several readers silently fell back to it, mixing units. Only `nuclei_per_mm2` is written now, and readers raise on pre-0.2 outputs.
- **IF: blur flag still flagged the worst 10% of fields by construction**, and `blur_abs_threshold` was rejected by the schema. An absolute threshold is now required (percentile mode is explicit opt-in).
- **IF figure title "Confirming QC Gating Does Not Alter Material Biological Trends"** was a hard-coded conclusion; it is now neutral.
- **OME-XML scale parsing ignored `PhysicalSizeXUnit`** (650 nm read as 650 µm/px) and had no plausibility check.
- **Provenance:** git commit was read from the working directory (`"unknown"` outside the repo), `lockfile_sha256` was always null, and classical runs claimed the cpsam weights hash.

### Fixed — reproducibility, CI, usability
- `requirements-lock.txt` regenerated with `uv pip compile --universal --generate-hashes`; it was missing `pydantic`, so the CI locked job could not import the package. CI installs it with `--require-hashes`.
- `models.lock.json` URL pointed to an endpoint returning HTTP 500; it now points to Hugging Face `mouseland/cellpose-sam/cpsam_v2`, whose published SHA-256 matches the pin.
- Shipped YAMLs use `${DATA_ROOT}`, `${OUTPUT_ROOT}`, `${MODELS_ROOT}`, `${CACHE_ROOT}` instead of working-directory paths. CLI defaults no longer override YAML paths, and the review folder is no longer created as a side effect.
- CI gates added: ruff (syntax/undefined names; 11 undefined names fixed), mypy on `core/`, coverage ≥ 60% overall and ≥ 80% on `core/`.
- Tests no longer depend on the working directory. The repository-pollution guard now covers the whole session. DL tests skip cleanly without torch.

### Fixed — CLI and docs (follow-up)
- `matrix-mat if sweep` ignored `--runs-dir`, `--data-dir` and `--sample`; they are now passed through, and the command exits non-zero on failure.
- IF `compare*` commands defaulted to `./runs` while `if sweep` writes to `${OUTPUT_ROOT}/runs`; they now share one default. `compare-prepost --biological` now draws figures for the PP11 run instead of the runs folder.
- Brightfield figure 1A showed stars from an unrelated test (per-spheroid fold change) on pooled well biomass; it is now descriptive (n wells).
- `requirements-lock.txt` now also locks the `dev` extra, so the documented install gives a complete, hash-verified environment.
- README, beginner, maintainer, visualization and troubleshooting guides rewritten for the `matrix-mat` CLI. Commands in the docs are parsed by tests.
- IF golden test: integer columns are compared exactly, float columns to within 0.5% (OpenCV 8-bit rounding differs between x86 and ARM).

### Validation status
- The manual-count validation is suspended: `tests/data/if_manual_counts.csv` has no provenance and matches pipeline output to within 0–4 nuclei (see `tests/data/README.md`). Unsupported claims were removed from `docs/VALIDATION.md`.

## [0.2.0] - 2026-10-24

### Changed (Breaking Changes)
- **Areal Density Standardization**: Replaced pseudo-volumetric density `nuclei_density_per_mm3` with physically sound areal planar density `nuclei_per_mm2` across the immunofluorescence sweep pipeline (`matrix_mat.if_density`).
- **Hierarchical Statistical Inference**: Brightfield inference moved from pseudo-replicated spheroid-level sampling to gel-level hierarchical bootstrap and cluster permutation tests with Holm-Bonferroni family-wise error rate control.
- **Strict Content-Addressed Caching**: Mask caching now keys off image SHA-256, model SHA-256, and full parameter payloads to prevent stale mask reuse across datasets and parameter changes.
- **Config Validation with Pydantic v2**: Replaced unvalidated dictionaries with strict Pydantic models forbidding unknown or misspelled configuration keys.
- **Monorepo Layout Migration**: Relocated core code to `src/matrix_mat/{core,brightfield,if_density,reporting}`, leaving backward-compatible shims in `spheroid_pipeline_v2` and `spheroid_if_sweep`.
- **Dynamic Reporting**: Removed hardcoded outcome statements and paired testing on unpaired Day 0 / Day 7 distributions; all reporting figures now consume precomputed statistics directly from `core.stats` tables.

### Fixed
- **Post-Processing Execution Order**: Reordered IF pipeline steps so clump splitting precedes size gating, preventing valid clusters from being erroneously discarded as debris.
- **Reproducibility & Determinism**: Enforced seeded pseudo-random generators (`np.random.default_rng`), pinned random seeds, and decoupled heavy PyTorch imports from CPU-only unit test paths.
- **CLI and Documentation Alignment**: Aligned CLI flags and documentation in `README.md`, verified through automated argument-parsing tests (`test_readme_commands_parse`).

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
