# Changelog

All notable changes to the **Matrix-Mat** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - unreleased

Fixes from the post-0.2.1 review. Every fix has a test that failed first.

### Core, CLI, reporting and packaging

#### Fixed — could change or silently corrupt results
- **Mask cache key ignored array shape and dtype** (a 3×4 and a 4×3 image with the same bytes collided). The key now hashes shape + dtype + bytes, and the config part accepts dicts/lists/tuples mixing dataclasses, pydantic models and plain values, serialised deterministically (non-algorithmic fields such as `cache_dir` are dropped at every level). *All existing cache entries miss once after upgrading.*
- **Mask cache:** labels above 65,535 were clipped silently; they are now stored losslessly as `.npz`. Re-saving without metadata left a stale sidecar (e.g. `qc_flag: SEG_FAIL`) behind; it is now removed. Mask and sidecar are written atomically. 3-D or negative-label masks raise a clear `ValueError`. Corrupt entries are deleted and reported as a miss, so `has()` and `load()` agree.
- **Scale:** ImageJ/Fiji TIFFs (XResolution in the ImageJ `unit`, ResolutionUnit NONE) were never read and fell back to the default; they are now read, and ImageJ units (nm, mm, cm, inch, …) are converted instead of assumed µm. Rejected calibrations (72/96 DPI screen defaults, implausible EVOS/OME/ImageJ/TIFF values) used to return the rejected value (e.g. 352.78 µm/px); they now return the configured default with `is_calibrated=False` and a `scale_source` naming the reason. OME-XML with a leading Bio-Formats comment, other schema years or prefixed namespaces is parsed. EVOS `Magnification: "4x"` no longer discards a valid `MicronsPerPixel`. `extract_pixel_size(..., return_info=True)` and `extract_scale_info()` expose `is_calibrated`. Real-data parsing (285 EVOS TRANS files, 55 IF files) is unchanged.
- **Cluster permutation test returned p = 0 when a gel had only NaN values.** Non-finite values and empty gels are now dropped first, and fewer than two gels per group gives p = NaN. `compute_mwu_pvalue` and the paired Wilcoxon test drop NaN/inf.
- **Units:** NaN diameters/areas became 0 volume and NaN/0 pixel sizes gave ~1e-8 µm. NaN now propagates and a non-positive or non-finite pixel size raises `ValueError`.
- **Filename parser** accepted partial matches (`Day0 S34 D30 gel1…` → condition `D30`; `Day0_Mat_gel1…` → `Day0_Mat`; `Mat+drug` → `drug`), kept case (`mat` ≠ `Mat`), kept zero padding (`gel01_001` ≠ `gel1_0001`) and paired `… (1).tif` copies. Names must now match in full (space or underscore separators), are normalised, and copies / ambiguous conditions raise `ValueError`. All real TRANS names parse exactly as before.
- **manifest.csv recorded a scale the run did not use** and turned NaN `is_calibrated` into True. `export_complete_manifest` accepts the run's `scale_extractor` / `default_pixel_size_um` / `pixel_size_override_um` and treats NaN as missing.
- **Reports:** the call rate could exceed 100% (it now counts matched FOV pairs with PASS objects at every timepoint, `n/a` when there are none); "Denominator Gate Pass" counted FAIL/REVIEW pairs; the version was hard-coded `2.0.0`; the pixel size shown was always the config default (now the scale(s) actually used, with source); the 60 µm gate was hard-coded; the footnote claimed a Monte Carlo test with M = 10,000 (it now describes exact enumeration / Monte Carlo with the configured M); a raw p-value fallback was labelled `p_adj`/Holm; p = 0.0004 printed as `0.000`; conditions not in a fixed list were dropped. `report.html` no longer loads Bootstrap from a CDN or shows raw LaTeX.
- **Figures:** fig 6 y-limits added Day 0 and Day 7 arrays element-wise; fig 1C crashed when a fold change was NaN; an all-NaN `qc_reasons` column crashed; empty groups were drawn from placeholder data (`[1.0]`, `[1e5]`); the V0–V7 scatter crashed with no valid pairs; 16-bit overlays wrapped around; conditions/timepoints outside a fixed list were dropped; `figure_dpi`/`contact_sheet_dpi` were ignored; figures leaked when plotting raised; the IF "Clean only" bar fell back to the all-cohort mean with SD 0; IF jitter was unseeded. Contact sheets can take this run's overlay list (`overlay_paths=`) or are restricted to the manifest's images, and overlays ignore FAIL records when colouring labels.
- **IF summary labels:** the docstring said gel *means* are tested; the test compares the median of per-gel medians. Labels now say so (`gel_mean_*` columns are descriptive).

#### Fixed — CLI, reproducibility, tooling
- A `-c/--config` path that does not exist now exits 2 instead of silently using defaults; the default config resolves next to the package, not the working directory, with a WARNING if absent. `-c` on `if compare*` now supplies the default runs directory.
- `--pixel-size 0` was ignored and negative values accepted; it must now be > 0. `--device` accepts `auto|cpu|cuda|cuda:N|mps`.
- `-o/--output-dir` on every subcommand (old spellings kept), no option abbreviations, `-v/-q` after the subcommand, `matrix-mat --version`, `bf preflight --input-dir`, `--t0-subdir/--t7-subdir`, `--dry-run` for `bf run` and `if sweep`, `if sweep --strict-pairing`, `if compare --run-names/--scope`, `if compare-prepost --scope`; `if preflight` honours the config's `allow_uncalibrated`. The IF `manifest.csv` records `scale_source` and `is_calibrated`.
- `bf plots` exited 0 when figures failed (now 1 with an error summary) and ignored the config `output_dir`. `if compare*` printed a traceback on a missing runs directory (now a one-line error, exit 1). `compare-prepost --biological` silently picked the first `*PP11*` run (now an error when several match; `--run-pattern`). Hard-coded dated input folders and `104_PP11_full` were removed. `--full-dataset` now clears a YAML `sample_n`; `--overwrite` / `--sample` help describe what they do.
- `spheroid-if`, `python -m spheroid_if_sweep` and `python run_pipeline.py` propagate exit codes; `run_pipeline.py` forwards to `matrix_mat.brightfield.run_pipeline`. Added `spheroid_if_sweep.sweep` and `spheroid_if_sweep.comparator_biological` shims.
- Model weights verification fails closed: a model missing from `models.lock.json` or without a `sha256` is an error (`tools/fetch_models.py --verify-only` prints INVALID and exits non-zero). `models.lock.json` ships as package data. `fetch_models.py` downloads to a temporary file with a timeout, checks size and SHA-256 before `os.replace`, and removes partial files on any error or Ctrl-C.
- Paths: `~` is expanded and an unresolved `${VAR}` raises a clear error.
- `requirements.txt` now matches the core dependencies in `pyproject.toml` (adds pydantic; torch/cellpose moved to the `dl` extra). CI: weekly/manual slow-test job, macOS runner, ruff `E,F,B` over the whole repo, mypy over core/reporting/CLI. Ruff settings live in `pyproject.toml`; pre-commit pins ruff to the locked version.

### Brightfield

#### Fixed — could change results
- **Pairing:** FAIL/SEG_FAIL objects are excluded from the Hungarian assignment, and the distance gate is applied before assignment (a nearby debris object could steal a spheroid's match, and a valid pair could be dropped because the global optimum picked over-limit pairs). The greedy fallback obeys the same gate.
- **QC gates:** `stats.min_t0_d_um_denominator` / `min_fold_change` / `max_fold_change` were parsed but ignored; they now take effect (conflicting non-default values under `pairing:` raise). `max_objects_per_fov` no longer promotes FAIL objects to REVIEW.
- **Cross-FOV duplicate check:** now configurable (`qc.dedup_cross_fov`, `qc.dedup_max_distance_um`, converted per image) instead of a hard-coded 35 px; overlays show post-dedup flags.
- **Object IDs:** FAIL objects get `<pair_key>_<tp>_fail<label>` so IDs are unique per image and debris can no longer inherit a spheroid's `pair_id`.
- **Mask cache key** now includes pixel size and preprocessing settings (one-time cache miss). Fallback-backend masks are plausibility-checked and not cached under the primary key.
- **Statistics:** `random_seed` is passed through; NaN log2 values are dropped before bootstrap/permutation; conditions with < 3 gels get `low_gel_n=True` and a `ci_method` suffix.
- **Config honoured:** `segmentation.device` (default `auto`), `save_corrected_images`, fallback backend tiers, classical-backend thresholds (`classical_min_d_um`, `classical_min_contrast`, `classical_ring_width_px`). Root placeholders are substituted after YAML parsing (Windows paths work).

#### Fixed — crashes and silent failures
- `spheroid-brightfield` / root `run_pipeline.py` crashed on every real run (duplicate `run_spheroid_pipeline` definition).
- Report steps run independently; failures are recorded and the run status is `SUCCESS`, `COMPLETED_WITH_ERRORS` or `FAILED` (non-zero exit unless SUCCESS).
- Unreadable / failed images are recorded in `images_status.csv`; the exclusion audit counts them; RGB TIFFs are converted to grayscale; manual masks are treated as binary unless they are a clean label image, and wrong-shaped masks are rejected.
- Audit figures derive conditions from the data, read TIFFs at full bit depth, accept `.tif/.tiff` case-insensitively, and respect `--condition`/`--sample`.
- `decisions.jsonl` is per run; report overlays/contact sheets use this run's overlays; `--smoke-test` uses a temp dir.

#### Usability
- `[i/N]` progress with ETA; `--dry-run`; `objects.csv` written incrementally; preflight warns on missing weights/cellpose and duplicate pair keys; legacy flags (`--input-dir`, `-o`, `--review-dir`, `--quiet`, `--sample`) behave as documented.

### Immunofluorescence

#### Fixed — could change results
- **Density denominator:** spheroid extent is computed from the raw actin channel, so it no longer varies across preprocessing arms; the border strip excluded from the nucleus count is also excluded from the area. `nuclei_per_mm2` changes for PP10/PP11/201–204 (area) and PP01/PP11/201–204 (border). New columns: `density_area_um2`, `spheroid_extent_method`, `spheroid_extent_hull_fallback`, `scale_source`.
- **Comparators:** a run missing an image is NaN (not 0) and excluded from spreads; missing ablation arms are reported "not available" instead of 0; robustness CVs keep 3 decimals and undefined CVs are NaN; full and `_subset` runs are not mixed; canned conclusions removed from `comparison_report.md`.
- **>8-bit images** are rescaled instead of wrapping modulo 256; saturation QC uses the dtype maximum.
- **Mask cache key** = preprocessed image + Cellpose params + seed + model SHA-256 (no longer the slug), so identical arms share masks; corrupt entries are recomputed.
- **Config honoured or rejected:** `model.use_gpu`, `model_type`, Gaussian denoise (`sigma_um`), `blank_glob`; method names are validated enums; size gates derive from `reference_vol_um3`; dead keys removed; `do_3D: true` rejected. Provenance lists only steps that ran.
- **Filenames:** generalised pattern (cell line, day, decimal/underscore materials); unparseable names are skipped with a warning instead of pooled into "Unknown"; uncalibrated files are refused unless `allow_uncalibrated: true`.
- `004_D_3d`/`204_D_3d` now run 2D (identical to 001/201; comparators warn).

#### Fixed — crashes and usability
- `--scope subset` slugs are `<slug>_subset` (could previously overwrite full runs); comparator chosen by a `comparator:` key; resume via `RUN_COMPLETE.json` (`--overwrite` re-runs); `--strict-pairing`; `--dry-run`; stratified `sample_n`; `main()` returns an exit code; preflight fails on missing weights with the fetch command; log handlers no longer duplicate/accumulate; progress with ETA; each field read once per combo.

## [0.2.1] - 2026-10-24

Verification of 0.2.0 against the post-0.1 review spec, including a full real-data run, found the problems below.
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
