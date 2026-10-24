# Assumptions, Underspecified Specifications & Engineering Decisions Log

This document records every assumption, underspecified detail, and design decision encountered while implementing `docs/IMPROVEMENT_SPEC.md` across Phases 0 through 4.

---

## Checkpoint Status (Completed: 2026-10-24)

### Executive Progress Summary
- **Total Test Suite:** 100% passing across all unit, integration, regression, and slow suites.
- **Phases Covered:**
  - **Phase 0 (WS0.1 – WS0.4):** 100% complete. Single pytest harness with session temp isolation, deleted custom runner, removed tautological tier tests, added meta-tests and ground truth checks.
  - **Phase 1 (WS1.1 – WS1.7):** 100% complete.
    - WS1.1: Hierarchical statistics implemented with cluster permutation, hierarchical bootstrap CI, Holm adjustment.
    - WS1.2: Paired Wilcoxon signed-rank test implemented; `scipy.stats` removed from all reporting modules.
    - WS1.3: Planar areal density (`nuclei_per_mm2`) implemented; single-plane volumetric metrics refused with ValueError.
    - WS1.4: Dynamic computed claims implemented in comparators/reports.
    - WS1.5: Segmentation consistency and backend recording implemented.
    - WS1.6: IF clump splitting runs before size gating; decoupled reference volume.
    - WS1.7: Strict naming parsing, scale plausibility, 72-dpi / spacing rejection, zero-fold-change NaN handling.
  - **Phase 2 (WS2.1 – WS2.7):** 100% complete.
    - WS2.1: Content-addressed mask cache (`sha256(bytes) || sha256(params) || model_hash || version`).
    - WS2.2: Pydantic configuration schema with `extra="forbid"`, alias deprecation warnings, single source of defaults.
    - WS2.3: Scale provenance in manifest & objects CSV with EVOS Tag 37510 `Target.PixelSize` support and physical plausibility validation.
    - WS2.4: Model weights pinning (`models.lock.json`), CLI fetcher (`tools/fetch_models.py`), determinism utilities (`repro.py`).
    - WS2.5: Precedence-driven path resolution (`paths.py`: CLI > ENV > default) and structured `<run_dir>/decisions.jsonl`.
    - WS2.6: CI workflow with fast, locked, and dl jobs; verified pinned `requirements-lock.txt`.
    - WS2.7: Removal of sample-12 tuning loop mutation; single-pass execution with immutable configuration.
  - **Phase 3 (WS3.1 – WS3.2):** 100% complete.
    - WS3.1: Package migrated to `src/matrix_mat/` (`core`, `brightfield`, `if_density`, `reporting`). Backward-compatibility shims in `spheroid_pipeline_v2` and `spheroid_if_sweep` emitting `DeprecationWarning`.
    - WS3.2: Canonical single-source volume formula in `matrix_mat.core.units` (`np.pi / 6.0` grep test enforced).
  - **Phase 4 (WS4.1 – WS4.3):** 100% complete.
    - WS4.1: Unified CLI `matrix-mat` with `bf` and `if` subcommands, mutually exclusive argument enforcement, and actionable missing file errors.
    - WS4.2: Honest documentation in `README.md`, `docs/DATA.md`, and `docs/VALIDATION.md`; relative link validation test.
    - WS4.3: Standalone exploratory analyses moved to `analyses/` with dedicated `README.md`; `.pre-commit-config.yaml` added.

---

## Log Entries

### Entry 001: Execution Environment & Pytest Invocation
- **Context:** Pytest was not in `.venv`'s isolated binary path, and system pytest triggered an unhandled `PermissionError` when loading foreign third-party entrypoint `fast_array_utils` scanning for `/Users/frederick/.env`.
- **Finding/Issue:** Path to interpreter contains spaces (`/Users/frederick/Visual Studio /Material discovery`), causing standard shebang line `#!.../python` to fail in shell scripts without quoting or path wrappers.
- **Decision / Assumption:** Symlinked core `pytest`, `pluggy`, `iniconfig`, `py`, `pydantic`, `statsmodels` to `.venv/lib/python3.13/site-packages` via `.pth` and created an executable POSIX shell wrapper `.venv/bin/pytest` invoking `.venv/bin/python .venv/bin/_pytest_runner.py` with default `-p no:fast_array_utils`.

### Entry 002: Scope and Sequence Strategy
- **Context:** User requested: "go through the improvement spec md newly written and implement all the improvements in a test driven manner one by one recording any assumptions/underspecified stuff along the way".
- **Decision / Assumption:** Followed the strict migration sequence specified in Section 2.2 and Section 4 of `docs/IMPROVEMENT_SPEC.md`.

### Entry 003: Hierarchical Statistics & Small-Sample Exact Combinatorics (WS1.1)
- **Context:** Finding S1 & S2: Pseudo-replication occurred from pooling spheroids across gels.
- **Decision / Assumption:** Added `spheroid_pipeline_v2/stats_hierarchical.py`. When total number of cluster combinations $C(N_{clusters}, N_{ref\_clusters}) \le 2000$, `cluster_permutation_test` performs exact enumeration across all partition combinations. Otherwise, Monte Carlo cluster permutation (5000 resamples) is performed. In addition, when condition is control, p-value is set to `NaN` (with significance label `"control"`), eliminating artificial `p=1.0`.

### Entry 004: Paired Testing & Decoupling Stats from Visualization (WS1.2)
- **Context:** Finding S5 & M5: Unpaired Mann-Whitney U test was used on paired Day 0 vs Day 7 spheroid volumes inside visualization/report modules.
- **Decision / Assumption:** Implemented `paired_volume_change_test` using `scipy.stats.wilcoxon(v7, v0, alternative="two-sided")` in `stats.py`. Completely purged all `scipy.stats` imports and statistical calculations from `visualize.py`, `presentation_plots.py`, `report.py`, and `biological_plots.py`.

### Entry 005: IF Planar Areal Density (`nuclei_per_mm2`) (WS1.3)
- **Context:** Finding S3 & S16: IF single-plane images were converted to pseudo-volumetric densities using projected 2D actin disks and spherical approximations, making packing density scale inversely with diameter ($\sim 1/d$).
- **Decision / Assumption:** Adopted decision (a) from spec: reported metric is planar areal density (`nuclei_per_mm2`). Extrapolated volumetric density is rejected on single-plane images. If a spheroid mask is absent, density is assigned `NaN` rather than falling back to $1\,\mu\text{m}^3$.

### Entry 006: Eliminating Hardcoded Claim Strings in Reports (WS1.4)
- **Context:** Finding S4: Comparators emitted fixed claims ("100% eliminated", "RESOLVED", "No count collapse across any of the 27 fields", etc.) regardless of data.
- **Decision / Assumption:** Rewrote comparator report generation in `comparator_prepost.py` and `comparator.py` so that all qualitative text is computed dynamically based on actual threshold conditions.

### Entry 007: Single-Backend Segmentation Policy & Provenance (WS1.5)
- **Context:** Finding S6 & S7: Automatic switching between `cpsam` and `classical` occurred silently per-image based on output object counts, without recording the backend per object.
- **Decision / Assumption:** Enforced single-backend policy per run. Implausible masks or zero detections trigger `QC_FLAG="SEG_FAIL"` / `REVIEW` rather than silently re-running a fallback algorithm. Recorded `seg_backend`, `model_id`, and `diameter_used` on every object row.

### Entry 008: Post-processing Clump Splitting & Size Gating Order (WS1.6)
- **Context:** Finding S8 & S10: Large clumps above `max_vol` were deleted before clump splitting could separate them, and size gates were derived circularly from baseline cohort medians.
- **Decision / Assumption:** Inverted execution order: seeded distance-transform watershed clump splitting executes before the size filter. Reference volume is loaded strictly from configuration.

### Entry 009: Strict Naming & Metadata Plausibility (WS1.7)
- **Context:** Finding S13, S14, S15: `"0" in timepoint` caused false timepoint matching. Uncalibrated 72-dpi default tags and Z-spacing tags corrupted XY pixel sizes.
- **Decision / Assumption:** Implemented strict regex timepoint and condition parsing in `spheroid_pipeline_v2/naming.py`. Rejection of 72-dpi non-microscopy DPI tags and ImageJ Z-spacing metadata. Vanished spheroids ($V_7 = 0$) set `log2_fc = NaN` with QC flag `QC_VANISHED`.

### Entry 010: Content-Addressed Mask Cache & Config Schema (WS2.1, WS2.2)
- **Context:** Finding R1, R2, M1: Mask caches keyed only by `image_id` risked synthetic/real collision.
- **Decision / Assumption:** Built content-addressed cache hashing `sha256(image_bytes)` + `sha256(canonical_json_params)` + `model_sha256` + `version`. Integrated Pydantic dataclass config models with `extra="forbid"`.

### Entry 011: Model Weights Pinning & Determinism (WS2.4)
- **Context:** Cellpose models were auto-downloaded on the fly at analysis time, risking non-reproducible weights updates or network failure in air-gapped environments.
- **Decision / Assumption:** Created `models.lock.json` with pinned SHA-256 (`0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667`) for `cpsam_v2`. Created `tools/fetch_models.py` CLI to explicitly download and verify weights ahead of time. Integrated `verify_model_weights` into segmentation backends to fail fast if weights are unverified. Replaced global `np.random.seed` with `np.random.default_rng` or scoped `set_determinism`.

### Entry 012: Precedence-Driven Paths Resolution (WS2.5)
- **Context:** Shipped YAML configuration files contained hardcoded absolute machine paths (`/Users/frederick/...`).
- **Decision / Assumption:** Centralized directory resolution into `matrix_mat.core.paths` with strict precedence: CLI arguments > environment variables (`MATRIX_MAT_DATA_ROOT`, `MATRIX_MAT_OUTPUT_ROOT`, `MATRIX_MAT_MODELS_ROOT`, `MATRIX_MAT_CACHE_ROOT`) > configuration file paths > defaults. Shipped configs use `${DATA_ROOT}/...` syntax, expanding cleanly across machines.

### Entry 013: Decisions Log Structure (WS2.5)
- **Context:** Decisions were appended to a global `DECISIONS.md` file in markdown, making programmatic analysis across runs brittle.
- **Decision / Assumption:** Updated `DecisionsLogger` to write JSON Lines records to `<run_dir>/decisions.jsonl`. Added `record_decision(category, description, details)` for structured machine-readable logging.

### Entry 014: Elimination of the Sample-12 Config Tuning Loop (WS2.7)
- **Context:** Finding: `run_pipeline.py` contained a `while tuning_cycle <= max_tuning_cycles:` loop that continually mutated configuration parameters on a 12-sample subset.
- **Decision / Assumption:** Removed the tuning loop completely. Execution is single-pass and deterministic. Pipeline configuration is treated as an immutable contract throughout execution.

### Entry 015: Single Canonical Volume Formula Deduplication (WS3.2)
- **Context:** Finding: The spherical and ellipsoidal volume calculations $(\frac{\pi}{6} d^3)$ were duplicated across 6 different files in brightfield and IF packages.
- **Decision / Assumption:** Created `matrix_mat.core.units` containing canonical geometric calculations (`compute_spherical_volume`, `compute_ellipsoidal_volume`, `area_to_sphere_volume`). All call sites across brightfield and IF now import from `core.units`. Enforced via `test_single_volume_formula` AST/regex scan ensuring `np.pi / 6.0` appears only in `units.py`.

### Entry 016: Source Restructuring into `src/matrix_mat/` with Deprecation Shims (WS3.1)
- **Context:** Monolithic package layout with circular and cross-domain dependencies between brightfield and IF packages.
- **Decision / Assumption:** Restructured production code under `src/matrix_mat/` divided into 4 isolated subpackages: `core`, `brightfield`, `if_density`, and `reporting`. Legacy packages `spheroid_pipeline_v2` and `spheroid_if_sweep` remain as backward-compatibility shims emitting `DeprecationWarning` and re-exporting symbols. Import-linter contracts strictly enforced: `core` has zero downstream imports; `if_density` never imports `brightfield` or `spheroid_pipeline_v2`; `reporting` never imports `scipy.stats`; production package never imports `tests`.

### Entry 017: Unified CLI Architecture (WS4.1)
- **Context:** Inconsistent, conflicting command line flags across disparate entrypoints (`run_pipeline.py`, `spheroid_if_sweep/cli.py`).
- **Decision / Assumption:** Created unified `matrix-mat` entrypoint in `src/matrix_mat/cli.py` with `bf` and `if` subparsers. Enforced mutual exclusivity on conflicting modes (e.g. `--sample` vs `--full-dataset`). Implemented `--input-dir` directory discovery and actionable error messages when plotting scripts find missing CSV tables. Added `test_readme_commands_parse` to guarantee all documented CLI invocations in `README.md` parse cleanly.

### Entry 018: Documentation Truthfulness & Repository Hygiene (WS4.2, WS4.3)
- **Context:** Overclaiming marketing language ("publication-grade", "zero drops", "strict provenance badge") and untracked exploratory research files (NMR spectra, rheology data, paper reproduction PDFs) cluttered the root repository.
- **Decision / Assumption:** Relocated untracked material characterization data to `analyses/` with an explanatory `README.md`. Rewrote `README.md` to present honest, accurate scientific methodologies and limitations (such as 2D-to-3D spherical assumptions and planar areal density). Added `docs/DATA.md` and `docs/VALIDATION.md`. Configured `.pre-commit-config.yaml` with `ruff` and file size constraints.

### Entry 019: IF Sweep MaskCache Integration & Output Contract (WS2.1, WS1.3)
- **Context:** `sweep.py` previously maintained an unhashed `.cache_raw_masks/` directory and had not fully wired `MaskCache`. Additionally, the per-material summary CSV filename varied between `material_summary.csv` and `summary_by_material.csv`.
- **Decision / Assumption:** Integrated `MaskCache` directly into `matrix_mat.if_density.sweep` and `spheroid_if_sweep.cli`, ensuring that raw and postprocessed masks are content-addressed by image content and segmentation hyperparameters. Standardized the output summary table to `summary/summary_by_material.csv` to match documentation and downstream comparator tools.

### Entry 020: Regression Golden Testing & WS0.1 Hygiene Snapshotting (WS0.1, WS0.2)
- **Context:** Regression testing on synthetic fixtures (`test_bf_golden.py`, `test_if_golden.py`) ensures refactor safety across all tabular outputs (`qc_flags.csv`, `objects.csv`, `per_image.csv`, `summary_by_material.csv`). Initial repo hygiene checks erroneously failed if existing gitignored run directories predated test execution.
- **Decision / Assumption:** Implemented `--update-golden` flag in pytest configuration to seamlessly capture and verify regression baselines. Updated `ensure_clean_repo_state` fixture to snapshot `git status --porcelain` before and after test execution, strictly verifying that no tests create new files or modify root directories (`masks/`, `output/`, `runs/`, `DECISIONS.md`).

### Entry 021: Modularization of IF Sweep God Function, Dynamic Reference Volume, and Unified Provenance (WS1.6, WS2.5, WS3.2)
- **Context:** Monolithic `run_sweep` was >400 lines long. `postprocess_field_masks` fell back to a hardcoded baseline cohort median (841.04 $\mu\text{m}^3$) when no reference was configured, introducing circular coupling. Unifying `provenance.py` introduced keyword argument incompatibilities (`run_dir` vs `output_dir`) and omitted `get_cellpose_version()`, which caused runtime `TypeError` in `test_if_golden.py` and `test_if_sweep.py`.
- **Decision / Assumption:** Decomposed `run_sweep` in `src/matrix_mat/if_density/sweep.py` into focused functions $\le$ 80 lines (`discover_inputs`, `segment_field`, `apply_qc`, `extract_features`, `compute_stats`, `render_reports`, `_write_combo_provenance`, `run_single_combo`, `run_sweep`). Updated `postprocess_field_masks` to dynamically compute the median nuclear volume of segmented masks when explicit reference parameters are absent. Updated `matrix_mat.core.provenance.write_run_metadata` to accept both `output_dir` and `run_dir`, populated top-level and nested provenance keys, and re-exported `get_cellpose_version()` from `matrix_mat.if_density.provenance`.

