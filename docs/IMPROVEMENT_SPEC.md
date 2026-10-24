# matrix-mat — Code Review & Test-Driven Improvement Spec

*Reviewed: 2026-10-23 at commit `75060eb` (main, in sync with origin). Scope: `spheroid_pipeline_v2/` (brightfield), `spheroid_if_sweep/` (immunofluorescence), configs, tests, CI, packaging, docs. Untracked local folders are covered only under hygiene (§1.6).*

Finding IDs (e.g. **S1**) are used for cross-reference in the spec (§3). Every finding was checked against the code, and line numbers refer to `75060eb`.

---

## 0. Executive summary

The repository is well past "notebook science". It has packaging, configs, provenance JSON, SHA-256 manifests, QC flags rather than silent drops, seeded resampling, and a synthetic fixture. The problems are concentrated in four places:

1. **The statistics answer the wrong question.**
   - The unit of analysis is the spheroid (brightfield) or the field (IF), not the gel or biological replicate. The result is pseudo-replicated p-values and CIs that are too narrow.
   - The IF "density per mm³" is not a volumetric density; it scales as about 1/d.
   - Several reported conclusions are hard-coded strings rather than computed values.
2. **The method can change per image without leaving a trace.**
   - The segmentation backend is chosen per image after looking at the output.
   - The "cyto3/cpdino" fallback actually re-runs `cpsam`.
   - The backend, pixel-size source and calibration status are not stored per object.
   - The mask caches are keyed only by filename, so stale or synthetic masks can be reused.
3. **The test suite gives false confidence.** About 190 of the CI tests never call production code. The tests that do exercise real code (m1–m4) are not run in CI, and the E2E test only checks that files exist.
4. **Docs and CLI have drifted.**
   - Three README commands fail at argparse.
   - The config aliases silently override explicit values.
   - Paths are cwd-relative and hard-coded to `Experimental data /Chuling cells/...`.

The recommended order is: **safety-net tests → scientific correctness → provenance/reproducibility → restructure → docs**. Do not restructure first. Moving code before characterisation tests exist makes it impossible to tell refactor regressions from intended fixes.

---

## 1. Findings

Severity: 🔴 high (can change a scientific conclusion or silently corrupt results), 🟠 medium, 🟡 low.

### 1.1 Scientific rigor

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| S1 | 🔴 | **Pseudo-replication (brightfield).** Bootstrap CIs and permutation tests resample individual spheroid pairs. Spheroids from the same gel/well are not independent, so effective n is overstated and p-values are anti-conservative. `replicate` (gel) is parsed but never used as a statistical unit. | `stats.py:59-147`, `:228-309` |
| S2 | 🔴 | **Pseudo-replication (IF).** Mann-Whitney is run on per-field densities, and several gels contribute 2 fields. S46D10 has n=1 field. | `biological_plots.py:113-120`, `summaries.py:49-56` |
| S3 | 🔴 | **IF density is not a volumetric density.** It is nuclei counted in one optical section ÷ (π/6)·d³ of the projected actin blob, so it scales ≈ 1/d: larger spheroids look less dense at identical packing. All 55 raw TIFFs are single-plane uint8, and nuclear "volumes" are also spheres extrapolated from 2D area. | `features.py:60-63`, `:197`; `cli.py:274`; `postprocess.py:62-70` |
| S4 | 🔴 | **Hard-coded conclusions in generated reports**, e.g. "100% eliminated", "+2 flags", "RESOLVED", "No count collapse across any of the 27 fields", "Mat-Gel1-3 exhibits severe focal drift". These print regardless of the data. | `comparator_prepost.py:455-557`, `comparator.py:250-260` |
| S5 | 🔴 | **Paired data tested with an unpaired test.** The Day0 vs Day7 volumes of tracked pairs use `mannwhitneyu(v7, v0)`. The statistics are also computed in two plotting/report modules rather than in `stats.py`. | `visualize.py:430`, `report.py:196` |
| S6 | 🔴 | **Method chosen per image by looking at the result.** Zero objects, or one mask over 25 % of the FOV, switches the whole image to another algorithm. Images in one condition are measured by different methods with different boundary bias: historical `DECISIONS.md` shows 548 cpsam vs 249 classical. The backend is not recorded on objects (`mask_source=automated` for all 1507 rows). | `segment.py:382-402`, `:441-528`; `run_pipeline.py:497-510` |
| S7 | 🔴 | **"cpdino/cyto3" tier is cpsam again.** Cellpose ≥4.0.1 ignores `model_type`, so tier 2 re-runs `cpsam_v2` but logs `backend: "cpdino"`. The docs and YAML describe a model that never runs. | `segment.py:229-232`, `:288-291` |
| S8 | 🔴 | **Circular parameter selection (IF).** The size-gate reference (841 µm³, gate 210–3364 µm³) is derived from the cohort median of the baseline run on the same data. PP11 is then declared "winning" on that same data. Robustness is measured as absolute-count spread, which any object-removing step shrinks automatically. | `postprocess.py:28-31`; `comparator_prepost.py:215-219`; `cli.py:52-58` |
| S9 | 🔴 | **Outcome-defined "clean" subset (IF).** `flag_density_outlier` (1.5×IQR on the endpoint itself) feeds `flag_any`, which defines "clean". The figure title then says clean data "confirm" the trend, and that wording is hard-coded. `flag_blurry` flags the bottom 10 % by construction. QC runs on preprocessed images. | `qc.py:274-302`, `:323`; `biological_plots.py:194-214`; `cli.py:282` |
| S10 | 🟠 | **Order bug: size gate before clump splitting.** Clumps above `max_vol` are deleted instead of split, and the removals are reported as a quality gain. | `postprocess.py:276-300` |
| S11 | 🟠 | **Statistical hygiene.** There are no multiplicity corrections across the ~5 conditions vs control. FC is tested on the raw ratio scale instead of log. Well-level Wilcoxon is one-sided `greater` with no recorded pre-registration. The control gets `p=1.0` instead of NA. | `stats.py:299-309`; `presentation_plots.py:233` |
| S12 | 🟠 | **Silent analysis-population switch.** If a condition has 0 PASS pairs, stats silently fall back to PASS+REVIEW for that condition only, so different conditions are analysed under different inclusion rules. | `stats.py:252-258`, `:348-353` |
| S13 | 🟠 | **`log2_fc = 0` when FC ≤ 0.** A vanished spheroid is encoded as "no change". | `pair.py:180` |
| S14 | 🟠 | **Fragile timepoint parsing.** `"0" in timepoint` classifies t0 vs t7 in 4 places, so e.g. `Day10` would be t0. The fallback filename parser assigns `gel1/0001/t7` to any unmatched name, so unrelated images get paired. | `stats.py:233`, `:400`, `:430`; `pair.py:362`; `run_pipeline.py:119-126` |
| S15 | 🟠 | **Pixel-size fallback can be wildly wrong but marked calibrated.** The 72-dpi default gives 352.8 µm/px with `is_calibrated=True`. ImageJ `spacing=` (the Z step) is read as XY size. | `scale_extractor.py:269`, `:291-304` |
| S16 | 🟠 | **Unit/label errors.** The IF y-axis says "10³ nuclei/mm³", but the plotted values are not divided by 1000. A fallback spheroid volume of 1 µm³ makes density ≈ 10⁹ × count. | `biological_plots.py:124`, `:213`; `features.py:57` |
| S17 | 🟡 | **Classical tier uses the full-resolution pixel size on the downsampled image** (latent while the scale factor is 1.0). Nearest-neighbour upscaling of labels biases area by about perimeter/(2s). | `segment.py:120`, `:152`; `preprocess.py:117-129` |
| S18 | 🟡 | **IF dedup is a no-op.** A label image can't overlap itself, and the test asserts `n_removed >= 0`. | `qc.py:54-133` |

### 1.2 Reproducibility & provenance

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| R1 | 🔴 | **Mask caches keyed only by image_id** (BF) or by an incomplete parameter key with no model hash (IF). Changing thresholds or the backend silently reuses old masks. The test fixture shares an image_id with real data: `masks/Day0 Mat gel1_0001_TRANS.png` is currently a **512×512 synthetic mask** in the real-data cache. | `mask_cache.py:35-40`; `segment.py:420-430`; IF `cli.py:215-226` |
| R2 | 🔴 | **IF baseline-mask shortcut** reuses `runs/001_A_baseline/masks/*` (which are *final*, post-processed masks) as raw masks for any combo matching flow/cellprob. It ignores diameter, min_size and model. | `spheroid_if_sweep/cli.py:223-226` |
| R3 | 🔴 | **Defaults disagree, so provenance is wrong.** `sweep.yaml` has no pre/post sections. The module defaults are `enabled=True` and so they run, but `cli.py` reads `.get("enabled", False)` and records them as off in `run_metadata.json`. | `preprocess.py:295`; `postprocess.py:298`; `cli.py:196`, `:351-352` |
| R4 | 🟠 | **Scale source and calibration status are discarded.** `manifest.csv` writes `pixel_size_um=1.518817, scale_source=default_fallback` for every row. SHA-256 hashing is "skipped". Preflight checks tag 37510 only on the first file. | `run_pipeline.py:409`, `:464`; `provenance.py:202-203`; `preflight.py:217` |
| R5 | 🟠 | **Unpinned model weights.** `models/cpsam_v2` (1.2 GB) is gitignored, with no checksum or fetch script. A missing path silently downloads whatever is current. Provenance records `"cpsam"` but no weights hash. | `segment.py:221`; IF `engine.py:71-78` |
| R6 | 🟠 | **Lock file is unused.** CI and the docs install the floating floors (`torch>=2.1`, `cellpose>=4.0`). The lock has no hashes and pins versions that may not support py3.10. There are no torch seeds or deterministic flags, and MPS/CUDA nondeterminism is undocumented. | `ci.yml:29-33`; `requirements-lock.txt` |
| R7 | 🟠 | **Everything is cwd-relative.** `DECISIONS.md` (append-only across runs: 1324 mixed entries), `masks/`, `models/`, `runs/` and data paths all resolve against the working directory. Tests silently skip when run from elsewhere. | `config.py:46`, `:54`; `segment.py:373-374` |
| R8 | 🟠 | **Sample-12 "tuning loop" does nothing.** It sets the same constant every cycle, reads the cached masks anyway, and mutates the shared config, which leaks into metadata. | `run_pipeline.py:543-554`, `:814` |
| R9 | 🟡 | **Full resolved config not saved** in BF `run_metadata` (`to_yaml` is never called). Older IF runs show `git.commit = "unknown"`. | `provenance.py:103-146` |

### 1.3 Maintainability

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| M1 | 🟠 | **Config aliases silently override explicit values**, e.g. `SegmentationConfig(primary_backend="classical")` gives `"cpsam"`, and the same pattern hits `single_mask_max_area_pct` and `min_t0_d_um_denominator`. There are also dead keys (`cpsam_model_path`, `hough_*`, `target_diameter_px`, IF `overwrite`, `use_gpu`, `model_type`, …) and two config systems (dataclasses vs raw dict `.get` with scattered defaults). | `config.py:59-61`, `:90-91`, `:115-116`, `:166-176`; IF `cli.py:362-431` |
| M2 | 🟠 | **God functions.** `run_spheroid_pipeline` is ~325 lines. IF `run_sweep` is ~400 lines and mixes I/O, caching, QC, provenance and dispatch. Comparator dispatch is chosen by config *filename substrings*. | `run_pipeline.py`; IF `cli.py:61-461`, `:454` |
| M3 | 🟠 | **Duplication.** The volume-from-area formula appears ×4 in IF and again in BF. `provenance.get_git_info`/`get_gpu_info`/sha256 exist twice. Manual-review logic is copied inline without the shape check. Two filename regexes and two IF pairing rules (preflight accepts `.tiff`, pairing does not) can disagree. | `postprocess.py:62`; `qc.py:193`, `:212`; `features.py:130`; `run_pipeline.py:474-494` vs `review.py:128-179` |
| M4 | 🟠 | **Production code imports test code** (`run_pipeline.py:61` imports `tests.synthetic_generator`). The IF package imports the BF package for scale extraction (`pairing.py:15`). | |
| M5 | 🟡 | **Statistics live in plotting/report modules.** Globals are mutated by `np.random.seed(42)`. The redundant `Visualizer` wrapper and dead outer `except` blocks remain. The O(N·H·W) plausibility check could be a single `np.bincount`. | `visualize.py:333`, `:799-833`; `presentation_plots.py:223`; `segment.py:396` |
| M6 | 🟡 | **Magic numbers without cited sources.** These include `1.518817` as a default in ~15 signatures, a hard-coded 1536×2048 FOV, `h_maxima=10` in intensity units (meaning differs with/without normalisation), a pixel-unit Gaussian kernel, and `clearance_um 5.85`. | |

### 1.4 Usability & documentation

| ID | Sev | Finding |
|---|---|---|
| U1 | 🔴 | Three README IF commands do not exist: `preflight --data-dir`, `run --config`, and `compare-prepost --biological`. The README calls `sweep.yaml` the "winning production config", but it is the 4-combo hyperparameter sweep. |
| U2 | 🟠 | The README `--plots` step claims to "run analysis", but it only re-plots and fails on a fresh clone. It still references the removed `config.yaml` (and so does `preflight.py:99,112`). It also uses `cd matrix-mat` and an `.venv` that a clone does not contain. `--input-dir` and `--full-dataset` are no-ops, and `--plots`/`--preflight-only` are outside the mutually exclusive group. |
| U3 | 🟠 | Data paths contain a trailing space and a colon (`"Experimental data /"`, `"spheroids staining:IF "`). They are hostile to shells, Windows and other machines. There is no `DATA_ROOT` concept. |
| U4 | 🟡 | Marketing language overclaims ("publication-grade", "Strict Provenance" badge, "Zero drops", "deterministic"), given S1–S9 and R1–R3. |

### 1.5 Testing & CI

| ID | Sev | Finding |
|---|---|---|
| T1 | 🔴 | **Tier 1–3 tests (~190) never import production code** except `synthetic_generator`. They re-implement logic inline or assert tautologies (`assertEqual("cpsam","cpsam")`). `test_synthetic_smoke.py` tests its own reference watershed. |
| T2 | 🔴 | **The m1–m4 tests that do exercise production code are not in CI.** The only E2E (`test_synthetic_dataset_fixture.py`) asserts that files exist, not any values. |
| T3 | 🟠 | **CI smoke test pulls about 1.2 GB of weights on each of 8 matrix jobs.** If the download fails it silently falls back to classical, so the job passes regardless and nobody knows which backend ran. |
| T4 | 🟠 | **Tests write into the repo** (`masks/`, `DECISIONS.md`), which is the root cause of the R1 collision. |
| T5 | 🟡 | **Structure.** There are two test roots, a custom runner (`run_e2e_tests.py`), unittest rather than pytest, and milestone-based test names (`m1_adversarial_challenger`) that say nothing about behaviour. |
| T6 | 🟡 | **Untested areas.** Scale tags (72-dpi, ImageJ), backend recording, cache invalidation, filename fallback, config precedence, density formula, post-processing order, comparators, CLI parsing (would have caught U1). |

### 1.6 Repository hygiene (local working copy)

- **Untracked work in the tree.** `NMR/`, `Rheology/`, `Reproducing paper /` (which contains a publisher PDF and supplement), root-level `nmr_audit_*.png/pdf`, `pegdbco_nmr_audit_summary.json`, `scratch_nmr_overview.png` and four untracked `scripts/*.py`. These are separate analyses (rheology/NMR/paper reproduction) unrelated to the spheroid pipelines.
- **Nested repos.** `Hydrogel_BO/` and `peptide_materials_ml/` are nested repositories inside the working tree (gitignored).
- **Large outputs in the checkout.** `output/` (1.6 GB), `runs/` (3.0 GB), `models/` (1.1 GB) and `masks/` all live inside the checkout.
- `.agents/` scratch and `.DS_Store` files are present (gitignored).

---

## 2. Cleanup plan: how and where

### 2.1 Target layout

```
matrix-mat/
├── pyproject.toml              # single source of deps; extras: [dl], [dev]; [tool.pytest], [tool.ruff]
├── requirements-lock.txt       # generated by `uv pip compile --generate-hashes`; used by CI
├── src/matrix_mat/
│   ├── core/                   # shared, dependency-light, 100% unit-tested
│   │   ├── units.py            # area→volume, px↔µm, ONE implementation each
│   │   ├── scale.py            # ← spheroid_pipeline_v2/scale_extractor.py (+ plausibility)
│   │   ├── naming.py           # filename → (condition, gel, fov, timepoint); strict, no fallbacks
│   │   ├── stats.py            # hierarchical bootstrap, gel-level permutation, Holm/BH, log-FC
│   │   ├── cache.py            # content-addressed mask cache (image hash + params + model hash)
│   │   ├── provenance.py       # merged from both packages; writes resolved config + weights SHA
│   │   └── paths.py            # DATA_ROOT / OUTPUT_ROOT / MODELS_ROOT resolution (env + config)
│   ├── brightfield/            # ← spheroid_pipeline_v2 (segment, qc, measure, pair, pipeline)
│   ├── if_density/             # ← spheroid_if_sweep (preprocess, postprocess, features, qc, sweep)
│   ├── reporting/              # figures + md/html; consumes stats tables, computes NO statistics
│   └── cli.py                  # one entry point: `matrix-mat bf ...`, `matrix-mat if ...`
├── configs/                    # validated against a schema (pydantic) — unknown keys = error
├── tests/
│   ├── conftest.py             # tmp_path-based OUTPUT/CACHE roots, synthetic builders
│   ├── unit/                   # pure functions, fast, no torch
│   ├── integration/            # pipeline on synthetic fixture, classical backend, golden values
│   ├── regression/             # golden CSVs (snapshot) for refactor safety
│   └── slow/                   # @pytest.mark.slow / @pytest.mark.dl — cellpose, real data
├── scripts/                    # one-off utilities; never imported by the package
├── tools/fetch_models.py       # downloads weights, verifies SHA-256 against models.lock.json
└── docs/
```

**Where the untracked analyses go.** Move `NMR/`, `Rheology/`, `Reproducing paper /` and the NMR/rheology scripts and plots into their own repository, or into `analyses/<topic>/` with its own README. Do not commit the publisher PDF or supplement; reference them by DOI. Move `output/`, `runs/`, `masks/` and `models/` outside the checkout, under paths set by `MATRIX_MAT_DATA_ROOT` / `MATRIX_MAT_OUTPUT_ROOT`, or keep them in place but gitignored.

### 2.2 How to migrate (order matters)

1. **Phase 0 — Safety net (no behaviour change).**
   - Switch to pytest, using `pytest` collection of both existing roots.
   - Add golden-output regression tests on the synthetic fixture with the classical backend forced.
   - Isolate all test I/O to `tmp_path`.
   - Delete the tautological tier 1–3 tests *after* their real-code replacements land.
2. **Phase 1 — Scientific correctness (§3.1).** Each fix has an intentional, reviewed golden-file update. Do these on the current layout so the diffs are small and reviewable.
3. **Phase 2 — Provenance and reproducibility (§3.2).** Cache, weights, config schema, paths.
4. **Phase 3 — Restructure.**
   - `git mv` into `src/matrix_mat/…`, keeping thin re-export shims in `spheroid_pipeline_v2/__init__.py` and `spheroid_if_sweep/__init__.py` for one release, with a `DeprecationWarning`.
   - Golden tests must pass unchanged through the move.
   - Then deduplicate into `core/`.
5. **Phase 4 — Usability and docs (§3.4).** Unified CLI, README regenerated from `--help`, honest claims.

Each phase is one PR, or one PR per workstream inside it. Every PR must keep CI green, and every PR that changes numbers must include a short `CHANGELOG` note explaining *which* scientific outputs moved and why.

---

## 3. Test-driven improvement spec

Conventions:
- **Red → Green → Refactor.** The listed tests are written first and must fail on `75060eb`, or, where marked *characterisation*, must pass and pin current behaviour.
- **Test locations** use the target layout. During Phase 1–2, put them in `tests/unit/` etc. and import from the current package names.
- **Fixtures** are tiny synthetic arrays built in `conftest.py`. No real data, no network and no torch in `unit/` or `integration/`.
- **Acceptance criteria (AC)** are the done-definition for each item.

### 3.0 Phase 0 — Test infrastructure

**WS0.1 — Single pytest harness**
- Tests: `tests/test_harness.py::test_no_test_writes_outside_tmp`. An autouse fixture snapshots `git status --porcelain` before and after the session, and the test asserts nothing new appears under `masks/`, `DECISIONS.md`, `output/` or `runs/`.
- Implementation:
  - Add `[tool.pytest.ini_options]` with `testpaths`, markers `slow`, `dl` and `realdata`, and `--strict-markers`.
  - Add a `conftest.py` that sets `MATRIX_MAT_OUTPUT_ROOT`, the cache dir and the decisions log path to `tmp_path_factory`.
- AC:
  - `pytest -m "not slow and not dl and not realdata"` completes in under 60 s on CPU with no torch import (`python -X importtime` check, or `sys.modules` assertion in a test).
  - The custom `run_e2e_tests.py` is deleted.

**WS0.2 — Golden-output characterisation**
- Tests: `tests/regression/test_bf_golden.py::test_synthetic_fixture_matches_golden`.
  - Run the BF pipeline on `tests/fixtures/synthetic_dataset` with `backend=classical`, cache disabled, and output to `tmp_path`.
  - Compare `objects.csv`, `pairs.csv` and `condition_summary.csv` to `tests/regression/golden/*.csv` with `pandas.testing.assert_frame_equal(rtol=1e-6)` on numeric columns.
  - Provide an `--update-golden` option to regenerate.
- Same for IF: build a 2-field synthetic IF fixture (ch00 nuclei disks, ch02 actin disk, single plane) and use a stub segmenter (see WS2.4).
- AC: the golden files are committed. Any later change to numbers must touch the golden files in the same PR.

**WS0.3 — Ground-truth accuracy (replaces "files exist")**
- Tests: `tests/integration/test_bf_accuracy.py`.
  - `test_known_diameter_recovered`: a synthetic dark disk of d=200 µm at 1.5 µm/px gives a measured `equivalent_diameter_um` within ±3 %.
  - `test_known_fold_change_recovered`: a Day0 disk d=100 µm and a Day7 disk d=150 µm at the same centroid pair, with `fold_change_volume` ≈ 3.375 ± 5 %.
  - `test_bright_blob_rejected`: a bright object gives 0 PASS objects.
- AC: these replace `test_synthetic_smoke.py`'s reference-implementation tests, which are deleted.

**WS0.4 — Purge tautological tests**
- Action: for each test in `test_tier1_features.py`, `test_tier2_boundaries.py` and `test_tier3_pairwise.py`, either rewrite it against the real function or delete it.
- Tests:
  - A meta-test, `tests/test_meta.py::test_every_test_module_imports_production_code`. It uses AST scanning to check that each `tests/unit|integration/*.py` imports from `matrix_mat` or a legacy package (excluding fixture builders).
  - A meta-test that grep-fails on `assertEqual(<literal>, <same literal>)`.
- AC: test count may fall, but `pytest --cov=spheroid_pipeline_v2 --cov=spheroid_if_sweep` coverage *increases*. Record the before/after numbers in the PR.

### 3.1 Phase 1 — Scientific correctness

**WS1.1 — Hierarchical statistics with the correct unit of analysis** (S1, S2, S11, S12)
- New API in `core/stats.py`:
  - `summarise_by_unit(df, value, unit_cols=("condition","gel"), agg="median")`
  - `hierarchical_bootstrap_ci(df, value, cluster="gel", stat=np.median, n_boot, seed)`, which resamples gels and then objects within gels.
  - `cluster_permutation_test(df, value, group_col, cluster="gel", n_perm, seed)`, which permutes condition labels at gel level and is exact when the number of permutations is small.
  - `adjust_pvalues(p, method="holm")`
- Tests (`tests/unit/test_stats_hierarchical.py`):
  - `test_bootstrap_ci_widens_with_cluster_correlation`: simulate 4 gels × 30 spheroids with a gel random effect (σ_gel=0.5, σ_within=0.1 on log FC). The hierarchical CI must be ≥2× wider than the naive per-spheroid CI.
  - `test_cluster_permutation_type1_error`: simulate 500 null datasets (4 vs 4 gels, strong gel effect, no condition effect). The rejection rate at α=0.05 must be ≤ 0.07. The naive per-spheroid test must exceed 0.15 on the same data (documents why the change matters).
  - `test_cluster_permutation_exact_when_few_gels`: with 3 vs 3 gels the minimum achievable p is 1/C(6,3)=0.05. Assert the function enumerates all 20 splits and returns p ≥ 0.05.
  - `test_holm_adjustment_matches_statsmodels`: compare against `statsmodels.stats.multitest.multipletests` (dev dependency only).
  - `test_fold_change_tested_on_log_scale`: FC values {0.5, 2} vs {1, 1} give a symmetric log2 statistic of 0 difference in median.
  - `test_control_p_value_is_na_not_one`
  - `test_no_silent_population_fallback`: a condition with 0 PASS pairs and `include_review_in_stats=False` gives `n_stat=0`, NaN statistics, and an explicit `analysis_population="PASS"` column. It must not include REVIEW pairs.
- `condition_summary.csv` gains these columns:
  - `n_gels`, `n_spheroids`
  - `analysis_population`
  - `log2fc_median`, `ci_method`
  - `p_raw`, `p_adj`, `p_adj_method`
  - `min_achievable_p`
- AC:
  - The README states that inference is at gel level.
  - Spheroid-level numbers are labelled descriptive.
  - The IF pipeline computes density per field, then aggregates to gel, then tests. S46D10 (n=1 gel) is reported as "insufficient replicates", with no p-value.

**WS1.2 — Correct paired testing; stats out of plotting code** (S5, M5)
- Tests:
  - `tests/unit/test_reporting_no_stats.py::test_reporting_does_not_import_scipy_stats`: an AST check that `reporting/*`, `visualize.py`, `presentation_plots.py` and `report.py` don't call `scipy.stats`.
  - `tests/unit/test_stats_paired.py::test_paired_day0_day7_uses_signed_rank`: given 10 pairs where v7 = 1.1·v0 with tiny noise, the paired test p < 0.01 while the unpaired MWU p > 0.3. This documents the power and correctness difference.
- AC: all p-values shown in figures come from a stats table passed into the plotting function, and figures carry the n (gels) in the annotation.

**WS1.3 — IF density metric** (S3, S16)
- Decision required: either (a) report **areal density** (nuclei per mm² of spheroid cross-section in the imaged plane), or (b) require z-stacks and compute true 3D density. Default to (a) for the existing single-plane data.
- Tests (`tests/unit/test_if_density.py`):
  - `test_areal_density_invariant_to_spheroid_size`: synthetic hexagonally packed nuclei (spacing 12 µm) inside disks of radius 50, 100 and 200 µm. The areal density must agree within ±5 % across sizes. The current volumetric metric must fail this (assert ratio ≈ 1/d as a documented-regression test, then delete).
  - `test_density_units_label_matches_values`: the plot y-label scale factor equals the divisor applied.
  - `test_missing_spheroid_mask_yields_nan_density`: replaces the 1 µm³ fallback.
  - `test_single_plane_input_refuses_volumetric_metric`: calling the 3D metric on a 2D image raises `ValueError`.
- AC: column renamed to `nuclei_per_mm2`. Old run comparisons are regenerated, and the CHANGELOG explains the change.

**WS1.4 — Computed, not hard-coded, conclusions** (S4, S9)
- Tests (`tests/unit/test_comparator_claims.py`):
  - `test_report_text_changes_with_data`: feed the comparator two synthetic summary tables with opposite effects. The generated markdown must differ in every sentence containing a claim word (`eliminated|confirmed|resolved|collapse|drift`).
  - `test_no_literal_claims_in_source`: grep the comparator source for those words outside f-strings that reference variables, and fail if any are found.
  - `test_clean_subset_excludes_no_endpoint_flags`: the `flag_any` composition must not include `flag_density_outlier`. The outlier flag is reported separately as a sensitivity analysis.
  - `test_blur_flag_uses_absolute_threshold`: a cohort of uniformly sharp images flags 0 as blurry.
  - `test_qc_metrics_computed_on_raw_image`: the saturation fraction on a raw image with no 255 values is 0, even when preprocessing is enabled.
- AC: narrative interpretation moves to a clearly labelled hand-written `docs/findings/*.md`, which is not generated.

**WS1.5 — Segmentation method consistency** (S6, S7)
- Policy: one backend per run, set by config. If an image produces 0 objects or an implausible mask, the image is flagged `REVIEW`/`SEG_FAIL` and its objects are excluded from inference. It is not re-segmented with another algorithm. A separate opt-in `fallback_policy: per_image` remains for exploration and is logged loudly.
- Tests (`tests/unit/test_segment_policy.py`, using a stub backend that returns a configured mask):
  - `test_implausible_mask_flags_image_not_switches_backend`
  - `test_backend_recorded_on_every_object`: `objects.csv` has non-null `seg_backend`, `model_id` and `diameter_used`, including on cache hits (the cache stores the original backend).
  - `test_fallback_policy_per_image_requires_opt_in`
  - `test_condition_summary_reports_backend_mix`: adds a `backend_counts` column.
  - `@pytest.mark.dl test_cellpose_model_identity`: when the config asks for `cpdino`, the loaded model's `pretrained_model` path or name contains `cpdino`. Otherwise it raises (no silent cpsam).
- AC: the tier-2 "cyto3" text is removed from the docs and YAML, or the model is actually loaded with `pretrained_model=`.

**WS1.6 — IF post-processing order and selection circularity** (S8, S10, S18)
- Tests:
  - `test_split_runs_before_size_gate`: a synthetic clump of 4 touching nuclei with total volume > `max_vol` gives 4 objects after post-processing, not 0.
  - `test_size_gate_reference_from_config_not_cohort`: the reference volume comes from config (literature/manual-annotation value with citation field). Running on two cohorts with different medians uses the same gate.
  - `test_robustness_metric_is_relative`: spread is a CV, so removing a constant fraction of objects leaves it unchanged.
  - `test_dedup_removed_or_documented`: either the module is removed, or the test asserts it is a documented no-op on label images.
- AC: config selection is validated on held-out fields with manual counts. Add `tests/slow/test_if_vs_manual_counts.py` (marked `realdata`) using a small committed CSV of manual counts, with a pre-declared acceptance (e.g. |bias| < 10 %, Spearman ρ > 0.9).

**WS1.7 — Parsing and edge-case correctness** (S13, S14, S15)
- Tests (`tests/unit/test_naming.py`, `test_scale.py`, `test_pair.py`):
  - `test_parse_timepoint` (parametrised): `Day0`→0, `Day7`→7, `D7`→7, `Day10`→10, `t0`→0. An unknown form raises.
  - `test_unmatched_filename_raises_or_flags`: no defaulting to `gel1/0001/t7`.
  - `test_timepoint_taken_from_source_directory_when_configured`
  - `test_72dpi_resolution_tag_not_trusted`: a TIFF with XResolution=72 and unit=inch gives a scale of `None` or `is_calibrated=False`, plus a warning.
  - `test_imagej_spacing_is_not_xy`: ImageJ metadata with `spacing=2.0` and resolution 0.65 µm/px gives 0.65.
  - `test_scale_plausibility_range`: values outside 0.05–20 µm/px are rejected.
  - `test_vanished_spheroid_log2fc_is_nan_and_flagged`: v7=0 gives `log2_fc` NaN and a `qc_reasons` entry, not 0.
  - `test_classical_tier_uses_effective_pixel_size`: with scale factor 0.5, the min-area gate in px is computed from `pixel_size/0.5`.
- AC: all four `"0" in timepoint` call sites go through `core.naming.parse_timepoint`.

### 3.2 Phase 2 — Reproducibility & provenance

**WS2.1 — Content-addressed mask cache** (R1, R2, T4)
- Key: `sha256(image bytes) ‖ sha256(canonical JSON of segmentation params) ‖ model weights sha256 ‖ package version`. The cache directory defaults to `<output_root>/cache/masks`.
- Tests (`tests/unit/test_cache.py`):
  - `test_cache_miss_when_any_seg_param_changes` (parametrised over every field of `SegmentationConfig`)
  - `test_cache_miss_when_model_hash_changes`
  - `test_same_filename_different_pixels_do_not_collide`: covers the current fixture-vs-real collision.
  - `test_cache_records_original_backend`
  - `test_if_baseline_shortcut_removed`: an AST/grep check that the IF CLI never reads `001_A_baseline/masks`.
- AC: after this lands, delete the existing `masks/` directory, as it holds contaminated synthetic masks.

**WS2.2 — Config schema and single source of defaults** (R3, M1)
- Use pydantic (or dataclasses + validation) models for both pipelines, with `extra="forbid"`. All defaults live in the model. No `.get(key, default)` outside it.
- Tests (`tests/unit/test_config.py`):
  - `test_unknown_key_rejected`
  - `test_explicit_value_not_overridden_by_alias`: `primary_backend="classical"` stays classical. Aliases raise a deprecation warning, and alias plus canonical together with conflicting values raises.
  - `test_every_config_field_is_read_somewhere`: an AST scan for attribute access to each field, which fails on dead keys such as `hough_*` and `overwrite`.
  - `test_shipped_configs_validate` (parametrised over `configs/*.yaml`)
  - `test_metadata_reflects_effective_steps`: run IF on the fixture with `sweep.yaml`, then assert `run_metadata.preprocessing.enabled` equals whether preprocessing actually ran (spy on `preprocess_field`).
  - `test_precedence_cli_over_yaml_over_default`
- AC: `run_metadata.json` embeds the full resolved config.

**WS2.3 — Per-image scale provenance** (R4)
- Tests:
  - `test_objects_csv_has_scale_source_and_calibrated` (non-null)
  - `test_manifest_reflects_per_image_scale`: with two fixture images, one carrying tag 37510 and one without, `manifest.csv` shows different `scale_source` values.
  - `test_uncalibrated_image_flags_review_by_default`
  - `test_manifest_contains_sha256` (not "skipped")
  - `test_preflight_checks_all_files_for_scale_tag`
- AC: there is a `--allow-uncalibrated` flag, off by default.

**WS2.4 — Model weights and determinism** (R5, R6, T3)
- Add `models.lock.json` (`{name, url, sha256, size}`) and `tools/fetch_models.py`. The backend refuses to load weights whose SHA doesn't match, and never auto-downloads at analysis time.
- Add `core.repro.set_determinism(seed)`, which sets numpy Generator, torch, `torch.use_deterministic_algorithms(True, warn_only=True)` and cudnn flags.
- Replace global `np.random.seed` with local `np.random.default_rng`.
- Tests:
  - `test_missing_weights_raises_with_fetch_hint` (no network)
  - `test_weights_sha_mismatch_raises`
  - `test_run_metadata_records_weights_sha_and_backend`
  - `test_no_global_numpy_seed_calls`: grep for `np.random.seed(`.
  - `test_pipeline_bitwise_reproducible_cpu`: run the fixture twice with the classical backend and compare CSVs byte-for-byte.
  - `@pytest.mark.dl test_cpsam_reproducible_within_tolerance` (GPU/MPS: object count is identical and area within 0.5 %)
- A `StubSegmenter` (returns precomputed label masks from the fixture) makes the IF and BF integration tests run without torch.
- AC: the docs state the known MPS/CUDA nondeterminism and the measured tolerance.

**WS2.5 — Paths and outputs** (R7, U3)
- Add `core.paths` with `DATA_ROOT`, `OUTPUT_ROOT` and `MODELS_ROOT`, resolved from CLI, then env, then config. No cwd-relative defaults in library code. `DECISIONS.md` becomes `<run_dir>/decisions.jsonl`, one file per run, as structured JSON lines.
- Tests:
  - `test_library_never_writes_to_cwd`: run from a temp cwd and assert the cwd stays empty.
  - `test_decisions_log_is_per_run`
  - `test_no_hardcoded_data_paths_in_package`: grep `src/` for "Experimental data", "Chuling" and "/Users/".
- AC: the configs reference `${DATA_ROOT}/...`. Rename the data folders to remove the trailing space and colon, and document the mapping.

**WS2.6 — Dependency locking and CI** (R6, T2, T3)
- Generate `requirements-lock.txt` with hashes (`uv pip compile pyproject.toml --extra dl --generate-hashes`).
- CI:
  - **Fast job matrix** (3.10–3.13, ubuntu): `pip install -e .[dev]` (no torch), `ruff`, `mypy core/`, `pytest -m "not slow and not dl and not realdata"`, coverage ≥ 80 % on `core/` with a gate.
  - **Locked job** (1×, ubuntu, py3.12): `pip install -r requirements-lock.txt` then the integration and golden tests.
  - **DL job** (manual or nightly): cached weights verified by SHA, then `pytest -m dl`.
- Tests: `test_ci_smoke_reports_backend`. The smoke test asserts the backend it expected actually ran, so a silent fallback fails the job.
- AC: the 8× 1.2 GB downloads per push are gone.

**WS2.7 — Remove the fake tuning loop** (R8)
- Test: `test_sample_mode_does_not_mutate_config` (deep-copy compare before and after).
- AC: the loop is deleted. If calibration is wanted, it becomes `matrix-mat bf calibrate --holdout <list>`, which writes a frozen `configs/calibrated_<date>.yaml` and records the held-out image IDs. Analysis runs refuse to include the calibration images unless `--include-calibration-images` is passed.

### 3.3 Phase 3 — Restructure & deduplicate

**WS3.1 — `src/` layout with shims**
- Tests:
  - All golden and regression tests pass unchanged.
  - `test_legacy_imports_warn`: `import spheroid_pipeline_v2` emits `DeprecationWarning` and re-exports.
  - `test_package_does_not_import_tests`: `matrix_mat` must not import `tests.*` (an `importlinter` contract).
- Import-linter contracts:
  - `core` imports nothing from `brightfield`, `if_density` or `reporting`.
  - `if_density` does not import `brightfield`.
  - `reporting` does not import `scipy.stats`.

**WS3.2 — Single implementations**
- Tests: `tests/unit/test_units.py` covers `area_to_sphere_volume`, `px_to_um` and `equivalent_diameter`. These are property-based (Hypothesis) round trips, e.g. `diameter→area→diameter` is identity within 1e-9, and volume scales as s³ under unit change.
- Grep test: `test_single_volume_formula`. The expression `np.pi / 6.0` appears only in `core/units.py`.
- AC:
  - `provenance`, `preflight` patterns, filename parsing and review-mask ingestion each exist once.
  - `run_spheroid_pipeline` and `run_sweep` are each ≤ 80 lines and orchestrate named stages (`discover → preflight → segment → measure → pair → stats → report`).
  - Each stage is unit-tested with in-memory inputs.

### 3.4 Phase 4 — Usability & docs

**WS4.1 — Unified, honest CLI** (U1, U2)
- One entry point, `matrix-mat`, with subcommands `bf {preflight,run,plots}` and `if {preflight,sweep,compare,compare-prepost,compare-biological}`.
- Tests (`tests/unit/test_cli.py`):
  - `test_readme_commands_parse`: extract every fenced `bash` command in `README.md` and `docs/*.md` that starts with `matrix-mat`/`python -m`, then run `parser.parse_args()` on it (no execution). This would have caught U1.
  - `test_plots_without_csvs_gives_actionable_error`
  - `test_mutually_exclusive_modes`
  - `test_no_dead_flags`: every argparse dest is consumed.
- AC: `--input-dir`, `--full-dataset` and `--overwrite` are removed or implemented.

**WS4.2 — Docs truthfulness**
- Tests: `test_docs_reference_existing_files`, which link-checks relative paths in markdown.
- Rewrite the README quick-start for a fresh clone:
  1. `uv venv`
  2. `pip install -e .[dl]`
  3. `python tools/fetch_models.py`
  4. `export MATRIX_MAT_DATA_ROOT=…`
  5. `matrix-mat bf preflight`
- Remove the overclaiming badges and adjectives. Add a "Statistical methods" section stating the unit of analysis, the tests, the multiplicity correction, the effect sizes reported, and known limitations (2D→3D volume assumption, single-plane IF).
- Add `docs/DATA.md`, describing the expected directory structure, naming convention and how to obtain the data, plus `docs/VALIDATION.md` (manual-count agreement, backend comparison).

**WS4.3 — Repository hygiene**
- Move the untracked NMR, rheology and paper-reproduction work to a separate repo, or to `analyses/` with its own README. Do not commit publisher PDFs.
- Add a `pre-commit` config with ruff, end-of-file fixes and `check-added-large-files --maxkb=1000`.
- `test_no_large_files_tracked` in CI (size check on `git ls-files`).

---

## 4. Suggested PR sequence

| # | PR | Workstreams | Changes numbers? |
|---|---|---|---|
| 1 | pytest harness, tmp isolation, golden snapshots, accuracy tests | 0.1–0.3 | no |
| 2 | Delete tautological tests; CI runs m1–m4 equivalents | 0.4, 2.6 (fast job) | no |
| 3 | Content-addressed cache + delete contaminated `masks/` | 2.1 | possibly (stale masks) |
| 4 | Config schema, single defaults, provenance of effective steps | 2.2 | IF: yes (documents real pre/post state) |
| 5 | Naming/scale/pair edge cases | 1.7, 2.3 | possibly |
| 6 | Segmentation policy + backend recording + model identity | 1.5, 2.4 | **yes** |
| 7 | Hierarchical stats, Holm, log-FC, paired tests, stats out of plots | 1.1, 1.2 | **yes** (CIs widen, p rises) |
| 8 | IF density → areal; post-proc order; computed claims; clean-subset definition | 1.3, 1.4, 1.6 | **yes** |
| 9 | Paths/DATA_ROOT, weights lock, determinism | 2.4, 2.5 | no |
| 10 | `src/` restructure + dedupe + import contracts | 3.1, 3.2 | no (golden-gated) |
| 11 | Unified CLI, docs rewrite, hygiene | 4.x | no |

PRs 6–8 are the ones that can change scientific conclusions. Re-run the full real-data analysis after each, and keep a short `docs/findings/RESULTS_DELTA.md` recording how each headline number moved.

---

## 5. Definition of done (repository-level)

- [ ] `pytest -m "not slow and not dl and not realdata"` passes in < 60 s without torch; coverage ≥ 80 % on `core/`, ≥ 60 % overall.
- [ ] Every number in a generated report traces to a column in a CSV produced by `core.stats`; no hard-coded claims.
- [ ] Every object row records `seg_backend`, `model_id`, `scale_source` and `is_calibrated`; every run records the resolved config, the weights SHA, the lockfile hash and the git commit (dirty flag included).
- [ ] Inference is at gel level with multiplicity correction; spheroid/field-level statistics are labelled descriptive.
- [ ] Running from any cwd writes only under `OUTPUT_ROOT`; tests write only under `tmp_path`.
- [ ] Every README command parses; a fresh clone reaches `preflight` following the README alone.
