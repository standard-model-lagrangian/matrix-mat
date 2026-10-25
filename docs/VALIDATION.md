# Pipeline Validation & Verification

This page lists what has actually been checked, how, and where the evidence lives. Anything not listed here
has not been validated.

---

## 1. Synthetic ground truth (automated, runs in CI)

| Check | Ground truth | Acceptance | Test |
| :--- | :--- | :--- | :--- |
| Spheroid diameter recovery (classical backend) | dark disk, d = 200 µm at 1.5 µm/px | within ±3% | `tests/integration/test_bf_accuracy.py::test_known_diameter_recovered` |
| Volume fold-change recovery | d₀ = 100 µm → d₇ = 150 µm (FC = 3.375) | within ±5% | `test_known_fold_change_recovered` |
| Bright artefact rejection | bright disk on grey background | 0 PASS objects | `test_bright_blob_rejected` |
| IF planar density invariance to spheroid size | hexagonally packed nuclei in actin disks of r = 60, 120, 200 px | within ±5% (observed ±1.3%) | `tests/unit/test_if_density.py::test_areal_density_invariant_to_spheroid_size_through_feature_pipeline` |
| Clump splitting before size gate | 4 touching nuclei, total volume above the gate | 4 objects | `tests/unit/test_if_postprocess.py::test_split_runs_before_size_gate` |
| Gel-level statistics | simulated gel random effects | permutation type-I error ≤ 0.08 at α = 0.05; hierarchical CI ≥ 2× naive | `tests/unit/test_stats_hierarchical.py` |

## 2. Agreement with manual counts — NOT YET VALIDATED

`tests/data/if_manual_counts.csv` has no recorded provenance, and its values match the pipeline's own PP11
counts to within 0–4 nuclei (see `tests/data/README.md`). It therefore cannot serve as independent evidence.
The comparison test (`tests/slow/test_if_vs_manual_counts.py`) skips until each row records `counted_by`,
`count_date` and `count_method`.

No agreement statistic (R², bias, Spearman ρ) is currently claimed for the IF pipeline.

## 3. Segmentation backends

The pipeline runs **one backend per run** (`segmentation.fallback_policy: strict`). The backend, model id and
diameter are recorded on every object row (`seg_backend`, `model_id`, `diameter_used`), and each condition's
backend mix is reported in `condition_summary.csv` (`backend_counts`).

| Backend | Model | Status |
| :--- | :--- | :--- |
| `cpsam` | Cellpose-SAM `cpsam_v2`, SHA-256 pinned in `models.lock.json` | Default. Used for all 1507 objects in the 2026-10-24 real-data run. |
| `cpdino` | Cellpose foundation model `cpdino` | Needs the optional `dinov3` package; initialisation fails loudly if Cellpose loads any other model. Not benchmarked. |
| `classical` | Multi-scale adaptive threshold + watershed | Deterministic CPU backend used by the synthetic tests. Not benchmarked against cpsam on real data. |

Cellpose 4 has no `cyto3` model. Earlier documentation describing a cyto3 tier was wrong: that tier silently
re-ran cpsam.

No cross-backend comparison on real data has been performed. Earlier claims (e.g. "~15% more spheroids",
">60% artefact reduction") had no supporting analysis in the repository and have been removed.

## 4. Reproducibility

- **CPU / classical backend**: `objects.csv` is bitwise identical across repeated runs
  (`tests/unit/test_determinism.py::test_pipeline_bitwise_reproducible_cpu`).
- **cpsam on GPU/MPS**: object counts are identical and foreground area agrees within 0.5% on a synthetic image
  (`test_cpsam_reproducible_within_tolerance`, DL job). Bitwise identity is not guaranteed on GPU/MPS.
