# matrix-mat

[![CI](https://github.com/standard-model-lagrangian/matrix-mat/actions/workflows/ci.yml/badge.svg)](https://github.com/standard-model-lagrangian/matrix-mat/actions/workflows/ci.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Image analysis for cancer spheroids grown in engineered hydrogels. Two pipelines share one CLI:

| Pipeline | Input | Measures | Key output |
| :--- | :--- | :--- | :--- |
| **Brightfield** (`matrix-mat bf`) | Day 0 and Day 7 transmitted-light TIFFs (EVOS) | Spheroid size and shape; Day 0 → Day 7 volume fold change of tracked spheroids | `condition_summary.csv`, figures 1–6, `report.html` |
| **Immunofluorescence** (`matrix-mat if`) | Paired nuclear (`_ch00`) and F-actin (`_ch02`) confocal TIFFs | Nuclei per mm² of spheroid cross-section, nuclear morphology | `summary/summary_by_material.csv`, biological figures |

Segmentation uses [Cellpose-SAM](https://www.cellpose.org/) (`cpsam_v2`, checksum-pinned), with a classical CPU backend for tests. Statistical inference is done at the level of the **gel** (biological replicate), not the individual spheroid. Every run records its inputs, configuration, code version, dependency lock and model hash.

> **What's new in 0.2.1:** verification fixes for the 0.2 restructure. The IF configs no longer crash, and missing model weights now fail the run instead of returning empty results. CIs are on the correct scale and IF has gel-level tests. Thresholds are no longer derived from the data being judged, and dependencies are locked with hashes. CI now enforces lint, type checks and coverage. Details: [CHANGELOG](CHANGELOG.md) · real-data impact: [RESULTS_DELTA](docs/findings/RESULTS_DELTA.md).

---

## Contents

- [Installation](#installation)
- [Data layout](#data-layout)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Statistics and limitations](#statistics-and-limitations)
- [Development](#development)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)

---

## Installation

**Requirements:** Python 3.10–3.13 on macOS or Linux, about 1.3 GB of disk for the model weights, and optionally an Apple-silicon (MPS) or NVIDIA (CUDA) GPU. The steps below use [`uv`](https://docs.astral.sh/uv/); plain `pip` works the same way (see the note further down).

```bash
git clone https://github.com/standard-model-lagrangian/matrix-mat.git
cd matrix-mat

# 1. Environment
uv venv --python 3.12
source .venv/bin/activate

# 2. Exact, hash-verified dependencies (includes torch + cellpose), then the package itself
uv pip install --require-hashes -r requirements-lock.txt
uv pip install -e ".[dl,dev]" --no-deps

# 3. Model weights (~1.2 GB) into ./models, verified against models.lock.json
python tools/fetch_models.py
```

Check the install:

```bash
matrix-mat --help
pytest -m "not slow and not dl and not realdata"   # ~1 min, no GPU or data needed
```

> **Using `pip` instead of `uv`:** run `python -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements-lock.txt` (pip switches to hash-checking automatically) and `pip install -e ".[dl,dev]" --no-deps`.
>
> **No GPU:** everything runs on CPU, just more slowly. The device is detected automatically, or set it with `--device cpu|mps|cuda`.
>
> **Optional `cpdino` backend:** this needs `pip install git+https://github.com/facebookresearch/dinov3`. The default `cpsam` backend does not.

---

## Data layout

Paths are resolved from four roots, set by environment variables. Each has a sensible default:

| Variable | Default | Holds |
| :--- | :--- | :--- |
| `MATRIX_MAT_DATA_ROOT` | `./data` | input images |
| `MATRIX_MAT_OUTPUT_ROOT` | `./output` | results (`./output/runs` for IF) |
| `MATRIX_MAT_MODELS_ROOT` | `./models` | `cpsam_v2` weights |
| `MATRIX_MAT_CACHE_DIR` | `$OUTPUT_ROOT/cache/masks` | content-addressed mask cache |

The expected input layout:

```
data/
├── brightfield/
│   ├── Day0/   Day0 Mat gel1_0001_TRANS.tif, Day0 S34D30 gel2_0003_TRANS.tif, …
│   └── Day7/   Day7 Mat gel1_0001_TRANS.tif, …
└── if/         SKOV3 Spheroid D7 Mor_Mat-Gel1-2_ch00.tif  (+ matching _ch02.tif), …
```

Filenames encode condition, gel (the replicate) and field of view. A name that doesn't match the pattern raises an error rather than being guessed. Folders named differently can be symlinked into this layout or passed with `--t0-dir/--t7-dir` (brightfield) or `--data-dir` (IF). See [docs/DATA.md](docs/DATA.md) for the full conventions and calibration rules.

---

## Quick start

### Brightfield: Day 0 → Day 7 growth

```bash
export MATRIX_MAT_DATA_ROOT=/path/to/data

matrix-mat bf preflight                      # checks pairing, calibration, control group
matrix-mat bf run --sample 12                # quick look at 12 stratified images
matrix-mat bf run                            # full dataset → ./output
matrix-mat bf plots                          # re-draw figures from existing CSVs
```

The brightfield run writes to `./output/`:

- `objects.csv`: every spheroid, with morphology, QC flag and segmentation backend
- `pairs.csv`: tracked Day 0 → Day 7 pairs and their fold changes
- `condition_summary.csv`: gel-level statistics per condition
- `figures/`, `overlays/`, `contact_sheets/`, and `report.md` / `report.html`
- `run_metadata.json` and `manifest.csv`: full provenance

### Immunofluorescence: nuclear packing density

```bash
matrix-mat if preflight                                        # channel pairing, weights
matrix-mat if sweep --config configs/prepost_ablation.yaml     # 2×2 pre/post ablation → ./output/runs
matrix-mat if compare-prepost --biological                     # ablation report + PP11 figures
```

Each run is written to `./output/runs/<run>/`, containing `features/per_image.csv`, `qc/qc_flags.csv`, `summary/summary_by_material.csv` (gel-level tests) and `run_metadata.json`.

---

## CLI reference

| Command | Purpose | Useful options |
| :--- | :--- | :--- |
| `matrix-mat bf preflight` | Validate inputs before a run | `-c CONFIG`, `--t0-dir`, `--t7-dir`, `--pixel-size` |
| `matrix-mat bf run` | Segment, measure, pair, test, report | `--sample N`, `--condition NAME`, `--backend {cpsam,cpdino,classical}`, `--device`, `--no-cache`, `-o DIR` |
| `matrix-mat bf plots` | Figures from existing CSVs | `-o DIR` |
| `matrix-mat if preflight` | Validate IF channel pairs and weights | `--data-dir`, `--model-path`, `--allow-unpaired` |
| `matrix-mat if sweep` | Run every combo in a sweep config | `-c CONFIG`, `--runs-dir`, `--data-dir`, `--sample N`, `--overwrite` |
| `matrix-mat if compare` | Cross-run comparison | `--runs DIR` |
| `matrix-mat if compare-prepost` | Ablation report (+ `--biological` figures) | `--runs DIR`, `--biological` |
| `matrix-mat if compare-biological` | Figures for one run | `--run-dir DIR` |

Every command exits non-zero on failure, including missing weights, invalid configs and unknown backends. Run `matrix-mat <bf|if> <command> --help` for the full option list.

---

## Configuration

All analysis parameters live in `configs/`. Each file is validated against a schema, so a typo or unknown key is an error.

| File | Use |
| :--- | :--- |
| `spheroid_brightfield.yaml` | Brightfield pipeline: QC gates, pairing, statistics, artefacts (commented line by line) |
| `prepost_ablation.yaml` | IF 2×2 pre/post-processing ablation (PP00 / PP10 / PP01 / PP11) |
| `sweep.yaml` | IF Cellpose hyper-parameter sweep (no pre/post-processing) |
| `robustness_check.yaml` | IF hyper-parameter sweep with pre/post-processing |

Two IF thresholds are **provisional**. Both were derived from this dataset and are marked in the YAML:

- the reference nuclear volume, `reference_vol_um3: 841.04`
- the blur cut-off, `blur_abs_threshold: 150`

Replace them with independent values before publication.

---

## Statistics and limitations

- **Unit of inference: the gel.**
  - Brightfield: medians of log₂ fold change with a two-stage (gel → spheroid) bootstrap CI.
  - Both pipelines: conditions are compared with Mat by an exact gel-label permutation test, Holm-corrected across conditions.
  - Spheroid- and field-level numbers are descriptive only.
- **Paired growth:** Day 0 vs Day 7 volumes of tracked spheroids use a Wilcoxon signed-rank test.
- **QC never drops data silently.** Objects are flagged PASS, REVIEW or FAIL, or SEG_FAIL when segmentation failed. Statistics use PASS pairs, and the analysed population is named in the output.
- **Limitations**
  - **Power.** The current data has 2–4 gels per condition, so the smallest attainable p-value is about 0.03–0.33; after Holm correction nothing can reach 0.05. Read the results as effect sizes with CIs. See [RESULTS_DELTA](docs/findings/RESULTS_DELTA.md).
  - **Volume assumes a sphere.** Volume is estimated from 2D area, V = π/6 · d³.
  - **IF density is planar.** IF fields are single optical sections, so density is per mm² of cross-section; true 3D density needs z-stacks.
  - **Manual-count validation is pending.** The reference file has no recorded provenance; see [tests/data/README.md](tests/data/README.md).

What has and hasn't been validated is listed in [docs/VALIDATION.md](docs/VALIDATION.md).

---

## Development

```bash
pytest -m "not slow and not dl and not realdata"   # fast suite (what CI runs on 3.10–3.13)
pytest -m dl                                       # needs torch, cellpose and the weights
pytest --update-golden tests/regression            # regenerate golden outputs after an intended change
ruff check --select E9,F63,F7,F82 src tests tools
mypy src/matrix_mat/core --ignore-missing-imports
```

CI enforces ruff, mypy on `core/`, and test coverage of at least 60% overall and 80% on `core/`. Tests never write into the repository; a session-wide guard fails the run if one does.

Golden regression files compare integer columns exactly and float columns to within 0.5%, because OpenCV's 8-bit image operations round slightly differently on x86 and ARM.

---

## Repository layout

```
src/matrix_mat/
├── core/          # units, scale extraction, naming, statistics, cache, paths, provenance
├── brightfield/   # segmentation, QC, measurement, pairing, statistics, pipeline
├── if_density/    # preprocessing, post-processing, features, QC, sweep, comparators
├── reporting/     # figures and reports (display only, no statistics)
└── cli.py         # `matrix-mat` entry point
configs/           # validated YAML configs
tests/             # unit / integration / regression / slow, plus synthetic fixtures
tools/             # fetch_models.py
docs/              # guides, validation, findings
analyses/          # separate NMR / rheology / paper-reproduction work (not part of the package)
```

`spheroid_pipeline_v2` and `spheroid_if_sweep` are deprecated aliases kept for old imports. They emit a `DeprecationWarning`.

---

## Documentation

| Guide | For |
| :--- | :--- |
| [Beginner's guide](docs/BEGINNER_GUIDE.md) | First run, step by step, and how to read the outputs |
| [Data](docs/DATA.md) | Folder layout, filename conventions, calibration |
| [Validation](docs/VALIDATION.md) | What is verified, and how |
| [Visualization guide](docs/VISUALIZATION_GUIDE.md) | What each figure shows |
| [Reproducibility & troubleshooting](docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md) | Provenance files, common errors |
| [Maintainers](docs/AGENTS_AND_MAINTAINERS.md) | Architecture and data contracts |
| [Results delta](docs/findings/RESULTS_DELTA.md) | How the real-data results changed in 0.2 |

## Citation & licence

If you use this software, please cite it via [CITATION.cff](CITATION.cff). Released under the [MIT License](LICENSE).
