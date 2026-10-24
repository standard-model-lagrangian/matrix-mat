# matrix-mat: Spheroid & Immunofluorescence Image Analysis Pipelines

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch MPS/CUDA](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org/)
[![Cellpose-SAM](https://img.shields.io/badge/Cellpose--SAM-cpsam__v2-green.svg)](https://www.cellpose.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, seeded and provenance-tracked image analysis suite to quantify cancer spheroid morphology, volume change, and planar nuclear packing density across engineered biomaterial hydrogels. CPU runs are bitwise reproducible; GPU/MPS inference is reproducible within a small tolerance (see [Validation](docs/VALIDATION.md)).

---

## Architecture Overview

The repository hosts two decoupled image analysis pipelines:

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │               RAW EXPERIMENTAL MICROSCOPY                   │
                                    └──────────────────────────────┬──────────────────────────────┘
                                                                   │
                                  ┌────────────────────────────────┴────────────────────────────────┐
                                  ▼                                                                 ▼
                    ┌───────────────────────────┐                                     ┌───────────────────────────┐
                    │  Brightfield Spheroids    │                                     │  Immunofluorescence (IF)  │
                    │   Day 0 vs Day 7 Growth   │                                     │   Nuclear Packing Density │
                    └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                                  │                                                                 │
                                  ▼                                                                 ▼
                    • Multi-spheroid segmentation                                       • Cellpose-SAM nuclear seg
                    • Optical scale extraction (um)                                     • Rolling-ball background
                    • Multi-criteria QC gating                                          • Clump splitting & size gate
                    • Hungarian centroid pairing                                        • Border clearance exclusion
                    • Biomass & volume change kinetics                                  • Planar areal density (nuclei/mm²)
                    • Presentation Figures 1–6                                          • Cross-material comparisons
```

---

## Documentation Roadmap

For in-depth specifications, consult our documentation:

| Guide | Description |
| :--- | :--- |
| **[Dataset & Data Structure](docs/DATA.md)** | Directory layouts, filename conventions, optical calibration, and data acquisition. |
| **[Pipeline Validation](docs/VALIDATION.md)** | What has been validated (synthetic ground truth, statistics) and what has not yet (manual counts, cross-backend). |
| **[Beginner's Guide](docs/BEGINNER_GUIDE.md)** | Step-by-step tutorial, configuration breakdown, and result interpretation. |
| **[Agents & Maintainers](docs/AGENTS_AND_MAINTAINERS.md)** | Architecture, data contracts, caching mechanisms, and engineering guidelines. |
| **[Reproducibility & Troubleshooting](docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md)** | Preflight auditing, `run_metadata.json`, SHA-256 manifests, and issue remediation. |
| **[Scientific Visualization Guide](docs/VISUALIZATION_GUIDE.md)** | Presentation Figures 1–6 gallery, IF plots, and statistical annotations. |

---

## Quickstart Guide

For a fresh clone:

```bash
# 1. Create and activate virtual environment
uv venv
source .venv/bin/activate

# 2. Install pinned dependencies, then the package (add [dev] for tests)
uv pip install --require-hashes -r requirements-lock.txt
uv pip install -e ".[dl]"

# 3. Download and verify pinned Cellpose-SAM model weights
python tools/fetch_models.py

# 4. Configure data root path
export MATRIX_MAT_DATA_ROOT=data

# 5. Brightfield preflight audit and run
matrix-mat bf preflight
matrix-mat bf run --config configs/spheroid_brightfield.yaml
matrix-mat bf plots --output-dir output

# 6. Immunofluorescence preflight audit and sweep
matrix-mat if preflight
matrix-mat if sweep --config configs/sweep.yaml
matrix-mat if compare-prepost --runs runs/ --biological
```

---

## Unified CLI Reference (`matrix-mat`)

The unified CLI provides dedicated subcommands for brightfield (`bf`) and immunofluorescence (`if`) analysis:

### Brightfield Commands (`matrix-mat bf`)

```bash
# Preflight data and calibration integrity check
matrix-mat bf preflight --config configs/spheroid_brightfield.yaml

# Run full brightfield pipeline
matrix-mat bf run --config configs/spheroid_brightfield.yaml --output-dir output

# Run on a sample of images for fast verification
matrix-mat bf run --config configs/spheroid_brightfield.yaml --sample 10

# Generate presentation figures from existing CSVs
matrix-mat bf plots --output-dir output
```

### Immunofluorescence Commands (`matrix-mat if`)

```bash
# Preflight channel pairing and hardware acceleration check
matrix-mat if preflight --config configs/sweep.yaml

# Execute parameter sweep across configurations
matrix-mat if sweep --config configs/sweep.yaml --runs-dir runs

# Cross-run pre/post ablation comparison
matrix-mat if compare-prepost --runs runs/ --biological

# Biological figures comparison across materials
matrix-mat if compare-biological --run-dir runs/104_PP11_full
```

---

## Statistical Methods & Rigor

- **Unit of Analysis**: Formal statistical inference is performed at the **biological unit level (gel/well)** rather than the spheroid level to prevent pseudo-replication. Spheroid-level measurements are reported as descriptive population statistics.
- **Hierarchical Resampling**: Confidence intervals for median fold changes are computed using a two-stage hierarchical bootstrap (resampling gels, then spheroids within gels).
- **Cluster Permutation Testing**: Differences between hydrogel conditions and the control condition (`Mat`) are tested using exact cluster label permutations for small sample sizes, with Holm-Bonferroni multiplicity correction.
- **Paired Growth Testing**: Volume fold changes between Day 0 and Day 7 for matched spheroids are evaluated using two-sided Wilcoxon signed-rank tests.
- **Known Limitations**:
  - Spherical volumes are estimated from 2D cross-sectional area: $V = \frac{\pi}{6} d_{\text{eq}}^3$. This assumes spherical geometry.
  - IF nuclear density readouts reflect single-plane confocal optical sections and are reported as planar areal density ($\text{nuclei/mm}^2$ of spheroid cross-section), not true 3D volumetric packing. True volumetric density requires 3D confocal z-stacks.
  - **Statistical power is limited by the number of gels.** With 3–4 gels per condition, the smallest attainable two-sided gel-level permutation p-value is about 0.03–0.17, so after Holm correction across five materials no comparison can reach p < 0.05. Report effect sizes with their confidence intervals; adding gels is the only way to gain power.
  - Bootstrap intervals are computed on the log₂ fold-change scale (`log2fc_ci_95_*`) and back-transformed (`bootstrap_ci_95_*`, fold-change scale). With very few gels the hierarchical bootstrap interval is itself unstable.
- **IF inference**: per-field planar densities are averaged per gel and compared with the control by the same gel-level permutation test with Holm correction (`summary/summary_by_material.csv`: `p_raw`, `p_adj`, `min_achievable_p`). Field-level means/SDs are descriptive only.
- **Provisional thresholds**: the IF nuclear reference volume (841.04 µm³) and blur threshold (Laplacian variance 150) were derived from this dataset and are marked provisional in `configs/*.yaml`.
