# Spheroid & Immunofluorescence Image Analysis Pipelines

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch MPS/CUDA](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org/)
[![Cellpose-SAM](https://img.shields.io/badge/Cellpose--SAM-cpsam__v2-green.svg)](https://www.cellpose.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Strict%20Provenance-brightgreen.svg)](docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md)

An end-to-end, deterministic, and fully auditable computer vision suite designed to quantify cancer spheroid morphology, volumetric growth kinetics, and 3D nuclear packing density across engineered biomaterial hydrogels.

---

## Architecture Overview

The repository hosts two decoupled, publication-grade image analysis pipelines:

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
                    • Strict QC gating (Zero drops)                                     • Clump splitting & size gate
                    • Hungarian centroid pairing                                        • Border clearance exclusion
                    • Biomass & hypertrophy metrics                                     • Nuclei density per mm³
                    • Presentation Figures 1–6                                          • Cross-material comparisons
```

---

## Documentation Roadmap

For in-depth explanations tailored to specific roles, consult our documentation hub in `docs/`:

| Guide | Description | Target Audience |
| :--- | :--- | :--- |
| **[Beginner's Guide](docs/BEGINNER_GUIDE.md)** | Zero-to-hero manual: what folders stuff goes into, line-by-line configuration parameter breakdown, running instructions, and output interpretation. | Students, wet-lab biologists, new users |
| **[Agents & Maintainers](docs/AGENTS_AND_MAINTAINERS.md)** | Architecture, execution graphs, module data contracts, caching mechanisms, deduplication algorithms, and extension guidelines. | AI Agents, software engineers, bioinformaticians |
| **[Reproducibility & Troubleshooting](docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md)** | Preflight auditing, `run_metadata.json`, SHA-256 manifests, warnings dictionary, and bad scenario remediation recipes. | Quality auditors, computational biologists |
| **[Scientific Visualization Guide](docs/VISUALIZATION_GUIDE.md)** | Gallery and scientific interpretation of Presentation Figures 1–6, IF biological plots, and statistical testing methodologies. | PIs, manuscript authors, presenters |

---

## Quickstart Guide

### 1. Environment Activation
```bash
# Clone or navigate to the repository
cd "/Users/frederick/Visual Studio /Material discovery"

# Activate the pre-configured virtual environment
source .venv/bin/activate
```

### 2. Brightfield Spheroid Volume Analysis Pipeline

```bash
# Step 1: Preflight Data & Optical Scale Audit (Instant sanity check)
python run_pipeline.py --preflight-only

# Step 2: Synthetic Ground-Truth Smoke Test (Asserts <10% recovery error)
python run_pipeline.py --smoke-test

# Step 3: Fast 12-Sample Stratified Test with Automated Sanity Gates
python run_pipeline.py --sample 12 --output-dir output_sample12

# Step 4: Full Dataset Batch Execution (All 285 raw brightfield images)
python run_pipeline.py --config configs/spheroid_brightfield.yaml --output-dir output

# Step 5: Regenerate Presentation Figures (Figs 1-6) On Demand
python run_pipeline.py --plots --output-dir output
```

### 3. Immunofluorescence (IF) Nuclei Density Pipeline

```bash
# Step 1: Run Production IF Segmentation Stack (Winning PP11 Stack)
python -m spheroid_if_sweep sweep --config configs/robustness_check.yaml

# Step 2: Generate Cross-Material Biological Comparison Figures
python -m spheroid_if_sweep compare-biological --run-dir runs/104_PP11_full

# Step 3: Run Pre/Post Processing Ablation Comparison & Verdict
python compare_prepost.py --runs runs/ --biological
```

---

## CLI Command Quick Reference

### Brightfield Spheroid CLI (`run_pipeline.py`)

| Flag / Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `--config` | `<path>` | Path to YAML configuration file | `config.yaml` / `configs/spheroid_brightfield.yaml` |
| `--preflight-only`| *None* | Run proactive input and optical scale audit and exit | `False` |
| `--smoke-test` | *None* | Run synthetic ground-truth mathematical smoke test | `False` |
| `--sample` | `<N>` | Run stratified $N$-sample test with programmatic sanity gates | `None` (Full) |
| `--full-dataset` | *None* | Process all images in Day 0 and Day 7 directories | `False` |
| `--plots` | *None* | Re-generate all 6 presentation figures from existing CSVs | `False` |
| `--output-dir` | `<path>` | Directory to write CSVs, figures, overlays, and reports | `output` |
| `--pixel-size` | `<val>` | Override optical scale in $\mu\mathrm{m/pixel}$ | Auto (EVOS Tag 37510) |
| `--no-cache` | *None* | Force recomputation without loading cached masks | `False` |

### Immunofluorescence CLI (`python -m spheroid_if_sweep`)

| Subcommand | Flags | Description |
| :--- | :--- | :--- |
| `sweep` | `--config <path> [--scope subset\|full] [--allow-unpaired]` | Executes multi-run parameter sweep or pre/post ablation. |
| `compare-biological` | `--run-dir <path> [--output <path>]` | Generates cross-material nuclei density, clean-vs-all, and volume plots. |
| `compare-prepost` | `--runs <dir> [--output <dir>]` | Evaluates 2×2 factorial ablation effects and robustness spread reduction. |
| `compare` | `--runs <dir> [--output <dir>]` | Cross-run divergence audit across original hyperparameter sweeps. |

---

## Directory & Configuration Structure

```
Material discovery/
├── configs/
│   ├── spheroid_brightfield.yaml     # Master Brightfield Config (exhaustively commented)
│   ├── sweep.yaml                    # Master IF Hyperparameter Sweep Config
│   ├── prepost_ablation.yaml         # 2x2 Factorial Pre/Post Ablation Config
│   └── robustness_check.yaml         # Robustness Regression Config
├── config.yaml                       # Root alias to configs/spheroid_brightfield.yaml
│
├── Experimental data /Chuling cells/ # Raw experimental images (Read-Only)
│   ├── Spheroid Day0 260805/         # Day 0 Brightfield TIFFs
│   ├── Spheroid D7 260812/           # Day 7 Brightfield TIFFs
│   └── spheroids staining:IF /       # IF confocal channels (_ch00, _ch02)
│
├── output/                           # Primary Brightfield Outputs
│   ├── manifest.csv                  # Complete input audit with SHA-256 hashes
│   ├── run_metadata.json             # Execution environment provenance
│   ├── objects.csv                   # Per-spheroid morphology and QC flags
│   ├── pairs.csv                     # Matched spheroid pairs and volume fold changes
│   ├── well_growth_comparison.csv    # Well-replicate pooled biomass and proliferation
│   ├── condition_summary.csv         # Population stats, bootstrap CIs, permutation p-values
│   ├── report.md / report.html       # Executive report and interactive dashboard
│   ├── overlays/                     # Color-coded supervision PNGs
│   ├── contact_sheets/               # Condition montage grids
│   └── figures/                      # Publication Figures 1–6 (PNG + vector PDF)
│
└── runs/                             # IF Sweep & Ablation Outputs
    ├── 104_PP11_full/                # Production winning IF run
    └── comparison_prepost/           # Ablation audit and robustness reports
```

---

## Scientific Rigor & Provenance

- **Quality Control**: Every object is explicitly classified as `PASS`, `REVIEW`, or `FAIL`. No data is ever silently dropped.
- **Statistical Testing**:
  - Non-parametric **2,000-iteration bootstrap** 95% Confidence Intervals on median fold change.
  - Non-parametric **10,000-iteration permutation tests** comparing hydrogels against control (`Mat`).
  - One-tailed **Wilcoxon signed-rank tests** evaluating well-replicate growth against $H_0: \text{FC}=1.0$.
  - Two-sided **Mann-Whitney U tests** for cross-material IF nuclear density differences.
- **Hardware Acceleration**: Automatically detects and leverages Apple Silicon MPS (Metal) on Mac M-series or CUDA on Linux/Windows, with seamless CPU fallback.
- **Audit Logging**: Every model fallback, plausibility rejection, and parameter tuning event is appended chronologically to `DECISIONS.md`.

For troubleshooting or detailed parameter specifications, please refer to the **[Reproducibility & Troubleshooting Guide](docs/REPRODUCIBILITY_AND_TROUBLESHOOTING.md)**.
