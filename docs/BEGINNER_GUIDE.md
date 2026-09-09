# Beginner's Guide: Spheroid & Immunofluorescence Image Analysis Pipelines

Welcome to the **Material Discovery Image Analysis Workspace**! This guide is written from the ground up for students, wet-lab biologists, and researchers who are new to computational image processing or Python CLI tools. No prior software engineering expertise is assumed.

---

## 1. What Are These Pipelines For?

This repository contains two specialized, production-grade image analysis pipelines designed to study cancer spheroids (SKOV3 ovarian cancer cell aggregates) cultured in engineered hydrogels:

```
                                    ┌─────────────────────────────────────────────────────────────┐
                                    │               RAW EXPERIMENTAL MICROSCOPY                   │
                                    └──────────────────────────────┬──────────────────────────────┘
                                                                   │
                                  ┌────────────────────────────────┴────────────────────────────────┐
                                  ▼                                                                 ▼
                    ┌───────────────────────────┐                                     ┌───────────────────────────┐
                    │    Brightfield Spheroids  │                                     │ Immunofluorescence (IF)   │
                    │   Day 0 vs Day 7 Growth   │                                     │  Nuclear Density & Actin  │
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

1. **Brightfield Spheroid Volume Pipeline (`spheroid_pipeline_v2`)**:
   - Analyzes brightfield transmission images of multi-spheroid cultures taken at **Day 0 ($t_0$)** and **Day 7 ($t_7$)** in the same multi-well hydrogel plates.
   - Measures each individual spheroid's area, diameter, circularity, spherical volume ($V_{\text{sphere}} = \frac{\pi}{6}d^3$), and ellipsoidal volume.
   - Tracks spheroids across time using Hungarian mutual nearest-centroid matching.
   - Determines whether hydrogel formulations promote growth (biomass accumulation, spheroid enlargement, colony proliferation) or inhibit growth.

2. **Immunofluorescence (IF) Nuclei Density Pipeline (`spheroid_if_sweep`)**:
   - Analyzes high-resolution fluorescence confocal/epifluorescence microscopy of fixed spheroids stained with:
     - **DAPI / Hoechst (`ch00`)**: Stains cell nuclei (blue/UV).
     - **Phalloidin (`ch02`)**: Stains filamentous F-actin (cytoskeleton / spheroid boundaries).
   - Segments thousands of individual cell nuclei in dense 3D clumps using deep learning (Cellpose-SAM).
   - Quantifies volumetric nuclear packing density ($\text{nuclei/mm}^3$) and nuclear morphology across hydrogel materials.

---

## 2. Directory Layout: What Goes Where?

### Workspace Structure
```
Material discovery/
├── Experimental data /                      # RAW MICROSCOPY DATA (Read-Only!)
│   └── Chuling cells/
│       ├── Spheroid Day0 260805/            # Day 0 Brightfield TIFF images
│       ├── Spheroid D7 260812/              # Day 7 Brightfield TIFF images
│       └── spheroids staining:IF /          # IF Staining Microscopy
│           └── nuclei density analysis/     # Paired _ch00 (nuclei) and _ch02 (actin) TIFFs
│
├── configs/                                 # CONFIGURATION HUB (YAML files)
│   ├── spheroid_brightfield.yaml            # Master config for brightfield spheroid volume pipeline
│   ├── sweep.yaml                           # Master config for IF hyperparameter sweep
│   ├── prepost_ablation.yaml                # 2x2 Factorial ablation config for IF
│   └── robustness_check.yaml                # Robustness regression config for IF
│
├── models/                                  # PRETRAINED NEURAL NETWORK WEIGHTS
│   └── cpsam_v2                             # Local Cellpose-SAM weights (1.2 GB, avoids redownloading)
│
├── output/                                  # BRIGHTFIELD PIPELINE RESULTS (Auto-generated)
│   ├── manifest.csv                         # Full audit of all input images, hashes, optical scales
│   ├── run_metadata.json                    # Exact execution environment, git commit, packages, hardware
│   ├── objects.csv                          # Every segmented spheroid, morphology, and QC flag
│   ├── pairs.csv                            # Matched Day 0 -> Day 7 spheroid pairs and fold changes
│   ├── well_growth_comparison.csv           # Replicate well-level pooled biomass and proliferation
│   ├── condition_summary.csv                # Population-level statistics, bootstrap CIs, permutation p-values
│   ├── report.md / report.html              # Executive summary and interactive web dashboard
│   ├── overlays/                            # Color-coded supervision PNGs (Green=PASS, Orange=REVIEW, Red=FAIL)
│   ├── contact_sheets/                      # Condition montage grids
│   └── figures/                             # Publication figures (PNG + vector PDF)
│
├── runs/                                    # IF PIPELINE RESULTS (Auto-generated per run)
│   ├── 104_PP11_full/                       # Winning production IF run
│   │   ├── run_metadata.json                # Environment provenance
│   │   ├── manifest.csv                     # Channel pairing manifest
│   │   ├── features/                        # objects.csv and per_image.csv
│   │   ├── qc/                              # qc_flags.csv and outlier detection
│   │   ├── masks/                           # 16-bit label TIFFs and overlay PNGs
│   │   └── figures/                         # Biological comparison figures across hydrogels
│   └── comparison_prepost/                  # Cross-run ablation and robustness audit reports
│
├── docs/                                    # IN-DEPTH DOCUMENTATION GUIDES
│   ├── BEGINNER_GUIDE.md                    # This file!
│   ├── AGENTS_AND_MAINTAINERS.md            # Architectural specs for developers and AI agents
│   ├── REPRODUCIBILITY_AND_TROUBLESHOOTING.md # Error codes, warnings, and preflight auditing
│   └── VISUALIZATION_GUIDE.md               # Scientific interpretation of all publication figures
│
├── run_pipeline.py                          # Canonical CLI entry point for Brightfield Pipeline
├── pyproject.toml                           # Package metadata and console scripts
└── requirements.txt                         # Pipeline dependencies
```

> [!NOTE]
> **Model Weights Availability (`models/cpsam_v2`):**
> `models/cpsam_v2` contains the local pretrained weights for Cellpose-SAM (Cellpose 4.x vision transformer, ~1.2 GB). If these weights are absent (e.g. on a fresh clone where `models/` is gitignored), both pipelines automatically download the official weights from HuggingFace / MouseLand upon first invocation.

---

## 3. Image File Naming Conventions

### Brightfield Spheroids
Raw images in `Spheroid Day0 260805/` and `Spheroid D7 260812/` must follow this standardized naming format:
$$\text{Day}\langle 0|7\rangle\quad\langle\text{Condition}\rangle\quad\text{gel}\langle\text{WellNumber}\rangle\_\langle\text{FOV:04d}\rangle\_\text{TRANS.tif}$$

**Examples:**
- `Day0 Mat gel1_0001_TRANS.tif`: Day 0, Matrigel control, Well replicate 1, Field of view 0001, Transmission brightfield.
- `Day7 Mat gel1_0001_TRANS.tif`: The exact matching Day 7 image for the above FOV.
- `Day0 S34D30 gel5_0003_TRANS.tif`: Day 0, S34D30 hydrogel, Well replicate 5, Field of view 0003.

### Immunofluorescence (IF) Channels
Raw images in `nuclei density analysis/` must come in pairs sharing the same base name with channel suffixes:
- Nuclear stain (DAPI / Hoechst): `..._ch00.tif`
- Actin stain (Phalloidin): `..._ch02.tif`

**Example:**
- `SKOV3 Spheroid D7 Mor_S34D30-Gel1-1_ch00.tif` (Nuclei)
- `SKOV3 Spheroid D7 Mor_S34D30-Gel1-1_ch02.tif` (Actin)

---

## 4. Setup & Installation (One-Time)

### Option A: Using the Pre-Configured Virtual Environment
If you are working directly in this workspace, the environment is already created:
```bash
# 1. Navigate to the project root
cd matrix-mat

# 2. Activate the pre-configured Python virtual environment
source .venv/bin/activate

# 3. Verify Python and GPU acceleration
python -c "import torch, cellpose, cv2; print('PyTorch:', torch.__version__, '| Cellpose:', cellpose.__version__, '| MPS Available:', torch.backends.mps.is_available(), '| CUDA Available:', torch.cuda.is_available())"
```

### Option B: Installing from Scratch via `pip` (New Machine or Fresh Setup)
All dependencies are completely specified in [`requirements.txt`](../requirements.txt). No special build tools are required:
```bash
# 1. Create a clean Python virtual environment (Python 3.10 - 3.13)
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate       # On macOS/Linux
# or: .venv\Scripts\activate    # On Windows

# 3. Upgrade pip and install all required packages in one command:
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> **Is Homebrew (`brew`) required?**
> **No.** All Python libraries (NumPy, PyTorch, OpenCV, Cellpose, Scikit-Image, Matplotlib, etc.) are installed directly from standard pre-built wheels via `pip`.
> You only need `brew` if your Mac does not have Python 3 installed at all, in which case you can run `brew install python@3.11`.

> [!TIP]
> **Hardware Acceleration:**
> - **Apple Silicon Macs (M1/M2/M3/M4)**: PyTorch automatically leverages Apple's **MPS (Metal Performance Shaders)** GPU accelerator.
> - **Linux / Windows PCs with NVIDIA**: PyTorch automatically leverages **CUDA**.
> - **CPU Fallback**: If no GPU is present, both pipelines fall back cleanly to CPU with multi-threading without any configuration changes.

---

## 5. Step-by-Step Tutorial: Running Brightfield Analysis

### Step 1: Preflight Sanity Check
Before doing any heavy computing, always run a preflight audit. This checks that your folders exist, images are readable, optical calibration is intact, and control groups are present:
```bash
python run_pipeline.py --preflight-only
```
*Expected Output:*
```
======================================================================
         SPHEROID PIPELINE v2 — PREFLIGHT INTEGRITY AUDIT
======================================================================
Status: PASSED
Total t0 Images Found: 136
Total t7 Images Found: 149
Matched Pair FOVs:     120
Singleton FOVs (t0):   16
Singleton FOVs (t7):   29
Conditions Detected:   Mat, S34D30, S40D30, S43D20, S46D10, S50
Control Condition:     'Mat' (Found: True)
Optical Scale Source:  TIFF Tag 37510 (EVOS Calibration Metadata)
Hardware Accelerator:  Apple Silicon MPS
======================================================================
```

### Step 2: Synthetic Smoke Test (Verification)
Run the synthetic ground-truth test. This generates 3 mathematical synthetic images (known sphere diameters and a non-spheroid bright blob) and asserts volume recovery error $<10\%$:
```bash
python run_pipeline.py --smoke-test
```
*Takes ~2 seconds and exits 0 if all algorithms are functioning correctly.*

### Step 3: Rapid Subsample Run (`--sample 12`)
To test parameters on a small, stratified sample of real experimental images (2 FOVs per condition):
```bash
python run_pipeline.py --sample 12 --output-dir output_sample12
```
This will:
- Process 12 images in ~15 seconds.
- Run automated sanity gates (checks that spheroids are found and no giant artifacts cover $>25\%$ of the image).
- Produce visual overlays in `output_sample12/overlays/`. You can open these PNGs to inspect contours!

### Step 4: Full Dataset Batch Run
When you are satisfied, process the entire dataset of 285 raw brightfield images:
```bash
python run_pipeline.py --config configs/spheroid_brightfield.yaml --output-dir output
```
This runs the full multi-tier pipeline:
1. Segments all Day 0 and Day 7 images (caching 16-bit masks in `masks/`).
2. Applies multi-focal z-plane deduplication (flags duplicate spheroids captured in slightly offset focal planes).
3. Executes Hungarian bipartite centroid matching across $t_0 \to t_7$.
4. Computes 2,000 bootstrap iterations for 95% CIs and 10,000 permutation test iterations against Mat control.
5. Generates all 6 presentation dashboard figures, overlays, and `report.html`.

### Step 5: Regenerate Presentation Plots on Demand
If you only want to re-render the presentation figures without re-segmenting images:
```bash
python run_pipeline.py --plots --output-dir output
```
Figures 1 through 6 will appear in `output/figures/` in both 300 DPI PNG and publication-ready vector PDF formats!

---

## 6. Step-by-Step Tutorial: Running Immunofluorescence (IF) Analysis

### Step 1: Run the Production IF Stack (`PP11_full`)
To run the winning production pipeline (Rolling-ball background subtraction + dynamic range normalization + Cellpose-SAM + seeded watershed clump splitting + physical size gating + border clearance):
```bash
python -m spheroid_if_sweep sweep --config configs/robustness_check.yaml
```

### Step 2: Generate Publication Biological Figures
To generate the three cross-hydrogel biological comparison figures from a completed run:
```bash
python -m spheroid_if_sweep compare-biological --run-dir runs/104_PP11_full
```
*Generated in `runs/104_PP11_full/figures/`:*
1. `fig_if_nuclei_density_by_material.png` / `.pdf`: Boxplot & jittered strip plot of nuclei density ($\text{nuclei/mm}^3$) across all hydrogels with Mann-Whitney U test vs Mat control.
2. `fig_if_clean_vs_all_density.png` / `.pdf`: All Cohort vs Clean Fields comparison demonstrating biological stability under QC filtering.
3. `fig_if_nuclear_volume_distributions.png` / `.pdf`: Boxplot comparing nuclear volume distributions ($\mu\text{m}^3$) across formulations.

### Step 3: Compare Pre/Post Processing Ablations
To compare how background subtraction and clump splitting improved segmentation across all 27 fields:
```bash
python compare_prepost.py --runs runs/ --biological
```
This generates `runs/comparison_prepost/effect_summary.md`, side-by-side disagreement panels, and biological figure comparisons.

---

## 7. Configuration Parameters Explained Line-by-Line

### Brightfield Config: `configs/spheroid_brightfield.yaml`

| Parameter | Type | Default | Physical Meaning & Biological Guidance |
| :--- | :--- | :--- | :--- |
| `t0_dir` | path | `"..."` | Directory containing Day 0 brightfield TIFFs. |
| `t7_dir` | path | `"..."` | Directory containing Day 7 brightfield TIFFs. |
| `default_pixel_size_um` | float | `1.518817` | Optical calibration ($\mu\mathrm{m/px}$) for EVOS 4x. Used if TIFF Tag 37510 is absent. |
| `random_seed` | int | `42` | Controls random sampling for bootstrap CIs and permutation tests for 100% reproducibility. |
| `preprocess.flat_field_sigma`| float | `0.0` | Gaussian sigma for background division. Set to `0.0` (disabled) to avoid flattening spheroid core contrast. |
| `preprocess.percentile_low` | float | `1.0` | Lower intensity percentile clipping bound (suppresses dead pixels). |
| `preprocess.percentile_high`| float | `99.0`| Upper intensity percentile clipping bound (anchors bright gel background). |
| `segmentation.backend` | str | `"cpsam"` | Deep learning instance segmentation backend: `'cpsam'`, `'cyto3'`, or `'classical'`. |
| `qc.min_d_um` | float | `40.0` | Rejects objects with equivalent diameter $< 40\,\mu\mathrm{m}$ (camera dust, debris). |
| `qc.max_d_um` | float | `1500.0` | Rejects objects $> 1500\,\mu\mathrm{m}$ (optical artifacts, meniscus edges). |
| `qc.border_margin_px` | int | `15` | Rejects spheroids touching or within 15 px of image boundaries (clipped volumes). |
| `qc.ring_contrast_factor` | float | `0.90` | Requires spheroid interior to be at least 10% darker than surrounding background ring. |
| `qc.max_single_mask_area_fraction` | float | `0.25` | Rejects any mask covering $> 25\%$ of the image area (prevents giant hallucinated circles). |
| `qc.min_circularity` | float | `0.65` | Spheroids with circularity $< 0.65$ are marked `REVIEW` (irregular borders / touching pairs). |
| `stats.control_condition` | str | `"Mat"` | Reference baseline condition for 2-sample permutation tests. |
| `stats.min_denominator_d0_um`| float | `60.0` | Pairs with $t_0$ diameter $< 60\,\mu\mathrm{m}$ are excluded from fold-change stats ($V_7/V_0$). |
| `stats.bootstrap_iterations` | int | `2000` | Number of bootstrap resamples used to compute 95% Confidence Intervals on median fold change. |
| `stats.permutation_iterations`| int| `10000`| Number of Monte Carlo permutations for hypothesis testing against control. |

---

## 8. Interpreting Your Results

### What do the QC Flags mean?
- **`PASS` (Green)**: High-confidence, isolated spheroid that meets all physical size, border margin, contrast, and circularity criteria. Included in all primary statistics.
- **`REVIEW` (Orange)**: Biologically authentic spheroid with non-spherical morphology (e.g. circularity $< 0.65$ or fold change outside $[0.1, 20]$). Never silently dropped; preserved for sensitivity auditing.
- **`FAIL` (Red)**: Rejected candidate instance (e.g. touches image border, undersized speck $< 40\,\mu\mathrm{m}$, bright gel bubble with no dark contrast, or multi-focal plane duplicate). Logged with exact reason in `objects.csv`.

### What does `well_growth_comparison.csv` tell you?
- `fc_pooled_vol_density`: Fold change of total spheroid biomass in that well ($\Sigma V_7 / \Sigma V_0$). This reflects **overall growth**.
- `fc_mean_volume`: Fold change of average spheroid size. Reflects **hypertrophy** (individual spheroids swelling/enlarging).
- `fc_count_density`: Fold change in number of colonies per field. Reflects **hyperplasia** (colony proliferation vs survival).

---

## 9. Dataset Scaling: Handling Different Amounts of Pictures & Pairs

A common question from researchers is: **"Does this pipeline work if I have a different number of pictures, missing pairs, or a brand-new experimental plate?"**

**Yes.** Both pipelines were engineered from first principles to be dynamically scale-invariant:

### 1. Dynamic Discovery (No Hardcoded Image Lists)
The pipelines never rely on static lists of filenames. They dynamically inspect target directories using filesystem globbing (`*.tif`, `*.tiff`). Whether your directory contains **1 pair**, **12 pairs**, **120 pairs**, or **10,000 images**, every file is automatically inventoried, verified for file integrity, and registered in `manifest.csv`.

### 2. Graceful Handling of Missing Pairs & Singletons
In real biological experiments, plates are sometimes dropped, wells dry out, or focal drift makes a timepoint uncapturable:
- **Brightfield Spheroid Volume**:
  - If a well only has a Day 0 image (and no matching Day 7 image), or vice-versa, the pipeline **does not crash**.
  - Singletons are flagged with `is_matched_pair = False` in `manifest.csv`.
  - Individual spheroids from singleton images are still segmented, measured, and stored in `objects.csv` (so you still have Day 0 baseline distributions and Day 7 endpoint measurements).
  - Only genuine matched pairs with mutual nearest-centroid assignment proceed to fold-change tracking in `pairs.csv`.
  - If a well has 3 spheroids on Day 0 and 4 spheroids on Day 7, the Hungarian algorithm optimally pairs the 3 closest spheroids and marks the 4th as an unmatched single object without error.
- **Immunofluorescence (IF)**:
  - If an actin channel (`_ch02.tif`) or nuclear channel (`_ch00.tif`) is missing for a field, the preflight auditor warns you immediately.
  - In standard runs, you can pass `--allow-unpaired` to process all complete pairs while cleanly skipping incomplete single-channel files.

### 3. Arbitrary Batch Sizing & Subsampling
You can run subsets of your data at any time:
```bash
# Run a quick 12-image subset to verify parameters
python run_pipeline.py --sample 12

# Run only a specific condition (e.g. S34D30)
python run_pipeline.py --condition S34D30

# Run on an entirely new experimental folder
python run_pipeline.py --t0-dir /path/to/my_day0 --t7-dir /path/to/my_day7 --out /path/to/results
```

### 4. Statistical Resilience with Small or Unequal Sample Sizes
- **Bootstrap Confidence Intervals**: If a condition has only 1 pass pair ($N=1$), the bootstrap CI safely outputs `[val, val]` rather than failing on zero-variance. If $N=0$, it outputs `[NaN, NaN]`.
- **Permutation Tests**: Exact Monte Carlo permutation testing checks for group sizes $N \ge 2$. If a condition has $<2$ pairs, the test gracefully returns `n/a` (not applicable) rather than crashing.
- **Figure Plotting**: All 6 presentation figures and 3 IF biological plots dynamically determine condition categories from the data. If you have 2 conditions, 6 conditions, or 20 conditions with custom names, the figures adjust their axes and legends automatically.

