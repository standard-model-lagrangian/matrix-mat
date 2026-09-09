# Original User Request

## Initial Request — 2026-07-02T11:50:55Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Analyze the machine learning pipeline and Bayesian Optimization methodology from https://github.com/YahwahNissi/Peptide_Materials_ML_Codes/. Set up the environment, perform a detailed analysis of the code's flow, and provide proposals for different levels of reproduction runs.

Working directory: /Users/frederick/Visual Studio /Material discovery/peptide_materials_ml
Integrity mode: benchmark

## Requirements

### R1. Environment Setup
Clone the repository `https://github.com/YahwahNissi/Peptide_Materials_ML_Codes/` into the working directory. Resolve dependencies and set up a functional Python virtual environment.

### R2. Code Analysis & ML Flow Breakdown
Perform a detailed review of the codebase to understand the machine learning algorithms and the Bayesian Optimization methodology implemented. Create a comprehensive markdown report breaking down the end-to-end execution flow.

### R3. Reproduction Proposals
Based on the code analysis, propose different levels of trial runs or reproduction strategies (e.g., quick smoke tests vs. full reproduction). For each proposal, describe the expected runtime, hardware requirements, and what it specifically validates about the Bayesian Optimization methodology.

## Acceptance Criteria

### Execution Verification
- [ ] The repository is cloned successfully into the working directory.
- [ ] The virtual environment is created and all listed dependencies install without conflict.
- [ ] A simple dry-run or import test passes successfully, proving the environment is correct.

### Evaluation Criteria
- [ ] A detailed markdown report (`ml_flow_breakdown.md`) is produced explaining the ML flow and Bayesian optimization methodology.
- [ ] The report includes 2-3 clear, actionable proposals for reproducing the results at varying levels of intensity.

## Follow-up — 2026-08-31T21:26:47Z

# Teamwork Project Prompt — Rebuild Spheroid Volume Pipeline

Build an end-to-end Python image processing pipeline that quantifies per-spheroid volume changes between brightfield imaging timepoints (day 0 and day 7) across hydrogel formulations and replicates with multi-spheroid instance segmentation, robust background contrast gating, denominator gating, and self-verification loops.

Working directory: /Users/frederick/Visual Studio /Material discovery/spheroid_pipeline_v2
Integrity mode: development

## Ground Truth & Context
- Brightfield images of cancer spheroids embedded in hydrogel, two timepoints (day 0 and day 7), same wells imaged twice.
- CRITICAL: Each image contains MULTIPLE spheroids (typically 3–15), not one. They appear as dark, roughly circular to irregular blobs on a bright, unevenly lit gel background, with halos, specks/debris, and occasional touching pairs. Spheroids are always DARKER than the local background.
- Any result where a single mask covers >25% of the image or encloses the entire field is a bug and must be rejected.
- Optical calibration: Automatically extract from TIFF metadata (e.g. 1.518817 um/px on EVOS 4x) or accept config/CLI override.

## Requirements

### R1. Segmentation Hierarchy & Disk Caching
- Segmentation backend hierarchy:
  1. Primary: Cellpose-SAM (`pip install cellpose`, `model_type='cpsam'`)
  2. Fallback: Cellpose (`cyto3`, diameter sweep)
  3. Last resort: Classical multi-window adaptive thresholding + watershed
- Use MPS (Apple Silicon M4) / CUDA if available, with CPU fallback (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Cache every label mask to disk (`masks/<image_id>.png`) so re-runs never repeat inference.
- Downsample images before inference so expected spheroid diameter is 150-250 px, storing scale factor to convert measured pixels back to um.
- Record any fallback decisions in `DECISIONS.md`.

### R2. Spheroid Filtering & QC Gating
Apply these post-filters to every mask in order:
1. Resolve overlapping instances (keep larger area or higher score).
2. Size gate in um: reject objects outside `[min_d_um, max_d_um]` (default 40 to 1500 um).
3. Reject border-touching objects (configurable margin).
4. Interior-intensity check: reject any mask whose mean interior intensity is not at least N% darker than its surrounding background ring (configurable N).
5. Per-image plausibility: if 0 objects pass, or any single object covers >25% of the image area, mark image `SEG_FAIL` and route to fallback backend.
6. QC metrics per object: area, perimeter, circularity ($4\pi A / P^2$), solidity, eccentricity, equivalent diameter $d = 2\sqrt{A/\pi}$, volume_sphere = $(\pi/6) d^3$, volume_ellipsoid = $(\pi/6) \text{major} \times \text{minor}^2$.
7. Flag (never silently drop): `REVIEW` if circularity < threshold.

### R3. Pairing, Denominator Gating & Statistical Hygiene
- Mutual nearest centroid pairing within each well across $t_0 \to t_7$ subject to max displacement tolerance.
- Denominator gate: exclude pairs whose $t_0$ diameter $< 60\,\mu\mathrm{m}$ from fold-change statistics (to prevent near-zero $V_0$ debris blowing up fold changes).
- Flag for `REVIEW` any pair with fold change outside $[0.1, 20]$.
- Per-condition summary: $n$ images, $n$ objects, $n$ matched pairs, median/IQR of $V_0, V_7$, and fold change; bootstrap 95% CI on median fold change; permutation test between conditions. Report exclusion counts at every step.

### R4. Self-Verification & Supervision Artifacts
1. Synthetic smoke test: 3 synthetic images (dark blurred disks of known radius on uneven bright background, touching pair, bright gel blob that must NOT be segmented). Assert recovered diameters within 10% and 0 masks for bright blob.
2. Stratified `--sample 12` run with programmatic sanity gates:
   - Objects per image in $[1, 25]$
   - No mask >25% of image
   - $\ge 60\%$ of images yield $\ge 1$ PASS object
   - Diameters in $[50, 1500]\,\mu\mathrm{m}$
   - If any gate fails, inspect overlays, adjust parameters, and iterate (up to 8 cycles).
3. Full dataset run over all images.
4. Outputs: `objects.csv`, `pairs.csv`, `condition_summary.csv`, overlay PNGs for every image (with $d$ in um, ID, QC flag), contact sheets, publication figures (log-scale violin, scatter with identity line), executive `report.md`, `report.html`, and `DECISIONS.md`.
5. `review/` folder convention for human manual mask validation with Dice/IoU in `validation.csv`.

## Acceptance Criteria

### Execution & Verification
- [ ] Synthetic smoke test passes with recovered diameters within 10% of truth and bright gel blobs rejected.
- [ ] Sample 12 test passes all programmatic sanity gates.
- [ ] Full dataset execution processes all image pairs without giant single-circle artifacts.
- [ ] All required CSV tables, publication figures (PNG+PDF), overlays, contact sheets, and `report.md` are generated in output directory.
- [ ] `DECISIONS.md` records all architectural, backend, and fallback decisions.

