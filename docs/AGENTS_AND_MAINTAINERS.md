# Architecture & Maintainer's Specification: Image Analysis Pipelines

This document provides a comprehensive technical reference for software engineers, bioinformaticians, and autonomous AI agents working on, extending, or refactoring the image processing pipelines in this repository.

---

## 1. System Architecture & Design Principles

The workspace implements two decoupled, seeded and provenance-tracked scientific image analysis pipelines:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                ARCHITECTURAL COMPONENT MAP                                  │
├──────────────────────────────────────────────────────────────┬──────────────────────────────┤
│               BRIGHTFIELD SPHEROID PIPELINE                  │          IF PIPELINE         │
│               (src/matrix_mat/brightfield)                   │  (src/matrix_mat/if_density) │
├──────────────────────────────────────────────────────────────┼──────────────────────────────┤
│ 1. Preflight Auditor (preflight.py)                          │ 1. Preflight Auditor         │
│ 2. Optical Scale Extractor (scale_extractor.py)              │ 2. Channel Pairing Engine    │
│ 3. Illumination Preprocessor (preprocess.py)                 │ 3. Preprocessor (RollingBall)│
│ 4. 3-Tier Segmentation Hierarchy (segment.py)                │ 4. CPSAM Inference Engine    │
│ 5. Lossless Mask Cache (mask_cache.py)                       │ 5. Seeded Watershed Splitter │
│ 6. Multi-focal Deduplicator (qc_filter.py)                   │ 6. Border Clearance Gater    │
│ 7. Morphometrics & Volumetrics Engine (measure.py)           │ 7. QC & Artifact Gater       │
│ 8. Hungarian Bipartite Centroid Matcher (pair.py)            │ 8. Morphometrics & Extent    │
│ 9. Hypothesis Testing & Bootstrap CI (stats.py)              │ 9. Provenance Tracker        │
│ 10. Presentation Visualizer (presentation_plots.py)          │ 10. Biological Plotter       │
│ 11. Supervisory Overlays & Contact Sheets (visualize.py)     │ 11. Cross-Run Comparator     │
│ 12. Provenance & Manifest Tracker (provenance.py)            │                              │
│ 13. Executive Report Generator (report.py)                   │                              │
└──────────────────────────────────────────────────────────────┴──────────────────────────────┘
```

### Core Tenets:
1. **Zero Silent Drops**: No candidate instance is ever discarded silently. Every single detected object is assigned a status (`PASS`, `REVIEW`, `FAIL`) and recorded in `objects.csv` alongside its geometric metrics and rejection rationale (`qc_reasons`).
2. **Reproducibility**: With the classical backend on CPU, identical inputs and configuration give bit-for-bit identical outputs. cpsam inference on GPU/MPS is reproducible within a small tolerance (identical counts, area within 0.5%), not bitwise. All stochastic steps (bootstrap, permutation tests) use `config.random_seed`.
3. **Content-addressed cache**: Segmentation masks (and their QC flags) are stored as 16-bit PNGs under `$MATRIX_MAT_CACHE_DIR`, keyed by image bytes + segmentation parameters + model SHA-256 + package version. Failed segmentations are never cached. `--no-cache` forces recomputation.
4. **Hardware Portability**: Full support for Apple Silicon MPS (`torch.backends.mps`), NVIDIA CUDA (`torch.cuda`), and automated CPU fallback without platform-specific code branching.

---

## 2. Brightfield Pipeline (`matrix_mat.brightfield`)

### 2.1 Execution Graph & Data Flow
```
Raw TIFF (1536x2048, EVOS Tag 37510)
  │
  ├──► [PreflightAuditor] ──► PreflightReport (Checks files, scales, control group)
  │
  ├──► [ScaleExtractor] ──► pixel_size_um (Tag 37510 JSON / OME-XML / Fallback)
  │
  ├──► [Preprocessor] ──► Flattened, normalized, scale-factor downsampled float32 image
  │
  ├──► [SegmentationHierarchy]
  │      ├─ One backend per run (fallback_policy: strict): cpsam (default, cpsam_v2) | cpdino | classical
  │      ├─ Empty / implausible / failed image → flagged SEG_FAIL, never re-segmented with another method
  │      └──► Content-addressed 16-bit PNG cache in ${CACHE_ROOT} (image bytes + params + model hash + version)
  │
  ├──► [QC Filter & Measure]
  │      ├─ Resolves overlapping mask instances
  │      ├─ Size gate in um: [40, 1500] um
  │      ├─ Border margin clearance: 15 px
  │      ├─ Annular ring contrast check: I_int <= 0.90 * I_ring
  │      ├─ Plausibility gate: Single mask area < 25% FOV
  │      ├─ Cross-FOV multi-focal z-plane deduplication (dist < 35px)
  │      └─ Computes A, P, circ, sol, ecc, d, V_sphere, V_ellip, discrepancy
  │
  ├──► [Hungarian Pairing]
  │      ├─ Bipartite cost matrix on Euclidean centroid displacement
  │      └─ Maximum displacement tolerance: max(50px, 1.5 * mean_radius)
  │
  ├──► [Stats & Auditing]
  │      ├─ Denominator gate: excludes pairs with d0 < 60 um from FC
  │      ├─ Bootstrap 95% CI on median fold change (B = 2,000)
  │      ├─ Two-sample permutation test vs Mat control (M = 10,000)
  │      └─ 10-stage exclusion auditing table
  │
  └──► [Visual & Artifact Generation]
         ├─ manifests.csv & run_metadata.json (Environment provenance)
         ├─ well_growth_comparison.csv (Replicate well-level pooling)
         ├─ Presentation Figures 1–6 (300 DPI PNG + vector PDF)
         ├─ High-contrast RGB supervision overlays (Green/Orange/Red)
         ├─ Contact sheets grouped by condition & replicate
         └─ report.md & standalone interactive report.html
```

### 2.2 Data Classes & Interface Contracts

#### `SpheroidObjectRecord` (`measure.py`)
Encapsulates a single detected spheroid candidate:
```python
@dataclass
class SpheroidObjectRecord:
    object_id: str                   # Unique: <image_id>_obj<N>
    image_id: str                    # e.g. Day0 Mat gel1_0001_TRANS
    timepoint: str                   # 't0' or 't7'
    condition: str                   # e.g. 'Mat', 'S34D30'
    replicate: str                   # e.g. 'gel1'
    fov: str                         # e.g. '0001'
    pair_key: str                    # <condition>_<replicate>_<fov>
    pair_id: Optional[str]           # Populated after pairing: <pair_key>_pair<M>
    centroid_x_px: float
    centroid_y_px: float
    area_px: float
    perimeter_px: float
    equivalent_diameter_um: float    # 2 * sqrt(Area / pi) * pixel_size_um
    volume_sphere_um3: float         # (pi / 6) * d^3
    volume_ellipsoid_um3: float      # (pi / 6) * major * minor^2
    volume_discrepancy_ratio: float  # |V_sphere - V_ellip| / V_sphere
    circularity: float               # 4 * pi * Area / Perimeter^2
    solidity: float                  # Area / ConvexHullArea
    eccentricity: float              # sqrt(1 - (minor/major)^2)
    interior_mean_intensity: float
    background_ring_mean_intensity: float
    contrast_ratio: float            # 1.0 - (I_int / I_ring)
    qc_flag: str                     # 'PASS', 'REVIEW', or 'FAIL'
    qc_reasons: str                  # Semicolon-delimited rejection explanations
    mask_source: str                 # 'automated' or 'manual_review'; backend is in seg_backend ('cpsam' | 'cpdino' | 'classical')
```

#### `SpheroidPairRecord` (`pair.py`)
Encapsulates a tracked temporal match across $t_0 \to t_7$:
```python
@dataclass
class SpheroidPairRecord:
    pair_id: str                     # <pair_key>_pair<N>
    pair_key: str                    # <condition>_<replicate>_<fov>
    condition: str
    replicate: str
    fov: str
    t0_object_id: str
    t7_object_id: str
    t0_volume_sphere_um3: float
    t7_volume_sphere_um3: float
    delta_volume_um3: float          # V7 - V0
    fold_change_volume: float        # V7 / V0
    log2_fold_change: float          # log2(V7 / V0)
    centroid_distance_px: float      # Euclidean shift
    denominator_gate_pass: bool      # True if d0 >= 60 um
    combined_qc_flag: str            # 'PASS', 'REVIEW', or 'FAIL'
    qc_reasons: str
```

### 2.3 Key Algorithms

#### Multi-Focal Plane Deduplication (`qc_filter.py`)
When multi-well plates are imaged with slight focal z-plane offsets across adjacent FOVs, the same physical spheroid can appear in multiple images. The pipeline identifies duplicate focal planes using spatial centroid proximity:
1. Candidate pairs across FOVs within the same well have displacement $< 35\,\text{px}$.
2. If two FOVs share $\ge 65\%$ overlapping spheroids or $\ge 3$ matching objects, an adjacency graph is constructed and connected components resolve physical FOV clusters.
3. Redundant spheroids are flagged `FAIL: duplicate_focal_plane` while retaining the best-focus representative.

#### Annular Ring Contrast Check (`qc_filter.py`)
To prevent segmenting bright hydrogel artifacts, phase halos, or bubbles:
1. Mask is dilated by $R_{\text{inner}} = 3\,\text{px}$ and $R_{\text{outer}} = 18\,\text{px}$.
2. Annular ring $A = \text{Dilate}(M, 18) \setminus \text{Dilate}(M, 3)$.
3. Spheroid interior must satisfy: $\bar{I}_{\text{interior}} \le 0.90 \times \bar{I}_{\text{ring}}$.

---

## 3. Immunofluorescence Pipeline (`matrix_mat.if_density`)

### 3.1 Execution Graph & Data Flow
```
Paired IF Channels (ch00=DAPI, ch02=Actin)
  │
  ├──► [audit_if_dataset] ──► IFPreflightReport (Validates pairs & models)
  │
  ├──► [discover_and_pair_fields] ──► Matches ch00 and ch02 files
  │
  ├──► [preprocess_field]
  │      ├─ Rolling-ball background subtraction (radius = 3.5 * expected_radius)
  │      └─ Percentile normalization to float32 [0.0, 1.0]
  │
  ├──► [CPSAMEngine] ──► Cellpose-SAM inference (MPS/CUDA/CPU)
  │
  ├──► [postprocess_field_masks]
  │      ├─ Deduplication (IoU > 0.50)
  │      ├─ Size gate in um³: [210, 3364] um³ (~0.25x to 4.0x median)
  │      ├─ Seeded watershed clump splitting (h-maxima peaks, split_factor=2.0)
  │      └─ Perimeter border clearance exclusion (5.85 um)
  │
  ├──► [qc.py] ──► Laplacian blur, SNR, saturation fraction, cohort density outliers
  │
  ├──► [features.py] ──► objects.csv, per_image.csv, summary_by_material.csv
  │
  ├──► [biological_plots.py] ──► 3 Publication figures (density, clean vs all, volume)
  │
  └──► [comparator_prepost.py] ──► ablation_effects.csv, robustness summary, panels
```

### 3.2 Post-Processing Module: `PP11_full` Winning Stack
The production pipeline uses the frozen configuration validated in Phase 4/5:
- **Preprocessing**: Rolling-ball background subtraction ($R = 3.5 \times r_{\text{nuc}} \approx 41\,\mu\mathrm{m}$) + Dynamic range percentile clipping $[1.0\%, 99.8\%]$.
- **Inference**: Cellpose-SAM `cpsam_v2` with `flow_threshold=0.4`, `cellprob_threshold=0.0`.
- **Postprocessing**:
  - Clump splitting: Triggered on masks $> 2.0 \times$ the configured `reference_vol_um3`. Computes $h$-maxima intensity peaks ($h=10.0$) and applies marker-controlled seeded watershed.
  - Border clearance: Nuclei within $5.85\,\mu\mathrm{m}$ (1 nuclear radius) of the image border are excluded from density calculations (`excluded_border=True`) to remove boundary truncation distortion.

---

## 4. Extension Guidelines for Maintainers

### How to Add a New Segmentation Backend
1. Implement a `BaseSegmenter` subclass in `src/matrix_mat/brightfield/segment.py` whose `segment(image, pixel_size_um)` returns `(labels_int32, meta)`, where `meta` has `backend`, `model_id`, `diameter_used` and `status` (`"success" | "empty" | "failed"`). Raise on model-initialisation errors; return `status="failed"` only for per-image inference errors.
2. Add the name to `SUPPORTED_BACKENDS` (segment.py) and the validation in `SegmentationConfig.__post_init__` (config.py), plus the `--backend` choices in `src/matrix_mat/cli.py`.
3. Add tests with a stub in `tests/unit/test_segment_policy.py` (strict policy, cache round trip, backend recorded on objects).

### How to Add a New Statistical Test
1. Implement it in `src/matrix_mat/core/stats.py` (pure functions, gel as the cluster unit).
2. Call it from `brightfield/stats.py` or `reporting/summaries.py` and write its columns into the summary CSV. Reporting modules must not compute statistics (enforced by `tests/unit/test_reporting_no_stats.py`).
3. Add unit tests with an independent oracle (e.g. `statsmodels` or brute-force enumeration) in `tests/unit/`.

### How to Run the Tests
```bash
pytest -m "not slow and not dl and not realdata"   # fast suite, as in CI
pytest -m dl                                       # needs torch, cellpose and the weights
pytest --update-golden tests/regression            # after an intended output change; review the diff
```
