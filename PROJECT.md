# Project: Spheroid Volume Pipeline v2 (spheroid_pipeline_v2)

## Architecture
End-to-end multi-spheroid instance segmentation, robust background contrast gating, denominator gating, temporal tracking, statistical hypothesis testing, and self-verification pipeline for brightfield hydrogel spheroid cultures.

```
TIFF Image (1536x2048, EVOS Tag 37510)
  │
  ▼
[scale_extractor] ──► pixel_size_um (1.518817 um/px)
  │
  ▼
[preprocess] ──► Background flattening + Downsampling (150-250px expected d)
  │
  ▼
[segmentation] ──► 3-Tier Hierarchy: Cellpose-SAM ──► Cellpose cyto3 ──► Classical Multi-Scale Watershed
  │                Disk Caching: masks/<image_id>.png | Logging: DECISIONS.md
  │
  ▼
[qc_filter] ──► 1. Overlap Resolution
                2. Size Gate (40 - 1500 um)
                3. Border Margin Clearance (15 px)
                4. Background Ring Contrast Check (I_int <= 0.90 * I_ring)
                5. Area Plausibility Gate (< 25% FOV area)
                6. Morphometrics (Area, Perim, Circ, Sol, Ecc, Vol_sphere, Vol_ellip)
                7. QC Flagging (PASS / REVIEW / FAIL)
  │
  ▼
[pairing] ──► Hungarian Mutual Nearest Centroid Matching (Day 0 ──► Day 7)
              Max displacement tolerance: max(50px, 1.5 * mean_radius)
  │
  ▼
[stats] ──► Denominator Gate (d_0 >= 60 um)
            Fold Change Calculation (V7 / V0), Flag FC not in [0.1, 20.0]
            Bootstrap 95% CI on Median Fold Change (B = 2000)
            Permutation Test vs Control 'Mat' (M = 10000)
            Exclusion Auditing Table
  │
  ▼
[supervision] ──► objects.csv, pairs.csv, condition_summary.csv, validation.csv
                  Overlay PNGs with contours & labels
                  Contact sheets by condition & replicate
                  Publication figures: log-scale violin & scatter with identity (PNG + PDF)
                  Executive report.md & interactive report.html
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Optical Scale Extraction | Extract 1.518817 um/px from TIFF Tag 37510 JSON or CLI override | M1 | Survey / Spec |
| 2 | Image Preprocessing & Downsampling | Illumination flattening and scale factor downsampling (150-250px) | M1 | Survey / Spec |
| 3 | Segmentation Hierarchy (Cellpose-SAM) | Primary zero-shot instance segmentation backend (`cpsam`) | M1 | Survey / Spec |
| 4 | Segmentation Fallback (Cellpose cyto3) | Secondary backend with diameter sweep | M1 | Survey / Spec |
| 5 | Segmentation Last Resort (Classical Watershed) | Deterministic multi-window adaptive thresholding + distance watershed | M1 | Survey / Spec |
| 6 | Hardware Acceleration & CPU Fallback | MPS (Apple M4), CUDA, CPU fallback with `PYTORCH_ENABLE_MPS_FALLBACK=1` | M1 | Survey / Spec |
| 7 | Mask Disk Caching | Lossless 16-bit PNG disk caching in `masks/<image_id>.png` | M1 | Survey / Spec |
| 8 | Architectural Decision Logging | Append all runtime backend decisions/fallbacks to `DECISIONS.md` | M1 | Survey / Spec |
| 9 | Overlap Resolution | Sort candidate instances by area/score and partition | M2 | Survey / Spec |
| 10 | Physical Size Gate | Reject objects outside [40, 1500] um | M2 | Survey / Spec |
| 11 | Border Margin Gate | Reject objects touching or within 15 px of image boundaries | M2 | Survey / Spec |
| 12 | Background Ring Contrast Gate | Annular ring extraction; reject objects not >=10% darker than ring | M2 | Survey / Spec |
| 13 | Single Mask Max Area Gate | Reject any single mask occupying >25% of FOV area (`SEG_FAIL`) | M2 | Survey / Spec |
| 14 | Morphological & Volumetric Suite | Compute Area, Perim, Circ, Sol, Ecc, d, V_sphere, V_ellip, Discrepancy | M2 | Survey / Spec |
| 15 | QC Flagging Protocol | Assign PASS / REVIEW / FAIL without dropping data | M2 | Survey / Spec |
| 16 | Mutual Nearest Centroid Pairing | Bipartite Hungarian matching across t0 -> t7 with max displacement | M3 | Survey / Spec |
| 17 | Denominator Gate | Exclude pairs with t0 diameter < 60 um from fold change stats | M3 | Survey / Spec |
| 18 | Trajectory Fold Change Gate | Flag REVIEW for matched pairs with V7/V0 outside [0.1, 20.0] | M3 | Survey / Spec |
| 19 | Condition-Level Statistics | Median/IQR of V0, V7, delta V, FC; Bootstrap 95% CI on median FC | M3 | Survey / Spec |
| 20 | Permutation Hypothesis Testing | Non-parametric permutation test (M=10000) vs Mat control | M3 | Survey / Spec |
| 21 | Multilevel Exclusion Auditing | Comprehensive 10-step exclusion tracking table | M3 | Survey / Spec |
| 22 | Synthetic Smoke Test Suite | 3 synthetic images (isolated, touching pair, bright blob), assert err<=10%, 0 masks on blob | E2E | Survey / Spec |
| 23 | Programmatic Sanity Gates (--sample 12) | Stratified 12-sample test with 4 sanity criteria, up to 8 tuning cycles | E2E / M4 | Survey / Spec |
| 24 | Full Dataset Batch Runner | Batch execution over all 285 TIFF images (120 matched pairs + 45 singletons) | M4 | Survey / Spec |
| 25 | Supervision Overlays | High-contrast RGB overlay PNGs with contours, IDs, d (um), QC flags | M4 | Survey / Spec |
| 26 | Contact Sheets | Condition/replicate grid contact sheets for high-throughput QC | M4 | Survey / Spec |
| 27 | Publication Figures (PNG+PDF) | Log-scale violin plots & scatter plots with identity line (PNG + vector PDF) | M4 | Survey / Spec |
| 28 | Executive Reports (MD & HTML) | Summary `report.md` and interactive responsive `report.html` | M4 | Survey / Spec |
| 29 | Review Folder & Validation Manager | Manual mask ingestion from `review/`, compute Dice/IoU in `validation.csv` | M4 | Survey / Spec |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Test Track | Test runner, synthetic smoke test harness, sample 12 sanity gate harness, Tiers 1-4 tests, TEST_INFRA.md, TEST_READY.md | None | IN_PROGRESS |
| 1 | M1: Foundation, Segmentation & Caching | Package structure, scale extractor, preprocessing, 3-tier segmentation hierarchy, MPS/CPU, mask caching, DECISIONS.md | None | IN_PROGRESS |
| 2 | M2: Filtering, Contrast & QC Gating | Overlap resolution, size gate, border gate, background ring contrast check, 25% area gate, morphometrics, QC flags | M1 | PLANNED |
| 3 | M3: Pairing, Denominator Gate & Stats | Hungarian pairing, denominator gate (d0>=60um), fold change, Bootstrap 95% CI, Permutation test vs Mat, exclusion audit | M2 | PLANNED |
| 4 | M4: Integration, Verification & Artifacts | CLI runner, sample 12 tuning loop, full dataset execution, overlays, contact sheets, publication figures, report.md/html, review manager | M3, E2E | PLANNED |

## Code Layout
Target directory: `/Users/frederick/Visual Studio /Material discovery/spheroid_pipeline_v2`
```
spheroid_pipeline_v2/
├── __init__.py
├── config.py             # Dataclasses, YAML loader, defaults, threshold constants
├── scale_extractor.py    # TIFF metadata extraction (Tag 37510 / OME-XML) + CLI override
├── preprocess.py         # Illumination correction, contrast normalization, downsampling
├── segment.py            # 3-tier hierarchy: Cellpose-SAM, Cellpose cyto3, Classical watershed
├── mask_cache.py         # Lossless 16-bit PNG mask caching in masks/<image_id>.png
├── decisions.py          # DECISIONS.md structured logging
├── qc_filter.py          # Size gate, border margin, background ring contrast, 25% area gate
├── measure.py            # Morphometrics (A, P, circ, sol, ecc, d, V_sphere, V_ellip)
├── pair.py               # Hungarian bipartite centroid matching with displacement limits
├── stats.py              # Denominator gating, Bootstrap 95% CI, permutation test, exclusions
├── visualize.py          # Overlay PNGs, contact sheets, publication figures (PNG + PDF)
├── report.py             # report.md and standalone report.html generation
├── review.py             # Ingestion of manual masks from review/, Dice/IoU calculation
├── run_pipeline.py       # Main CLI entry point (--smoke-test, --sample 12, --full-dataset)
└── tests/
    ├── __init__.py
    ├── test_synthetic_smoke.py  # 3 synthetic scenarios (isolated, touching pair, bright blob)
    ├── test_tier1_features.py   # Unit & functional tests per feature
    ├── test_tier2_boundaries.py # Boundary value & edge cases
    ├── test_tier3_pairwise.py   # Cross-feature combinations
    ├── test_tier4_workloads.py  # Real-world image & sample 12 sanity gates
    └── run_e2e_tests.py         # Comprehensive E2E test runner
```

## Interface Contracts
### `scale_extractor.py` ↔ `preprocess.py` / `segment.py`
- `extract_pixel_size(image_path: str, override_um: Optional[float] = None) -> Tuple[float, str]`
  - Returns `(pixel_size_um, source_description)` (e.g. `(1.518817, "TIFF Tag 37510 JSON")`).

### `segment.py` ↔ `mask_cache.py` ↔ `qc_filter.py`
- `SegmentationHierarchy.segment(image: np.ndarray, image_id: str, pixel_size_um: float) -> Tuple[np.ndarray, Dict[str, Any]]`
  - Returns `(label_mask_int32, metadata_dict)` where `label_mask_int32` has shape `(H, W)`.

### `qc_filter.py` ↔ `measure.py`
- `filter_and_measure_objects(image: np.ndarray, raw_label_mask: np.ndarray, image_metadata: Dict, config: PipelineConfig) -> Tuple[np.ndarray, List[SpheroidObjectRecord]]`
  - Applies overlap resolution, size gate, border margin, background ring contrast check, and morphometrics.

### `pair.py` ↔ `stats.py`
- `pair_timepoints(t0_objects: List[SpheroidObjectRecord], t7_objects: List[SpheroidObjectRecord], config: PipelineConfig) -> Tuple[List[SpheroidPairRecord], List[SpheroidObjectRecord], List[SpheroidObjectRecord]]`
- `compute_condition_statistics(pairs: List[SpheroidPairRecord], config: PipelineConfig) -> pd.DataFrame`
