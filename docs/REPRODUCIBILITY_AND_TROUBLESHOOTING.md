# Reproducibility Protocol & Troubleshooting Guide

This guide details the mechanisms built into the image processing pipelines to ensure **100% computational reproducibility**, auditable scientific provenance, and foolproof error recovery across environments.

---

## 1. The Reproducibility Contract

Image analysis in biomaterials and pharmacology is often plagued by hidden sources of irreproducibility:
- Floating-point differences across CPU, CUDA, and Apple Silicon MPS hardware.
- Uncalibrated or mismatched microscope optical scales ($\mu\mathrm{m/px}$).
- Silent fallback to unverified defaults.
- Unrecorded environment states or software library version drift.

To eliminate these failure modes, our pipelines enforce a strict **Reproducibility Contract**:

```
                              ┌─────────────────────────────────────────────────────────┐
                              │                 PREFLIGHT INTEGRITY CHECK               │
                              │ • Verifies image paths & channel pairing                │
                              │ • Validates TIFF bit depths & dynamic ranges            │
                              │ • Audits optical scale metadata (Tag 37510)             │
                              │ • Confirms control condition ('Mat') presence           │
                              └────────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │               AUDITABLE RUN METADATA & MANIFEST         │
                              │ • manifest.csv: File paths, shapes, SHA-256 checksums   │
                              │ • run_metadata.json: Git commit, packages, GPU, seed    │
                              │ • DECISIONS.md: Every fallback decision logged in time  │
                              └────────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                              ┌─────────────────────────────────────────────────────────┐
                              │                 DETERMINISTIC PROCESSING                │
                              │ • Fixed random seeds for bootstrap CIs & permutations   │
                              │ • Zero silent drops: PASS / REVIEW / FAIL logged        │
                              │ • Lossless 16-bit PNG disk caching                      │
                              └─────────────────────────────────────────────────────────┘
```

---

## 2. Auditing Run Metadata & Data Manifests

Every single execution generates two self-documenting audit files in the output directory:

### 2.1 `run_metadata.json`
Captures complete environment provenance and execution parameters:
```json
{
  "pipeline_name": "spheroid_pipeline_v2",
  "timestamp_utc": "2026-09-09T14:28:09.123456+00:00",
  "wall_time_seconds": 14.82,
  "environment": {
    "python_version": "3.13.0",
    "platform": "macOS-15.0-arm64-arm-64bit",
    "hostname": "workstation-01",
    "architecture": "arm64",
    "hardware_accelerator": "Apple Silicon MPS (Apple M4 / arm64)",
    "git": {
      "commit": "a1b2c3d4e5f6...",
      "branch": "main",
      "is_dirty": false,
      "status_summary": "clean"
    }
  },
  "dependency_versions": {
    "torch": "2.5.0",
    "numpy": "2.1.0",
    "pandas": "2.2.0",
    "cellpose": "3.1.0",
    "scipy": "1.14.0",
    "skimage": "0.24.0",
    "opencv": "4.10.0"
  },
  "random_seed": 42,
  "resolved_parameters": {
    "default_pixel_size_um": 1.518817,
    "segmentation_backend": "cpsam",
    "qc_min_d_um": 40.0,
    "qc_max_d_um": 1500.0,
    "stats_control_condition": "Mat",
    "stats_bootstrap_iterations": 2000,
    "stats_permutation_iterations": 10000
  },
  "warnings_logged": [
    "Found 16 unpaired Day 0 and 29 unpaired Day 7 images."
  ],
  "warnings_count": 1
}
```

### 2.2 `manifest.csv`
Documents every input image file processed, its SHA-256 cryptographic checksum, physical dimensions, bit depth, optical scale, and pairing status:
```csv
image_id,timepoint,condition,replicate,fov,pair_key,file_path,file_size_bytes,sha256,dimensions,bit_depth,pixel_size_um,scale_source,is_matched_pair
Day0 Mat gel1_0001_TRANS,t0,Mat,gel1,0001,Mat_gel1_0001,/path/to/img.tif,3146244,7f8e3b...,1536x2048,uint8,1.518817,evos_metadata,True
```

---

## 3. Warnings Dictionary & Remediation

When the pipeline encounters an unexpected or degraded scenario, it does not guess silently. It logs a structured warning and records it in `run_metadata.json`:

| Warning Message | Root Cause | Impact on Results | Remediation Step |
| :--- | :--- | :--- | :--- |
| `TIFF optical scale tag (37510) not detected... Pipeline will assume default 1.518817 um/pixel.` | Image metadata is missing or was stripped during export from microscope. | Diameters and volumes will be scaled by the default. If actual objective was not 4x, volumes will be inaccurate. | Pass the known optical scale via CLI: `--pixel-size <val>` or set `default_pixel_size_um` in config. |
| `Found N unpaired Day 0 and M unpaired Day 7 images.` | Certain wells were only imaged at Day 0 or Day 7 (singletons). | These FOVs cannot have paired fold change ($V_7/V_0$) computed; they are included in population stats only. | Verify well imaging inventory; ensure naming matches `Day<0\|7> <Cond> gel<N>_<FOV>_TRANS.tif`. |
| `Control condition 'Mat' not found in dataset conditions.` | Dataset does not contain the default reference control group (`Mat`). | Two-sample permutation tests cannot run; $p$-values will be omitted. | Specify your baseline group in config under `stats.control_condition: "YourControl"` or `--config`. |
| `Degraded image contrast detected in sample: <file> (zero variance / blank).` | Image contains all black/white pixels or illumination failed during imaging. | FOV will produce 0 segmented objects. | Inspect raw TIFF; check microscope light source or camera shutter for that well. |
| `Skipping N unpaired field(s) without complete ch00/ch02 pairs.` | An IF field has a nuclear channel (`ch00`) but is missing an actin channel (`ch02`), or vice-versa. | Unpaired field is excluded from sweep to prevent invalid multi-channel quantification. | Provide missing channel TIFF, or run with `--allow-unpaired` to continue with complete pairs. |
| `Local Cellpose-SAM weights not found at 'models/cpsam_v2'.` | Local weights file was deleted or moved. | Cellpose will attempt to download model weights from HuggingFace via internet. | Place `cpsam_v2` model weights in `models/cpsam_v2`. |
| `Reached max tuning cycles (8) in sample 12 mode.` | Sample images failed one of the 4 programmatic sanity gates after 8 auto-tuning cycles. | Parameters may need manual adjustment for this batch. | Inspect `output/cpsam_tuning_inspection/` or overlay PNGs; adjust adaptive window sizes in config. |

---

## 4. Bad Scenario Recipes & Troubleshooting

### Scenario A: Microscope Scale Was Stripped / ImageJ Uncalibrated
**Symptom**: `TIFF optical scale tag (37510) not detected`.
**Solution**:
1. Check objective magnification used during acquisition:
   - EVOS 4x objective: `1.518817` $\mu\mathrm{m/pixel}$
   - EVOS 10x objective: `0.607527` $\mu\mathrm{m/pixel}$
   - EVOS 20x objective: `0.303763` $\mu\mathrm{m/pixel}$
2. Re-run pipeline passing the explicit scale:
   ```bash
   python run_pipeline.py --pixel-size 0.607527 --output-dir output_10x
   ```

### Scenario B: Running on a Remote Cluster Without GPU
**Symptom**: No GPU detected; execution is slow.
**Solution**:
The pipeline automatically falls back to CPU with thread optimization. For faster execution on CPU:
- Use stratified subsampling first: `python run_pipeline.py --sample 12`.
- When running the full plate, use `spheroid_brightfield.yaml` with cached masks enabled (`cache_dir: "masks"`). Once a mask is computed, re-runs take $<5$ seconds.

### Scenario C: Outlier FOVs with Severe Focal Drift (Blurry)
**Symptom**: In IF analysis, `flag_blurry=true` appears in `qc_flags.csv`.
**Scientific Rationale**:
In single-plane fluorescence microscopy, focal drift during acquisition places spheroids out of the depth of field. Deconvolution cannot restore missing high-frequency optical information.
**Remediation**:
- The pipeline flags these fields (`flag_blurry`) and provides clean-vs-all cohort comparisons (`fig_if_clean_vs_all_density.png`).
- For future acquisitions, configure hardware autofocus z-stacks.
