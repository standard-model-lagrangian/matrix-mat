# Dataset Structure and Acquisition Specification

This document details the expected directory layout, image acquisition metadata, filename conventions, and optical calibration for brightfield spheroid and immunofluorescence (IF) microscopy datasets processed by `matrix-mat`.

---

## 1. Directory Structure

By default, datasets are organized under the path specified by the `MATRIX_MAT_DATA_ROOT` environment variable (defaulting to `./data`):

```
data/
├── brightfield/
│   ├── Day0/                  # Baseline Day 0 brightfield TIFFs
│   │   ├── S34D30-Gel1-FOV1.tif
│   │   ├── S34D30-Gel1-FOV2.tif
│   │   └── ...
│   └── Day7/                  # Endpoint Day 7 brightfield TIFFs
│       ├── S34D30-Gel1-FOV1.tif
│       ├── S34D30-Gel1-FOV2.tif
│       └── ...
└── if/                        # Immunofluorescence confocal TIFFs
    ├── SKOV3 Spheroid D7 Mor_Mat-Gel1_ch00.tif   # DAPI / Hoechst nuclear channel
    ├── SKOV3 Spheroid D7 Mor_Mat-Gel1_ch02.tif   # Phalloidin actin channel
    └── ...
```

Precedence order for path resolution is:
1. Explicit CLI arguments (`--t0-dir`, `--t7-dir`, `--data-dir`)
2. Environment variables (`MATRIX_MAT_DATA_ROOT`, `MATRIX_MAT_OUTPUT_ROOT`)
3. Shipped configuration defaults (`configs/*.yaml`)

---

## 2. Filename Conventions

### Brightfield Spheroids
Brightfield files follow the format `<Condition>-<GelID>-<FOV>.<ext>`:
- **Condition**: Hydrogel formulation code, e.g., `Mat` (Matrigel baseline control), `S34D30`, `S46D10`.
- **GelID**: Biological replicate identifier representing a discrete hydrogel culture unit, e.g., `Gel1`, `Gel2`, `Gel3`.
- **FOV**: Field-of-view identifier within the gel, e.g., `FOV1`, `FOV2`.

Pairs are matched between `Day0` and `Day7` by matching identical `(Condition, GelID, FOV)` coordinates.

### Immunofluorescence (IF) Confocal
IF files follow the EVOS/confocal channel naming:
`SKOV3 Spheroid D7 Mor_<Condition>-<GelID>_<Channel>.tif`
- `ch00`: Nuclear channel (DAPI / Hoechst 33342, 405 nm excitation).
- `ch02`: Spheroid boundary / cytoplasm channel (Phalloidin F-actin, 594 nm excitation).

Pairs of `(ch00, ch02)` sharing the same base identifier are automatically discovered and co-registered.

---

## 3. Optical Calibration & Pixel Size

Accurate physical scale extraction is mandatory for morphological measurement:
- **EVOS TIFF Tag 37510**: The pipeline inspects Tag 37510 JSON metadata (`Target.PixelSize` or `MicronsPerPixel`). Valid objective scales typically range between 0.05 and 20.0 $\mu\mathrm{m/pixel}$ (e.g. 1.518817 $\mu\mathrm{m/px}$ for 4× EVOS air objective, 0.505049 $\mu\mathrm{m/px}$ for 20× objective).
- **Fallback Policy**: If metadata is absent and no CLI override (`--pixel-size`) is provided, uncalibrated runs are halted unless explicitly permitted via `--force-uncalibrated`.

---

## 4. Obtaining the Experimental Data

For institutional access to the raw experimental datasets, contact the repository maintainers or refer to the laboratory data repository archive. Synthetic test fixtures and smoke test datasets are provided out-of-the-box in `tests/data/` and via programmatic generators.
