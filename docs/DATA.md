# Dataset Structure and Acquisition Specification

This document details the expected directory layout, image acquisition metadata, filename conventions, and optical calibration for brightfield spheroid and immunofluorescence (IF) microscopy datasets processed by `matrix-mat`.

---

## 1. Directory Structure

Shipped configs locate data through `${DATA_ROOT}` (environment variable `MATRIX_MAT_DATA_ROOT`, default `./data`):

```
data/
├── brightfield/
│   ├── Day0/        # e.g. "Day0 Mat gel1_0001_TRANS.tif"
│   └── Day7/        # e.g. "Day7 Mat gel1_0001_TRANS.tif"
└── if/              # e.g. "SKOV3 Spheroid D7 Mor_Mat-Gel1-2_ch00.tif" (+ matching _ch02)
```

The original acquisition folders are named differently (`Spheroid Day0 260805`, `Spheroid D7 260812`,
`spheroids staining:IF /nuclei density analysis`, note the trailing spaces and colon). Either symlink them into
the layout above or pass them explicitly, e.g.
`matrix-mat bf run --t0-dir "<…>/Spheroid Day0 260805" --t7-dir "<…>/Spheroid D7 260812"`.

Other roots: `${OUTPUT_ROOT}` (`MATRIX_MAT_OUTPUT_ROOT`, default `./output`), `${MODELS_ROOT}`
(`MATRIX_MAT_MODELS_ROOT`, default `./models`), and `${CACHE_ROOT}` (`MATRIX_MAT_CACHE_DIR`, default
`${OUTPUT_ROOT}/cache/masks`).

Precedence: explicit CLI flags > `MATRIX_MAT_*` environment variables > defaults.

---

## 2. Filename Conventions

### Brightfield spheroids

The parser (`matrix_mat.core.naming.parse_image_filename`) expects

`Day<N> <Condition> gel<G>_<FOV>_TRANS.tif`, e.g. `Day0 S34D30 gel2_0003_TRANS.tif`

- **Day** `<N>`: the timepoint (`Day0`, `D7`, …). It can instead come from the source folder when configured.
- **Condition**: hydrogel formulation (`Mat` is the Matrigel control, `S34D30`, `S46D10`, …).
- **gel** `<G>`: the biological replicate. This is the unit of statistical inference.
- **FOV**: zero-padded field of view within the gel.

Day 0 and Day 7 objects are paired within the same `(Condition, gel, FOV)`. A filename that does not match
raises an error; there is no default gel or FOV.

### Immunofluorescence (IF)

`SKOV3 Spheroid D7 Mor_<Material>-Gel<G>[-…]_<channel>.tif`
- `ch00`: nuclear channel (DAPI / Hoechst).
- `ch02`: F-actin (phalloidin), used for the spheroid cross-section mask.

`Gel<G>` is the replicate (inference unit). Multiple fields from one gel (`Gel1-2`, `Gel1-3`) are averaged
before testing.

---

## 3. Optical Calibration & Pixel Size

Accurate physical scale extraction is mandatory for morphological measurement:
- **EVOS TIFF Tag 37510**: The pipeline inspects Tag 37510 JSON metadata (`Target.PixelSize` or `MicronsPerPixel`). Valid objective scales typically range between 0.05 and 20.0 $\mu\mathrm{m/pixel}$ (e.g. 1.518817 $\mu\mathrm{m/px}$ for 4× EVOS air objective, 0.505049 $\mu\mathrm{m/px}$ for 20× objective).
- **Fallback Policy**: If metadata is absent and no CLI override (`--pixel-size`) is provided, uncalibrated runs are halted unless explicitly permitted via `--force-uncalibrated`.

---

## 4. Obtaining the Experimental Data

For institutional access to the raw experimental datasets, contact the repository maintainers or refer to the laboratory data repository archive. Synthetic test fixtures are provided in `tests/fixtures/` (brightfield and IF).
