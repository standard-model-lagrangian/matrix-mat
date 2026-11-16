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

The whole name must match. Tokens may be separated by spaces or underscores (`Day0_Mat_gel1_0001_TRANS.tif`).
Names are normalised: the control is always spelled `Mat` (`mat`, `MAT`), `S<n>D<n>` conditions are
upper-cased and joined (`S34 D30` and `s34d30` become `S34D30`), `gel01` becomes `gel1`, and FOV numbers
are zero-padded to four digits (`001` becomes `0001`).

Day 0 and Day 7 objects are paired within the same `(Condition, gel, FOV)`. A filename that does not match
raises `ValueError`; there is no default gel or FOV. These names are also rejected:

- file-manager copies such as `… _TRANS (1).tif` or `… _TRANS copy.tif` (they would pair twice);
- conditions that cannot be resolved unambiguously, such as `Mat+drug` or `Mat extra`.

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
- **OME-XML** (ImageDescription): `PhysicalSizeX` in `PhysicalSizeXUnit` (default µm), any OME schema year,
  with or without a leading Bio-Formats comment.
- **ImageJ / Fiji**: `pixelWidth`, or (the usual case) XResolution in pixels per ImageJ `unit` with TIFF
  ResolutionUnit NONE. The unit (`um`, `micron`, `µm`, `nm`, `mm`, `cm`, `inch`, …) is converted to µm;
  `unit=pixel` means uncalibrated.
- **TIFF resolution tags** in pixels per cm or per inch.
- **Rejected values**: 72/96 DPI screen defaults and values outside 0.05–20 µm/px are never used. The configured
  default pixel size is used instead, with `is_calibrated = False` and a `scale_source` naming the reason
  (e.g. `config_default_uncalibrated (72 DPI screen default)`).
- **Fallback Policy**: If metadata is absent and no CLI override (`--pixel-size`, must be > 0) is provided, uncalibrated runs are halted unless explicitly permitted via `--force-uncalibrated`.

---

## 4. Obtaining the Experimental Data

For institutional access to the raw experimental datasets, contact the repository maintainers or refer to the laboratory data repository archive. Synthetic test fixtures are provided in `tests/fixtures/` (brightfield and IF).
