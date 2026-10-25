# Beginner's Guide

This guide is for biologists and students running the pipelines for the first time. It assumes you can open a
terminal and paste commands. Installation is in the [README](../README.md#installation); everything below assumes
the virtual environment is active (`source .venv/bin/activate`).

---

## 1. What the two pipelines do

**Brightfield (`matrix-mat bf`)** works on transmitted-light images of spheroids in hydrogels, taken at Day 0 and
Day 7 in the same fields.

- Finds every spheroid and measures its area, diameter, circularity and an estimated volume (V = π/6 · d³,
  which assumes a sphere).
- Follows each spheroid from Day 0 to Day 7 by matching nearby centroids within the same field, then computes its
  volume fold change.
- Compares each hydrogel with the Matrigel control (`Mat`) at the level of the gel.

**Immunofluorescence (`matrix-mat if`)** works on fixed spheroids stained for nuclei (DAPI/Hoechst, `_ch00`) and
F-actin (phalloidin, `_ch02`).

- Segments individual nuclei with Cellpose-SAM.
- Outlines the spheroid from the actin channel.
- Reports nuclear packing as nuclei per mm² of spheroid cross-section. The images are single optical sections, so
  this is a planar density, not a 3D one.

---

## 2. Put your images in place

```
data/
├── brightfield/
│   ├── Day0/   Day0 Mat gel1_0001_TRANS.tif …
│   └── Day7/   Day7 Mat gel1_0001_TRANS.tif …
└── if/         SKOV3 Spheroid D7 Mor_Mat-Gel1-2_ch00.tif  +  …_ch02.tif
```

Then tell the pipelines where `data/` is:

```bash
export MATRIX_MAT_DATA_ROOT=/path/to/data
```

### File names matter

| Pipeline | Pattern | Example |
| :--- | :--- | :--- |
| Brightfield | `Day<N> <Condition> gel<G>_<FOV>_TRANS.tif` | `Day0 S34D30 gel5_0003_TRANS.tif` is Day 0, hydrogel S34D30, gel 5, field 3 |
| IF | `SKOV3 Spheroid D7 Mor_<Material>-Gel<G>[-…]_ch00.tif` / `_ch02.tif` | `…Mor_S34D30-Gel1-1_ch00.tif` with its `_ch02` partner |

The **gel** is the biological replicate, and statistics are done per gel. A brightfield file whose name doesn't
fit the pattern is rejected with an error rather than guessed. If your folders have other names, symlink them into
the layout above or pass them directly with `--t0-dir`/`--t7-dir` or `--data-dir`. See [DATA.md](DATA.md).

---

## 3. Brightfield, step by step

1. **Preflight.** Checks the folders, Day 0/Day 7 pairing, the control group and the optical calibration.
   ```bash
   matrix-mat bf preflight
   ```
   You should see `Status: PASSED`, the number of images per day, and `Optical Scale Source: TIFF Tag 37510`.

2. **Quick sample run.** A stratified subset of 12 images.
   ```bash
   matrix-mat bf run --sample 12 --output-dir output_sample
   ```
   Open a few images in `output_sample/overlays/` and check that the outlines sit on the spheroids.

3. **Full run.**
   ```bash
   matrix-mat bf run
   ```
   Results go to `./output/` (or `$MATRIX_MAT_OUTPUT_ROOT`). Segmentation masks are cached, so a re-run with the
   same settings is fast.

4. **Re-draw the figures only.**
   ```bash
   matrix-mat bf plots
   ```

If a step fails, the command exits with an error that says why: for example, missing model weights, an
uncalibrated image, or a typo in the config.

---

## 4. Immunofluorescence, step by step

```bash
matrix-mat if preflight                                        # checks _ch00/_ch02 pairs and the weights
matrix-mat if sweep --config configs/prepost_ablation.yaml     # 4 pipeline variants → ./output/runs/
matrix-mat if compare-prepost --biological                     # comparison report + figures for the full pipeline
```

To try it on a couple of fields first, add `--sample 2` to `if sweep`.

The four variants turn background correction (pre-processing) and clump splitting / size gate / border exclusion
(post-processing) on and off. `104_PP11_full` has both on.

---

## 5. Reading the results

### QC flags (brightfield `objects.csv`, overlay colours)

| Flag | Colour | Meaning |
| :--- | :--- | :--- |
| `PASS` | green | Isolated spheroid that meets the size, border, contrast and shape criteria. Used in statistics. |
| `REVIEW` | orange | Probably real but unusual, e.g. irregular outline or extreme growth. Kept, but excluded from statistics by default. |
| `FAIL` | red | Rejected: debris, touching the image border, bright bubble, or duplicate focal plane. The reason is in `qc_reasons`. |
| `SEG_FAIL` | — | Segmentation of the whole image failed or was implausible. The image is not re-segmented with another method. |

Nothing is dropped silently: every object stays in `objects.csv` with its flag.

### `condition_summary.csv` (brightfield)

| Column | Meaning |
| :--- | :--- |
| `n_spheroids`, `n_gels` | Analysed PASS pairs, and the gels they come from (the inference unit) |
| `fold_change_median` | Median Day 7 / Day 0 volume ratio |
| `bootstrap_ci_95_low/high` | 95% CI for that median, resampling gels then spheroids |
| `p_raw`, `p_adj` | Gel-level permutation test against Mat, raw and Holm-adjusted |
| `min_achievable_p` | Smallest p-value this number of gels can produce. If it's above 0.05, no result can be significant. |
| `significance` | Stars from `p_adj`, `ns`, `control`, or `untested` |

### `well_growth_comparison.csv` (brightfield, descriptive)

- `fc_pooled_vol_density`: total spheroid volume in the well, Day 7 / Day 0. Overall growth.
- `fc_mean_volume`: mean spheroid size, Day 7 / Day 0. Are spheroids getting bigger?
- `fc_count_density`: number of spheroids, Day 7 / Day 0. Are there more of them?

### `summary/summary_by_material.csv` (IF)

- `gel_mean_nuclei_per_mm2`: mean density per gel, averaged over gels. This is the number that is compared.
- `mean_nuclei_per_mm2_all` / `_clean`: field-level means for all fields, and for fields without QC flags. Descriptive.
- `p_raw`, `p_adj`, `min_achievable_p`, `significance`: as above. Materials with a single gel are `untested`.

### How much to trust a "not significant"

With 2–4 gels per condition, the smallest possible p-value is often above 0.05 before any correction. A
non-significant result here usually means "not enough gels", not "no effect". Report the effect size and its CI.
See [RESULTS_DELTA](findings/RESULTS_DELTA.md) for the current dataset.

---

## 6. Changing parameters

All parameters are in `configs/*.yaml`, with comments. A misspelled or unknown key is an error, so typos can't be
silently ignored. Commonly adjusted brightfield settings:

| Key | Default | Meaning |
| :--- | :--- | :--- |
| `default_pixel_size_um` | 1.518817 | µm per pixel (EVOS 4×). Only used if the image has no calibration; such images are flagged unless `--force-uncalibrated` is given. |
| `segmentation.primary_backend` | `cpsam` | `cpsam`, `cpdino` (extra install) or `classical`. One backend per run. |
| `qc.min_d_um` / `qc.max_d_um` | 40 / 1500 | Size limits for an object to count as a spheroid (µm) |
| `qc.border_margin_px` | 15 | Objects closer than this to the image edge are rejected |
| `qc.ring_contrast_factor` | 0.90 | The spheroid must be at least 10% darker than the ring of background around it |
| `qc.single_mask_max_area_pct` | 0.25 | A single object covering more than 25% of the image means the segmentation failed |
| `qc.min_circularity` | 0.65 | Below this the object is flagged `REVIEW` |
| `pairing.min_t0_d_um_denominator` | 60 | Pairs with a Day 0 diameter under 60 µm are excluded from fold-change statistics |
| `stats.control_condition` | `Mat` | Reference condition for tests |
| `stats.include_review_in_stats` | false | Whether REVIEW pairs are analysed |

For IF, the two key thresholds are `postprocessing.size_gate.reference_vol_um3` and `qc.blur_abs_threshold`. Both
are currently provisional.

Override paths or settings for a single run from the command line, e.g.
`matrix-mat bf run --pixel-size 0.607527 --condition S34D30 -o output_10x`.

---

## 7. Smaller, larger or messier datasets

- **Any number of images.** Files are discovered from the folders, so there is no list to maintain.
- **Missing Day 0 or Day 7 partner.** Those images are still segmented and measured, but contribute no fold
  changes. Preflight reports how many singletons there are.
- **Missing IF channel.** Preflight warns; add `--allow-unpaired` to skip incomplete pairs.
- **Few spheroids.** One PASS pair gives a degenerate CI (`[x, x]`), zero gives NaN, and a condition with one gel
  is reported as `untested`.
- **New conditions.** Figures and tables pick up whatever condition names are in the filenames.
