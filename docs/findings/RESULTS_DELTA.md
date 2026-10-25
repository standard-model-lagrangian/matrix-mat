# Results Delta: v0.1 → v0.2.1 (real data)

How the headline numbers moved when the statistics, QC and provenance changes were applied to the real
datasets. Both runs used cpsam_v2 (SHA-256 `0f1cc3f7…c667`) on the same 285 brightfield TIFFs.

- **v0.1:** `output/condition_summary.csv`, generated 2026-09-03.
- **v0.2.1:** re-run on 2026-10-24 with a fresh mask cache.

## Brightfield (Day 0 → Day 7 volume fold change)

Segmentation is unchanged: 1507 objects (1019 PASS / 30 REVIEW / 458 FAIL), 57 matched pairs (31 PASS), all
cpsam, all calibrated from EVOS tag 37510. The medians are identical. Only the inference changed.

| Condition | PASS pairs | Gels in analysis | Median FC | v0.1 95% CI (spheroid bootstrap) | v0.2.1 95% CI (gel→spheroid bootstrap) | v0.1 p (spheroid permutation) | v0.2.1 p_raw (gel permutation) | min attainable p | v0.2.1 p_adj (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| Mat (control) | 2 | 2 | 3.44 | 2.76–4.12 | 2.76–4.12 | — | — | — | — |
| S34D30 | 2 | 2 | 1.25 | 0.50–2.00 | 0.50–2.00 | 0.331 ns | 0.333 | 0.333 | 1.0 ns |
| S40D30 | 13 | 3 | 2.30 | 1.95–2.38 | 1.86–3.35 | **0.020 \*** | 0.300 | 0.100 | 1.0 ns |
| S43D20 | 6 | 2 | 1.60 | 1.19–1.92 | 1.08–1.99 | **0.039 \*** | 0.333 | 0.333 | 1.0 ns |
| S46D10 | 2 | 2 | 2.81 | 2.55–3.06 | 2.55–3.06 | 0.668 ns | 0.667 | 0.333 | 1.0 ns |
| S50 | 6 | 4 | 1.78 | 1.38–2.77 | 1.22–3.37 | 0.077 ns | 0.267 | 0.067 | 1.0 ns |

### What changed and why

- **The two v0.1 "significant" results (S40D30 \*, S43D20 \*) do not survive gel-level inference.** They came
  from treating spheroids from the same gel as independent.
- **This design cannot produce a significant result.** Only 2 control gels contribute PASS pairs, so the
  smallest attainable gel-level p-value is 0.07–0.33 per comparison, and Holm correction across five
  materials pushes every `p_adj` to 1.0. This is a limit of the data, not of the method.
- **Confidence intervals widened** where several spheroids came from few gels (S40D30: 13 spheroids in 3 gels).
  Where each gel contributed a single spheroid the interval is unchanged. CIs are now computed on log₂ FC and
  back-transformed; v0.2.0 had reported the log₂ interval next to ratio-scale medians.
- **The control rests on 2 PASS pairs**, so every comparison inherits that uncertainty.

### Recommended reading of the brightfield data

Report median fold change with gel-level CIs as effect sizes. Do not report significance stars. More gels per
condition (and more PASS pairs in the control) are needed for confirmatory inference.

## Immunofluorescence (planar nuclei density)

v0.1 reported a pseudo-volumetric "nuclei per mm³" that scales with 1/diameter. v0.2.1 reports planar density
(nuclei/mm² of spheroid cross-section), so the two versions' IF densities cannot be compared numerically.

The 2×2 pre/post ablation (`configs/prepost_ablation.yaml`) was re-run on all 27 paired fields with cpsam_v2 on
MPS. Gel-level density (mean of per-field densities within each gel), with a gel-label permutation test against
Mat and Holm correction across materials:

| Material | Fields | Gels | PP00 gel mean ± SD (/mm²) | PP00 p_raw → p_adj | PP11 gel mean ± SD (/mm²) | PP11 p_raw → p_adj | min attainable p |
|---|---|---|---|---|---|---|---|
| Mat (control) | 6 | 5 | 4498 ± 1000 | — | 4344 ± 966 | — | — |
| S34D30 | 3 | 3 | 4367 ± 345 | 1.00 → 1.00 | 4261 ± 245 | 1.00 → 1.00 | 0.054 |
| S40D30 | 4 | 4 | 4440 ± 2077 | 0.69 → 1.00 | 4420 ± 2713 | 0.41 → 1.00 | 0.024 |
| S43D20 | 6 | 5 | 5323 ± 1798 | 0.71 → 1.00 | 4672 ± 1526 | 1.00 → 1.00 | 0.048 |
| S46D10 | 1 | 1 | 3574 | untested (1 gel) | 2992 | untested (1 gel) | — |
| S50 | 7 | 5 | 3021 ± 1522 | 0.048 → 0.19 | 2880 ± 1386 | 0.12 → 0.48 | 0.048 |

- **No material differs from Mat after correction,** under either pipeline variant. The lower S50 density is the
  largest effect but is not significant at gel level.
- **Pre/post-processing changes counts by only −1.5% to −3.5% per field** (83.2 → 80.3 nuclei/field, PP00 → PP11).
  The drop in flagged images (44% → 15%) is mostly the `debris_or_merged` flag, which the size gate removes
  by construction.
- **Blur QC (absolute threshold 150) flags only Mat-Gel1-3.** The old percentile rule always flagged about 3 fields.
- **The robustness comparison (001–004 vs 201–204) has not been re-run under v0.2.1.** Its earlier conclusions
  used the old density units and are not carried forward.
- **Thresholds are provisional:** the reference nuclear volume (841.04 µm³) and blur threshold (150) were derived
  from this dataset.
