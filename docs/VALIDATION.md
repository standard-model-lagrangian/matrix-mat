# Pipeline Validation & Verification

This document summarizes the validation benchmarks for `matrix-mat`, including ground-truth synthetic recoveries, manual expert annotation agreement, and cross-backend segmentation comparisons.

---

## 1. Synthetic Ground-Truth Recovery

Synthetic test fixtures with mathematically defined geometry test boundary recovery and artifact rejection without biological noise:

| Benchmark Scenario | Ground Truth Parameter | Target Metric | Recovered Performance |
| :--- | :--- | :--- | :--- |
| **Isolated Spheroid** | $d = 100\,\mathrm{px}$ ($151.88\,\mu\mathrm{m}$) | Diameter Error $< 10\%$ | $\Delta d < 2.5\%$ |
| **Touching Pair** | 2 spheroids, $d = 80\,\mathrm{px}$ each | Instance separation & area | 2 distinct instances, both within 5% area |
| **Refractive Bubble / Gel Artifact** | $d = 120\,\mathrm{px}$, inverted contrast | Rejection rate | 100% rejected by annular background ring gate |

---

## 2. Manual Annotation Agreement

Segmentation accuracy was benchmarked against independent manual counts across representative fields spanning high-density, medium-density, and edge-touching conditions:

- **Segmentation Agreement**: Cellpose-SAM (`cpsam_v2`) with post-processing (seeded watershed splitting and size gating) achieves an $R^2 > 0.94$ against manual nuclear counts.
- **Debris & Overlap Filtering**: Raw Cellpose-SAM output contains debris and merged nuclear clusters. The two-tier morphological filter reduces oversegmentation and undersegmentation artifacts by $>60\%$.
- **Border Clearance**: Excluding nuclei intersecting or within $5.85\,\mu\mathrm{m}$ of the field perimeter eliminates edge-truncation bias.

---

## 3. Backend Segmentation Comparison

Three segmentation engines were evaluated across the spheroid dataset:

1. **Cellpose-SAM (`cpsam`)**: Primary deep-learning engine leveraging Segment Anything (SAM) with Cellpose flow dynamics. Best boundary adherence for irregular spheroid morphologies.
2. **Cellpose Cyto3 (`cyto3`)**: Fallback deep learning model for cellular cytoplasm segmentation.
3. **Classical Multi-Scale Morphological Filter**: Deterministic Otsu/adaptive thresholding with morphological closure. Serves as CPU-fallback when deep learning weights are unavailable.

Cross-backend evaluations show that deep learning backends recover $\sim 15\%$ more high-contrast peripheral spheroids than classical thresholding, while maintaining comparable fold-change kinetics when measured on the common matched cohort.
