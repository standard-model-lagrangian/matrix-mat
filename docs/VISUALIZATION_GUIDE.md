# Scientific Visualization & Publication Figures Guide

This guide details all publication-ready visual artifacts, statistical dashboards, and supervisory overlays generated across the **Brightfield Spheroid Pipeline** and the **Immunofluorescence (IF) Pipeline**.

---

## 1. Brightfield Spheroid Presentation Figures (Figures 1–6)

All figures are automatically generated in `output/figures/` in both **high-resolution 300 DPI PNG** (for presentations and web) and **vector PDF** (for vector editing in Illustrator / publication submission).

### Figure 1: Primary Biomass & Proliferation Dashboard
`presentation_fig1_primary_biomass_dashboard.png` / `.pdf`

```
┌─────────────────────────────────┬─────────────────────────────────┬─────────────────────────────────┐
│ Panel A: Pooled Well Fold Change│ Panel B: Population Shifts      │ Panel C: Paired Spheroid Growth │
│ Boxplot of ΣV7 / ΣV0 per well   │ t0 vs t7 volume distributions   │ Fold change vs baseline volume  │
│ Wilcoxon test vs H0: FC = 1.0   │ Spheroid enlargement            │ Hungarian matched pairs         │
└─────────────────────────────────┴─────────────────────────────────┴─────────────────────────────────┘
```
- **Panel A (Total Pooled Biomass Fold Change)**: Evaluates overall biomass growth at the well-replicate level ($N=6\text{--}8$ independent hydrogels per condition). A diamond marks the condition mean, and a horizontal dashed red line marks the no-growth boundary ($\text{FC}=1.0$). Statistically honest one-tailed Wilcoxon signed-rank tests assess significant growth ($^*p<0.05$, $^{**}p<0.01$).
- **Panel B (Population Volume Shifts)**: Side-by-side boxplots showing all individual `PASS` spheroids at Day 0 ($t_0$, blue) vs Day 7 ($t_7$, amber) on a logarithmic scale.
- **Panel C (Growth vs Initial Size)**: Scatter plot of matched spheroid pairs showing whether smaller or larger spheroids grow faster in each gel formulation.

---

### Figure 2: Hypertrophy vs Hyperplasia Decomposition
`presentation_fig2_hypertrophy_vs_hyperplasia.png` / `.pdf`

Disentangles the biological mode of growth across hydrogels by separating:
- **X-axis (Hypertrophy / Spheroid Enlargement)**: Mean spheroid volume fold change ($\bar{V}_{t7} / \bar{V}_{t0}$).
- **Y-axis (Hyperplasia / Colony Yield)**: Spheroid count density fold change ($\text{Count}_{t7} / \text{Count}_{t0}$).

**Biological Quadrant Interpretation:**
1. **Upper-Right (Dual Expansion)**: Spheroids both grow larger and proliferate into more colonies (e.g. Matrigel control, S34D30).
2. **Lower-Right (Hypertrophy Only)**: Spheroids swell in volume, but count does not increase (fused colonies or no new budding).
3. **Lower-Left (Growth Arrest / Shrinkage)**: Colony loss and volume regression.

---

### Figure 3: Per-Spheroid Volume Distributions
`presentation_fig3_per_spheroid_volume_distributions.png` / `.pdf`

- Logarithmic distribution of all detected, valid spheroid volumes across all conditions.
- Demonstrates condition-wide spread, dynamic range, and skewness across hundreds of quantified spheroids.

---

### Figure 4: Pipeline Quality Control Yield & Forensic Audit
`presentation_fig4_qc_pipeline_yield.png` / `.pdf`

```
┌──────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
│ Panel A: QC Flag Yield Breakdown                 │ Panel B: Forensic Exclusion Breakdown            │
│ Stacked horizontal bar chart:                    │ Bar chart of specific rejection reasons:         │
│ Green = PASS, Amber = REVIEW, Red = FAIL         │ Border clipped, Low contrast, Undersized, Dups   │
└──────────────────────────────────────────────────┴──────────────────────────────────────────────────┘
```
- **Panel A (Yield Breakdown)**: Shows the percentage of candidate instances classified as `PASS` (valid spheroids), `REVIEW` (irregular morphology), and `FAIL` (debris/border clipped) per condition.
- **Panel B (Forensic Root-Cause)**: Quantifies the exact reasons for exclusion across all rejected objects (e.g. `touches_image_border`, `insufficient_dark_contrast`, `debris_undersized`, `duplicate_focal_plane`).

---

### Figure 5: Hungarian-Tracked Paired Trajectories
`presentation_fig5_paired_trajectories.png` / `.pdf`

- **Panel A (Log-Log Scatter with Reference Lines)**: Baseline volume ($V_0$) vs endpoint volume ($V_7$) for all tracked pairs. Includes reference dashed lines for:
  - **1:1** (Gray dashed line: No net volume change)
  - **2:1** (Green dotted line: Volume doubling / 1 population doubling)
  - **4:1** (Orange dotted line: Volume quadrupling / 2 population doublings)
- **Panel B (Paired Growth Slopegraphs)**: Direct paired slope lines connecting $t_0 \to t_7$ for every tracked spheroid, with thick condition median lines highlighting condition trajectories.

---

### Figure 6: Raw Volume Numbers Per Well (Anti-Collision Staggered Labels)
`presentation_fig6_raw_well_volumes.png` / `.pdf`

- Displays the absolute total volume per well in **nanoliters** ($10^6\,\mu\mathrm{m}^3 = 1\,\mathrm{nL}$) for every single replicate well across all hydrogel formulations.
- Green lines represent growing wells ($\ge 1.2\times$), red lines represent shrinking wells ($\le 0.9\times$), and gray lines represent neutral wells.
- Incorporates an **anti-collision logarithmic text staggering algorithm**: When multiple wells share similar volumes, label text is dynamically offset vertically with connecting dotted leader lines to prevent text overlapping.

---

## 2. Immunofluorescence (IF) Biological Comparison Figures

Generated in `runs/<run_slug>/figures/` or `runs/comparison/biological_figures/`:

### Figure 1: Cross-Material Nuclei Density Distributions
`fig_if_nuclei_density_by_material.png` / `.pdf`
- **Panel A (Linear Scale)**: Boxplot of nuclear packing density ($10^3\,\text{nuclei/mm}^3$) across hydrogel materials with overlay jittered points for each FOV. Condition means are indicated by yellow diamonds. Two-sided Mann-Whitney U tests annotate significance against the `Mat` control ($^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$).
- **Panel B (Log Scale)**: Captures the full dynamic range of nuclear density variations across soft vs stiff hydrogels.

### Figure 2: Clean vs All-Cohort Biological Robustness
`fig_if_clean_vs_all_density.png` / `.pdf`
- Paired comparison bars showing mean $\pm$ SD nuclei density for **All Cohort (Unfiltered)** vs **Clean Only (QC Outliers Excluded)**.
- Demonstrates that removing blurry or artifact-heavy fields does not alter the biological rank order or relative stiffness response across formulations.

### Figure 3: Single-Nucleus Volume Distributions Across Hydrogels
`fig_if_nuclear_volume_distributions.png` / `.pdf`
- Boxplot of individual nuclear volumes ($V_{\text{nuc}}$, $\mu\mathrm{m}^3$) across hydrogel formulations for all non-border nuclei.
- Annotates median volume values and sample sizes ($n$) to evaluate whether dense hydrogel networks exert physical compressive stress on cell nuclei.

---

## 3. Supervisory Overlays & Inspection Conventions

### Brightfield Supervision Overlays (`output/overlays/<image_id>_overlay.png`)
Each raw image produces an annotated RGB supervision image:
- **Green Contours**: `PASS` objects. Valid spheroids meeting all size, border margin, and contrast criteria.
- **Amber / Orange Contours**: `REVIEW` objects. Authentically segmented spheroids with low circularity ($< 0.65$) or extreme fold changes.
- **Red Contours**: `FAIL` objects. Rejected debris, border-clipped fragments, or non-contrast gel artifacts.
- **Centroid Markers**: Colored crosshair at the geometric center of each object.
- **Label Text**: High-contrast text box displaying Object ID, equivalent diameter $d$ in $\mu\mathrm{m}$, circularity, and QC flag.

### Contact Sheets (`output/contact_sheets/`)
Montage grids grouping all FOV overlays by condition and timepoint (e.g. `Mat_t0_contact_sheet.png`, `S34D30_t7_contact_sheet.png`), allowing instant high-throughput visual quality control across dozens of wells simultaneously.
