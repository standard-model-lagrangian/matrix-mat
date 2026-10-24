# Immunofluorescence Pipeline Pre/Post-Processing Findings (Narrative)

*Hand-written observations and interpretations for the SKOV3 IF dataset. These are author commentary, not generated results; each should cite the metric it rests on.*

## 1. Clumping and Merged Masks
- **Observation:** In dense spheroid centers, single-plane 2D projection causes nuclear overlap.
- **Intervention:** Seeded watershed splitting using local distance peaks and intensity h-maxima partitions clump candidates larger than 2× the configured reference nuclear volume (841.04 µm³, provisional). Whether this reduces sensitivity to Cellpose parameters is what `robustness_vs_original.csv` measures; it has not been re-measured since the 0.2 changes.

## 2. Autofluorescence and Size Thresholding
- **Observation:** Soft hydrogels (e.g. S50, S43D20) exhibit diffuse matrix haze and small autofluorescent puncta.
- **Intervention:** Rolling-ball background subtraction removes diffuse gradients, while physical size gating removes sub-nuclear specks.

## 3. Perimeter Truncation
- **Observation:** Nuclei cut by the image boundary distort morphology and cross-sectional density estimates.
- **Intervention:** Border margin clearance excludes boundary-touching objects within 5.85 µm from density calculations.

## 4. Optical Quality & Focal Variations
- **Observation (author, not yet confirmed by visual review):** Field `SKOV3 Spheroid D7 Mor_Mat-Gel1-3` has the lowest raw nuclear-channel Laplacian variance of the 27 fields (133 vs 183–481 for the rest), consistent with defocus. It is the only field below the provisional absolute blur threshold of 150.
- **Conclusion:** 2D computational restoration cannot recover missing optical high-frequency details; future acquisition should enforce automated confocal z-stack autofocus.
