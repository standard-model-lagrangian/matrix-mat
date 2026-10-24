# Immunofluorescence Pipeline Pre/Post-Processing Findings (Narrative)

*Archived manual scientific observations and interpretations for the SKOV3 IF dataset.*

## 1. Clumping and Merged Masks
- **Observation:** In dense spheroid centers, single-plane 2D projection causes nuclear overlap.
- **Intervention:** Seeded watershed splitting using local distance peaks and intensity h-maxima partitions large clump candidates (>2× image median volume), reducing parameter sensitivity between sensitive and conservative Cellpose-SAM configurations.

## 2. Autofluorescence and Size Thresholding
- **Observation:** Soft hydrogels (e.g. S50, S43D20) exhibit diffuse matrix haze and small autofluorescent puncta.
- **Intervention:** Rolling-ball background subtraction removes diffuse gradients, while physical size gating removes sub-nuclear specks.

## 3. Perimeter Truncation
- **Observation:** Nuclei cut by the image boundary distort morphology and cross-sectional density estimates.
- **Intervention:** Border margin clearance excludes boundary-touching objects within 5.85 µm from density calculations.

## 4. Optical Quality & Focal Variations
- **Observation:** Field `SKOV3 Spheroid D7 Mor_Mat-Gel1-3` exhibits severe optical defocus at the imaged plane, triggering Laplacian blur flags.
- **Conclusion:** 2D computational restoration cannot recover missing optical high-frequency details; future acquisition should enforce automated confocal z-stack autofocus.
