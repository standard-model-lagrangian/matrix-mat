"""
Measurement and Morphometrics Module for Spheroid Volume Pipeline v2 (spheroid_pipeline_v2).
Provides:
  - Quantitative 2D and 3D calibrated morphological metrics:
    * Area (px, um^2), Perimeter (px)
    * Equivalent diameter d = 2 * sqrt(A / pi) (px, um)
    * Major / minor axis lengths (um), Aspect ratio
    * Spherical volume V_sphere = (pi / 6) * d^3
    * Ellipsoidal volume V_ellipsoid = (pi / 6) * a * b^2
    * Volume discrepancy ratio |V_sphere - V_ellipsoid| / V_sphere
    * Circularity (4 * pi * A / P^2), Solidity (A / ConvexHullArea), Eccentricity
  - Contrast readouts: mean interior intensity, mean background ring intensity, contrast ratio
  - QC Flagging Protocol: PASS, REVIEW, FAIL with semicolon-delimited reasons
  - SpheroidObjectRecord dataclass matching the full 32-column objects.csv schema
  - CSV export utility: save_objects_csv
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from skimage import measure

from spheroid_pipeline_v2.config import QCConfig

logger = logging.getLogger("spheroid_pipeline_v2.measure")

# Exact 32-column header sequence mandated by PROJECT.md / spec report
OBJECTS_CSV_COLUMNS: List[str] = [
    "object_id",
    "image_id",
    "timepoint",
    "condition",
    "replicate",
    "fov",
    "pair_key",
    "label_idx",
    "pixel_size_um",
    "area_px",
    "perimeter_px",
    "equivalent_diameter_px",
    "centroid_y",
    "centroid_x",
    "area_um2",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "aspect_ratio",
    "volume_sphere_um3",
    "volume_ellipsoid_um3",
    "volume_discrepancy_ratio",
    "circularity",
    "solidity",
    "eccentricity",
    "mean_interior_intensity",
    "mean_background_ring_intensity",
    "contrast_ratio",
    "qc_flag",
    "qc_reasons",
    "pair_id",
    "mask_source",
]


@dataclass
class SpheroidObjectRecord:
    """Dataclass representing a single segmented spheroid object with all 32 schema fields."""
    object_id: str
    image_id: str
    timepoint: str
    condition: str
    replicate: str
    fov: str
    pair_key: str
    label_idx: int
    pixel_size_um: float
    area_px: float
    perimeter_px: float
    equivalent_diameter_px: float
    centroid_y: float
    centroid_x: float
    area_um2: float
    equivalent_diameter_um: float
    major_axis_um: float
    minor_axis_um: float
    aspect_ratio: float
    volume_sphere_um3: float
    volume_ellipsoid_um3: float
    volume_discrepancy_ratio: float
    circularity: float
    solidity: float
    eccentricity: float
    mean_interior_intensity: float
    mean_background_ring_intensity: float
    contrast_ratio: float
    qc_flag: str
    qc_reasons: str
    pair_id: str = "UNPAIRED"
    mask_source: str = "automated"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to ordered dictionary matching OBJECTS_CSV_COLUMNS."""
        raw = asdict(self)
        return {col: raw[col] for col in OBJECTS_CSV_COLUMNS}


def compute_spherical_volume(diameter_um: float) -> float:
    """
    Compute spherical volume from equivalent diameter:
    V_sphere = (pi / 6) * d^3
    """
    d = max(0.0, float(diameter_um))
    return float((math.pi / 6.0) * (d ** 3))


def compute_ellipsoidal_volume(major_axis_um: float, minor_axis_um: float) -> float:
    """
    Compute ellipsoidal (prolate/oblate spheroid) volume:
    V_ellipsoid = (pi / 6) * major * minor^2
    """
    a = max(0.0, float(major_axis_um))
    b = max(0.0, float(minor_axis_um))
    return float((math.pi / 6.0) * a * (b ** 2))


def compute_volume_discrepancy_ratio(volume_sphere: float, volume_ellipsoid: float) -> float:
    """
    Compute volume discrepancy ratio to audit spherical geometry assumption:
    Discrepancy = |V_sphere - V_ellipsoid| / V_sphere
    """
    v_sph = max(0.0, float(volume_sphere))
    v_ell = max(0.0, float(volume_ellipsoid))
    denom = max(v_sph, 1e-6)
    return float(abs(v_sph - v_ell) / denom)


def compute_circularity(area_px: float, perimeter_px: float) -> float:
    """
    Compute 2D circularity (isoperimetric quotient):
    Circularity = 4 * pi * Area / (Perimeter^2)
    Normalized to range [0.0, 1.0].
    """
    a = max(0.0, float(area_px))
    p = max(1e-4, float(perimeter_px))
    circ = (4.0 * math.pi * a) / (p ** 2)
    return float(min(1.0, max(0.0, circ)))


def evaluate_morphological_qc(
    equivalent_diameter_um: float,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    mean_interior_intensity: float,
    mean_ring_intensity: float,
    contrast_ratio: float,
    circularity: float,
    solidity: float,
    eccentricity: float,
    config: Optional[QCConfig] = None,
) -> Tuple[str, List[str]]:
    """
    Evaluate morphological, size, border, and contrast gates.
    Assigns:
      - 'FAIL': rejected from clean label map (debris, artifact, border, low contrast, giant mask)
      - 'REVIEW': retained in clean label map, flagged for manual check (low circularity/solidity, high eccentricity)
      - 'PASS': passes all gates
    Returns (qc_flag, list_of_reasons).
    """
    cfg = config if config is not None else QCConfig()
    img_h, img_w = image_shape[:2]
    min_r, min_c, max_r, max_c = bbox
    reasons: List[str] = []
    flag = "PASS"

    # --- FAIL CHECKS ---
    # 1. Physical size gate
    if equivalent_diameter_um < cfg.min_d_um:
        flag = "FAIL"
        reasons.append(f"debris_undersized (d={equivalent_diameter_um:.1f}um < {cfg.min_d_um:.1f}um)")
    elif equivalent_diameter_um > cfg.max_d_um:
        flag = "FAIL"
        reasons.append(f"oversized_artifact (d={equivalent_diameter_um:.1f}um > {cfg.max_d_um:.1f}um)")

    # 2. Border margin gate (touches or within border_margin_px of FOV boundary)
    margin = max(0, int(cfg.border_margin_px))
    if min_r < margin or min_c < margin or max_r > (img_h - margin) or max_c > (img_w - margin):
        flag = "FAIL"
        reasons.append(f"touches_image_border (margin={margin}px, bbox=[{min_r},{min_c},{max_r},{max_c}])")

    # 3. Contrast gate (must be at least N% darker than surrounding ring)
    # mean_int <= (1.0 - N) * mean_ring + 1e-7  <=>  contrast_ratio >= N - 1e-7
    if contrast_ratio < (cfg.interior_contrast_pct - 1e-7):
        flag = "FAIL"
        reasons.append(
            f"insufficient_dark_contrast (contrast={contrast_ratio:.3f} < {cfg.interior_contrast_pct:.3f}, "
            f"I_int={mean_interior_intensity:.3f}, I_ring={mean_ring_intensity:.3f})"
        )

    # --- REVIEW CHECKS (Only if not already FAIL) ---
    if flag != "FAIL":
        if circularity < cfg.min_circularity:
            flag = "REVIEW"
            reasons.append(f"low_circularity ({circularity:.2f} < {cfg.min_circularity:.2f})")

        if solidity < cfg.min_solidity:
            flag = "REVIEW"
            reasons.append(f"low_solidity ({solidity:.2f} < {cfg.min_solidity:.2f})")

        if eccentricity > cfg.max_eccentricity:
            flag = "REVIEW"
            reasons.append(f"high_eccentricity ({eccentricity:.2f} > {cfg.max_eccentricity:.2f})")

    return flag, reasons


def measure_single_object(
    prop: Any,
    image_shape: Tuple[int, int],
    pixel_size_um: float,
    mean_interior_intensity: float,
    mean_ring_intensity: float,
    contrast_ratio: float,
    image_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[QCConfig] = None,
    mask_source: str = "automated",
    pair_id: str = "UNPAIRED",
) -> SpheroidObjectRecord:
    """
    Measure full 32-column geometric, volumetric, and QC attributes for a single skimage RegionProperties object.
    """
    cfg = config if config is not None else QCConfig()
    meta = image_metadata or {}

    scale = max(1e-6, float(pixel_size_um))
    area_px = float(prop.area)
    perimeter_px = float(prop.perimeter) if prop.perimeter > 0 else 1.0
    eq_d_px = float(2.0 * math.sqrt(area_px / math.pi)) if area_px > 0 else 0.0

    cy, cx = float(prop.centroid[0]), float(prop.centroid[1])
    min_r, min_c, max_r, max_c = prop.bbox

    # Calibrated physical units
    area_um2 = float(area_px * (scale ** 2))
    eq_d_um = float(eq_d_px * scale)

    # Major and Minor axis lengths (handling skimage property name variants)
    if hasattr(prop, "axis_major_length"):
        major_px = float(prop.axis_major_length)
    elif hasattr(prop, "major_axis_length"):
        major_px = float(prop.major_axis_length)
    else:
        major_px = eq_d_px

    if hasattr(prop, "axis_minor_length"):
        minor_px = float(prop.axis_minor_length)
    elif hasattr(prop, "minor_axis_length"):
        minor_px = float(prop.minor_axis_length)
    else:
        minor_px = eq_d_px

    # Guard against NaN/zero axis lengths
    if math.isnan(major_px) or major_px <= 0:
        major_px = eq_d_px
    if math.isnan(minor_px) or minor_px <= 0:
        minor_px = eq_d_px

    major_um = float(major_px * scale)
    minor_um = float(minor_px * scale)
    aspect_ratio = float(major_um / max(minor_um, 1e-4))

    # Volumetric metrics
    v_sphere = compute_spherical_volume(eq_d_um)
    v_ellipsoid = compute_ellipsoidal_volume(major_um, minor_um)
    v_disc = compute_volume_discrepancy_ratio(v_sphere, v_ellipsoid)

    # Shape descriptors
    circ = compute_circularity(area_px, perimeter_px)
    sol = float(prop.solidity) if hasattr(prop, "solidity") and not math.isnan(prop.solidity) else 1.0
    sol = float(min(1.0, max(0.0, sol)))

    ecc = float(prop.eccentricity) if hasattr(prop, "eccentricity") and not math.isnan(prop.eccentricity) else 0.0
    ecc = float(min(1.0, max(0.0, ecc)))

    # QC Flagging
    qc_flag, reasons = evaluate_morphological_qc(
        equivalent_diameter_um=eq_d_um,
        bbox=(int(min_r), int(min_c), int(max_r), int(max_c)),
        image_shape=image_shape,
        mean_interior_intensity=mean_interior_intensity,
        mean_ring_intensity=mean_ring_intensity,
        contrast_ratio=contrast_ratio,
        circularity=circ,
        solidity=sol,
        eccentricity=ecc,
        config=cfg,
    )

    qc_reasons_str = "; ".join(reasons) if reasons else "None"

    # Identifiers
    image_id = str(meta.get("image_id", meta.get("file_name", "unknown_image")))
    timepoint = str(meta.get("timepoint", "t0"))
    condition = str(meta.get("condition", "unknown"))
    replicate = str(meta.get("replicate", "rep1"))
    fov = str(meta.get("fov", "0001"))
    pair_key = str(meta.get("pair_key", f"{condition}_{replicate}_{fov}"))
    label_idx = int(prop.label)
    object_id = f"{pair_key}_{timepoint}_obj{label_idx}"

    return SpheroidObjectRecord(
        object_id=object_id,
        image_id=image_id,
        timepoint=timepoint,
        condition=condition,
        replicate=replicate,
        fov=fov,
        pair_key=pair_key,
        label_idx=label_idx,
        pixel_size_um=scale,
        area_px=area_px,
        perimeter_px=perimeter_px,
        equivalent_diameter_px=eq_d_px,
        centroid_y=cy,
        centroid_x=cx,
        area_um2=area_um2,
        equivalent_diameter_um=eq_d_um,
        major_axis_um=major_um,
        minor_axis_um=minor_um,
        aspect_ratio=aspect_ratio,
        volume_sphere_um3=v_sphere,
        volume_ellipsoid_um3=v_ellipsoid,
        volume_discrepancy_ratio=v_disc,
        circularity=circ,
        solidity=sol,
        eccentricity=ecc,
        mean_interior_intensity=float(mean_interior_intensity),
        mean_background_ring_intensity=float(mean_ring_intensity),
        contrast_ratio=float(contrast_ratio),
        qc_flag=qc_flag,
        qc_reasons=qc_reasons_str,
        pair_id=pair_id,
        mask_source=mask_source,
    )


def save_objects_csv(records: List[SpheroidObjectRecord], output_path: Union[str, Path]) -> pd.DataFrame:
    """
    Save list of SpheroidObjectRecord instances to CSV matching the 32-column schema.
    Returns the created pandas DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows, columns=OBJECTS_CSV_COLUMNS)
    df.to_csv(output_path, index=False)
    logger.debug(f"Saved {len(records)} object records to {output_path}")
    return df
