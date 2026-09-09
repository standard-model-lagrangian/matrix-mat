"""
Feature extraction: per-object physical morphometrics and per-image spheroid-level density readouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

logger = logging.getLogger("spheroid_if_sweep.features")


def compute_spheroid_extent(
    actin_img: np.ndarray,
    nuclear_masks: np.ndarray,
    pixel_size_um: float,
    method: str = "actin_threshold",
) -> Tuple[float, float, float, float]:
    """
    Estimate spheroid volume, centroid (y, x in um), and max radius (um).
    Returns (spheroid_volume_um3, centroid_y_um, centroid_x_um, max_radius_um).
    """
    h, w = actin_img.shape[:2]
    spheroid_mask = np.zeros((h, w), dtype=bool)

    if method == "actin_threshold":
        # Otsu or background cutoff
        smooth = cv2.GaussianBlur(actin_img, (15, 15), 0)
        thresh_val, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Binary closing and hole filling
        filled = ndimage.binary_fill_holes(binary > 0)
        # Select largest component
        labeled, num = measure.label(filled, return_num=True)
        if num > 0:
            props = measure.regionprops(labeled)
            largest = max(props, key=lambda p: p.area)
            spheroid_mask = (labeled == largest.label)

    # Fallback to convex hull of nuclear masks if actin mask is empty or requested
    if np.sum(spheroid_mask) < 100 and nuclear_masks.max() > 0:
        nuc_binary = (nuclear_masks > 0)
        try:
            from skimage.morphology import convex_hull_image
            spheroid_mask = convex_hull_image(nuc_binary)
        except Exception:
            spheroid_mask = nuc_binary

    if np.sum(spheroid_mask) == 0:
        # Ultimate fallback: image center and default volume
        return 1.0, (h / 2.0) * pixel_size_um, (w / 2.0) * pixel_size_um, 50.0

    # Calculate properties
    area_px = float(np.sum(spheroid_mask))
    area_um2 = area_px * (pixel_size_um ** 2)
    eq_d_um = 2.0 * np.sqrt(area_um2 / np.pi)
    spheroid_vol_um3 = (np.pi / 6.0) * (eq_d_um ** 3)

    M = measure.moments(spheroid_mask.astype(np.uint8))
    cy_px = M[1, 0] / max(1.0, M[0, 0])
    cx_px = M[0, 1] / max(1.0, M[0, 0])
    cy_um = cy_px * pixel_size_um
    cx_um = cx_px * pixel_size_um
    max_radius_um = (eq_d_um / 2.0)

    return spheroid_vol_um3, cy_um, cx_um, max_radius_um


def extract_objects_and_image_features(
    image_id: str,
    material: str,
    replicate: str,
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    dedup_masks: np.ndarray,
    pixel_size_um: float,
    qc_flags_list: str,
    feature_config: Dict[str, Any],
    border_excluded_labels: Optional[Set[int]] = None,
    clearance_um: float = 0.0,
    spheroid_extent: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract per-object morphology and per-image density and aggregate statistics.
    Objects within clearance_um get excluded_border=True and are excluded from density calculations.
    Returns (objects_rows, per_image_row).
    """
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(f"Invalid physical pixel calibration: {pixel_size_um} um/pixel for {image_id}")

    h, w = nuclear_img.shape[:2]
    n_nuclei = int(dedup_masks.max())

    if spheroid_extent is not None:
        sph_vol_um3, sph_cy_um, sph_cx_um, max_r_um = spheroid_extent
    else:
        extent_method = feature_config.get("spheroid_extent_method", "actin_threshold")
        sph_vol_um3, sph_cy_um, sph_cx_um, max_r_um = compute_spheroid_extent(
            actin_img, dedup_masks, pixel_size_um, method=extent_method
        )

    # Determine border labels if not provided
    if border_excluded_labels is None and clearance_um > 0:
        clearance_px = max(1, int(round(clearance_um / pixel_size_um)))
        border_excluded_labels = set()
        for p in measure.regionprops(dedup_masks):
            minr, minc, maxr, maxc = p.bbox
            if minr <= clearance_px or minc <= clearance_px or maxr >= (h - clearance_px) or maxc >= (w - clearance_px):
                border_excluded_labels.add(p.label)
    elif border_excluded_labels is None:
        border_excluded_labels = set()

    objects_rows: List[Dict[str, Any]] = []
    nuclear_volumes: List[float] = []

    if n_nuclei > 0:
        props = measure.regionprops(dedup_masks, intensity_image=actin_img)
        margin = int(feature_config.get("border_margin_px", 5))

        for p in props:
            obj_id = f"{image_id}_obj_{p.label:04d}"
            area_px = p.area
            area_um2 = area_px * (pixel_size_um ** 2)
            eq_d_um = 2.0 * np.sqrt(area_um2 / np.pi)
            vol_um3 = (np.pi / 6.0) * (eq_d_um ** 3)
            nuclear_volumes.append(vol_um3)

            # Sphericity (2D circularity 4*pi*A / P^2)
            perimeter_px = max(1.0, p.perimeter)
            sphericity = min(1.0, (4.0 * np.pi * area_px) / (perimeter_px ** 2))

            # Solidity
            solidity = p.solidity

            # Centroids (z, y, x in um)
            cy_px, cx_px = p.centroid
            cy_um = cy_px * pixel_size_um
            cx_um = cx_px * pixel_size_um
            cz_um = 0.0

            # Actin intensity features under nuclear mask
            mean_actin = float(p.intensity_mean if hasattr(p, "intensity_mean") else getattr(p, "mean_intensity", 0.0))
            integrated_actin = float(np.sum(actin_img[dedup_masks == p.label]))

            # Normalized radial distance from spheroid centroid
            dist_to_center_um = np.sqrt((cy_um - sph_cy_um) ** 2 + (cx_um - sph_cx_um) ** 2)
            norm_radial_dist = dist_to_center_um / max(1.0, max_r_um)

            # Edge touching
            minr, minc, maxr, maxc = p.bbox
            edge_touching = bool(
                minr <= margin or minc <= margin or maxr >= (h - margin) or maxc >= (w - margin)
            )

            # Border exclusion
            excluded_border = bool(p.label in border_excluded_labels)

            # Object-level flags
            obj_flags = []
            if edge_touching:
                obj_flags.append("edge_touching")
            if excluded_border:
                obj_flags.append("excluded_border")
            if vol_um3 < feature_config.get("min_nuclear_vol_um3", 150.0):
                obj_flags.append("debris")
            elif vol_um3 > feature_config.get("max_nuclear_vol_um3", 4000.0):
                obj_flags.append("merged")

            objects_rows.append({
                "image_id": image_id,
                "material": material,
                "replicate": replicate,
                "object_id": obj_id,
                "volume_um3": round(vol_um3, 2),
                "equivalent_diameter_um": round(eq_d_um, 2),
                "sphericity": round(sphericity, 3),
                "solidity": round(solidity, 3),
                "centroid_z_um": cz_um,
                "centroid_y_um": round(cy_um, 2),
                "centroid_x_um": round(cx_um, 2),
                "mean_actin_intensity": round(mean_actin, 2),
                "integrated_actin_intensity": round(integrated_actin, 1),
                "normalized_radial_distance": round(norm_radial_dist, 3),
                "edge_touching": edge_touching,
                "excluded_border": excluded_border,
                "flags": ";".join(obj_flags) if obj_flags else "none",
            })

    # Per-image calculations excluding border-truncated nuclei from density
    n_valid = sum(1 for obj in objects_rows if not obj.get("excluded_border", False))
    density_per_mm3 = (float(n_valid) / max(1e-6, sph_vol_um3)) * 1e9
    median_nuc_vol = float(np.median(nuclear_volumes)) if nuclear_volumes else 0.0
    median_actin = float(np.median(actin_img))

    per_image_row = {
        "image_id": image_id,
        "material": material,
        "replicate": replicate,
        "n_nuclei": n_valid,
        "n_nuclei_total": n_nuclei,
        "n_border_excluded": n_nuclei - n_valid,
        "spheroid_volume_um3": round(sph_vol_um3, 1),
        "nuclei_density_per_mm3": round(density_per_mm3, 2),
        "median_nuclear_volume_um3": round(median_nuc_vol, 2),
        "median_actin_intensity": round(median_actin, 2),
        "flag_summary": qc_flags_list if qc_flags_list else "clean",
    }

    return objects_rows, per_image_row


def save_features_csvs(
    all_objects: List[Dict[str, Any]],
    all_per_image: List[Dict[str, Any]],
    features_dir: Path,
) -> Tuple[Path, Path]:
    """Save features/objects.csv and features/per_image.csv."""
    features_dir.mkdir(parents=True, exist_ok=True)
    obj_df = pd.DataFrame(all_objects)
    obj_path = features_dir / "objects.csv"
    obj_df.to_csv(obj_path, index=False)

    img_df = pd.DataFrame(all_per_image)
    img_path = features_dir / "per_image.csv"
    img_df.to_csv(img_path, index=False)

    logger.info(f"Saved {len(obj_df)} objects to {obj_path} and {len(img_df)} images to {img_path}")
    return obj_path, img_path
