"""
Feature extraction: per-object physical morphometrics and per-image spheroid-level density readouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from matrix_mat.core.units import compute_spherical_volume

logger = logging.getLogger("matrix_mat.if_density.features")


def compute_areal_density(n_valid: int, spheroid_area_um2: float) -> float:
    """Computes planar areal density (nuclei per mm^2) of spheroid cross-section in imaged plane."""
    if spheroid_area_um2 is None or np.isnan(spheroid_area_um2) or spheroid_area_um2 <= 0:
        return float("nan")
    spheroid_area_mm2 = spheroid_area_um2 * 1e-6
    return float(n_valid / spheroid_area_mm2)


def compute_volumetric_density_3d(n_nuclei: int, z_stack_or_volume_um3: Any, is_2d_plane: bool = False) -> float:
    """Computes true 3D volumetric density. Refuses single-plane 2D input with ValueError."""
    if is_2d_plane or (isinstance(z_stack_or_volume_um3, np.ndarray) and z_stack_or_volume_um3.ndim == 2):
        raise ValueError("Single-plane 2D image cannot support true 3D volumetric density calculation; requires 3D z-stack")
    if isinstance(z_stack_or_volume_um3, (int, float)):
        vol = float(z_stack_or_volume_um3)
    elif isinstance(z_stack_or_volume_um3, np.ndarray) and z_stack_or_volume_um3.ndim == 3:
        vol = float(np.sum(z_stack_or_volume_um3 > 0))
    else:
        raise ValueError("Invalid volume representation for 3D volumetric density")
    if vol <= 0 or np.isnan(vol):
        return float("nan")
    return float(n_nuclei / vol) * 1e9


# Below this many actin-mask pixels the actin threshold is treated as failed and the nuclei convex hull
# is used instead (flagged per field).
MIN_ACTIN_MASK_PX = 100


@dataclass
class SpheroidExtent:
    """
    Spheroid cross-section estimate for one field.
    Unpacks like the legacy 4-tuple (volume_um3, centroid_y_um, centroid_x_um, max_radius_um).
    """
    volume_um3: float
    centroid_y_um: float
    centroid_x_um: float
    max_radius_um: float
    area_um2: float
    method_used: str  # 'actin_threshold' | 'nuclei_convex_hull' | 'nuclei_convex_hull_fallback' | 'none'
    hull_fallback: bool
    mask: Optional[np.ndarray] = None

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.volume_um3, self.centroid_y_um, self.centroid_x_um, self.max_radius_um)

    def __iter__(self) -> Iterator[float]:
        return iter(self.as_tuple())

    def __getitem__(self, i: int) -> float:
        return self.as_tuple()[i]

    def __len__(self) -> int:
        return 4


def prepare_actin_for_extent(raw_actin: np.ndarray) -> np.ndarray:
    """
    Fixed, arm-independent actin preparation for the spheroid extent: the RAW actin channel scaled to
    uint8 by bit depth (8-bit data unchanged). No background subtraction or percentile normalisation,
    so every pre/post-processing arm of a field gets the same density denominator.
    """
    from matrix_mat.if_density.preprocess import to_uint8
    return to_uint8(raw_actin)


def compute_spheroid_extent_details(
    actin_img: np.ndarray,
    nuclear_masks: np.ndarray,
    pixel_size_um: float,
    method: str = "actin_threshold",
) -> SpheroidExtent:
    """
    Estimate the spheroid cross-section mask, its area, equivalent volume, centroid and radius.
    method='actin_threshold': Otsu on the blurred actin image, holes filled, largest component; when that
    mask has < MIN_ACTIN_MASK_PX pixels the convex hull of the nuclei masks is used and hull_fallback=True.
    method='nuclei_convex_hull': convex hull of the nuclei masks.
    """
    if method not in ("actin_threshold", "nuclei_convex_hull"):
        raise ValueError(f"Unknown spheroid_extent_method '{method}'")
    actin_u8 = actin_img if actin_img.dtype == np.uint8 else prepare_actin_for_extent(actin_img)
    h, w = actin_u8.shape[:2]
    spheroid_mask = np.zeros((h, w), dtype=bool)
    method_used = "none"
    hull_fallback = False

    if method == "actin_threshold":
        # Otsu or background cutoff
        smooth = cv2.GaussianBlur(actin_u8, (15, 15), 0)
        thresh_val, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Binary closing and hole filling
        filled = ndimage.binary_fill_holes(binary > 0)
        # Select largest component
        labeled, num = measure.label(filled, return_num=True)
        if num > 0:
            props = measure.regionprops(labeled)
            largest = max(props, key=lambda p: p.area)
            spheroid_mask = (labeled == largest.label)
            method_used = "actin_threshold"

    use_hull = method == "nuclei_convex_hull" or np.sum(spheroid_mask) < MIN_ACTIN_MASK_PX
    if use_hull:
        hull_fallback = method == "actin_threshold"
        if nuclear_masks.max() > 0:
            nuc_binary = (nuclear_masks > 0)
            try:
                from skimage.morphology import convex_hull_image
                spheroid_mask = convex_hull_image(nuc_binary)
            except Exception:
                spheroid_mask = nuc_binary
            method_used = "nuclei_convex_hull_fallback" if hull_fallback else "nuclei_convex_hull"
        elif hull_fallback:
            # Nothing to fall back to; keep whatever (tiny) actin mask there was.
            hull_fallback = False

    if np.sum(spheroid_mask) == 0:
        # Spheroid mask empty: return NaN rather than 1 um^3 fallback
        return SpheroidExtent(
            float("nan"), (h / 2.0) * pixel_size_um, (w / 2.0) * pixel_size_um, float("nan"),
            area_um2=float("nan"), method_used="none", hull_fallback=False, mask=spheroid_mask,
        )

    area_px = float(np.sum(spheroid_mask))
    area_um2 = area_px * (pixel_size_um ** 2)
    eq_d_um = 2.0 * np.sqrt(area_um2 / np.pi)
    spheroid_vol_um3 = compute_spherical_volume(eq_d_um)

    M = measure.moments(spheroid_mask.astype(np.uint8))
    cy_px = M[1, 0] / max(1.0, M[0, 0])
    cx_px = M[0, 1] / max(1.0, M[0, 0])
    cy_um = cy_px * pixel_size_um
    cx_um = cx_px * pixel_size_um
    max_radius_um = (eq_d_um / 2.0)

    return SpheroidExtent(
        spheroid_vol_um3, cy_um, cx_um, max_radius_um,
        area_um2=area_um2, method_used=method_used, hull_fallback=hull_fallback, mask=spheroid_mask,
    )


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
    return compute_spheroid_extent_details(actin_img, nuclear_masks, pixel_size_um, method=method).as_tuple()


def density_area_um2(extent: SpheroidExtent, pixel_size_um: float, border_clearance_px: int = 0) -> float:
    """
    Spheroid area used as the density denominator: the extent mask minus the image-border strip whose
    nuclei are excluded from the count (postprocess.border_strip_mask), so numerator and denominator
    cover the same region.
    """
    if extent.mask is None or np.isnan(extent.area_um2):
        return extent.area_um2
    if border_clearance_px <= 0:
        return extent.area_um2
    from matrix_mat.if_density.postprocess import border_strip_mask
    strip = border_strip_mask(extent.mask.shape, border_clearance_px)
    area_px = float(np.count_nonzero(extent.mask & ~strip))
    return area_px * (pixel_size_um ** 2) if area_px > 0 else float("nan")


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
    spheroid_extent: Optional[Union[SpheroidExtent, Tuple[float, float, float, float]]] = None,
    border_clearance_px: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract per-object morphology and per-image density and aggregate statistics.
    Objects within clearance_um get excluded_border=True and are excluded from density calculations.
    The density denominator excludes the same image-border strip (border_clearance_px, or clearance_um
    when labels are derived here). Returns (objects_rows, per_image_row).
    """
    if pixel_size_um is None or pixel_size_um <= 0:
        raise ValueError(f"Invalid physical pixel calibration: {pixel_size_um} um/pixel for {image_id}")

    h, w = nuclear_img.shape[:2]
    n_nuclei = int(dedup_masks.max())

    extent: Optional[SpheroidExtent]
    if isinstance(spheroid_extent, SpheroidExtent):
        extent = spheroid_extent
    elif spheroid_extent is not None:
        extent = None
    else:
        extent_method = feature_config.get("spheroid_extent_method", "actin_threshold")
        extent = compute_spheroid_extent_details(
            prepare_actin_for_extent(actin_img), dedup_masks, pixel_size_um, method=extent_method
        )
    if extent is not None:
        sph_vol_um3, sph_cy_um, sph_cx_um, max_r_um = extent.as_tuple()
    else:
        sph_vol_um3, sph_cy_um, sph_cx_um, max_r_um = spheroid_extent  # type: ignore[misc]

    # Determine border labels if not provided
    if border_excluded_labels is None and clearance_um > 0:
        clearance_px = max(1, int(round(clearance_um / pixel_size_um)))
        border_clearance_px = border_clearance_px or clearance_px
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
            vol_um3 = compute_spherical_volume(eq_d_um)
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
            integrated_actin = float(p.image_intensity[p.image].sum())

            # Normalized radial distance from spheroid centroid
            dist_to_center_um = np.sqrt((cy_um - sph_cy_um) ** 2 + (cx_um - sph_cx_um) ** 2)
            radius_ok = max_r_um is not None and np.isfinite(max_r_um) and max_r_um > 0
            norm_radial_dist = dist_to_center_um / max(1.0, max_r_um) if radius_ok else float("nan")

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

    if max_r_um is not None and not np.isnan(max_r_um) and max_r_um > 0:
        spheroid_area_um2 = np.pi * (max_r_um ** 2)
    else:
        spheroid_area_um2 = float("nan")
    if extent is not None:
        denom_area_um2 = density_area_um2(extent, pixel_size_um, border_clearance_px)
    else:
        denom_area_um2 = spheroid_area_um2

    nuclei_per_mm2 = compute_areal_density(n_valid, denom_area_um2)
    median_nuc_vol = float(np.median(nuclear_volumes)) if nuclear_volumes else 0.0
    median_actin = float(np.median(actin_img))

    per_image_row = {
        "image_id": image_id,
        "material": material,
        "replicate": replicate,
        "n_nuclei": n_valid,
        "n_nuclei_total": n_nuclei,
        "n_border_excluded": n_nuclei - n_valid,
        "spheroid_area_um2": round(spheroid_area_um2, 1) if not np.isnan(spheroid_area_um2) else np.nan,
        "spheroid_volume_um3": round(sph_vol_um3, 1) if (sph_vol_um3 is not None and not np.isnan(sph_vol_um3)) else np.nan,
        "nuclei_per_mm2": round(nuclei_per_mm2, 2) if not np.isnan(nuclei_per_mm2) else np.nan,
        "median_nuclear_volume_um3": round(median_nuc_vol, 2),
        "median_actin_intensity": round(median_actin, 2),
        "flag_summary": qc_flags_list if qc_flags_list else "clean",
        # Appended columns (existing names unchanged): the denominator actually used, how the extent was
        # obtained, and whether the actin threshold failed and the nuclei convex hull was used instead.
        "density_area_um2": round(denom_area_um2, 1) if not np.isnan(denom_area_um2) else np.nan,
        "spheroid_extent_method": extent.method_used if extent is not None else "provided",
        "spheroid_extent_hull_fallback": bool(extent.hull_fallback) if extent is not None else False,
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
