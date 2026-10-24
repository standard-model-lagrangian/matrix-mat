"""
Physical units conversion and geometric volume calculations for matrix-mat.
Single canonical source of the (np.pi / 6.0) volume calculation.
"""

from __future__ import annotations

import math
from typing import Union
import numpy as np


def compute_spherical_volume(diameter_um: float) -> float:
    """
    Compute spherical volume from equivalent diameter:
    V_sphere = (pi / 6) * d^3
    """
    d = max(0.0, float(diameter_um))
    return float((np.pi / 6.0) * (d ** 3))


def compute_ellipsoidal_volume(major_axis_um: float, minor_axis_um: float) -> float:
    """
    Compute ellipsoidal volume from major and minor axes:
    V_ellipsoid = (pi / 6) * major * minor^2
    """
    a = max(0.0, float(major_axis_um))
    b = max(0.0, float(minor_axis_um))
    return float((np.pi / 6.0) * a * (b ** 2))


def area_to_sphere_volume(area_um2: float) -> float:
    """
    Convert 2D cross-sectional area in um^2 to equivalent 3D spherical volume in um^3:
    d = 2 * sqrt(area / pi)
    V = (pi / 6) * d^3
    """
    a = max(0.0, float(area_um2))
    d = 2.0 * math.sqrt(a / np.pi)
    return compute_spherical_volume(d)


def sphere_volume_to_diameter(volume_um3: float) -> float:
    """
    Convert 3D spherical volume in um^3 to equivalent diameter in um:
    d = (6 * V / pi) ^ (1/3)
    """
    v = max(0.0, float(volume_um3))
    return float((6.0 * v / np.pi) ** (1.0 / 3.0))


def equivalent_diameter(area_px: float, pixel_size_um: float = 1.0) -> float:
    """
    Compute equivalent diameter in physical micrometers from 2D pixel area and pixel size:
    d = 2 * sqrt(area_px / pi) * pixel_size_um
    """
    a = max(0.0, float(area_px))
    scale = max(1e-9, float(pixel_size_um))
    return float(2.0 * math.sqrt(a / np.pi) * scale)


def px_to_um(px: float, pixel_size_um: float) -> float:
    """Convert distance from pixels to micrometers."""
    return float(px) * float(pixel_size_um)


def um_to_px(um: float, pixel_size_um: float) -> float:
    """Convert distance from micrometers to pixels."""
    scale = max(1e-9, float(pixel_size_um))
    return float(um) / scale


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
    circ = (4.0 * np.pi * a) / (p ** 2)
    return float(min(1.0, max(0.0, circ)))
