import math
import re
from pathlib import Path
import numpy as np
import pytest

from spheroid_pipeline_v2.units import (
    area_to_sphere_volume,
    sphere_volume_to_diameter,
    equivalent_diameter,
    px_to_um,
    um_to_px,
    compute_spherical_volume,
    compute_ellipsoidal_volume,
)


def test_area_to_sphere_volume_known_value():
    """Disk of radius 100 um has area pi * 100^2. Sphere volume is (4/3)*pi*r^3 = (pi/6)*d^3."""
    r = 100.0
    d = 200.0
    area = np.pi * (r ** 2)
    expected_vol = (4.0 / 3.0) * np.pi * (r ** 3)
    computed_vol = area_to_sphere_volume(area)
    assert np.isclose(computed_vol, expected_vol, rtol=1e-6)

    # Round trip volume -> diameter
    recovered_d = sphere_volume_to_diameter(computed_vol)
    assert np.isclose(recovered_d, d, rtol=1e-6)


def test_equivalent_diameter_scaling():
    """Area scales as s^2, equivalent diameter scales linearly with scale."""
    area_px = 1000.0
    scale1 = 1.0
    scale2 = 2.0
    d1 = equivalent_diameter(area_px, pixel_size_um=scale1)
    d2 = equivalent_diameter(area_px, pixel_size_um=scale2)
    assert np.isclose(d2, 2.0 * d1, rtol=1e-6)


def test_px_um_roundtrip():
    """Converting px -> um -> px returns original value."""
    px = 123.456
    scale = 1.518817
    um = px_to_um(px, scale)
    recovered_px = um_to_px(um, scale)
    assert np.isclose(recovered_px, px, rtol=1e-9)


def test_volume_scales_cubically():
    """Volume scales as scale^3 under optical magnification change."""
    d_um = 150.0
    vol1 = compute_spherical_volume(d_um)
    vol2 = compute_spherical_volume(2.0 * d_um)
    assert np.isclose(vol2, 8.0 * vol1, rtol=1e-6)


def test_single_volume_formula():
    """Grep test: The expression 'np.pi / 6.0' (or 'math.pi / 6.0') must appear ONLY in units.py in production code."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pattern = re.compile(r"(?:np|math)\.pi\s*/\s*6(?:\.0)?")

    scan_dirs = [
        repo_root / "spheroid_pipeline_v2",
        repo_root / "spheroid_if_sweep",
    ]
    if (repo_root / "src" / "matrix_mat").exists():
        scan_dirs.append(repo_root / "src" / "matrix_mat")

    violating_lines = []
    for sdir in scan_dirs:
        if not sdir.exists():
            continue
        for py_file in sdir.rglob("*.py"):
            if "__pycache__" in str(py_file) or "tests" in str(py_file):
                continue
            if py_file.name == "units.py":
                continue
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                if pattern.search(clean_line):
                    violating_lines.append(f"{py_file.relative_to(repo_root)}:{line_no}: {clean_line}")

    assert not violating_lines, (
        f"Volume formula duplicated outside units.py:\n" + "\n".join(violating_lines)
    )
