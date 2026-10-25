"""Property-based checks of the single canonical geometry implementation."""
import math

import pytest

pytest.importorskip("hypothesis")  # dev extra
from hypothesis import given, strategies as st  # noqa: E402

from matrix_mat.core.units import (
    area_to_sphere_volume,
    compute_spherical_volume,
    equivalent_diameter,
    px_to_um,
    sphere_volume_to_diameter,
    um_to_px,
)

pos = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False)


@given(pos)
def test_diameter_volume_round_trip(d):
    assert math.isclose(sphere_volume_to_diameter(compute_spherical_volume(d)), d, rel_tol=1e-9)


@given(pos, st.floats(min_value=0.1, max_value=10.0))
def test_volume_scales_cubically_with_length_unit(d, s):
    assert math.isclose(compute_spherical_volume(d * s), compute_spherical_volume(d) * s ** 3, rel_tol=1e-9)


@given(pos)
def test_area_to_volume_matches_equivalent_diameter(area):
    d = 2.0 * math.sqrt(area / math.pi)
    assert math.isclose(area_to_sphere_volume(area), compute_spherical_volume(d), rel_tol=1e-9)


@given(pos, st.floats(min_value=0.05, max_value=20.0))
def test_px_um_round_trip(px, size):
    assert math.isclose(um_to_px(px_to_um(px, size), size), px, rel_tol=1e-9)


@given(pos, st.floats(min_value=0.05, max_value=20.0))
def test_equivalent_diameter_scales_with_pixel_size(area_px, size):
    assert math.isclose(equivalent_diameter(area_px, size), equivalent_diameter(area_px, 1.0) * size, rel_tol=1e-9)
