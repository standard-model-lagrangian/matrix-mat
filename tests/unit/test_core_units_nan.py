"""NaN propagation and pixel-size validation in core.units (0.2.2)."""

import math

import pytest

from matrix_mat.core.units import (
    area_to_sphere_volume,
    compute_circularity,
    compute_ellipsoidal_volume,
    compute_spherical_volume,
    compute_volume_discrepancy_ratio,
    equivalent_diameter,
    px_to_um,
    sphere_volume_to_diameter,
    um_to_px,
)

NAN = float("nan")


@pytest.mark.parametrize("fn,args", [
    (compute_spherical_volume, (NAN,)),
    (compute_ellipsoidal_volume, (NAN, 1.0)),
    (compute_ellipsoidal_volume, (1.0, NAN)),
    (area_to_sphere_volume, (NAN,)),
    (sphere_volume_to_diameter, (NAN,)),
    (equivalent_diameter, (NAN, 1.0)),
    (px_to_um, (NAN, 1.0)),
    (um_to_px, (NAN, 1.0)),
    (compute_volume_discrepancy_ratio, (NAN, 1.0)),
    (compute_circularity, (NAN, 10.0)),
])
def test_nan_input_propagates(fn, args):
    assert math.isnan(fn(*args))


@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, float("inf")])
@pytest.mark.parametrize("fn", [lambda s: equivalent_diameter(100.0, s), lambda s: um_to_px(10.0, s),
                                lambda s: px_to_um(10.0, s)])
def test_non_positive_or_nan_pixel_size_raises(fn, bad):
    with pytest.raises(ValueError, match="pixel"):
        fn(bad)


def test_negative_values_still_clip_to_zero():
    assert compute_spherical_volume(-5.0) == 0.0
    assert equivalent_diameter(-5.0, 1.0) == 0.0
