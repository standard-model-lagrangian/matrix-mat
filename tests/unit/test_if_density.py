import numpy as np
import pytest
import spheroid_if_sweep
from spheroid_if_sweep.features import (
    compute_spheroid_extent,
    compute_areal_density,
    compute_volumetric_density_3d,
    extract_objects_and_image_features,
)


def _generate_hex_packed_nuclei(radius_um: float, spacing_um: float = 12.0) -> int:
    """Count how many hexagonally packed nuclei fit inside a circular disk of given radius."""
    count = 0
    y_step = spacing_um * np.sqrt(3) / 2.0
    x_step = spacing_um
    
    max_y = int(np.ceil(radius_um / y_step))
    max_x = int(np.ceil(radius_um / x_step))
    
    for row in range(-max_y, max_y + 1):
        y = row * y_step
        x_offset = (spacing_um / 2.0) if (row % 2 != 0) else 0.0
        for col in range(-max_x - 1, max_x + 2):
            x = col * x_step + x_offset
            if x * x + y * y <= radius_um * radius_um:
                count += 1
    return count


def test_areal_density_invariant_to_spheroid_size():
    """Synthetic hexagonally packed nuclei (12 um spacing) inside disks of r=50, 100, 200 um.
    Areal density must agree within +-5% across sizes.
    """
    radii = [50.0, 100.0, 200.0]
    densities = []
    
    for r in radii:
        area_um2 = np.pi * (r ** 2)
        n_nuclei = _generate_hex_packed_nuclei(r, spacing_um=12.0)
        dens = compute_areal_density(n_nuclei, area_um2)
        densities.append(dens)
    
    mean_dens = float(np.mean(densities))
    for d in densities:
        rel_diff = abs(d - mean_dens) / mean_dens
        assert rel_diff < 0.05, f"Areal density {d} differs from mean {mean_dens} by {rel_diff:.1%}"


def test_missing_spheroid_mask_yields_nan_density():
    """Missing spheroid mask yields NaN density instead of 1 um^3 fallback."""
    actin_empty = np.zeros((100, 100), dtype=np.uint8)
    nuc_empty = np.zeros((100, 100), dtype=np.int32)
    
    sph_vol, cy, cx, r_max = compute_spheroid_extent(actin_empty, nuc_empty, pixel_size_um=1.5)
    assert np.isnan(sph_vol) or sph_vol == 0.0
    
    objs, per_img = extract_objects_and_image_features(
        image_id="test_empty",
        material="Mat",
        replicate="Gel1",
        nuclear_img=actin_empty,
        actin_img=actin_empty,
        dedup_masks=nuc_empty,
        pixel_size_um=1.5,
        qc_flags_list="",
        feature_config={},
    )
    assert np.isnan(per_img["nuclei_per_mm2"])


def test_single_plane_input_refuses_volumetric_metric():
    """Calling 3D volumetric density metric on a 2D image raises ValueError."""
    img_2d = np.zeros((512, 512), dtype=np.uint8)
    with pytest.raises(ValueError, match="[Ss]ingle-plane.*2D.*3D.*volumetric"):
        compute_volumetric_density_3d(n_nuclei=50, z_stack_or_volume_um3=img_2d)

    with pytest.raises(ValueError, match="[Ss]ingle-plane.*2D.*3D.*volumetric"):
        compute_volumetric_density_3d(n_nuclei=50, z_stack_or_volume_um3=1e6, is_2d_plane=True)


def test_density_units_label_matches_values():
    """The plot y-label unit matches the numeric scale."""
    from spheroid_if_sweep.biological_plots import DENSITY_Y_LABEL
    assert "mm^2" in DENSITY_Y_LABEL or "mm²" in DENSITY_Y_LABEL
    assert "mm^3" not in DENSITY_Y_LABEL and "mm³" not in DENSITY_Y_LABEL


def _synthetic_spheroid_field(radius_px: int, spacing_px: int = 12, size: int = 460):
    """Actin-filled disk with hexagonally packed 3-px nuclei inside it, rendered as images and labels."""
    from skimage import draw

    actin = np.full((size, size), 10, dtype=np.uint8)
    rr, cc = draw.disk((size // 2, size // 2), radius_px, shape=actin.shape)
    actin[rr, cc] = 180
    nuclear = np.zeros_like(actin)
    labels = np.zeros(actin.shape, dtype=np.int32)
    lab = 0
    y_step = spacing_px * np.sqrt(3) / 2
    for row in range(-int(radius_px / y_step) - 1, int(radius_px / y_step) + 2):
        off = spacing_px / 2 if row % 2 else 0.0
        for col in range(-radius_px // spacing_px - 2, radius_px // spacing_px + 3):
            y, x = row * y_step, col * spacing_px + off
            if y * y + x * x <= (radius_px - 4) ** 2:
                lab += 1
                r2, c2 = draw.disk((size // 2 + y, size // 2 + x), 3, shape=actin.shape)
                labels[r2, c2] = lab
                nuclear[r2, c2] = 200
    return nuclear, actin, labels


def test_areal_density_invariant_to_spheroid_size_through_feature_pipeline():
    """Same packing in spheroids of different size must give the same nuclei/mm2 via the production code path."""
    densities = []
    for radius in (60, 120, 200):
        nuclear, actin, labels = _synthetic_spheroid_field(radius)
        _, per_img = extract_objects_and_image_features(
            image_id=f"r{radius}", material="Mat", replicate="Gel1",
            nuclear_img=nuclear, actin_img=actin, dedup_masks=labels,
            pixel_size_um=1.0, qc_flags_list="", feature_config={},
        )
        densities.append(per_img["nuclei_per_mm2"])
        assert "nuclei_density_per_mm3" not in per_img
    densities = np.array(densities)
    assert np.all(np.abs(densities / densities.mean() - 1.0) < 0.05), densities
