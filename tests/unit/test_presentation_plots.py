import pandas as pd
import pytest

from matrix_mat.reporting.presentation_plots import get_effective_fov_counts


@pytest.mark.parametrize("fov_values", [["0001", "0004"], [1, 4]])
def test_effective_fov_counts_handles_string_and_int_fovs(fov_values):
    """In-memory pipeline frames carry zero-padded string FOVs; CSV round-trips give ints. Both must work."""
    f1, f4 = fov_values
    rows = []
    for i in range(3):
        rows.append({"condition": "Mat", "replicate": "gel1", "timepoint": "t0", "fov": f1, "qc_reasons": "None"})
        rows.append({
            "condition": "Mat", "replicate": "gel1", "timepoint": "t0", "fov": f4,
            "qc_reasons": "duplicate_focal_plane of object in fov 1",
        })
    counts = get_effective_fov_counts(pd.DataFrame(rows))
    # Three duplicate links between FOV 1 and FOV 4 collapse them into one physical field.
    assert counts.iloc[0]["n_fovs"] == 1


def test_duplicate_link_to_fov_outside_group_is_ignored():
    rows = [
        {"condition": "Mat", "replicate": "gel1", "timepoint": "t0", "fov": "0002", "qc_reasons": "None"},
        {"condition": "Mat", "replicate": "gel1", "timepoint": "t0", "fov": "0003",
         "qc_reasons": "duplicate_focal_plane in fov 9"},
    ]
    counts = get_effective_fov_counts(pd.DataFrame(rows))
    assert counts.iloc[0]["n_fovs"] == 2


def test_pooled_biomass_panel_does_not_borrow_significance_from_another_test():
    """Panel A plots well-pooled biomass FC; condition_summary significance tests spheroid FC, so it must not be shown there."""
    import inspect
    from matrix_mat.reporting import presentation_plots

    src = inspect.getsource(presentation_plots.generate_all_presentation_figures)
    panel_a = src[src.index("FIGURE 1"):src.index("ax1.axhline")]
    assert "significance" not in panel_a
