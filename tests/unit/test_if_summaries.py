import numpy as np
import pandas as pd
import pytest

from matrix_mat.core.stats import adjust_pvalues, cluster_permutation_test
from matrix_mat.reporting.summaries import compute_summary_by_material


def _row(material, gel, dens, clean=True):
    return {
        "image_id": f"{material}-{gel}-{dens}",
        "material": material,
        "replicate": gel,
        "nuclei_per_mm2": dens,
        "flag_summary": "clean" if clean else "blurry",
        "median_nuclear_volume_um3": 800.0,
        "spheroid_area_um2": 1e5,
    }


def _rows():
    rows = []
    for g, base in enumerate([1000.0, 1050.0, 980.0], start=1):
        rows += [_row("Mat", f"Gel{g}", base), _row("Mat", f"Gel{g}", base + 20)]
    for g, base in enumerate([1500.0, 1600.0, 1550.0], start=1):
        rows += [_row("S50", f"Gel{g}", base), _row("S50", f"Gel{g}", base + 10)]
    for g, base in enumerate([1010.0, 990.0, 1030.0], start=1):
        rows.append(_row("S34D30", f"Gel{g}", base))
    rows.append(_row("S46D10", "Gel1", 1200.0, clean=False))
    return rows


def test_gel_level_test_and_holm_adjustment(tmp_path):
    rows = _rows()
    summary = compute_summary_by_material(rows, tmp_path / "s.csv", control_material="Mat").set_index("material")
    df = pd.DataFrame(rows)

    raw = {}
    for m in ("S34D30", "S50"):
        sub = df[df["material"].isin(["Mat", m])]
        raw[m], min_p = cluster_permutation_test(sub, "nuclei_per_mm2", "material", cluster_col="replicate", ref_group="Mat")
        assert summary.loc[m, "p_raw"] == pytest.approx(raw[m])
        assert summary.loc[m, "min_achievable_p"] == pytest.approx(min_p)
    # S46D10 has a single gel: no test, so it does not enter the multiplicity family.
    adj = adjust_pvalues([raw["S34D30"], raw["S50"]], method="holm")
    assert summary.loc["S34D30", "p_adj"] == pytest.approx(adj[0])
    assert summary.loc["S50", "p_adj"] == pytest.approx(adj[1])
    assert summary.loc["Mat", "significance"] == "control"
    assert summary.loc["S50", "n_gels"] == 3
    assert summary.loc["S50", "gel_mean_nuclei_per_mm2"] == pytest.approx(np.mean([1505.0, 1605.0, 1555.0]))


def test_single_gel_material_gets_no_p_value(tmp_path):
    summary = compute_summary_by_material(_rows(), tmp_path / "s.csv", control_material="Mat").set_index("material")
    assert pd.isna(summary.loc["S46D10", "p_raw"])
    assert summary.loc["S46D10", "significance"] == "untested"
    assert "insufficient replicates" in summary.loc["S46D10", "statistical_note"]


def test_empty_subsets_are_nan_not_zero(tmp_path):
    summary = compute_summary_by_material(_rows(), tmp_path / "s.csv", control_material="Mat").set_index("material")
    # S46D10: one image, flagged -> no clean images and SD undefined
    assert summary.loc["S46D10", "n_images_clean"] == 0
    assert pd.isna(summary.loc["S46D10", "mean_nuclei_per_mm2_clean"])
    assert pd.isna(summary.loc["S46D10", "sd_nuclei_per_mm2_all"])


def test_summary_refuses_legacy_volumetric_density(tmp_path):
    legacy = [{"image_id": "a", "material": "Mat", "replicate": "Gel1", "nuclei_density_per_mm3": 1e5,
               "flag_summary": "clean"}]
    with pytest.raises(ValueError, match="nuclei_per_mm2"):
        compute_summary_by_material(legacy, tmp_path / "s.csv")


def test_summary_has_no_mislabelled_alias_columns(tmp_path):
    summary = compute_summary_by_material(_rows(), tmp_path / "s.csv")
    assert not [c for c in summary.columns if "density" in c and "mm2" not in c]


def test_density_figure_title_only_claims_tests_that_were_run():
    from matrix_mat.reporting.biological_plots import density_figure_subtitle

    untested = density_figure_subtitle(None)
    assert "p<" not in untested and "Holm" not in untested
    assert "no hypothesis test" in untested
    summary = pd.DataFrame({"material": ["Mat", "S50"], "p_adj": [np.nan, 0.2], "significance": ["control", "ns"]})
    sub = density_figure_subtitle(summary, control_material="Mat")
    assert "gel-level" in sub and "Holm" in sub and "Mat" in sub


def test_biological_figures_receive_gel_level_summary(tmp_path, monkeypatch):
    import matrix_mat.reporting.biological_plots as bp

    run = tmp_path / "run"
    (run / "features").mkdir(parents=True)
    (run / "summary").mkdir()
    pd.DataFrame(_rows()).to_csv(run / "features" / "per_image.csv", index=False)
    compute_summary_by_material(_rows(), run / "summary" / "summary_by_material.csv")

    seen = {}

    def fake_plot(per_img, out_dir, summary_df=None, **kw):
        seen["summary"] = summary_df
        return (out_dir / "a.png", out_dir / "a.pdf")

    monkeypatch.setattr(bp, "plot_nuclei_density_by_material", fake_plot)
    bp.generate_all_if_biological_figures(run)
    assert seen["summary"] is not None and "p_adj" in seen["summary"].columns


def test_clean_vs_all_title_makes_no_conclusion():
    import inspect
    import matrix_mat.reporting.biological_plots as bp

    src = inspect.getsource(bp.plot_clean_vs_all_density)
    assert "Confirming" not in src and "Does Not Alter" not in src


def test_significance_labels_survive_csv_round_trip(tmp_path):
    """Labels must not be strings pandas reads back as missing (e.g. 'n/a', 'NA')."""
    out = tmp_path / "s.csv"
    compute_summary_by_material(_rows(), out, control_material="Mat")
    back = pd.read_csv(out).set_index("material")
    assert back["significance"].notna().all()
    assert back.loc["S46D10", "significance"] == "untested"
