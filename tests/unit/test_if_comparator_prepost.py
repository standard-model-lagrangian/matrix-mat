"""Pre/post comparator: missing/duplicate ablation arms, robustness CV table, scope detection."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from matrix_mat.if_density import comparator_prepost as cp


def _abl_frame(arms=("PP00", "PP10", "PP01", "PP11"), n_images=3):
    rows = []
    slugs = {"PP00": "101_PP00_none_none", "PP10": "102_PP10_bg_only", "PP01": "103_PP01_post_only", "PP11": "104_PP11_full"}
    for arm in arms:
        for i in range(n_images):
            rows.append({"scope": "full", "image_id": f"img_{i}", "combo_slug": slugs[arm],
                         "n_nuclei_final": 100.0 + 10 * i, "flag_any": False,
                         "flag_debris_or_merged": False, "flags_list": ""})
    return pd.DataFrame(rows)


def test_missing_arm_reported_not_available_not_zero(tmp_path):
    out = tmp_path / "r.md"
    cp.generate_effect_summary_md(_abl_frame(arms=("PP00", "PP01", "PP11")), pd.DataFrame(), out)
    text = out.read_text()
    assert "PP10" in text and "not available" in text.lower()
    # The PP10 main effect must not be computed as if PP10 counted 0 nuclei.
    assert "-110.0" not in text and "-100.0%" not in text
    assert "Interaction: not available" in text
    # Effects whose arms exist are still reported.
    assert "`PP01` vs `PP00`" in text


def test_duplicate_arm_dirs_raise_with_names(tmp_path):
    runs = tmp_path / "runs"
    for name in ("101_PP00_none_none", "901_PP00_rerun"):
        (runs / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="101_PP00_none_none.*901_PP00_rerun"):
        cp.find_arm_dirs(runs)


def test_duplicate_arm_in_effects_table_raises(tmp_path):
    df = _abl_frame()
    extra = df[df["combo_slug"] == "101_PP00_none_none"].assign(combo_slug="999_PP00_again")
    with pytest.raises(ValueError, match="PP00"):
        cp.generate_effect_summary_md(pd.concat([df, extra]), pd.DataFrame(), tmp_path / "r.md")


def test_scope_from_subset_suffix_only():
    assert cp.slug_scope("101_PP00_none_none_subset") == "subset"
    assert cp.slug_scope("1s_PP00_x") == "full"
    assert cp.slug_scope("101_PP00_none_none") == "full"
    assert cp.ablation_arm("104_PP11_full") == "PP11"
    assert cp.ablation_arm("PP110_x") is None


def test_cv_spread_undefined_is_nan():
    assert np.isnan(cp.compute_cv_spread([5.0]))
    assert np.isnan(cp.compute_cv_spread([]))
    assert np.isnan(cp.compute_cv_spread([0.0, 0.0]))


def _per_image(runs: Path, name: str, counts: dict, dens_scale: float = 10.0):
    d = runs / name / "features"
    d.mkdir(parents=True)
    pd.DataFrame([
        {"image_id": k, "material": "Mat" if k < "m" else "S34D30", "n_nuclei": v, "nuclei_per_mm2": v * dens_scale}
        for k, v in counts.items()
    ]).to_csv(d / "per_image.csv", index=False)


def test_robustness_tolerates_missing_runs_intersects_both_families_and_keeps_precision(tmp_path):
    runs = tmp_path / "runs"
    _per_image(runs, "001_A_baseline", {"a": 100, "b": 50, "z": 10})
    _per_image(runs, "002_B_sensitive", {"a": 103, "b": 51, "z": 11})
    # 003 missing entirely
    _per_image(runs, "004_D_3d", {"a": 100, "b": 50, "z": 10})  # identical to 001
    _per_image(runs, "201_A_baseline_pp", {"a": 100, "b": 50})   # 'z' absent from the pp family
    _per_image(runs, "202_B_sensitive_pp", {"a": 101, "b": 50})
    out = tmp_path / "rob.csv"
    df = cp.generate_robustness_vs_original_csv(runs, out)
    allc = df[df["material"] == "ALL_COHORT"].iloc[0]
    assert allc["n_images"] == 2  # 'z' excluded because it is not in every run of both families
    # 3-decimal CVs, not rounded to 0.0 at 1 dp
    assert 0.0 < allc["orig_mean_count_spread"] < 0.05
    assert allc["orig_mean_count_spread"] == round(allc["orig_mean_count_spread"], 3)
    warns = " ".join(df.attrs["warnings"])
    assert "003_C_conservative" in warns
    assert "001_A_baseline" in warns and "004_D_3d" in warns and "identical" in warns

    md = tmp_path / "e.md"
    cp.generate_effect_summary_md(_abl_frame(), df, md)
    text = md.read_text()
    assert "identical outputs" in text
    assert f"{allc['orig_mean_count_spread']:.3f}" in text


def test_ablation_csv_labels_material_sd_as_field_level(tmp_path):
    runs = tmp_path / "runs"
    for slug in ("101_PP00_none_none", "104_PP11_full"):
        d = runs / slug
        (d / "qc").mkdir(parents=True)
        (d / "features").mkdir()
        pd.DataFrame([{"image_id": f"i{k}", "material": "Mat", "replicate": "Gel1", "flag_any": False,
                       "nuclei_per_mm2": 100.0 + k} for k in range(3)]).to_csv(d / "qc" / "qc_flags.csv", index=False)
        pd.DataFrame([{"image_id": f"i{k}", "material": "Mat", "replicate": "Gel1", "n_nuclei": 10 + k,
                       "nuclei_per_mm2": 100.0 + k, "median_nuclear_volume_um3": 500.0} for k in range(3)]).to_csv(
            d / "features" / "per_image.csv", index=False)
    df = cp.generate_ablation_effects_csv(runs, tmp_path / "out" / "ablation_effects.csv")
    summary = df[df["image_id"].astype(str).str.startswith("SUMMARY_")]
    assert summary["sd_level"].str.contains("field-level").all()
    pivot = pd.read_csv(tmp_path / "out" / "ablation_material_density_summary.csv")
    assert pivot["sd_level"].str.contains("field-level").all()
