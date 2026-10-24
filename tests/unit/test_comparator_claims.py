import ast
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest
import spheroid_if_sweep
from spheroid_if_sweep.qc import ImageQCMetrics, apply_cohort_qc_flags, compute_image_qc_metrics
from spheroid_if_sweep.comparator_prepost import run_comparator_prepost

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_metric(image_id: str, material: str, replicate: str, lap_var: float, density: float) -> ImageQCMetrics:
    return ImageQCMetrics(
        image_id=image_id,
        material=material,
        replicate=replicate,
        n_nuclei_raw=50,
        n_nuclei_dedup=50,
        n_duplicates_removed=0,
        nuclear_snr=25.0,
        nuclear_laplacian_var=lap_var,
        nuclear_saturation_frac=0.0,
        actin_saturation_frac=0.0,
        edge_touching_frac=0.0,
        debris_count=0,
        merged_count=0,
        flag_empty_seg=False,
        flag_count_outlier=False,
        flag_dup_masks=False,
        flag_debris_or_merged=False,
        flag_edge_heavy=False,
        flag_saturated=False,
        flag_blurry=False,
        flag_low_snr=False,
        flag_any=False,
        flags_list="",
        flag_density_outlier=False,
        nuclei_per_mm2=density,
    )


def test_clean_subset_excludes_no_endpoint_flags():
    """flag_any must NOT include flag_density_outlier or flag_count_outlier (endpoint flags)."""
    metric = _make_metric("img_1", "Mat", "Gel1", 300.0, 100000.0)  # extreme outlier
    norm1 = _make_metric("norm1", "Mat", "Gel2", 300.0, 20000.0)
    norm2 = _make_metric("norm2", "Mat", "Gel3", 300.0, 21000.0)
    norm3 = _make_metric("norm3", "Mat", "Gel4", 300.0, 20500.0)

    cohort = apply_cohort_qc_flags([metric, norm1, norm2, norm3], qc_config={"outlier_iqr_factor": 1.5, "blur_abs_threshold": 50.0})
    res_metric = [m for m in cohort if m.image_id == "img_1"][0]

    assert res_metric.flag_density_outlier is True, "Expected density outlier to be flagged"
    assert res_metric.flag_any is False, "flag_any must NOT include outcome/endpoint outlier flags"


def test_blur_flag_uses_absolute_threshold():
    """A cohort of uniformly sharp images flags 0 as blurry when using absolute threshold."""
    sharp_images = [
        _make_metric(f"sharp_{i}", "Mat", f"Gel{i}", var, 20000.0)
        for i, var in enumerate([400.0, 450.0, 500.0, 600.0, 800.0])
    ]
    res = apply_cohort_qc_flags(sharp_images, qc_config={"blur_abs_threshold": 50.0})
    blurry_count = sum(1 for m in res if m.flag_blurry)
    assert blurry_count == 0, f"Expected 0 blurry images in sharp cohort, got {blurry_count}"


def test_qc_metrics_computed_on_raw_image():
    """Saturation fraction on a raw image with no 255 values is 0, even if preprocessed has saturated pixels."""
    raw_img = np.full((100, 100), 100, dtype=np.uint8)  # max value 100, not saturated
    empty_masks = np.zeros((100, 100), dtype=np.int32)
    metric = compute_image_qc_metrics(
        image_id="test_raw_sat",
        material="Mat",
        replicate="Gel1",
        nuclear_img=raw_img,
        actin_img=raw_img,
        raw_masks=empty_masks,
        dedup_masks=empty_masks,
        n_dups_removed=0,
        pixel_size_um=1.5,
        qc_config={},
    )
    assert metric.nuclear_saturation_frac == 0.0
    assert metric.flag_saturated is False


def test_no_literal_claims_in_source():
    """Static claim words (eliminated|confirmed|resolved|collapse|drift) must not appear as uncomputed literals."""
    comparator_file = (REPO_ROOT / "src/matrix_mat/if_density/comparator_prepost.py")
    content = comparator_file.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(comparator_file))

    claim_pattern = re.compile(r"\b(eliminated|confirmed|resolved|collapse|drift)\b", re.IGNORECASE)

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            matches = claim_pattern.findall(node.value)
            if matches and any(word.lower() in ("eliminated", "confirmed", "resolved", "collapse", "drift") for word in matches):
                if any(k in node.value for k in ("100% eliminated", "Decision Confirmed", "RESOLVED", "No count collapse", "focal drift")):
                    violations.append((node.lineno, node.value.strip()[:80]))

    assert not violations, f"Hardcoded literal claims found in comparator: {violations}"


def test_report_text_changes_with_data(tmp_path):
    """Feed the comparator two synthetic summary tables with opposite effects; generated reports must differ."""
    from spheroid_if_sweep.comparator_prepost import generate_effect_summary_md

    def _make_dfs(reduced: bool):
        ablation_rows = []
        for slug in ["PP00", "PP10", "PP01", "PP11"]:
            ablation_rows.append({
                "scope": "full",
                "image_id": f"img_{slug}",
                "combo_slug": slug,
                "n_nuclei_final": 100.0 if slug == "PP00" else (120.0 if reduced else 50.0),
                "flag_any": False,
                "flag_debris_or_merged": (slug == "PP00") if reduced else (slug == "PP11"),
                "flags_list": "",
            })
        ablation_df = pd.DataFrame(ablation_rows)
        robustness_df = pd.DataFrame([{
            "material": "ALL_COHORT",
            "spread_reduction_mean_pct": 50.0 if reduced else -50.0,
            "spread_reduction_max_pct": 60.0 if reduced else -30.0,
            "orig_mean_count_spread": 20.0,
            "pp_mean_count_spread": 10.0 if reduced else 30.0,
            "orig_mean_density_spread": 200.0,
            "pp_mean_density_spread": 100.0 if reduced else 300.0,
            "density_spread_reduction_pct": 50.0 if reduced else -50.0,
            "n_images": 4,
        }])
        return ablation_df, robustness_df

    abl1, rob1 = _make_dfs(reduced=True)
    out1 = tmp_path / "report1.md"
    generate_effect_summary_md(abl1, rob1, out1)
    text1 = out1.read_text(encoding="utf-8")

    abl2, rob2 = _make_dfs(reduced=False)
    out2 = tmp_path / "report2.md"
    generate_effect_summary_md(abl2, rob2, out2)
    text2 = out2.read_text(encoding="utf-8")

    assert text1 != text2, "Reports generated from opposite datasets should differ"
    assert "+50.0% reduction" in text1 and "-50.0% reduction" in text2



def test_blur_flag_requires_explicit_threshold_or_opt_in():
    """Without an absolute threshold, blur flagging must not silently fall back to 'worst 10% of the cohort'."""
    imgs = [_make_metric(f"i{i}", "Mat", f"Gel{i}", v, 1000.0) for i, v in enumerate([400.0, 450.0, 500.0])]
    with pytest.raises(ValueError, match="blur_abs_threshold"):
        apply_cohort_qc_flags(imgs, qc_config={})


def test_blur_percentile_mode_is_explicit_opt_in():
    imgs = [_make_metric(f"i{i}", "Mat", f"Gel{i}", v, 1000.0) for i, v in enumerate(range(100, 1100, 100))]
    res = apply_cohort_qc_flags(imgs, qc_config={"blur_mode": "cohort_percentile", "blur_percentile_cutoff": 10.0})
    assert sum(m.flag_blurry for m in res) == 1


def test_shipped_if_configs_use_absolute_blur_threshold():
    import yaml
    from matrix_mat.if_density.config import IFSweepConfig

    for name in ("sweep.yaml", "prepost_ablation.yaml", "robustness_check.yaml"):
        cfg = IFSweepConfig.from_yaml(REPO_ROOT / "configs" / name)
        assert cfg.qc.blur_mode == "absolute", name
        assert cfg.qc.blur_abs_threshold is not None and cfg.qc.blur_abs_threshold > 0, name


def _ablation_frame(n_images=3):
    rows = []
    for slug in ["101_PP00_none_none", "102_PP10_bg_only", "103_PP01_post_only", "104_PP11_full"]:
        for i in range(n_images):
            rows.append({
                "scope": "full", "image_id": f"img_{i}", "combo_slug": slug,
                "n_nuclei_final": 100.0 - i, "flag_any": i == 0,
                "flag_debris_or_merged": i == 0 and "PP0" in slug, "flags_list": "",
            })
    return pd.DataFrame(rows)


def test_missing_robustness_runs_are_reported_as_not_evaluated(tmp_path):
    from spheroid_if_sweep.comparator_prepost import generate_effect_summary_md

    out = tmp_path / "r.md"
    generate_effect_summary_md(_ablation_frame(), pd.DataFrame(), out)
    text = out.read_text()
    assert "not evaluated" in text.lower()
    assert "unaltered" not in text and "was reduced" not in text


def test_report_counts_fields_from_data(tmp_path):
    from spheroid_if_sweep.comparator_prepost import generate_effect_summary_md

    out = tmp_path / "r.md"
    generate_effect_summary_md(_ablation_frame(n_images=5), pd.DataFrame(), out)
    text = out.read_text()
    assert "5 fields" in text
    assert "27" not in text


def test_generated_report_makes_no_uncomputed_mechanistic_or_verdict_claims(tmp_path):
    """Mechanistic explanations and recommendations are author judgement; they belong in docs/findings."""
    from spheroid_if_sweep.comparator_prepost import generate_effect_summary_md

    out = tmp_path / "r.md"
    generate_effect_summary_md(_ablation_frame(), pd.DataFrame(), out)
    text = out.read_text().lower()
    banned = ["purges", "synergistic", "adopt", "permanent production", "effectively", "neutraliz",
              "anchors signal", "stabilizing", "s43d20-gel1-1-2", "image median"]
    assert not [w for w in banned if w in text]


def test_report_flags_debris_reduction_as_mechanical(tmp_path):
    from spheroid_if_sweep.comparator_prepost import generate_effect_summary_md

    out = tmp_path / "r.md"
    generate_effect_summary_md(_ablation_frame(), pd.DataFrame(), out)
    assert "by construction" in out.read_text()
