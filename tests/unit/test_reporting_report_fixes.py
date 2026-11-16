"""reporting.report fixes (0.2.2)."""

import re

import numpy as np
import pandas as pd
import pytest

import matrix_mat
from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.reporting.report import ReportGenerator, df_to_markdown_table, format_p_value


def _data():
    manifest = pd.DataFrame([
        {"pair_key": f"Mat_gel1_000{i}", "is_matched_pair": True} for i in (1, 2)
    ] * 2 + [{"pair_key": "Mat_gel1_0009", "is_matched_pair": False}])
    objs = []
    # FOV 1: PASS at both timepoints (called). FOV 2: PASS only at t0. Unmatched FOV 9 also has PASS.
    for key, tp in (("Mat_gel1_0001", "t0"), ("Mat_gel1_0001", "t7"), ("Mat_gel1_0002", "t0"),
                    ("Mat_gel1_0009", "t0"), ("PEG9_gel1_0001", "t0"), ("PEG9_gel1_0001", "t7")):
        for _k in range(4):
            objs.append({"pair_key": key, "timepoint": tp, "qc_flag": "PASS", "condition": key.split("_")[0],
                         "image_id": f"{key}_{tp}", "volume_sphere_um3": 1e5, "equivalent_diameter_um": 50.0,
                         "pixel_size_um": 1.518817 if key != "PEG9_gel1_0001" else 0.65,
                         "scale_source": "evos_tag_37510" if key != "PEG9_gel1_0001" else "cli_override"})
    pairs = pd.DataFrame([
        {"condition": "Mat", "combined_qc_flag": "PASS", "denominator_gate_pass": True, "fold_change_volume": 2.0},
        {"condition": "Mat", "combined_qc_flag": "FAIL", "denominator_gate_pass": True, "fold_change_volume": 2.0},
        {"condition": "Mat", "combined_qc_flag": "REVIEW", "denominator_gate_pass": True, "fold_change_volume": 2.0},
        {"condition": "Mat", "combined_qc_flag": "PASS", "denominator_gate_pass": False, "fold_change_volume": 2.0},
    ])
    cond = pd.DataFrame([{"condition": "Mat", "p_raw": np.nan, "p_adj": np.nan, "significance": "control"},
                         {"condition": "PEG9", "p_raw": 0.0004, "p_adj": np.nan, "significance": "***"}])
    return manifest, pd.DataFrame(objs), pairs, cond


def _gen(tmp_path, **cfg):
    config = PipelineConfig(output_dir=str(tmp_path))
    for k, v in cfg.items():
        obj = config
        *path, last = k.split(".")
        for pth in path:
            obj = getattr(obj, pth)
        setattr(obj, last, v)
    m, o, p, c = _data()
    md, html = ReportGenerator(config, tmp_path).generate(m, o, p, c)
    return md.read_text(), html.read_text()


def test_call_rate_bounded_and_defined(tmp_path):
    md, html = _gen(tmp_path)
    m = re.search(r"Segmentation Call Rate\*\* \| `([^`]+)`", md)
    assert m.group(1) == "50.0%"  # 1 of 2 matched FOVs has PASS objects at both timepoints


def test_call_rate_zero_denominator_is_na(tmp_path):
    config = PipelineConfig(output_dir=str(tmp_path))
    _, o, p, c = _data()
    md = ReportGenerator(config, tmp_path).generate(pd.DataFrame(), o, p, c)[0].read_text()
    assert re.search(r"Segmentation Call Rate\*\* \| `n/a`", md)


def test_version_from_package(tmp_path):
    md, html = _gen(tmp_path)
    assert f"`{matrix_mat.__version__}`" in md and "2.0.0" not in md


def test_effective_scales_reported(tmp_path):
    md, html = _gen(tmp_path)
    assert "1.518817" in md and "evos_tag_37510" in md
    assert "0.650000" in md and "cli_override" in md
    assert "evos_tag_37510" in html


def test_scale_falls_back_to_override(tmp_path):
    config = PipelineConfig(output_dir=str(tmp_path), cli_pixel_size_um=0.9)
    m, o, p, c = _data()
    o = o.drop(columns=["pixel_size_um", "scale_source"])
    md = ReportGenerator(config, tmp_path).generate(m, o, p, c)[0].read_text()
    assert "0.900000" in md and "cli_override" in md


def test_denominator_gate_counts_only_stats_pairs(tmp_path):
    md, _ = _gen(tmp_path)
    assert re.search(r"Denominator Gate Pass\*\* \| `1`", md)
    md2, _ = _gen(tmp_path, **{"stats.include_review_in_stats": True})
    assert re.search(r"Denominator Gate Pass\*\* \| `2`", md2)


def test_gate_threshold_from_config(tmp_path):
    md, html = _gen(tmp_path, **{"pairing.min_t0_d_um_denominator": 75.0})
    assert "75" in md and "60\\," not in md
    assert "75" in html


def test_footnote_matches_core_stats(tmp_path):
    md, _ = _gen(tmp_path, **{"stats.permutation_iterations": 1234})
    assert "M=10,000" not in md
    assert "exact" in md.lower() and "1,234" in md


def test_p_raw_fallback_not_labelled_adjusted(tmp_path):
    md, _ = _gen(tmp_path)
    assert "unadjusted" in md
    assert "p_adj (vs Mat)" not in md


@pytest.mark.parametrize("p,expected", [(0.0004, "<0.001"), (0.0123, "0.012"), (np.nan, "—"), (1.0, "1.000")])
def test_format_p_value(p, expected):
    assert format_p_value(p) == expected


def test_small_floats_not_rendered_as_zero():
    out = df_to_markdown_table(pd.DataFrame({"x": [0.0004]}))
    assert "0.000 " not in out and "4.00e-04" in out


def test_conditions_derived_from_data(tmp_path):
    md, _ = _gen(tmp_path)
    assert "| PEG9" in md


def test_html_has_no_cdn_and_no_raw_latex(tmp_path):
    _, html = _gen(tmp_path)
    assert "cdn." not in html and "<link" not in html
    assert "\\mu" not in html and "$" not in html
