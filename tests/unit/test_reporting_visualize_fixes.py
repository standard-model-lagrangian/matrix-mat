"""reporting.visualize fixes (0.2.2)."""

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from matrix_mat.reporting import visualize as vz  # noqa: E402


@pytest.fixture(autouse=True)
def _no_savefig(monkeypatch):
    calls = []

    def fake(self, path, *a, **k):
        calls.append((str(path), k.get("dpi")))

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fake)
    return calls


def _pairs(conds=("Mat", "S50", "PEG9")):
    rows = []
    for c in conds:
        for k in range(3):
            rows.append({"condition": c, "combined_qc_flag": "PASS", "denominator_gate_pass": True,
                         "t0_volume_sphere_um3": 1e5 * (k + 1), "t7_volume_sphere_um3": 2e5 * (k + 1),
                         "fold_change_volume": 2.0})
    return pd.DataFrame(rows)


def test_scatter_with_no_valid_pairs_does_not_crash(tmp_path):
    df = _pairs()
    df["t0_volume_sphere_um3"] = -1.0
    before = set(plt.get_fignums())
    vz.plot_v0_vs_v7_scatter(df, tmp_path / "s")
    assert set(plt.get_fignums()) == before


def test_overlay_scales_16bit_instead_of_wrapping(tmp_path):
    img = np.full((20, 20), 1000, dtype=np.uint16)
    img[:10] = 3000  # astype(uint8) would wrap 3000 -> 184 and 1000 -> 232 (inverting contrast)
    out = vz.create_overlay(img, np.zeros_like(img, dtype=np.int32), [], tmp_path / "o.png")
    bgr = cv2.imread(str(out))
    assert bgr[0, 0, 0] > bgr[15, 0, 0]
    assert bgr[0, 0, 0] == 255 and bgr[15, 0, 0] == 0


def test_violin_empty_group_draws_no_placeholder(tmp_path, monkeypatch):
    df = _pairs()
    df.loc[df.condition == "S50", "fold_change_volume"] = np.nan
    seen = {}
    orig = matplotlib.axes.Axes.violinplot

    def spy(self, dataset, *a, **k):
        seen["n"] = len(dataset)
        seen["data"] = [list(d) for d in dataset]
        return orig(self, dataset, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "violinplot", spy)
    vz.plot_fold_change_violin(df, tmp_path / "v")
    assert seen["n"] == 2 and [1.0] not in seen["data"]


def test_violin_title_uses_configured_gate(tmp_path, monkeypatch):
    titles = []
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", lambda self, t, *a, **k: titles.append(t))
    vz.plot_fold_change_violin(_pairs(), tmp_path / "v", min_t0_d_um=75.0)
    assert any("75" in t for t in titles) and not any("60" in t for t in titles)


def _objects():
    rows = []
    for c in ("Mat", "S50", "PEG9"):
        for tp in ("t0", "t7", "t10"):
            for k in range(3):
                rows.append({"condition": c, "timepoint": tp, "qc_flag": "PASS", "replicate": f"gel{k}",
                             "volume_sphere_um3": 1e5 * (k + 1)})
    df = pd.DataFrame(rows)
    return df[~((df.condition == "S50") & (df.timepoint == "t7"))]


def test_population_plot_derives_conditions_and_timepoints(tmp_path, monkeypatch):
    seen = {}
    orig = matplotlib.axes.Axes.boxplot

    def spy(self, data, *a, **k):
        seen["n_boxes"] = len(data)
        seen["data"] = [list(d) for d in data]
        return orig(self, data, *a, **k)

    labels = []
    orig_lab = matplotlib.axes.Axes.set_xticklabels
    monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", spy)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticklabels",
                        lambda self, lab, *a, **k: (labels.extend(lab), orig_lab(self, lab, *a, **k))[1])
    vz.plot_population_volume_distributions(_objects(), tmp_path / "p")
    # 3 conditions x 3 timepoints, minus the empty S50/t7 group (no fake 1e5 box).
    assert seen["n_boxes"] == 8
    assert "PEG9" in labels


def test_population_annotation_labels_raw_p_honestly(tmp_path, monkeypatch):
    texts = []
    monkeypatch.setattr(matplotlib.axes.Axes, "text", lambda self, x, y, s, *a, **k: texts.append(s))
    cond = pd.DataFrame([{"condition": "Mat", "p_raw": 0.01, "n_gels": 3}])
    vz.plot_population_volume_distributions(_objects(), tmp_path / "p", condition_df=cond)
    joined = "\n".join(texts)
    assert "Holm" not in joined and "unadjusted" in joined


def test_contact_sheets_group_by_parsed_condition(tmp_path):
    ov = tmp_path / "ov"
    ov.mkdir()
    for name in ("Day0 Mat gel1_0001_TRANS_overlay.png", "Day0 PEG9 gel1_0001_TRANS_overlay.png",
                 "Day0 S34D30 gel1_0001_TRANS_overlay.png"):
        Image.new("RGB", (10, 10)).save(ov / name)
    res = vz.generate_all_contact_sheets(ov, tmp_path / "out")
    assert set(res) == {"Mat", "PEG9", "S34D30"}


def test_contact_sheet_dpi(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "a.png")
    out = vz.create_contact_sheet([tmp_path / "a.png"], tmp_path / "cs.png", dpi=150)
    with Image.open(out) as im:
        assert round(im.info["dpi"][0]) == 150


def test_visualizer_uses_figure_dpi(tmp_path, _no_savefig):
    from matrix_mat.brightfield.config import ArtifactConfig

    v = vz.Visualizer(ArtifactConfig(figure_dpi=123))
    objs = _objects().assign(circularity=0.9, volume_discrepancy_ratio=0.05)
    v.generate_figures(_pairs(), objs, tmp_path)
    pngs = [d for p, d in _no_savefig if p.endswith(".png")]
    assert pngs and all(d == 123 for d in pngs)


def test_figures_closed_on_error(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", boom)
    before = set(plt.get_fignums())
    with pytest.raises(OSError):
        vz.plot_fold_change_violin(_pairs(), tmp_path / "v")
    assert set(plt.get_fignums()) == before


def test_contact_sheets_use_explicit_overlay_list(tmp_path):
    ov = tmp_path / "ov"
    ov.mkdir()
    for name in ("Day0 Mat gel1_0001_TRANS_overlay.png", "Day0 Mat gel9_0009_TRANS_overlay.png"):
        Image.new("RGB", (10, 10)).save(ov / name)
    res = vz.generate_condition_contact_sheets(ov, tmp_path / "out", pd.DataFrame(),
                                               overlay_paths=[ov / "Day0 Mat gel1_0001_TRANS_overlay.png"])
    assert set(res) == {"Mat"}
    # Without an explicit list, the manifest's image_ids filter out stale overlays.
    manifest = pd.DataFrame({"image_id": ["Day0 Mat gel1_0001_TRANS"], "condition": ["Mat"]})
    seen = []
    orig = vz.create_contact_sheet
    try:
        vz.create_contact_sheet = lambda overlay_paths, *a, **k: seen.extend(overlay_paths) or orig(overlay_paths, *a, **k)
        vz.generate_condition_contact_sheets(ov, tmp_path / "out2", manifest)
    finally:
        vz.create_contact_sheet = orig
    assert [p.name for p in seen] == ["Day0 Mat gel1_0001_TRANS_overlay.png"]


def test_overlay_ignores_fail_records_and_unknown_labels(tmp_path):
    img = np.full((40, 40), 128, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=np.int32)
    mask[5:15, 5:15] = 1
    recs = [{"label_idx": 1, "qc_flag": "FAIL", "object_id": "x_FAILREC"},
            {"label_idx": 1, "qc_flag": "PASS", "object_id": "x_PASSREC"},
            {"label_idx": 7, "qc_flag": "REVIEW", "object_id": "x_GHOST"}]
    # FAIL record listed last must not override the PASS record for label 1.
    out = vz.create_overlay(img, mask, recs[1:] + recs[:1], tmp_path / "o.png")
    bgr = cv2.imread(str(out))
    green = np.all(np.abs(bgr.astype(int) - np.array(vz.QC_COLORS_BGR["PASS"])) < 40, axis=-1)
    red = np.all(np.abs(bgr.astype(int) - np.array(vz.QC_COLORS_BGR["FAIL"])) < 40, axis=-1)
    assert green.sum() > 0 and red.sum() == 0
