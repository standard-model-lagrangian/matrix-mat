"""Cross-run comparator (generic): missing images, run selection, panels, computed report text."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from matrix_mat.if_density import comparator as comp


def _write_run(runs: Path, name: str, rows, overlay_ids=()):
    d = runs / name
    (d / "features").mkdir(parents=True)
    (d / "qc").mkdir()
    (d / "masks").mkdir()
    pd.DataFrame(rows).to_csv(d / "features" / "per_image.csv", index=False)
    pd.DataFrame([{"image_id": r["image_id"], "flag_any": False} for r in rows]).to_csv(d / "qc" / "qc_flags.csv", index=False)
    for i in overlay_ids:
        cv2.imwrite(str(d / "masks" / f"{i}_overlay.png"), np.full((64, 64, 3), 80, np.uint8))
    return d


def _row(img, n, dens, mat="Mat"):
    return {"image_id": img, "material": mat, "n_nuclei": n, "nuclei_per_mm2": dens, "flag_summary": "clean"}


def test_missing_image_is_nan_and_excluded_from_spread():
    per_images = {
        "r1": pd.DataFrame([_row("a", 100, 1000.0), _row("b", 50, 500.0)]),
        "r2": pd.DataFrame([_row("a", 110, 1100.0)]),  # 'b' missing
        "r3": pd.DataFrame([_row("a", 90, 900.0), _row("b", 52, 520.0), _row("c", 10, 100.0)]),
    }
    d = comp.compute_per_image_deltas(per_images).set_index("image_id")
    # union of image ids, not just the first run's
    assert set(d.index) == {"a", "b", "c"}
    assert np.isnan(d.loc["b", "r2_count"])
    assert d.loc["b", "n_runs_present"] == 2
    assert d.loc["b", "count_spread"] == 2  # 52 - 50, not 52 - 0
    assert d.loc["a", "count_spread"] == 20
    # 'c' only in one run -> no spread, not ranked
    assert d.loc["c", "n_runs_present"] == 1
    assert np.isnan(d.loc["c", "count_spread"]) and np.isnan(d.loc["c", "rank"])
    assert d.loc["a", "rank"] == 1


def test_spread_independent_of_run_order_with_nan_density():
    a = {"r1": pd.DataFrame([_row("x", 10, float("nan"))]), "r2": pd.DataFrame([_row("x", 20, 300.0)]),
         "r3": pd.DataFrame([_row("x", 30, 100.0)])}
    b = dict(reversed(list(a.items())))
    da = comp.compute_per_image_deltas(a).iloc[0]
    db = comp.compute_per_image_deltas(b).iloc[0]
    assert da["density_spread"] == db["density_spread"] == 200.0
    assert da["count_spread"] == db["count_spread"] == 20


def test_default_selection_separates_full_and_subset_runs(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "001_A", [_row("a", 1, 1.0)])
    _write_run(runs, "002_B", [_row("a", 2, 2.0)])
    _write_run(runs, "001_A_subset", [_row("a", 3, 3.0)])
    (runs / "cache" / "masks").mkdir(parents=True)
    assert [d.name for d in comp.select_run_dirs(runs)] == ["001_A", "002_B"]
    assert [d.name for d in comp.select_run_dirs(runs, scope="subset")] == ["001_A_subset"]
    assert [d.name for d in comp.select_run_dirs(runs, run_names=["002_B", "nope"])] == ["002_B"]


@pytest.mark.parametrize("n_runs", [2, 3, 5])
def test_disagreement_panels_for_any_run_count(tmp_path, n_runs):
    runs = tmp_path / "runs"
    for i in range(n_runs):
        _write_run(runs, f"00{i}_run", [_row("img", 10 + 5 * i, 100.0 + i)], overlay_ids=["img"])
    out = comp.generate_cross_run_comparison(runs)
    panels = list((out / "disagreement_panels").glob("*.png"))
    assert len(panels) == 1
    assert cv2.imread(str(panels[0])) is not None


def test_panel_write_failure_raises(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for i in range(2):
        _write_run(runs, f"00{i}_run", [_row("img", 10 + i, 100.0)], overlay_ids=["img"])
    monkeypatch.setattr(comp.cv2, "imwrite", lambda *a, **k: False)
    with pytest.raises(IOError, match="imwrite"):
        comp.generate_cross_run_comparison(runs)


def test_report_has_no_canned_conclusions_and_states_run_count(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "001_x", [_row("a", 10, 100.0), _row("b", 20, 200.0)])
    _write_run(runs, "002_y", [_row("a", 30, 300.0)])
    _write_run(runs, "003_z", [_row("a", 12, 120.0), _row("b", 22, 220.0)])
    out = comp.generate_cross_run_comparison(runs)
    text = (out / "comparison_report.md").read_text()
    assert "3 run(s)" in text
    for canned in ("B_sensitive", "C_conservative", "aggressively fragments", "autofluorescence", "D_3d"):
        assert canned not in text
    assert "`002_y`" in text and "Highest mean count: `002_y`" in text
    assert "`b` missing from: 002_y" in text


def test_report_text_follows_data(tmp_path):
    def make(counts):
        runs = tmp_path / f"runs_{counts[0]}"
        _write_run(runs, "001_x", [_row("a", counts[0], 1.0)])
        _write_run(runs, "002_y", [_row("a", counts[1], 1.0)])
        return (comp.generate_cross_run_comparison(runs) / "comparison_report.md").read_text()

    assert "Highest mean count: `001_x`" in make((50, 10))
    assert "Highest mean count: `002_y`" in make((10, 50))
