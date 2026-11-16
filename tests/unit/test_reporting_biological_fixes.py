"""biological_plots / summaries fixes (0.2.2)."""

import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import pandas as pd  # noqa: E402

from matrix_mat.reporting import biological_plots as bp  # noqa: E402
from matrix_mat.reporting import summaries  # noqa: E402


def _per_image():
    rows = []
    for m, flags in (("Mat", [False, False, True]), ("S50", [True, True, True])):
        for i, _f in enumerate(flags):
            rows.append({"image_id": f"{m}{i}", "material": m, "replicate": f"Gel{i}", "nuclei_per_mm2": 100.0 + 10 * i})
    qc = pd.DataFrame([{"image_id": r["image_id"], "flag_any": f}
                       for r, f in zip(rows, [False, False, True, True, True, True], strict=True)])
    return pd.DataFrame(rows), qc


def test_clean_only_bar_is_not_faked_when_no_clean_fields(tmp_path, monkeypatch):
    bars = []
    texts = []
    orig_bar = matplotlib.axes.Axes.bar
    orig_text = matplotlib.axes.Axes.text

    def spy_bar(self, x, height, *a, **k):
        bars.append((k.get("label"), list(height), list(k.get("yerr", []))))
        return orig_bar(self, x, height, *a, **k)

    def spy_text(self, x, y, s, *a, **k):
        texts.append(s)
        return orig_text(self, x, y, s, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)
    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: None)
    per_image, qc = _per_image()
    bp.plot_clean_vs_all_density(per_image, qc, tmp_path)
    clean = next(b for b in bars if b[0] and b[0].startswith("Clean"))
    assert np.isnan(clean[1][1])  # S50 has no clean field -> no bar, not the all-cohort mean
    assert any("n=0" in t for t in texts)


def test_jitter_is_seeded():
    src = inspect.getsource(bp)
    assert "np.random.normal" not in src


def test_summary_labels_match_test_statistic(tmp_path):
    rows = []
    for g, vals in enumerate([[1.0, 2.0, 9.0], [3.0, 4.0, 5.0], [6.0, 7.0, 100.0]]):
        for v in vals:
            rows.append({"image_id": f"Mat{g}{v}", "material": "Mat", "replicate": f"G{g}",
                         "nuclei_per_mm2": v, "flag_summary": "clean"})
    out = summaries.compute_summary_by_material(rows, tmp_path / "s.csv", control_material="Mat")
    # gel_mean column really is the mean of per-gel means (4, 4, 37.67) and is labelled descriptive.
    assert out["gel_mean_nuclei_per_mm2"].iloc[0] == pytest.approx((4.0 + 4.0 + 113.0 / 3) / 3)
    doc = summaries.__doc__.lower()
    assert "median" in doc and "averaged within each gel" not in doc and "descriptive" in doc
