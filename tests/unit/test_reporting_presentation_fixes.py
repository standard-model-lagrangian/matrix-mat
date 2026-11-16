"""presentation_plots fixes (0.2.2): NaN fold changes, all-NaN qc_reasons, fig6 y-limits, figure cleanup."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from matrix_mat.reporting import presentation_plots as pp  # noqa: E402


def _synthetic(qc_reasons_nan=False):
    rng = np.random.default_rng(0)
    objs, pairs = [], []
    for cond in ("Mat", "S50"):
        for g in (1, 2):
            for tp in ("t0", "t7"):
                for fov in (1, 2):
                    for _k in range(2):
                        objs.append({
                            "condition": cond, "replicate": f"gel{g}", "timepoint": tp, "fov": fov,
                            "qc_flag": "PASS", "qc_reasons": np.nan if qc_reasons_nan else "None",
                            "volume_sphere_um3": float(rng.uniform(1e5, 1e6)),
                        })
            for k in range(3):
                v0 = float(rng.uniform(1e5, 1e6))
                pairs.append({
                    "condition": cond, "replicate": f"gel{g}", "combined_qc_flag": "PASS",
                    "t0_volume_sphere_um3": v0, "t7_volume_sphere_um3": v0 * 1.5,
                    "fold_change_volume": 1.5 if k else np.nan,  # one NaN fold change per gel
                })
    return pd.DataFrame(objs), pd.DataFrame(pairs)


def test_all_nan_qc_reasons_column():
    objs, _ = _synthetic(qc_reasons_nan=True)
    assert objs["qc_reasons"].dtype == float
    counts = pp.get_effective_fov_counts(objs)
    assert (counts["n_fovs"] == 2).all()


def test_full_presentation_set_with_nan_fold_changes(tmp_path, monkeypatch):
    objs, pairs = _synthetic()
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: None)
    before = set(plt.get_fignums())
    figs = pp.generate_all_presentation_figures(tmp_path, objects_df=objs, pairs_df=pairs, condition_df=None)
    assert set(figs) == {f"fig{i}" for i in range(1, 7)}
    assert set(plt.get_fignums()) == before


def test_fig6_ylim_uses_concatenated_values(tmp_path, monkeypatch):
    objs, pairs = _synthetic()
    calls = []
    orig = matplotlib.axes.Axes.set_ylim

    def spy(self, *a, **k):
        calls.append((self, a))
        return orig(self, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylim", spy)
    wells = pp.compute_well_growth_comparison(objs)
    # One well's Day-7 volume is tiny: element-wise v0+v7 would hide it from the minimum.
    wells.loc[(wells.condition == "Mat") & (wells.replicate == "gel1"), "total_volume_um3_t7"] = 1e3
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: None)
    pp.generate_all_presentation_figures(tmp_path, objects_df=objs, pairs_df=pairs, wells_df=wells)
    lo, hi = [a for ax, a in calls if ax.get_title().startswith("Mat (N") and len(a) == 2][-1]
    assert lo == pytest.approx(max(1e-4, (1e3 / 1e6) * 0.5))


def test_figures_closed_when_generation_fails(tmp_path, monkeypatch):
    objs, pairs = _synthetic()
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    before = set(plt.get_fignums())
    with pytest.raises(OSError):
        pp.generate_all_presentation_figures(tmp_path, objects_df=objs, pairs_df=pairs)
    assert set(plt.get_fignums()) == before
