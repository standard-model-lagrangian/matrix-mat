"""
Cross-run comparison: delta quantification, multi-run side-by-side panels, and failure mode analysis.

Run selection is explicit: pass `run_names`, or the default picks every completed run directory of one
scope (full runs, or `*_subset` runs) and logs what was included and excluded. An image missing from a
run is NaN for that run and does not take part in that image's spread; images present in fewer than
two runs are not ranked.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger("matrix_mat.if_density.comparator")

SUBSET_SUFFIX = "_subset"


def run_scope(run_name: str) -> str:
    """'subset' for run directories named '*_subset', otherwise 'full'."""
    return "subset" if run_name.endswith(SUBSET_SUFFIX) else "full"


def select_run_dirs(
    runs_dir: Path,
    run_names: Optional[Sequence[str]] = None,
    scope: str = "full",
) -> List[Path]:
    """
    Choose the run directories to compare.
    - run_names given: exactly those (missing or incomplete ones are logged and dropped).
    - otherwise: every subdirectory with features/per_image.csv whose scope matches `scope`
      ('full' excludes '*_subset' runs, 'subset' keeps only them, 'any' keeps both).
    """
    runs_dir = Path(runs_dir)
    if run_names is not None:
        selected = []
        for name in run_names:
            d = runs_dir / name
            if (d / "features" / "per_image.csv").exists():
                selected.append(d)
            else:
                logger.warning(f"Requested run '{name}' has no features/per_image.csv in {runs_dir}; dropped")
        logger.info(f"Comparing explicitly requested runs: {[d.name for d in selected]}")
        return selected

    candidates = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "features" / "per_image.csv").exists()
    ) if runs_dir.exists() else []
    if scope == "any":
        selected = candidates
    else:
        selected = [d for d in candidates if run_scope(d.name) == scope]
    excluded = [d.name for d in candidates if d not in selected]
    logger.info(f"Comparing {len(selected)} {scope}-scope run(s): {[d.name for d in selected]}")
    if excluded:
        logger.info(f"Excluded run(s) of another scope: {excluded}")
    return selected


def _load_csvs(run_dirs: Sequence[Path], rel: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for r in run_dirs:
        f = r / rel
        if f.exists():
            out[r.name] = pd.read_csv(f)
    return out


def load_run_summaries(runs_dir: Path, run_names: Optional[Sequence[str]] = None, scope: str = "full") -> Dict[str, pd.DataFrame]:
    """Load summary_by_material.csv from the selected runs."""
    return _load_csvs(select_run_dirs(runs_dir, run_names, scope), "summary/summary_by_material.csv")


def load_run_per_images(runs_dir: Path, run_names: Optional[Sequence[str]] = None, scope: str = "full") -> Dict[str, pd.DataFrame]:
    """Load per_image.csv from the selected runs."""
    return _load_csvs(select_run_dirs(runs_dir, run_names, scope), "features/per_image.csv")


def load_run_qc_flags(runs_dir: Path, run_names: Optional[Sequence[str]] = None, scope: str = "full") -> Dict[str, pd.DataFrame]:
    """Load qc_flags.csv from the selected runs."""
    return _load_csvs(select_run_dirs(runs_dir, run_names, scope), "qc/qc_flags.csv")


def compute_per_image_deltas(per_images: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-image counts/densities side by side over the union of image ids.
    Missing image in a run -> NaN for that run, excluded from the spread. count_spread, density_spread
    and relative_spread (CV) use only runs that have the image (n_runs_present); images present in fewer
    than two runs get NaN spreads and rank NaN. Rows are sorted by count_spread (desc), unranked last.
    """
    run_names = list(per_images.keys())
    for r_name, p_df in per_images.items():
        if "nuclei_per_mm2" not in p_df.columns:
            raise ValueError(f"{r_name}: " + "per-image table has no nuclei_per_mm2 column (pre-0.2 run with pseudo-volumetric density); re-run it before comparing")

    indexed = {r: df.drop_duplicates("image_id").set_index("image_id") for r, df in per_images.items()}
    all_img_ids = sorted(set().union(*[set(df.index) for df in indexed.values()])) if indexed else []

    delta_rows = []
    for img_id in all_img_ids:
        mat = next(df.loc[img_id, "material"] for df in indexed.values() if img_id in df.index)
        row: Dict[str, Any] = {"image_id": img_id, "material": mat}
        counts: List[float] = []
        densities: List[float] = []
        flags_by_run = []
        missing_in = []
        for r_name in run_names:
            df = indexed[r_name]
            if img_id in df.index:
                rec = df.loc[img_id]
                c = float(rec["n_nuclei"])
                d = float(rec["nuclei_per_mm2"])
                fl = str(rec["flag_summary"])
                counts.append(c)
                if np.isfinite(d):
                    densities.append(d)
                if fl != "clean":
                    flags_by_run.append(f"{r_name}:{fl}")
            else:
                c, d = float("nan"), float("nan")
                missing_in.append(r_name)
            row[f"{r_name}_count"] = c
            row[f"{r_name}_density"] = round(d, 2) if np.isfinite(d) else float("nan")

        n_present = len(counts)
        row["n_runs_present"] = n_present
        if n_present >= 2:
            row["count_spread"] = max(counts) - min(counts)
            mean_c = float(np.mean(counts))
            row["relative_spread"] = round(float(np.std(counts, ddof=1) / mean_c), 3) if mean_c > 0 else float("nan")
        else:
            row["count_spread"] = float("nan")
            row["relative_spread"] = float("nan")
        row["density_spread"] = round(max(densities) - min(densities), 2) if len(densities) >= 2 else float("nan")
        row["active_flags"] = "; ".join(flags_by_run) if flags_by_run else "clean"
        row["missing_in_runs"] = ";".join(missing_in)
        delta_rows.append(row)
        if missing_in:
            logger.warning(f"Image '{img_id}' missing from run(s) {missing_in}; excluded from its spread")

    deltas_df = pd.DataFrame(delta_rows)
    if deltas_df.empty:
        return deltas_df
    deltas_df = deltas_df.sort_values(
        by=["count_spread", "image_id"], ascending=[False, True], na_position="last", kind="mergesort"
    ).reset_index(drop=True)
    ranked = deltas_df["n_runs_present"] >= 2
    deltas_df["rank"] = np.where(ranked, np.cumsum(ranked), np.nan)
    n_unranked = int((~ranked).sum())
    if n_unranked:
        logger.warning(f"{n_unranked} image(s) present in fewer than 2 runs are not ranked")
    return deltas_df


def _tile_grid(panels: List[np.ndarray], labels: List[str], tile: int = 512) -> np.ndarray:
    """Arrange any number of panels in a near-square grid of tile x tile cells, each labelled."""
    n = len(panels)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    grid = np.zeros((nrows * tile, ncols * tile, 3), dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if panel.shape[:2] != (tile, tile):
            panel = cv2.resize(panel, (tile, tile), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, ncols)
        cell = panel.copy()
        cv2.putText(cell, label, (10, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, label, (10, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        grid[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = cell
    return grid


def write_disagreement_panels(
    run_dirs: Dict[str, Path],
    deltas_df: pd.DataFrame,
    panels_dir: Path,
    top_n: int = 10,
) -> List[Path]:
    """Side-by-side overlays (one tile per run) for the top_n ranked images. Needs >= 2 runs."""
    written: List[Path] = []
    if len(run_dirs) < 2 or deltas_df.empty:
        return written
    ranked = deltas_df[deltas_df["n_runs_present"] >= 2].head(top_n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        img_id = row["image_id"]
        panels, labels = [], []
        for r_name, r_dir in run_dirs.items():
            overlay_file = r_dir / "masks" / f"{img_id}_overlay.png"
            panel = cv2.imread(str(overlay_file)) if overlay_file.exists() else None
            if panel is None:
                panel = np.zeros((512, 512, 3), dtype=np.uint8)
                labels.append(f"{r_name} (missing)")
            else:
                labels.append(r_name)
            panels.append(panel)
        grid = _tile_grid(panels, labels)
        out_panel_path = panels_dir / f"rank{rank:02d}_{img_id}_disagreement.png"
        if not cv2.imwrite(str(out_panel_path), grid):
            raise IOError(f"cv2.imwrite failed to write disagreement panel {out_panel_path}")
        written.append(out_panel_path)
    return written


def generate_cross_run_comparison(
    runs_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    run_names: Optional[Sequence[str]] = None,
    scope: str = "full",
) -> Path:
    """
    Generate comparison outputs (metrics_by_run.csv, per_image_deltas.csv, disagreement panels,
    comparison_report.md) for the selected runs; see select_run_dirs for how runs are chosen.
    """
    runs_dir = Path(runs_dir)
    comp_dir = Path(output_dir) if output_dir else runs_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = comp_dir / "disagreement_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    run_dirs_list = [d for d in select_run_dirs(runs_dir, run_names, scope) if d.resolve() != comp_dir.resolve()]
    run_dirs = {d.name: d for d in run_dirs_list}
    summaries = _load_csvs(run_dirs_list, "summary/summary_by_material.csv")
    per_images = _load_csvs(run_dirs_list, "features/per_image.csv")
    qc_flags = _load_csvs(run_dirs_list, "qc/qc_flags.csv")

    if not per_images:
        logger.warning(f"No run outputs found in {runs_dir}")
        return comp_dir

    run_names_used = list(per_images.keys())
    logger.info(f"Comparing runs: {run_names_used}")
    if len(run_names_used) < 2:
        logger.warning("Fewer than 2 runs selected; spreads and panels need at least 2 runs")

    # 1. metrics_by_run.csv: per-material metrics side-by-side
    mat_rows = []
    all_materials = set()
    for s_df in summaries.values():
        all_materials.update(s_df["material"].tolist())

    for mat in sorted(all_materials):
        row: Dict[str, Any] = {"material": mat}
        for r_name in run_names_used:
            s_df = summaries.get(r_name, pd.DataFrame())
            m_sub = s_df[s_df["material"] == mat] if not s_df.empty else pd.DataFrame()
            if not m_sub.empty:
                rec = m_sub.iloc[0]
                if "mean_nuclei_per_mm2_all" not in rec.index:
                    raise ValueError(f"{r_name}: summary_by_material.csv lacks nuclei_per_mm2 columns; re-run with matrix-mat >= 0.2")
                row[f"{r_name}_density_all"] = rec["mean_nuclei_per_mm2_all"]
                row[f"{r_name}_density_clean"] = rec["mean_nuclei_per_mm2_clean"]
                row[f"{r_name}_median_nuc_vol"] = rec["median_nuclear_volume_um3"]
                row[f"{r_name}_flag_rate"] = rec["flagged_image_rate"]
            else:
                row[f"{r_name}_density_all"] = np.nan
        mat_rows.append(row)

    metrics_df = pd.DataFrame(mat_rows)
    metrics_path = comp_dir / "metrics_by_run.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved {metrics_path}")

    # 2. per_image_deltas.csv
    deltas_df = compute_per_image_deltas(per_images)
    deltas_path = comp_dir / "per_image_deltas.csv"
    deltas_df.to_csv(deltas_path, index=False)
    logger.info(f"Saved {deltas_path}")

    # 3. Top-10 highest-disagreement images: one tile per run
    write_disagreement_panels(run_dirs, deltas_df, panels_dir, top_n=10)

    # 4. comparison_report.md
    report_path = comp_dir / "comparison_report.md"
    _write_comparison_report(
        report_path=report_path,
        run_names=run_names_used,
        metrics_df=metrics_df,
        deltas_df=deltas_df,
        qc_flags=qc_flags,
        per_images=per_images,
    )
    logger.info(f"Generated cross-run comparison report: {report_path}")

    return comp_dir


def _fmt(x: Any, spec: str = ".1f") -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "n/a" if not np.isfinite(xf) else format(xf, spec)


def _write_comparison_report(
    report_path: Path,
    run_names: List[str],
    metrics_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    qc_flags: Dict[str, pd.DataFrame],
    per_images: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Markdown comparison report. Every statement is computed from the tables; no canned conclusions."""
    per_images = per_images or {}
    n_runs = len(run_names)
    ranked = deltas_df[deltas_df["n_runs_present"] >= 2] if not deltas_df.empty else deltas_df
    unranked = deltas_df[deltas_df["n_runs_present"] < 2] if not deltas_df.empty else deltas_df

    lines = [
        "# Cross-Run Segmentation Comparison",
        "",
        "## Summary",
        f"Compared **{n_runs} run(s)** across **{len(deltas_df)} fields of view** "
        f"({len(ranked)} present in at least 2 runs and ranked):",
    ]
    for r in run_names:
        lines.append(f"- `{r}`")

    # Per-run descriptive table, computed
    lines.extend([
        "",
        "## 1. Per-Run Totals",
        "",
        "| Run | Fields | Mean nuclei/field | Mean nuclei/mm² | Flagged fields (%) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    run_means: Dict[str, float] = {}
    for r in run_names:
        p = per_images.get(r, pd.DataFrame())
        q = qc_flags.get(r, pd.DataFrame())
        n = len(p)
        mean_c = float(p["n_nuclei"].mean()) if n else float("nan")
        mean_d = float(p["nuclei_per_mm2"].mean()) if n else float("nan")
        flag_pct = 100.0 * float(q["flag_any"].astype(bool).mean()) if (not q.empty and "flag_any" in q.columns) else float("nan")
        run_means[r] = mean_c
        lines.append(f"| `{r}` | {n} | {_fmt(mean_c)} | {_fmt(mean_d)} | {_fmt(flag_pct)} |")

    finite = {k: v for k, v in run_means.items() if np.isfinite(v)}
    if len(finite) >= 2:
        hi = max(finite, key=lambda k: finite[k])
        lo = min(finite, key=lambda k: finite[k])
        lines.extend([
            "",
            f"Highest mean count: `{hi}` ({finite[hi]:.1f} nuclei/field); lowest: `{lo}` ({finite[lo]:.1f}). "
            "Mean counts are over the fields each run contains.",
        ])

    if not ranked.empty:
        mat_grouped = ranked.groupby("material")["count_spread"].agg(["mean", "max", "count"]).reset_index()
        mat_grouped = mat_grouped.sort_values(by="mean", ascending=False)
        lines.extend([
            "",
            "## 2. Count Spread by Material",
            "Spread = max − min nuclei count over the runs that contain the field.",
            "",
            "| Material | Images | Mean Count Spread | Max Count Spread |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for _, row in mat_grouped.iterrows():
            lines.append(f"| `{row['material']}` | {int(row['count'])} | {row['mean']:.1f} | {row['max']:.0f} |")

        lines.extend([
            "",
            "## 3. Top-5 Highest-Disagreement Images",
            "",
            "| Rank | Image ID | Material | Runs present | Count Spread | Density Spread (nuclei/mm²) | Active Flags |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for _, row in ranked.head(5).iterrows():
            lines.append(
                f"| {int(row['rank'])} | `{row['image_id']}` | {row['material']} | {int(row['n_runs_present'])} | "
                f"{_fmt(row['count_spread'], '.0f')} | {_fmt(row['density_spread'])} | {row['active_flags']} |"
            )

    if not deltas_df.empty and (deltas_df["missing_in_runs"] != "").any():
        lines.extend(["", "## 4. Missing Images", f"{len(unranked)} image(s) are in fewer than 2 runs and unranked.", ""])
        for _, row in deltas_df[deltas_df["missing_in_runs"] != ""].iterrows():
            lines.append(f"- `{row['image_id']}` missing from: {row['missing_in_runs'].replace(';', ', ')}")

    identical = _identical_run_pairs(per_images)
    if identical:
        lines.extend(["", "## Identical Runs", ""])
        for a, b in identical:
            lines.append(f"- `{a}` and `{b}` produced identical per-image counts and densities.")

    if n_runs >= 2:
        lines.extend([
            "",
            f"Side-by-side overlays ({n_runs} tiles each) of the top-10 ranked images are in `disagreement_panels/`.",
        ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _identical_run_pairs(per_images: Dict[str, pd.DataFrame]) -> List[tuple]:
    """Pairs of runs whose per-image n_nuclei and nuclei_per_mm2 are identical on all shared images."""
    names = list(per_images.keys())
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da = per_images[a].set_index("image_id")
            db = per_images[b].set_index("image_id")
            common = sorted(set(da.index) & set(db.index))
            if not common or len(common) != len(da) or len(common) != len(db):
                continue
            ca, cb = da.loc[common], db.loc[common]
            if (ca["n_nuclei"].values == cb["n_nuclei"].values).all() and np.allclose(
                ca["nuclei_per_mm2"].values.astype(float), cb["nuclei_per_mm2"].values.astype(float), equal_nan=True
            ):
                out.append((a, b))
                logger.warning(f"Runs '{a}' and '{b}' produced identical per-image outputs")
    return out
