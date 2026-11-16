"""
Cross-run comparison and verdict generation for Pre/Post-processing Ablation and Robustness sweeps.
Generates:
1. runs/comparison_prepost/ablation_effects.csv
2. runs/comparison_prepost/effect_summary.md
3. runs/comparison_prepost/robustness_vs_original.csv
4. runs/comparison_prepost/panels/ (preprocessing montages and split overlays for top 5 disagreement images)
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage import segmentation

logger = logging.getLogger("matrix_mat.if_density.comparator_prepost")

SEEDED_DISAGREEMENT_IDS = [
    "SKOV3 Spheroid D7 Mor_S40D30-Gel2-1",
    "SKOV3 Spheroid D7 Mor_S34D30-Gel1-1",
    "SKOV3 Spheroid D7 Mor_S34D30-Gel8-1-2",
    "SKOV3 Spheroid D7 Mor_Mat-Gel1-3",
    "SKOV3 Spheroid D7 Mor_S43D20-Gel2-1-2",
]


ARMS = ("PP00", "PP10", "PP01", "PP11")
_ARM_PATTERN = re.compile(r"(?<![A-Za-z0-9])(PP00|PP10|PP01|PP11)(?![0-9])")
ORIGINAL_RUNS = ["001_A_baseline", "002_B_sensitive", "003_C_conservative", "004_D_3d"]
ROBUSTNESS_RUNS = ["201_A_baseline_pp", "202_B_sensitive_pp", "203_C_conservative_pp", "204_D_3d_pp"]
FIELD_LEVEL_SD_NOTE = (
    "descriptive: mean and SD over fields (field-level); fields of one gel are not independent replicates"
)


def ablation_arm(slug: str) -> Optional[str]:
    """'PP00' | 'PP10' | 'PP01' | 'PP11' when the slug names an ablation arm as a token, else None."""
    m = _ARM_PATTERN.search(slug)
    return m.group(1) if m else None


def slug_scope(slug: str) -> str:
    """Run scope from the directory name: '*_subset' is a subset run, anything else a full run."""
    return "subset" if slug.endswith("_subset") else "full"


def find_arm_dirs(runs_dir: Path) -> Dict[Tuple[str, str], Path]:
    """
    Map (scope, arm) -> run directory for completed ablation runs. Several directories for the same
    scope and arm is ambiguous and raises ValueError listing them.
    """
    found: Dict[Tuple[str, str], List[Path]] = {}
    if not runs_dir.exists():
        return {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        arm = ablation_arm(d.name)
        if arm is None:
            continue
        found.setdefault((slug_scope(d.name), arm), []).append(d)
    dup = {k: v for k, v in found.items() if len(v) > 1}
    if dup:
        detail = "; ".join(f"{s}/{a}: {', '.join(x.name for x in v)}" for (s, a), v in sorted(dup.items()))
        raise ValueError(f"Multiple run directories match the same ablation arm ({detail}); keep one per arm")
    return {k: v[0] for k, v in found.items()}


def load_run_metadata(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compute_cv_spread(values: List[float]) -> float:
    """
    Compute Coefficient of Variation (CV = sample std / mean) as a scale-invariant spread metric.
    If all values are scaled by a constant factor c > 0, CV is invariant: CV(c * X) == CV(X).
    Undefined (NaN) for fewer than 2 finite values or a non-positive mean; callers exclude NaN.
    """
    arr = np.asarray([v for v in (values or []) if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    mean_val = float(np.mean(arr))
    if mean_val <= 1e-9:
        return float("nan")
    return float(np.std(arr, ddof=1) / mean_val)



def generate_ablation_effects_csv(
    runs_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Compile per-image and per-material metrics across ablation runs:
    PP00_none_none, PP10_bg_only, PP01_post_only, PP11_full (both subset and full).
    Includes per-image rows and per-material mean +- SD density per combo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    arm_dirs = find_arm_dirs(runs_dir)
    for (scope, arm) in sorted(arm_dirs):
        logger.info(f"Ablation arm {arm} ({scope}): {arm_dirs[(scope, arm)].name}")

    for (scope, arm), r_dir in sorted(arm_dirs.items()):
        slug = r_dir.name
        meta = load_run_metadata(r_dir)
        post_stats = meta.get("postprocessing", {}).get("per_image_stats", {})

        qc_csv = r_dir / "qc" / "qc_flags.csv"
        features_csv = r_dir / "features" / "per_image.csv"

        if not qc_csv.exists() or not features_csv.exists():
            continue

        qc_df = pd.read_csv(qc_csv)
        feat_df = pd.read_csv(features_csv)

        merged = pd.merge(qc_df, feat_df, on=["image_id", "material", "replicate"], suffixes=("_qc", "_feat"))

        for _, r in merged.iterrows():
            fld = str(r["image_id"])
            p_stat = post_stats.get(fld, {})
            r_dens_raw = r.get("nuclei_per_mm2_feat", r.get("nuclei_per_mm2"))
            if r_dens_raw is None:
                raise ValueError(f"{slug}: " + "per-image table has no nuclei_per_mm2 column (pre-0.2 run with pseudo-volumetric density); re-run it before comparing")
            r_dens = float(r_dens_raw)
            is_flagged = bool(r.get("flag_any", False))

            rows.append({
                "image_id": fld,
                "combo_slug": slug,
                "scope": scope,
                "material": r["material"],
                "replicate": r["replicate"],
                "n_nuclei_raw": r.get("n_nuclei_raw", r.get("n_nuclei_total", r.get("n_nuclei_dedup", 0))),
                "n_nuclei_final": r.get("n_nuclei", 0),
                "nuclei_per_mm2": r_dens,
                "density_mean_sd": f"{r_dens:.1f}",
                "median_nuclear_volume_um3": r.get("median_nuclear_volume_um3", 0.0),
                "flag_any": is_flagged,
                "flag_rate": 1.0 if is_flagged else 0.0,
                "flags_list": str(r.get("flags_list", "")),
                "flag_debris_or_merged": r.get("flag_debris_or_merged", False),
                "flag_density_outlier": r.get("flag_density_outlier", False),
                "flag_count_outlier": r.get("flag_count_outlier", False),
                "n_dedup_removed": p_stat.get("n_dedup_removed", r.get("n_duplicates_removed", 0)),
                "n_size_gate_removed_small": p_stat.get("n_size_gate_removed_small", 0),
                "n_size_gate_removed_large": p_stat.get("n_size_gate_removed_large", 0),
                "n_split_candidates": p_stat.get("n_split_candidates", 0),
                "n_split_accepted": p_stat.get("n_split_accepted", 0),
                "n_border_excluded": p_stat.get("n_border_excluded", r.get("n_border_excluded", 0)),
                "arm": arm,
                "sd_level": "",
            })

    # Per-material mean +- SD density per combo (descriptive, over fields)
    if rows:
        base_df = pd.DataFrame(rows)
        summary_rows = []
        for (scope, slug, mat), grp in base_df.groupby(["scope", "combo_slug", "material"], sort=False):
            mean_dens = float(grp["nuclei_per_mm2"].mean())
            std_dens = float(grp["nuclei_per_mm2"].std(ddof=1)) if len(grp) > 1 else float("nan")
            mean_final = float(grp["n_nuclei_final"].mean())
            mean_raw = float(grp["n_nuclei_raw"].mean())
            mean_vol = float(grp["median_nuclear_volume_um3"].mean())
            flag_pct = float(grp["flag_any"].mean())

            summary_rows.append({
                "image_id": f"SUMMARY_{mat}",
                "combo_slug": slug,
                "scope": scope,
                "material": mat,
                "replicate": "ALL",
                "n_nuclei_raw": round(mean_raw, 1),
                "n_nuclei_final": round(mean_final, 1),
                "nuclei_per_mm2": round(mean_dens, 2),
                "density_mean_sd": f"{mean_dens:.1f} ± {std_dens:.1f}",
                "median_nuclear_volume_um3": round(mean_vol, 2),
                "flag_any": bool(grp["flag_any"].any()),
                "flag_rate": round(flag_pct, 3),
                "flags_list": ";".join(sorted(set(f for fl in grp["flags_list"] for f in str(fl).split(";") if f))),
                "flag_debris_or_merged": int(grp["flag_debris_or_merged"].sum()),
                "flag_density_outlier": int(grp["flag_density_outlier"].sum()),
                "flag_count_outlier": int(grp["flag_count_outlier"].sum()),
                "n_dedup_removed": int(grp["n_dedup_removed"].sum()),
                "n_size_gate_removed_small": int(grp["n_size_gate_removed_small"].sum()),
                "n_size_gate_removed_large": int(grp["n_size_gate_removed_large"].sum()),
                "n_split_candidates": int(grp["n_split_candidates"].sum()),
                "n_split_accepted": int(grp["n_split_accepted"].sum()),
                "n_border_excluded": int(grp["n_border_excluded"].sum()),
                "arm": grp["arm"].iloc[0],
                "sd_level": FIELD_LEVEL_SD_NOTE,
            })

        all_rows = rows + summary_rows
    else:
        all_rows = []

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved ablation effects ({len(df)} rows, including per-material summaries) to {output_path}")

    # Also save dedicated per-material pivot table
    if rows:
        mat_table = []
        for (scope, mat), grp_m in base_df.groupby(["scope", "material"]):
            row_dict = {"scope": scope, "material": mat, "n_images": len(grp_m["image_id"].unique()),
                        "sd_level": FIELD_LEVEL_SD_NOTE}
            for slug, grp_s in grp_m.groupby("combo_slug"):
                m_d = float(grp_s["nuclei_per_mm2"].mean())
                s_d = float(grp_s["nuclei_per_mm2"].std(ddof=1)) if len(grp_s) > 1 else float("nan")
                row_dict[f"{slug}_density_mean_sd"] = f"{m_d:.1f} ± {s_d:.1f}"
                row_dict[f"{slug}_mean_count"] = round(float(grp_s["n_nuclei_final"].mean()), 1)
            mat_table.append(row_dict)
        pd.DataFrame(mat_table).to_csv(output_path.parent / "ablation_material_density_summary.csv", index=False)

    return df


def _load_family(runs_dir: Path, names: List[str], scope: str, warnings: List[str]) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for r in names:
        name = f"{r}_subset" if scope == "subset" else r
        f = runs_dir / name / "features" / "per_image.csv"
        if f.exists():
            d = pd.read_csv(f)
            if "nuclei_per_mm2" not in d.columns:
                raise ValueError(f"{name}: " + "per-image table has no nuclei_per_mm2 column (pre-0.2 run with pseudo-volumetric density); re-run it before comparing")
            dfs[name] = d.drop_duplicates("image_id").set_index("image_id")
        else:
            msg = f"Run '{name}' not found in {runs_dir}; excluded from the robustness comparison"
            logger.warning(msg)
            warnings.append(msg)
    return dfs


def find_identical_runs(dfs: Dict[str, pd.DataFrame], images: List[str]) -> List[Tuple[str, str]]:
    """Pairs of runs whose n_nuclei and nuclei_per_mm2 are identical on every given image."""
    names = list(dfs)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not images:
                continue
            ca, cb = dfs[a].loc[images], dfs[b].loc[images]
            if (ca["n_nuclei"].values == cb["n_nuclei"].values).all() and np.allclose(
                ca["nuclei_per_mm2"].values.astype(float), cb["nuclei_per_mm2"].values.astype(float), equal_nan=True
            ):
                pairs.append((a, b))
    return pairs


def _r3(x: float) -> float:
    return round(float(x), 3) if np.isfinite(x) else float("nan")


def _red_pct(orig: float, pp: float) -> float:
    if not (np.isfinite(orig) and np.isfinite(pp)) or orig <= 1e-12:
        return float("nan")
    return round(100.0 * (orig - pp) / orig, 1)


def generate_robustness_vs_original_csv(
    runs_dir: Path,
    output_path: Path,
    scope: str = "full",
) -> pd.DataFrame:
    """
    Compare the original runs (001-004) with the pre/post-processed runs (201-204): per-image CV of the
    count and density across each family's runs, summarised per material.

    - Only images present in every available run of BOTH families are compared.
    - A missing run is warned about and excluded (no KeyError); a family needs >= 2 runs for a CV.
    - CV is NaN when undefined and excluded from means; values are rounded to 3 decimals.
    - Runs with identical outputs (e.g. 004_D_3d == 001_A_baseline on 2D data) are reported.
    Warnings are returned in DataFrame.attrs['warnings'] for the effect summary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    warns: List[str] = []
    orig_dfs = _load_family(runs_dir, ORIGINAL_RUNS, scope, warns)
    pp_dfs = _load_family(runs_dir, ROBUSTNESS_RUNS, scope, warns)

    if not orig_dfs or not pp_dfs:
        logger.warning("Missing original or robustness runs for comparison.")
        empty = pd.DataFrame()
        empty.attrs["warnings"] = warns
        return empty
    for fam, dfs in (("original", orig_dfs), ("robustness", pp_dfs)):
        if len(dfs) < 2:
            msg = f"Only {len(dfs)} {fam} run(s) available; the cross-parameter CV needs at least 2"
            logger.warning(msg)
            warns.append(msg)

    all_dfs = {**orig_dfs, **pp_dfs}
    common_images = sorted(set.intersection(*[set(d.index) for d in all_dfs.values()]))
    dropped = sorted(set.union(*[set(d.index) for d in all_dfs.values()]) - set(common_images))
    if dropped:
        msg = f"{len(dropped)} image(s) not present in every run were excluded: {', '.join(dropped[:10])}"
        logger.warning(msg)
        warns.append(msg)

    for fam_dfs in (orig_dfs, pp_dfs):
        for a, b in find_identical_runs(fam_dfs, common_images):
            msg = (f"Runs '{a}' and '{b}' produced identical outputs on all {len(common_images)} common images; "
                   "they are not independent parameter settings (on 2D data the 3D/stitch slot equals baseline)")
            logger.warning(msg)
            warns.append(msg)

    first = next(iter(orig_dfs.values()))
    image_rows = []
    for img_id in common_images:
        image_rows.append({
            "image_id": img_id,
            "material": first.loc[img_id, "material"],
            "orig_count_spread": compute_cv_spread([d.loc[img_id, "n_nuclei"] for d in orig_dfs.values()]),
            "pp_count_spread": compute_cv_spread([d.loc[img_id, "n_nuclei"] for d in pp_dfs.values()]),
            "orig_density_spread": compute_cv_spread([d.loc[img_id, "nuclei_per_mm2"] for d in orig_dfs.values()]),
            "pp_density_spread": compute_cv_spread([d.loc[img_id, "nuclei_per_mm2"] for d in pp_dfs.values()]),
        })
    img_df = pd.DataFrame(image_rows, columns=["image_id", "material", "orig_count_spread", "pp_count_spread",
                                               "orig_density_spread", "pp_density_spread"])

    def _summary(label: str, group: pd.DataFrame) -> Dict[str, Any]:
        valid = group[np.isfinite(group["orig_count_spread"]) & np.isfinite(group["pp_count_spread"])]
        o_mean, p_mean = float(valid["orig_count_spread"].mean()), float(valid["pp_count_spread"].mean())
        o_max, p_max = float(valid["orig_count_spread"].max()), float(valid["pp_count_spread"].max())
        valid_d = group[np.isfinite(group["orig_density_spread"]) & np.isfinite(group["pp_density_spread"])]
        o_d, p_d = float(valid_d["orig_density_spread"].mean()), float(valid_d["pp_density_spread"].mean())
        return {
            "material": label,
            "n_images": int(len(valid)),
            "orig_mean_count_spread": _r3(o_mean),
            "orig_max_count_spread": _r3(o_max),
            "pp_mean_count_spread": _r3(p_mean),
            "pp_max_count_spread": _r3(p_max),
            "spread_reduction_mean_pct": _red_pct(o_mean, p_mean),
            "spread_reduction_max_pct": _red_pct(o_max, p_max),
            "orig_mean_density_spread": _r3(o_d),
            "pp_mean_density_spread": _r3(p_d),
            "density_spread_reduction_pct": _red_pct(o_d, p_d),
        }

    mat_rows = [_summary(mat, group) for mat, group in img_df.groupby("material")]
    mat_rows.append(_summary("ALL_COHORT", img_df))

    mat_df = pd.DataFrame(mat_rows)
    mat_df.to_csv(output_path, index=False)
    mat_df.attrs["warnings"] = warns
    mat_df.attrs["orig_runs"] = list(orig_dfs)
    mat_df.attrs["pp_runs"] = list(pp_dfs)
    logger.info(f"Saved robustness comparison to {output_path}")
    return mat_df


def generate_panels(
    runs_dir: Path,
    output_dir: Path,
) -> None:
    """Generate before/after preprocessing and split-mask overlays for top 5 disagreement images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference run folders, found by ablation arm (full scope preferred, then subset)
    arm_dirs = find_arm_dirs(runs_dir)
    pp11_dir = arm_dirs.get(("full", "PP11")) or arm_dirs.get(("subset", "PP11")) or runs_dir / "104_PP11_full"
    pp00_dir = arm_dirs.get(("full", "PP00")) or arm_dirs.get(("subset", "PP00")) or runs_dir / "001_A_baseline"
    pre_methods = load_run_metadata(pp11_dir).get("preprocessing", {}).get("methods", [])
    pre_label = " + ".join(pre_methods) if pre_methods else "no preprocessing recorded"

    for img_id in SEEDED_DISAGREEMENT_IDS:
        # 1. Preprocessing before / after montage
        preproc_tif = pp11_dir / "preprocessed" / f"{img_id}_nuclear.tif"
        raw_mask_tif = pp00_dir / "masks" / f"{img_id}_mask.tif"
        final_mask_tif = pp11_dir / "masks" / f"{img_id}_mask.tif"

        # Locate raw nuclear file from manifest
        manifest_p = pp11_dir / "manifest.csv"
        if not manifest_p.exists():
            manifest_p = runs_dir / "001_A_baseline" / "manifest.csv"

        raw_nuc = None
        proc_nuc = None

        if manifest_p.exists():
            m_df = pd.read_csv(manifest_p)
            match = m_df[m_df["field_id"] == img_id]
            if not match.empty:
                raw_nuc_path = Path(str(match.iloc[0]["nuclear_path"]))
                if raw_nuc_path.exists() and preproc_tif.exists():
                    raw_nuc = tifffile.imread(raw_nuc_path)
                    proc_nuc = tifffile.imread(preproc_tif)

                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    fig.patch.set_facecolor("black")
                    ax[0].imshow(raw_nuc, cmap="magma")
                    ax[0].set_title(f"Raw Input (min={raw_nuc.min()}, max={raw_nuc.max()})", color="white")
                    ax[0].axis("off")

                    ax[1].imshow(proc_nuc, cmap="magma")
                    ax[1].set_title(f"Preprocessed ({pre_label})", color="white")
                    ax[1].axis("off")

                    plt.suptitle(f"Preprocessing Impact: {img_id}", color="white", fontsize=14)
                    plt.tight_layout()
                    out_p = output_dir / f"{img_id}_preproc_montage.png"
                    plt.savefig(out_p, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
                    plt.close(fig)

        # 2. Mask overlay comparing raw vs post-processed with microscopy image backdrop
        if raw_mask_tif.exists() and final_mask_tif.exists():
            raw_m = tifffile.imread(raw_mask_tif)
            final_m = tifffile.imread(final_mask_tif)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            fig.patch.set_facecolor("black")

            # Boundaries
            raw_bound = segmentation.find_boundaries(raw_m > 0, mode="thick")
            final_bound = segmentation.find_boundaries(final_m > 0, mode="thick")

            bg_raw = raw_nuc if raw_nuc is not None else (raw_m > 0).astype(float)
            bg_proc = proc_nuc if proc_nuc is not None else (final_m > 0).astype(float)

            ax[0].imshow(bg_raw, cmap="gray")
            ax[0].imshow(np.ma.masked_where(~raw_bound, raw_bound), cmap="autumn", alpha=0.85)
            ax[0].set_title(f"Raw Baseline Mask (Count={int(raw_m.max())})", color="white")
            ax[0].axis("off")

            ax[1].imshow(bg_proc, cmap="gray")
            ax[1].imshow(np.ma.masked_where(~final_bound, final_bound), cmap="spring", alpha=0.85)
            ax[1].set_title(f"Post-processed Mask (Count={int(final_m.max())})", color="white")
            ax[1].axis("off")

            plt.suptitle(f"Segmentation Mask Refinement: {img_id}", color="white", fontsize=14)
            plt.tight_layout()
            out_mask_p = output_dir / f"{img_id}_split_overlay.png"
            plt.savefig(out_mask_p, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)


def generate_effect_summary_md(
    ablation_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write effect_summary.md detailing main effects, interactions, failure mode resolutions, and verdict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"scope", "image_id", "combo_slug"}
    if ablation_df is None or ablation_df.empty or not required.issubset(ablation_df.columns):
        output_path.write_text(
            "# Pre/Post-Processing Effect Summary\n\n"
            "No PP00/PP10/PP01/PP11 ablation runs were found in this runs directory, so no main effects "
            "or interactions can be estimated. Run `configs/prepost_ablation.yaml` into the same runs "
            "directory to populate this report.\n",
            encoding="utf-8",
        )
        logger.warning(f"No ablation runs found; wrote placeholder effect summary to {output_path}")
        return

    # Filter to image-level rows for full scope
    full_df = ablation_df[
        (ablation_df["scope"] == "full") & (~ablation_df["image_id"].astype(str).str.startswith("SUMMARY_"))
    ]
    if full_df.empty:
        full_df = ablation_df[~ablation_df["image_id"].astype(str).str.startswith("SUMMARY_")]

    # Map each arm to exactly one run (slug); several runs for one arm is ambiguous.
    slugs_by_arm: Dict[str, List[str]] = {}
    for slug in sorted(full_df["combo_slug"].astype(str).unique()):
        arm = ablation_arm(slug)
        if arm:
            slugs_by_arm.setdefault(arm, []).append(slug)
    ambiguous = {a: s for a, s in slugs_by_arm.items() if len(s) > 1}
    if ambiguous:
        raise ValueError(f"Multiple runs for the same ablation arm: {ambiguous}")
    arm_slug = {a: s[0] for a, s in slugs_by_arm.items()}
    missing_arms = [a for a in ARMS if a not in arm_slug]

    # Quantitative metrics per arm
    counts: Dict[str, float] = {}
    flags: Dict[str, float] = {}
    debris: Dict[str, int] = {}
    low_snr: Dict[str, int] = {}
    for arm, slug in arm_slug.items():
        grp = full_df[full_df["combo_slug"] == slug]
        counts[arm] = float(grp["n_nuclei_final"].mean())
        flags[arm] = float(grp["flag_any"].astype(bool).mean()) * 100.0
        debris[arm] = int(grp["flag_debris_or_merged"].astype(bool).sum())
        low_snr[arm] = int(grp["flags_list"].astype(str).str.contains("low_snr").sum())

    def have(*arms: str) -> bool:
        return all(a in arm_slug for a in arms)

    def not_available(*arms: str) -> str:
        miss = [a for a in arms if a not in arm_slug]
        return f"- Not available: ablation arm(s) {', '.join(miss)} not found in this runs directory."

    def pct(new: float, old: float) -> str:
        return f"{(new - old) / old * 100.0:+.1f}%" if old > 1e-6 else "n/a"

    def effect_lines(arm: str, ref: str = "PP00") -> List[str]:
        if not have(arm, ref):
            return [not_available(arm, ref)]
        c_d, f_d, b_d = counts[arm] - counts[ref], flags[arm] - flags[ref], debris[arm] - debris[ref]
        return [
            f"- **Quantitative Deltas (`{arm}` vs `{ref}`)**:",
            f"  - Delta Nuclei Count: **{c_d:+.1f}** nuclei/field ({pct(counts[arm], counts[ref])})",
            f"  - Delta Overall Flag Rate: **{f_d:+.1f}** percentage points (from {flags[ref]:.1f}% to {flags[arm]:.1f}%)",
            f"  - `debris_or_merged` Flagged Images: **{debris[ref]} -> {debris[arm]}** ({b_d:+d})",
            f"  - `low_snr` Flagged Images: **{low_snr[ref]} -> {low_snr[arm]}** ({low_snr[arm] - low_snr[ref]:+d})",
        ]

    # Robustness metrics
    rob_warnings: List[str] = list(getattr(robustness_df, "attrs", {}).get("warnings", []))
    c_red_mean = c_red_max = orig_mean_spread = pp_mean_spread = float("nan")
    if not robustness_df.empty and "material" in robustness_df.columns:
        all_cohort_row = robustness_df[robustness_df["material"] == "ALL_COHORT"]
        if not all_cohort_row.empty:
            c_red_mean = float(all_cohort_row.iloc[0]["spread_reduction_mean_pct"])
            c_red_max = float(all_cohort_row.iloc[0]["spread_reduction_max_pct"])
            orig_mean_spread = float(all_cohort_row.iloc[0]["orig_mean_count_spread"])
            pp_mean_spread = float(all_cohort_row.iloc[0]["pp_mean_count_spread"])
    robustness_evaluated = not np.isnan(c_red_mean)
    n_fields = int(full_df["image_id"].nunique()) if not full_df.empty else 0

    if robustness_evaluated:
        robustness_summary = [
            f"- **Robustness**: mean cross-parameter count CV changed from **{orig_mean_spread:.3f}** to "
            f"**{pp_mean_spread:.3f}** ({c_red_mean:+.1f}% reduction; max {c_red_max:+.1f}%).",
        ]
    else:
        robustness_summary = [
            "- **Robustness**: not evaluated (original 001–004 and robustness 201–204 runs were not both found "
            "in this runs directory, or no image had a defined spread).",
        ]

    lines = [
        "# Pre/Post-Processing Ablation Comparison",
        "",
        "## Executive Summary",
        f"Computed comparison of the 2×2 pre/post-processing ablation across {n_fields} fields. "
        "All numbers below are calculated from the run outputs; interpretation is left to `docs/findings/`.",
        "",
        *robustness_summary,
    ]
    if missing_arms:
        lines.append(f"- **Missing ablation arms**: {', '.join(missing_arms)}; effects that need them are reported as not available.")
    lines.extend([
        "",
        "---",
        "",
        "## 1. 2×2 Factorial Ablation Analysis",
        "",
        "| Configuration | Preprocessing | Post-processing | Mean Nuclei Count | Flagged Images (%) | Debris/Merged Flags |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for arm in ARMS:
        if arm not in arm_slug:
            lines.append(f"| *{arm}: not available* | - | - | - | - | - |")
            continue
        is_pre = "BG Sub + Norm" if arm in ("PP10", "PP11") else "Off"
        is_post = "Size + Split + Border" if arm in ("PP01", "PP11") else "Off"
        lines.append(f"| `{arm_slug[arm]}` | {is_pre} | {is_post} | {counts[arm]:.1f} | {flags[arm]:.1f}% | {debris[arm]} |")
    lines.extend([
        "",
        f"Per-material mean ± SD in `ablation_effects.csv` and `ablation_material_density_summary.csv` are "
        f"{FIELD_LEVEL_SD_NOTE}.",
        "",
        "### Main Effect of Preprocessing (Background Subtraction + Normalization)",
        *effect_lines("PP10"),
        "",
        "### Main Effect of Post-Processing (Size Gate + Splitting + Border Exclusion)",
        *effect_lines("PP01"),
    ])
    if have("PP01", "PP00"):
        lines.append(
            "  - Note: the size gate removes the very objects that raise `debris_or_merged`, so its drop under "
            "post-processing is expected by construction and is not evidence of better segmentation."
        )
    lines.extend(["", "### Combined Pre + Post (`PP11`)", *effect_lines("PP11")])
    if have(*ARMS):
        lines.append(
            f"  - Interaction (count): **{(counts['PP11'] - counts['PP01']) - (counts['PP10'] - counts['PP00']):+.1f}** "
            "nuclei/field (effect of preprocessing with post-processing on, minus without)"
        )
    else:
        lines.append("  - Interaction: not available (needs all four arms).")

    def f3(x: Any) -> str:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return str(x)
        return format(xf, ".3f") if np.isfinite(xf) else "n/a"

    def f1(x: Any) -> str:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return str(x)
        return f"{xf:.1f}%" if np.isfinite(xf) else "n/a"

    orig_used = getattr(robustness_df, "attrs", {}).get("orig_runs", [])
    pp_used = getattr(robustness_df, "attrs", {}).get("pp_runs", [])
    lines.extend([
        "",
        "---",
        "",
        "## 2. Robustness Regression Test (Original CPSAM Configs vs the Same Configs with PP11)",
        "",
        "Per-material mean coefficient of variation (CV) of the per-field count and density across the "
        "runs of each family; images must be present in every run of both families.",
        f"Original runs used: {', '.join(f'`{r}`' for r in orig_used) or 'none'}; "
        f"robustness runs used: {', '.join(f'`{r}`' for r in pp_used) or 'none'}.",
        "",
        "| Material | Images | Orig Mean CV | PP Mean CV | Count CV Reduction (%) | Orig Density CV | PP Density CV | Density CV Reduction (%) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    if not robustness_df.empty and "material" in robustness_df.columns:
        for _, row in robustness_df.iterrows():
            mat = row["material"]
            bold = "**" if mat == "ALL_COHORT" else ""
            lines.append(
                f"| {bold}`{mat}`{bold} | {row['n_images']} | {f3(row['orig_mean_count_spread'])} | "
                f"{f3(row['pp_mean_count_spread'])} | {bold}{f1(row['spread_reduction_mean_pct'])}{bold} | "
                f"{f3(row['orig_mean_density_spread'])} | {f3(row['pp_mean_density_spread'])} | "
                f"{bold}{f1(row['density_spread_reduction_pct'])}{bold} |"
            )
    else:
        lines.append("| *(not evaluated)* | - | - | - | - | - | - | - |")
    if rob_warnings:
        lines.extend(["", "**Warnings:**", *[f"- {w}" for w in rob_warnings]])

    module_lines = []
    if have("PP11", "PP10"):
        module_lines.append(f"1. **Clump splitting**: count delta PP11 − PP10: {counts['PP11'] - counts['PP10']:+.1f} nuclei/field.")
    else:
        module_lines.append("1. **Clump splitting**: " + not_available("PP11", "PP10")[2:])
    if have("PP11", "PP00"):
        module_lines.append(f"2. **Background subtraction + size gating**: debris flag delta PP11 − PP00: {debris['PP11'] - debris['PP00']:+d}.")
    else:
        module_lines.append("2. **Background subtraction + size gating**: " + not_available("PP11", "PP00")[2:])
    module_lines.append("3. **Border exclusion**: border-strip nuclei excluded from the count and the strip excluded from the density area.")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Module Evaluation Summary",
        "",
        *module_lines,
        "",
        "*(Note: Narrative biological interpretation is archived in `docs/findings/` and is not auto-generated.)*",
        "",
        "---",
        "",
        "## 4. Recommendation",
        "",
        "Not generated automatically. Choosing a production configuration is an author decision; record it, "
        "with the numbers above, in `docs/findings/IF_PREPOST_FINDINGS.md`.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Saved effect summary report to {output_path}")


def run_comparator_prepost(
    runs_dir: str | Path = "runs",
    output_dir: Optional[str | Path] = None,
    scope: str = "full",
) -> Path:
    """Execute all pre/post comparison steps (robustness compares runs of the given scope)."""
    runs_dir = Path(runs_dir)
    comp_dir = Path(output_dir) if output_dir else runs_dir / "comparison_prepost"
    comp_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = comp_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # 1. ablation_effects.csv
    ablation_csv = comp_dir / "ablation_effects.csv"
    ablation_df = generate_ablation_effects_csv(runs_dir, ablation_csv)

    # 2. robustness_vs_original.csv
    robustness_csv = comp_dir / "robustness_vs_original.csv"
    robustness_df = generate_robustness_vs_original_csv(runs_dir, robustness_csv, scope=scope)

    # 3. panels
    generate_panels(runs_dir, panels_dir)

    # 4. effect_summary.md
    summary_md = comp_dir / "effect_summary.md"
    generate_effect_summary_md(ablation_df, robustness_df, summary_md)

    logger.info(f"Phase 5 pre/post comparison complete. Outputs saved in {comp_dir}")
    return comp_dir
