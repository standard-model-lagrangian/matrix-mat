"""
CLI entry point for multi-run Cellpose-SAM sweep, ablation analysis, and cross-run comparison.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import numpy as np
import tifffile
import yaml

from matrix_mat.core.cache import MaskCache, compute_cache_key
from matrix_mat.if_density.config import IFSweepConfig
from matrix_mat.reporting.biological_plots import generate_all_if_biological_figures
from matrix_mat.if_density.comparator import generate_cross_run_comparison
from matrix_mat.if_density.comparator_prepost import run_comparator_prepost
from matrix_mat.if_density.engine import CPSAMEngine, setup_run_logger
from matrix_mat.if_density.features import (
    compute_areal_density,
    compute_spheroid_extent,
    extract_objects_and_image_features,
    save_features_csvs,
)
from matrix_mat.if_density.pairing import discover_and_pair_fields
from matrix_mat.if_density.postprocess import postprocess_field_masks
from matrix_mat.if_density.preflight import audit_if_dataset
from matrix_mat.if_density.preprocess import preprocess_field
from matrix_mat.if_density.provenance import write_manifest_csv, write_run_metadata
from matrix_mat.if_density.qc import (
    apply_cohort_qc_flags,
    compute_image_qc_metrics,
    deduplicate_masks,
    export_qc_flags_csv,
)
from matrix_mat.reporting.summaries import compute_summary_by_material
from matrix_mat.if_density.visualize import save_mask_and_overlay, save_qc_overlay_panels

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("matrix_mat.if_density.cli")

SUBSET_FIELD_IDS: List[str] = [
    "SKOV3 Spheroid D7 Mor_S40D30-Gel2-1",
    "SKOV3 Spheroid D7 Mor_S34D30-Gel1-1",
    "SKOV3 Spheroid D7 Mor_S34D30-Gel8-1-2",
    "SKOV3 Spheroid D7 Mor_Mat-Gel1-3",
    "SKOV3 Spheroid D7 Mor_S43D20-Gel2-1-2",
]


def discover_inputs(
    config: Dict[str, Any],
    scope: str = "full",
    allow_unpaired: bool = True,
    engine: Optional[Any] = None,
) -> Tuple[List[Any], List[str], Any]:
    """Preflights dataset integrity and discovers/pairs IF field channels."""
    data_dir = Path(config["data_dir"])
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("pretrained_model", "models/cpsam_v2")
    random_seed = int(config.get("random_seed", 42))
    pixel_size_default = float(config.get("default_pixel_size_um", 0.505049))

    if engine is None:
        preflight = audit_if_dataset(data_dir=data_dir, model_path=model_path, allow_unpaired=allow_unpaired)
        logger.info("\n" + preflight.format_summary())
        preflight.log_and_raise_if_fatal()

    logger.info(f"Scanning data directory: {data_dir}")
    field_records, unpaired_bases = discover_and_pair_fields(
        data_dir=data_dir,
        allow_unpaired=allow_unpaired,
        default_pixel_size_um=pixel_size_default,
    )

    if scope == "subset":
        field_records = [r for r in field_records if r.field_id in SUBSET_FIELD_IDS]
        logger.info(f"Scope is SUBSET: filtered to {len(field_records)} high-disagreement fields.")
    else:
        logger.info(f"Scope is FULL: processing {len(field_records)} fields.")

    if engine is None:
        engine = CPSAMEngine(model_path=model_path, random_seed=random_seed)

    return field_records, unpaired_bases, engine


def segment_field(
    rec: Any,
    combo: Dict[str, Any],
    preproc_cfg: Dict[str, Any],
    postproc_cfg: Dict[str, Any],
    qc_cfg: Dict[str, Any],
    feat_cfg: Dict[str, Any],
    masks_dir: Path,
    preproc_dir: Path,
    runs_dir: Path,
    overwrite: bool,
    engine: Any,
    field_idx: int,
    run_name: str,
) -> Dict[str, Any]:
    """Processes a single IF field through preprocessing, segmentation, postprocessing, and QC."""
    field_id = rec.field_id
    nuclear_img = tifffile.imread(rec.nuclear_path)
    actin_img = tifffile.imread(rec.actin_path)

    preproc_res = preprocess_field(
        image_id=field_id,
        nuclear_img=nuclear_img,
        actin_img=actin_img,
        pixel_size_um=rec.pixel_size_um,
        preprocess_config=preproc_cfg,
        output_dir=preproc_dir if preproc_cfg.get("enabled", False) else None,
        save_montage=(field_idx <= 5),
    )
    preproc_meta = {
        "p_low_val": preproc_res.nuclear.p_low_val,
        "p_high_val": preproc_res.nuclear.p_high_val,
        "bg_radius_px": preproc_res.nuclear.bg_radius_px,
        "bg_radius_um": preproc_res.nuclear.bg_radius_um,
    }

    proc_nuclear_u8 = preproc_res.nuclear.preprocessed_u8
    proc_actin_u8 = preproc_res.actin.preprocessed_u8

    mask_cache = MaskCache(cache_dir=runs_dir / "cache" / "masks")
    cache_key = compute_cache_key(proc_nuclear_u8, combo)

    if not overwrite and mask_cache.has(cache_key):
        logger.info(f"[{field_id}] Reusing cached raw segmentation for key {cache_key[:12]}")
        raw_masks = mask_cache.load(cache_key)
    else:
        raw_masks, mode_used = engine.segment_image(proc_nuclear_u8, combo, image_id=field_id)
        mask_cache.save(cache_key, raw_masks)

    final_masks, border_labels, post_stats = postprocess_field_masks(
        image_id=field_id,
        raw_masks=raw_masks,
        nuclear_img=proc_nuclear_u8,
        pixel_size_um=rec.pixel_size_um,
        postprocess_config=postproc_cfg,
    )

    mask_path, overlay_path = save_mask_and_overlay(
        image_id=field_id,
        nuclear_img=proc_nuclear_u8,
        actin_img=proc_actin_u8,
        masks=final_masks,
        masks_dir=masks_dir,
        run_name=run_name,
    )

    sph_extent = compute_spheroid_extent(
        proc_actin_u8,
        final_masks,
        rec.pixel_size_um,
        method=feat_cfg.get("spheroid_extent_method", "actin_threshold"),
        actin_background_percentile=float(feat_cfg.get("actin_background_percentile", 50.0)),
    )
    sph_vol_um3 = sph_extent[0]
    max_r_um = sph_extent[3]
    sph_area_um2 = np.pi * (max_r_um ** 2) if (max_r_um is not None and not np.isnan(max_r_um) and max_r_um > 0) else float("nan")
    density_areal = compute_areal_density(post_stats.n_final_included, sph_area_um2)
    density_vol = (float(post_stats.n_final_included) / sph_vol_um3 * 1e9) if (sph_vol_um3 is not None and not np.isnan(sph_vol_um3) and sph_vol_um3 > 0) else float("nan")

    raw_nuc_img = tifffile.imread(rec.nuclear_path)
    raw_act_img = tifffile.imread(rec.actin_path)

    qc_metric = compute_image_qc_metrics(
        image_id=field_id,
        material=rec.material,
        replicate=rec.replicate,
        nuclear_img=raw_nuc_img,
        actin_img=raw_act_img,
        raw_masks=raw_masks,
        dedup_masks=final_masks,
        n_dups_removed=post_stats.n_dedup_removed,
        pixel_size_um=rec.pixel_size_um,
        qc_config=qc_cfg,
        nuclei_density_per_mm3=density_vol,
        nuclei_per_mm2=density_areal,
        n_nuclei_valid=post_stats.n_final_included,
        n_border_excluded=post_stats.n_border_excluded,
    )

    return {
        "field_id": field_id,
        "qc_metric": qc_metric,
        "post_stats": post_stats,
        "preproc_meta": preproc_meta,
        "border_labels": border_labels,
        "sph_extent": sph_extent,
        "overlay_path": overlay_path,
    }


def apply_qc(
    raw_qc_metrics: List[Any],
    qc_cfg: Dict[str, Any],
    qc_dir: Path,
) -> List[Any]:
    """Applies cohort-relative QC flags and writes qc_flags.csv."""
    final_qc_metrics = apply_cohort_qc_flags(raw_qc_metrics, qc_cfg)
    export_qc_flags_csv(final_qc_metrics, qc_dir / "qc_flags.csv")
    return final_qc_metrics


def extract_features(
    field_records: List[Any],
    masks_dir: Path,
    final_qc_metrics: List[Any],
    border_labels_by_id: Dict[str, Set[int]],
    spheroid_extents_by_id: Dict[str, Any],
    feat_cfg: Dict[str, Any],
    features_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts per-object and per-image morphological and intensity features."""
    all_objects: List[Dict[str, Any]] = []
    per_image_rows: List[Dict[str, Any]] = []
    qc_lookup = {m.image_id: m for m in final_qc_metrics}

    for rec in field_records:
        field_id = rec.field_id
        nuclear_img = tifffile.imread(rec.nuclear_path)
        actin_img = tifffile.imread(rec.actin_path)
        mask_path = masks_dir / f"{field_id}_mask.tif"
        masks = tifffile.imread(mask_path).astype(np.int32)

        qc_m = qc_lookup[field_id]
        b_labels = border_labels_by_id.get(field_id, set())
        sph_ext = spheroid_extents_by_id.get(field_id)

        obj_rows, per_img_row = extract_objects_and_image_features(
            image_id=field_id,
            material=rec.material,
            replicate=rec.replicate,
            nuclear_img=nuclear_img,
            actin_img=actin_img,
            dedup_masks=masks,
            pixel_size_um=rec.pixel_size_um,
            qc_flags_list=qc_m.flags_list,
            feature_config=feat_cfg,
            border_excluded_labels=b_labels,
            spheroid_extent=sph_ext,
        )
        all_objects.extend(obj_rows)
        per_image_rows.append(per_img_row)

    save_features_csvs(all_objects, per_image_rows, features_dir)
    return all_objects, per_image_rows


def compute_stats(
    per_image_rows: List[Dict[str, Any]],
    summary_dir: Path,
) -> None:
    """Computes condition-level summary statistics across hydrogels."""
    compute_summary_by_material(per_image_rows, summary_dir / "summary_by_material.csv")


def render_reports(
    final_qc_metrics: List[Any],
    overlay_paths: Dict[str, Path],
    qc_dir: Path,
    run_dir: Path,
    slug: str,
) -> None:
    """Saves worst-field QC overlay panels and publication biological figures."""
    flagged_images = [
        {"image_id": m.image_id, "flags": m.flags_list, "overlay_path": overlay_paths[m.image_id]}
        for m in final_qc_metrics if m.flag_any and m.image_id in overlay_paths
    ]
    save_qc_overlay_panels(flagged_images[:5], qc_dir)

    try:
        generate_all_if_biological_figures(run_dir=run_dir)
    except Exception as e:
        logger.warning(f"Failed to generate biological figures for {slug}: {e}")


def _write_combo_provenance(
    run_dir: Path,
    combo: Dict[str, Any],
    preproc_cfg: Dict[str, Any],
    postproc_cfg: Dict[str, Any],
    per_image_preproc_meta: Dict[str, Any],
    per_image_postproc_stats: Dict[str, Any],
    pre_debris_or_merged_count: int,
    post_debris_or_merged_count: int,
    unpaired_bases: List[str],
    random_seed: int,
    elapsed: float,
) -> None:
    """Builds and writes comprehensive run_metadata.json for a sweep run."""
    from matrix_mat.if_density.postprocess import (
        REFERENCE_NUCLEAR_VOLUME_UM3,
        DEFAULT_MIN_VOL_UM3,
        DEFAULT_MAX_VOL_UM3,
    )
    post_enabled = bool(postproc_cfg.get("enabled", False))
    pre_enabled = bool(preproc_cfg.get("enabled", False))
    bg_cfg = preproc_cfg.get("background", {})
    norm_cfg = preproc_cfg.get("normalize", {})
    denoise_cfg = preproc_cfg.get("denoise", {})
    size_cfg = postproc_cfg.get("size_gate", {})
    split_cfg = postproc_cfg.get("splitting", {})
    border_cfg = postproc_cfg.get("border_exclusion", {})

    ref_vol = float(
        size_cfg.get("reference_vol_um3")
        or postproc_cfg.get("reference_vol_um3")
        or REFERENCE_NUCLEAR_VOLUME_UM3
    )

    preproc_metadata = {
        "enabled": pre_enabled,
        "methods": [
            m for m in [
                f"background_subtraction_{bg_cfg.get('method', 'none')}" if bg_cfg.get("method", "none") != "none" and pre_enabled else None,
                "normalize_percentile" if norm_cfg.get("enabled", True) and pre_enabled else None,
                f"denoise_{denoise_cfg.get('method', 'none')}" if denoise_cfg.get("method", "none") != "none" and pre_enabled else None,
            ] if m is not None
        ],
        "resolved_parameters": {
            "expected_nuclear_diameter_um": float(preproc_cfg.get("expected_nuclear_diameter_um", 11.7)),
            "background": {
                "method": bg_cfg.get("method", "none"),
                "radius_multiplier": float(bg_cfg.get("radius_multiplier", 3.5)),
                "downsample_factor": int(bg_cfg.get("downsample_factor", 2)),
            },
            "normalize": {
                "enabled": bool(norm_cfg.get("enabled", True)),
                "p_low": float(norm_cfg.get("p_low", 1.0)),
                "p_high": float(norm_cfg.get("p_high", 99.8)),
            },
            "denoise": {
                "method": denoise_cfg.get("method", "none"),
                "sigma_um": float(denoise_cfg.get("sigma_um", 0.0)),
            },
        },
        "per_image_percentiles": per_image_preproc_meta,
    }

    postproc_metadata = {
        "enabled": post_enabled,
        "methods": [
            m for m in [
                "deduplicate" if postproc_cfg.get("deduplicate", {}).get("iou_threshold", 0) > 0 and post_enabled else None,
                "size_gate" if size_cfg.get("enabled", True) and post_enabled else None,
                f"split_merged_{split_cfg.get('method', 'seeded_watershed')}" if split_cfg.get("enabled", True) and post_enabled else None,
                "border_exclusion" if border_cfg.get("enabled", True) and post_enabled else None,
            ] if m is not None
        ],
        "resolved_parameters": {
            "reference_nuclear_volume_um3": ref_vol,
            "deduplicate": {
                "iou_threshold": float(postproc_cfg.get("deduplicate", {}).get("iou_threshold", 0.50)),
            },
            "size_gate": {
                "enabled": bool(size_cfg.get("enabled", True)),
                "min_vol_um3": float(size_cfg.get("min_vol_um3", DEFAULT_MIN_VOL_UM3)),
                "max_vol_um3": float(size_cfg.get("max_vol_um3", DEFAULT_MAX_VOL_UM3)),
                "vol_multiplier_min": 0.25,
                "vol_multiplier_max": 4.0,
            },
            "splitting": {
                "enabled": bool(split_cfg.get("enabled", True)),
                "method": split_cfg.get("method", "seeded_watershed"),
                "split_factor": float(split_cfg.get("split_factor", 2.0)),
                "h_maxima": float(split_cfg.get("h_maxima", 10.0)),
                "min_distance_um": float(split_cfg.get("min_distance_um", 4.0)),
            },
            "border_exclusion": {
                "enabled": bool(border_cfg.get("enabled", True)),
                "clearance_um": float(border_cfg.get("clearance_um", 5.85)),
            },
        },
        "per_image_stats": per_image_postproc_stats,
        "cohort_deltas": {
            "debris_or_merged_pre_count": pre_debris_or_merged_count,
            "debris_or_merged_post_count": post_debris_or_merged_count,
            "resolved_count": max(0, pre_debris_or_merged_count - post_debris_or_merged_count),
        },
    }

    write_run_metadata(
        run_dir=run_dir,
        resolved_params=combo,
        random_seed=random_seed,
        wall_time_sec=elapsed,
        extra_info={"unpaired_skipped": unpaired_bases},
        preprocessing_info=preproc_metadata,
        postprocessing_info=postproc_metadata,
    )


def run_single_combo(
    combo_idx: int,
    combo: Dict[str, Any],
    config: Dict[str, Any],
    field_records: List[Any],
    unpaired_bases: List[str],
    engine: Any,
    runs_dir: Path,
    scope: str,
    random_seed: int,
    pixel_size_default: float,
    overwrite: bool,
) -> None:
    """Executes a single parameter or pre/post-processing combo in the sweep."""
    raw_slug = combo.get("slug", f"{combo_idx:03d}_{combo.get('name', 'run')}")
    if scope == "subset":
        if raw_slug.startswith("10") and not raw_slug.startswith("101s") and not raw_slug.startswith("102s") and not raw_slug.startswith("103s") and not raw_slug.startswith("104s"):
            slug = raw_slug[:3] + "s_" + raw_slug[4:]
        elif not raw_slug.endswith("_subset") and not "s_" in raw_slug:
            slug = f"{raw_slug}_subset"
        else:
            slug = raw_slug
    else:
        slug = raw_slug

    run_name = combo.get("name", slug)
    run_dir = runs_dir / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    preproc_cfg = combo.get("preprocessing", config.get("preprocessing", {}))
    postproc_cfg = combo.get("postprocessing", config.get("postprocessing", {}))
    qc_cfg = config.get("qc", {})
    feat_cfg = config.get("features", {})

    log_handler = setup_run_logger(run_dir)
    logger.info(f"\n{'='*70}\nStarting Run: {slug} ({combo.get('description', '')})\n{'='*70}")
    start_time = time.time()

    run_config = {
        "run_slug": slug, "run_name": run_name, "scope": scope, "parameters": combo,
        "preprocessing": preproc_cfg, "postprocessing": postproc_cfg, "qc": qc_cfg,
        "features": feat_cfg, "random_seed": random_seed, "pixel_size_um": pixel_size_default,
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)

    write_manifest_csv(run_dir, field_records)

    masks_dir, qc_dir, features_dir = run_dir / "masks", run_dir / "qc", run_dir / "features"
    summary_dir, preproc_dir = run_dir / "summary", run_dir / "preprocessed"

    raw_qc_metrics, overlay_paths = [], {}
    per_image_preproc_meta, per_image_postproc_stats = {}, {}
    border_labels_by_id, spheroid_extents_by_id = {}, {}
    pre_debris_or_merged_count, post_debris_or_merged_count = 0, 0

    for field_idx, rec in enumerate(field_records, start=1):
        res = segment_field(
            rec=rec, combo=combo, preproc_cfg=preproc_cfg, postproc_cfg=postproc_cfg,
            qc_cfg=qc_cfg, feat_cfg=feat_cfg, masks_dir=masks_dir, preproc_dir=preproc_dir,
            runs_dir=runs_dir, overwrite=overwrite, engine=engine, field_idx=field_idx, run_name=run_name,
        )
        fid, qm, ps = res["field_id"], res["qc_metric"], res["post_stats"]
        raw_qc_metrics.append(qm)
        overlay_paths[fid] = res["overlay_path"]
        per_image_preproc_meta[fid] = res["preproc_meta"]
        border_labels_by_id[fid] = res["border_labels"]
        spheroid_extents_by_id[fid] = res["sph_extent"]
        per_image_postproc_stats[fid] = {
            "n_raw": ps.n_raw, "n_dedup_removed": ps.n_dedup_removed,
            "n_size_gate_removed_small": ps.n_size_gate_removed_small,
            "n_size_gate_removed_large": ps.n_size_gate_removed_large,
            "n_split_candidates": ps.n_split_candidates, "n_split_accepted": ps.n_split_accepted,
            "n_split_reverted": ps.n_split_reverted, "n_children_created": ps.n_children_created,
            "n_border_excluded": ps.n_border_excluded, "n_final_total": ps.n_final_total,
            "n_final_included": ps.n_final_included,
        }
        if (qm.debris_count_pre > 0) or (qm.merged_count_pre > 0):
            pre_debris_or_merged_count += 1
        if qm.flag_debris_or_merged:
            post_debris_or_merged_count += 1

    final_qc_metrics = apply_qc(raw_qc_metrics, qc_cfg, qc_dir)
    all_objects, per_image_rows = extract_features(
        field_records=field_records, masks_dir=masks_dir, final_qc_metrics=final_qc_metrics,
        border_labels_by_id=border_labels_by_id, spheroid_extents_by_id=spheroid_extents_by_id,
        feat_cfg=feat_cfg, features_dir=features_dir,
    )
    compute_stats(per_image_rows, summary_dir)
    render_reports(final_qc_metrics, overlay_paths, qc_dir, run_dir, slug)

    elapsed = time.time() - start_time
    _write_combo_provenance(
        run_dir=run_dir, combo=combo, preproc_cfg=preproc_cfg, postproc_cfg=postproc_cfg,
        per_image_preproc_meta=per_image_preproc_meta, per_image_postproc_stats=per_image_postproc_stats,
        pre_debris_or_merged_count=pre_debris_or_merged_count,
        post_debris_or_merged_count=post_debris_or_merged_count,
        unpaired_bases=unpaired_bases, random_seed=random_seed, elapsed=elapsed,
    )
    logger.info(f"Run {slug} finished in {elapsed:.1f}s.")
    logger.removeHandler(log_handler)
    log_handler.close()


def run_sweep(
    config_path: str | Path,
    allow_unpaired_override: bool = False,
    overwrite_override: bool = False,
    scope: str = "full",
    engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """Execute sweep runs defined in config with optional subset scope and pre/post-processing."""
    config_obj = IFSweepConfig.from_yaml(config_path)
    config = config_obj.model_dump()

    runs_dir = Path(config["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    allow_unpaired = allow_unpaired_override or config.get("allow_unpaired", True)
    overwrite = overwrite_override or config.get("overwrite", False)
    random_seed = int(config.get("random_seed", 42))
    pixel_size_default = float(config.get("default_pixel_size_um", 0.505049))

    combos = config.get("combos", [])
    if not combos:
        raise ValueError("No combos defined in configuration file.")

    field_records, unpaired_bases, engine = discover_inputs(
        config=config, scope=scope, allow_unpaired=allow_unpaired, engine=engine
    )

    logger.info(f"Beginning sweep across {len(combos)} configurations on {len(field_records)} fields.")

    for combo_idx, combo in enumerate(combos, start=1):
        run_single_combo(
            combo_idx=combo_idx,
            combo=combo,
            config=config,
            field_records=field_records,
            unpaired_bases=unpaired_bases,
            engine=engine,
            runs_dir=runs_dir,
            scope=scope,
            random_seed=random_seed,
            pixel_size_default=pixel_size_default,
            overwrite=overwrite,
        )

    # Trigger appropriate comparison
    if "prepost" in str(config_path).lower() or "robustness" in str(config_path).lower():
        logger.info("\n" + "="*70 + "\nExecuting Pre/Post Cross-Run Comparison\n" + "="*70)
        run_comparator_prepost(runs_dir=runs_dir)
    else:
        logger.info("\n" + "="*70 + "\nExecuting Phase 5 Cross-Run Comparison\n" + "="*70)
        generate_cross_run_comparison(runs_dir=runs_dir)

    logger.info("Sweep execution and comparison complete.")
    return {"status": "SUCCESS", "runs_dir": runs_dir}


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-run Cellpose-SAM (cpsam) IF segmentation sweep, ablation, and comparison pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: sweep
    sweep_parser = subparsers.add_parser("sweep", help="Execute hyperparameter or pre/post sweep.")
    sweep_parser.add_argument("--config", type=str, default="configs/sweep.yaml", help="Path to YAML configuration.")
    sweep_parser.add_argument("--scope", type=str, choices=["subset", "full"], default="full", help="Evaluation scope: 'subset' (top-5 disagreement fields) or 'full' (all 27 fields).")
    sweep_parser.add_argument("--allow-unpaired", action="store_true", help="Ignore unpaired files.")
    sweep_parser.add_argument("--overwrite", action="store_true", help="Force recomputation of existing runs.")

    # Subcommand: compare
    compare_parser = subparsers.add_parser("compare", help="Compare completed original runs.")
    compare_parser.add_argument("--runs", type=str, default="runs", help="Path to runs directory.")
    compare_parser.add_argument("--output", type=str, default=None, help="Path to comparison output directory.")

    # Subcommand: compare-prepost
    compare_pp_parser = subparsers.add_parser("compare-prepost", help="Compare pre/post ablation and robustness runs.")
    compare_pp_parser.add_argument("--runs", type=str, default="runs", help="Path to runs directory.")
    compare_pp_parser.add_argument("--output", type=str, default=None, help="Path to comparison output directory.")

    # Subcommand: compare-biological
    comp_bio_parser = subparsers.add_parser("compare-biological", help="Generate publication-grade biological comparison figures across hydrogels.")
    comp_bio_parser.add_argument("--run-dir", type=str, default="runs/104_PP11_full", help="Path to run directory containing features/ and qc/.")
    comp_bio_parser.add_argument("--output", type=str, default=None, help="Directory to save biological figures (default: <run_dir>/figures).")

    args = parser.parse_args()

    if args.command == "sweep":
        run_sweep(
            config_path=args.config,
            allow_unpaired_override=args.allow_unpaired,
            overwrite_override=args.overwrite,
            scope=args.scope,
        )
    elif args.command == "compare":
        generate_cross_run_comparison(runs_dir=args.runs, output_dir=args.output)
    elif args.command == "compare-prepost":
        run_comparator_prepost(runs_dir=args.runs, output_dir=args.output)
    elif args.command == "compare-biological":
        figs = generate_all_if_biological_figures(run_dir=args.run_dir, output_dir=args.output)
        print(f"[+] Generated biological comparison figures in {args.output or Path(args.run_dir) / 'figures'}")


if __name__ == "__main__":
    main()
