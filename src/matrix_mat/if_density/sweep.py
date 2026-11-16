"""
CLI entry point for multi-run Cellpose-SAM sweep, ablation analysis, and cross-run comparison.

Per combo (run): every field is read once, preprocessed, segmented (or taken from the mask cache),
post-processed, measured and QC'd; cohort QC flags are applied at the end. A run directory is complete
only when its completion marker (RUN_COMPLETE.json, written last) exists; complete runs whose config and
inputs are unchanged are skipped unless overwrite is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from matrix_mat.if_density.engine import CPSAMEngine, setup_run_logger, teardown_run_logger
from matrix_mat.if_density.features import (
    compute_areal_density,
    compute_spheroid_extent_details,
    density_area_um2,
    extract_objects_and_image_features,
    prepare_actin_for_extent,
    save_features_csvs,
)
from matrix_mat.if_density.pairing import discover_and_pair_fields_detailed
from matrix_mat.if_density.postprocess import postprocess_field_masks, resolve_size_gate_bounds
from matrix_mat.if_density.preflight import audit_if_dataset
from matrix_mat.if_density.preprocess import ensure_2d_grayscale, preprocess_field
from matrix_mat.if_density.provenance import write_manifest_csv, write_run_metadata
from matrix_mat.if_density.qc import (
    apply_cohort_qc_flags,
    compute_image_qc_metrics,
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
SUBSET_SUFFIX = "_subset"
COMPLETION_MARKER = "RUN_COMPLETE.json"
# Combo keys that change what Cellpose returns; everything else (slug, name, description, pre/post
# config) is excluded from the mask cache key so arms with identical segmentation inputs share masks.
CELLPOSE_PARAM_KEYS = ("flow_threshold", "cellprob_threshold", "diameter", "min_size", "stitch_threshold", "do_3D")
ARTIFACT_FLAGS = ("empty_seg", "dup_masks", "debris_or_merged", "edge_heavy", "saturated", "blurry", "low_snr")


def subset_slug(raw_slug: str) -> str:
    """Slug of the subset-scope run for a combo: '<slug>_subset' (unchanged if it already ends so)."""
    return raw_slug if raw_slug.endswith(SUBSET_SUFFIX) else f"{raw_slug}{SUBSET_SUFFIX}"


def resolve_run_slug(combo: Dict[str, Any], combo_idx: int, scope: str) -> str:
    raw_slug = combo.get("slug") or f"{combo_idx:03d}_{combo.get('name', 'run')}"
    return subset_slug(raw_slug) if scope == "subset" else raw_slug


def stratified_sample(field_records: Sequence[Any], n: int) -> List[Any]:
    """First n fields taken round-robin across materials (sorted), so a sample covers every material."""
    by_mat: Dict[str, List[Any]] = {}
    for r in sorted(field_records, key=lambda r: r.field_id):
        by_mat.setdefault(r.material, []).append(r)
    out: List[Any] = []
    queues = [list(v) for _, v in sorted(by_mat.items())]
    while len(out) < n and any(queues):
        for q in queues:
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def segmentation_cache_params(combo: Dict[str, Any], random_seed: int) -> Dict[str, Any]:
    """Only what changes the Cellpose output: its parameters and the seed."""
    params = {k: combo.get(k) for k in CELLPOSE_PARAM_KEYS}
    params["random_seed"] = int(random_seed)
    return params


def engine_model_hash(engine: Any) -> str:
    """SHA-256 of the engine's weights; engines without weights (test stubs) are identified by class."""
    sha = getattr(engine, "weights_sha256", None)
    if sha:
        return str(sha)
    return f"engine:{type(engine).__module__}.{type(engine).__qualname__}"


def load_field_images(rec: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Read a field's nuclear and actin channels once, as 2D arrays (RGB -> grayscale; stacks refused)."""
    nuc = ensure_2d_grayscale(tifffile.imread(rec.nuclear_path), rec.field_id)
    act = ensure_2d_grayscale(tifffile.imread(rec.actin_path), rec.field_id)
    if nuc.shape != act.shape:
        raise ValueError(f"{rec.field_id}: nuclear shape {nuc.shape} != actin shape {act.shape}")
    return nuc, act


def discover_inputs(
    config: Dict[str, Any],
    scope: str = "full",
    allow_unpaired: bool = True,
    engine: Optional[Any] = None,
    sample_n: Optional[int] = None,
    create_engine: bool = True,
) -> Tuple[List[Any], List[str], Any]:
    """Preflights dataset integrity and discovers/pairs IF field channels."""
    data_dir = Path(config["data_dir"])
    model_cfg = config.get("model") or {}
    model_path = model_cfg.get("pretrained_model", "models/cpsam_v2")
    random_seed = int(config.get("random_seed", 42))
    pixel_size_default = float(config.get("default_pixel_size_um", 0.505049))
    allow_uncalibrated = bool(config.get("allow_uncalibrated", False))

    if engine is None:
        preflight = audit_if_dataset(
            data_dir=data_dir, model_path=model_path, allow_unpaired=allow_unpaired,
            allow_uncalibrated=allow_uncalibrated,
        )
        logger.info("\n" + preflight.format_summary())
        preflight.log_and_raise_if_fatal()

    logger.info(f"Scanning data directory: {data_dir}")
    pairing = discover_and_pair_fields_detailed(
        data_dir=data_dir,
        allow_unpaired=allow_unpaired,
        default_pixel_size_um=pixel_size_default,
        allow_uncalibrated=allow_uncalibrated,
    )
    field_records = pairing.records
    skipped = list(pairing.unpaired_bases) + list(pairing.unparsed_bases)

    if scope == "subset":
        field_records = [r for r in field_records if r.field_id in SUBSET_FIELD_IDS]
        if not field_records:
            raise ValueError(
                f"--scope subset matched none of the {len(SUBSET_FIELD_IDS)} subset field ids "
                f"({', '.join(SUBSET_FIELD_IDS)}) in {data_dir}"
            )
        logger.info(f"Scope is SUBSET: filtered to {len(field_records)} high-disagreement fields.")
    else:
        logger.info(f"Scope is FULL: processing {len(field_records)} fields.")

    if sample_n is not None:
        field_records = stratified_sample(field_records, int(sample_n))
        logger.info(f"Sampled {len(field_records)} field(s) stratified by material: {[r.field_id for r in field_records]}")
    if not field_records:
        raise ValueError(f"No analysable IF fields found in {data_dir}")

    if engine is None and create_engine:
        engine = CPSAMEngine(
            model_path=model_path,
            random_seed=random_seed,
            use_gpu=bool(model_cfg.get("use_gpu", True)),
            model_type=str(model_cfg.get("model_type", "cpsam")),
        )

    return field_records, skipped, engine


def segment_field(
    rec: Any,
    nuclear_img: np.ndarray,
    actin_img: np.ndarray,
    combo: Dict[str, Any],
    preproc_cfg: Dict[str, Any],
    postproc_cfg: Dict[str, Any],
    qc_cfg: Dict[str, Any],
    feat_cfg: Dict[str, Any],
    masks_dir: Path,
    preproc_dir: Path,
    mask_cache: MaskCache,
    model_hash: str,
    random_seed: int,
    overwrite: bool,
    engine: Any,
    field_idx: int,
    run_name: str,
) -> Dict[str, Any]:
    """
    Processes one IF field (raw channels already loaded) through preprocessing, segmentation,
    postprocessing, spheroid extent, features and per-image QC.
    """
    field_id = rec.field_id

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
        "methods_applied": list(preproc_res.nuclear.methods_applied),
    }

    proc_nuclear_u8 = preproc_res.nuclear.preprocessed_u8
    proc_actin_u8 = preproc_res.actin.preprocessed_u8

    # Cache key: preprocessed image + Cellpose parameters + seed + weights hash (not slug/name/pre-post).
    cache_key = compute_cache_key(proc_nuclear_u8, segmentation_cache_params(combo, random_seed), model_hash=model_hash)
    raw_masks: Optional[np.ndarray] = None
    if not overwrite and mask_cache.has(cache_key):
        raw_masks = mask_cache.load(cache_key)
        if raw_masks is None:
            logger.warning(f"[{field_id}] Cached mask {cache_key[:12]} is unreadable; recomputing")
        elif raw_masks.shape != proc_nuclear_u8.shape:
            logger.warning(f"[{field_id}] Cached mask {cache_key[:12]} has shape {raw_masks.shape}; recomputing")
            raw_masks = None
        else:
            logger.info(f"[{field_id}] Reusing cached raw segmentation for key {cache_key[:12]}")
    if raw_masks is None:
        raw_masks, _mode_used = engine.segment_image(proc_nuclear_u8, combo, image_id=field_id)
        mask_cache.save(cache_key, raw_masks)

    final_masks, border_labels, post_stats = postprocess_field_masks(
        image_id=field_id,
        raw_masks=raw_masks,
        nuclear_img=proc_nuclear_u8,
        pixel_size_um=rec.pixel_size_um,
        postprocess_config=postproc_cfg,
    )

    _mask_path, overlay_path = save_mask_and_overlay(
        image_id=field_id,
        nuclear_img=proc_nuclear_u8,
        actin_img=proc_actin_u8,
        masks=final_masks,
        masks_dir=masks_dir,
        run_name=run_name,
    )

    # Spheroid extent from the RAW actin channel (bit-depth scaled only), so the density denominator is
    # the same for every pre/post-processing arm of this field. The nuclei-hull fallback is flagged.
    extent = compute_spheroid_extent_details(
        prepare_actin_for_extent(actin_img),
        final_masks,
        rec.pixel_size_um,
        method=feat_cfg.get("spheroid_extent_method", "actin_threshold"),
    )
    if extent.hull_fallback:
        logger.warning(f"[{field_id}] Actin spheroid mask too small; using the nuclei convex hull (flagged)")
    area_um2 = density_area_um2(extent, rec.pixel_size_um, post_stats.border_clearance_px)
    density_areal = compute_areal_density(post_stats.n_final_included, area_um2)

    qc_metric = compute_image_qc_metrics(
        image_id=field_id,
        material=rec.material,
        replicate=rec.replicate,
        nuclear_img=nuclear_img,
        actin_img=actin_img,
        raw_masks=raw_masks,
        dedup_masks=final_masks,
        n_dups_removed=post_stats.n_dedup_removed,
        pixel_size_um=rec.pixel_size_um,
        qc_config=qc_cfg,
        nuclei_per_mm2=density_areal,
        n_nuclei_valid=post_stats.n_final_included,
        n_border_excluded=post_stats.n_border_excluded,
    )

    obj_rows, per_img_row = extract_objects_and_image_features(
        image_id=field_id,
        material=rec.material,
        replicate=rec.replicate,
        nuclear_img=nuclear_img,
        actin_img=actin_img,
        dedup_masks=final_masks.astype(np.int32),
        pixel_size_um=rec.pixel_size_um,
        qc_flags_list="",  # filled in after the cohort QC pass
        feature_config=feat_cfg,
        border_excluded_labels=border_labels,
        spheroid_extent=extent,
        border_clearance_px=post_stats.border_clearance_px,
    )
    per_img_row["scale_source"] = getattr(rec, "scale_source", "")

    return {
        "field_id": field_id,
        "qc_metric": qc_metric,
        "post_stats": post_stats,
        "preproc_meta": preproc_meta,
        "preproc_methods": list(preproc_res.nuclear.methods_applied),
        "border_labels": border_labels,
        "sph_extent": extent,
        "overlay_path": overlay_path,
        "object_rows": obj_rows,
        "per_image_row": per_img_row,
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


def compute_stats(
    per_image_rows: List[Dict[str, Any]],
    summary_dir: Path,
) -> None:
    """Computes condition-level summary statistics across hydrogels."""
    compute_summary_by_material(per_image_rows, summary_dir / "summary_by_material.csv")


def flag_severity(metric: Any) -> int:
    """Number of artifact QC flags raised on a field (endpoint outlier flags excluded)."""
    flags = [f for f in str(getattr(metric, "flags_list", "")).split(";") if f]
    return sum(1 for f in flags if f in ARTIFACT_FLAGS)


def render_reports(
    final_qc_metrics: List[Any],
    overlay_paths: Dict[str, Path],
    qc_dir: Path,
    run_dir: Path,
    slug: str,
) -> str:
    """
    Saves QC overlay panels for the 5 worst flagged fields (most artifact flags first) and the biological
    figures. Returns the figure status ('ok' or 'failed: <reason>') for the run status record.
    """
    flagged = sorted(
        (m for m in final_qc_metrics if m.flag_any and m.image_id in overlay_paths),
        key=lambda m: (-flag_severity(m), m.image_id),
    )
    flagged_images = [
        {"image_id": m.image_id, "flags": m.flags_list, "overlay_path": overlay_paths[m.image_id]}
        for m in flagged
    ]
    save_qc_overlay_panels(flagged_images[:5], qc_dir)

    try:
        generate_all_if_biological_figures(run_dir=run_dir)
        return "ok"
    except Exception as e:
        logger.warning(f"Failed to generate biological figures for {slug}: {e}")
        return f"failed: {type(e).__name__}: {e}"


def _ordered_union(lists: Sequence[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for lst in lists:
        for m in lst:
            if m not in out:
                out.append(m)
    return out


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
    preproc_methods_by_field: Optional[Dict[str, List[str]]] = None,
    postproc_methods_by_field: Optional[Dict[str, List[str]]] = None,
    extent_by_field: Optional[Dict[str, Dict[str, Any]]] = None,
    scale_by_field: Optional[Dict[str, Dict[str, Any]]] = None,
    model_hash: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Builds and writes run_metadata.json for a sweep run. `methods` lists only the steps that actually
    ran (union over fields); resolved parameters record the values used.
    """
    post_enabled = bool(postproc_cfg.get("enabled", False))
    pre_enabled = bool(preproc_cfg.get("enabled", False))
    bg_cfg = preproc_cfg.get("background") or {}
    norm_cfg = preproc_cfg.get("normalize") or {}
    denoise_cfg = preproc_cfg.get("denoise") or {}
    split_cfg = postproc_cfg.get("splitting") or {}
    border_cfg = postproc_cfg.get("border_exclusion") or {}
    size_cfg = postproc_cfg.get("size_gate") or {}
    bounds = resolve_size_gate_bounds(postproc_cfg)
    pre_methods = _ordered_union(list((preproc_methods_by_field or {}).values()))
    post_methods = _ordered_union(list((postproc_methods_by_field or {}).values()))

    preproc_metadata = {
        "enabled": pre_enabled,
        "methods": pre_methods,
        "resolved_parameters": {
            "expected_nuclear_diameter_um": float(preproc_cfg.get("expected_nuclear_diameter_um", 11.7)),
            "background": {
                "method": bg_cfg.get("method", "none"),
                "radius_multiplier": float(bg_cfg.get("radius_multiplier", 3.5)),
                "downsample_factor": int(bg_cfg.get("downsample_factor", 2)),
                "blank_glob": bg_cfg.get("blank_glob"),
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
        "methods": post_methods,
        "resolved_parameters": {
            "reference_nuclear_volume_um3": bounds["reference_vol_um3"],
            "reference_citation": size_cfg.get("reference_citation"),
            "deduplicate": {
                "iou_threshold": float((postproc_cfg.get("deduplicate") or {}).get("iou_threshold", 0.50)),
                "applied": "deduplicate" in post_methods,
                "note": "2D label images are disjoint partitions; deduplication cannot remove anything on them",
            },
            "size_gate": {
                "enabled": bool(size_cfg.get("enabled", True)),
                "min_vol_um3": bounds["min_vol_um3"],
                "max_vol_um3": bounds["max_vol_um3"],
                "min_vol_um3_source": bounds["min_vol_um3_source"],
                "max_vol_um3_source": bounds["max_vol_um3_source"],
                "vol_multiplier_min": bounds["min_multiplier_of_reference"],
                "vol_multiplier_max": bounds["max_multiplier_of_reference"],
            },
            "splitting": {
                "enabled": bool(split_cfg.get("enabled", True)),
                "method": "seeded_watershed",
                "split_factor": float(split_cfg.get("split_factor", 2.0)),
                "h_maxima": float(split_cfg.get("h_maxima", 10.0)),
                "min_distance_um": float(split_cfg.get("min_distance_um", 4.0)),
            },
            "border_exclusion": {
                "enabled": bool(border_cfg.get("enabled", True)),
                "clearance_um": float(border_cfg.get("clearance_um", 5.85)),
                "density_area": "border strip excluded from the spheroid area when nuclei in it are excluded",
            },
        },
        "per_image_stats": per_image_postproc_stats,
        "cohort_deltas": {
            "debris_or_merged_pre_count": pre_debris_or_merged_count,
            "debris_or_merged_post_count": post_debris_or_merged_count,
            "resolved_count": max(0, pre_debris_or_merged_count - post_debris_or_merged_count),
        },
    }

    extra_info: Dict[str, Any] = {
        "unpaired_skipped": unpaired_bases,
        "segmentation_cache": {
            "key_inputs": "preprocessed nuclear image + " + ", ".join(CELLPOSE_PARAM_KEYS) + " + random_seed + model hash",
            "model_hash": model_hash,
        },
        "spheroid_extent": {
            "actin_preparation": "raw actin channel scaled to uint8 by bit depth (independent of pre/post arm)",
            "per_field": extent_by_field or {},
        },
        "pixel_scale": scale_by_field or {},
    }
    extra_info.update(extra or {})
    sha = model_hash if (model_hash and not model_hash.startswith("engine:")) else None

    write_run_metadata(
        run_dir=run_dir,
        resolved_params=combo,
        random_seed=random_seed,
        wall_time_sec=elapsed,
        extra_info=extra_info,
        preprocessing_info=preproc_metadata,
        postprocessing_info=postproc_metadata,
        weights_sha256=sha,
    )


def _patch_json(path: Path, updates: Dict[str, Any]) -> None:
    """Merge top-level keys into a JSON file, written atomically."""
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    data.update(updates)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_fingerprint(run_config: Dict[str, Any], field_records: Sequence[Any], model_hash: str) -> str:
    """Hash of everything that determines a run's outputs; a completed run is reused only if it matches."""
    payload = {
        "config": run_config,
        "model_hash": model_hash,
        "fields": [
            (r.field_id, r.nuclear_sha256, r.actin_sha256, round(float(r.pixel_size_um), 9))
            for r in sorted(field_records, key=lambda r: r.field_id)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def is_run_complete(run_dir: Path, fingerprint: Optional[str] = None) -> bool:
    """True when the completion marker exists (and, if given, records the same fingerprint)."""
    marker = run_dir / COMPLETION_MARKER
    if not marker.exists():
        return False
    if fingerprint is None:
        return True
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
    except (OSError, ValueError):
        return False


def _build_run_config(slug: str, run_name: str, scope: str, combo: Dict[str, Any], config: Dict[str, Any],
                      preproc_cfg: Dict[str, Any], postproc_cfg: Dict[str, Any], random_seed: int,
                      pixel_size_default: float) -> Dict[str, Any]:
    return {
        "run_slug": slug, "run_name": run_name, "scope": scope, "parameters": combo,
        "preprocessing": preproc_cfg, "postprocessing": postproc_cfg, "qc": config.get("qc", {}),
        "features": config.get("features", {}), "random_seed": random_seed, "pixel_size_um": pixel_size_default,
        "allow_uncalibrated": bool(config.get("allow_uncalibrated", False)),
    }


def _effective_sections(combo: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # A combo without its own section inherits the top-level one; after schema validation the
    # missing section is present as None, so `.get(key, default)` alone is not enough.
    preproc_cfg = combo.get("preprocessing") or config.get("preprocessing") or {"enabled": False}
    postproc_cfg = combo.get("postprocessing") or config.get("postprocessing") or {"enabled": False}
    return preproc_cfg, postproc_cfg


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
    model_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a single parameter or pre/post-processing combo in the sweep.
    Returns {'slug', 'status': 'complete'|'skipped', 'figures': ...}.
    """
    slug = resolve_run_slug(combo, combo_idx, scope)
    run_name = combo.get("name", slug)
    run_dir = runs_dir / slug
    model_hash = model_hash or engine_model_hash(engine)

    preproc_cfg, postproc_cfg = _effective_sections(combo, config)
    qc_cfg = config.get("qc", {})
    feat_cfg = config.get("features", {})
    run_config = _build_run_config(slug, run_name, scope, combo, config, preproc_cfg, postproc_cfg,
                                   random_seed, pixel_size_default)
    fingerprint = run_fingerprint(run_config, field_records, model_hash)

    if not overwrite and is_run_complete(run_dir, fingerprint):
        logger.info(f"Run {slug} is already complete with the same config and inputs; skipping (use --overwrite to redo)")
        return {"slug": slug, "status": "skipped"}
    if is_run_complete(run_dir):
        logger.info(f"Run {slug} has a completion marker for a different config/inputs or --overwrite was given; re-running")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / COMPLETION_MARKER).unlink(missing_ok=True)

    log_handler = setup_run_logger(run_dir)
    try:
        logger.info(f"\n{'='*70}\nStarting Run: {slug} ({combo.get('description', '')})\n{'='*70}")
        start_time = time.time()

        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(run_config, f, default_flow_style=False, sort_keys=False)
        write_manifest_csv(run_dir, field_records)

        masks_dir, qc_dir, features_dir = run_dir / "masks", run_dir / "qc", run_dir / "features"
        summary_dir, preproc_dir = run_dir / "summary", run_dir / "preprocessed"
        mask_cache = MaskCache(cache_dir=runs_dir / "cache" / "masks")
        meta_path = run_dir / "run_metadata.json"
        n_fields = len(field_records)

        # Provenance is written up front and patched per field, so an interrupted run says how far it got.
        _write_combo_provenance(
            run_dir=run_dir, combo=combo, preproc_cfg=preproc_cfg, postproc_cfg=postproc_cfg,
            per_image_preproc_meta={}, per_image_postproc_stats={}, pre_debris_or_merged_count=0,
            post_debris_or_merged_count=0, unpaired_bases=unpaired_bases, random_seed=random_seed, elapsed=0.0,
            model_hash=model_hash,
        )
        _patch_json(meta_path, {"progress": {"status": "running", "fields_completed": 0, "fields_total": n_fields}})

        raw_qc_metrics, overlay_paths = [], {}
        per_image_preproc_meta: Dict[str, Any] = {}
        per_image_postproc_stats: Dict[str, Any] = {}
        pre_methods: Dict[str, List[str]] = {}
        post_methods: Dict[str, List[str]] = {}
        extent_by_field: Dict[str, Dict[str, Any]] = {}
        scale_by_field: Dict[str, Dict[str, Any]] = {}
        all_objects: List[Dict[str, Any]] = []
        per_image_rows: List[Dict[str, Any]] = []
        pre_debris_or_merged_count, post_debris_or_merged_count = 0, 0
        t_loop = time.time()

        for field_idx, rec in enumerate(field_records, start=1):
            t_field = time.time()
            nuclear_img, actin_img = load_field_images(rec)
            res = segment_field(
                rec=rec, nuclear_img=nuclear_img, actin_img=actin_img, combo=combo, preproc_cfg=preproc_cfg,
                postproc_cfg=postproc_cfg, qc_cfg=qc_cfg, feat_cfg=feat_cfg, masks_dir=masks_dir,
                preproc_dir=preproc_dir, mask_cache=mask_cache, model_hash=model_hash, random_seed=random_seed,
                overwrite=overwrite, engine=engine, field_idx=field_idx, run_name=run_name,
            )
            fid, qm, ps = res["field_id"], res["qc_metric"], res["post_stats"]
            raw_qc_metrics.append(qm)
            overlay_paths[fid] = res["overlay_path"]
            per_image_preproc_meta[fid] = res["preproc_meta"]
            pre_methods[fid] = res["preproc_methods"]
            post_methods[fid] = list(ps.methods_applied)
            ext = res["sph_extent"]
            extent_by_field[fid] = {"method_used": ext.method_used, "hull_fallback": bool(ext.hull_fallback)}
            scale_by_field[fid] = {"pixel_size_um": rec.pixel_size_um, "scale_source": getattr(rec, "scale_source", "")}
            all_objects.extend(res["object_rows"])
            per_image_rows.append(res["per_image_row"])
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

            done = field_idx
            avg = (time.time() - t_loop) / done
            eta = avg * (n_fields - done)
            logger.info(f"[{slug}] field {done}/{n_fields} '{fid}' done in {time.time() - t_field:.1f}s; ETA {eta:.0f}s")
            _patch_json(meta_path, {"progress": {"status": "running", "fields_completed": done,
                                                 "fields_total": n_fields, "last_field": fid,
                                                 "eta_seconds": round(eta, 1)}})

        final_qc_metrics = apply_qc(raw_qc_metrics, qc_cfg, qc_dir)
        flags_by_id = {m.image_id: m.flags_list for m in final_qc_metrics}
        for row in per_image_rows:
            row["flag_summary"] = flags_by_id.get(row["image_id"]) or "clean"
        save_features_csvs(all_objects, per_image_rows, features_dir)
        compute_stats(per_image_rows, summary_dir)
        figures_status = render_reports(final_qc_metrics, overlay_paths, qc_dir, run_dir, slug)

        elapsed = time.time() - start_time
        n_fallback = sum(1 for e in extent_by_field.values() if e["hull_fallback"])
        _write_combo_provenance(
            run_dir=run_dir, combo=combo, preproc_cfg=preproc_cfg, postproc_cfg=postproc_cfg,
            per_image_preproc_meta=per_image_preproc_meta, per_image_postproc_stats=per_image_postproc_stats,
            pre_debris_or_merged_count=pre_debris_or_merged_count,
            post_debris_or_merged_count=post_debris_or_merged_count,
            unpaired_bases=unpaired_bases, random_seed=random_seed, elapsed=elapsed,
            preproc_methods_by_field=pre_methods, postproc_methods_by_field=post_methods,
            extent_by_field=extent_by_field, scale_by_field=scale_by_field, model_hash=model_hash,
            extra={"figures": figures_status, "n_spheroid_extent_hull_fallback": n_fallback},
        )
        _patch_json(meta_path, {"progress": {"status": "complete", "fields_completed": n_fields,
                                             "fields_total": n_fields, "figures": figures_status}})
        # Completion marker last: its presence means every output above was written.
        _patch_json(run_dir / COMPLETION_MARKER, {
            "slug": slug, "fingerprint": fingerprint, "fields": n_fields, "figures": figures_status,
            "completed_unix": time.time(),
        })
        logger.info(f"Run {slug} finished in {elapsed:.1f}s.")
        return {"slug": slug, "status": "complete", "figures": figures_status}
    finally:
        teardown_run_logger(log_handler)


def _resolve_allow_unpaired(config: Dict[str, Any], allow_unpaired_override: bool, strict_pairing: bool) -> bool:
    if allow_unpaired_override and strict_pairing:
        raise ValueError("--allow-unpaired and --strict-pairing are mutually exclusive")
    if strict_pairing:
        return False
    return bool(allow_unpaired_override or config.get("allow_unpaired", True))


def _dry_run(config_obj: IFSweepConfig, config: Dict[str, Any], scope: str, allow_unpaired: bool,
             overwrite: bool, sample_n: Optional[int], runs_dir: Path) -> Dict[str, Any]:
    """Preflight + pairing + resolved run plan, without importing torch or loading the model."""
    model_path = (config.get("model") or {}).get("pretrained_model", "models/cpsam_v2")
    preflight = audit_if_dataset(
        data_dir=config["data_dir"], model_path=model_path, allow_unpaired=allow_unpaired,
        allow_uncalibrated=bool(config.get("allow_uncalibrated", False)),
    )
    print(preflight.format_summary())
    preflight.log_and_raise_if_fatal()
    field_records, skipped, _ = discover_inputs(
        config=config, scope=scope, allow_unpaired=allow_unpaired, engine=None, sample_n=sample_n,
        create_engine=False,
    )
    comparator = config_obj.resolved_comparator()
    plan = []
    lines = ["", "DRY RUN: resolved sweep plan (nothing executed)", f"Fields: {len(field_records)}  "
             f"(skipped: {len(skipped)})  scope={scope}  comparator={comparator}  runs_dir={runs_dir}"]
    for idx, combo in enumerate(config.get("combos", []), start=1):
        slug = resolve_run_slug(combo, idx, scope)
        run_dir = runs_dir / slug
        complete = is_run_complete(run_dir)
        action = "run" if (overwrite or not complete) else "skip if config unchanged (complete)"
        pre, post = _effective_sections(combo, config)
        plan.append({"slug": slug, "run_dir": str(run_dir), "action": action,
                     "preprocessing": bool(pre.get("enabled")), "postprocessing": bool(post.get("enabled"))})
        lines.append(f"  [{idx}] {slug:<28} -> {run_dir}  pre={bool(pre.get('enabled'))} "
                     f"post={bool(post.get('enabled'))}  action={action}")
    print("\n".join(lines))
    return {"status": "DRY_RUN", "runs_dir": runs_dir, "plan": plan, "n_fields": len(field_records)}


def run_sweep(
    config_path: str | Path,
    allow_unpaired_override: bool = False,
    overwrite_override: bool = False,
    scope: str = "full",
    engine: Optional[Any] = None,
    runs_dir_override: Optional[str] = None,
    data_dir_override: Optional[str] = None,
    sample_n: Optional[int] = None,
    strict_pairing: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute sweep runs defined in config with optional subset scope and pre/post-processing.
    allow_unpaired_override=True forces unpaired fields to be skipped; strict_pairing=True makes any
    unpaired field fatal (overrides the YAML default). dry_run prints the plan without running.
    """
    config_obj = IFSweepConfig.from_yaml(config_path)
    config = config_obj.model_dump()
    if runs_dir_override:
        config["runs_dir"] = str(runs_dir_override)
    if data_dir_override:
        config["data_dir"] = str(data_dir_override)

    runs_dir = Path(config["runs_dir"])
    allow_unpaired = _resolve_allow_unpaired(config, allow_unpaired_override, strict_pairing)
    overwrite = bool(overwrite_override or config.get("overwrite", False))
    random_seed = int(config.get("random_seed", 42))
    pixel_size_default = float(config.get("default_pixel_size_um", 0.505049))

    combos = config.get("combos", [])
    if not combos:
        raise ValueError("No combos defined in configuration file.")

    if dry_run:
        return _dry_run(config_obj, config, scope, allow_unpaired, overwrite, sample_n, runs_dir)

    runs_dir.mkdir(parents=True, exist_ok=True)
    field_records, unpaired_bases, engine = discover_inputs(
        config=config, scope=scope, allow_unpaired=allow_unpaired, engine=engine, sample_n=sample_n,
    )
    model_hash = engine_model_hash(engine)

    logger.info(f"Beginning sweep across {len(combos)} configurations on {len(field_records)} fields.")

    results = []
    for combo_idx, combo in enumerate(combos, start=1):
        results.append(run_single_combo(
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
            model_hash=model_hash,
        ))
    slugs = [r["slug"] for r in results]

    comparator = config_obj.resolved_comparator()
    suffix = SUBSET_SUFFIX if scope == "subset" else ""
    if comparator == "prepost":
        logger.info("\n" + "="*70 + "\nExecuting Pre/Post Cross-Run Comparison\n" + "="*70)
        run_comparator_prepost(runs_dir=runs_dir, output_dir=runs_dir / f"comparison_prepost{suffix}", scope=scope)
    else:
        logger.info("\n" + "="*70 + "\nExecuting Cross-Run Comparison of this sweep's runs\n" + "="*70)
        generate_cross_run_comparison(runs_dir=runs_dir, output_dir=runs_dir / f"comparison{suffix}", run_names=slugs)

    logger.info("Sweep execution and comparison complete.")
    return {
        "status": "SUCCESS",
        "runs_dir": runs_dir,
        "runs": slugs,
        "skipped_runs": [r["slug"] for r in results if r["status"] == "skipped"],
        "figure_failures": {r["slug"]: r["figures"] for r in results if str(r.get("figures", "ok")).startswith("failed")},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-run Cellpose-SAM (cpsam) IF segmentation sweep, ablation, and comparison pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sweep_parser = subparsers.add_parser("sweep", help="Execute hyperparameter or pre/post sweep.")
    sweep_parser.add_argument("--config", type=str, default="configs/sweep.yaml", help="Path to YAML configuration.")
    sweep_parser.add_argument("--scope", type=str, choices=["subset", "full"], default="full", help="Evaluation scope: 'subset' (named high-disagreement fields) or 'full' (all fields).")
    pairing = sweep_parser.add_mutually_exclusive_group()
    pairing.add_argument("--allow-unpaired", action="store_true", help="Skip unpaired files (overrides YAML).")
    pairing.add_argument("--strict-pairing", action="store_true", help="Fail on any unpaired file (overrides YAML allow_unpaired: true).")
    sweep_parser.add_argument("--overwrite", action="store_true", help="Recompute runs that are already complete and bypass the mask cache.")
    sweep_parser.add_argument("--runs-dir", type=str, default=None, help="Override runs_dir from the config.")
    sweep_parser.add_argument("--data-dir", type=str, default=None, help="Override data_dir from the config.")
    sweep_parser.add_argument("--sample", type=int, default=None, help="Process N fields, stratified by material.")
    sweep_parser.add_argument("--dry-run", action="store_true", help="Preflight, pair and print the run plan without segmenting.")

    compare_parser = subparsers.add_parser("compare", help="Compare completed runs.")
    compare_parser.add_argument("--runs", type=str, default="runs", help="Path to runs directory.")
    compare_parser.add_argument("--output", type=str, default=None, help="Path to comparison output directory.")
    compare_parser.add_argument("--run-names", type=str, default=None, help="Comma-separated run directory names to compare.")
    compare_parser.add_argument("--scope", type=str, choices=["full", "subset", "any"], default="full", help="Runs to include when --run-names is not given.")

    compare_pp_parser = subparsers.add_parser("compare-prepost", help="Compare pre/post ablation and robustness runs.")
    compare_pp_parser.add_argument("--runs", type=str, default="runs", help="Path to runs directory.")
    compare_pp_parser.add_argument("--output", type=str, default=None, help="Path to comparison output directory.")
    compare_pp_parser.add_argument("--scope", type=str, choices=["full", "subset"], default="full")

    comp_bio_parser = subparsers.add_parser("compare-biological", help="Generate publication-grade biological comparison figures across hydrogels.")
    comp_bio_parser.add_argument("--run-dir", type=str, default="runs/104_PP11_full", help="Path to run directory containing features/ and qc/.")
    comp_bio_parser.add_argument("--output", type=str, default=None, help="Directory to save biological figures (default: <run_dir>/figures).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point. Returns 0 on success, 1 on failure."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "sweep":
            run_sweep(
                config_path=args.config,
                allow_unpaired_override=args.allow_unpaired,
                overwrite_override=args.overwrite,
                scope=args.scope,
                runs_dir_override=args.runs_dir,
                data_dir_override=args.data_dir,
                sample_n=args.sample,
                strict_pairing=args.strict_pairing,
                dry_run=args.dry_run,
            )
        elif args.command == "compare":
            names = [n.strip() for n in args.run_names.split(",") if n.strip()] if args.run_names else None
            generate_cross_run_comparison(runs_dir=args.runs, output_dir=args.output, run_names=names, scope=args.scope)
        elif args.command == "compare-prepost":
            run_comparator_prepost(runs_dir=args.runs, output_dir=args.output, scope=args.scope)
        elif args.command == "compare-biological":
            generate_all_if_biological_figures(run_dir=args.run_dir, output_dir=args.output)
            print(f"[+] Generated biological comparison figures in {args.output or Path(args.run_dir) / 'figures'}")
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as e:
        logger.error(f"{args.command} failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
