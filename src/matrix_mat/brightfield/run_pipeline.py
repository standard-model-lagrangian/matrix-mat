"""
Master Pipeline Orchestrator and CLI Entry Point for Brightfield Spheroids.

Provides:
  - Full end-to-end multi-spheroid analysis over raw brightfield TIFF datasets.
  - Manifest generation, scale extraction, 3-tier segmentation, QC filtering,
    morphometrics measurement, temporal Hungarian pairing, hierarchical statistics,
    presentation figures, and executive reporting.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import numpy as np
import pandas as pd
import tifffile

from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.brightfield.decisions import DecisionsLogger
from matrix_mat.core.cache import MaskCache
from matrix_mat.brightfield.measure import SpheroidObjectRecord, save_objects_csv
from matrix_mat.brightfield.pair import (
    SpheroidPairRecord,
    pair_dataset,
    save_pairs_csv,
    update_objects_with_pair_ids,
)
from matrix_mat.brightfield.preprocess import Preprocessor
from matrix_mat.brightfield.qc_filter import deduplicate_dataset_objects, filter_and_measure_objects
from matrix_mat.brightfield.review import ReviewManager
from matrix_mat.core.scale import ScaleExtractor
from matrix_mat.brightfield.segment import SegmentationHierarchy
from matrix_mat.brightfield.stats import (
    compute_condition_statistics,
    compute_exclusion_audit,
    generate_exclusion_audit_table,
    save_condition_summary_csv,
)
from matrix_mat.brightfield.audit import (
    audit_fail_class,
    create_fail_audit_crops_grid,
    audit_session_quality,
)
from matrix_mat.brightfield.preflight import PreflightAuditor
from matrix_mat.core.provenance import export_complete_manifest, write_run_metadata
from matrix_mat.core.naming import parse_image_filename, parse_timepoint, is_t0

logger = logging.getLogger("matrix_mat.brightfield.run_pipeline")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configures structured logging output."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# -----------------------------------------------------------------------------
# Manifest & File Ingestion
# -----------------------------------------------------------------------------

def build_manifest(t0_dir: Union[str, Path], t7_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Scans Day 0 and Day 7 directories, builds inventory manifest, and identifies matched pairs.
    """
    t0_dir = Path(t0_dir)
    t7_dir = Path(t7_dir)

    records: List[Dict[str, Any]] = []

    # Scan Day 0
    if t0_dir.exists():
        for p in sorted(list(t0_dir.glob("*.tif")) + list(t0_dir.glob("*.tiff"))):
            try:
                meta = parse_image_filename(p.name, source_dir_timepoint="Day0")
                meta["file_path"] = str(p)
                records.append(meta)
            except ValueError as e:
                logger.warning(f"Skipping unmatched Day0 image {p.name}: {e}")

    # Scan Day 7
    if t7_dir.exists():
        for p in sorted(list(t7_dir.glob("*.tif")) + list(t7_dir.glob("*.tiff"))):
            try:
                meta = parse_image_filename(p.name, source_dir_timepoint="Day7")
                meta["file_path"] = str(p)
                records.append(meta)
            except ValueError as e:
                logger.warning(f"Skipping unmatched Day7 image {p.name}: {e}")

    if not records:
        logger.warning(f"No TIFF images found in {t0_dir} or {t7_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Determine matched pairs
    pair_counts = df.groupby("pair_key")["timepoint"].nunique()
    matched_keys = set(pair_counts[pair_counts >= 2].index)
    df["is_matched_pair"] = df["pair_key"].isin(matched_keys)

    logger.info(
        f"Built manifest: {len(df)} total images ({len(df[df['timepoint']=='t0'])} D0, "
        f"{len(df[df['timepoint']=='t7'])} D7), {len(matched_keys)} matched FOV pairs across "
        f"{df['condition'].nunique()} conditions."
    )
    return df


# -----------------------------------------------------------------------------
# Synthetic Smoke Test Runner (No test code imported)
# -----------------------------------------------------------------------------

def run_synthetic_smoke_test(config: Optional[PipelineConfig] = None) -> bool:
    """
    Executes the 3 synthetic smoke test scenarios:
      1. Isolated dark circular spheroid (ground truth d=100px): asserts recovered d within 10%.
      2. Touching pair of spheroids (ground truth d=80px each): asserts 2 separated instances, both d within 10%.
      3. Bright gel blob / refractive artifact (ground truth d=120px): asserts 0 masks segmented (contrast gate).
    """
    print("\n" + "=" * 80)
    print("   RUNNING SYNTHETIC GROUND-TRUTH SMOKE TESTS")
    print("=" * 80)

    cfg = config or PipelineConfig()
    cfg.segmentation.primary_backend = "classical"
    passed_all = True
    scale = 1.518817

    # Scenario 1: Isolated Dark Spheroid (d = 100 px)
    img1 = np.full((600, 800), 0.65, dtype=np.float32)
    cv2.circle(img1, (400, 300), 50, 0.20, -1)
    cv2.circle(img1, (400, 300), 50, 0.15, 1)

    seg1 = SegmentationHierarchy(cfg.segmentation)
    mask1, _ = seg1.segment(img1, "synthetic_isolated", pixel_size_um=scale)
    clean_mask1, objs1 = filter_and_measure_objects(
        img1, mask1, {"image_id": "synthetic_isolated", "pixel_size_um": scale, "is_calibrated": True}, cfg
    )

    pass_objs1 = [o for o in objs1 if o.qc_flag == "PASS"]
    if len(pass_objs1) == 1:
        rec_d = pass_objs1[0].equivalent_diameter_px
        err = abs(rec_d - 100.0) / 100.0 * 100.0
        status1 = "PASS" if err <= 10.0 else "FAIL"
        print(f"[*] Scenario 1 (Isolated Spheroid d=100px): [{status1}] (Recovered d={rec_d:.1f}px, Error={err:.2f}%)")
        if status1 == "FAIL":
            passed_all = False
    else:
        print(f"[*] Scenario 1 (Isolated Spheroid): [FAIL] (Expected 1 PASS object, got {len(pass_objs1)})")
        passed_all = False

    # Scenario 2: Touching Pair (d1 = 80 px, d2 = 80 px)
    img2 = np.full((600, 800), 0.65, dtype=np.float32)
    cv2.circle(img2, (365, 300), 40, 0.20, -1)
    cv2.circle(img2, (435, 300), 40, 0.20, -1)
    cv2.circle(img2, (365, 300), 40, 0.15, 1)
    cv2.circle(img2, (435, 300), 40, 0.15, 1)

    mask2, _ = seg1.segment(img2, "synthetic_touching", pixel_size_um=scale)
    clean_mask2, objs2 = filter_and_measure_objects(
        img2, mask2, {"image_id": "synthetic_touching", "pixel_size_um": scale, "is_calibrated": True}, cfg
    )

    pass_objs2 = [o for o in objs2 if o.qc_flag == "PASS"]
    if len(pass_objs2) == 2:
        d_vals = [o.equivalent_diameter_px for o in pass_objs2]
        errs = [abs(d - 80.0) / 80.0 * 100.0 for d in d_vals]
        status2 = "PASS" if all(e <= 10.0 for e in errs) else "FAIL"
        print(f"[*] Scenario 2 (Touching Pair d=80px): [{status2}] (Separated into 2 instances, d=[{d_vals[0]:.1f}, {d_vals[1]:.1f}], errs=[{errs[0]:.1f}%, {errs[1]:.1f}%])")
        if status2 == "FAIL":
            passed_all = False
    else:
        print(f"[*] Scenario 2 (Touching Pair): [FAIL] (Expected 2 PASS objects, got {len(pass_objs2)})")
        passed_all = False

    # Scenario 3: Bright Gel Blob (d = 120 px)
    img3 = np.full((600, 800), 0.65, dtype=np.float32)
    cv2.circle(img3, (400, 300), 60, 0.95, -1)

    mask3, _ = seg1.segment(img3, "synthetic_bright_blob", pixel_size_um=scale)
    clean_mask3, objs3 = filter_and_measure_objects(
        img3, mask3, {"image_id": "synthetic_bright_blob", "pixel_size_um": scale, "is_calibrated": True}, cfg
    )

    pass_objs3 = [o for o in objs3 if o.qc_flag == "PASS"]
    status3 = "PASS" if len(pass_objs3) == 0 else "FAIL"
    print(f"[*] Scenario 3 (Bright Gel Blob): [{status3}] (Expected 0 PASS objects, got {len(pass_objs3)})")
    if status3 == "FAIL":
        passed_all = False

    print("=" * 80)
    overall_status = "ALL SMOKE TESTS PASSED" if passed_all else "SMOKE TESTS FAILED"
    print(f"[*] RESULT: {overall_status}")
    print("=" * 80 + "\n")
    return passed_all


# -----------------------------------------------------------------------------
# Master Pipeline Orchestrator
# -----------------------------------------------------------------------------

def run_spheroid_pipeline(
    config: Optional[PipelineConfig] = None,
    sample_n: Optional[int] = None,
    is_sample_12_loop: bool = False,
    max_tuning_cycles: int = 1,
) -> Dict[str, Any]:
    """
    Executes the end-to-end brightfield spheroid analysis pipeline.
    """
    start_time = time.time()
    config = config or PipelineConfig()

    from matrix_mat.reporting.visualize import (
        create_overlay,
        generate_all_figures,
        generate_condition_contact_sheets,
    )
    from matrix_mat.reporting.presentation_plots import (
        compute_well_growth_comparison,
        generate_all_presentation_figures,
    )
    from matrix_mat.reporting.report import ReportGenerator

def discover_inputs(
    config: PipelineConfig,
    sample_n: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Path]:
    """Runs preflight integrity audit, prepares output dirs, and builds/samples manifest."""
    logger.info("Running preflight environment and dataset integrity audit...")
    preflight = PreflightAuditor(config).run_preflight_check()
    logger.info("\n" + preflight.format_summary())
    preflight.log_and_raise_if_fatal()
    active_warnings = list(preflight.warnings)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / config.artifacts.overlay_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / config.artifacts.contact_sheets_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / config.artifacts.figures_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / config.artifacts.reports_dir).mkdir(parents=True, exist_ok=True)
    if config.preprocess.save_corrected_images:
        (out_dir / "corrected_images").mkdir(parents=True, exist_ok=True)

    manifest_df = build_manifest(config.t0_dir, config.t7_dir)
    if manifest_df.empty:
        logger.error("No images found to process.")
        return pd.DataFrame(), active_warnings, out_dir

    if config.condition:
        manifest_df = manifest_df[manifest_df["condition"] == config.condition].copy()
        if manifest_df.empty:
            logger.error(f"No images found matching condition '{config.condition}'.")
            return pd.DataFrame(), active_warnings, out_dir
        logger.info(f"Filtered dataset to condition '{config.condition}': {len(manifest_df)} images found.")

    export_complete_manifest(manifest_df, out_dir, compute_checksums=True)

    effective_sample_n = sample_n if sample_n is not None else config.sample_n
    if effective_sample_n is not None and effective_sample_n > 0:
        conditions = sorted(manifest_df["condition"].unique())
        selected_rows = []
        pairs_per_cond = max(1, effective_sample_n // (len(conditions) * 2))
        for cond in conditions:
            c_pairs = manifest_df[(manifest_df["condition"] == cond) & (manifest_df["is_matched_pair"] == True)]
            unique_keys = c_pairs["pair_key"].unique()
            for key in unique_keys[:pairs_per_cond]:
                selected_rows.append(c_pairs[c_pairs["pair_key"] == key])
        sample_manifest = pd.concat(selected_rows).drop_duplicates().head(effective_sample_n) if selected_rows else manifest_df.head(effective_sample_n)
        manifest_to_process = sample_manifest
        logger.info(f"Running stratified sample execution on {len(manifest_to_process)} images")
    else:
        manifest_to_process = manifest_df

    return manifest_to_process, active_warnings, out_dir


def segment_cohort(
    manifest_to_process: pd.DataFrame,
    config: PipelineConfig,
    out_dir: Path,
) -> Tuple[List[SpheroidObjectRecord], List[Dict[str, Any]]]:
    """Segments images across the cohort, measures objects, and generates supervision overlays."""
    from matrix_mat.reporting.visualize import create_overlay

    overlays_dir = out_dir / config.artifacts.overlay_dir
    decisions_logger = DecisionsLogger(out_dir / "decisions.jsonl")
    review_manager = ReviewManager(config.review_dir)
    mask_cache = MaskCache(config.segmentation.cache_dir)
    scale_extractor = ScaleExtractor(
        default_pixel_size_um=config.default_pixel_size_um,
        cli_override_um=config.cli_pixel_size_um,
    )
    preprocessor = Preprocessor(config.preprocess)
    seg_hierarchy = SegmentationHierarchy(
        config=config.segmentation,
        decisions_logger=decisions_logger,
        mask_cache=mask_cache,
        preprocessor=preprocessor,
    )

    all_objects: List[SpheroidObjectRecord] = []
    all_validation_records: List[Dict[str, Any]] = []

    logger.info(f"--- Pipeline Execution ({len(manifest_to_process)} images) ---")
    for _, row in manifest_to_process.iterrows():
        img_path = Path(row["file_path"])
        image_id, pair_key = row["image_id"], row["pair_key"]
        timepoint, condition, replicate, fov = row["timepoint"], row["condition"], row["replicate"], row["fov"]

        scale_info = scale_extractor.extract(img_path)
        pixel_size_um = scale_info.pixel_size_um

        try:
            raw_img = tifffile.imread(str(img_path))
        except Exception as e:
            logger.warning(f"Could not read TIFF {img_path}: {e}")
            continue

        manual_mask_path = review_manager.find_manual_mask(image_id, pair_key=pair_key, timepoint=timepoint)
        if manual_mask_path is not None:
            manual_mask = review_manager.load_manual_mask(manual_mask_path)
            auto_mask, _ = seg_hierarchy.segment(raw_img, image_id, pixel_size_um)
            dice, iou = review_manager.compute_agreement(auto_mask, manual_mask)
            active_mask, mask_source = manual_mask, "manual_review"
            seg_meta = {"backend": "manual_review", "model_id": "human", "diameter_used": None}
            all_validation_records.append({
                "image_id": image_id, "pair_key": pair_key, "timepoint": timepoint,
                "manual_mask_path": str(manual_mask_path), "n_auto_objects": int(np.max(auto_mask)),
                "n_manual_objects": int(np.max(manual_mask)), "dice_coefficient": float(round(dice, 4)),
                "iou_jaccard": float(round(iou, 4)), "pixel_size_um": float(pixel_size_um),
                "notes": f"Manual override applied. Dice={dice:.3f}, IoU={iou:.3f}",
            })
        else:
            active_mask, seg_meta = seg_hierarchy.segment(raw_img, image_id, pixel_size_um)
            mask_source = "automated"

        active_backend = config.segmentation.primary_backend
        img_meta = {
            "image_id": image_id, "timepoint": timepoint, "condition": condition,
            "replicate": replicate, "fov": fov, "pair_key": pair_key,
            "pixel_size_um": pixel_size_um, "mask_source": mask_source,
            "seg_backend": seg_meta.get("backend", active_backend),
            "model_id": seg_meta.get("model_id", active_backend),
            "diameter_used": seg_meta.get("diameter_used", None),
            "scale_source": scale_info.source, "is_calibrated": scale_info.is_calibrated,
            "allow_uncalibrated": getattr(config, "allow_uncalibrated", False) or not getattr(config.qc, "require_calibration", True),
            "seg_qc_flag": seg_meta.get("qc_flag"), "seg_qc_reasons": seg_meta.get("qc_reasons"),
            "plausibility_status": seg_meta.get("plausibility_status", "PASS"),
        }
        clean_mask, obj_records = filter_and_measure_objects(raw_img, active_mask, img_meta, config)
        all_objects.extend(obj_records)

        overlay_out = overlays_dir / f"{image_id}_overlay.png"
        try:
            create_overlay(raw_img, clean_mask, obj_records, overlay_out, pixel_size_um=pixel_size_um)
        except Exception as e:
            logger.warning(f"Failed to create overlay for {image_id}: {e}")

    logger.info("Executing cross-FOV multi-focal plane deduplication pass...")
    all_objects, dedup_meta = deduplicate_dataset_objects(all_objects, max_duplicate_distance_px=35.0)
    logger.info(f"Deduplication pass complete: {dedup_meta['duplicates_flagged']} duplicates flagged FAIL.")
    return all_objects, all_validation_records


def pair_timepoints(
    all_objects: List[SpheroidObjectRecord],
    config: PipelineConfig,
    out_dir: Path,
) -> Tuple[List[SpheroidPairRecord], List[SpheroidObjectRecord], pd.DataFrame, pd.DataFrame]:
    """Pairs Day 0 and Day 7 spheroids and exports objects.csv and pairs.csv."""
    objects_csv_path = out_dir / "objects.csv"
    save_objects_csv(all_objects, objects_csv_path)

    logger.info("Executing Hungarian mutual nearest centroid pairing...")
    all_pairs, all_objects = pair_dataset(all_objects, config)
    save_objects_csv(all_objects, objects_csv_path)

    pairs_csv_path = out_dir / "pairs.csv"
    save_pairs_csv(all_pairs, pairs_csv_path)

    objects_df = pd.DataFrame([vars(o) for o in all_objects]) if all_objects else pd.DataFrame()
    pairs_df = pd.DataFrame([vars(p) for p in all_pairs]) if all_pairs else pd.DataFrame()
    return all_pairs, all_objects, objects_df, pairs_df


def compute_stats(
    all_pairs: List[SpheroidPairRecord],
    all_objects: List[SpheroidObjectRecord],
    config: PipelineConfig,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes condition-level summary statistics and 10-stage exclusion audit."""
    logger.info("Computing condition-level summary statistics and hypothesis tests...")
    condition_summary_df = compute_condition_statistics(all_pairs, all_objects=all_objects, config=config)
    save_condition_summary_csv(condition_summary_df, out_dir / "condition_summary.csv")

    exclusion_dict = compute_exclusion_audit(all_objects=all_objects, all_pairs=all_pairs)
    exclusion_audit_df = generate_exclusion_audit_table(exclusion_dict)
    return condition_summary_df, exclusion_audit_df


def render_reports(
    manifest_to_process: pd.DataFrame,
    all_objects: List[SpheroidObjectRecord],
    all_pairs: List[SpheroidPairRecord],
    objects_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    condition_summary_df: pd.DataFrame,
    exclusion_audit_df: pd.DataFrame,
    all_validation_records: List[Dict[str, Any]],
    config: PipelineConfig,
    out_dir: Path,
    start_time: float,
    active_warnings: List[str],
    sample_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Generates figures, contact sheets, presentation plots, and markdown/HTML reports."""
    from matrix_mat.reporting.visualize import (
        generate_all_figures,
        generate_condition_contact_sheets,
    )
    from matrix_mat.reporting.presentation_plots import (
        compute_well_growth_comparison,
        generate_all_presentation_figures,
    )
    from matrix_mat.reporting.report import ReportGenerator

    overlays_dir = out_dir / config.artifacts.overlay_dir
    contact_dir = out_dir / config.artifacts.contact_sheets_dir
    review_manager = ReviewManager(config.review_dir)
    validation_df = review_manager.save_validation_report(all_validation_records, out_dir / "validation.csv")

    logger.info("Generating publication figures (PNG + PDF)...")
    fig_paths = generate_all_figures(pairs_df, objects_df, out_dir, save_pdf=True, save_png=True, condition_df=condition_summary_df)

    fail_fig_paths = audit_fail_class(objects_df, out_dir, save_pdf=True)
    fig_paths["fail_class_audit"] = fail_fig_paths

    contact_sheets: Dict[str, Path] = {}
    fail_grid_path = create_fail_audit_crops_grid(
        objects_df, config.t0_dir, config.t7_dir, contact_dir / "fail_objects_audit_grid.png", num_samples=20
    )
    if fail_grid_path:
        contact_sheets["FAIL_Audit_Grid"] = fail_grid_path

    session_df, session_fig_paths = audit_session_quality(config.t0_dir, config.t7_dir, out_dir, save_pdf=True)
    fig_paths["session_focus_exposure_audit"] = session_fig_paths
    contact_sheets.update(generate_condition_contact_sheets(overlays_dir, contact_dir, manifest_to_process))

    try:
        wells_df = compute_well_growth_comparison(objects_df, out_dir / "well_growth_comparison.csv")
        pres_figs = generate_all_presentation_figures(out_dir, objects_df, pairs_df, wells_df, condition_df=condition_summary_df)
        fig_paths.update(pres_figs)
    except Exception as e:
        logger.warning(f"Failed to generate presentation figures: {e}")

    report_gen = ReportGenerator(config, out_dir)
    md_path, html_path = report_gen.generate(
        manifest_df=manifest_to_process, objects_df=objects_df, pairs_df=pairs_df,
        condition_summary_df=condition_summary_df, exclusion_audit_df=exclusion_audit_df,
        fig_paths=fig_paths, overlay_samples=list(overlays_dir.glob("*.png"))[:12],
        contact_sheets=contact_sheets, validation_df=validation_df,
        decisions_path=out_dir / "decisions.jsonl", session_df=session_df,
    )

    wall_time = time.time() - start_time
    write_run_metadata(
        output_dir=out_dir, config=config, wall_time_sec=wall_time,
        warnings=active_warnings, backend=config.segmentation.primary_backend,
        extra_info={"objects_count": len(all_objects), "pairs_count": len(all_pairs), "is_sample_mode": sample_n is not None},
    )

    logger.info(f"Pipeline execution completed successfully. Artifacts available in {out_dir}/")
    return {
        "status": "SUCCESS", "objects_count": len(all_objects), "pairs_count": len(all_pairs),
        "condition_summary": condition_summary_df, "md_report": md_path,
        "html_report": html_path, "figures": fig_paths, "contact_sheets": contact_sheets,
    }


def run_spheroid_pipeline(
    config: Optional[PipelineConfig] = None,
    sample_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Master pipeline runner function for spheroid volume changes and kinetics.
    Decomposed into clean, modular steps (discover_inputs, segment_cohort,
    pair_timepoints, compute_stats, render_reports).
    """
    start_time = time.time()
    config = config or PipelineConfig()

    logger.info("================================================================================")
    logger.info("       LAUNCHING BRIGHTFIELD SPHEROID ANALYSIS PIPELINE (matrix-mat)           ")
    logger.info("================================================================================")

    manifest_to_process, active_warnings, out_dir = discover_inputs(config, sample_n)
    if manifest_to_process.empty:
        return {"status": "FAILED", "error": "No images found to process"}

    all_objects, all_validation_records = segment_cohort(manifest_to_process, config, out_dir)
    all_pairs, all_objects, objects_df, pairs_df = pair_timepoints(all_objects, config, out_dir)
    condition_summary_df, exclusion_audit_df = compute_stats(all_pairs, all_objects, config, out_dir)

    return render_reports(
        manifest_to_process=manifest_to_process,
        all_objects=all_objects,
        all_pairs=all_pairs,
        objects_df=objects_df,
        pairs_df=pairs_df,
        condition_summary_df=condition_summary_df,
        exclusion_audit_df=exclusion_audit_df,
        all_validation_records=all_validation_records,
        config=config,
        out_dir=out_dir,
        start_time=start_time,
        active_warnings=active_warnings,
        sample_n=sample_n,
    )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """Builds comprehensive CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Brightfield Spheroid Volume Analysis Pipeline (matrix-mat bf)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute synthetic ground-truth smoke test (3 scenarios).",
    )
    mode_group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Run stratified N-sample validation with programmatic sanity gates.",
    )
    mode_group.add_argument(
        "--full-dataset",
        action="store_true",
        help="Run full dataset batch execution.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/spheroid_brightfield.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--input-dir", type=str, help="Unified input directory.")
    parser.add_argument("--t0-dir", type=str, help="Day 0 input image directory.")
    parser.add_argument("--t7-dir", type=str, help="Day 7 input image directory.")
    parser.add_argument(
        "-o", "--out", "--output-dir",
        dest="output_dir",
        type=str,
        default="output",
        help="Output directory for CSVs and artifacts.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Filter dataset to specific hydrogel condition (e.g. S34D30, Mat).",
    )
    parser.add_argument("--review-dir", type=str, default="review", help="Directory monitoring manual ground-truth masks.")
    parser.add_argument("--pixel-size", type=float, help="Optical calibration override in um/pixel.")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation without loading cached masks.")
    parser.add_argument("--plots", action="store_true", help="Generate all presentation figures from existing CSVs.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight input and optical calibration validation and exit.")
    parser.add_argument("--allow-uncalibrated", "--force-uncalibrated", dest="allow_uncalibrated", action="store_true", help="Allow uncalibrated optical scale images without flagging REVIEW.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error log output.")

    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI execution entry point."""
    parser = build_cli_parser()
    cli_args = parser.parse_args(args)

    setup_logging(verbose=cli_args.verbose, quiet=cli_args.quiet)

    # 1. Handle Synthetic Smoke Test mode
    if cli_args.smoke_test:
        success = run_synthetic_smoke_test()
        return 0 if success else 1

    # 2. Build Pipeline Configuration
    cli_overrides: Dict[str, Any] = {}
    if cli_args.output_dir:
        cli_overrides["output_dir"] = cli_args.output_dir
    if cli_args.review_dir:
        cli_overrides["review_dir"] = cli_args.review_dir
    if cli_args.t0_dir:
        cli_overrides["t0_dir"] = cli_args.t0_dir
    if cli_args.t7_dir:
        cli_overrides["t7_dir"] = cli_args.t7_dir
    if cli_args.pixel_size:
        cli_overrides["cli_pixel_size_um"] = cli_args.pixel_size
    if cli_args.condition:
        cli_overrides["condition"] = cli_args.condition
    if cli_args.no_cache:
        cli_overrides.setdefault("segmentation", {})["force_recompute"] = True
    if cli_args.allow_uncalibrated:
        cli_overrides["allow_uncalibrated"] = True

    config_path = Path(cli_args.config)
    if config_path.exists():
        config = PipelineConfig.from_yaml(config_path, cli_overrides=cli_overrides)
    else:
        config = PipelineConfig()
        if "output_dir" in cli_overrides:
            config.output_dir = cli_overrides["output_dir"]
        if "review_dir" in cli_overrides:
            config.review_dir = cli_overrides["review_dir"]
        if "t0_dir" in cli_overrides:
            config.t0_dir = cli_overrides["t0_dir"]
        if "t7_dir" in cli_overrides:
            config.t7_dir = cli_overrides["t7_dir"]
        if "cli_pixel_size_um" in cli_overrides:
            config.cli_pixel_size_um = cli_overrides["cli_pixel_size_um"]
        if cli_args.no_cache:
            config.segmentation.force_recompute = True
        if cli_args.allow_uncalibrated:
            config.allow_uncalibrated = True

    # 3. Handle Preflight-Only Mode
    if cli_args.preflight_only:
        print("\n[*] Executing Preflight Integrity & Optical Calibration Audit...\n")
        report = PreflightAuditor(config).run_preflight_check()
        print(report.format_summary())
        return 0 if report.passed else 1

    # 4. Handle Presentation Plots Generation Mode
    if cli_args.plots:
        out_p = Path(cli_args.output_dir)
        objs_p = out_p / "objects.csv"
        pairs_p = out_p / "pairs.csv"
        if not objs_p.exists():
            logger.error(f"Cannot generate presentation figures: '{objs_p}' does not exist. Run pipeline first.")
            return 1
        print(f"\n[*] Generating presentation figures in {out_p}/figures/ ...\n")
        objs_df = pd.read_csv(objs_p)
        pairs_df = pd.read_csv(pairs_p) if pairs_p.exists() else pd.DataFrame()
        wells_csv = out_p / "well_growth_comparison.csv"
        wells_df = compute_well_growth_comparison(objs_df, wells_csv)
        generate_all_presentation_figures(out_p, objs_df, pairs_df, wells_df)
        print(f"[+] Successfully generated presentation figures in {out_p}/figures/\n")
        return 0

    # 5. Handle Sample Mode
    if cli_args.sample is not None:
        sample_n = cli_args.sample
        print(f"\n[*] Launching Stratified {sample_n}-Sample Execution...\n")
        res = run_spheroid_pipeline(
            config,
            sample_n=sample_n,
            is_sample_12_loop=False,
            max_tuning_cycles=1,
        )
        return 0 if res.get("status") == "SUCCESS" else 1

    # 6. Handle Full Dataset or Default Execution
    print("\n[*] Launching Full Dataset Execution across all raw images...\n")
    res = run_spheroid_pipeline(
        config,
        sample_n=None,
        is_sample_12_loop=False,
    )
    return 0 if res.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
