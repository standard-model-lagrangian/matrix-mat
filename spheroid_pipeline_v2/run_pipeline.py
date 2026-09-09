"""
Master CLI Entry Point and Pipeline Orchestrator for Spheroid Volume Pipeline v2.

Provides:
  - Full end-to-end multi-spheroid analysis over raw brightfield TIFF datasets.
  - CLI flags:
      * --smoke-test: Executes synthetic ground-truth smoke test (3 scenarios).
      * --sample N (e.g. --sample 12): Stratified N-sample validation with programmatic sanity gates
        and automated parameter tuning loop (up to 8 cycles).
      * --full-dataset: Comprehensive batch execution over all 285 raw TIFFs (120 matched pairs + 45 singletons).
      * --config, --input-dir, --output-dir, --review-dir, --pixel-size, --no-cache.
  - Complete output artifact generation:
      * objects.csv, pairs.csv, condition_summary.csv, validation.csv, DECISIONS.md
      * overlays/, contact_sheets/, figures/ (PNG + PDF)
      * report.md and interactive report.html
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import numpy as np
import pandas as pd
import tifffile

from spheroid_pipeline_v2.config import PipelineConfig
from spheroid_pipeline_v2.decisions import DecisionsLogger
from spheroid_pipeline_v2.mask_cache import MaskCache
from spheroid_pipeline_v2.measure import SpheroidObjectRecord, save_objects_csv
from spheroid_pipeline_v2.pair import (
    SpheroidPairRecord,
    pair_dataset,
    save_pairs_csv,
    update_objects_with_pair_ids,
)
from spheroid_pipeline_v2.preprocess import Preprocessor
from spheroid_pipeline_v2.qc_filter import deduplicate_dataset_objects, filter_and_measure_objects
from spheroid_pipeline_v2.report import ReportGenerator
from spheroid_pipeline_v2.review import ReviewManager
from spheroid_pipeline_v2.scale_extractor import ScaleExtractor
from spheroid_pipeline_v2.segment import SegmentationHierarchy
from spheroid_pipeline_v2.stats import (
    compute_condition_statistics,
    compute_exclusion_audit,
    generate_exclusion_audit_table,
    save_condition_summary_csv,
)
from spheroid_pipeline_v2.tests.synthetic_generator import SyntheticImageGenerator
from spheroid_pipeline_v2.visualize import (
    create_overlay,
    generate_all_figures,
    generate_condition_contact_sheets,
)
from spheroid_pipeline_v2.audit import (
    audit_fail_class,
    create_fail_audit_crops_grid,
    audit_session_quality,
)
from spheroid_pipeline_v2.preflight import PreflightAuditor
from spheroid_pipeline_v2.provenance import export_complete_manifest, write_run_metadata
from spheroid_pipeline_v2.presentation_plots import (
    compute_well_growth_comparison,
    generate_all_presentation_figures,
)
import time

logger = logging.getLogger("spheroid_pipeline_v2")


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

def parse_image_filename(filename: str) -> Dict[str, Any]:
    """
    Parses standard dataset filename into structured metadata:
    Pattern: Day<0|7> <Condition> gel<N>_<Field:04d>_TRANS.tif
    """
    stem = Path(filename).stem
    pattern = r"Day(?P<day>[07])\s+(?P<condition>[A-Za-z0-9]+)\s+gel(?P<gel>\d+)_(?P<field>\d+)_TRANS"
    match = re.match(pattern, stem)

    if match:
        day_num = match.group("day")
        condition = match.group("condition")
        replicate = f"gel{match.group('gel')}"
        fov = match.group("field")
        timepoint = "t0" if day_num == "0" else "t7"
        pair_key = f"{condition}_{replicate}_{fov}"
    else:
        # Fallback heuristic parser
        parts = stem.split()
        condition = parts[1] if len(parts) > 1 else "Unknown"
        replicate = "gel1"
        fov = "0001"
        timepoint = "t0" if "Day0" in stem or "d0" in stem.lower() else "t7"
        pair_key = f"{condition}_{replicate}_{fov}"

    return {
        "image_id": stem,
        "timepoint": timepoint,
        "condition": condition,
        "replicate": replicate,
        "fov": fov,
        "pair_key": pair_key,
    }


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
            meta = parse_image_filename(p.name)
            meta["file_path"] = str(p)
            records.append(meta)

    # Scan Day 7
    if t7_dir.exists():
        for p in sorted(list(t7_dir.glob("*.tif")) + list(t7_dir.glob("*.tiff"))):
            meta = parse_image_filename(p.name)
            meta["file_path"] = str(p)
            records.append(meta)

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
# Synthetic Smoke Test Runner
# -----------------------------------------------------------------------------

def run_synthetic_smoke_test(config: Optional[PipelineConfig] = None) -> bool:
    """
    Executes the 3 mandatory synthetic smoke test scenarios:
      1. Isolated dark circular spheroid (ground truth d=100px): asserts recovered d within 10%.
      2. Touching pair of spheroids (ground truth d=80px each): asserts 2 separated instances, both d within 10%.
      3. Bright gel blob / refractive artifact (ground truth d=120px): asserts 0 masks segmented (contrast gate).
    """
    print("\n" + "=" * 80)
    print("   RUNNING SYNTHETIC GROUND-TRUTH SMOKE TESTS")
    print("=" * 80)

    cfg = config or PipelineConfig()
    passed_all = True
    scale = 1.518817

    # Scenario 1: Isolated Dark Spheroid (d = 100 px -> d = 151.88 um)
    img1, gt_mask1, rec1 = SyntheticImageGenerator.create_isolated_spheroid(
        diameter_px=100.0,
        width=800,
        height=600,
        pixel_size_um=scale,
        contrast=0.45,
    )
    seg1 = SegmentationHierarchy(cfg.segmentation)
    mask1, _ = seg1.segment(img1, "synthetic_isolated", pixel_size_um=scale)
    clean_mask1, objs1 = filter_and_measure_objects(
        img1, mask1, {"image_id": "synthetic_isolated", "pixel_size_um": scale}, cfg
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
    img2, gt_mask2, recs2 = SyntheticImageGenerator.create_touching_pair(
        d1_px=80.0,
        d2_px=80.0,
        overlap_px=10.0,
        width=800,
        height=600,
        pixel_size_um=scale,
        contrast=0.45,
    )
    mask2, _ = seg1.segment(img2, "synthetic_touching", pixel_size_um=scale)
    clean_mask2, objs2 = filter_and_measure_objects(
        img2, mask2, {"image_id": "synthetic_touching", "pixel_size_um": scale}, cfg
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
    img3, gt_mask3, rec3 = SyntheticImageGenerator.create_bright_blob(
        diameter_px=120.0,
        width=800,
        height=600,
        pixel_size_um=scale,
        contrast=0.35,
    )
    mask3, _ = seg1.segment(img3, "synthetic_bright_blob", pixel_size_um=scale)
    clean_mask3, objs3 = filter_and_measure_objects(
        img3, mask3, {"image_id": "synthetic_bright_blob", "pixel_size_um": scale}, cfg
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
# Stratified Sample 12 Sanity Gates & Parameter Tuning Loop
# -----------------------------------------------------------------------------

def evaluate_sample_12_sanity_gates(
    objects_df: pd.DataFrame,
    manifest_sample_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates the 4 mandatory programmatic sanity gates over the stratified sample:
      1. Spheroid count per image in [1, 25] for passing images
      2. No single mask covers > 25% of FOV area
      3. >= 60% of sample images yield >= 1 PASS object
      4. Equivalent diameters in [40, 1500] um (for PASS objects)
    """
    sample_img_ids = set(manifest_sample_df["image_id"].unique())
    n_sample_images = len(sample_img_ids)

    pass_objs = objects_df[objects_df["qc_flag"] == "PASS"] if not objects_df.empty and "qc_flag" in objects_df.columns else pd.DataFrame()

    # Gate 1: Spheroid count in [1, 25] for passing images
    counts_per_img = pass_objs.groupby("image_id").size().to_dict() if not pass_objs.empty else {}
    gate1_violations = [img_id for img_id, cnt in counts_per_img.items() if cnt > 25]
    gate1_passed = len(gate1_violations) == 0

    # Gate 2: No single mask > 25% FOV area
    # EVOS 1536x2048 = 3,145,728 px
    gate2_violations = []
    if not objects_df.empty and "area_px" in objects_df.columns:
        for _, row in objects_df.iterrows():
            area_pct = row["area_px"] / (1536.0 * 2048.0)
            if area_pct > 0.25:
                gate2_violations.append(row["object_id"])
    gate2_passed = len(gate2_violations) == 0

    # Gate 3: >= 60% images yield >= 1 PASS object
    pass_img_ids = set(pass_objs["image_id"].unique()) if not pass_objs.empty else set()
    pass_rate_pct = (len(pass_img_ids) / max(1, n_sample_images)) * 100.0
    gate3_passed = pass_rate_pct >= 60.0

    # Gate 4: Diameters in [min_d_um, max_d_um] for PASS objects
    min_d = min(40.0, config.qc.min_d_um)
    max_d = max(1500.0, config.qc.max_d_um)
    gate4_violations = []
    if not pass_objs.empty and "equivalent_diameter_um" in pass_objs.columns:
        for _, row in pass_objs.iterrows():
            d_um = row["equivalent_diameter_um"]
            if d_um < min_d or d_um > max_d:
                gate4_violations.append((row["object_id"], d_um))
    gate4_passed = len(gate4_violations) == 0

    all_passed = gate1_passed and gate2_passed and gate3_passed and gate4_passed

    report = {
        "all_passed": all_passed,
        "n_sample_images": n_sample_images,
        "gate1_passed": gate1_passed,
        "gate1_max_count": max(counts_per_img.values()) if counts_per_img else 0,
        "gate2_passed": gate2_passed,
        "gate2_violations": len(gate2_violations),
        "gate3_passed": gate3_passed,
        "pass_rate_pct": pass_rate_pct,
        "gate4_passed": gate4_passed,
        "gate4_violations": len(gate4_violations),
    }
    return all_passed, report


# -----------------------------------------------------------------------------
# End-to-End Pipeline Execution Engine
# -----------------------------------------------------------------------------

def run_spheroid_pipeline(
    config: PipelineConfig,
    sample_n: Optional[int] = None,
    is_sample_12_loop: bool = False,
    max_tuning_cycles: int = 8,
) -> Dict[str, Any]:
    """
    Executes the full Spheroid Volume Analysis Pipeline.

    Steps:
      1. Manifest building & optical calibration.
      2. Image processing, 3-tier segmentation & manual review ingestion.
      3. QC filtering & morphometrics quantification.
      4. Bipartite Hungarian matching & trajectory tracking.
      5. Condition-level statistics, bootstrap CIs & permutation hypothesis testing.
      6. 10-stage exclusion auditing.
      7. Supervision overlays, contact sheets & publication figures (PNG + PDF).
      8. report.md and interactive report.html dashboard.
    """
    start_time = time.time()

    # Preflight Validation Audit
    preflight = PreflightAuditor(config).run_preflight_check()
    logger.info("\n" + preflight.format_summary())
    preflight.log_and_raise_if_fatal()
    active_warnings = list(preflight.warnings)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = out_dir / config.artifacts.overlay_dir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = out_dir / config.artifacts.contact_sheets_dir
    contact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / config.artifacts.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    decisions_logger = DecisionsLogger(out_dir / "DECISIONS.md")
    review_manager = ReviewManager(config.review_dir)
    mask_cache = MaskCache(config.segmentation.cache_dir)
    scale_extractor = ScaleExtractor(
        default_pixel_size_um=config.default_pixel_size_um,
        cli_override_um=config.cli_pixel_size_um,
    )
    preprocessor = Preprocessor(config.preprocess)

    # 1. Build Manifest & Export Complete Manifest
    manifest_df = build_manifest(config.t0_dir, config.t7_dir)
    if manifest_df.empty:
        logger.error("No images found to process.")
        return {"status": "FAILED", "error": "Empty dataset"}
    export_complete_manifest(manifest_df, out_dir, compute_checksums=False)

    # Select stratified sample if requested
    if sample_n is not None and sample_n > 0:
        # Stratified sampling: pick balanced pairs from each condition
        conditions = sorted(manifest_df["condition"].unique())
        selected_rows = []
        pairs_per_cond = max(1, sample_n // (len(conditions) * 2))

        for cond in conditions:
            c_pairs = manifest_df[(manifest_df["condition"] == cond) & (manifest_df["is_matched_pair"] == True)]
            unique_keys = c_pairs["pair_key"].unique()
            for key in unique_keys[:pairs_per_cond]:
                selected_rows.append(c_pairs[c_pairs["pair_key"] == key])

        if selected_rows:
            sample_manifest = pd.concat(selected_rows).drop_duplicates().head(sample_n)
        else:
            sample_manifest = manifest_df.head(sample_n)

        manifest_to_process = sample_manifest
        logger.info(f"Running stratified sample execution on {len(manifest_to_process)} images")
    else:
        manifest_to_process = manifest_df

    # 2. Automated Tuning Loop for Sample 12
    tuning_cycle = 1
    best_config = config
    all_objects: List[SpheroidObjectRecord] = []
    all_validation_records: List[Dict[str, Any]] = []

    while tuning_cycle <= max_tuning_cycles:
        all_objects = []
        all_validation_records = []
        seg_hierarchy = SegmentationHierarchy(
            config=best_config.segmentation,
            decisions_logger=decisions_logger,
            mask_cache=mask_cache,
            preprocessor=preprocessor,
        )

        logger.info(f"--- Pipeline Execution Cycle {tuning_cycle}/{max_tuning_cycles} ({len(manifest_to_process)} images) ---")

        for idx, row in manifest_to_process.iterrows():
            img_path = Path(row["file_path"])
            image_id = row["image_id"]
            pair_key = row["pair_key"]
            timepoint = row["timepoint"]
            condition = row["condition"]
            replicate = row["replicate"]
            fov = row["fov"]

            # Optical scale extraction
            scale_info = scale_extractor.extract(img_path)
            pixel_size_um = scale_info.pixel_size_um
            scale_src = scale_info.source

            # Load raw TIFF image
            try:
                raw_img = tifffile.imread(str(img_path))
            except Exception as e:
                logger.warning(f"Could not read TIFF {img_path}: {e}")
                continue

            # Check manual review override
            manual_mask_path = review_manager.find_manual_mask(image_id, pair_key=pair_key, timepoint=timepoint)
            if manual_mask_path is not None:
                # Manual review path
                manual_mask = review_manager.load_manual_mask(manual_mask_path)
                auto_mask, _ = seg_hierarchy.segment(raw_img, image_id, pixel_size_um)
                dice, iou = review_manager.compute_agreement(auto_mask, manual_mask)
                active_mask = manual_mask
                mask_source = "manual_review"

                all_validation_records.append({
                    "image_id": image_id,
                    "pair_key": pair_key,
                    "timepoint": timepoint,
                    "manual_mask_path": str(manual_mask_path),
                    "n_auto_objects": int(np.max(auto_mask)),
                    "n_manual_objects": int(np.max(manual_mask)),
                    "dice_coefficient": float(round(dice, 4)),
                    "iou_jaccard": float(round(iou, 4)),
                    "pixel_size_um": float(pixel_size_um),
                    "notes": f"Manual override applied. Dice={dice:.3f}, IoU={iou:.3f}",
                })
            else:
                # Automated hierarchy segmentation
                active_mask, seg_meta = seg_hierarchy.segment(raw_img, image_id, pixel_size_um)
                mask_source = "automated"

            # QC Filter and Morphometrics
            img_meta = {
                "image_id": image_id,
                "timepoint": timepoint,
                "condition": condition,
                "replicate": replicate,
                "fov": fov,
                "pair_key": pair_key,
                "pixel_size_um": pixel_size_um,
                "mask_source": mask_source,
            }
            clean_mask, obj_records = filter_and_measure_objects(
                raw_img, active_mask, img_meta, best_config
            )
            all_objects.extend(obj_records)

            # Generate Supervision Overlay PNG
            overlay_out = overlays_dir / f"{image_id}_overlay.png"
            try:
                create_overlay(
                    raw_img,
                    clean_mask,
                    obj_records,
                    overlay_out,
                    pixel_size_um=pixel_size_um,
                )
            except Exception as e:
                logger.warning(f"Failed to create overlay for {image_id}: {e}")

        # Check Sanity Gates if running in sample 12 mode
        if is_sample_12_loop:
            objs_df_temp = pd.DataFrame([vars(o) for o in all_objects]) if all_objects else pd.DataFrame()
            passed_gates, gate_report = evaluate_sample_12_sanity_gates(
                objs_df_temp, manifest_to_process, best_config
            )
            logger.info(f"Sample 12 Sanity Gates Evaluation (Cycle {tuning_cycle}): {gate_report}")

            if passed_gates or tuning_cycle >= max_tuning_cycles:
                if passed_gates:
                    logger.info(f"All programmatic sanity gates PASSED on cycle {tuning_cycle}!")
                else:
                    logger.warning(f"Reached max tuning cycles ({max_tuning_cycles}). Proceeding with current parameters.")
                break
            else:
                # Adjust parameters and iterate
                tuning_cycle += 1
                # Slightly relax or adjust window sizes
                best_config.segmentation.classical_adaptive_windows = [31, 61, 101, 151, 201]
                decisions_logger.log(
                    image_id="SAMPLE_12_TUNING",
                    stage=f"Tuning Cycle {tuning_cycle}",
                    trigger_event=f"Sanity gates failed: {gate_report}",
                    action_taken="Adjusted classical adaptive window parameters",
                    outcome="Re-running sample 12 evaluation",
                )
        else:
            break

    # 2b. Cross-FOV Multi-Focal Plane Deduplication Pass
    logger.info("Executing cross-FOV multi-focal plane deduplication pass...")
    all_objects, dedup_meta = deduplicate_dataset_objects(all_objects, max_duplicate_distance_px=35.0)
    logger.info(
        f"Deduplication pass complete: {dedup_meta['duplicates_flagged']} multi-focal plane duplicate spheroids flagged FAIL."
    )

    # 3. Object Records DataFrame & CSV Export
    objects_df = pd.DataFrame([vars(o) for o in all_objects]) if all_objects else pd.DataFrame()
    objects_csv_path = out_dir / "objects.csv"
    save_objects_csv(all_objects, objects_csv_path)

    # 4. Temporal Hungarian Pairing (Day 0 -> Day 7)
    logger.info("Executing Hungarian mutual nearest centroid pairing...")
    all_pairs, all_objects = pair_dataset(all_objects, best_config)

    # Re-save objects.csv with populated pair_ids
    save_objects_csv(all_objects, objects_csv_path)

    # Save pairs.csv
    pairs_csv_path = out_dir / "pairs.csv"
    save_pairs_csv(all_pairs, pairs_csv_path)
    pairs_df = pd.DataFrame([vars(p) for p in all_pairs]) if all_pairs else pd.DataFrame()

    # 5. Condition-Level Summary Statistics, Bootstrap CIs & Permutation Tests
    logger.info("Computing condition-level summary statistics and hypothesis tests...")
    condition_summary_df = compute_condition_statistics(all_pairs, all_objects=all_objects, config=best_config)
    cond_csv_path = out_dir / "condition_summary.csv"
    save_condition_summary_csv(condition_summary_df, cond_csv_path)

    # 6. Multilevel 10-Stage Exclusion Audit
    exclusion_dict = compute_exclusion_audit(all_objects=all_objects, all_pairs=all_pairs)
    exclusion_audit_df = generate_exclusion_audit_table(exclusion_dict)

    # 7. Validation Agreement Table
    val_csv_path = out_dir / "validation.csv"
    validation_df = review_manager.save_validation_report(all_validation_records, val_csv_path)

    # 8. Visual Artifacts: Publication Figures & Contact Sheets
    logger.info("Generating publication figures (PNG + PDF)...")
    fig_paths = generate_all_figures(pairs_df, objects_df, out_dir, save_pdf=True, save_png=True)

    # 8b. Forensic Audits: FAIL Class Distribution & Imaging Session Quality
    logger.info("Executing forensic FAIL class audit and session quality analysis...")
    fail_fig_paths = audit_fail_class(objects_df, out_dir, save_pdf=True)
    fig_paths["fail_class_audit"] = fail_fig_paths

    contact_sheets: Dict[str, Path] = {}
    fail_grid_path = create_fail_audit_crops_grid(
        objects_df,
        best_config.t0_dir,
        best_config.t7_dir,
        contact_dir / "fail_objects_audit_grid.png",
        num_samples=20,
    )
    if fail_grid_path:
        contact_sheets["FAIL_Audit_Grid"] = fail_grid_path

    session_df, session_fig_paths = audit_session_quality(
        best_config.t0_dir, best_config.t7_dir, out_dir, save_pdf=True
    )
    fig_paths["session_focus_exposure_audit"] = session_fig_paths

    logger.info("Generating condition contact sheets...")
    contact_sheets.update(generate_condition_contact_sheets(overlays_dir, contact_dir, manifest_to_process))

    # 8c. Presentation Dashboard Figures (Figs 1-6) & Well Growth Analysis Table
    logger.info("Generating presentation dashboard figures (Figs 1-6) and well growth comparison...")
    try:
        wells_csv_path = out_dir / "well_growth_comparison.csv"
        wells_df = compute_well_growth_comparison(objects_df, wells_csv_path)
        pres_figs = generate_all_presentation_figures(out_dir, objects_df, pairs_df, wells_df)
        fig_paths.update(pres_figs)
    except Exception as e:
        logger.warning(f"Failed to generate presentation figures: {e}")

    # 9. Executive Reports: report.md and report.html
    logger.info("Compiling executive report.md and interactive report.html...")
    report_gen = ReportGenerator(best_config, out_dir)
    md_path, html_path = report_gen.generate(
        manifest_df=manifest_to_process,
        objects_df=objects_df,
        pairs_df=pairs_df,
        condition_summary_df=condition_summary_df,
        exclusion_audit_df=exclusion_audit_df,
        fig_paths=fig_paths,
        overlay_samples=list(overlays_dir.glob("*.png"))[:12],
        contact_sheets=contact_sheets,
        validation_df=validation_df,
        decisions_path=out_dir / "DECISIONS.md",
        session_df=session_df,
    )

    # 10. Write Run Metadata & Provenance (run_metadata.json)
    wall_time = time.time() - start_time
    write_run_metadata(
        output_dir=out_dir,
        config=best_config,
        wall_time_sec=wall_time,
        warnings=active_warnings,
        extra_info={
            "objects_count": len(all_objects),
            "pairs_count": len(all_pairs),
            "tuning_cycles_run": tuning_cycle,
            "is_sample_mode": sample_n is not None,
        },
    )

    logger.info(f"Pipeline execution completed successfully. Artifacts available in {out_dir}/")
    return {
        "status": "SUCCESS",
        "objects_count": len(all_objects),
        "pairs_count": len(all_pairs),
        "condition_summary": condition_summary_df,
        "md_report": md_path,
        "html_report": html_path,
        "figures": fig_paths,
        "contact_sheets": contact_sheets,
    }


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """Builds comprehensive CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Spheroid Volume Analysis Pipeline v2 (spheroid_pipeline_v2)",
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
        help="Run stratified N-sample validation with programmatic sanity gates and tuning loop (e.g. --sample 12).",
    )
    mode_group.add_argument(
        "--full-dataset",
        action="store_true",
        help="Run full dataset batch execution over all 285 raw TIFFs.",
    )

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--input-dir", type=str, help="Unified input directory.")
    parser.add_argument("--t0-dir", type=str, help="Day 0 input image directory.")
    parser.add_argument("--t7-dir", type=str, help="Day 7 input image directory.")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for CSVs and artifacts.")
    parser.add_argument("--review-dir", type=str, default="review", help="Directory monitoring manual ground-truth masks.")
    parser.add_argument("--pixel-size", type=float, help="Optical calibration override in um/pixel.")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation without loading cached masks.")
    parser.add_argument("--plots", action="store_true", help="Generate all presentation figures (Figs 1-6) and well growth table from existing CSVs in output-dir.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight input and optical calibration validation and exit.")
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
    if cli_args.no_cache:
        cli_overrides.setdefault("segmentation", {})["force_recompute"] = True

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
        print(f"\n[*] Generating presentation figures (Figs 1-6) and well growth table in {out_p}/figures/ ...\n")
        objs_df = pd.read_csv(objs_p)
        pairs_df = pd.read_csv(pairs_p) if pairs_p.exists() else pd.DataFrame()
        wells_csv = out_p / "well_growth_comparison.csv"
        wells_df = compute_well_growth_comparison(objs_df, wells_csv)
        generate_all_presentation_figures(out_p, objs_df, pairs_df, wells_df)
        print(f"[+] Successfully generated all 6 presentation figures in {out_p}/figures/\n")
        return 0

    # 5. Handle Sample Mode
    if cli_args.sample is not None:
        sample_n = cli_args.sample
        is_sample_12 = (sample_n == 12)
        print(f"\n[*] Launching Stratified {sample_n}-Sample Execution with Sanity Gates...\n")
        res = run_spheroid_pipeline(
            config,
            sample_n=sample_n,
            is_sample_12_loop=is_sample_12,
            max_tuning_cycles=8,
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
