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
import copy
import csv
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import cv2
import numpy as np
import pandas as pd

from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.brightfield.decisions import DecisionsLogger
from matrix_mat.core.cache import MaskCache
from matrix_mat.brightfield.io import list_tiff_files, load_grayscale_image
from matrix_mat.brightfield.measure import OBJECTS_CSV_COLUMNS, SpheroidObjectRecord, save_objects_csv
from matrix_mat.brightfield.pair import (
    SpheroidPairRecord,
    pair_dataset,
    save_pairs_csv,
)
from matrix_mat.brightfield.preprocess import Preprocessor, flatten_illumination, normalize_percentiles
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
from matrix_mat.core.naming import parse_image_filename

logger = logging.getLogger("matrix_mat.brightfield.run_pipeline")

# Per-image processing outcomes recorded in images_status.csv
IMAGE_STATUS_COLUMNS: List[str] = [
    "image_id", "pair_key", "timepoint", "condition", "replicate", "fov", "file_path",
    "status", "reason", "n_objects", "n_pass", "seg_backend", "mask_source", "cached",
    "pixel_size_um", "scale_source", "is_calibrated",
]
IMAGE_OK_STATUSES = ("OK",)

RUN_STATUS_SUCCESS = "SUCCESS"
RUN_STATUS_PARTIAL = "COMPLETED_WITH_ERRORS"
RUN_STATUS_FAILED = "FAILED"

INPUT_DIR_T0_CANDIDATES = ("Day0", "Day 0", "Spheroid Day0 260805", "t0")
INPUT_DIR_T7_CANDIDATES = ("Day7", "Day 7", "Spheroid D7 260812", "t7")


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
    logging.getLogger("matrix_mat").setLevel(level)


# -----------------------------------------------------------------------------
# Manifest & File Ingestion
# -----------------------------------------------------------------------------

def build_manifest(t0_dir: Union[str, Path], t7_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Scans Day 0 and Day 7 directories (.tif/.tiff, case-insensitive), builds the inventory
    manifest, and identifies matched pairs. Several files mapping to the same
    (pair_key, timepoint) are reported and only the first is kept.
    """
    t0_dir = Path(t0_dir)
    t7_dir = Path(t7_dir)

    records: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], str] = {}

    for directory, day in ((t0_dir, "Day0"), (t7_dir, "Day7")):
        for p in list_tiff_files(directory):
            try:
                meta = parse_image_filename(p.name, source_dir_timepoint=day)
            except ValueError as e:
                logger.warning(f"Skipping unmatched {day} image {p.name}: {e}")
                continue
            key = (meta["pair_key"], meta["timepoint"])
            if key in seen:
                if Path(seen[key]).resolve() != p.resolve():
                    logger.warning(
                        f"Duplicate image for {key[0]} {key[1]}: keeping {Path(seen[key]).name}, ignoring {p.name}"
                    )
                continue
            seen[key] = str(p)
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


def select_images(
    manifest_df: pd.DataFrame,
    condition: Optional[str] = None,
    sample_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Applies the --condition filter and the stratified --sample N selection.

    Sampling is round-robin across conditions: each round adds one matched FOV pair (both
    timepoints) per condition, then unmatched single images, until N images are selected, so a
    small N never drops later conditions entirely while larger conditions still have pairs left.
    """
    if manifest_df.empty:
        return manifest_df
    df = manifest_df
    if condition:
        df = df[df["condition"] == condition].copy()
    if sample_n is None or sample_n <= 0 or df.empty:
        return df

    units_by_cond: Dict[str, List[List[int]]] = {}
    for cond in sorted(df["condition"].unique()):
        c_df = df[df["condition"] == cond]
        units: List[List[int]] = []
        matched = c_df[c_df["is_matched_pair"].astype(bool)] if "is_matched_pair" in c_df.columns else c_df.iloc[0:0]
        for key in sorted(matched["pair_key"].unique()):
            units.append(list(matched.index[matched["pair_key"] == key]))
        for idx in c_df.index:
            if idx not in matched.index:
                units.append([idx])
        units_by_cond[cond] = units

    selected: List[int] = []
    cursors = {c: 0 for c in units_by_cond}
    progress = True
    while len(selected) < sample_n and progress:
        progress = False
        for cond, units in units_by_cond.items():
            if len(selected) >= sample_n:
                break
            i = cursors[cond]
            if i >= len(units):
                continue
            unit = units[i]
            if len(selected) + len(unit) > sample_n:
                unit = unit[: sample_n - len(selected)]
            selected.extend(unit)
            cursors[cond] = i + 1
            progress = True

    n_conds = len(units_by_cond)
    covered = df.loc[selected, "condition"].nunique() if selected else 0
    if covered < n_conds:
        logger.warning(
            f"--sample {sample_n} covers only {covered} of {n_conds} conditions; use N >= {2 * n_conds} "
            "for at least one Day0/Day7 pair per condition."
        )
    return df.loc[selected]


def resolve_input_dir(input_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Maps a unified --input-dir to (t0_dir, t7_dir) using the known Day 0 / Day 7 subfolder names."""
    in_p = Path(input_dir)
    found_t0 = next((in_p / c for c in INPUT_DIR_T0_CANDIDATES if (in_p / c).is_dir()), in_p)
    found_t7 = next((in_p / c for c in INPUT_DIR_T7_CANDIDATES if (in_p / c).is_dir()), in_p)
    return found_t0, found_t7


# -----------------------------------------------------------------------------
# Synthetic Smoke Test Runner (No test code imported)
# -----------------------------------------------------------------------------

def run_synthetic_smoke_test(config: Optional[PipelineConfig] = None) -> bool:
    """
    Executes the 3 synthetic smoke test scenarios:
      1. Isolated dark circular spheroid (ground truth d=100px): asserts recovered d within 10%.
      2. Touching pair of spheroids (ground truth d=80px each): asserts 2 separated instances, both d within 10%.
      3. Bright gel blob / refractive artifact (ground truth d=120px): asserts 0 masks segmented (contrast gate).

    Runs in a temporary directory with the mask cache disabled, so it never reads or writes the
    real cache, decisions log or output directory, and never mutates the caller's config.
    """
    logger.info("=" * 80)
    logger.info("   RUNNING SYNTHETIC GROUND-TRUTH SMOKE TESTS")
    logger.info("=" * 80)

    cfg = copy.deepcopy(config) if config is not None else PipelineConfig()
    cfg.segmentation.primary_backend = "classical"
    cfg.segmentation.force_recompute = True
    passed_all = True
    scale = 1.518817

    with tempfile.TemporaryDirectory(prefix="matrix_mat_smoke_") as td:
        tmp = Path(td)
        cfg.output_dir = str(tmp / "output")
        cfg.segmentation.cache_dir = str(tmp / "cache")
        cfg.segmentation.decisions_log_path = str(tmp / "decisions.jsonl")
        seg1 = SegmentationHierarchy(
            cfg.segmentation,
            decisions_logger=DecisionsLogger(tmp / "decisions.jsonl"),
            mask_cache=MaskCache(tmp / "cache", enabled=False),
        )

        # Scenario 1: Isolated Dark Spheroid (d = 100 px)
        img1 = np.full((600, 800), 0.65, dtype=np.float32)
        cv2.circle(img1, (400, 300), 50, 0.20, -1)
        cv2.circle(img1, (400, 300), 50, 0.15, 1)

        mask1, _ = seg1.segment(img1, "synthetic_isolated", pixel_size_um=scale, force_recompute=True)
        _, objs1 = filter_and_measure_objects(
            img1, mask1, {"image_id": "synthetic_isolated", "pixel_size_um": scale, "is_calibrated": True}, cfg
        )

        pass_objs1 = [o for o in objs1 if o.qc_flag == "PASS"]
        if len(pass_objs1) == 1:
            rec_d = pass_objs1[0].equivalent_diameter_px
            err = abs(rec_d - 100.0) / 100.0 * 100.0
            status1 = "PASS" if err <= 10.0 else "FAIL"
            logger.info(f"[*] Scenario 1 (Isolated Spheroid d=100px): [{status1}] (Recovered d={rec_d:.1f}px, Error={err:.2f}%)")
            if status1 == "FAIL":
                passed_all = False
        else:
            logger.error(f"[*] Scenario 1 (Isolated Spheroid): [FAIL] (Expected 1 PASS object, got {len(pass_objs1)})")
            passed_all = False

        # Scenario 2: Touching Pair (d1 = 80 px, d2 = 80 px)
        img2 = np.full((600, 800), 0.65, dtype=np.float32)
        cv2.circle(img2, (365, 300), 40, 0.20, -1)
        cv2.circle(img2, (435, 300), 40, 0.20, -1)
        cv2.circle(img2, (365, 300), 40, 0.15, 1)
        cv2.circle(img2, (435, 300), 40, 0.15, 1)

        mask2, _ = seg1.segment(img2, "synthetic_touching", pixel_size_um=scale, force_recompute=True)
        _, objs2 = filter_and_measure_objects(
            img2, mask2, {"image_id": "synthetic_touching", "pixel_size_um": scale, "is_calibrated": True}, cfg
        )

        pass_objs2 = [o for o in objs2 if o.qc_flag == "PASS"]
        if len(pass_objs2) == 2:
            d_vals = [o.equivalent_diameter_px for o in pass_objs2]
            errs = [abs(d - 80.0) / 80.0 * 100.0 for d in d_vals]
            status2 = "PASS" if all(e <= 10.0 for e in errs) else "FAIL"
            logger.info(
                f"[*] Scenario 2 (Touching Pair d=80px): [{status2}] (Separated into 2 instances, "
                f"d=[{d_vals[0]:.1f}, {d_vals[1]:.1f}], errs=[{errs[0]:.1f}%, {errs[1]:.1f}%])"
            )
            if status2 == "FAIL":
                passed_all = False
        else:
            logger.error(f"[*] Scenario 2 (Touching Pair): [FAIL] (Expected 2 PASS objects, got {len(pass_objs2)})")
            passed_all = False

        # Scenario 3: Bright Gel Blob (d = 120 px)
        img3 = np.full((600, 800), 0.65, dtype=np.float32)
        cv2.circle(img3, (400, 300), 60, 0.95, -1)

        mask3, _ = seg1.segment(img3, "synthetic_bright_blob", pixel_size_um=scale, force_recompute=True)
        _, objs3 = filter_and_measure_objects(
            img3, mask3, {"image_id": "synthetic_bright_blob", "pixel_size_um": scale, "is_calibrated": True}, cfg
        )

        pass_objs3 = [o for o in objs3 if o.qc_flag == "PASS"]
        status3 = "PASS" if len(pass_objs3) == 0 else "FAIL"
        logger.info(f"[*] Scenario 3 (Bright Gel Blob): [{status3}] (Expected 0 PASS objects, got {len(pass_objs3)})")
        if status3 == "FAIL":
            passed_all = False

    overall_status = "ALL SMOKE TESTS PASSED" if passed_all else "SMOKE TESTS FAILED"
    (logger.info if passed_all else logger.error)(f"[*] RESULT: {overall_status}")
    return passed_all


# -----------------------------------------------------------------------------
# Master Pipeline Orchestrator
# -----------------------------------------------------------------------------

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
        manifest_df = select_images(manifest_df, condition=config.condition)
        if manifest_df.empty:
            logger.error(f"No images found matching condition '{config.condition}'.")
            return pd.DataFrame(), active_warnings, out_dir
        logger.info(f"Filtered dataset to condition '{config.condition}': {len(manifest_df)} images found.")

    export_complete_manifest(
        manifest_df,
        out_dir,
        compute_checksums=True,
        default_pixel_size_um=config.default_pixel_size_um,
        pixel_size_override_um=config.cli_pixel_size_um,
    )

    effective_sample_n = sample_n if sample_n is not None else config.sample_n
    if effective_sample_n is not None and effective_sample_n > 0:
        manifest_to_process = select_images(manifest_df, sample_n=effective_sample_n)
        logger.info(
            f"Running stratified sample execution on {len(manifest_to_process)} images across "
            f"{manifest_to_process['condition'].nunique()} conditions"
        )
    else:
        manifest_to_process = manifest_df

    return manifest_to_process, active_warnings, out_dir


def _status_row(row: pd.Series, **kw: Any) -> Dict[str, Any]:
    base = {
        "image_id": row.get("image_id"), "pair_key": row.get("pair_key"), "timepoint": row.get("timepoint"),
        "condition": row.get("condition"), "replicate": row.get("replicate"), "fov": row.get("fov"),
        "file_path": row.get("file_path"), "status": "OK", "reason": "", "n_objects": 0, "n_pass": 0,
        "seg_backend": None, "mask_source": None, "cached": None, "pixel_size_um": None,
        "scale_source": None, "is_calibrated": None,
    }
    base.update(kw)
    return base


def save_image_status_csv(rows: List[Dict[str, Any]], output_path: Union[str, Path]) -> pd.DataFrame:
    """Writes the per-image processing status table (images_status.csv)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=IMAGE_STATUS_COLUMNS)
    df.to_csv(output_path, index=False)
    return df


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _compress_mask(mask: np.ndarray) -> Tuple[bytes, Tuple[int, ...], str]:
    arr = np.ascontiguousarray(mask)
    return zlib.compress(arr.tobytes(), 1), arr.shape, arr.dtype.str


def _decompress_mask(blob: Tuple[bytes, Tuple[int, ...], str]) -> np.ndarray:
    data, shape, dtype = blob
    return np.frombuffer(zlib.decompress(data), dtype=np.dtype(dtype)).reshape(shape)


def segment_cohort(
    manifest_to_process: pd.DataFrame,
    config: PipelineConfig,
    out_dir: Path,
    run_state: Optional[Dict[str, Any]] = None,
) -> Tuple[List[SpheroidObjectRecord], List[Dict[str, Any]]]:
    """
    Segments images across the cohort, measures objects, deduplicates across FOVs and generates
    supervision overlays (drawn with the final, post-deduplication QC flags).

    Every image gets a row in ``<out_dir>/images_status.csv`` (status OK / READ_FAIL / SEG_FAIL /
    PROCESS_FAIL), and object rows are appended to ``objects.csv`` as each image completes so an
    interrupted run keeps its segmentation results. ``run_state`` (optional dict) receives
    ``image_status`` and ``overlay_paths`` for later stages.
    """
    from matrix_mat.reporting.visualize import create_overlay

    run_state = run_state if run_state is not None else {}
    overlays_dir = out_dir / config.artifacts.overlay_dir
    overlays_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = out_dir / "corrected_images"
    if config.preprocess.save_corrected_images:
        corrected_dir.mkdir(parents=True, exist_ok=True)
    # One decisions log per run: truncate any previous run's records.
    decisions_logger = DecisionsLogger(out_dir / "decisions.jsonl", reset=True)
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
    image_status: List[Dict[str, Any]] = run_state.setdefault("image_status", [])
    overlay_paths: List[Path] = run_state.setdefault("overlay_paths", [])
    overlay_inputs: Dict[str, Dict[str, Any]] = {}

    # Incremental objects.csv (rewritten with final flags and pair ids by pair_timepoints)
    objects_csv_path = out_dir / "objects.csv"
    with open(objects_csv_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(OBJECTS_CSV_COLUMNS)

    n_total = len(manifest_to_process)
    t_start = time.time()
    logger.info(f"--- Pipeline Execution ({n_total} images) ---")
    for i, (_, row) in enumerate(manifest_to_process.iterrows(), start=1):
        img_path = Path(row["file_path"])
        image_id, pair_key = row["image_id"], row["pair_key"]
        timepoint, condition, replicate, fov = row["timepoint"], row["condition"], row["replicate"], row["fov"]

        scale_info = scale_extractor.extract(img_path)
        pixel_size_um = scale_info.pixel_size_um
        scale_kw = {"pixel_size_um": pixel_size_um, "scale_source": scale_info.source,
                    "is_calibrated": scale_info.is_calibrated}

        try:
            # Single 2-D plane at native bit depth; RGB(A) TIFFs are converted to grayscale here.
            raw_img = load_grayscale_image(img_path)
        except Exception as e:
            logger.error(f"[{i}/{n_total}] Could not read TIFF {img_path}: {e}")
            image_status.append(_status_row(row, status="READ_FAIL", reason=str(e), **scale_kw))
            continue

        if config.preprocess.save_corrected_images:
            try:
                norm, _, _ = normalize_percentiles(
                    flatten_illumination(raw_img, sigma=config.preprocess.flat_field_sigma),
                    p_low=config.preprocess.percentile_low, p_high=config.preprocess.percentile_high,
                )
                Preprocessor.save_image(norm, corrected_dir / f"{image_id}.png")
            except Exception as e:
                logger.warning(f"Could not save corrected image for {image_id}: {e}")

        status_note = ""
        manual_mask_path = review_manager.find_manual_mask(image_id, pair_key=pair_key, timepoint=timepoint)
        manual_mask = None
        if manual_mask_path is not None:
            try:
                manual_mask = review_manager.load_manual_mask(manual_mask_path, expected_shape=raw_img.shape[:2])
            except Exception as e:
                status_note = f"manual mask rejected ({e}); automated mask used"
                logger.error(f"{image_id}: {status_note}")

        if manual_mask is not None:
            auto_mask, _ = seg_hierarchy.segment(raw_img, image_id, pixel_size_um)
            dice, iou = review_manager.compute_agreement(auto_mask, manual_mask)
            active_mask, mask_source = manual_mask, "manual_review"
            seg_meta: Dict[str, Any] = {"backend": "manual_review", "model_id": "human", "diameter_used": None}
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
        common_kw = dict(seg_backend=img_meta["seg_backend"], mask_source=mask_source,
                         cached=bool(seg_meta.get("cached", False)), **scale_kw)
        try:
            clean_mask, obj_records = filter_and_measure_objects(raw_img, active_mask, img_meta, config)
        except Exception as e:
            logger.exception(f"[{i}/{n_total}] QC/measurement failed for {image_id}: {e!r}")
            image_status.append(_status_row(row, status="PROCESS_FAIL", reason=repr(e), **common_kw))
            continue
        all_objects.extend(obj_records)

        n_pass = sum(1 for r in obj_records if r.qc_flag == "PASS")
        if seg_meta.get("qc_flag") == "SEG_FAIL":
            status, reason = "SEG_FAIL", str(seg_meta.get("qc_reasons") or "implausible segmentation")
        else:
            status, reason = "OK", status_note
        if status_note and status != "OK":
            reason = f"{reason}; {status_note}"
        image_status.append(_status_row(row, status=status, reason=reason, n_objects=len(obj_records),
                                        n_pass=n_pass, **common_kw))

        with open(objects_csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for rec in obj_records:
                writer.writerow([rec.to_dict()[c] for c in OBJECTS_CSV_COLUMNS])

        # Only objects present in the clean label map are drawn (their label_idx indexes it).
        in_map = [r for r in obj_records if r.qc_flag not in ("FAIL", "SEG_FAIL")]
        overlay_out = overlays_dir / f"{image_id}_overlay.png"
        overlay_inputs[image_id] = {
            "path": img_path, "mask": _compress_mask(clean_mask), "records": in_map,
            "pixel_size_um": pixel_size_um, "out": overlay_out,
        }
        try:
            create_overlay(raw_img, clean_mask, in_map, overlay_out, pixel_size_um=pixel_size_um)
            overlay_paths.append(overlay_out)
        except Exception as e:
            logger.warning(f"Failed to create overlay for {image_id}: {e}")

        elapsed = time.time() - t_start
        eta = elapsed / i * (n_total - i)
        logger.info(
            f"[{i}/{n_total}] {image_id}: {status}, {len(obj_records)} objects ({n_pass} PASS) "
            f"| elapsed {_format_eta(elapsed)}, ETA {_format_eta(eta)}"
        )

    logger.info("Executing cross-FOV multi-focal plane deduplication pass...")
    all_objects, dedup_meta = deduplicate_dataset_objects(
        all_objects,
        max_duplicate_distance_um=float(config.qc.dedup_max_distance_um),
        enabled=bool(config.qc.dedup_cross_fov),
    )
    logger.info(f"Deduplication pass complete: {dedup_meta['duplicates_flagged']} duplicates flagged FAIL.")

    # Redraw overlays whose objects changed flag during deduplication.
    flagged = set(dedup_meta.get("flagged_object_ids", []))
    if flagged:
        for image_id, info in overlay_inputs.items():
            if not any(r.object_id in flagged for r in info["records"]):
                continue
            try:
                raw_img = load_grayscale_image(info["path"])
                create_overlay(raw_img, _decompress_mask(info["mask"]), info["records"], info["out"],
                               pixel_size_um=info["pixel_size_um"])
            except Exception as e:
                logger.warning(f"Failed to redraw overlay for {image_id} after deduplication: {e}")

    save_image_status_csv(image_status, out_dir / "images_status.csv")
    n_failed = sum(1 for r in image_status if r["status"] not in IMAGE_OK_STATUSES)
    if n_failed:
        logger.warning(f"{n_failed}/{len(image_status)} images failed (see images_status.csv)")
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
    image_status: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes condition-level summary statistics and 10-stage exclusion audit."""
    logger.info("Computing condition-level summary statistics and hypothesis tests...")
    condition_summary_df = compute_condition_statistics(
        all_pairs, all_objects=all_objects, config=config, random_seed=config.random_seed
    )
    save_condition_summary_csv(condition_summary_df, out_dir / "condition_summary.csv")

    counts: Dict[str, int] = {}
    if image_status:
        st = pd.DataFrame(image_status)
        counts["t0_images_count"] = int((st["timepoint"] == "t0").sum())
        counts["t7_images_count"] = int((st["timepoint"] != "t0").sum())
        counts["plausibility_fallbacks"] = int(
            (st["status"] == "SEG_FAIL").sum()
            + ((st["status"] == "OK") & (st["seg_backend"].notna())
               & (~st["seg_backend"].isin([config.segmentation.primary_backend, "manual_review"]))).sum()
        )
    exclusion_dict = compute_exclusion_audit(all_objects=all_objects, all_pairs=all_pairs, **counts)
    exclusion_audit_df = generate_exclusion_audit_table(exclusion_dict)
    return condition_summary_df, exclusion_audit_df


def _run_step(name: str, fn: Callable[[], Any], failures: List[Dict[str, str]], default: Any = None) -> Any:
    """Runs one report step; a failure is logged and recorded instead of aborting later steps."""
    try:
        return fn()
    except Exception as e:
        logger.exception(f"Report step '{name}' failed: {e!r}")
        failures.append({"step": name, "error": repr(e)})
        return default


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
    overlay_paths: Optional[List[Path]] = None,
    image_status: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generates figures, contact sheets, presentation plots, and markdown/HTML reports.

    Each step runs independently: a failing figure is recorded in ``failed_steps`` and the
    remaining steps still run. The returned status is SUCCESS only when every step succeeded and
    at least one image was processed; COMPLETED_WITH_ERRORS when a step failed; FAILED when no
    image could be processed.
    """
    from matrix_mat.reporting.visualize import (
        generate_all_figures,
        generate_condition_contact_sheets,
    )
    from matrix_mat.reporting.presentation_plots import (
        compute_well_growth_comparison,
        generate_all_presentation_figures,
    )
    from matrix_mat.reporting.report import ReportGenerator

    failures: List[Dict[str, str]] = []
    overlays_dir = out_dir / config.artifacts.overlay_dir
    contact_dir = out_dir / config.artifacts.contact_sheets_dir
    review_manager = ReviewManager(config.review_dir)
    validation_df = _run_step(
        "validation_csv",
        lambda: review_manager.save_validation_report(all_validation_records, out_dir / "validation.csv"),
        failures, default=pd.DataFrame(),
    )

    logger.info("Generating publication figures (PNG + PDF)...")
    fig_paths: Dict[str, Any] = _run_step(
        "figures",
        lambda: generate_all_figures(
            pairs_df, objects_df, out_dir, save_pdf=True, save_png=True,
            condition_df=condition_summary_df, config=config,
        ),
        failures, default={},
    ) or {}

    fail_fig_paths = _run_step("fail_class_audit", lambda: audit_fail_class(objects_df, out_dir, save_pdf=True, qc_config=config.qc), failures)
    if fail_fig_paths:
        fig_paths["fail_class_audit"] = fail_fig_paths

    contact_sheets: Dict[str, Path] = {}
    fail_grid_path = _run_step(
        "fail_audit_grid",
        lambda: create_fail_audit_crops_grid(
            objects_df, config.t0_dir, config.t7_dir, contact_dir / "fail_objects_audit_grid.png", num_samples=20
        ),
        failures,
    )
    if fail_grid_path:
        contact_sheets["FAIL_Audit_Grid"] = fail_grid_path

    session_res = _run_step(
        "session_quality_audit",
        lambda: audit_session_quality(config.t0_dir, config.t7_dir, out_dir, save_pdf=True, manifest_df=manifest_to_process),
        failures,
    )
    session_df = session_res[0] if session_res else pd.DataFrame()
    if session_res:
        fig_paths["session_focus_exposure_audit"] = session_res[1]

    sheets = _run_step(
        "contact_sheets",
        lambda: generate_condition_contact_sheets(
            overlays_dir, contact_dir, manifest_to_process, overlay_paths=overlay_paths
        ),
        failures, default={},
    )
    if sheets:
        contact_sheets.update(sheets)

    def _presentation() -> Dict[str, Any]:
        wells_df = compute_well_growth_comparison(objects_df, out_dir / "well_growth_comparison.csv")
        return generate_all_presentation_figures(out_dir, objects_df, pairs_df, wells_df, condition_df=condition_summary_df)

    pres_figs = _run_step("presentation_figures", _presentation, failures, default={})
    if pres_figs:
        fig_paths.update(pres_figs)

    # Overlay samples: only overlays written by THIS run, in a stable order.
    produced = sorted({Path(p) for p in (overlay_paths or []) if Path(p).exists()})
    overlay_samples = produced[:12]

    report_paths = _run_step(
        "report",
        lambda: ReportGenerator(config, out_dir).generate(
            manifest_df=manifest_to_process, objects_df=objects_df, pairs_df=pairs_df,
            condition_summary_df=condition_summary_df, exclusion_audit_df=exclusion_audit_df,
            fig_paths=fig_paths, overlay_samples=overlay_samples,
            contact_sheets=contact_sheets, validation_df=validation_df,
            decisions_path=out_dir / "decisions.jsonl", session_df=session_df,
        ),
        failures,
    )
    md_path, html_path = report_paths if report_paths else (None, None)

    statuses = [r.get("status") for r in (image_status or [])]
    n_images = len(statuses)
    n_ok = sum(1 for s in statuses if s in IMAGE_OK_STATUSES)
    image_failures = [r for r in (image_status or []) if r.get("status") not in IMAGE_OK_STATUSES]
    if image_status is not None and n_images > 0 and n_ok == 0:
        run_status = RUN_STATUS_FAILED
    elif failures:
        run_status = RUN_STATUS_PARTIAL
    else:
        run_status = RUN_STATUS_SUCCESS

    wall_time = time.time() - start_time
    _run_step(
        "run_metadata",
        lambda: write_run_metadata(
            output_dir=out_dir, config=config, wall_time_sec=wall_time,
            warnings=active_warnings, backend=config.segmentation.primary_backend,
            extra_info={
                "objects_count": len(all_objects), "pairs_count": len(all_pairs),
                "is_sample_mode": sample_n is not None, "run_status": run_status,
                "failed_steps": failures, "images_processed": n_images,
                "images_failed": len(image_failures),
            },
        ),
        failures,
    )
    if failures and run_status == RUN_STATUS_SUCCESS:
        run_status = RUN_STATUS_PARTIAL

    if run_status == RUN_STATUS_SUCCESS:
        logger.info(f"Pipeline execution completed successfully. Artifacts available in {out_dir}/")
    elif run_status == RUN_STATUS_PARTIAL:
        logger.error(
            f"Pipeline completed WITH ERRORS in {len(failures)} report step(s): "
            f"{', '.join(f['step'] for f in failures)}. Artifacts in {out_dir}/"
        )
    else:
        logger.error(f"Pipeline FAILED: none of the {n_images} images could be processed (see images_status.csv).")
    return {
        "status": run_status, "objects_count": len(all_objects), "pairs_count": len(all_pairs),
        "condition_summary": condition_summary_df, "md_report": md_path,
        "html_report": html_path, "figures": fig_paths, "contact_sheets": contact_sheets,
        "failed_steps": failures, "image_failures": image_failures,
    }


def run_spheroid_pipeline(
    config: Optional[PipelineConfig] = None,
    sample_n: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Master pipeline runner function for spheroid volume changes and kinetics.
    Decomposed into clean, modular steps (discover_inputs, segment_cohort,
    pair_timepoints, compute_stats, render_reports).

    Returns a dict whose ``status`` is SUCCESS, COMPLETED_WITH_ERRORS (a report step failed;
    see ``failed_steps``) or FAILED (no images found / none could be processed).
    """
    start_time = time.time()
    config = config or PipelineConfig()

    logger.info("=" * 80)
    logger.info("       LAUNCHING BRIGHTFIELD SPHEROID ANALYSIS PIPELINE (matrix-mat)           ")
    logger.info("=" * 80)

    manifest_to_process, active_warnings, out_dir = discover_inputs(config, sample_n)
    if manifest_to_process.empty:
        return {"status": RUN_STATUS_FAILED, "error": "No images found to process"}

    run_state: Dict[str, Any] = {}
    all_objects, all_validation_records = segment_cohort(manifest_to_process, config, out_dir, run_state=run_state)
    image_status = run_state.get("image_status", [])
    all_pairs, all_objects, objects_df, pairs_df = pair_timepoints(all_objects, config, out_dir)
    condition_summary_df, exclusion_audit_df = compute_stats(all_pairs, all_objects, config, out_dir, image_status=image_status)

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
        overlay_paths=run_state.get("overlay_paths"),
        image_status=image_status,
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
        help="Execute synthetic ground-truth smoke test (3 scenarios) in a temporary directory.",
    )
    mode_group.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Process a stratified sample of N images (round-robin Day0/Day7 pairs across conditions).",
    )
    mode_group.add_argument(
        "--full-dataset",
        action="store_true",
        help="No-op kept for backward compatibility: the full dataset is processed whenever --sample is not given.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/spheroid_brightfield.yaml",
        help="Path to YAML configuration file (must exist).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Unified input directory containing Day 0 / Day 7 subfolders "
             f"(one of {', '.join(INPUT_DIR_T0_CANDIDATES)} / {', '.join(INPUT_DIR_T7_CANDIDATES)}); "
             "--t0-dir / --t7-dir take precedence.",
    )
    parser.add_argument("--t0-dir", type=str, help="Day 0 input image directory.")
    parser.add_argument("--t7-dir", type=str, help="Day 7 input image directory.")
    parser.add_argument(
        "-o", "--out", "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Output directory for CSVs and artifacts (default: output_dir from the config).",
    )
    parser.add_argument(
        "--condition",
        type=str,
        help="Filter dataset to specific hydrogel condition (e.g. S34D30, Mat).",
    )
    parser.add_argument("--review-dir", type=str, default=None, help="Directory with manual ground-truth masks (default: review_dir from the config).")
    parser.add_argument("--pixel-size", type=float, help="Optical calibration override in um/pixel.")
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation without loading cached masks.")
    parser.add_argument("--plots", action="store_true", help="Generate all presentation figures from existing CSVs.")
    parser.add_argument("--preflight-only", action="store_true", help="Run preflight input and optical calibration validation and exit.")
    parser.add_argument("--dry-run", action="store_true", help="List the images that would be processed (after --condition/--sample) and exit without loading models.")
    parser.add_argument(
        "--allow-uncalibrated", "--force-uncalibrated", dest="allow_uncalibrated", action="store_true",
        help="Accept the default pixel size for images without scale metadata instead of flagging their objects REVIEW.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error log output.")

    return parser


def main(args: Optional[Sequence[str]] = None) -> int:
    """CLI execution entry point. Returns 0 on full success, 1 on any failure, 2 on usage errors."""
    parser = build_cli_parser()
    cli_args = parser.parse_args(args)

    setup_logging(verbose=cli_args.verbose, quiet=cli_args.quiet)

    # 1. Handle Synthetic Smoke Test mode (self-contained, needs no config file)
    if cli_args.smoke_test:
        success = run_synthetic_smoke_test()
        return 0 if success else 1

    # 2. Build Pipeline Configuration
    config_path = Path(cli_args.config)
    if not config_path.is_file():
        logger.error(
            f"Config file not found: '{config_path}'. Pass --config <path to spheroid_brightfield.yaml> "
            "(run from the repository root to use the default configs/spheroid_brightfield.yaml)."
        )
        return 2

    cli_overrides: Dict[str, Any] = {}
    if cli_args.input_dir:
        in_t0, in_t7 = resolve_input_dir(cli_args.input_dir)
        cli_overrides["t0_dir"] = str(in_t0)
        cli_overrides["t7_dir"] = str(in_t7)
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

    try:
        config = PipelineConfig.from_yaml(config_path, cli_overrides=cli_overrides)
    except ValueError as e:
        logger.error(f"Invalid configuration in '{config_path}': {e}")
        return 2

    if cli_args.full_dataset:
        logger.info("--full-dataset is a no-op: the full dataset is processed unless --sample is given.")

    # 3. Dry run: list the selection, load no models
    if cli_args.dry_run:
        manifest_df = build_manifest(config.t0_dir, config.t7_dir)
        selection = select_images(manifest_df, condition=config.condition, sample_n=cli_args.sample)
        if selection.empty:
            logger.error("Dry run: no images selected.")
            return 1
        print(f"Dry run: {len(selection)} image(s) would be processed "
              f"({selection['condition'].nunique()} condition(s)):")
        for _, r in selection.iterrows():
            print(f"  {r['condition']:<10} {r['timepoint']:<3} {r['pair_key']:<28} {r['file_path']}")
        return 0

    # 4. Handle Preflight-Only Mode
    if cli_args.preflight_only:
        logger.info("Executing Preflight Integrity & Optical Calibration Audit...")
        report = PreflightAuditor(config).run_preflight_check()
        print(report.format_summary())
        return 0 if report.passed else 1

    # 5. Handle Presentation Plots Generation Mode
    if cli_args.plots:
        out_p = Path(config.output_dir)
        objs_p = out_p / "objects.csv"
        pairs_p = out_p / "pairs.csv"
        if not objs_p.exists():
            logger.error(f"Cannot generate presentation figures: '{objs_p}' does not exist. Run pipeline first.")
            return 1
        from matrix_mat.reporting.presentation_plots import (
            compute_well_growth_comparison,
            generate_all_presentation_figures,
        )
        logger.info(f"Generating presentation figures in {out_p}/figures/ ...")
        objs_df = pd.read_csv(objs_p)
        pairs_df = pd.read_csv(pairs_p) if pairs_p.exists() else pd.DataFrame()
        wells_csv = out_p / "well_growth_comparison.csv"
        wells_df = compute_well_growth_comparison(objs_df, wells_csv)
        generate_all_presentation_figures(out_p, objs_df, pairs_df, wells_df)
        logger.info(f"Generated presentation figures in {out_p}/figures/")
        return 0

    # 6. Sample or full dataset execution
    if cli_args.sample is not None:
        logger.info(f"Launching stratified {cli_args.sample}-image sample execution...")
    else:
        logger.info("Launching full dataset execution across all raw images...")
    try:
        res = run_spheroid_pipeline(config, sample_n=cli_args.sample)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Brightfield run aborted: {e}")
        return 1
    return 0 if res.get("status") == RUN_STATUS_SUCCESS else 1


if __name__ == "__main__":
    sys.exit(main())
