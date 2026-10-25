"""
matrix-mat: Unified CLI for spheroid brightfield and immunofluorescence analysis.

Subcommands:
  matrix-mat bf {preflight, run, plots}
  matrix-mat if {preflight, sweep, compare, compare-prepost, compare-biological}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from matrix_mat.core.paths import resolve_data_root, resolve_models_root, resolve_output_root


def resolve_runs_dir(cli_val: Optional[str]) -> Path:
    """IF runs live under ${OUTPUT_ROOT}/runs unless a directory is given explicitly."""
    return Path(cli_val) if cli_val else resolve_output_root() / "runs"

logger = logging.getLogger("matrix_mat.cli")


def build_parser() -> argparse.ArgumentParser:
    """Builds unified matrix-mat CLI argument parser with subparsers."""
    parser = argparse.ArgumentParser(
        prog="matrix-mat",
        description="matrix-mat: High-precision, QC-audited image analysis for spheroid volume kinetics and IF density.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error log output.")

    subparsers = parser.add_subparsers(dest="domain", required=True, help="Domain subpackage to invoke: 'bf' or 'if'.")

    # =========================================================================
    # Brightfield Domain: matrix-mat bf ...
    # =========================================================================
    bf_parser = subparsers.add_parser("bf", help="Brightfield spheroid volume and kinetics pipeline.")
    bf_sub = bf_parser.add_subparsers(dest="subcommand", required=True, help="Brightfield action: preflight, run, plots.")

    # bf preflight
    bf_preflight = bf_sub.add_parser("preflight", help="Proactive dataset integrity and optical calibration audit.")
    bf_preflight.add_argument("-c", "--config", type=str, default="configs/spheroid_brightfield.yaml", help="Path to YAML config.")
    bf_preflight.add_argument("--t0-dir", type=str, help="Day 0 image directory override.")
    bf_preflight.add_argument("--t7-dir", type=str, help="Day 7 image directory override.")
    bf_preflight.add_argument("-o", "--out", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory (default: output_dir from the YAML config).")
    bf_preflight.add_argument("--pixel-size", dest="pixel_size", type=float, help="Optical calibration override (um/px).")

    # bf run
    bf_run = bf_sub.add_parser("run", help="Run full brightfield segmentation, pairing, and statistical analysis.")
    bf_run.add_argument("-c", "--config", type=str, default="configs/spheroid_brightfield.yaml", help="Path to YAML config.")
    bf_run.add_argument("--t0-dir", type=str, help="Day 0 image directory override.")
    bf_run.add_argument("--t7-dir", type=str, help="Day 7 image directory override.")
    bf_run.add_argument("--input-dir", type=str, help="Unified input root directory containing Day0 and Day7 subdirectories.")
    bf_run.add_argument("-o", "--out", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory (default: output_dir from the YAML config).")
    bf_run.add_argument("--condition", type=str, help="Filter dataset to specific condition (e.g. Mat, S34D30).")
    bf_run.add_argument("--review-dir", type=str, default=None, help="Directory of manual review masks (default: review_dir from the YAML config).")
    bf_run.add_argument("--pixel-size", dest="pixel_size", type=float, help="Optical calibration override in um/pixel.")
    bf_run.add_argument("--backend", type=str, choices=["cpsam", "cpdino", "classical"], help="Primary segmentation backend.")
    bf_run.add_argument("--device", type=str, choices=["cpu", "mps", "cuda"], help="Hardware acceleration device.")
    bf_run.add_argument("--no-cache", action="store_true", help="Force recomputation without reading cached masks.")
    bf_run.add_argument("--overwrite", action="store_true", help="Overwrite existing output files and bypass mask cache.")
    bf_run.add_argument(
        "--force-uncalibrated",
        "--allow-uncalibrated",
        dest="force_uncalibrated",
        action="store_true",
        help="Allow running on uncalibrated images (bypasses strict calibration check).",
    )

    # Mutually exclusive sample vs full dataset
    sample_group = bf_run.add_mutually_exclusive_group()
    sample_group.add_argument("--sample", type=int, metavar="N", help="Run on N samples only.")
    sample_group.add_argument("--full-dataset", action="store_true", help="Process all images in dataset (default behavior).")

    # bf plots
    bf_plots = bf_sub.add_parser("plots", help="Generate all presentation figures (Figs 1-6) and growth tables from CSVs.")
    bf_plots.add_argument("-c", "--config", type=str, default="configs/spheroid_brightfield.yaml", help="Path to YAML config.")
    bf_plots.add_argument("-o", "--out", "--output-dir", dest="output_dir", type=str, default=None, help="Directory containing objects.csv and pairs.csv (default: ${OUTPUT_ROOT}).")
    bf_plots.add_argument("--input-dir", type=str, help="Alias for directory containing objects.csv and pairs.csv.")

    # =========================================================================
    # Immunofluorescence Domain: matrix-mat if ...
    # =========================================================================
    if_parser = subparsers.add_parser("if", help="Immunofluorescence (IF) nuclear density and sweep pipeline.")
    if_sub = if_parser.add_subparsers(dest="subcommand", required=True, help="IF action: preflight, sweep, compare, compare-prepost, compare-biological.")

    # if preflight
    if_preflight = if_sub.add_parser("preflight", help="Proactive audit for channel pairing and GPU acceleration.")
    if_preflight.add_argument("-c", "--config", type=str, default="configs/sweep.yaml", help="Path to sweep YAML config.")
    if_preflight.add_argument("--data-dir", "--input-dir", dest="data_dir", type=str, default=None, help="Directory containing IF microscopy images.")
    if_preflight.add_argument("--model-path", type=str, default=None, help="Model weights file (default: from the config).")
    if_preflight.add_argument("--allow-unpaired", action="store_true", help="Allow unpaired files without error.")

    # if sweep
    if_sweep = if_sub.add_parser("sweep", help="Execute multi-run hyperparameter sweep or pre/post ablation.")
    if_sweep.add_argument("-c", "--config", type=str, default="configs/sweep.yaml", help="Path to sweep YAML config.")
    if_sweep.add_argument("-o", "--out", "--output-dir", "--runs-dir", dest="runs_dir", type=str, default=None, help="Output directory for runs (default: runs_dir from the YAML).")
    if_sweep.add_argument("--data-dir", "--input-dir", dest="data_dir", type=str, default=None, help="Input directory containing IF images.")
    if_sweep.add_argument("--scope", type=str, choices=["subset", "full"], default="full", help="Evaluation scope.")
    if_sweep.add_argument("--allow-unpaired", action="store_true", help="Allow unpaired channel files.")
    if_sweep.add_argument("--overwrite", action="store_true", help="Force recomputation of existing runs.")
    if_sweep.add_argument("--sample", type=int, metavar="N", help="Run sweep on first N fields only.")

    # if compare
    if_compare = if_sub.add_parser("compare", help="Compare completed runs across hyperparameter sweep.")
    if_compare.add_argument("--runs", "--runs-dir", dest="runs_dir", type=str, default=None, help="Path to runs directory (default: ${OUTPUT_ROOT}/runs).")
    if_compare.add_argument("-o", "--output", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory for comparison.")
    if_compare.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML.")

    # if compare-prepost
    if_comp_pp = if_sub.add_parser("compare-prepost", help="Compare pre/post ablation and robustness runs.")
    if_comp_pp.add_argument("--runs", "--runs-dir", dest="runs_dir", type=str, default=None, help="Path to runs directory (default: ${OUTPUT_ROOT}/runs).")
    if_comp_pp.add_argument("-o", "--output", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory for comparison.")
    if_comp_pp.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML.")
    if_comp_pp.add_argument("--biological", action="store_true", help="Also generate biological figures.")

    # if compare-biological
    if_comp_bio = if_sub.add_parser("compare-biological", help="Generate biological comparison figures across hydrogels.")
    if_comp_bio.add_argument("--run-dir", type=str, default=None, help="Run directory (default: ${OUTPUT_ROOT}/runs/104_PP11_full).")
    if_comp_bio.add_argument("-o", "--output", "--output-dir", dest="output_dir", type=str, default=None, help="Output directory for figures.")
    if_comp_bio.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML.")

    return parser


def setup_cli_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configures root logging format and level."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)


def handle_bf_preflight(args: argparse.Namespace) -> int:
    """Executes brightfield preflight integrity audit."""
    from matrix_mat.brightfield.config import PipelineConfig
    from matrix_mat.brightfield.preflight import PreflightAuditor

    cli_overrides: Dict[str, Any] = {}
    if getattr(args, "output_dir", None):
        cli_overrides["output_dir"] = args.output_dir
    if getattr(args, "t0_dir", None):
        cli_overrides["t0_dir"] = args.t0_dir
    if getattr(args, "t7_dir", None):
        cli_overrides["t7_dir"] = args.t7_dir
    if getattr(args, "pixel_size", None):
        cli_overrides["cli_pixel_size_um"] = args.pixel_size

    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = PipelineConfig.from_yaml(cfg_path, cli_overrides=cli_overrides)
    else:
        config = PipelineConfig()
        for k, v in cli_overrides.items():
            setattr(config, k, v)

    print("\n[*] Executing Brightfield Preflight Integrity & Calibration Audit...\n")
    report = PreflightAuditor(config).run_preflight_check()
    print(report.format_summary())
    return 0 if report.passed else 1


def handle_bf_run(args: argparse.Namespace) -> int:
    """Executes brightfield analysis pipeline."""
    from matrix_mat.brightfield.config import PipelineConfig
    from matrix_mat.brightfield.run_pipeline import run_spheroid_pipeline

    cli_overrides: Dict[str, Any] = {}
    if getattr(args, "output_dir", None):
        cli_overrides["output_dir"] = args.output_dir
    if getattr(args, "review_dir", None):
        cli_overrides["review_dir"] = args.review_dir
    if getattr(args, "t0_dir", None):
        cli_overrides["t0_dir"] = args.t0_dir
    if getattr(args, "t7_dir", None):
        cli_overrides["t7_dir"] = args.t7_dir
    if getattr(args, "pixel_size", None):
        cli_overrides["cli_pixel_size_um"] = args.pixel_size
    if getattr(args, "condition", None):
        cli_overrides["condition"] = args.condition
    if getattr(args, "backend", None):
        cli_overrides.setdefault("segmentation", {})["primary_backend"] = args.backend
    if getattr(args, "device", None):
        cli_overrides.setdefault("segmentation", {})["device"] = args.device
    if getattr(args, "no_cache", False) or getattr(args, "overwrite", False):
        cli_overrides.setdefault("segmentation", {})["force_recompute"] = True
    if getattr(args, "force_uncalibrated", False):
        cli_overrides["allow_uncalibrated"] = True
        cli_overrides.setdefault("qc", {})["require_calibration"] = False

    # Handle input-dir: resolve Day0 and Day7 subdirectories
    input_dir = getattr(args, "input_dir", None)
    if input_dir:
        in_p = Path(input_dir)
        candidates_t0 = ["Day0", "Day 0", "Spheroid Day0 260805", "t0"]
        candidates_t7 = ["Day7", "Day 7", "Spheroid D7 260812", "t7"]
        found_t0 = next((in_p / c for c in candidates_t0 if (in_p / c).exists()), in_p)
        found_t7 = next((in_p / c for c in candidates_t7 if (in_p / c).exists()), in_p)
        cli_overrides.setdefault("t0_dir", str(found_t0))
        cli_overrides.setdefault("t7_dir", str(found_t7))

    cfg_path = Path(args.config)
    if cfg_path.exists():
        config = PipelineConfig.from_yaml(cfg_path, cli_overrides=cli_overrides)
    else:
        config = PipelineConfig()
        for k, v in cli_overrides.items():
            if isinstance(v, dict):
                sub = getattr(config, k, None)
                if sub:
                    for sk, sv in v.items():
                        setattr(sub, sk, sv)
            else:
                setattr(config, k, v)

    sample_n = getattr(args, "sample", None)
    try:
        res = run_spheroid_pipeline(
            config=config,
            sample_n=sample_n,
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Brightfield run aborted: {e}")
        return 1
    is_success = isinstance(res, dict) and res.get("status") == "SUCCESS"
    return 0 if is_success else 1


def handle_bf_plots(args: argparse.Namespace) -> int:
    """Generates brightfield presentation figures from existing CSVs."""
    from matrix_mat.brightfield.config import PipelineConfig
    from matrix_mat.reporting.presentation_plots import generate_all_presentation_figures
    from matrix_mat.reporting.visualize import generate_all_figures

    out_dir_str = args.input_dir or args.output_dir or str(resolve_output_root())
    out_p = Path(out_dir_str)
    objs_p = out_p / "objects.csv"
    pairs_p = out_p / "pairs.csv"

    if not objs_p.exists():
        logger.error(
            f"Cannot generate presentation figures: '{objs_p}' does not exist. "
            f"Run 'matrix-mat bf run' first to generate required output tables."
        )
        return 1
    if not pairs_p.exists():
        logger.error(
            f"Cannot generate presentation figures: '{pairs_p}' does not exist. "
            f"Run 'matrix-mat bf run' first to generate required output tables."
        )
        return 1

    cfg_path = Path(args.config)
    config = PipelineConfig.from_yaml(cfg_path) if cfg_path.exists() else PipelineConfig()

    logger.info(f"Generating presentation figures from CSVs in '{out_p}'...")
    import pandas as pd
    objs_df = pd.read_csv(objs_p)
    pairs_df = pd.read_csv(pairs_p)
    cond_p = out_p / "condition_summary.csv"
    cond_df = pd.read_csv(cond_p) if cond_p.exists() else None

    generate_all_figures(pairs_df=pairs_df, objects_df=objs_df, output_dir=out_p)
    try:
        generate_all_presentation_figures(
            output_dir=out_p,
            objects_df=objs_df,
            pairs_df=pairs_df,
            condition_df=cond_df,
        )
    except Exception as e:
        logger.warning(f"Note on presentation figures generation: {e}")

    logger.info("Figure generation complete.")
    return 0


def handle_if_preflight(args: argparse.Namespace) -> int:
    """Executes immunofluorescence preflight audit."""
    from matrix_mat.if_density.preflight import run_if_preflight

    # Check the same data, weights and unpaired-field rule the sweep config will use.
    cfg = None
    cfg_path = Path(getattr(args, "config", "") or "")
    if cfg_path.is_file():
        from matrix_mat.if_density.config import IFSweepConfig
        cfg = IFSweepConfig.from_yaml(cfg_path)
    data_dir = getattr(args, "data_dir", None) or (cfg.data_dir if cfg else resolve_data_root() / "if")
    model_path = getattr(args, "model_path", None) or (
        str(cfg.model.pretrained_model) if cfg else str(resolve_models_root() / "cpsam_v2")
    )
    allow_unpaired = bool(getattr(args, "allow_unpaired", False) or (cfg.allow_unpaired if cfg else False))

    print("\n[*] Executing Immunofluorescence Preflight Audit...\n")
    audit = run_if_preflight(data_dir=str(data_dir), model_path=model_path, allow_unpaired=allow_unpaired)
    print(audit.format_summary())
    return 0 if audit.passed else 1


def handle_if_sweep(args: argparse.Namespace) -> int:
    """Executes immunofluorescence parameter sweep."""
    from matrix_mat.if_density.sweep import run_sweep

    import matrix_mat.if_density.sweep as sweep_mod

    cfg_path = Path(args.config)
    try:
        sweep_mod.run_sweep(
            config_path=str(cfg_path),
            allow_unpaired_override=getattr(args, "allow_unpaired", False),
            overwrite_override=getattr(args, "overwrite", False),
            scope=getattr(args, "scope", "full"),
            runs_dir_override=getattr(args, "runs_dir", None),
            data_dir_override=getattr(args, "data_dir", None),
            sample_n=getattr(args, "sample", None),
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error(f"IF sweep aborted: {e}")
        return 1
    return 0


def handle_if_compare(args: argparse.Namespace) -> int:
    """Executes cross-run IF comparison."""
    from matrix_mat.if_density.comparator import generate_cross_run_comparison

    runs_dir = resolve_runs_dir(getattr(args, "runs_dir", None))
    output_dir = getattr(args, "output_dir", None)
    generate_cross_run_comparison(runs_dir=runs_dir, output_dir=output_dir)
    return 0


def handle_if_compare_prepost(args: argparse.Namespace) -> int:
    """Executes pre/post ablation comparison."""
    import matrix_mat.if_density.comparator_prepost as cp

    runs_dir = resolve_runs_dir(getattr(args, "runs_dir", None))
    output_dir = getattr(args, "output_dir", None)
    cp.run_comparator_prepost(runs_dir=runs_dir, output_dir=output_dir)

    if getattr(args, "biological", False):
        import matrix_mat.if_density.comparator_biological as cb
        # Biological figures describe one run: the full pre+post pipeline (PP11).
        pp11 = sorted(p for p in runs_dir.glob("*PP11*") if p.is_dir())
        if not pp11:
            logger.error(f"--biological: no *PP11* run found in {runs_dir}")
            return 1
        cb.generate_all_if_biological_figures(run_dir=pp11[0], output_dir=output_dir)
    return 0


def handle_if_compare_biological(args: argparse.Namespace) -> int:
    """Generates biological comparison figures across hydrogels."""
    from matrix_mat.if_density.comparator_biological import generate_all_if_biological_figures

    run_dir = getattr(args, "run_dir", None) or resolve_runs_dir(None) / "104_PP11_full"
    output_dir = getattr(args, "output_dir", None)
    generate_all_if_biological_figures(run_dir=run_dir, output_dir=output_dir)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_cli_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    if args.domain == "bf":
        if args.subcommand == "preflight":
            return handle_bf_preflight(args)
        elif args.subcommand == "run":
            return handle_bf_run(args)
        elif args.subcommand == "plots":
            return handle_bf_plots(args)
    elif args.domain == "if":
        if args.subcommand == "preflight":
            return handle_if_preflight(args)
        elif args.subcommand == "sweep":
            return handle_if_sweep(args)
        elif args.subcommand == "compare":
            return handle_if_compare(args)
        elif args.subcommand == "compare-prepost":
            return handle_if_compare_prepost(args)
        elif args.subcommand == "compare-biological":
            return handle_if_compare_biological(args)

    logger.error(f"Unrecognized command: {args.domain} {getattr(args, 'subcommand', '')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
