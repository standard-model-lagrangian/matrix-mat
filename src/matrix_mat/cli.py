"""
matrix-mat: Unified CLI for spheroid brightfield and immunofluorescence analysis.

Subcommands:
  matrix-mat bf {preflight, run, plots}
  matrix-mat if {preflight, sweep, compare, compare-prepost, compare-biological}

Exit codes: 0 success, 1 run/validation failure, 2 usage error (bad flags, missing config file).
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matrix_mat.core.paths import resolve_data_root, resolve_models_root, resolve_output_root

logger = logging.getLogger("matrix_mat.cli")

# Shipped configs live in <repo>/configs; resolved relative to the package, never the cwd.
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
BF_DEFAULT_CONFIG = "spheroid_brightfield.yaml"
IF_DEFAULT_CONFIG = "sweep.yaml"

# --input-dir layout discovery: the brightfield pipeline's own folder names first
# (matrix_mat.brightfield.run_pipeline.resolve_input_dir), then any single sub-folder whose name
# contains a Day-0 / Day-7 token (e.g. "Spheroid Day0 <date>", "Spheroid D7 <date>").
T0_SUBDIR_PATTERN = re.compile(r"(?i)(?:^|[\s_-])(?:day\s?0|d0|t0)(?:$|[\s_-])")
T7_SUBDIR_PATTERN = re.compile(r"(?i)(?:^|[\s_-])(?:day\s?7|d7|t7)(?:$|[\s_-])")

# IF biological figures describe the full pre+post pipeline run (PP11) by default.
DEFAULT_BIOLOGICAL_RUN_PATTERN = "*PP11*"

DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")


class CLIUsageError(Exception):
    """Bad user input detected after parsing (exit code 2)."""


def _package_version() -> str:
    # matrix_mat.__version__ is kept equal to pyproject's version (tested); installed
    # dist-info metadata can be stale in editable installs, so it is only a fallback.
    try:
        import matrix_mat

        return str(matrix_mat.__version__)
    except Exception:  # pragma: no cover
        import importlib.metadata

        return importlib.metadata.version("matrix-mat")


def resolve_runs_dir(cli_val: Optional[str]) -> Path:
    """IF runs live under ${OUTPUT_ROOT}/runs unless a directory is given explicitly."""
    return Path(cli_val) if cli_val else resolve_output_root() / "runs"


def resolve_config_path(cli_val: Optional[str], default_name: str) -> Optional[Path]:
    """Resolve ``-c/--config``.

    An explicit path must exist (``CLIUsageError`` otherwise). Without ``-c`` the shipped
    ``configs/<default_name>`` next to the package is used; if that is absent a WARNING is
    logged and ``None`` is returned (built-in defaults apply).
    """
    if cli_val:
        p = Path(cli_val).expanduser()
        if not p.is_file():
            raise CLIUsageError(f"config file not found: {cli_val}")
        return p
    p = DEFAULT_CONFIG_DIR / default_name
    if p.is_file():
        return p
    logger.warning(f"Default config {p} not found; using built-in defaults (pass -c/--config to choose a file).")
    return None


def resolve_input_subdirs(
    input_dir: str | Path,
    t0_subdir: Optional[str] = None,
    t7_subdir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Find the Day 0 / Day 7 folders inside ``--input-dir``.

    ``t0_subdir``/``t7_subdir`` (``--t0-subdir``/``--t7-subdir``) win. Otherwise the folder found
    by ``matrix_mat.brightfield.run_pipeline.resolve_input_dir``, else the single sub-folder
    matching T0/T7_SUBDIR_PATTERN. Falls back to ``input_dir`` itself; raises ``ValueError`` when
    several folders match.
    """
    from matrix_mat.brightfield.run_pipeline import resolve_input_dir

    root = Path(input_dir)
    subdirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    known_t0, known_t7 = resolve_input_dir(root)

    def pick(explicit: Optional[str], known: Path, pattern: re.Pattern, label: str) -> Path:
        if explicit:
            p = root / explicit
            if not p.is_dir():
                raise ValueError(f"{label} sub-folder {explicit!r} not found in {root}")
            return p
        if known != root:
            return known
        hits = [p for p in subdirs if pattern.search(p.name)]
        if len(hits) > 1:
            raise ValueError(
                f"ambiguous {label} folders in {root}: {[p.name for p in hits]}; "
                f"choose one with --{label.lower()}-subdir"
            )
        return hits[0] if hits else root

    return (
        pick(t0_subdir, known_t0, T0_SUBDIR_PATTERN, "T0"),
        pick(t7_subdir, known_t7, T7_SUBDIR_PATTERN, "T7"),
    )


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {text!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text!r}")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text!r}")
    return v


def _device(text: str) -> str:
    d = text.strip().lower()
    if d in DEVICE_CHOICES or re.fullmatch(r"cuda:\d+", d):
        return d
    raise argparse.ArgumentTypeError(f"invalid device {text!r}; choose from {', '.join(DEVICE_CHOICES)} or cuda:N")


def build_parser() -> argparse.ArgumentParser:
    """Builds unified matrix-mat CLI argument parser with subparsers."""
    fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        prog="matrix-mat",
        description="matrix-mat: High-precision, QC-audited image analysis for spheroid volume kinetics and IF density.",
        formatter_class=fmt,
        allow_abbrev=False,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_package_version()}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error log output.")

    # -v / -q are also accepted after the subcommand (SUPPRESS keeps the top-level default).
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="Enable verbose debug logging.")
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="Suppress non-error log output.")

    def leaf(sub: Any, name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, parents=[common], formatter_class=fmt, allow_abbrev=False)

    subparsers = parser.add_subparsers(dest="domain", required=True, help="Domain subpackage to invoke: 'bf' or 'if'.")

    # =========================================================================
    # Brightfield Domain: matrix-mat bf ...
    # =========================================================================
    bf_parser = subparsers.add_parser("bf", help="Brightfield spheroid volume and kinetics pipeline.",
                                      parents=[common], allow_abbrev=False)
    bf_sub = bf_parser.add_subparsers(dest="subcommand", required=True, help="Brightfield action: preflight, run, plots.")

    bf_cfg_help = f"Path to YAML config (default: <repo>/configs/{BF_DEFAULT_CONFIG}; missing explicit file -> exit 2)."

    def add_bf_inputs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--t0-dir", type=str, help="Day 0 image directory override.")
        p.add_argument("--t7-dir", type=str, help="Day 7 image directory override.")
        p.add_argument("--input-dir", type=str,
                       help="Input root containing the Day 0 and Day 7 sub-folders (found by name, e.g. 'Day0' or "
                            "'Spheroid Day0 <date>'; see --t0-subdir/--t7-subdir).")
        p.add_argument("--t0-subdir", type=str, default=None, help="Name of the Day 0 sub-folder inside --input-dir.")
        p.add_argument("--t7-subdir", type=str, default=None, help="Name of the Day 7 sub-folder inside --input-dir.")

    # bf preflight
    bf_preflight = leaf(bf_sub, "preflight", "Proactive dataset integrity and optical calibration audit.")
    bf_preflight.add_argument("-c", "--config", type=str, default=None, help=bf_cfg_help)
    add_bf_inputs(bf_preflight)
    bf_preflight.add_argument("-o", "--output-dir", "--out", dest="output_dir", type=str, default=None,
                              help="Output directory (default: output_dir from the YAML config).")
    bf_preflight.add_argument("--pixel-size", dest="pixel_size", type=_positive_float,
                              help="Optical calibration override (um/px, > 0).")

    # bf run
    bf_run = leaf(bf_sub, "run", "Run full brightfield segmentation, pairing, and statistical analysis.")
    bf_run.add_argument("-c", "--config", type=str, default=None, help=bf_cfg_help)
    add_bf_inputs(bf_run)
    bf_run.add_argument("-o", "--output-dir", "--out", dest="output_dir", type=str, default=None,
                        help="Output directory (default: output_dir from the YAML config).")
    bf_run.add_argument("--condition", type=str, help="Filter dataset to specific condition (e.g. Mat, S34D30).")
    bf_run.add_argument("--review-dir", type=str, default=None,
                        help="Directory of manual review masks (default: review_dir from the YAML config).")
    bf_run.add_argument("--pixel-size", dest="pixel_size", type=_positive_float,
                        help="Optical calibration override in um/pixel (> 0); applied to every image.")
    bf_run.add_argument("--backend", type=str, choices=["cpsam", "cpdino", "classical"], help="Primary segmentation backend.")
    bf_run.add_argument("--device", type=_device, default=None,
                        help="Segmentation device: auto, cpu, cuda (or cuda:N), mps (default: segmentation.device "
                             "from the YAML config).")
    bf_run.add_argument("--no-cache", action="store_true", help="Recompute every mask instead of reading the mask cache.")
    bf_run.add_argument("--overwrite", action="store_true",
                        help="Same as --no-cache: recompute every mask (output tables and figures are rewritten "
                             "on every run regardless).")
    bf_run.add_argument("--dry-run", action="store_true",
                        help="List the images that would be processed (after --condition/--sample) and exit "
                             "without loading models or writing outputs.")
    bf_run.add_argument(
        "--force-uncalibrated",
        "--allow-uncalibrated",
        dest="force_uncalibrated",
        action="store_true",
        help="Allow running on uncalibrated images (bypasses strict calibration check).",
    )

    sample_group = bf_run.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample", type=_positive_int, metavar="N",
        help="Brightfield: process a subset of about N images, i.e. N // (2 x n_conditions) matched Day 0/Day 7 "
             "pairs per condition (at least one pair each).",
    )
    sample_group.add_argument(
        "--full-dataset", action="store_true",
        help="Process every image (the default); also clears any sample_n set in the YAML config.",
    )

    # bf plots
    bf_plots = leaf(bf_sub, "plots", "Generate all presentation figures (Figs 1-6) and growth tables from CSVs.")
    bf_plots.add_argument("-c", "--config", type=str, default=None, help=bf_cfg_help)
    bf_plots.add_argument("-o", "--output-dir", "--out", dest="output_dir", type=str, default=None,
                          help="Directory containing objects.csv and pairs.csv; figures are written there "
                               "(default: output_dir from the YAML config).")
    bf_plots.add_argument("--input-dir", type=str, help="Alias for the directory containing objects.csv and pairs.csv.")

    # =========================================================================
    # Immunofluorescence Domain: matrix-mat if ...
    # =========================================================================
    if_parser = subparsers.add_parser("if", help="Immunofluorescence (IF) nuclear density and sweep pipeline.",
                                      parents=[common], allow_abbrev=False)
    if_sub = if_parser.add_subparsers(dest="subcommand", required=True,
                                      help="IF action: preflight, sweep, compare, compare-prepost, compare-biological.")
    if_cfg_help = f"Path to IF YAML config (default: <repo>/configs/{IF_DEFAULT_CONFIG}; missing explicit file -> exit 2)."

    # if preflight
    if_preflight = leaf(if_sub, "preflight", "Proactive audit for channel pairing and GPU acceleration.")
    if_preflight.add_argument("-c", "--config", type=str, default=None, help=if_cfg_help)
    if_preflight.add_argument("--data-dir", "--input-dir", dest="data_dir", type=str, default=None,
                              help="Directory containing IF microscopy images.")
    if_preflight.add_argument("--model-path", type=str, default=None, help="Model weights file (default: from the config).")
    if_preflight.add_argument("--allow-unpaired", action="store_true", help="Allow unpaired files without error.")
    if_preflight.add_argument("-o", "--output-dir", dest="output_dir", type=str, default=None,
                              help="Also write the preflight summary to <dir>/if_preflight_report.txt.")

    # if sweep
    if_sweep = leaf(if_sub, "sweep", "Execute multi-run hyperparameter sweep or pre/post ablation.")
    if_sweep.add_argument("-c", "--config", type=str, default=None, help=if_cfg_help)
    if_sweep.add_argument("-o", "--output-dir", "--out", "--runs-dir", dest="runs_dir", type=str, default=None,
                          help="Output directory for runs (default: runs_dir from the YAML).")
    if_sweep.add_argument("--data-dir", "--input-dir", dest="data_dir", type=str, default=None,
                          help="Input directory containing IF images.")
    if_sweep.add_argument("--scope", type=str, choices=["subset", "full"], default="full", help="Evaluation scope.")
    if_sweep.add_argument("--allow-unpaired", action="store_true", help="Allow unpaired channel files.")
    if_sweep.add_argument("--overwrite", action="store_true",
                          help="Recompute every run: ignore existing run outputs and cached masks.")
    if_sweep.add_argument("--strict-pairing", action="store_true",
                          help="Fail on any pairing ambiguity instead of warning.")
    if_sweep.add_argument("--dry-run", action="store_true",
                          help="Preflight, pair and print the run plan without segmenting.")
    if_sweep.add_argument("--sample", type=_positive_int, metavar="N",
                          help="IF: process only the first N paired fields of view (nuclear+actin channel pairs).")

    def add_runs(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs", "--runs-dir", dest="runs_dir", type=str, default=None,
                       help="Runs directory (default: runs_dir from -c/--config if given, else ${OUTPUT_ROOT}/runs).")
        p.add_argument("-c", "--config", type=str, default=None,
                       help="IF YAML config; only its runs_dir is used, as the default for --runs.")

    # if compare
    if_compare = leaf(if_sub, "compare", "Compare completed runs across hyperparameter sweep.")
    add_runs(if_compare)
    if_compare.add_argument("-o", "--output-dir", "--output", dest="output_dir", type=str, default=None,
                            help="Output directory for comparison.")
    if_compare.add_argument("--run-names", nargs="+", default=None, metavar="RUN",
                            help="Compare only these run folders (default: all runs of --scope).")
    if_compare.add_argument("--scope", choices=["subset", "full"], default="full",
                            help="Which runs to compare: those run with --scope subset or full.")

    # if compare-prepost
    if_comp_pp = leaf(if_sub, "compare-prepost", "Compare pre/post ablation and robustness runs.")
    add_runs(if_comp_pp)
    if_comp_pp.add_argument("-o", "--output-dir", "--output", dest="output_dir", type=str, default=None,
                            help="Output directory for comparison.")
    if_comp_pp.add_argument("--scope", choices=["subset", "full"], default="full",
                            help="Robustness comparison uses runs of this scope.")
    if_comp_pp.add_argument("--biological", action="store_true",
                            help="Also generate biological figures for the single run matching --run-pattern.")
    if_comp_pp.add_argument("--run-pattern", type=str, default=DEFAULT_BIOLOGICAL_RUN_PATTERN,
                            help="Glob selecting the run for --biological; exactly one run must match.")

    # if compare-biological
    if_comp_bio = leaf(if_sub, "compare-biological", "Generate biological comparison figures across hydrogels.")
    add_runs(if_comp_bio)
    if_comp_bio.add_argument("--run-dir", type=str, default=None,
                             help="Run directory (default: the single run in --runs matching --run-pattern).")
    if_comp_bio.add_argument("--run-pattern", type=str, default=DEFAULT_BIOLOGICAL_RUN_PATTERN,
                             help="Glob selecting the run inside --runs when --run-dir is not given.")
    if_comp_bio.add_argument("-o", "--output-dir", "--output", dest="output_dir", type=str, default=None,
                             help="Output directory for figures.")

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


def _error(msg: str) -> None:
    """Print a user-facing error to stderr (independent of logging configuration)."""
    print(f"matrix-mat: error: {msg}", file=sys.stderr)


def _load_bf_config(args: argparse.Namespace, cli_overrides: Dict[str, Any]) -> Any:
    from matrix_mat.brightfield.config import PipelineConfig

    cfg_path = resolve_config_path(getattr(args, "config", None), BF_DEFAULT_CONFIG)
    if cfg_path is not None:
        return PipelineConfig.from_yaml(cfg_path, cli_overrides=cli_overrides)
    config = PipelineConfig()
    for k, v in cli_overrides.items():
        if isinstance(v, dict):
            sub = getattr(config, k, None)
            if sub is not None:
                for sk, sv in v.items():
                    setattr(sub, sk, sv)
        else:
            setattr(config, k, v)
    return config


def _bf_input_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    cli_overrides: Dict[str, Any] = {}
    if getattr(args, "output_dir", None):
        cli_overrides["output_dir"] = args.output_dir
    if getattr(args, "t0_dir", None):
        cli_overrides["t0_dir"] = args.t0_dir
    if getattr(args, "t7_dir", None):
        cli_overrides["t7_dir"] = args.t7_dir
    if getattr(args, "pixel_size", None) is not None:
        cli_overrides["cli_pixel_size_um"] = args.pixel_size
    input_dir = getattr(args, "input_dir", None)
    if input_dir:
        if not Path(input_dir).is_dir():
            raise CLIUsageError(f"--input-dir {input_dir} is not a directory")
        try:
            t0, t7 = resolve_input_subdirs(input_dir, getattr(args, "t0_subdir", None), getattr(args, "t7_subdir", None))
        except ValueError as e:
            raise CLIUsageError(str(e)) from None
        cli_overrides.setdefault("t0_dir", str(t0))
        cli_overrides.setdefault("t7_dir", str(t7))
    return cli_overrides


def handle_bf_preflight(args: argparse.Namespace) -> int:
    """Executes brightfield preflight integrity audit."""
    from matrix_mat.brightfield.preflight import PreflightAuditor

    config = _load_bf_config(args, _bf_input_overrides(args))

    print("\n[*] Executing Brightfield Preflight Integrity & Calibration Audit...\n")
    report = PreflightAuditor(config).run_preflight_check()
    print(report.format_summary())
    return 0 if report.passed else 1


def handle_bf_run(args: argparse.Namespace) -> int:
    """Executes brightfield analysis pipeline."""
    import matrix_mat.brightfield.run_pipeline as rp

    cli_overrides = _bf_input_overrides(args)
    if getattr(args, "review_dir", None):
        cli_overrides["review_dir"] = args.review_dir
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

    config = _load_bf_config(args, cli_overrides)
    if getattr(args, "full_dataset", False):
        config.sample_n = None

    sample_n = getattr(args, "sample", None)
    if getattr(args, "dry_run", False):
        manifest_df = rp.build_manifest(config.t0_dir, config.t7_dir)
        selection = rp.select_images(manifest_df, condition=config.condition, sample_n=sample_n)
        if selection.empty:
            _error("dry run: no images selected")
            return 1
        print(f"Dry run: {len(selection)} image(s) would be processed "
              f"({selection['condition'].nunique()} condition(s)):")
        for _, r in selection.iterrows():
            print(f"  {r['condition']:<10} {r['timepoint']:<3} {r['pair_key']:<28} {r['file_path']}")
        return 0
    try:
        res = rp.run_spheroid_pipeline(config=config, sample_n=sample_n)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error(f"Brightfield run aborted: {e}")
        return 1
    is_success = isinstance(res, dict) and res.get("status") == "SUCCESS"
    if not is_success and isinstance(res, dict) and res.get("error"):
        _error(f"brightfield run failed: {res['error']}")
    return 0 if is_success else 1


def handle_bf_plots(args: argparse.Namespace) -> int:
    """Generates brightfield presentation figures from existing CSVs."""
    import matrix_mat.reporting.presentation_plots as pp
    import matrix_mat.reporting.visualize as vz

    config = _load_bf_config(args, {})
    out_dir_str = args.input_dir or args.output_dir or str(getattr(config, "output_dir", "") or resolve_output_root())
    out_p = Path(out_dir_str)
    objs_p = out_p / "objects.csv"
    pairs_p = out_p / "pairs.csv"

    for p in (objs_p, pairs_p):
        if not p.exists():
            _error(
                f"Cannot generate presentation figures: '{p}' does not exist. "
                f"Run 'matrix-mat bf run' first to generate required output tables."
            )
            return 1

    logger.info(f"Generating presentation figures from CSVs in '{out_p}'...")
    import pandas as pd
    objs_df = pd.read_csv(objs_p)
    pairs_df = pd.read_csv(pairs_p)
    cond_p = out_p / "condition_summary.csv"
    cond_df = pd.read_csv(cond_p) if cond_p.exists() else None

    failures: List[str] = []
    try:
        vz.generate_all_figures(pairs_df=pairs_df, objects_df=objs_df, output_dir=out_p)
    except Exception as e:
        failures.append(f"QC/summary figures: {type(e).__name__}: {e}")
    try:
        pp.generate_all_presentation_figures(
            output_dir=out_p,
            objects_df=objs_df,
            pairs_df=pairs_df,
            condition_df=cond_df,
        )
    except Exception as e:
        failures.append(f"presentation figures: {type(e).__name__}: {e}")

    if failures:
        _error("figure generation failed:\n  - " + "\n  - ".join(failures))
        return 1
    logger.info("Figure generation complete.")
    return 0


def _load_if_config(args: argparse.Namespace) -> Tuple[Optional[Path], Any]:
    cfg_path = resolve_config_path(getattr(args, "config", None), IF_DEFAULT_CONFIG)
    if cfg_path is None:
        return None, None
    from matrix_mat.if_density.config import IFSweepConfig

    return cfg_path, IFSweepConfig.from_yaml(cfg_path)


def handle_if_preflight(args: argparse.Namespace) -> int:
    """Executes immunofluorescence preflight audit."""
    from matrix_mat.if_density.preflight import run_if_preflight

    # Check the same data, weights and unpaired-field rule the sweep config will use.
    _, cfg = _load_if_config(args)
    data_dir = getattr(args, "data_dir", None) or (cfg.data_dir if cfg else resolve_data_root() / "if")
    model_path = getattr(args, "model_path", None) or (
        str(cfg.model.pretrained_model) if cfg else str(resolve_models_root() / "cpsam_v2")
    )
    allow_unpaired = bool(getattr(args, "allow_unpaired", False) or (cfg.allow_unpaired if cfg else False))

    print("\n[*] Executing Immunofluorescence Preflight Audit...\n")
    audit = run_if_preflight(
        data_dir=str(data_dir), model_path=model_path, allow_unpaired=allow_unpaired,
        allow_uncalibrated=(bool(getattr(cfg, "allow_uncalibrated", False)) if cfg else None),
    )
    summary = audit.format_summary()
    print(summary)
    if getattr(args, "output_dir", None):
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "if_preflight_report.txt").write_text(summary, encoding="utf-8")
    return 0 if audit.passed else 1


def handle_if_sweep(args: argparse.Namespace) -> int:
    """Executes immunofluorescence parameter sweep."""
    import matrix_mat.if_density.sweep as sweep_mod

    cfg_path = resolve_config_path(getattr(args, "config", None), IF_DEFAULT_CONFIG)
    if cfg_path is None:
        raise CLIUsageError(f"if sweep needs a config file; pass -c/--config (no {IF_DEFAULT_CONFIG} found)")
    try:
        res = sweep_mod.run_sweep(
            config_path=str(cfg_path),
            allow_unpaired_override=getattr(args, "allow_unpaired", False),
            overwrite_override=getattr(args, "overwrite", False),
            scope=getattr(args, "scope", "full"),
            runs_dir_override=getattr(args, "runs_dir", None),
            data_dir_override=getattr(args, "data_dir", None),
            sample_n=getattr(args, "sample", None),
            strict_pairing=bool(getattr(args, "strict_pairing", False)),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
        logger.error(f"IF sweep aborted: {e}")
        return 1
    if isinstance(res, dict) and res.get("status") not in (None, "SUCCESS"):
        _error(f"IF sweep finished with status {res.get('status')}: {res.get('error', '')}")
        return 1
    return 0


def _runs_dir_for(args: argparse.Namespace) -> Path:
    """--runs, else runs_dir from an explicit -c config, else ${OUTPUT_ROOT}/runs. Must exist."""
    runs = getattr(args, "runs_dir", None)
    if not runs and getattr(args, "config", None):
        _, cfg = _load_if_config(args)
        runs = str(getattr(cfg, "runs_dir", "") or "") or None
    runs_dir = resolve_runs_dir(runs)
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory {runs_dir} does not exist (run 'matrix-mat if sweep' first or pass --runs)")
    return runs_dir


def _select_run(runs_dir: Path, pattern: str) -> Path:
    hits = sorted(p for p in runs_dir.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern))
    if not hits:
        raise FileNotFoundError(f"no run matching {pattern!r} in {runs_dir}")
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} runs match {pattern!r} in {runs_dir}: {[p.name for p in hits]}; "
            "pass --run-dir or a narrower --run-pattern"
        )
    return hits[0]


def handle_if_compare(args: argparse.Namespace) -> int:
    """Executes cross-run IF comparison."""
    from matrix_mat.if_density.comparator import generate_cross_run_comparison

    runs_dir = _runs_dir_for(args)
    generate_cross_run_comparison(
        runs_dir=runs_dir, output_dir=getattr(args, "output_dir", None),
        run_names=getattr(args, "run_names", None), scope=getattr(args, "scope", "full"),
    )
    return 0


def handle_if_compare_prepost(args: argparse.Namespace) -> int:
    """Executes pre/post ablation comparison."""
    import matrix_mat.if_density.comparator_prepost as cp

    runs_dir = _runs_dir_for(args)
    output_dir = getattr(args, "output_dir", None)
    bio_run = _select_run(runs_dir, args.run_pattern) if getattr(args, "biological", False) else None
    cp.run_comparator_prepost(runs_dir=runs_dir, output_dir=output_dir, scope=getattr(args, "scope", "full"))

    if bio_run is not None:
        import matrix_mat.if_density.comparator_biological as cb
        cb.generate_all_if_biological_figures(run_dir=bio_run, output_dir=output_dir)
    return 0


def handle_if_compare_biological(args: argparse.Namespace) -> int:
    """Generates biological comparison figures across hydrogels."""
    import matrix_mat.if_density.comparator_biological as cb

    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory {run_dir} does not exist")
    else:
        run_dir = _select_run(_runs_dir_for(args), args.run_pattern)
    cb.generate_all_if_biological_figures(run_dir=run_dir, output_dir=getattr(args, "output_dir", None))
    return 0


HANDLERS = {
    ("bf", "preflight"): handle_bf_preflight,
    ("bf", "run"): handle_bf_run,
    ("bf", "plots"): handle_bf_plots,
    ("if", "preflight"): handle_if_preflight,
    ("if", "sweep"): handle_if_sweep,
    ("if", "compare"): handle_if_compare,
    ("if", "compare-prepost"): handle_if_compare_prepost,
    ("if", "compare-biological"): handle_if_compare_biological,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint. Returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_cli_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    handler = HANDLERS.get((str(args.domain), str(getattr(args, "subcommand", ""))))
    if handler is None:
        logger.error(f"Unrecognized command: {args.domain} {getattr(args, 'subcommand', '')}")
        return 2
    try:
        return handler(args)
    except CLIUsageError as e:
        _error(str(e))
        return 2
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError, RuntimeError) as e:
        _error(f"{args.domain} {args.subcommand}: {e}")
        logger.debug("Traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
