#!/usr/bin/env python3
"""
CLI entry point for pre/post processing comparison and verdict generation.
Delegates to spheroid_if_sweep.comparator_prepost.run_comparator_prepost.
Optionally generates publication-grade biological comparison figures.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Ensure writable MPLCONFIGDIR
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spheroid_if_sweep.biological_plots import generate_all_if_biological_figures
from spheroid_if_sweep.comparator_prepost import run_comparator_prepost


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pre/post-processing ablation and robustness runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default="runs",
        help="Path to runs directory (default: runs).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output comparison directory (default: runs/comparison_prepost).",
    )
    parser.add_argument(
        "--biological",
        action="store_true",
        help="Also generate publication-grade biological comparison figures across hydrogels.",
    )
    parser.add_argument(
        "--target-run",
        type=str,
        default="104_PP11_full",
        help="Target run slug for biological figures (default: 104_PP11_full).",
    )
    args = parser.parse_args()

    comp_dir = run_comparator_prepost(runs_dir=args.runs, output_dir=args.output)
    print(f"[+] Pre/post comparison successfully generated in: {comp_dir}")

    if args.biological:
        target_dir = Path(args.runs) / args.target_run
        if target_dir.exists():
            bio_out = comp_dir / "biological_figures"
            generate_all_if_biological_figures(run_dir=target_dir, output_dir=bio_out)
            print(f"[+] Biological comparison figures generated in: {bio_out}")
        else:
            print(f"Warning: Target run directory {target_dir} not found for biological figures.")


if __name__ == "__main__":
    main()
