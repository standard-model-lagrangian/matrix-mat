#!/usr/bin/env python3
"""
Forwarding CLI entry point for well-level growth analysis.
Delegates to spheroid_pipeline_v2.presentation_plots.compute_well_growth_comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spheroid_pipeline_v2.presentation_plots import compute_well_growth_comparison


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute well-level spheroid growth comparison table."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing objects.csv (default: output).",
    )
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    objs_p = out_p / "objects.csv"
    if not objs_p.exists():
        print(f"Error: {objs_p} not found. Run pipeline first or specify valid --output-dir.", file=sys.stderr)
        return 1

    objs_df = pd.read_csv(objs_p)
    out_csv = out_p / "well_growth_comparison.csv"
    compute_well_growth_comparison(objs_df, out_csv)
    print(f"[+] Saved well growth comparison table to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
