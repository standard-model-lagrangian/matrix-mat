#!/usr/bin/env python3
"""
Forwarding CLI entry point for presentation figure 6 (Raw Well Volumes).
Delegates to spheroid_pipeline_v2.presentation_plots.generate_all_presentation_figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spheroid_pipeline_v2.presentation_plots import (
    compute_well_growth_comparison,
    generate_all_presentation_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Presentation Figure 6: Raw Volume Numbers Per Well."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory containing objects.csv and pairs.csv (default: output).",
    )
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    objs_p = out_p / "objects.csv"
    pairs_p = out_p / "pairs.csv"

    if not objs_p.exists():
        print(f"Error: {objs_p} not found. Run pipeline first or specify valid --output-dir.", file=sys.stderr)
        return 1

    objs_df = pd.read_csv(objs_p)
    pairs_df = pd.read_csv(pairs_p) if pairs_p.exists() else pd.DataFrame()
    wells_csv = out_p / "well_growth_comparison.csv"
    wells_df = compute_well_growth_comparison(objs_df, wells_csv)
    generate_all_presentation_figures(out_p, objs_df, pairs_df, wells_df)
    print(f"[+] Presentation figure 6 saved in {out_p}/figures/presentation_fig6_raw_well_volumes.png and .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
