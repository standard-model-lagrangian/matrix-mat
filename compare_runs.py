#!/usr/bin/env python3
"""
Standalone entry point for cross-run comparison:
Scans runs/ and generates runs/comparison/ containing metrics_by_run.csv,
per_image_deltas.csv, disagreement_panels/, and comparison_report.md.
Optionally generates publication-grade biological comparison figures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spheroid_if_sweep.biological_plots import generate_all_if_biological_figures
from spheroid_if_sweep.comparator import generate_cross_run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-run comparison generator for spheroid IF segmentation.")
    parser.add_argument("--runs", type=str, default="runs", help="Path to runs directory (default: runs).")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: runs/comparison).")
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

    runs_path = Path(args.runs)
    if not runs_path.exists():
        print(f"Error: Runs directory not found: {runs_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = generate_cross_run_comparison(runs_dir=runs_path, output_dir=args.output)
    print(f"Cross-run comparison successfully generated in: {out_dir}")

    if args.biological:
        target_dir = runs_path / args.target_run
        if target_dir.exists():
            bio_out = out_dir / "biological_figures"
            generate_all_if_biological_figures(run_dir=target_dir, output_dir=bio_out)
            print(f"[+] Biological comparison figures generated in: {bio_out}")
        else:
            print(f"Warning: Target run directory {target_dir} not found for biological figures.")


if __name__ == "__main__":
    main()
