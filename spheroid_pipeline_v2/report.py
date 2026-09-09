"""
Executive Report Generator for Spheroid Volume Pipeline v2.
Generates comprehensive GitHub-Flavored Markdown report (report.md) and standalone
interactive, responsive HTML report (report.html).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from spheroid_pipeline_v2.config import PipelineConfig

logger = logging.getLogger("spheroid_pipeline_v2.report")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a GitHub-flavored Markdown table."""
    if df is None or df.empty:
        return "*No data available.*"

    headers = list(df.columns)
    col_widths = [max(len(str(h)), 8) for h in headers]

    rows_str: List[List[str]] = []
    for _, row in df.iterrows():
        r_str: List[str] = []
        for i, val in enumerate(row):
            if pd.isna(val):
                s = "—"
            elif isinstance(val, float):
                if abs(val) < 0.0001 and val != 0.0:
                    s = f"{val:.2e}"
                elif abs(val) < 10.0:
                    s = f"{val:.3f}"
                elif abs(val) < 1000.0:
                    s = f"{val:.1f}"
                else:
                    s = f"{val:,.0f}"
            else:
                s = str(val)
            col_widths[i] = max(col_widths[i], len(s))
            r_str.append(s)
        rows_str.append(r_str)

    header_line = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    data_lines = []
    for r in rows_str:
        line = "| " + " | ".join(f"{r[i]:<{col_widths[i]}}" for i in range(len(headers))) + " |"
        data_lines.append(line)

    return "\n".join([header_line, separator_line] + data_lines)


class ReportGenerator:
    """Creates report.md and report.html with KPIs, statistical tables, audit tables, and figure embeds."""

    def __init__(self, config: Optional[PipelineConfig] = None, output_dir: Union[str, Path] = "output"):
        self.config = config or PipelineConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        manifest_df: pd.DataFrame,
        objects_df: pd.DataFrame,
        pairs_df: pd.DataFrame,
        condition_summary_df: pd.DataFrame,
        exclusion_audit_df: Optional[pd.DataFrame] = None,
        fig_paths: Optional[Dict[str, Any]] = None,
        overlay_samples: Optional[List[Path]] = None,
        contact_sheets: Optional[Dict[str, Path]] = None,
        validation_df: Optional[pd.DataFrame] = None,
        decisions_path: Optional[Union[str, Path]] = None,
        session_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[Path, Path]:
        """Generates both report.md and report.html in the output directory."""
        md_path = self.output_dir / "report.md"
        html_path = self.output_dir / "report.html"

        # Calculate Executive KPIs
        total_images = len(manifest_df) if manifest_df is not None and not manifest_df.empty else 0
        matched_pairs_fov = int(manifest_df["is_matched_pair"].sum()) if manifest_df is not None and not manifest_df.empty and "is_matched_pair" in manifest_df.columns else 0
        total_objects = len(objects_df) if objects_df is not None and not objects_df.empty else 0
        total_pairs = len(pairs_df) if pairs_df is not None and not pairs_df.empty else 0

        qc_counts = objects_df["qc_flag"].value_counts().to_dict() if objects_df is not None and not objects_df.empty and "qc_flag" in objects_df.columns else {}
        pass_count = qc_counts.get("PASS", 0)
        review_count = qc_counts.get("REVIEW", 0)
        fail_count = qc_counts.get("FAIL", 0)
        pass_rate = (pass_count / total_objects * 100.0) if total_objects > 0 else 0.0

        fov_with_pass = 0
        if objects_df is not None and not objects_df.empty and "qc_flag" in objects_df.columns and "pair_key" in objects_df.columns:
            pass_objs = objects_df[objects_df["qc_flag"] == "PASS"]
            fov_with_pass = pass_objs["pair_key"].nunique()
        call_rate = (fov_with_pass / max(1, matched_pairs_fov)) * 100.0 if matched_pairs_fov > 0 else 100.0

        n_denom_pass = int(pairs_df["denominator_gate_pass"].sum()) if pairs_df is not None and not pairs_df.empty and "denominator_gate_pass" in pairs_df.columns else total_pairs

        # Unpaired Population Summary Table
        pop_summary_df = self._compute_population_summary(objects_df)

        # Read decisions log excerpt
        decisions_text = ""
        if decisions_path:
            d_p = Path(decisions_path)
            if d_p.exists():
                try:
                    decisions_text = d_p.read_text(encoding="utf-8")
                except Exception as e:
                    logger.debug(f"Could not read decisions file: {e}")

        # Build Markdown
        md_content = self._build_markdown(
            total_images=total_images,
            matched_pairs_fov=matched_pairs_fov,
            total_objects=total_objects,
            total_pairs=total_pairs,
            pass_count=pass_count,
            review_count=review_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            call_rate=call_rate,
            n_denom_pass=n_denom_pass,
            condition_summary_df=condition_summary_df,
            pop_summary_df=pop_summary_df,
            exclusion_audit_df=exclusion_audit_df,
            session_df=session_df,
            fig_paths=fig_paths or {},
            overlay_samples=overlay_samples or [],
            contact_sheets=contact_sheets or {},
            validation_df=validation_df,
            decisions_text=decisions_text,
        )
        md_path.write_text(md_content, encoding="utf-8")

        # Build HTML
        html_content = self._build_html(
            total_images=total_images,
            matched_pairs_fov=matched_pairs_fov,
            total_objects=total_objects,
            total_pairs=total_pairs,
            pass_count=pass_count,
            review_count=review_count,
            fail_count=fail_count,
            pass_rate=pass_rate,
            call_rate=call_rate,
            n_denom_pass=n_denom_pass,
            condition_summary_df=condition_summary_df,
            pop_summary_df=pop_summary_df,
            exclusion_audit_df=exclusion_audit_df,
            session_df=session_df,
            fig_paths=fig_paths or {},
            overlay_samples=overlay_samples or [],
            contact_sheets=contact_sheets or {},
            validation_df=validation_df,
            decisions_text=decisions_text,
        )
        html_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Generated executive reports: {md_path.name} and {html_path.name}")
        return md_path, html_path

    def _compute_population_summary(self, objects_df: pd.DataFrame) -> pd.DataFrame:
        """Computes unpaired population growth metrics across all verified PASS spheroids."""
        if objects_df is None or objects_df.empty or "condition" not in objects_df.columns or "qc_flag" not in objects_df.columns:
            return pd.DataFrame()

        pass_objs = objects_df[objects_df["qc_flag"] == "PASS"].copy()
        if pass_objs.empty:
            return pd.DataFrame()

        cond_order = ["Mat", "S34D30", "S40D30", "S43D20", "S46D10", "S50"]
        present_conds = [c for c in cond_order if c in pass_objs["condition"].unique()]

        rows = []
        for cond in present_conds:
            c_df = pass_objs[pass_objs["condition"] == cond]
            v0 = c_df[c_df["timepoint"] == "t0"]["volume_sphere_um3"].dropna().values if "volume_sphere_um3" in c_df.columns and "timepoint" in c_df.columns else np.array([])
            v7 = c_df[c_df["timepoint"] == "t7"]["volume_sphere_um3"].dropna().values if "volume_sphere_um3" in c_df.columns and "timepoint" in c_df.columns else np.array([])

            d0 = c_df[c_df["timepoint"] == "t0"]["equivalent_diameter_um"].dropna().values if "equivalent_diameter_um" in c_df.columns and "timepoint" in c_df.columns else np.array([])
            d7 = c_df[c_df["timepoint"] == "t7"]["equivalent_diameter_um"].dropna().values if "equivalent_diameter_um" in c_df.columns and "timepoint" in c_df.columns else np.array([])

            mwu_p = np.nan
            if len(v0) > 0 and len(v7) > 0:
                try:
                    _, mwu_p = stats.mannwhitneyu(v7, v0, alternative="two-sided")
                except Exception:
                    mwu_p = np.nan

            v0_med = float(np.median(v0)) if len(v0) > 0 else np.nan
            v7_med = float(np.median(v7)) if len(v7) > 0 else np.nan
            d0_med = float(np.median(d0)) if len(d0) > 0 else np.nan
            d7_med = float(np.median(d7)) if len(d7) > 0 else np.nan

            pop_fc = (v7_med / v0_med) if (v0_med > 0 and not np.isnan(v0_med)) else np.nan
            delta_v = (v7_med - v0_med) if (not np.isnan(v7_med) and not np.isnan(v0_med)) else np.nan

            sig = "****" if mwu_p < 0.0001 else "***" if mwu_p < 0.001 else "**" if mwu_p < 0.01 else "*" if mwu_p < 0.05 else "ns"

            rows.append({
                "condition": cond,
                "n_spheroids_t0": len(v0),
                "n_spheroids_t7": len(v7),
                "median_d0_um": d0_med,
                "median_d7_um": d7_med,
                "median_v0_um3": v0_med,
                "median_v7_um3": v7_med,
                "delta_median_v_um3": delta_v,
                "pop_fold_change (Med_V7/Med_V0)": pop_fc,
                "mwu_p_val (t7 vs t0)": mwu_p,
                "significance": sig,
            })

        return pd.DataFrame(rows)

    def _build_markdown(
        self,
        total_images: int,
        matched_pairs_fov: int,
        total_objects: int,
        total_pairs: int,
        pass_count: int,
        review_count: int,
        fail_count: int,
        pass_rate: float,
        call_rate: float,
        n_denom_pass: int,
        condition_summary_df: pd.DataFrame,
        pop_summary_df: pd.DataFrame,
        exclusion_audit_df: Optional[pd.DataFrame],
        session_df: Optional[pd.DataFrame],
        fig_paths: Dict[str, Any],
        overlay_samples: List[Path],
        contact_sheets: Dict[str, Path],
        validation_df: Optional[pd.DataFrame],
        decisions_text: str,
    ) -> str:
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "# Spheroid Volume Analysis Pipeline v2 — Executive Report & Comprehensive Audit",
            f"\n**Execution Timestamp**: `{now_str}`  ",
            f"**Pipeline Version**: `2.0.0` | **Optical Calibration**: `{self.config.default_pixel_size_um:.6f} um/px`\n",
            "---",
            "\n## 1. Executive Summary & Quality Control KPIs\n",
            "| Metric | Value | Description |",
            "|---|---|---|",
            f"| **Total Raw Images** | `{total_images}` | Ingested brightfield TIFF images (Day 0 + Day 7) |",
            f"| **Matched FOV Pairs** | `{matched_pairs_fov}` | FOVs imaged at both Day 0 and Day 7 |",
            f"| **Total Detected Objects** | `{total_objects}` | Individual candidate spheroid instances |",
            f"| **QC PASS Objects** | `{pass_count}` ({pass_rate:.1f}%) | Verified spheroids passing all QC & contrast filters |",
            f"| **QC REVIEW Objects** | `{review_count}` | Irregular/borderline instances flagged for audit |",
            f"| **QC FAIL Objects** | `{fail_count}` | Low-contrast gel artifacts & debris excluded |",
            f"| **Matched Trajectories** | `{total_pairs}` | Bipartite Hungarian matched pairs ($t_0 \\to t_7$) |",
            f"| **Denominator Gate Pass** | `{n_denom_pass}` | Pairs with $d_0 \\ge 60\\,\\mu\\mathrm{{m}}$ included in stats |",
            f"| **Segmentation Call Rate** | `{call_rate:.1f}%` | FOVs yielding $\\ge 1$ verified PASS spheroid |",
            "\n---",
            "\n## 2. Unpaired Population-Level Growth Summary (All PASS Spheroids)\n",
            "> **Note**: In un-registered 3D brightfield imaging where the plate is repositioned on a manual stage at Day 7, comparing the full population distribution of all spheroids at Day 0 vs Day 7 per condition is the most robust and biologically defensible readout.\n",
        ]

        if pop_summary_df is not None and not pop_summary_df.empty:
            lines.append(df_to_markdown_table(pop_summary_df))
        else:
            lines.append("*No population summary data generated.*")

        lines.extend([
            "\n---",
            "\n## 3. Condition-Level Growth & Paired Trajectory Statistics (PASS-Only & Denominator Gate)\n",
        ])

        if condition_summary_df is not None and not condition_summary_df.empty:
            lines.append(df_to_markdown_table(condition_summary_df))
        else:
            lines.append("*No condition summary data generated.*")

        lines.extend([
            "\n*Significance stars: `****` ($p < 0.0001$), `***` ($p < 0.001$), `**` ($p < 0.01$), `*` ($p < 0.05$), `ns` ($p \\ge 0.05$) vs Control `Mat` (Monte Carlo Permutation Test, $M=10,000$).*",
            "\n---",
            "\n## 4. Multilevel 10-Stage Exclusion Audit\n",
        ])

        if exclusion_audit_df is not None and not exclusion_audit_df.empty:
            lines.append(df_to_markdown_table(exclusion_audit_df))
        else:
            lines.append("*Exclusion audit table not available.*")

        lines.extend([
            "\n---",
            "\n## 5. Publication Figures\n",
            "### 5.1 Unpaired Population Volume Distributions (Day 0 vs Day 7)",
            "![Population Distributions](figures/population_volume_distributions.png)\n",
            "### 5.2 Paired Spheroid Volume Fold Change Distribution (PASS-Only)",
            "![Fold Change Violin](figures/fold_change_violin.png)\n",
            "### 5.3 Baseline vs Endpoint Growth Correlation ($V_0$ vs $V_7$)",
            "![V0 vs V7 Scatter](figures/v0_vs_v7_scatter.png)\n",
            "### 5.4 Paired Spheroid Growth Trajectories ($t_0 \\to t_7$)",
            "![Growth Slopegraph](figures/growth_slopegraph.png)\n",
            "### 5.5 Morphological Sensitivity Audit (Discrepancy vs Circularity)",
            "![Circularity Sensitivity Audit](figures/circularity_sensitivity_audit.png)\n",
            "### 5.6 Forensic Audit of Candidate Detection & FAIL Class Gating",
            "![FAIL Class Audit](figures/fail_class_audit.png)\n",
            "### 5.7 Imaging Session Sharpness & Illumination Stability",
            "![Session Focus Audit](figures/session_focus_exposure_audit.png)\n",
        ])

        if contact_sheets:
            lines.extend([
                "\n---",
                "\n## 6. Supervision Contact Sheets & FAIL Audit Grid\n",
            ])
            for cond, cs_p in sorted(contact_sheets.items()):
                lines.append(f"### View: `{cond}`\n![Contact Sheet {cond}](contact_sheets/{cs_p.name})\n")

        if validation_df is not None and not validation_df.empty:
            lines.extend([
                "\n---",
                "\n## 7. Human Supervision & Validation Agreement\n",
                df_to_markdown_table(validation_df),
            ])

        if decisions_text:
            lines.extend([
                "\n---",
                "\n## 8. Architectural & Fallback Decisions Log\n",
                decisions_text,
            ])

        return "\n".join(lines) + "\n"

    def _build_html(
        self,
        total_images: int,
        matched_pairs_fov: int,
        total_objects: int,
        total_pairs: int,
        pass_count: int,
        review_count: int,
        fail_count: int,
        pass_rate: float,
        call_rate: float,
        n_denom_pass: int,
        condition_summary_df: pd.DataFrame,
        pop_summary_df: pd.DataFrame,
        exclusion_audit_df: Optional[pd.DataFrame],
        session_df: Optional[pd.DataFrame],
        fig_paths: Dict[str, Any],
        overlay_samples: List[Path],
        contact_sheets: Dict[str, Path],
        validation_df: Optional[pd.DataFrame],
        decisions_text: str,
    ) -> str:
        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        pop_table_html = pop_summary_df.to_html(
            classes="table table-striped table-hover",
            index=False,
            na_rep="—",
            float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else (f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"),
        ) if pop_summary_df is not None and not pop_summary_df.empty else "<p>No data</p>"

        cond_table_html = condition_summary_df.to_html(
            classes="table table-striped table-hover",
            index=False,
            na_rep="—",
            float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else (f"{x:,.1f}" if abs(x) < 1000 else f"{x:,.0f}"),
        ) if condition_summary_df is not None and not condition_summary_df.empty else "<p>No data</p>"

        audit_table_html = exclusion_audit_df.to_html(
            classes="table table-striped table-hover",
            index=False,
            na_rep="—",
        ) if exclusion_audit_df is not None and not exclusion_audit_df.empty else "<p>No data</p>"

        fig_cards = []
        fig_names = [
            ("Unpaired Population Distributions", "population_volume_distributions.png"),
            ("Paired Fold Change Violins", "fold_change_violin.png"),
            ("Baseline vs Endpoint Scatter", "v0_vs_v7_scatter.png"),
            ("Paired Trajectory Slopegraphs", "growth_slopegraph.png"),
            ("Circularity Sensitivity Audit", "circularity_sensitivity_audit.png"),
            ("FAIL Class Forensic Audit", "fail_class_audit.png"),
            ("Session Focus & Illumination Audit", "session_focus_exposure_audit.png"),
        ]
        for title, fname in fig_names:
            fig_cards.append(f"""
            <div class="col-md-6 mb-4">
                <div class="card shadow-sm h-100">
                    <div class="card-header bg-dark text-white font-weight-bold">{title}</div>
                    <div class="card-body text-center">
                        <img src="figures/{fname}" class="img-fluid rounded" alt="{title}">
                    </div>
                </div>
            </div>
            """)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spheroid Volume Pipeline v2 — Spheroid Volume Analysis Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; }}
        .header-banner {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 2.5rem 1rem; margin-bottom: 2rem; border-radius: 0 0 12px 12px; }}
        .stat-card {{ background: white; border-radius: 8px; padding: 1.25rem; box-shadow: 0 2px 6px rgba(0,0,0,0.06); text-align: center; }}
        .stat-number {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
        .stat-label {{ font-size: 0.85rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
        .card {{ border: none; border-radius: 8px; margin-bottom: 1.5rem; }}
        .table th {{ background: #2c3e50; color: white; font-size: 0.9rem; }}
        .table td {{ font-size: 0.88rem; }}
    </style>
</head>
<body>
    <div class="header-banner text-center">
        <h1>Spheroid Volume Pipeline v2 — Spheroid Volume Analysis Dashboard</h1>
        <p class="lead">Supervisable Multi-Spheroid Image Processing & Growth Analysis (Day 0 vs Day 7)</p>
        <p class="small mb-0">Generated: {now_str} | Optical Scale: {self.config.default_pixel_size_um:.6f} &mu;m/px</p>
    </div>
    <div class="container-fluid" style="max-width: 1400px;">
        <!-- Quick Stats Row -->
        <div class="row mb-4">
            <div class="col-md-2"><div class="stat-card"><div class="stat-number">{total_images}</div><div class="stat-label">Raw Images</div></div></div>
            <div class="col-md-2"><div class="stat-card"><div class="stat-number">{matched_pairs_fov}</div><div class="stat-label">Matched FOVs</div></div></div>
            <div class="col-md-2"><div class="stat-card"><div class="stat-number text-success">{pass_count}</div><div class="stat-label">PASS Spheroids</div></div></div>
            <div class="col-md-2"><div class="stat-card"><div class="stat-number text-warning">{review_count}</div><div class="stat-label">REVIEW Flagged</div></div></div>
            <div class="col-md-2"><div class="stat-card"><div class="stat-number text-danger">{fail_count}</div><div class="stat-label">FAIL Excluded</div></div></div>
            <div class="col-md-2"><div class="stat-card"><div class="stat-number text-primary">{call_rate:.1f}%</div><div class="stat-label">FOV Call Rate</div></div></div>
        </div>

        <!-- Unpaired Population Summary -->
        <div class="card shadow-sm">
            <div class="card-header bg-white font-weight-bold h5 mb-0">Unpaired Population-Level Growth Summary (All PASS Spheroids)</div>
            <div class="card-body table-responsive">
                <p class="text-muted small mb-3">Evaluates the full cohort of verified spheroids at Day 0 vs Day 7 without requiring spatial coordinate persistence across manual re-positioning.</p>
                {pop_table_html}
            </div>
        </div>

        <!-- Paired Trajectories Summary -->
        <div class="card shadow-sm">
            <div class="card-header bg-white font-weight-bold h5 mb-0">Paired Trajectory Statistics (PASS-Only &amp; $d_0 \\geq 60\\,\\mu\\mathrm{{m}}$)</div>
            <div class="card-body table-responsive">
                {cond_table_html}
            </div>
        </div>

        <!-- Exclusion Audit -->
        <div class="card shadow-sm">
            <div class="card-header bg-white font-weight-bold h5 mb-0">Multilevel 10-Stage Exclusion Audit</div>
            <div class="card-body table-responsive">
                {audit_table_html}
            </div>
        </div>

        <!-- Figures Gallery -->
        <div class="row">
            {''.join(fig_cards)}
        </div>
    </div>
</body>
</html>
"""
        return html
