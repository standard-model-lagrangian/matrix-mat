"""
Executive Report Generator for Spheroid Volume Pipeline v2.
Generates comprehensive GitHub-Flavored Markdown report (report.md) and standalone
interactive, responsive HTML report (report.html).
"""

from __future__ import annotations

import datetime
import html as _html
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from matrix_mat.brightfield.config import PipelineConfig
from matrix_mat.reporting._figures import ordered_values

logger = logging.getLogger("matrix_mat.reporting.report")


def _fmt_float(val: float) -> str:
    if val != 0.0 and abs(val) < 0.001:
        return f"{val:.2e}"
    if abs(val) < 10.0:
        return f"{val:.3f}"
    if abs(val) < 1000.0:
        return f"{val:.1f}"
    return f"{val:,.0f}"


def format_p_value(p: Any) -> str:
    """Render a p-value: '<0.001' below 0.001, three decimals otherwise, '—' when missing."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v):
        return "—"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def _is_p_column(name: Any) -> bool:
    n = str(name).lower()
    return n.startswith("p_") or n.startswith("p ") or n in ("p", "p_value", "pvalue") or "min_achievable_p" in n or n == "min_p"


def _format_frame(df: pd.DataFrame) -> pd.DataFrame:
    """String-format a frame for display: p-value columns via format_p_value, other floats via _fmt_float."""
    out = df.copy().astype(object)
    for col in df.columns:
        is_p = _is_p_column(col)
        out[col] = [
            ("—" if (v is None or (isinstance(v, float) and math.isnan(v))) else
             format_p_value(v) if is_p and isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else
             _fmt_float(float(v)) if isinstance(v, (float, np.floating)) else str(v))
            for v in df[col].tolist()
        ]
    return out


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
            if not isinstance(val, str) and pd.isna(val):
                s = "—"
            elif isinstance(val, (float, np.floating)):
                s = format_p_value(val) if _is_p_column(headers[i]) else _fmt_float(float(val))
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
        matched_pairs_fov = int(manifest_df[manifest_df["is_matched_pair"] == True]["pair_key"].nunique()) if manifest_df is not None and not manifest_df.empty and "is_matched_pair" in manifest_df.columns else 0  # noqa: E712
        total_objects = len(objects_df) if objects_df is not None and not objects_df.empty else 0
        total_pairs = len(pairs_df) if pairs_df is not None and not pairs_df.empty else 0

        qc_counts = objects_df["qc_flag"].value_counts().to_dict() if objects_df is not None and not objects_df.empty and "qc_flag" in objects_df.columns else {}
        pass_count = qc_counts.get("PASS", 0)
        review_count = qc_counts.get("REVIEW", 0)
        fail_count = qc_counts.get("FAIL", 0)
        pass_rate = (pass_count / total_objects * 100.0) if total_objects > 0 else 0.0

        call_rate = self._call_rate(manifest_df, objects_df)
        n_denom_pass = self._n_pairs_in_stats(pairs_df)
        self._scale_lines = self._effective_scales(objects_df, pairs_df)

        # Unpaired Population Summary Table
        pop_summary_df = self._compute_population_summary(objects_df, condition_summary_df)

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

    # ------------------------------------------------------------------ helpers
    @property
    def _gate_um(self) -> float:
        for section in ("pairing", "stats"):
            v = getattr(getattr(self.config, section, None), "min_t0_d_um_denominator", None)
            if v is not None:
                return float(v)
        return float("nan")

    @property
    def _control(self) -> str:
        return str(getattr(getattr(self.config, "stats", None), "control_condition", "Mat"))

    @staticmethod
    def _call_rate(manifest_df: Optional[pd.DataFrame], objects_df: Optional[pd.DataFrame]) -> Optional[float]:
        """Percent of matched Day-0/Day-7 FOV pairs with >=1 PASS object at every timepoint.

        Numerator and denominator use the same set of matched pair keys, so the rate is in
        [0, 100]. Returns None (rendered 'n/a') when there are no matched FOV pairs.
        """
        if manifest_df is None or manifest_df.empty or not {"is_matched_pair", "pair_key"}.issubset(manifest_df.columns):
            return None
        matched = set(manifest_df.loc[manifest_df["is_matched_pair"] == True, "pair_key"].dropna())  # noqa: E712
        if not matched:
            return None
        called = 0
        if objects_df is not None and not objects_df.empty and {"qc_flag", "pair_key"}.issubset(objects_df.columns):
            pass_objs = objects_df[(objects_df["qc_flag"] == "PASS") & objects_df["pair_key"].isin(matched)]
            if "timepoint" in pass_objs.columns and "timepoint" in objects_df.columns:
                n_tp = objects_df.loc[objects_df["pair_key"].isin(matched), "timepoint"].nunique()
                per_key = pass_objs.groupby("pair_key")["timepoint"].nunique()
                called = int((per_key >= max(1, n_tp)).sum())
            else:
                called = int(pass_objs["pair_key"].nunique())
        return 100.0 * called / len(matched)

    def _n_pairs_in_stats(self, pairs_df: Optional[pd.DataFrame]) -> int:
        """Pairs that enter the condition statistics: gate passed, QC PASS (+REVIEW if configured), finite FC."""
        if pairs_df is None or pairs_df.empty:
            return 0
        mask = pd.Series(True, index=pairs_df.index)
        if "denominator_gate_pass" in pairs_df.columns:
            mask &= pairs_df["denominator_gate_pass"].fillna(False).astype(bool)
        if "combined_qc_flag" in pairs_df.columns:
            allowed = ["PASS", "REVIEW"] if getattr(getattr(self.config, "stats", None), "include_review_in_stats", False) else ["PASS"]
            mask &= pairs_df["combined_qc_flag"].isin(allowed)
        for col in ("log2_fold_change", "fold_change_volume"):
            if col in pairs_df.columns:
                mask &= np.isfinite(pd.to_numeric(pairs_df[col], errors="coerce"))
                break
        return int(mask.sum())

    def _effective_scales(self, objects_df: Optional[pd.DataFrame], pairs_df: Optional[pd.DataFrame]) -> List[str]:
        """Pixel scale(s) actually used, from the per-object scale columns; else override / config default."""
        for df in (objects_df, pairs_df):
            if df is not None and not df.empty and {"pixel_size_um", "scale_source"}.issubset(df.columns):
                cols = ["pixel_size_um", "scale_source"]
                sub = df.drop_duplicates("image_id") if "image_id" in df.columns else df
                counts = sub[cols].dropna().assign(pixel_size_um=lambda d: d["pixel_size_um"].astype(float).round(6))
                counts = counts.groupby(cols).size().reset_index(name="n")
                if not counts.empty:
                    unit = "images" if "image_id" in df.columns else "rows"
                    return [f"{r.pixel_size_um:.6f} um/px ({r.scale_source}, {int(r.n)} {unit})" for r in counts.itertuples()]
        override = getattr(self.config, "cli_pixel_size_um", None)
        if override is not None:
            return [f"{float(override):.6f} um/px (cli_override)"]
        return [f"{float(self.config.default_pixel_size_um):.6f} um/px (config default; per-image scale not recorded)"]

    def _footnote(self) -> str:
        perms = int(getattr(getattr(self.config, "stats", None), "permutation_iterations", 10000))
        return (
            f"Significance vs control `{self._control}`: gel-level (cluster) label permutation test on the paired "
            f"fold change — exact enumeration of all gel relabellings when there are at most 2,000, otherwise "
            f"Monte Carlo with M={perms:,} seeded permutations; p-values Holm-adjusted across conditions. "
            "Stars: `****` p<0.0001, `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` p>=0.05."
        )

    def _compute_population_summary(
        self,
        objects_df: pd.DataFrame,
        condition_summary_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Computes unpaired population growth metrics across all verified PASS spheroids."""
        if objects_df is None or objects_df.empty or "condition" not in objects_df.columns or "qc_flag" not in objects_df.columns:
            return pd.DataFrame()

        pass_objs = objects_df[objects_df["qc_flag"] == "PASS"].copy()
        if pass_objs.empty:
            return pd.DataFrame()

        present_conds = ordered_values(pass_objs["condition"].unique())

        rows = []
        for cond in present_conds:
            c_df = pass_objs[pass_objs["condition"] == cond]
            v0 = c_df[c_df["timepoint"] == "t0"]["volume_sphere_um3"].dropna().values if "volume_sphere_um3" in c_df.columns and "timepoint" in c_df.columns else np.array([])
            v7 = c_df[c_df["timepoint"] == "t7"]["volume_sphere_um3"].dropna().values if "volume_sphere_um3" in c_df.columns and "timepoint" in c_df.columns else np.array([])

            d0 = c_df[c_df["timepoint"] == "t0"]["equivalent_diameter_um"].dropna().values if "equivalent_diameter_um" in c_df.columns and "timepoint" in c_df.columns else np.array([])
            d7 = c_df[c_df["timepoint"] == "t7"]["equivalent_diameter_um"].dropna().values if "equivalent_diameter_um" in c_df.columns and "timepoint" in c_df.columns else np.array([])

            p_val = np.nan
            p_type = "—"
            sig = "ns"
            if condition_summary_df is not None and not condition_summary_df.empty and "condition" in condition_summary_df.columns:
                cond_row = condition_summary_df[condition_summary_df["condition"] == cond]
                if not cond_row.empty:
                    if "p_adj" in cond_row.columns and pd.notna(cond_row["p_adj"].values[0]):
                        p_val = float(cond_row["p_adj"].values[0])
                        p_type = "Holm-adjusted"
                    elif "p_raw" in cond_row.columns and pd.notna(cond_row["p_raw"].values[0]):
                        p_val = float(cond_row["p_raw"].values[0])
                        p_type = "unadjusted"
                    if "significance" in cond_row.columns and pd.notna(cond_row["significance"].values[0]):
                        sig = str(cond_row["significance"].values[0])

            v0_med = float(np.median(v0)) if len(v0) > 0 else np.nan
            v7_med = float(np.median(v7)) if len(v7) > 0 else np.nan
            d0_med = float(np.median(d0)) if len(d0) > 0 else np.nan
            d7_med = float(np.median(d7)) if len(d7) > 0 else np.nan

            pop_fc = (v7_med / v0_med) if (v0_med > 0 and not np.isnan(v0_med)) else np.nan
            delta_v = (v7_med - v0_med) if (not np.isnan(v7_med) and not np.isnan(v0_med)) else np.nan

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
                f"p (vs {self._control})": p_val,
                "p_type": p_type,
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
        call_rate: Optional[float],
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
        import matrix_mat

        call_rate_str = "n/a" if call_rate is None else f"{call_rate:.1f}%"
        scales = getattr(self, "_scale_lines", None) or self._effective_scales(None, None)

        lines = [
            "# Spheroid Volume Analysis Pipeline v2 — Executive Report & Comprehensive Audit",
            f"\n**Execution Timestamp**: `{now_str}`  ",
            f"**Pipeline Version**: `{matrix_mat.__version__}` | **Optical Calibration (effective)**: "
            + "; ".join(f"`{s}`" for s in scales) + "\n",
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
            f"| **Denominator Gate Pass** | `{n_denom_pass}` | Pairs entering the condition statistics (QC-accepted, $d_0 \\ge {self._gate_um:g}\\,\\mu\\mathrm{{m}}$, finite fold change) |",
            f"| **Segmentation Call Rate** | `{call_rate_str}` | Matched FOV pairs with $\\ge 1$ PASS spheroid at every timepoint |",
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
            f"\n*{self._footnote()}*",
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
        call_rate: Optional[float],
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
        """Self-contained HTML (inline CSS, no CDN, no LaTeX: units are written as plain HTML)."""
        import matrix_mat

        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        esc = _html.escape
        call_rate_str = "n/a" if call_rate is None else f"{call_rate:.1f}%"
        scales = getattr(self, "_scale_lines", None) or self._effective_scales(None, None)
        scale_str = "; ".join(esc(s).replace("um/px", "&micro;m/px") for s in scales)
        gate = f"d<sub>0</sub> &ge; {self._gate_um:g} &micro;m"

        def table(df: Optional[pd.DataFrame]) -> str:
            if df is None or df.empty:
                return "<p>No data</p>"
            return _format_frame(df).to_html(classes="tbl", index=False, na_rep="—", escape=True)

        fig_names = [
            ("Unpaired Population Distributions", "population_volume_distributions.png"),
            ("Paired Fold Change Violins", "fold_change_violin.png"),
            ("Baseline vs Endpoint Scatter", "v0_vs_v7_scatter.png"),
            ("Paired Trajectory Slopegraphs", "growth_slopegraph.png"),
            ("Circularity Sensitivity Audit", "circularity_sensitivity_audit.png"),
            ("FAIL Class Forensic Audit", "fail_class_audit.png"),
            ("Session Focus & Illumination Audit", "session_focus_exposure_audit.png"),
        ]
        fig_cards = "".join(
            f'<div class="card fig"><div class="card-h">{esc(t)}</div>'
            f'<img src="figures/{esc(f)}" alt="{esc(t)}"></div>'
            for t, f in fig_names
        )
        stats = [
            (total_images, "Raw Images", ""), (matched_pairs_fov, "Matched FOVs", ""),
            (pass_count, "PASS Spheroids", "ok"), (review_count, "REVIEW Flagged", "warn"),
            (fail_count, "FAIL Excluded", "bad"), (call_rate_str, "FOV Call Rate", "info"),
            (n_denom_pass, "Pairs in Stats", ""),
        ]
        stat_cards = "".join(
            f'<div class="stat"><div class="num {c}">{esc(str(v))}</div><div class="lbl">{esc(lbl)}</div></div>'
            for v, lbl, c in stats
        )
        footnote = esc(self._footnote()).replace("`", "")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spheroid Volume Analysis Report</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; color: #2c3e50; }}
  header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #fff; padding: 2rem 1rem; text-align: center; }}
  header h1 {{ margin: 0 0 .5rem; font-size: 1.8rem; }}
  header p {{ margin: .25rem 0; }}
  main {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem 1rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .stat, .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,.06); }}
  .stat {{ padding: 1rem; text-align: center; }}
  .num {{ font-size: 1.8rem; font-weight: bold; }}
  .num.ok {{ color: #27ae60; }} .num.warn {{ color: #e67e22; }} .num.bad {{ color: #c0392b; }} .num.info {{ color: #2a5298; }}
  .lbl {{ font-size: .8rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }}
  .card {{ margin-bottom: 1.5rem; overflow-x: auto; }}
  .card-h {{ font-weight: bold; font-size: 1.1rem; padding: .75rem 1rem; border-bottom: 1px solid #eee; }}
  .card-b {{ padding: 1rem; }}
  .muted {{ color: #7f8c8d; font-size: .85rem; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: .85rem; }}
  table.tbl th {{ background: #2c3e50; color: #fff; padding: .4rem .5rem; text-align: left; }}
  table.tbl td {{ padding: .35rem .5rem; border-bottom: 1px solid #eee; }}
  table.tbl tr:nth-child(even) td {{ background: #f6f8fa; }}
  .figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 1rem; }}
  .fig img {{ max-width: 100%; height: auto; display: block; margin: .5rem auto; }}
</style>
</head>
<body>
<header>
  <h1>Spheroid Volume Analysis Report</h1>
  <p>Multi-spheroid image processing and growth analysis (Day 0 vs Day 7)</p>
  <p class="small">Generated: {esc(now_str)} | matrix-mat {esc(str(matrix_mat.__version__))} | Optical scale (effective): {scale_str}</p>
</header>
<main>
  <section class="stats">{stat_cards}</section>

  <div class="card"><div class="card-h">Unpaired Population-Level Growth Summary (All PASS Spheroids)</div>
    <div class="card-b"><p class="muted">Evaluates the full cohort of verified spheroids at Day 0 vs Day 7 without requiring spatial coordinate persistence across manual re-positioning.</p>
    {table(pop_summary_df)}</div></div>

  <div class="card"><div class="card-h">Paired Trajectory Statistics (PASS-Only &amp; {gate})</div>
    <div class="card-b">{table(condition_summary_df)}<p class="muted">{footnote}</p></div></div>

  <div class="card"><div class="card-h">Multilevel Exclusion Audit</div>
    <div class="card-b">{table(exclusion_audit_df)}</div></div>

  <section class="figs">{fig_cards}</section>
</main>
</body>
</html>
"""
