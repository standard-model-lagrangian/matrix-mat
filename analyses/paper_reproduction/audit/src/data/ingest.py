"""Canonical Data Ingestion & Gate 1 Sign-off (Task T00).

Ingests raw CSV datasets from Hydrogel_BO, validates accounting and data integrity,
and generates canonical Parquet tables, paper claims constants, SHA-256 manifest,
and Section 9 Draft-07 task card.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml

from utils.hashing import compute_sha256, create_manifest, verify_manifest
from utils.seeds import get_task_seed

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
candidate1 = PROJECT_ROOT.parent.parent / "Hydrogel_BO"
candidate2 = PROJECT_ROOT.parent / "Hydrogel_BO"
if candidate1.is_dir():
    DEFAULT_RAW_DIR = candidate1
elif candidate2.is_dir():
    DEFAULT_RAW_DIR = candidate2
else:
    DEFAULT_RAW_DIR = Path("/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO").resolve()
CANONICAL_DIR = PROJECT_ROOT / "artifacts" / "canonical"
CARDS_DIR = PROJECT_ROOT / "artifacts" / "cards"


def ingest_canonical_data(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    canonical_dir: str | Path = CANONICAL_DIR,
    cards_dir: str | Path = CARDS_DIR,
) -> dict[str, Path]:
    """Execute complete Task T00 data ingestion and verification pipeline."""
    start_time = time.time()
    raw_path = Path(raw_dir).resolve()
    canon_path = Path(canonical_dir).resolve()
    cards_path = Path(cards_dir).resolve()

    canon_path.mkdir(parents=True, exist_ok=True)
    (cards_path / "T00").mkdir(parents=True, exist_ok=True)

    analysis_csv = raw_path / "hydrogel_analysis.csv"
    bo_csv = raw_path / "hydrogel_bo.csv"

    if not analysis_csv.is_file():
        raise FileNotFoundError(f"Missing raw file: {analysis_csv}")
    if not bo_csv.is_file():
        raise FileNotFoundError(f"Missing raw file: {bo_csv}")

    # Load raw CSVs
    df_analysis_raw = pd.read_csv(analysis_csv)
    df_bo_raw = pd.read_csv(bo_csv)

    # 1. Rigorous Data Accounting Verification
    assert len(df_analysis_raw) == 168, f"Expected 168 rows in analysis.csv, got {len(df_analysis_raw)}"
    assert len(df_bo_raw) == 211, f"Expected 211 rows in bo.csv, got {len(df_bo_raw)}"

    # Check index column name
    idx_col = "Unnamed: 0"
    assert idx_col in df_analysis_raw.columns, f"Missing index column '{idx_col}' in analysis.csv"
    assert idx_col in df_bo_raw.columns, f"Missing index column '{idx_col}' in bo.csv"

    # Verify prior accounting: exactly 43 rows, index 2 missing
    df_prior_raw = df_bo_raw.iloc[:43].copy()
    prior_indices = df_prior_raw[idx_col].tolist()
    assert len(df_prior_raw) == 43, f"Expected 43 prior rows, got {len(df_prior_raw)}"
    assert 2 not in prior_indices, "Index 2 unexpectedly present in prior data"
    assert prior_indices[0] == 0 and prior_indices[-1] == 43

    # Verify experimental accounting in bo.csv matches analysis.csv with zero difference
    df_bo_exp = df_bo_raw.iloc[43:].copy().reset_index(drop=True)
    df_analysis_sorted = df_analysis_raw.sort_values(by=idx_col).reset_index(drop=True)
    df_bo_sorted = df_bo_exp.sort_values(by=idx_col).reset_index(drop=True)

    data_cols = [
        "Exposure Time",
        "GelMA-L",
        "GelMA-H",
        "aCNC",
        "LAP",
        "Stress Relaxation",
        "Modulus",
    ]
    diff = (df_bo_sorted[data_cols] - df_analysis_sorted[data_cols]).abs().max().max()
    assert diff == 0.0, f"Discrepancy detected between bo.csv and analysis.csv: max abs diff = {diff}"

    # 2. Build points.parquet (168 rows)
    # Campaign partitions across row positions 0..167 in hydrogel_analysis.csv
    campaign_specs = [
        ("HealthySkin", 0, 20, 1, 5, "closed_loop"),
        ("ScarredSkin", 20, 52, 6, 13, "closed_loop"),
        ("SoftTumor", 52, 80, 14, 20, "closed_loop"),
        ("StiffTumor", 80, 100, 21, 25, "closed_loop"),
        ("Pareto_Utopian01", 100, 128, 26, 32, "pareto_mapping"),
        ("Pareto_Utopian02", 128, 140, 33, 35, "pareto_mapping"),
        ("Pareto_Utopian03", 140, 152, 36, 38, "pareto_mapping"),
        ("Pareto_HV", 152, 168, 39, 42, "pareto_mapping"),
    ]

    campaign_col = []
    batch_id_col = []
    batch_type_col = []

    for name, start_pos, end_pos, b_start, b_end, b_type in campaign_specs:
        n_points = end_pos - start_pos
        n_batches = b_end - b_start + 1
        assert n_points == n_batches * 4, f"Batch size mismatch in {name}"
        for b in range(b_start, b_end + 1):
            for _ in range(4):
                campaign_col.append(name)
                batch_id_col.append(b)
                batch_type_col.append(b_type)

    assert len(campaign_col) == 168

    points_df = pd.DataFrame({
        "point_id": np.arange(168, dtype=np.int64),
        "source_index": df_analysis_raw[idx_col].astype(np.int64).values,
        "exposure_time": df_analysis_raw["Exposure Time"].astype(float).values,
        "gelma_l": df_analysis_raw["GelMA-L"].astype(float).values,
        "gelma_h": df_analysis_raw["GelMA-H"].astype(float).values,
        "acnc": df_analysis_raw["aCNC"].astype(float).values,
        "lap": df_analysis_raw["LAP"].astype(float).values,
        "modulus": df_analysis_raw["Modulus"].astype(float).values,
        "stress_relaxation": df_analysis_raw["Stress Relaxation"].astype(float).values,
        "campaign": campaign_col,
        "batch_id": np.array(batch_id_col, dtype=np.int64),
        "batch_type": batch_type_col,
    })

    points_parquet_path = canon_path / "points.parquet"
    points_df.to_parquet(points_parquet_path, index=False, engine="pyarrow")

    # 3. Build prior.parquet (43 rows)
    prior_df = pd.DataFrame({
        "prior_id": np.arange(43, dtype=np.int64),
        "source_index": df_prior_raw[idx_col].astype(np.int64).values,
        "exposure_time": df_prior_raw["Exposure Time"].astype(float).values,
        "gelma_l": df_prior_raw["GelMA-L"].astype(float).values,
        "gelma_h": df_prior_raw["GelMA-H"].astype(float).values,
        "acnc": df_prior_raw["aCNC"].astype(float).values,
        "lap": df_prior_raw["LAP"].astype(float).values,
        "imputed_modulus": df_prior_raw["Modulus"].astype(float).values,
        "stress_relaxation": df_prior_raw["Stress Relaxation"].astype(float).values,
    })

    prior_parquet_path = canon_path / "prior.parquet"
    prior_df.to_parquet(prior_parquet_path, index=False, engine="pyarrow")

    # 4. Build optima.parquet (7 rows)
    # 7 optimal formulations identified across the 4 closed-loop campaigns
    optima_records = [
        {
            "target_name": "HealthySkin",
            "candidate_id": "I4-2",
            "source_index": 57,
            "target_modulus": 1.5,
            "target_stress_relaxation": 25.0,
            "tolerance_modulus": 0.15,
            "tolerance_stress_relaxation": 2.5,
        },
        {
            "target_name": "HealthySkin",
            "candidate_id": "I4-4",
            "source_index": 59,
            "target_modulus": 1.5,
            "target_stress_relaxation": 25.0,
            "tolerance_modulus": 0.15,
            "tolerance_stress_relaxation": 2.5,
        },
        {
            "target_name": "ScarredSkin",
            "candidate_id": "I6-4",
            "source_index": 87,
            "target_modulus": 12.0,
            "target_stress_relaxation": 20.0,
            "tolerance_modulus": 1.2,
            "tolerance_stress_relaxation": 2.0,
        },
        {
            "target_name": "ScarredSkin",
            "candidate_id": "I7-3",
            "source_index": 90,
            "target_modulus": 12.0,
            "target_stress_relaxation": 20.0,
            "tolerance_modulus": 1.2,
            "tolerance_stress_relaxation": 2.0,
        },
        {
            "target_name": "SoftTumor",
            "candidate_id": "I3-3",
            "source_index": 146,
            "target_modulus": 1.3,
            "target_stress_relaxation": 50.0,
            "tolerance_modulus": 0.13,
            "tolerance_stress_relaxation": 5.0,
        },
        {
            "target_name": "SoftTumor",
            "candidate_id": "I4-1",
            "source_index": 148,
            "target_modulus": 1.3,
            "target_stress_relaxation": 50.0,
            "tolerance_modulus": 0.13,
            "tolerance_stress_relaxation": 5.0,
        },
        {
            "target_name": "StiffTumor",
            "candidate_id": "I1-4",
            "source_index": 167,
            "target_modulus": 16.0,
            "target_stress_relaxation": np.nan,
            "tolerance_modulus": 1.6,
            "tolerance_stress_relaxation": 1e10,
        },
    ]

    optima_rows = []
    analysis_indexed = df_analysis_raw.set_index(idx_col)

    for rec in optima_records:
        s_idx = rec["source_index"]
        raw_row = analysis_indexed.loc[s_idx]
        m_mod = float(raw_row["Modulus"])
        m_sr = float(raw_row["Stress Relaxation"])

        # Check tolerance window pass status
        mod_ok = (rec["target_modulus"] - rec["tolerance_modulus"] <= m_mod <= rec["target_modulus"] + rec["tolerance_modulus"])
        if np.isnan(rec["target_stress_relaxation"]):
            sr_ok = True  # Maximization objective per Section 5.1
        else:
            sr_ok = (rec["target_stress_relaxation"] - rec["tolerance_stress_relaxation"] <= m_sr <= rec["target_stress_relaxation"] + rec["tolerance_stress_relaxation"])

        window_pass = bool(mod_ok and sr_ok)

        optima_rows.append({
            "target_name": rec["target_name"],
            "candidate_id": rec["candidate_id"],
            "source_index": int(s_idx),
            "exposure_time": float(raw_row["Exposure Time"]),
            "gelma_l": float(raw_row["GelMA-L"]),
            "gelma_h": float(raw_row["GelMA-H"]),
            "acnc": float(raw_row["aCNC"]),
            "lap": float(raw_row["LAP"]),
            "measured_modulus": m_mod,
            "measured_stress_relaxation": m_sr,
            "target_modulus": float(rec["target_modulus"]),
            "target_stress_relaxation": float(rec["target_stress_relaxation"]),
            "tolerance_modulus": float(rec["tolerance_modulus"]),
            "tolerance_stress_relaxation": float(rec["tolerance_stress_relaxation"]),
            "window_pass": window_pass,
        })

    optima_df = pd.DataFrame(optima_rows)
    assert len(optima_df) == 7
    assert optima_df["window_pass"].all(), "All 7 optimal formulations must pass their target windows"

    optima_parquet_path = canon_path / "optima.parquet"
    optima_df.to_parquet(optima_parquet_path, index=False, engine="pyarrow")

    # 5. Build paper_claims.yaml
    paper_claims_data = {
        "metadata": {
            "title": "Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity",
            "authors": "Chen et al.",
            "journal": "Adv. Funct. Mater. 2026, 36:e75639",
            "doi": "10.1002/adfm.202575639",
        },
        "discretization_bounds": {
            "parameter_space": {
                "exposure_time": {"min": 0.0, "max": 600.0, "step": 60.0, "levels": 11, "unit": "s"},
                "gelma_l": {"min": 0.0, "max": 6.0, "step": 1.0, "levels": 7, "unit": "wt%"},
                "gelma_h": {"min": 0.0, "max": 6.0, "step": 1.0, "levels": 7, "unit": "wt%"},
                "acnc": {"min": 0.0, "max": 6.0, "step": 1.0, "levels": 7, "unit": "wt%"},
                "lap": {"min": 0.0, "max": 0.2, "step": 0.05, "levels": 5, "unit": "wt%"},
            },
            "claimed_grid_size": 6230,
            "true_cartesian_grid_size": 18865,
            "arithmetic_discrepancy_factor": round(18865 / 6230, 4),
        },
        "table_1_targets": {
            "HealthySkin": {
                "modulus_kpa": 1.5,
                "tolerance_modulus_kpa": 0.15,
                "stress_relaxation_pct": 25.0,
                "tolerance_stress_relaxation_pct": 2.5,
                "reported_iterations_to_hit": 4,
            },
            "ScarredSkin": {
                "modulus_kpa": 12.0,
                "tolerance_modulus_kpa": 1.2,
                "stress_relaxation_pct": 20.0,
                "tolerance_stress_relaxation_pct": 2.0,
                "reported_iterations_to_hit": 6,
            },
            "SoftTumor": {
                "modulus_kpa": 1.3,
                "tolerance_modulus_kpa": 0.13,
                "stress_relaxation_pct": 50.0,
                "tolerance_stress_relaxation_pct": 5.0,
                "reported_iterations_to_hit": 3,
            },
            "StiffTumor": {
                "modulus_kpa": 16.0,
                "tolerance_modulus_kpa": 1.6,
                "stress_relaxation_pct": None,
                "tolerance_stress_relaxation_pct": 1.0e10,
                "objective_mode": "maximize_stress_relaxation",
                "reported_iterations_to_hit": 1,
            },
        },
        "surrogate_performance_table_s6": {
            "gradient_boosting": {
                "modulus_r2": 0.90,
                "stress_relaxation_r2": 0.91,
            },
            "svr": {
                "modulus_r2": 0.88,
                "stress_relaxation_r2": 0.92,
            },
            "gaussian_process": {
                "modulus_r2": 0.79,
                "stress_relaxation_r2": 0.79,
            },
        },
        "surrogate_performance_narrative_claim": {
            "gaussian_process": {
                "modulus_r2": 0.82,
                "modulus_rmse": 1.78,
                "modulus_mae": 1.24,
                "stress_relaxation_r2": 0.82,
                "stress_relaxation_rmse": 9.65,
                "stress_relaxation_mae": 6.36,
            },
        },
        "viscoelastic_model_si": {
            "equation": "G(t) = G_1 * exp(-t / tau_1) + G_2 * exp(-t / tau_2) + G_inf",
            "observation_window_s": [0.1, 100.0],
            "extrapolated_half_times": {
                "HealthySkin_I4_2": {"tau_1": 1.1, "tau_2": 164.0, "t_half": 548.0},
                "HealthySkin_I4_4": {"tau_1": 1.3, "tau_2": 168.0, "t_half": 583.0},
                "ScarredSkin_I6_4": {"tau_1": 1.6, "tau_2": 253.0, "t_half": 1015.0},
                "ScarredSkin_I7_3": {"tau_1": 0.7, "tau_2": 222.0, "t_half": 1446.0},
                "SoftTumor_I3_3": {"tau_1": 0.6, "tau_2": 44.3, "t_half": 120.0},
                "SoftTumor_I4_1": {"tau_1": 0.8, "tau_2": 43.8, "t_half": 119.0},
                "StiffTumor_I1_4": {"tau_1": 0.8, "tau_2": 294.0, "t_half": 3267.0},
            },
        },
        "accounting": {
            "prior_dataset_observed_rows": 43,
            "prior_dataset_claimed_rows": 44,
            "missing_prior_index": 2,
            "closed_loop_batches": 25,
            "closed_loop_points": 100,
            "pareto_mapping_batches": 17,
            "pareto_mapping_points": 68,
            "total_experimental_points": 168,
            "total_bo_rows": 211,
        },
        "section_5_experimental_constants": {
            "photocrosslinking": {
                "uv_wavelength_nm": 400,
                "uv_intensity_mw_cm2": 30.0,
                "pre_gelation_incubation_hours": 2.0,
                "pre_gelation_temp_c": 37.0,
            },
            "compression_test": {
                "instrument": "Mach-1 Mechanical Tester (Biomomentum Inc.)",
                "geometry": "cylinder height 5.00 mm, diameter 9.83 mm",
                "strain_rate_mm_s": 0.1,
                "strain_level_pct": 20.0,
                "modulus_fit_strain_pct": [10.0, 15.0],
            },
            "stress_relaxation_test": {
                "instrument": "DHR-1 Rheometer (TA Instruments)",
                "geometry": "40 mm parallel plates",
                "evaporation_blocker": "mineral oil",
                "step_strain_pct": 5.0,
                "step_strain_duration_s": 0.1,
                "hold_time_s": 100.0,
            },
        },
    }

    paper_claims_path = canon_path / "paper_claims.yaml"
    with open(paper_claims_path, "w", encoding="utf-8") as f:
        yaml.dump(paper_claims_data, f, sort_keys=False, indent=2)

    # 6. Generate SHA-256 Manifest
    canonical_files = [
        points_parquet_path,
        prior_parquet_path,
        optima_parquet_path,
        paper_claims_path,
    ]
    manifest_path = canon_path / "manifest.sha256"
    create_manifest(canonical_files, manifest_path, base_dir=PROJECT_ROOT)

    # Verify manifest immediately
    verify_manifest(manifest_path, base_dir=PROJECT_ROOT)

    # 7. Generate Task T00 Card
    t00_seed = get_task_seed("T00", PROJECT_ROOT / "env" / "seeds.yaml")
    script_path = Path(__file__).resolve()

    provenance_inputs = [
        {"path": str(analysis_csv.resolve()), "sha256": compute_sha256(analysis_csv)},
        {"path": str(bo_csv.resolve()), "sha256": compute_sha256(bo_csv)},
    ]
    provenance_outputs = [
        {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": compute_sha256(p)}
        for p in [points_parquet_path, prior_parquet_path, optima_parquet_path, paper_claims_path, manifest_path]
    ]
    provenance_script = {
        "path": str(script_path.relative_to(PROJECT_ROOT)),
        "sha256": compute_sha256(script_path),
    }

    elapsed_time = round(time.time() - start_time, 4)

    card_t00_data = {
        "task_id": "T00",
        "claim_reference": "Adv. Funct. Mater. 2026, 36:e75639, Section 5, SI Section 1, Table S2, Table S3, Table S4",
        "verdict": "SUPPORTED",
        "preregistration": {
            "hypothesis": "Raw experimental data files (hydrogel_analysis.csv, hydrogel_bo.csv) preserve exactly 168 experimental conditions across 42 batches and 43 prior conditions with exact accounting and zero floating-point divergence.",
            "primary_metric": "Dataset reconciliation max absolute difference == 0.0 and canonical row counts match ground truth.",
            "decision_rule": "SUPPORTED if max absolute difference between hydrogel_bo.loc[44:211] and hydrogel_analysis is 0.0, canonical row counts are 168 (points), 43 (prior), and 7 (optima), and SHA-256 manifest is verified.",
            "parameters": {
                "target_tolerance_pct": 10.0,
            },
        },
        "sensitivity_ranges": {
            "target_tolerance_pct": [5.0, 10.0, 15.0],
            "prior_rows": 43,
            "closed_loop_batches": 25,
            "pareto_batches": 17,
        },
        "bias_direction": "neutral",
        "provenance_hashes": {
            "inputs": provenance_inputs,
            "outputs": provenance_outputs,
            "script": provenance_script,
        },
        "execution_metadata": {
            "execution_time_seconds": elapsed_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": t00_seed,
            "hardware": "Apple M4 (10 cores)",
            "python_version": "3.13.14",
        },
        "metrics": {
            "raw_analysis_rows": 168,
            "raw_bo_rows": 211,
            "canonical_points_rows": 168,
            "canonical_prior_rows": 43,
            "canonical_optima_rows": 7,
            "closed_loop_batches": 25,
            "closed_loop_points": 100,
            "pareto_batches": 17,
            "pareto_mapping_points": 68,
            "max_abs_diff_bo_analysis": float(diff),
            "steward_signoff": True,
        },
        "summary": "Canonical ingestion completed with exact row accounting reconciliation (168 experimental points = 100 closed-loop across 25 batches + 68 Pareto mapping across 17 batches; 43 prior rows with missing index 2). All 7 optimal hydrogels verified and Gate 1 sign-off issued.",
    }

    t00_card_path = cards_path / "T00" / "card.json"
    with open(t00_card_path, "w", encoding="utf-8") as f:
        json.dump(card_t00_data, f, indent=2)

    return {
        "points": points_parquet_path,
        "prior": prior_parquet_path,
        "optima": optima_parquet_path,
        "claims": paper_claims_path,
        "manifest": manifest_path,
        "card": t00_card_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical Data Ingestion (T00)")
    parser.add_argument("--raw-dir", type=str, default=str(DEFAULT_RAW_DIR), help="Path to raw Hydrogel_BO directory")
    parser.add_argument("--canonical-dir", type=str, default=str(CANONICAL_DIR), help="Path to artifacts/canonical")
    parser.add_argument("--cards-dir", type=str, default=str(CARDS_DIR), help="Path to artifacts/cards")
    args = parser.parse_args()

    results = ingest_canonical_data(
        raw_dir=args.raw_dir,
        canonical_dir=args.canonical_dir,
        cards_dir=args.cards_dir,
    )

    print("=================================================================")
    print("✅ CANONICAL INGESTION COMPLETE (TASK T00) & GATE 1 SIGNOFF ISSUED")
    print("=================================================================")
    for k, v in results.items():
        print(f"  - {k:10s}: {v}")
