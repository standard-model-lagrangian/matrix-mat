# Project: Multi-Agent Reproduction & Adversarial Audit of Chen et al. 2026
**Target Paper**: Chen et al., *Adv. Funct. Mater.* 2026, 36:e75639 ("Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity")  
**Project Working Directory**: `/Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit`  
**Orchestration Metadata Directory**: `/Users/frederick/Visual Studio /Material discovery/.agents/orchestrator_audit`  
**Upstream Repository**: `/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO`  
**Master Seed**: `20260919` (`env/seeds.yaml`)

---

## Architecture
The audit is structured into a Dual-Track Architecture:
1. **Implementation Track**: Modular Python packages and audit modules under `src/` executing reproducible analysis tasks (T00–T12), writing validated Section 9 `card.json` files under `artifacts/cards/Txx/`, canonical tables in `artifacts/canonical/`, and synthesis reports in `report/`.
2. **E2E Testing Track**: Independent opaque-box verification suite under `tests/` verifying all schemas, manifests, assertions, and reproducible invariants, gated by `make verify` and publishing `TEST_READY.md`.

---

## Feature Inventory

| # | Feature / Task | Description | Milestone | Source |
|---|----------------|-------------|-----------|--------|
| 1 | **T00: Canonical Data Ingestion** | Ingest raw CSVs/SI docx, produce canonical parquet tables (`points.parquet`, `prior.parquet`, `optima.parquet`, `paper_claims.yaml`, `manifest.sha256`) with Gate 1 sign-off | M1 | R1, T00 |
| 2 | **T01: Prospective GP Surrogate** | Sequential 1-step-ahead GP evaluation ($t=1\dots 25$) on $D_{<t}$ across 8 kernel/noise configurations, soft-regime RMSE vs $\pm 0.13\text{--}0.15$ kPa tolerance, `fig_soft_regime.pdf` | M2 | R2, T01 |
| 3 | **T04: Imputation & Prior Fingerprint** | ARD GP lengthscale fingerprint on 43-row prior vs 100 closed-loop points (detecting LAP zero-variance), propagated imputation error ($R^2=0.80 \implies \text{RMSE} \approx 1.44$ kPa) | M2 | R5, T04 |
| 4 | **T02a: Model-Free Permutations** | 10,000 permutations of 168-pool to evaluate iterations-to-first-hit vs reported Atlas iterations (4, 6, 3, 1), quantifying AL enrichment bias | M3 | R3, T02a |
| 5 | **T02b: Digital Twin Benchmark** | Scarred-skin benchmark replication ($\ge 200$ seeds) on digital twin simulator with Atlas vs Random vs Grid vs Sobol/LHS, documenting structural circularity | M3 | R3, T02b |
| 6 | **T03: Cross-Validation Protocols** | Compare Random 5-fold CV vs forward-chaining vs LOCO vs regime-out; nearest-neighbor KS test vs 10k Sobol; selection leakage audit in 80/20 emulation | M3 | R4, T03 |
| 7 | **T05: Simpler-Model Counterfactual** | Response surfaces (OLS with 21 terms, log($E$) OLS, GB twin, GP) under LOCO; learning curves ($N \in \{25, 30, 40, 50, 60, 80\}$) vs 32-run fractional factorial | M3 | R6, T05 |
| 8 | **T06: Interpretability & SHAP** | Replicate Fig 5 SHAP on GB twin, stress-test under interventional vs path-dependent, uniform vs empirical background, SVR swap, bootstrap stability, a-CNC PDP/ICE | M4 | R7, T06 |
| 9 | **T07: Pareto & Hypervolume** | 500-order shuffle test, reference-point sensitivity ($(0,0)$ vs nadir vs theoretical), 1000-bootstrap front uncertainty, $P(E > 5\text{ kPa} \land \sigma_{100} > 40\%)$ | M4 | R8, T07 |
| 10 | **T11: Effective Explored Volume** | 5D convex hull volume of 168 points relative to nominal box and unmodeled feasibility constraints | M4 | R9, T11 |
| 11 | **T10: Arithmetic & Paper Consistency** | Grid discretization arithmetic ($18,865$ vs 6,230), main text vs SI reconciliation (Scarred Skin, Stiff Tumor), Table 1 pass/fail, Table S6 model ranking, citation errors, duplicate optima distance | M5 | R9, T10 |
| 12 | **T08: Rheology & Viscoelasticity** | Maxwell viscoelastic parameter identifiability and 33-fold extrapolation (0.1–100 s vs $t_{1/2} > 1000$ s) | M5 | R10, T08 |
| 13 | **T09: Cell Biology Biostatistics** | Pseudoreplication quantification in Figs 2, 3, S6, S7 (mixed-effects models vs pooled ordinary ANOVA on $N=32\text{--}240$ cells/spheroids from $N=3$ repeats) | M5 | R10, T09 |
| 14 | **T12: Red Team & Synthesis** | Adversarial red-team defense for every non-SUPPORTED claim (`red_team.md`), final synthesis (`findings.md`, `claims_table.csv`, `slides_outline.md`) | M5 | R10, T12 |
| 15 | **Final Gate Verification** | Full 7-gate `make verify` execution, SHA-256 manifest verification, card schema validation, adversarial test hardening | M6 | Acceptance |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| **M1** | Canonical Data Ingestion & Infra | Ingest raw data to `artifacts/canonical/`, generate manifest & Gate 1 sign-off, set up `env/seeds.yaml`, card schema, Makefile | none | IN_PROGRESS |
| **M2** | Surrogate Evaluation & Imputation | T01 prospective 1-step GP evaluation ($t=1\dots 25$), soft-regime RMSE, `fig_soft_regime.pdf`, and T04 ARD imputation fingerprint | M1 | PLANNED |
| **M3** | Baselines, CV & Counterfactuals | T02a pool permutations, T02b digital twin benchmark, T03 CV estimand audit, and T05 simpler-model response surfaces | M1 | PLANNED |
| **M4** | Interpretability, Pareto & Volume | T06 SHAP robustness, T07 Pareto hypervolume shuffle/bootstrap, and T11 5D explored volume | M1 | PLANNED |
| **M5** | Paper Consistency, Rheology, Stats & Red Team | T10 consistency ledger, T08 rheology extrapolation, T09 cell biostatistics pseudoreplication, and T12 red team defense + synthesis reports | M2, M3, M4 | PLANNED |
| **M6** | Final Gate & Verification Hardening | 100% pass of E2E verification test suite (`make verify`), Tier 5 challenger coverage hardening, final sign-off | M1, M2, M3, M4, M5 | PLANNED |

---

## Interface Contracts

### M1 ↔ Downstream Tasks (Data Contract)
- `artifacts/canonical/points.parquet`:
  - Columns: `point_id` (int, 0..167), `source_index` (int, 44..211), `exposure_time` (float, s), `gelma_l` (float, wt%), `gelma_h` (float, wt%), `acnc` (float, wt%), `lap` (float, wt%), `modulus` (float, kPa), `stress_relaxation` (float, %), `campaign` (string: HealthySkin, ScarredSkin, SoftTumor, StiffTumor, Pareto_Utopian01, Pareto_Utopian02, Pareto_Utopian03, Pareto_HV), `batch_id` (int, 1..42), `batch_type` (string: closed_loop or pareto_mapping).
- `artifacts/canonical/prior.parquet`:
  - Columns: `prior_id` (int, 0..42), `source_index` (int, 0..43 without 2), 5 formulation features, `imputed_modulus` (float, kPa), `stress_relaxation` (float, %).
- `artifacts/canonical/optima.parquet`:
  - Columns: `target_name`, `candidate_id`, `source_index`, 5 formulation features, `measured_modulus`, `measured_stress_relaxation`, `target_modulus`, `target_stress_relaxation`, `tolerance_modulus`, `tolerance_stress_relaxation`, `window_pass` (bool).
- `artifacts/canonical/paper_claims.yaml`:
  - Structured constants for Section 5, Table 1 targets, Table S6 metrics, discretization bounds.
- `artifacts/canonical/manifest.sha256`:
  - SHA-256 hex digests of the canonical tables.

### Section 9 Task Cards (`artifacts/cards/Txx/card.json`)
Every task $T \in \{T00, T01, T02a, T02b, T03, T04, T05, T06, T07a, T07b, T07c, T07d, T08, T09, T10, T11, T12\}$ must produce a `card.json` validating against `src/utils/card_schema.json`.
Mandatory fields: `task_id`, `claim_reference`, `verdict`, `preregistration`, `sensitivity_ranges`, `bias_direction`, `provenance_hashes`, `execution_metadata`, `metrics`, `summary`.

---

## Code Layout

```text
Reproducing paper /audit/
├── Makefile
├── README.md
├── TEST_READY.md (published by E2E track)
├── env/
│   ├── seeds.yaml
│   ├── environment.yaml
│   └── requirements.txt
├── artifacts/
│   ├── canonical/
│   │   ├── points.parquet
│   │   ├── prior.parquet
│   │   ├── optima.parquet
│   │   ├── paper_claims.yaml
│   │   └── manifest.sha256
│   └── cards/
│       ├── T00/card.json
│       ├── T01/card.json
│       ├── T02a/card.json
│       ├── T02b/card.json
│       ├── T03/card.json
│       ├── T04/card.json
│       ├── T05/card.json
│       ├── T06/card.json
│       ├── T07a/card.json
│       ├── T07b/card.json
│       ├── T07c/card.json
│       ├── T07d/card.json
│       ├── T08/card.json
│       ├── T09/card.json
│       ├── T10/card.json
│       ├── T11/card.json
│       └── T12/card.json
├── src/
│   ├── data/
│   │   └── ingest.py
│   ├── models/
│   │   ├── gp_surrogate.py
│   │   └── digital_twin.py
│   ├── audit/
│   │   ├── prospective.py
│   │   ├── baselines.py
│   │   ├── cv_estimand.py
│   │   ├── imputation.py
│   │   ├── counterfactual.py
│   │   ├── interpretability.py
│   │   ├── pareto.py
│   │   ├── consistency.py
│   │   ├── volume.py
│   │   └── rheology.py
│   └── utils/
│       ├── seeds.py
│       ├── hashing.py
│       └── card_validator.py
├── scripts/
│   ├── run_all_tasks.py
│   ├── verify_all.py
│   └── generate_report.py
├── tests/
│   ├── test_canonical_data.py
│   ├── test_cards_schema.py
│   ├── test_hashes.py
│   └── test_seeds.py
└── report/
    ├── findings.md
    ├── claims_table.csv
    ├── red_team.md
    ├── slides_outline.md
    └── figures/
        └── fig_soft_regime.pdf
```
