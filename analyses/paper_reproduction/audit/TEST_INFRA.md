# Systematic Test Infrastructure & E2E Verification Architecture
**Audit Project**: Multi-Agent Reproduction & Adversarial Audit of Chen et al., *Adv. Funct. Mater.* 2026, 36:e75639  
**Target Paper**: "Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity"  
**Audit Root Directory**: `/Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit`  
**Authoritative Request**: `/Users/frederick/Visual Studio /Material discovery/.agents/ORIGINAL_REQUEST.md` (§ Follow-up — 2026-09-19T08:39:53Z)  
**Authoritative Specification**: `/Users/frederick/Visual Studio /Material discovery/.agents/orchestrator_audit/PROJECT.md`  
**Master Seed**: `20260919` (`env/seeds.yaml`)  
**Track**: Independent E2E Testing Track  

---

## 1. Executive Summary & Verification Philosophy

The E2E Testing Track serves as an independent, opaque-box verification authority for the multi-agent reproduction and adversarial audit of Chen et al. 2026. Rather than relying on implementation assumptions or internal method structures, the test suite verifies behavior, data contracts, mathematical properties, cryptographic manifests, schema compliance, and pipeline invariants derived strictly from authoritative paper and repository sources.

### Core Testing Tenets:
1. **Opaque-Box Independence**: Tests evaluate system behavior against external specifications (paper tables, SI text, raw repository CSVs, draft-07 JSON schema), never against implementation shortcuts.
2. **Progressive Testability**: During milestone execution, tests dynamically verify all currently implemented features while gracefully discovering downstream artifacts as milestones advance.
3. **Cryptographic Provenance**: Every data artifact and task card is validated against SHA-256 digests and formal JSON schemas.
4. **Adversarial Edge-Case Stress**: Tests explicitly probe known paper contradictions, off-by-one accounting discrepancies, zero-variance feature artifacts, and boundary conditions.
5. **Deterministic Replicability**: All stochastic and derivation processes trace deterministically back to master seed `20260919`.

---

## 2. Authoritative Ground Truth Sources

All test assertions and expected values are anchored in explicit, authoritative sources:

| Source Identifier | Path / Location | Scope & Ground Truth Content |
|---|---|---|
| **Authoritative Request** | `.agents/ORIGINAL_REQUEST.md` | Requirements R1–R10, task definitions T00–T12, acceptance criteria. |
| **Audit Project Contract** | `.agents/orchestrator_audit/PROJECT.md` | Dual-track architecture, interface contracts, data schemas, code layout. |
| **Target Paper PDF** | `Reproducing paper /Adv Funct Materials - 2026 - Chen...pdf` | Section 5 experimental methods, Table 1 target properties, Fig 2–5 claims. |
| **Supplementary Info** | `Reproducing paper /adfm75639-sup-0001-suppmat.docx` | Table S1 bounds, Table S2 prior data, Table S3–S5 optima & rheology, Table S6 models. |
| **Upstream Raw Data** | `Hydrogel_BO/hydrogel_bo.csv` | 211 rows: 43 prior rows (index 2 missing) + 168 experimental conditions. |
| **Upstream Analysis Data** | `Hydrogel_BO/hydrogel_analysis.csv` | 168 rows: 100 closed-loop (batches 1–25) + 68 Pareto mapping (batches 26–42). |
| **Master Seed Manifest** | `env/seeds.yaml` | Master seed `20260919` and deterministic SHA-256 derivation formula. |
| **Card JSON Schema** | `src/utils/card_schema.json` | Draft-07 Section 9 audit task card specification. |

---

## 3. 4-Tier Test Architecture

The test suite is organized into a four-tier hierarchy following the Systematic Test Methodology:

```text
tests/
├── conftest.py                    # Shared fixtures, paths, schema loaders, oracle reference loaders
├── test_canonical_data.py         # Tier 1 & Tier 2: Canonical parquet schemas, counts, bounds, paper contradictions
├── test_cards_schema.py           # Tier 3: Draft-07 schema compliance, required fields, enum validation
├── test_hashes.py                 # Tier 3: SHA-256 manifest integrity & card provenance cross-referencing
├── test_seeds.py                  # Tier 4: Deterministic seed derivation formula & SeedSequence independence
└── test_e2e_reproducibility.py     # Tier 4: End-to-end pipeline invariants, hypervolume, convex hull volume
```

### Tier 1: Feature & Data Contract Coverage
Validates the structure, types, non-null guarantees, and domain bounds of all canonical data tables produced by Gate 1 (Task T00):
- **`points.parquet` (168 rows)**:
  - Columns: `point_id` (0..167), `source_index` (44..211), `exposure_time` ([0, 600] s), `gelma_l` ([0, 6] wt%), `gelma_h` ([0, 6] wt%), `acnc` ([0, 6] wt%), `lap` ([0, 0.2] wt%), `modulus` ($E > 0$ kPa), `stress_relaxation` ($0 \le \sigma_{100} \le 100$ %), `campaign`, `batch_id` (1..42), `batch_type` (`closed_loop` or `pareto_mapping`).
  - Strict assertions: No nulls, exact 168 rows, monotonic indices, valid campaign enum values.
- **`prior.parquet` (43 rows)**:
  - Columns: `prior_id` (0..42), `source_index` (0..43, missing 2), 5 formulation parameters, `imputed_modulus` ($E > 0$), `stress_relaxation`.
  - Strict assertions: No nulls, exact 43 rows, index 2 absent.
- **`optima.parquet` (7 rows)**:
  - Columns: `target_name`, `candidate_id`, `source_index`, 5 inputs, `measured_modulus`, `measured_stress_relaxation`, `target_modulus`, `target_stress_relaxation`, `tolerance_modulus`, `tolerance_stress_relaxation`, `window_pass`.
  - Strict assertions: Exactly 7 optimal candidates (I4-2, I4-4, I6-4, I7-3, I3-3, I4-1, I1-4), correct target windows.
- **`paper_claims.yaml`**:
  - Validates presence of keys: `section5_constants`, `table1_targets`, `table_s6_metrics`, `discretization_bounds`.

### Tier 2: Boundary & Corner Cases (Adversarial Stress)
Validates mathematical anomalies, paper contradictions, and domain boundaries:
1. **Missing Index 2 in Prior**: Verifies `source_index` contains `0, 1, 3, 4, ... 43` and does NOT contain `2`. Verifies row count is 43, refuting the paper's text claim of 44 points.
2. **LAP Zero/Low-Variance Artifact**: Asserts that exactly 35 of 43 prior rows (81.4%) have $\text{LAP} = 0.10$ wt.%, proving severe variance collapse that causes ARD lengthscale inflation.
3. **Partition Accounting Invariant**: Asserts $100\text{ (closed-loop)} + 68\text{ (Pareto)} = 168$, and $43\text{ (prior)} + 168 = 211\text{ (total)}$. Verifies exact numerical identity with upstream `hydrogel_analysis.csv` and `hydrogel_bo.csv`.
4. **Target Tolerance Window Boundaries**:
   - Soft Tumor I4-1 modulus ($1.428$ kPa) vs upper tolerance ($1.30 \times 1.10 = 1.430$ kPa) — asserts candidate passes within a 2 Pa margin.
   - Scarred Skin paper claim ($22.5 \pm 1.5\%$) vs Table 1 window ($[18.0, 22.0]\%$) — asserts the paper's text claim mathematically FAILS the window ($22.5 > 22.0$), while Table S3 raw data ($21.0\%, 20.5\%$) PASSES.
   - Stiff Tumor Modulus Discrepancy — asserts text claim ($16.8$ kPa) vs raw data ($15.923$ kPa) discrepancy is $\approx 0.88$ kPa ($> 5\%$).
5. **Discretization Arithmetic Contradiction**:
   - Asserts mathematical Cartesian product of Table S1 levels ($7 \times 7 \times 7 \times 5 \times 11 = 18,865$) contradicts paper's claim of 6,230.
   - Asserts prime factorization of 6,230 includes 89, which cannot arise from bounds without unstated constraints.
6. **Duplicate Hit Redundancy**: Asserts normalized Euclidean distance between Healthy Skin I4-2 and I4-4 is $< 0.05$ ($0.0448$), demonstrating near-duplicate sampling within iteration 4.

### Tier 3: Cross-Feature Interactions & Schema Checks
Validates contract compliance across modules and files:
1. **Draft-07 Card Schema Compliance (`test_cards_schema.py`)**:
   - Uses `fastjsonschema` (or `jsonschema`) to validate every `card.json` in `artifacts/cards/` against `src/utils/card_schema.json`.
   - Checks all mandatory top-level keys: `task_id`, `claim_reference`, `verdict`, `preregistration`, `sensitivity_ranges`, `bias_direction`, `provenance_hashes`, `execution_metadata`, `metrics`, `summary`.
   - Validates enum fields (`verdict`: `SUPPORTED`, `WEAKENED`, `CONTRADICTED`, `UNCHECKABLE`; `bias_direction`: `conservative`, `anti-conservative`, `neutral`, `favors_authors`, `penalizes_authors`).
   - Includes adversarial tests with mutated invalid cards (missing keys, invalid regex, bad enums) to verify schema rejecting power.
2. **Cryptographic Manifest Integrity (`test_hashes.py`)**:
   - Reads `artifacts/canonical/manifest.sha256` and computes live SHA-256 digests of all listed canonical files (`points.parquet`, `prior.parquet`, `optima.parquet`, `paper_claims.yaml`).
   - Asserts zero divergence between recorded hex digests and disk files.
   - Scans all `card.json` files and verifies that hashes in `provenance_hashes.inputs`, `outputs`, and `script` match disk reality.

### Tier 4: Real-World Workload Scenarios & Pipeline Invariants
Validates end-to-end execution consistency, mathematical properties, and deterministic reproducibility:
1. **Seed Protocol Integrity (`test_seeds.py`)**:
   - Verifies master seed is `20260919`.
   - Verifies the derivation formula:
     $$\text{seed} = \text{int}\Big(\text{SHA-256}\big(f\text{"}\{master\_seed\}\text{:}\{task\_id\}\text{"}\big)[:8], 16\Big) \pmod{2^{31}-1}$$
   - Confirms exact match for all tasks T00 through T12.
   - Verifies child seed generation using `np.random.SeedSequence(seed).spawn(200)` generates 200 unique, reproducible, positive 32-bit integers.
2. **Reproducible Pipeline Invariants (`test_e2e_reproducibility.py`)**:
   - **Hypervolume 2D Calculation**: Implements reference 2D hypervolume calculation with empirical nadir $(0.093, 8.0)$; asserts hypervolume at point 100 is $\sim 301$, at point 128 is $\sim 362.1$, and at point 168 is $\sim 364.1$, reproducing the paper's 362–364 plateau.
   - **Explored Parameter Volume (5D Convex Hull)**: Calculates convex hull volume of the 168 points scaled to unit hypercube $[0, 1]^5$; asserts volume is within $[2.7\%, 3.0\%]$ (specifically $\approx 2.815\%$), and for the initial 100 points is $< 0.6\%$ ($\approx 0.436\%$).
   - **Feasibility Constraint Filter**: Evaluates the `known_constraint` function on the 168 points; asserts 100% of experimental points satisfy `GelMA_L + GelMA_H >= 3` and the total solids inequality.

---

## 4. Progressive Testability Matrix

| Milestone | Target Scope | Active Test Modules | Expected Result | Gating Criteria |
|---|---|---|---|---|
| **M1** | Canonical Ingestion & Infra | `test_canonical_data.py`, `test_seeds.py`, `test_hashes.py`, `test_cards_schema.py` (T00) | **100% PASS** | Canonical tables created, manifest verified, T00 card valid. |
| **M2** | Surrogate & Imputation (T01, T04) | `test_cards_schema.py` (T01, T04), `test_e2e_reproducibility.py` | **100% PASS** | Prospective RMSE evaluated, ARD lengthscales verified. |
| **M3** | Baselines & CV (T02, T03, T05) | `test_cards_schema.py` (T02a, T02b, T03, T05) | **100% PASS** | 10k permutations complete, twin circularity logged. |
| **M4** | Interpretability & Pareto (T06, T07, T11) | `test_cards_schema.py` (T06, T07a-d, T11), `test_e2e_reproducibility.py` | **100% PASS** | HV plateau verified, 5D hull volume $\approx 2.815\%$. |
| **M5** | Consistency & Synthesis (T08, T09, T10, T12) | `test_cards_schema.py` (T08..T12), all consistency checks | **100% PASS** | Consistency ledger complete, red team defense valid. |
| **M6** | Final Gate Hardening | Full suite + `make verify` (7 gates) | **100% PASS** | `TEST_READY.md` verified, audit integrity sign-off issued. |

---

## 5. Execution Environment & Test Commands

Tests must be executed using the designated virtual environment and system site-packages:

```bash
# Standard test invocation from project root:
cd "/Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit"

PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages:$(PWD)/src:$(PYTHONPATH)" \
MPLCONFIGDIR="$(PWD)/.mplconfig" \
"/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO/venv/bin/python" -m pytest tests/ -v

# Run with coverage report:
PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages:$(PWD)/src:$(PYTHONPATH)" \
MPLCONFIGDIR="$(PWD)/.mplconfig" \
"/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO/venv/bin/python" -m pytest tests/ -v --tb=short

# Run specific test tier:
# Tier 1 & 2 (Canonical Data & Boundaries):
pytest tests/test_canonical_data.py -v
# Tier 3 (Schemas & Hashes):
pytest tests/test_cards_schema.py tests/test_hashes.py -v
# Tier 4 (Seeds & Pipeline Invariants):
pytest tests/test_seeds.py tests/test_e2e_reproducibility.py -v
```

---

## 6. Verification and Readiness Criteria (`TEST_READY.md`)

`TEST_READY.md` is published at the project root once:
1. `TEST_INFRA.md` is authored and committed.
2. All test modules in `tests/` are created with comprehensive non-tautological test cases.
3. The M1 test suite (`test_canonical_data.py`, `test_hashes.py`, `test_seeds.py`, and `test_cards_schema.py` on available cards) executes with 100% PASS status against the ingested canonical artifacts.
4. Downstream milestone tests are structurally integrated with progressive auto-discovery.
