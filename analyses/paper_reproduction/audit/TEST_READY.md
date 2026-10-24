# Test Readiness Publication (`TEST_READY.md`)
**Project**: Multi-Agent Reproduction & Adversarial Audit of Chen et al., *Adv. Funct. Mater.* 2026, 36:e75639  
**Audit Directory**: `/Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit`  
**Test Suite Status**: **READY (100% PASS, 69/69 Tests)**  
**Gate 1 Ingestion & Infrastructure Status**: **VERIFIED & SIGNED OFF**  
**Date & Timestamp**: 2026-09-19T09:01:00Z  
**Master Seed**: `20260919` (`env/seeds.yaml`)  
**Test Writer Agent**: `test_writer_e2e_1`  

---

## 1. Executive Summary

The independent E2E Testing Track has initialized and verified the full test infrastructure for the reproduction and adversarial audit of Chen et al. 2026. The test suite enforces a strict 4-tier testing hierarchy that validates data contracts, mathematical properties, cryptographic manifests, Draft-07 JSON schemas, and pipeline reproducibility without dependence on implementation internals.

The test suite was executed against canonical data ingested by Task T00 and passed all 69 test cases in 0.38 seconds.

---

## 2. Test Suite Inventory

| Test Module | Tier | Scope & Features Tested | Test Count | Status |
|---|---|---|---|---|
| `tests/test_canonical_data.py` | Tier 1 & 2 | `points.parquet` (168 rows, schema, bounds, campaign partitions, numerical identity), `prior.parquet` (43 rows, index 2 absent, LAP variance collapse), `optima.parquet` (7 rows, target windows, boundary cases), `paper_claims.yaml` (discretization arithmetic contradiction, constants), and physical constraint checks. | 20 | **PASS** |
| `tests/test_cards_schema.py` | Tier 3 | Draft-07 `card_schema.json` compilation via `fastjsonschema`, Section 9 compliance of generated cards, directory-wide validation, and 31 parameterized adversarial rejection tests (missing fields, invalid regex, bad enums, malformed SHA-256). | 36 | **PASS** |
| `tests/test_hashes.py` | Tier 3 | `manifest.sha256` integrity, full coverage of all canonical tables, independent re-hashing, and cross-validation of task card provenance hashes against files on disk. | 4 | **PASS** |
| `tests/test_seeds.py` | Tier 4 | `env/seeds.yaml` integrity, deterministic SHA-256 seed derivation formula for all 17 tasks (T00–T12), 31-bit positive int bounds, pairwise uniqueness, and multi-seed child sequence spawning via `np.random.SeedSequence`. | 5 | **PASS** |
| `tests/test_e2e_reproducibility.py` | Tier 4 | 2D Hypervolume plateau reproduction at 362–364, 5D Convex Hull explored parameter volume (~2.815%), physical feasibility Monte Carlo volume (~45.5%), and upstream raw CSV immutability. | 4 | **PASS** |
| **Total** | | **Comprehensive 4-Tier Test Suite** | **69** | **100% PASS** |

---

## 3. Verification Execution & Results

### Execution Command:
```bash
cd "/Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit"
make verify
```

Equivalent direct pytest invocation:
```bash
PYTHONPATH="/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages:$(PWD)/src:$(PYTHONPATH)" \
MPLCONFIGDIR="$(PWD)/.mplconfig" \
"/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO/venv/bin/python" -m pytest -p no:fast_array_utils tests/ -v
```

### Verification Output:
```text
=== Gate 1.1: Checking Environment & Dependencies ===
✅ All core packages verified!
=== Gate 2: Verifying Canonical Data SHA-256 Manifest ===
✅ Manifest artifacts/canonical/manifest.sha256 verified successfully (4 files checked).
=== Gate 3: Validating Section 9 Card Schemas ===
✅ All 1 task card(s) validated successfully in artifacts/cards
=== Gate 4: Verifying Card Provenance SHA-256 Hashes ===
✅ All 1 task card(s) validated successfully (with hash verification) in artifacts/cards
=== Gate 5: Verifying Deterministic Seed Derivation ===
✅ Seeds verified successfully against derivation formula (env/seeds.yaml)
=== Gate 6: Running Pytest Test Suite ===
tests/test_canonical_data.py::TestCanonicalDataTier1::test_canonical_files_exist PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_points_parquet_schema_and_row_count PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_points_parquet_bounds_and_physical_ranges PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_points_parquet_numerical_identity_with_upstream_oracle PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_points_campaign_and_batch_accounting PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_prior_parquet_schema_and_row_count PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_prior_parquet_matches_upstream_oracle PASSED
tests/test_canonical_data.py::TestCanonicalDataTier1::test_optima_parquet_schema_and_candidates PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_prior_missing_index_2_corner_case PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_prior_lap_zero_variance_artifact PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_prior_exposure_time_mode_clustering PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_exact_dataset_accounting_invariant PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_soft_tumor_i4_1_modulus_upper_tolerance_boundary PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_scarred_skin_target_window_paper_text_contradiction PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_stiff_tumor_modulus_discrepancy PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_duplicate_optima_euclidean_distance PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_paper_claims_yaml_content_and_structure PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_paper_claims_yaml_discretization_arithmetic_contradiction PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_all_168_points_satisfy_physical_feasibility_constraint PASSED
tests/test_canonical_data.py::TestCanonicalDataTier2::test_point_90_csv_rounding_boundary_overflow PASSED
tests/test_cards_schema.py::test_card_schema_compilation PASSED
tests/test_cards_schema.py::test_t00_card_validation PASSED
tests/test_cards_schema.py::test_cards_directory_validation PASSED
tests/test_cards_schema.py::test_valid_mock_card_passes_validator PASSED
tests/test_cards_schema.py::test_adversarial_missing_required_fields[...] (10 tests) PASSED
tests/test_cards_schema.py::test_adversarial_invalid_task_id_pattern[...] (7 tests) PASSED
tests/test_cards_schema.py::test_adversarial_invalid_verdict_enum[...] (6 tests) PASSED
tests/test_cards_schema.py::test_adversarial_invalid_bias_direction_enum[...] (4 tests) PASSED
tests/test_cards_schema.py::test_adversarial_malformed_sha256[...] (5 tests) PASSED
tests/test_e2e_reproducibility.py::TestE2EReproducibility::test_hypervolume_2d_plateau_reproduction PASSED
tests/test_e2e_reproducibility.py::TestE2EReproducibility::test_explored_parameter_volume_5d_convex_hull PASSED
tests/test_e2e_reproducibility.py::TestE2EReproducibility::test_physical_feasibility_space_volume_monte_carlo PASSED
tests/test_e2e_reproducibility.py::TestE2EReproducibility::test_upstream_oracle_csv_digests PASSED
tests/test_hashes.py::test_manifest_verification PASSED
tests/test_hashes.py::test_manifest_file_hashes PASSED
tests/test_hashes.py::test_manifest_covers_all_required_canonical_tables PASSED
tests/test_hashes.py::test_card_provenance_hashes_match_disk PASSED
tests/test_seeds.py::test_seeds_config_verification PASSED
tests/test_seeds.py::test_seed_derivation_deterministic PASSED
tests/test_seeds.py::test_all_17_task_seeds_match_oracle_derivation PASSED
tests/test_seeds.py::test_task_seeds_uniqueness PASSED
tests/test_seeds.py::test_multiseed_child_sequence_spawning PASSED

============================== 69 passed in 0.38s ==============================
======================================================================
✅ ALL GATE 1 INGESTION & INFRASTRUCTURE CHECKS PASSED SUCCESSFULLY
======================================================================
```

---

## 4. Key Adversarial Invariants Verified

1. **Prior Accounting Contradiction**: Index 2 is absent in `prior.parquet`. Total prior rows = exactly 43 (refuting 44 claimed in Section 2.1).
2. **Prior LAP Variance Collapse**: 35 of 43 prior rows (81.4%) have $\text{LAP} = 0.10$ wt.%, confirming severe feature degeneracy.
3. **Partition Accounting**: Exactly 100 closed-loop points + 68 Pareto mapping points = 168 points.
4. **Target Tolerance Boundaries**:
   - Soft Tumor I4-1 modulus ($1.428$ kPa) passes within 2 Pa of upper bound ($1.430$ kPa).
   - Scarred Skin text claim ($\sigma_{100} = 22.5\%$) fails Table 1 window ($[18.0, 22.0]\%$), whereas raw data ($21.0\%, 20.5\%$) passes.
   - Stiff Tumor modulus text claim ($16.8$ kPa) exhibits an unexplained $0.877$ kPa discrepancy vs raw data ($15.923$ kPa).
5. **Discretization Arithmetic Error**: Cartesian product of Table S1 levels is $18,865$, mathematically contradicting the claimed $6,230$.
6. **Near-Duplicate Optima**: Healthy Skin hits I4-2 and I4-4 have normalized Euclidean distance $0.0448 < 0.05$.
7. **Hypervolume Plateau**: Utopian 1 accounts for $96.9\%$ of total hypervolume gain beyond closed-loop, confirming the 362–364 plateau.
8. **Explored Parameter Volume**: Convex hull of 168 points occupies only $2.815\%$ of the nominal 5D box ($0.436\%$ for closed-loop).

---

## 5. Progressive Testing for Downstream Milestones

The test suite is built to support progressive testability as implementation milestones advance:
- **Milestone M2 (T01, T04)**: When surrogate and imputation cards/artifacts are created in `artifacts/cards/T01` and `artifacts/cards/T04`, `test_cards_schema.py` and `test_hashes.py` will auto-discover and validate them.
- **Milestone M3 (T02, T03, T05)**: Permutation baselines and CV estimands will automatically be verified against seed protocol and Section 9 schema.
- **Milestones M4–M6**: Pareto, interpretability, consistency, and final red-team syntheses will undergo automated gate validation via `make verify`.

---

## 6. Readiness Sign-Off

The E2E Testing Track officially issues **TEST_READY** sign-off for Milestone 1. The test runner, schema validators, hash checkers, seed generators, and 69 test cases are fully verified and locked.
