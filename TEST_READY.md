# Spheroid Pipeline v2 — Test Suite Ready Declaration (`TEST_READY.md`)

## 1. Readiness Status: **READY & VERIFIED**
- **Date**: 2026-08-31
- **Status**: 100% of E2E test suites implemented, verified, and passing.
- **Total Test Cases**: **202 Tests** across 45 test classes.
- **Pass Rate**: **100.0% (202/202 passing, 0 failures, 0 errors)**.
- **Execution Time**: **9.64 seconds** end-to-end.

---

## 2. Test Suite Inventory

| Test Module | Scope | Test Classes | Test Count | Status |
|---|---|---|---|---|
| `spheroid_pipeline_v2/tests/synthetic_generator.py` | Generative brightfield microscopy simulator with physics ground-truth | — | Utility | **VERIFIED** |
| `spheroid_pipeline_v2/tests/test_synthetic_smoke.py` | 3 Ground-truth scenarios (isolated $\pm 10\%$, touching pair, bright blob 100% rejected) | `TestSyntheticSmoke` | 5 | **PASS** |
| `spheroid_pipeline_v2/tests/test_tier1_features.py` | 29 Features from PROJECT.md ($\ge 5$ tests per feature) | 29 Feature Classes | 145 | **PASS** |
| `spheroid_pipeline_v2/tests/test_tier2_boundaries.py` | Boundaries, edge cases, micro-debris, giant masks $>25\%$, border touching, zero denom, NaNs | 8 Boundary Classes | 40 | **PASS** |
| `spheroid_pipeline_v2/tests/test_tier3_pairwise.py` | Cross-feature integration & full data flows | 6 Interaction Classes | 8 | **PASS** |
| `spheroid_pipeline_v2/tests/test_tier4_workloads.py` | Sample 12 stratified gate execution & dataset inventory checks | `TestTier4RealWorldWorkload` | 4 | **PASS** |
| `spheroid_pipeline_v2/tests/run_e2e_tests.py` | Master CLI runner & JSON reporter | — | CLI Runner | **VERIFIED** |

---

## 3. How to Run the Tests

### Master Test Runner (Recommended)
```bash
# Run all 202 tests across all tiers
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --all

# Export execution summary to JSON
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --all --json-output output/test_results.json

# Run specific tiers
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --smoke
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --tier1
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --tier2
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --tier3
./.venv/bin/python spheroid_pipeline_v2/tests/run_e2e_tests.py --tier4
```

### Standard Python Unittest Execution
```bash
# Run individual modules
./.venv/bin/python -m unittest spheroid_pipeline_v2/tests/test_synthetic_smoke.py
./.venv/bin/python -m unittest spheroid_pipeline_v2/tests/test_tier1_features.py
./.venv/bin/python -m unittest spheroid_pipeline_v2/tests/test_tier2_boundaries.py
./.venv/bin/python -m unittest spheroid_pipeline_v2/tests/test_tier3_pairwise.py
./.venv/bin/python -m unittest spheroid_pipeline_v2/tests/test_tier4_workloads.py

# Run all test files via discovery
./.venv/bin/python -m unittest discover -s spheroid_pipeline_v2/tests -p "test_*.py"
```

---

## 4. Discovered Implementation Defects & Advice for Worker Agents
1. **Module Import Path**: Ensure `spheroid_pipeline_v2` is importable by executing scripts with the workspace root in `sys.path`.
2. **Matplotlib Font Cache**: On macOS / sandboxed environments, set `os.environ['MPLCONFIGDIR']` to a temporary directory before importing matplotlib to avoid permission warnings.
3. **Floating Point Precision in Contrast Gating**: In `qc_filter.py`, evaluate the background ring contrast gate as `(mean_ring - mean_int) / max(mean_ring, 1e-4) >= 0.10 - 1e-7` or `mean_int <= (1.0 - 0.10) * mean_ring` to prevent binary floating point rounding false rejections.
4. **Markdown Table Generation**: Avoid relying on `tabulate` (`df.to_markdown()`) since it is not installed in `.venv`. Generate native markdown table strings.
