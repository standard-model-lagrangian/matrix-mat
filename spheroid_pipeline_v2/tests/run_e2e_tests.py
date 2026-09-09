#!/usr/bin/env python3
"""Master E2E Test Suite Runner for Spheroid Volume Pipeline v2.

Executes all test tiers (Smoke, Tier 1, Tier 2, Tier 3, Tier 4), aggregates metrics,
and produces structured summary reports in console and optional JSON.

Usage:
    python spheroid_pipeline_v2/tests/run_e2e_tests.py [--all] [--smoke] [--tier1] [--tier2] [--tier3] [--tier4] [-v] [--json-output report.json]
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

# Avoid matplotlib cache issues
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

from spheroid_pipeline_v2.tests.test_synthetic_smoke import TestSyntheticSmoke
from spheroid_pipeline_v2.tests.test_tier1_features import (
    TestFeature01_ScaleExtraction,
    TestFeature02_PreprocessingAndDownsampling,
    TestFeature03_CellposeSAMBackend,
    TestFeature04_CellposeCyto3Backend,
    TestFeature05_ClassicalWatershedBackend,
    TestFeature06_HardwareAccelerationFallback,
    TestFeature07_MaskDiskCaching,
    TestFeature08_DecisionLogging,
    TestFeature09_OverlapResolution,
    TestFeature10_PhysicalSizeGate,
    TestFeature11_BorderMarginGate,
    TestFeature12_BackgroundRingContrastGate,
    TestFeature13_SingleMaskMaxAreaGate,
    TestFeature14_MorphologicalVolumetricSuite,
    TestFeature15_QCFlaggingProtocol,
    TestFeature16_MutualNearestCentroidPairing,
    TestFeature17_DenominatorGate,
    TestFeature18_TrajectoryFoldChangeGate,
    TestFeature19_ConditionLevelStatistics,
    TestFeature20_PermutationHypothesisTesting,
    TestFeature21_MultilevelExclusionAuditing,
    TestFeature22_SyntheticSmokeHarness,
    TestFeature23_ProgrammaticSanityGates,
    TestFeature24_FullDatasetBatchRunner,
    TestFeature25_SupervisionOverlays,
    TestFeature26_ContactSheets,
    TestFeature27_PublicationFigures,
    TestFeature28_ExecutiveReports,
    TestFeature29_ReviewFolderValidationManager,
)
from spheroid_pipeline_v2.tests.test_tier2_boundaries import (
    TestBoundaryBorderTouching,
    TestBoundaryEmptyInputs,
    TestBoundaryExtremeCounts,
    TestBoundaryExtremeFoldChanges,
    TestBoundaryGiantMasks,
    TestBoundaryMicroObjects,
    TestBoundaryNaNAndInfResilience,
    TestBoundaryZeroDenominatorProtection,
)
from spheroid_pipeline_v2.tests.test_tier3_pairwise import (
    TestPairwiseBatchSupervisionAndReports,
    TestPairwiseCachingAndContrastQC,
    TestPairwiseHierarchyFallbackAndAudit,
    TestPairwisePairingDenominatorAndBootstrapStats,
    TestPairwiseReviewValidationAndOverride,
    TestPairwiseScaleAndDownsampling,
)
from spheroid_pipeline_v2.tests.test_tier4_workloads import (
    TestTier4RealWorldWorkload,
)
from spheroid_pipeline_v2.tests.test_m2_qc_measure import (
    TestOverlapResolution,
    TestPhysicalSizeGate,
    TestBorderMarginGate,
    TestBackgroundRingContrastGate,
    TestMorphologicalQCAndVolumetrics,
    TestPlausibilityAndAreaGate,
    TestObjectsCSVSchemaAndPersistence,
    TestEndToEndFilterAndMeasureIntegration,
)
from spheroid_pipeline_v2.tests.test_m3_pair_stats import (
    TestHungarianPairing,
    TestTrajectoryAndDenominatorGating,
    TestPopulationStatisticsAndHygiene,
    TestBootstrapAndPermutationTesting,
    TestExclusionAuditingAndCSVPersistence,
)
from spheroid_pipeline_v2.tests.test_m4_integration import (
    TestM4Visualization,
    TestM4ReviewManager,
    TestM4Reporting,
    TestM4PipelineExecution,
)


# ANSI formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def build_suite(tier_name: str) -> unittest.TestSuite:
  """Build a test suite for a specific tier."""
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()

  if tier_name in ["smoke", "all"]:
    suite.addTests(loader.loadTestsFromTestCase(TestSyntheticSmoke))

  if tier_name in ["tier1", "all"]:
    t1_classes = [
        TestFeature01_ScaleExtraction,
        TestFeature02_PreprocessingAndDownsampling,
        TestFeature03_CellposeSAMBackend,
        TestFeature04_CellposeCyto3Backend,
        TestFeature05_ClassicalWatershedBackend,
        TestFeature06_HardwareAccelerationFallback,
        TestFeature07_MaskDiskCaching,
        TestFeature08_DecisionLogging,
        TestFeature09_OverlapResolution,
        TestFeature10_PhysicalSizeGate,
        TestFeature11_BorderMarginGate,
        TestFeature12_BackgroundRingContrastGate,
        TestFeature13_SingleMaskMaxAreaGate,
        TestFeature14_MorphologicalVolumetricSuite,
        TestFeature15_QCFlaggingProtocol,
        TestFeature16_MutualNearestCentroidPairing,
        TestFeature17_DenominatorGate,
        TestFeature18_TrajectoryFoldChangeGate,
        TestFeature19_ConditionLevelStatistics,
        TestFeature20_PermutationHypothesisTesting,
        TestFeature21_MultilevelExclusionAuditing,
        TestFeature22_SyntheticSmokeHarness,
        TestFeature23_ProgrammaticSanityGates,
        TestFeature24_FullDatasetBatchRunner,
        TestFeature25_SupervisionOverlays,
        TestFeature26_ContactSheets,
        TestFeature27_PublicationFigures,
        TestFeature28_ExecutiveReports,
        TestFeature29_ReviewFolderValidationManager,
    ]
    for cls in t1_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  if tier_name in ["tier2", "all"]:
    t2_classes = [
        TestBoundaryEmptyInputs,
        TestBoundaryGiantMasks,
        TestBoundaryMicroObjects,
        TestBoundaryBorderTouching,
        TestBoundaryExtremeFoldChanges,
        TestBoundaryZeroDenominatorProtection,
        TestBoundaryNaNAndInfResilience,
        TestBoundaryExtremeCounts,
    ]
    for cls in t2_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  if tier_name in ["tier3", "all"]:
    t3_classes = [
        TestPairwiseScaleAndDownsampling,
        TestPairwiseCachingAndContrastQC,
        TestPairwiseHierarchyFallbackAndAudit,
        TestPairwisePairingDenominatorAndBootstrapStats,
        TestPairwiseReviewValidationAndOverride,
        TestPairwiseBatchSupervisionAndReports,
    ]
    for cls in t3_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  if tier_name in ["tier4", "all"]:
    suite.addTests(loader.loadTestsFromTestCase(TestTier4RealWorldWorkload))

  if tier_name in ["m2", "all"]:
    m2_classes = [
        TestOverlapResolution,
        TestPhysicalSizeGate,
        TestBorderMarginGate,
        TestBackgroundRingContrastGate,
        TestMorphologicalQCAndVolumetrics,
        TestPlausibilityAndAreaGate,
        TestObjectsCSVSchemaAndPersistence,
        TestEndToEndFilterAndMeasureIntegration,
    ]
    for cls in m2_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  if tier_name in ["m3", "all"]:
    m3_classes = [
        TestHungarianPairing,
        TestTrajectoryAndDenominatorGating,
        TestPopulationStatisticsAndHygiene,
        TestBootstrapAndPermutationTesting,
        TestExclusionAuditingAndCSVPersistence,
    ]
    for cls in m3_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  if tier_name in ["m4", "all"]:
    m4_classes = [
        TestM4Visualization,
        TestM4ReviewManager,
        TestM4Reporting,
        TestM4PipelineExecution,
    ]
    for cls in m4_classes:
      suite.addTests(loader.loadTestsFromTestCase(cls))

  return suite



def run_tier(
    tier_key: str, tier_label: str, verbose: bool
) -> Dict[str, Any]:
  """Run a specific tier and record detailed outcome."""
  suite = build_suite(tier_key)
  runner = unittest.TextTestRunner(
      verbosity=2 if verbose else 1,
      stream=open(os.devnull, "w") if not verbose else sys.stdout,
  )

  start_t = time.time()
  result = runner.run(suite)
  elapsed = time.time() - start_t

  total = result.testsRun
  failed = len(result.failures)
  errors = len(result.errors)
  passed = total - failed - errors
  status = "PASS" if (failed == 0 and errors == 0) else "FAIL"

  return {
      "tier_key": tier_key,
      "tier_label": tier_label,
      "total": total,
      "passed": passed,
      "failed": failed,
      "errors": errors,
      "elapsed_sec": elapsed,
      "status": status,
      "failure_details": [str(f[1]) for f in result.failures],
      "error_details": [str(e[1]) for e in result.errors],
  }


def main():
  parser = argparse.ArgumentParser(
      description="Run Spheroid Pipeline v2 E2E Test Suite"
  )
  parser.add_argument(
      "--smoke", action="store_true", help="Run Synthetic Smoke Test"
  )
  parser.add_argument(
      "--tier1", action="store_true", help="Run Tier 1 Feature Tests (29 feats)"
  )
  parser.add_argument(
      "--tier2", action="store_true", help="Run Tier 2 Boundary Tests"
  )
  parser.add_argument(
      "--tier3", action="store_true", help="Run Tier 3 Pairwise Tests"
  )
  parser.add_argument(
      "--tier4", action="store_true", help="Run Tier 4 Workload Sanity Gates"
  )
  parser.add_argument(
      "--m2", action="store_true", help="Run Milestone 2 QC & Measurement Suite"
  )
  parser.add_argument(
      "--m3", action="store_true", help="Run Milestone 3 Pairing & Stats Suite"
  )
  parser.add_argument(
      "--m4", action="store_true", help="Run Milestone 4 Integration & Artifacts Suite"
  )
  parser.add_argument(
      "--all",
      action="store_true",
      help="Run All Tiers (Default if none specified)",
  )
  parser.add_argument(
      "-v", "--verbose", action="store_true", help="Verbose test execution"
  )
  parser.add_argument(
      "--json-output", type=str, default=None, help="Path to save JSON summary"
  )

  args = parser.parse_args()

  # Default to all if no specific tier requested
  if not (
      args.smoke or args.tier1 or args.tier2 or args.tier3 or args.tier4 or args.m2 or args.m3 or args.m4
  ):
    args.all = True

  tiers_to_run = []
  if args.all:
    tiers_to_run = [
        ("smoke", "Synthetic Ground-Truth Smoke Tests"),
        ("tier1", "Tier 1: Feature Inventory (29 Features)"),
        ("tier2", "Tier 2: Boundary & Numeric Edge Cases"),
        ("tier3", "Tier 3: Pairwise Cross-Feature Interactions"),
        ("tier4", "Tier 4: Real Workload & Sample 12 Sanity Gates"),
        ("m2", "Milestone 2: QC Filtering & Measurement Suite"),
        ("m3", "Milestone 3: Pairing & Statistical Hygiene Suite"),
        ("m4", "Milestone 4: Integration, Verification & Supervision"),
    ]
  else:
    if args.smoke:
      tiers_to_run.append(("smoke", "Synthetic Ground-Truth Smoke Tests"))
    if args.tier1:
      tiers_to_run.append(("tier1", "Tier 1: Feature Inventory (29 Features)"))
    if args.tier2:
      tiers_to_run.append(("tier2", "Tier 2: Boundary & Numeric Edge Cases"))
    if args.tier3:
      tiers_to_run.append(
          ("tier3", "Tier 3: Pairwise Cross-Feature Interactions")
      )
    if args.tier4:
      tiers_to_run.append(
          ("tier4", "Tier 4: Real Workload & Sample 12 Sanity Gates")
      )
    if args.m2:
      tiers_to_run.append(
          ("m2", "Milestone 2: QC Filtering & Measurement Suite")
      )
    if args.m3:
      tiers_to_run.append(
          ("m3", "Milestone 3: Pairing & Statistical Hygiene Suite")
      )
    if args.m4:
      tiers_to_run.append(
          ("m4", "Milestone 4: Integration, Verification & Supervision")
      )


  print(f"{BOLD}{CYAN}{'='*80}{RESET}")
  print(f"{BOLD}{CYAN}   SPHEROID VOLUME PIPELINE v2 — E2E TEST SUITE RUNNER{RESET}")
  print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")

  results = []
  grand_total = 0
  grand_passed = 0
  grand_failed = 0
  grand_errors = 0
  total_start = time.time()

  for tier_key, tier_label in tiers_to_run:
    print(f"[*] Running {BOLD}{tier_label}{RESET}...", end="", flush=True)
    res = run_tier(tier_key, tier_label, args.verbose)
    results.append(res)

    grand_total += res["total"]
    grand_passed += res["passed"]
    grand_failed += res["failed"]
    grand_errors += res["errors"]

    if res["status"] == "PASS":
      print(
          f" {GREEN}[PASS]{RESET} ({res['passed']}/{res['total']} passed in"
          f" {res['elapsed_sec']:.2f}s)"
      )
    else:
      print(
          f" {RED}[FAIL]{RESET} ({res['failed']} failed, {res['errors']} errors"
          f" in {res['elapsed_sec']:.2f}s)"
      )

  total_elapsed = time.time() - total_start

  # Summary Table
  print(f"\n{BOLD}{'='*80}{RESET}")
  print(f"{BOLD}                        TEST EXECUTION SUMMARY TABLE{RESET}")
  print(f"{BOLD}{'='*80}{RESET}")
  print(
      f"{'Tier / Test Suite':<50} | {'Total':<6} | {'Pass':<6} | {'Fail':<6} |"
      f" {'Time (s)':<8} | {'Status':<6}"
  )
  print("-" * 88)

  for res in results:
    status_str = (
        f"{GREEN}PASS{RESET}"
        if res["status"] == "PASS"
        else f"{RED}FAIL{RESET}"
    )
    print(
        f"{res['tier_label']:<50} | {res['total']:<6} | {res['passed']:<6} |"
        f" {res['failed'] + res['errors']:<6} | {res['elapsed_sec']:<8.2f} |"
        f" {status_str}"
    )

  print("-" * 88)
  overall_status = (
      f"{GREEN}ALL PASSED{RESET}"
      if (grand_failed == 0 and grand_errors == 0)
      else f"{RED}SOME FAILED{RESET}"
  )
  print(
      f"{BOLD}{'GRAND TOTAL':<50} | {grand_total:<6} | {grand_passed:<6} |"
      f" {grand_failed + grand_errors:<6} | {total_elapsed:<8.2f} |"
      f" {overall_status}{RESET}"
  )
  print(f"{BOLD}{'='*80}{RESET}\n")

  # Print failure/error diagnostics if any
  if grand_failed > 0 or grand_errors > 0:
    print(f"{BOLD}{RED}--- FAILURE / ERROR DIAGNOSTICS ---{RESET}")
    for res in results:
      if res["status"] == "FAIL":
        print(f"\n{BOLD}[{res['tier_label']}]{RESET}")
        for f in res["failure_details"]:
          print(f"{RED}Failure:{RESET}\n{f}")
        for e in res["error_details"]:
          print(f"{RED}Error:{RESET}\n{e}")

  # Export JSON summary if requested
  if args.json_output:
    json_path = Path(args.json_output)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "grand_total": grand_total,
        "grand_passed": grand_passed,
        "grand_failed": grand_failed,
        "grand_errors": grand_errors,
        "total_elapsed_sec": total_elapsed,
        "overall_status": (
            "PASS" if (grand_failed == 0 and grand_errors == 0) else "FAIL"
        ),
        "tiers": results,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"[+] JSON report saved to: {json_path}")

  sys.exit(0 if (grand_failed == 0 and grand_errors == 0) else 1)


if __name__ == "__main__":
  main()
