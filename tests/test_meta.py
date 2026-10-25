import ast
from pathlib import Path
import re
import pytest

TESTS_ROOT = Path(__file__).resolve().parent

def _is_test_file(path: Path) -> bool:
    if (
        path.name.startswith("test_")
        and path.suffix == ".py"
        and path.name not in ("test_meta.py", "test_harness.py")
    ):
        return True
    return False

def test_every_test_module_imports_production_code():
    """Verify that all test modules import real production packages, not just mock helpers."""
    prod_packages = ("spheroid_pipeline_v2", "spheroid_if_sweep", "matrix_mat")
    test_files = [p for p in TESTS_ROOT.rglob("*.py") if _is_test_file(p)]
    assert test_files, "No test files discovered under tests/"

    failing_modules = []
    for test_file in test_files:
        content = test_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(test_file))

        imports_prod = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(pkg) for pkg in prod_packages):
                        imports_prod = True
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(pkg) for pkg in prod_packages):
                    imports_prod = True
                    break
            if imports_prod:
                break

        if not imports_prod:
            failing_modules.append(str(test_file.relative_to(TESTS_ROOT)))

    assert not failing_modules, (
        f"The following test modules do not import production code: {failing_modules}"
    )


def test_no_tautological_literal_assertions():
    """Checks for tautological assertions like assertEqual(val, val) on literals."""
    test_files = [p for p in TESTS_ROOT.rglob("*.py") if _is_test_file(p)]
    # Match patterns like assertEqual("foo", "foo") or assert 1 == 1
    literal_assert_pattern = re.compile(
        r'assertEqual\(\s*(["\'][^"\']+["\']|\d+)\s*,\s*\1\s*\)'
    )

    violations = []
    for test_file in test_files:
        content = test_file.read_text(encoding="utf-8")
        for match in literal_assert_pattern.finditer(content):
            violations.append((test_file.name, match.group(0)))

    assert not violations, f"Tautological literal assertions found: {violations}"
