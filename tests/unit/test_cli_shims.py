"""Legacy entry points propagate exit codes; shim modules exist (0.2.2)."""

import ast
import importlib
from pathlib import Path

import pytest

import matrix_mat  # noqa: F401  (production import for test_meta)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mod,target", [
    ("spheroid_if_sweep.sweep", "matrix_mat.if_density.sweep"),
    ("spheroid_if_sweep.comparator_biological", "matrix_mat.if_density.comparator_biological"),
])
def test_missing_if_shims_exist(mod, target):
    with pytest.deprecated_call():
        m = importlib.import_module(mod)
    t = importlib.import_module(target)
    public = [n for n in dir(t) if not n.startswith("_") and callable(getattr(t, n))]
    assert public and all(hasattr(m, n) for n in public)


def _main_call_is_sys_exit(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "exit":
            if node.args and isinstance(node.args[0], ast.Call) and getattr(node.args[0].func, "id", None) == "main":
                return True
    return False


@pytest.mark.parametrize("rel", ["spheroid_if_sweep/__main__.py", "spheroid_pipeline_v2/__main__.py", "run_pipeline.py"])
def test_entry_points_propagate_exit_code(rel):
    assert _main_call_is_sys_exit(REPO_ROOT / rel)


def test_root_run_pipeline_forwards_to_matrix_mat():
    text = (REPO_ROOT / "run_pipeline.py").read_text(encoding="utf-8")
    assert "from matrix_mat.brightfield.run_pipeline import main" in text
    assert "spheroid_pipeline_v2" not in text.split('"""')[-1]
