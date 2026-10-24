import ast
from pathlib import Path
import pytest
import warnings


def test_legacy_imports_warn():
    """Importing legacy packages must emit DeprecationWarning and re-export symbols."""
    import sys
    import importlib

    sys.modules.pop("spheroid_pipeline_v2", None)
    with pytest.deprecated_call():
        import spheroid_pipeline_v2
        assert hasattr(spheroid_pipeline_v2, "run_spheroid_pipeline")

    sys.modules.pop("spheroid_if_sweep", None)
    with pytest.deprecated_call():
        import spheroid_if_sweep
        assert hasattr(spheroid_if_sweep, "run_sweep") or hasattr(spheroid_if_sweep, "cli")


def test_import_matrix_mat_emits_no_deprecation_warnings():
    """Importing matrix_mat must NOT emit DeprecationWarning."""
    import sys
    for mod in list(sys.modules.keys()):
        if mod.startswith("matrix_mat"):
            sys.modules.pop(mod, None)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        import matrix_mat
        from matrix_mat import core, brightfield, if_density, reporting, cli
        dep_warnings = [w for w in record if issubclass(w.category, DeprecationWarning)]
        assert not dep_warnings, f"Unexpected deprecation warnings on matrix_mat import: {dep_warnings}"


def test_matrix_mat_imports_succeed():
    """Core, brightfield, if_density, and reporting subpackages can be imported."""
    from matrix_mat.core import units, scale, naming, stats, cache, provenance, paths, repro
    assert callable(units.compute_spherical_volume)
    assert hasattr(scale, "ScaleExtractor")
    assert callable(naming.parse_image_filename)
    assert callable(stats.hierarchical_bootstrap_ci)
    assert hasattr(cache, "MaskCache")
    assert callable(provenance.export_complete_manifest)
    assert callable(paths.resolve_data_root)
    assert callable(repro.set_determinism)

    from matrix_mat.brightfield import run_pipeline, segment, qc_filter, measure, pair
    assert callable(run_pipeline.run_spheroid_pipeline)

    from matrix_mat.if_density import sweep, preprocess, postprocess, features, qc
    assert callable(sweep.run_sweep)

    from matrix_mat.cli import main
    assert callable(main)


def test_core_has_no_downstream_imports():
    """Core subpackage must be pure and light: imports nothing from brightfield, if_density, reporting."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    core_dir = repo_root / "src" / "matrix_mat" / "core"
    assert core_dir.exists(), f"Directory not found: {core_dir}"

    forbidden = ("brightfield", "if_density", "reporting", "spheroid_pipeline_v2", "spheroid_if_sweep")
    violations = []

    for py_file in core_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(f in alias.name for f in forbidden):
                        violations.append(f"{py_file.name}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(f in node.module for f in forbidden):
                    violations.append(f"{py_file.name}: from {node.module} import ...")

    assert not violations, f"Forbidden downstream imports found in core:\n" + "\n".join(violations)


def test_if_density_does_not_import_brightfield():
    """if_density subpackage must not import brightfield or legacy spheroid_pipeline_v2."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    if_dir = repo_root / "src" / "matrix_mat" / "if_density"
    assert if_dir.exists(), f"Directory not found: {if_dir}"

    forbidden = ("brightfield", "spheroid_pipeline_v2")
    violations = []

    for py_file in if_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(f in alias.name for f in forbidden):
                        violations.append(f"{py_file.name}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(f in node.module for f in forbidden):
                    violations.append(f"{py_file.name}: from {node.module} import ...")

    assert not violations, f"Forbidden brightfield imports found in if_density:\n" + "\n".join(violations)


def test_package_does_not_import_tests():
    """Production package src/matrix_mat must NEVER import from tests.*."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pkg_dir = repo_root / "src" / "matrix_mat"
    assert pkg_dir.exists(), f"Directory not found: {pkg_dir}"

    violations = []
    for py_file in pkg_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("tests") or "synthetic_generator" in alias.name:
                        violations.append(f"{py_file.name}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and (node.module.startswith("tests") or "synthetic_generator" in node.module):
                    violations.append(f"{py_file.name}: from {node.module} import ...")

    assert not violations, f"Forbidden test imports in production code:\n" + "\n".join(violations)
