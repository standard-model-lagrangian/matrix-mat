import ast
from pathlib import Path
import pytest
import spheroid_pipeline_v2
import spheroid_if_sweep

REPORTING_MODULES = [
    Path("spheroid_pipeline_v2/visualize.py"),
    Path("spheroid_pipeline_v2/presentation_plots.py"),
    Path("spheroid_pipeline_v2/report.py"),
    Path("spheroid_if_sweep/biological_plots.py"),
    Path("src/matrix_mat/reporting/visualize.py"),
    Path("src/matrix_mat/reporting/presentation_plots.py"),
    Path("src/matrix_mat/reporting/report.py"),
    Path("src/matrix_mat/reporting/biological_plots.py"),
    Path("src/matrix_mat/reporting/summaries.py"),
]

def test_reporting_does_not_import_scipy_stats():
    """Reporting and visualization modules must only display computed stats from tables, not compute them."""
    violations = []
    for mod_path in REPORTING_MODULES:
        if not mod_path.exists():
            continue
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if "scipy.stats" in name.name or name.name == "stats":
                        violations.append((str(mod_path), f"import {name.name}"))
            elif isinstance(node, ast.ImportFrom):
                if node.module and ("scipy.stats" in node.module or node.module == "scipy" and any(n.name == "stats" for n in node.names)):
                    violations.append((str(mod_path), f"from {node.module} import ..."))

    assert not violations, f"Reporting modules found importing scipy.stats: {violations}"
