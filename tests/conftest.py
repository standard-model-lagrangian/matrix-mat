import os
import sys
from pathlib import Path
import pytest

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden regression CSV files",
    )

@pytest.fixture(scope="session", autouse=True)
def isolate_session_paths(tmp_path_factory):
    session_tmp = tmp_path_factory.mktemp("matrix_mat_session")
    output_root = session_tmp / "output"
    cache_root = session_tmp / "cache" / "masks"
    decisions_path = session_tmp / "decisions.log"

    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    old_env = {
        "MATRIX_MAT_OUTPUT_ROOT": os.environ.get("MATRIX_MAT_OUTPUT_ROOT"),
        "MATRIX_MAT_CACHE_DIR": os.environ.get("MATRIX_MAT_CACHE_DIR"),
        "MATRIX_MAT_DECISIONS_PATH": os.environ.get("MATRIX_MAT_DECISIONS_PATH"),
    }

    os.environ["MATRIX_MAT_OUTPUT_ROOT"] = str(output_root)
    os.environ["MATRIX_MAT_CACHE_DIR"] = str(cache_root)
    os.environ["MATRIX_MAT_DECISIONS_PATH"] = str(decisions_path)

    yield session_tmp

    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
