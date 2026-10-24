import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# Folders that pipeline runs used to write into; tests must never add to or modify them.
GUARDED_DIRS = ("masks", "output", "runs", "review", "cache")
GUARDED_FILES = ("DECISIONS.md", "decisions.jsonl")

src_path = str(REPO_ROOT / "src")
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


def _repo_snapshot():
    """Top-level entries plus (path, mtime) of every file under the guarded folders."""
    top = {p.name for p in REPO_ROOT.iterdir()}
    files = {}
    for d in GUARDED_DIRS:
        root = REPO_ROOT / d
        if root.is_dir():
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    fp = Path(dirpath) / fn
                    try:
                        files[str(fp.relative_to(REPO_ROOT))] = fp.stat().st_mtime_ns
                    except OSError:
                        pass
    for f in GUARDED_FILES:
        fp = REPO_ROOT / f
        if fp.exists():
            files[f] = fp.stat().st_mtime_ns
    return top, files


@pytest.fixture(scope="session", autouse=True)
def ensure_clean_repo_state():
    """Fails the session if any test created or modified pipeline outputs inside the repository."""
    top_before, files_before = _repo_snapshot()
    yield
    top_after, files_after = _repo_snapshot()
    # Build artefacts from an editable install are not test output.
    new_top = {n for n in top_after - top_before if not n.endswith(".egg-info") and n not in (".pytest_cache", ".coverage")}
    changed = sorted(k for k, v in files_after.items() if files_before.get(k) != v)
    assert not new_top and not changed, (
        f"Tests wrote into the repository: new top-level entries={sorted(new_top)}, "
        f"new/modified files={changed[:20]}"
    )
