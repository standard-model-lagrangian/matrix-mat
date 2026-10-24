import subprocess
from pathlib import Path
import sys
import pytest

@pytest.fixture(scope="session", autouse=True)
def ensure_clean_repo_state():
    """Session fixture that verifies no stray root outputs are left behind during tests."""
    repo_root = Path(__file__).resolve().parent.parent

    def get_status():
        res = subprocess.run(
            ["git", "status", "--porcelain", "--ignored"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        return set(res.stdout.splitlines())

    before = get_status()
    yield
    after = get_status()
    diff = after - before
    forbidden_prefixes = ("masks/", "output/", "runs/")
    forbidden_exact = ("DECISIONS.md",)
    violations = []
    for line in diff:
        line = line.strip()
        if not line:
            continue
        status_path = line[2:].strip()
        if any(status_path.startswith(prefix) for prefix in forbidden_prefixes):
            violations.append(status_path)
        if status_path in forbidden_exact:
            violations.append(status_path)
    assert not violations, f"Stray root files/directories created during test run: {violations}"


def test_no_test_writes_outside_tmp():
    """Asserts that no tests pollute the repository tree with outputs/caches."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    forbidden_prefixes = ("masks/", "output/", "runs/")
    forbidden_exact = ("DECISIONS.md",)

    violations = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # status code is first 2 chars
        status_path = line[2:].strip()
        if any(status_path.startswith(prefix) for prefix in forbidden_prefixes):
            violations.append(status_path)
        if status_path in forbidden_exact:
            # Check if it was modified or untracked
            violations.append(status_path)

    # Note: If masks/ existed before, git status porcelain would only show new/modified
    assert not violations, f"Forbidden repo mutations detected outside tmp: {violations}"


def test_no_large_files_tracked():
    """Asserts that no tracked files in git exceed 1000 KB (1 MB)."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    max_bytes = 1000 * 1024
    large_files = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        file_path = repo_root / line
        if file_path.exists() and file_path.is_file():
            size = file_path.stat().st_size
            if size > max_bytes:
                large_files.append((line, size))

    assert not large_files, f"Tracked files exceed 1000 KB limit: {large_files}"

