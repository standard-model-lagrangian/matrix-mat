"""
Shared pytest fixtures and configuration for E2E Testing Track.
Audit of Chen et al., Adv. Funct. Mater. 2026, 36:e75639.
"""

import hashlib
import os
from pathlib import Path
import pytest
import pandas as pd


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the absolute path to the audit project root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def upstream_dir() -> Path:
    """Return the absolute path to the upstream Hydrogel_BO repository."""
    candidate = Path("/Users/frederick/Visual Studio /Material discovery/Hydrogel_BO")
    if candidate.is_dir():
        return candidate
    # Fallback relative to project
    candidate_rel = Path(__file__).resolve().parent.parent.parent / "Hydrogel_BO"
    return candidate_rel


@pytest.fixture(scope="session")
def canonical_dir(project_root: Path) -> Path:
    """Return the path to artifacts/canonical."""
    return project_root / "artifacts" / "canonical"


@pytest.fixture(scope="session")
def cards_dir(project_root: Path) -> Path:
    """Return the path to artifacts/cards."""
    return project_root / "artifacts" / "cards"


@pytest.fixture(scope="session")
def env_dir(project_root: Path) -> Path:
    """Return the path to env."""
    return project_root / "env"


@pytest.fixture(scope="session")
def seeds_path(env_dir: Path) -> Path:
    """Return the path to env/seeds.yaml."""
    return env_dir / "seeds.yaml"


@pytest.fixture(scope="session")
def card_schema_path(project_root: Path) -> Path:
    """Return the path to card_schema.json in src/utils or project root."""
    p1 = project_root / "src" / "utils" / "card_schema.json"
    if p1.is_file():
        return p1
    return project_root / "card_schema.json"


@pytest.fixture(scope="session")
def manifest_path(canonical_dir: Path) -> Path:
    """Return the path to artifacts/canonical/manifest.sha256."""
    return canonical_dir / "manifest.sha256"


@pytest.fixture(scope="session")
def upstream_bo_df(upstream_dir: Path) -> pd.DataFrame:
    """Load the upstream hydrogel_bo.csv reference dataset (211 rows)."""
    csv_path = upstream_dir / "hydrogel_bo.csv"
    assert csv_path.is_file(), f"Upstream file not found: {csv_path}"
    return pd.read_csv(csv_path, index_col=0)


@pytest.fixture(scope="session")
def upstream_analysis_df(upstream_dir: Path) -> pd.DataFrame:
    """Load the upstream hydrogel_analysis.csv reference dataset (168 rows)."""
    csv_path = upstream_dir / "hydrogel_analysis.csv"
    assert csv_path.is_file(), f"Upstream file not found: {csv_path}"
    return pd.read_csv(csv_path, index_col=0)


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_known_constraint(gelma_l: float, gelma_h: float, acnc: float, lap: float) -> bool:
    """
    Physical feasibility constraint defined in Section 5 and run.py:
    1. Total polymer concentration: GelMA_L/15 + GelMA_H/15 + aCNC/6.1 + LAP/2 <= 1.0
    2. Minimum GelMA content for network integrity: GelMA_L + GelMA_H >= 3.0
    """
    if (gelma_l / 15.0) + (gelma_h / 15.0) + (acnc / 6.1) + (lap / 2.0) > 1.0:
        return False
    if (gelma_l + gelma_h) < 3.0:
        return False
    return True
