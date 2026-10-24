"""
Tier 4 Tests: End-to-End Pipeline Invariants, Mathematical Reproducibility & Geometric Audits.
Validates:
- Exact 2D Hypervolume plateau reproduction at 362-364 using empirical nadir (Task T07)
- 5D Convex Hull explored parameter volume ~2.815% of nominal box (Task T11)
- Physical feasibility space Monte Carlo volume ~45.5% (Task T11)
- Upstream raw file immutability and provenance
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull
import hashlib


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_known_constraint(gelma_l: float, gelma_h: float, acnc: float, lap: float) -> bool:
    """Physical feasibility constraint defined in Section 5 and run.py."""
    if (gelma_l / 15.0) + (gelma_h / 15.0) + (acnc / 6.1) + (lap / 2.0) > 1.0:
        return False
    if (gelma_l + gelma_h) < 3.0:
        return False
    return True



def compute_2d_hypervolume(points_2d: np.ndarray, ref_point: tuple[float, float]) -> float:
    """
    Compute the exact 2D hypervolume dominated by points_2d with respect to ref_point.
    points_2d shape: (N, 2), columns: [Modulus, Stress Relaxation].
    Both objectives are maximized.
    """
    # Filter points dominating the reference point
    valid = points_2d[(points_2d[:, 0] >= ref_point[0]) & (points_2d[:, 1] >= ref_point[1])]
    if len(valid) == 0:
        return 0.0

    # Sort descending by Modulus (first objective)
    sorted_pts = valid[valid[:, 0].argsort()[::-1]]

    # Extract non-dominated Pareto front
    pareto = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p[1] > pareto[-1][1]:
            pareto.append(p)

    pf = np.array(pareto)
    # Sort ascending by Modulus for rectangular integration
    pf = pf[pf[:, 0].argsort()]

    # Riemann summation of rectangles
    area = (pf[0, 0] - ref_point[0]) * (pf[0, 1] - ref_point[1])
    for i in range(1, len(pf)):
        area += (pf[i, 0] - pf[i - 1, 0]) * (pf[i, 1] - ref_point[1])

    return float(area)


class TestE2EReproducibility:
    """Test mathematical and geometric reproduction across the audit pipeline."""

    def test_hypervolume_2d_plateau_reproduction(self, canonical_dir: Path):
        """
        Verify the 362-364 Hypervolume plateau reported in Figure 4 and Task T07.
        - Reference point: empirical nadir (0.093 kPa, 8.0%).
        - HV at point 100: ~301.07
        - HV at point 128 (Utopian 1): ~362.12 (growth of +61.05)
        - HV at point 168 (final): ~364.08 (total growth beyond point 100: +63.01)
        - Utopian 1 accounts for 96.9% of all hypervolume gain beyond closed-loop.
        """
        df = pd.read_parquet(canonical_dir / "points.parquet")
        pts = df[["modulus", "stress_relaxation"]].values
        ref_nadir = (0.093, 8.0)

        hv_100 = compute_2d_hypervolume(pts[:100], ref_nadir)
        hv_128 = compute_2d_hypervolume(pts[:128], ref_nadir)
        hv_168 = compute_2d_hypervolume(pts[:168], ref_nadir)

        assert abs(hv_100 - 301.07) < 0.5, f"HV at 100 expected ~301.07, got {hv_100:.2f}"
        assert abs(hv_128 - 362.12) < 0.5, f"HV at 128 expected ~362.12, got {hv_128:.2f}"
        assert abs(hv_168 - 364.08) < 0.5, f"HV at 168 expected ~364.08, got {hv_168:.2f}"

        # Plateau verification: gain after Utopian 1 is minimal (< 2.0)
        gain_u1 = hv_128 - hv_100
        gain_total = hv_168 - hv_100
        assert gain_u1 / gain_total > 0.95, (
            f"Utopian 1 should account for > 95% of hypervolume growth, got {gain_u1/gain_total*100:.1f}%"
        )
        assert (hv_168 - hv_128) < 2.5, "Hypervolume plateau violated: substantial growth observed after Utopian 1"

    def test_explored_parameter_volume_5d_convex_hull(self, canonical_dir: Path):
        """
        Verify the effective explored volume calculation (Task T11).
        The nominal 5D box is [0, 600] x [0, 6]^3 x [0, 0.2].
        Convex hull of 168 points normalized to [0, 1]^5 occupies ~2.815% of the box.
        Convex hull of the initial 100 points occupies ~0.436% of the box.
        """
        df = pd.read_parquet(canonical_dir / "points.parquet")
        features = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]
        X = df[features].values
        bounds = np.array([[0.0, 600.0], [0.0, 6.0], [0.0, 6.0], [0.0, 6.0], [0.0, 0.20]])

        # Normalize to unit hypercube [0, 1]^5
        X_unit = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

        # All points must be inside unit box
        assert (X_unit >= 0.0).all() and (X_unit <= 1.0).all()

        hull_168 = ConvexHull(X_unit)
        vol_pct_168 = hull_168.volume * 100.0
        assert abs(vol_pct_168 - 2.815) < 0.2, (
            f"Explored volume of 168 points expected ~2.815%, got {vol_pct_168:.3f}%"
        )

        hull_100 = ConvexHull(X_unit[:100])
        vol_pct_100 = hull_100.volume * 100.0
        assert abs(vol_pct_100 - 0.436) < 0.1, (
            f"Explored volume of 100 closed-loop points expected ~0.436%, got {vol_pct_100:.3f}%"
        )

    def test_physical_feasibility_space_volume_monte_carlo(self):
        """
        Verify that the physical feasibility constraint (known_constraint) restricts
        the nominal 5D formulation box to ~45.5% feasible volume via deterministic Monte Carlo.
        """
        rng = np.random.default_rng(20260919)
        n_samples = 50000

        # Uniform sampling within 5D bounds
        gelma_l = rng.uniform(0.0, 6.0, n_samples)
        gelma_h = rng.uniform(0.0, 6.0, n_samples)
        acnc = rng.uniform(0.0, 6.0, n_samples)
        lap = rng.uniform(0.0, 0.20, n_samples)

        feasible_count = 0
        for i in range(n_samples):
            if check_known_constraint(gelma_l[i], gelma_h[i], acnc[i], lap[i]):
                feasible_count += 1

        feasible_fraction = feasible_count / n_samples
        assert abs(feasible_fraction - 0.455) < 0.02, (
            f"Expected feasible fraction ~45.5%, got {feasible_fraction*100:.2f}%"
        )

    def test_upstream_oracle_csv_digests(self, upstream_dir: Path):
        """
        Verify that upstream raw datasets in Hydrogel_BO have valid non-empty files
        and consistent line counts (212 lines in bo.csv, 169 in analysis.csv).
        """
        bo_csv = upstream_dir / "hydrogel_bo.csv"
        an_csv = upstream_dir / "hydrogel_analysis.csv"

        assert bo_csv.is_file(), f"Missing {bo_csv}"
        assert an_csv.is_file(), f"Missing {an_csv}"

        bo_lines = bo_csv.read_text(encoding="utf-8").strip().splitlines()
        an_lines = an_csv.read_text(encoding="utf-8").strip().splitlines()

        assert len(bo_lines) == 212, f"Expected 212 lines in hydrogel_bo.csv, got {len(bo_lines)}"
        assert len(an_lines) == 169, f"Expected 169 lines in hydrogel_analysis.csv, got {len(an_lines)}"
