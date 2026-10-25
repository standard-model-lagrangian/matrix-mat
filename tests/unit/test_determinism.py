import os
import re
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from spheroid_pipeline_v2.repro import (
    set_determinism,
    verify_model_weights,
    get_model_weights_info,
)


def test_missing_weights_raises_with_fetch_hint(tmp_path):
    """Missing model weights must raise FileNotFoundError with actionable fetch command."""
    non_existent = tmp_path / "non_existent_weights.pt"
    with pytest.raises(FileNotFoundError) as exc_info:
        verify_model_weights(non_existent, expected_sha256="abc123def456")
    msg = str(exc_info.value)
    assert "tools/fetch_models.py" in msg or "fetch_models" in msg


def test_weights_sha_mismatch_raises(tmp_path):
    """Corrupted weights with wrong SHA-256 must raise ValueError."""
    corrupt_file = tmp_path / "fake_model"
    corrupt_file.write_text("corrupted model bytes")
    with pytest.raises(ValueError) as exc_info:
        verify_model_weights(corrupt_file, expected_sha256="0000000000000000000000000000000000000000000000000000000000000000")
    assert "mismatch" in str(exc_info.value).lower()


def _locate_cpsam_weights():
    """Weights from MATRIX_MAT_MODELS_ROOT, else <repo>/models, independent of the cwd."""
    from matrix_mat.core.paths import resolve_models_root
    repo_root = Path(__file__).resolve().parents[2]
    for root in (resolve_models_root() if os.environ.get("MATRIX_MAT_MODELS_ROOT") else None, repo_root / "models"):
        if root is not None and (Path(root) / "cpsam_v2").exists():
            return Path(root) / "cpsam_v2"
    return None


@pytest.mark.dl
def test_set_determinism_sets_torch_and_numpy():
    """set_determinism configures seeds and returns independent numpy Generator."""
    torch = pytest.importorskip("torch")
    rng1 = set_determinism(42, include_torch=True)
    val1 = rng1.random()
    rng2 = set_determinism(42, include_torch=True)
    val2 = rng2.random()
    assert val1 == val2
    assert torch.initial_seed() == 42


def test_no_global_numpy_seed_calls():
    """Production codebase must NOT call global np.random.seed (must use Generator)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    pattern = re.compile(r"np\.random\.seed\(")

    scan_dirs = [
        repo_root / "spheroid_pipeline_v2",
        repo_root / "spheroid_if_sweep",
    ]
    if (repo_root / "src" / "matrix_mat").exists():
        scan_dirs.append(repo_root / "src" / "matrix_mat")

    violating_lines = []
    for sdir in scan_dirs:
        if not sdir.exists():
            continue
        for py_file in sdir.rglob("*.py"):
            # Skip test files and pycache
            if "__pycache__" in str(py_file) or "tests" in str(py_file):
                continue
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for line_no, line in enumerate(lines, 1):
                clean_line = line.strip()
                if clean_line.startswith("#"):
                    continue
                if pattern.search(clean_line):
                    violating_lines.append(f"{py_file.relative_to(repo_root)}:{line_no}: {clean_line}")

    assert not violating_lines, (
        f"Found forbidden calls to global np.random.seed:\n" + "\n".join(violating_lines)
    )


def test_run_metadata_records_weights_sha_and_backend(tmp_path):
    """Run metadata must record segmentation backend and weights SHA-256."""
    from spheroid_pipeline_v2.provenance import write_run_metadata
    from spheroid_pipeline_v2.config import PipelineConfig

    cfg = PipelineConfig(output_dir=str(tmp_path))
    out_json = write_run_metadata(
        output_dir=tmp_path,
        config=cfg,
        weights_sha256="test_sha_256_hash",
        backend="classical",
    )
    import json
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "weights_sha256" in data or "model_weights_sha256" in data
    assert "backend" in data or "segmentation_backend" in data


def test_pipeline_bitwise_reproducible_cpu(tmp_path):
    """Running synthetic fixture twice on CPU must produce bitwise identical objects CSV."""
    from spheroid_pipeline_v2.run_pipeline import run_spheroid_pipeline
    from spheroid_pipeline_v2.config import PipelineConfig

    fixture_dir = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_dataset"
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"

    cfg1 = PipelineConfig(
        t0_dir=str(fixture_dir / "Day0"),
        t7_dir=str(fixture_dir / "Day7"),
        output_dir=str(out1),
    )
    cfg1.segmentation.primary_backend = "classical"
    cfg1.segmentation.force_recompute = True

    cfg2 = PipelineConfig(
        t0_dir=str(fixture_dir / "Day0"),
        t7_dir=str(fixture_dir / "Day7"),
        output_dir=str(out2),
    )
    cfg2.segmentation.primary_backend = "classical"
    cfg2.segmentation.force_recompute = True

    set_determinism(42)
    run_spheroid_pipeline(cfg1)
    set_determinism(42)
    run_spheroid_pipeline(cfg2)

    df1 = pd.read_csv(out1 / "objects.csv")
    df2 = pd.read_csv(out2 / "objects.csv")
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.dl
def test_cpsam_reproducible_within_tolerance():
    """CPSAM GPU/MPS inference is reproducible: object count identical and area within 0.5%."""
    pytest.importorskip("cellpose")
    from matrix_mat.core.paths import resolve_models_root
    from matrix_mat.if_density.engine import CPSAMEngine
    model_path = _locate_cpsam_weights()
    if model_path is None:
        pytest.skip("cpsam_v2 weights not found (set MATRIX_MAT_MODELS_ROOT or run tools/fetch_models.py)")

    # Generate synthetic image
    img = np.zeros((256, 256), dtype=np.uint8)
    centers = [(64, 64), (64, 192), (192, 64), (192, 192)]
    radius = 15.0
    for cy, cx in centers:
        y, x = np.ogrid[:256, :256]
        dist_sq = (y - cy) ** 2 + (x - cx) ** 2
        blob = np.exp(-dist_sq / (2.0 * (radius / 2.0) ** 2)) * 220
        img = np.maximum(img, blob.astype(np.uint8))

    engine1 = CPSAMEngine(model_path=str(model_path), random_seed=42)
    masks1, _ = engine1.segment_image(
        img,
        {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "diameter": 30.0, "min_size": 15},
    )

    engine2 = CPSAMEngine(model_path=str(model_path), random_seed=42)
    masks2, _ = engine2.segment_image(
        img,
        {"flow_threshold": 0.4, "cellprob_threshold": 0.0, "diameter": 30.0, "min_size": 15},
    )

    n1 = int(masks1.max())
    n2 = int(masks2.max())
    assert n1 == n2, f"Object count differed: run1={n1}, run2={n2}"

    area1 = float(np.sum(masks1 > 0))
    area2 = float(np.sum(masks2 > 0))
    if area1 > 0:
        rel_diff = abs(area1 - area2) / area1
        assert rel_diff <= 0.005, f"Area relative difference {rel_diff:.4f} exceeds 0.5% tolerance"

