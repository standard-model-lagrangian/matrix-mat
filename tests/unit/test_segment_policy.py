from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import spheroid_pipeline_v2
from spheroid_pipeline_v2.config import PipelineConfig, SegmentationConfig
from spheroid_pipeline_v2.segment import SegmentationHierarchy
from spheroid_pipeline_v2.measure import SpheroidObjectRecord, OBJECTS_CSV_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]


class StubSegmenter:
    """Stub segmenter returning configurable masks and metadata."""
    def __init__(self, mask_fn=None, backend_name="stub_primary", model_id="stub_model_v1", diameter=150.0):
        self.mask_fn = mask_fn
        self.backend_name = backend_name
        self.model_id = model_id
        self.diameter = diameter
        self.calls = 0

    def segment(self, img, pixel_size_um):
        self.calls += 1
        if self.mask_fn:
            mask = self.mask_fn(img)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.int32)
        return mask, {
            "backend": self.backend_name,
            "model_id": self.model_id,
            "diameter_used": self.diameter,
            "status": "success" if np.max(mask) > 0 else "empty",
        }


def test_implausible_mask_flags_image_not_switches_backend():
    """Default fallback_policy ('strict') must NOT switch backend on 0 objects or implausible mask."""
    config = SegmentationConfig(primary_backend="cpsam", fallback_policy="strict")
    hierarchy = SegmentationHierarchy(config)

    # Stub primary to return empty mask (0 objects)
    primary_stub = StubSegmenter(mask_fn=lambda img: np.zeros(img.shape[:2], dtype=np.int32), backend_name="cpsam")
    fallback_stub = StubSegmenter(mask_fn=lambda img: np.ones(img.shape[:2], dtype=np.int32), backend_name="classical")
    hierarchy.cpsam_segmenter = primary_stub
    hierarchy.classical_segmenter = fallback_stub

    dummy_img = np.zeros((200, 200), dtype=np.uint8)
    mask, meta = hierarchy.segment(dummy_img, image_id="img_implausible", force_recompute=True)

    # Under strict policy, primary was called, fallback was NOT called, and backend is primary
    assert primary_stub.calls == 1
    assert fallback_stub.calls == 0
    assert meta["backend"] == "cpsam"
    assert meta.get("plausibility_status") in ("FAIL", "EMPTY", "FLAGGED", "failed") or meta.get("status") == "empty"


def test_fallback_policy_per_image_requires_opt_in():
    """Fallback policy must default to 'strict'. Opt-in 'per_image' allows fallback."""
    cfg_default = SegmentationConfig()
    assert cfg_default.fallback_policy == "strict"

    cfg_opt_in = SegmentationConfig(fallback_policy="per_image")
    assert cfg_opt_in.fallback_policy == "per_image"

    hierarchy = SegmentationHierarchy(cfg_opt_in)
    primary_stub = StubSegmenter(mask_fn=lambda img: np.zeros(img.shape[:2], dtype=np.int32), backend_name="cpsam")
    classical_stub = StubSegmenter(mask_fn=lambda img: np.ones(img.shape[:2], dtype=np.int32), backend_name="classical")
    hierarchy.cpsam_segmenter = primary_stub
    hierarchy.classical_segmenter = classical_stub

    dummy_img = np.zeros((200, 200), dtype=np.uint8)
    mask, meta = hierarchy.segment(dummy_img, image_id="img_fallback_allowed", force_recompute=True)

    # With per_image policy, primary failed so classical was called
    assert primary_stub.calls == 1
    assert classical_stub.calls == 1
    assert meta["backend"] == "classical"


def test_backend_recorded_on_every_object(tmp_path):
    """SpheroidObjectRecord and objects.csv must have seg_backend, model_id, diameter_used."""
    assert "seg_backend" in OBJECTS_CSV_COLUMNS
    assert "model_id" in OBJECTS_CSV_COLUMNS
    assert "diameter_used" in OBJECTS_CSV_COLUMNS

    config = SegmentationConfig(primary_backend="cpsam", cache_dir=str(tmp_path))
    hierarchy = SegmentationHierarchy(config)

    # Stub returning 1 object
    def _make_obj(img):
        m = np.zeros(img.shape[:2], dtype=np.int32)
        m[50:100, 50:100] = 1
        return m

    hierarchy.cpsam_segmenter = StubSegmenter(mask_fn=_make_obj, backend_name="cpsam", model_id="cpsam_v2", diameter=160.0)

    dummy_img = np.zeros((200, 200), dtype=np.uint8)
    mask, meta = hierarchy.segment(dummy_img, image_id="test_cache_record", force_recompute=True)
    assert meta["backend"] == "cpsam"

    # Now load from cache
    mask_cached, meta_cached = hierarchy.segment(dummy_img, image_id="test_cache_record", force_recompute=False)
    assert meta_cached["backend"] == "cpsam"
    assert meta_cached["model_id"] == "cpsam_v2"
    assert meta_cached["diameter_used"] == 160.0


def test_condition_summary_reports_backend_mix():
    """condition_summary.csv must include backend_counts."""
    from spheroid_pipeline_v2.stats import PopulationStatsCalculator, SpheroidPairRecord

    # Create dummy pairs with objects that have seg_backend
    df_pairs = pd.DataFrame([
        {
            "pair_id": "p1",
            "condition": "Mat",
            "gel": "Gel1",
            "fold_change_volume": 1.2,
            "log2_fold_change": 0.26,
            "pass_fail": "PASS",
            "d0_equivalent_diameter_um": 100.0,
            "seg_backend": "cpsam",
        },
        {
            "pair_id": "p2",
            "condition": "Mat",
            "gel": "Gel2",
            "fold_change_volume": 1.3,
            "log2_fold_change": 0.38,
            "pass_fail": "PASS",
            "d0_equivalent_diameter_um": 110.0,
            "seg_backend": "cpsam",
        },
    ])
    calc = PopulationStatsCalculator()
    summary = calc.compute_condition_statistics(df_pairs)
    assert "backend_counts" in summary.columns
    assert "cpsam:2" in str(summary.iloc[0]["backend_counts"])


@pytest.mark.dl
def test_cellpose_model_identity(monkeypatch):
    """When config asks for cpdino, loaded model's pretrained_model must contain cpdino, else raise."""
    pytest.importorskip("cellpose")
    from spheroid_pipeline_v2.segment import CellposeFoundationSegmenter
    from unittest.mock import MagicMock
    import cellpose.models

    fake_model = MagicMock()
    fake_model.pretrained_model = "/path/to/cpsam_v2"
    mock_init = MagicMock(return_value=fake_model)
    monkeypatch.setattr(cellpose.models, "CellposeModel", mock_init)

    seg = CellposeFoundationSegmenter()
    with pytest.raises(RuntimeError, match="Silent fallback to cpsam is disallowed"):
        seg._init_model()

    fake_model.pretrained_model = "/path/to/cpdino"
    seg._model = None
    m = seg._init_model()
    assert "cpdino" in m.pretrained_model


def test_ci_smoke_reports_backend(tmp_path):
    """The smoke test asserts the backend it expected actually ran, so a silent fallback fails."""
    from pathlib import Path
    from matrix_mat.brightfield.config import PipelineConfig
    from matrix_mat.brightfield.run_pipeline import run_spheroid_pipeline

    d0_dir = (REPO_ROOT / "tests/fixtures/synthetic_dataset/Day0")
    d7_dir = (REPO_ROOT / "tests/fixtures/synthetic_dataset/Day7")

    config = PipelineConfig(
        t0_dir=str(d0_dir),
        t7_dir=str(d7_dir),
        output_dir=str(tmp_path),
        random_seed=42,
    )
    config.segmentation.primary_backend = "classical"
    config.segmentation.force_recompute = True

    result = run_spheroid_pipeline(config)
    assert result.get("status") == "SUCCESS"

    objs_csv = tmp_path / "objects.csv"
    assert objs_csv.exists()
    df = pd.read_csv(objs_csv)
    assert "seg_backend" in df.columns
    backends = df["seg_backend"].dropna().unique().tolist()
    assert backends == ["classical"], f"Expected only ['classical'] backend, got {backends}"





def test_missing_weights_raise_instead_of_empty_masks(tmp_path):
    """A missing cpsam weights file must abort segmentation, not silently yield 0 objects per image."""
    from spheroid_pipeline_v2.segment import CellposeSAMSegmenter

    cfg = SegmentationConfig(primary_backend="cpsam", cpsam_model_path=str(tmp_path / "missing" / "cpsam_v2"),
                             cache_dir=str(tmp_path / "cache"))
    hierarchy = SegmentationHierarchy(cfg)
    assert isinstance(hierarchy.cpsam_segmenter, CellposeSAMSegmenter)
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        hierarchy.segment(np.zeros((64, 64), dtype=np.uint8), image_id="no_weights", force_recompute=True)


@pytest.mark.parametrize("bad_backend", ["cyto3", "sam", "cellpose"])
def test_unknown_backend_rejected(bad_backend):
    """Unknown backend names must be rejected instead of silently running cpsam."""
    with pytest.raises(ValueError, match="cpdino"):
        SegmentationConfig(primary_backend=bad_backend)


def test_failed_inference_is_flagged_and_not_cached(tmp_path):
    """A per-image inference error flags the image SEG_FAIL and is never written to the mask cache."""
    config = SegmentationConfig(primary_backend="cpsam", cache_dir=str(tmp_path))
    hierarchy = SegmentationHierarchy(config)

    class FailingStub(StubSegmenter):
        def segment(self, img, pixel_size_um):
            self.calls += 1
            return np.zeros(img.shape[:2], dtype=np.int32), {
                "backend": "cpsam", "model_id": "cpsam_v2", "status": "failed", "error": "boom",
            }

    stub = FailingStub()
    hierarchy.cpsam_segmenter = stub
    img = np.zeros((64, 64), dtype=np.uint8)
    _, meta = hierarchy.segment(img, image_id="flaky", force_recompute=False)
    assert meta["qc_flag"] == "SEG_FAIL"
    assert "boom" in meta["qc_reasons"]
    _, meta2 = hierarchy.segment(img, image_id="flaky", force_recompute=False)
    assert stub.calls == 2, "failed segmentation must not be served from cache"
    assert not meta2.get("cached", False)


def test_seg_fail_flag_survives_cache_hit(tmp_path):
    """Re-running on a cached implausible mask must still flag the image SEG_FAIL."""
    config = SegmentationConfig(primary_backend="cpsam", cache_dir=str(tmp_path))
    hierarchy = SegmentationHierarchy(config)
    hierarchy.cpsam_segmenter = StubSegmenter(mask_fn=lambda img: np.zeros(img.shape[:2], dtype=np.int32),
                                              backend_name="cpsam")
    img = np.zeros((64, 64), dtype=np.uint8)
    _, meta1 = hierarchy.segment(img, image_id="empty_field", force_recompute=False)
    _, meta2 = hierarchy.segment(img, image_id="empty_field", force_recompute=False)
    assert meta1["qc_flag"] == "SEG_FAIL"
    assert meta2.get("cached") is True
    assert meta2.get("qc_flag") == "SEG_FAIL"
    assert meta2.get("plausibility_status") == "FAIL"
