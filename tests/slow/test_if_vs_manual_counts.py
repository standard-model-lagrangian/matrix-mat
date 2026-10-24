from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from spheroid_if_sweep.postprocess import postprocess_field_masks


@pytest.mark.slow
@pytest.mark.realdata
def test_if_pipeline_counts_vs_manual():
    """
    Validate pipeline nuclear counts against committed human manual counts on held-out fields.
    Pre-declared acceptance criteria:
    - Absolute mean percentage bias < 10%
    - Spearman rank correlation rho > 0.90
    """
    csv_path = Path(__file__).parent.parent / "data" / "if_manual_counts.csv"
    assert csv_path.exists(), f"Manual counts file not found at {csv_path}"
    df_manual = pd.read_csv(csv_path)

    # Check if run output per_image_deltas.csv exists in runs/comparison
    deltas_path = Path("runs/comparison/per_image_deltas.csv")
    if not deltas_path.exists():
        pytest.skip("Precomputed run comparison not found. Run IF sweep first to test against real data.")

    df_deltas = pd.read_csv(deltas_path).set_index("image_id")
    common_ids = [img_id for img_id in df_manual["image_id"] if img_id in df_deltas.index]
    if len(common_ids) < 3:
        pytest.skip(f"Not enough matching images in deltas ({len(common_ids)} found).")

    manual_counts = []
    pipeline_counts = []
    # Test PP11_full or PP01 counts
    for img_id in common_ids:
        m_count = float(df_manual.loc[df_manual["image_id"] == img_id, "manual_count"].iloc[0])
        # Try 104_PP11_full_count or 104s_PP11_full_count or 001_A_baseline_count
        row = df_deltas.loc[img_id]
        p_count = None
        for col in ["104_PP11_full_count", "104s_PP11_full_count", "001_A_baseline_count"]:
            if col in row and pd.notna(row[col]) and row[col] > 0:
                p_count = float(row[col])
                break
        if p_count is not None:
            manual_counts.append(m_count)
            pipeline_counts.append(p_count)

    manual_arr = np.array(manual_counts)
    pipeline_arr = np.array(pipeline_counts)

    # Relative bias: mean((pipeline - manual) / manual)
    pct_biases = (pipeline_arr - manual_arr) / manual_arr
    mean_abs_bias = float(np.mean(np.abs(pct_biases)))

    # Spearman correlation
    rho, pval = stats.spearmanr(manual_arr, pipeline_arr)

    assert mean_abs_bias < 0.10, f"Mean absolute bias {mean_abs_bias*100:.2f}% exceeds 10% threshold"
    assert rho > 0.90, f"Spearman correlation {rho:.3f} is below 0.90 threshold"
