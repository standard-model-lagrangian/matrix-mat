"""
Script to generate Figure: Deceptive Prior & Slowed Convergence Audit.
Demonstrates:
1. 1D Landscape Transect: GP Prior overconfidence and distortion vs true physical response.
2. ARD Lengthscale Paralysis: GelMA-L lengthscale collapse from 48.3 to 0.48.
3. Early Prospective Prediction Bias: Severe overestimation in early closed-loop batches.
4. Acquisition Function Trap: False minima diverting search from true tissue targets.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AUDIT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from models.gp_surrogate import GPSurrogate, DEFAULT_BOUNDS

def generate_deception_figure():
    canonical_dir = AUDIT_ROOT / "artifacts" / "canonical"
    prior_df = pd.read_parquet(canonical_dir / "prior.parquet")
    points_df = pd.read_parquet(canonical_dir / "points.parquet")
    cl = points_df[points_df["batch_type"] == "closed_loop"].copy()
    batches = sorted(cl["batch_id"].unique())
    features = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]

    X_prior = prior_df[features].values
    y_prior = prior_df["imputed_modulus"].values

    # -------------------------------------------------------------
    # 1. ARD Lengthscale History across Batches
    # -------------------------------------------------------------
    batch_indices = [0, 1, 2, 3, 4, 5, 8, 12, 16, 20, 25]
    ls_history = []

    for b_idx in batch_indices:
        if b_idx == 0:
            X_tr, y_tr = X_prior, y_prior
        else:
            sub = cl[cl["batch_id"].isin(batches[:b_idx])]
            X_tr = np.vstack([X_prior, sub[features].values])
            y_tr = np.concatenate([y_prior, sub["modulus"].values])
        gp = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=42)
        gp.fit(X_tr, y_tr)
        ls_history.append(gp.get_lengthscales())

    ls_history = np.array(ls_history)

    # -------------------------------------------------------------
    # 2. 1D Transect: Healthy Skin formulation slice
    # Target: E = 1.5 kPa at GelMA-L ~ 2.64 wt%
    # -------------------------------------------------------------
    grid_gelma_l = np.linspace(0.0, 6.0, 100)
    slice_X = np.zeros((100, 5))
    slice_X[:, 0] = 31.0   # exposure
    slice_X[:, 1] = grid_gelma_l
    slice_X[:, 2] = 2.55   # GelMA-H
    slice_X[:, 3] = 1.58   # aCNC
    slice_X[:, 4] = 0.08   # LAP

    # Fit Prior-only GP
    gp_prior = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=42)
    gp_prior.fit(X_prior, y_prior)
    mu_prior, sig_prior = gp_prior.predict(slice_X)

    # Fit Full Ground-Truth Proxy (Prior + 100 CL points)
    X_all = np.vstack([X_prior, cl[features].values])
    y_all = np.concatenate([y_prior, cl["modulus"].values])
    gp_full = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=42)
    gp_full.fit(X_all, y_all)
    mu_full, sig_full = gp_full.predict(slice_X)

    # -------------------------------------------------------------
    # 3. Early Batch Prediction Errors (Epistemic Debt)
    # -------------------------------------------------------------
    batch_errors = []
    batch_z_scores = []
    hist_X = list(X_prior)
    hist_y = list(y_prior)

    for b_id in batches:
        b_data = cl[cl["batch_id"] == b_id]
        gp_curr = GPSurrogate(kernel="matern52", ard=True, noise_model="learned", random_seed=42)
        gp_curr.fit(np.array(hist_X), np.array(hist_y))
        
        preds, uncerts = gp_curr.predict(b_data[features].values)
        trues = b_data["modulus"].values
        
        errors = trues - preds
        batch_errors.append(np.mean(errors))
        z = (trues - preds) / uncerts
        batch_z_scores.append(np.mean(np.abs(z)))

        # Update historical data
        hist_X.extend(b_data[features].values)
        hist_y.extend(trues)

    # -------------------------------------------------------------
    # 4. Acquisition Function Trap: Loss = |E - 1.5|
    # -------------------------------------------------------------
    loss_prior = np.abs(mu_prior - 1.5)
    loss_true = np.abs(mu_full - 1.5)

    # -------------------------------------------------------------
    # Render Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: 1D Landscape Transect & Prior Hallucination
    ax = axes[0, 0]
    ax.plot(grid_gelma_l, mu_full, "k-", lw=2.5, label="Empirical Reality (Full GP Proxy)")
    ax.fill_between(grid_gelma_l, mu_full - 1.96 * sig_full, mu_full + 1.96 * sig_full, color="gray", alpha=0.2)
    ax.plot(grid_gelma_l, mu_prior, "r--", lw=2.5, label="Imputed GP Prior (Table S2)")
    ax.fill_between(grid_gelma_l, mu_prior - 1.96 * sig_prior, mu_prior + 1.96 * sig_prior, color="red", alpha=0.15)
    ax.axhline(1.5, color="green", linestyle=":", lw=2, label="Healthy Skin Target (1.5 kPa)")
    ax.axhspan(1.35, 1.65, color="green", alpha=0.15, label="Target Tolerance (±0.15 kPa)")
    ax.axvline(2.64, color="blue", linestyle="--", alpha=0.7, label="True Optimum GelMA-L (2.64%)")
    ax.set_xlabel("GelMA-L Concentration (wt%)", fontsize=11)
    ax.set_ylabel("Elastic Modulus E (kPa)", fontsize=11)
    ax.set_title("A. Prior Hallucination & Upward Modulus Bias", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 10)

    # Panel B: Acquisition Function Distortion (Loss Landscape)
    ax = axes[0, 1]
    ax.plot(grid_gelma_l, loss_true, "k-", lw=2.5, label="True Target Loss |E - 1.5|")
    ax.plot(grid_gelma_l, loss_prior, "r--", lw=2.5, label="Perceived Loss Under Imputed Prior")
    ax.axvline(2.64, color="blue", linestyle="--", alpha=0.7, label="True Minimum (GelMA-L=2.64%)")
    min_prior_idx = np.argmin(loss_prior)
    ax.axvline(grid_gelma_l[min_prior_idx], color="red", linestyle=":", lw=2, 
               label=f"Perceived Minimum (GelMA-L={grid_gelma_l[min_prior_idx]:.2f}%)")
    ax.set_xlabel("GelMA-L Concentration (wt%)", fontsize=11)
    ax.set_ylabel("Target Loss |E - 1.5 kPa|", fontsize=11)
    ax.set_title("B. Acquisition Distortion: False Attractors", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel C: ARD Lengthscale Paralysis & Collapse
    ax = axes[1, 0]
    ax.plot(batch_indices, ls_history[:, 1], "r-o", lw=2.5, label="GelMA-L Lengthscale (ARD)")
    ax.plot(batch_indices, ls_history[:, 2], "b-s", lw=2, label="GelMA-H Lengthscale")
    ax.plot(batch_indices, ls_history[:, 4], "g-^", lw=2, label="LAP Lengthscale")
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="Nominal Normalized Lengthscale (1.0)")
    ax.set_yscale("log")
    ax.set_xlabel("Closed-Loop Batch Index t", fontsize=11)
    ax.set_ylabel("ARD Lengthscale (log scale)", fontsize=11)
    ax.set_title("C. ARD Parameter Paralysis: 'Unlearning' Delay", fontsize=12, fontweight="bold")
    ax.annotate("Prior Pegged at 48.3\n(Attentional Blindness)", xy=(0, 48.3), xytext=(2, 20),
                arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, color='darkred', fontweight='bold')
    ax.annotate("Lengthscale Collapse (0.48)\nafter Real Data Arrives", xy=(2, 0.48), xytext=(5, 0.15),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, color='black')
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel D: Prospective Prediction Error (Epistemic Debt)
    ax = axes[1, 1]
    batch_nums = np.arange(1, len(batches) + 1)
    colors = ["#d9534f" if err < 0 else "#0275d8" for err in batch_errors]
    ax.bar(batch_nums, batch_errors, color=colors, alpha=0.7, edgecolor="black", width=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.axhspan(-0.15, 0.15, color="green", alpha=0.2, label="Target Tolerance Window (±0.15 kPa)")
    ax.set_xlabel("Closed-Loop Batch Number t", fontsize=11)
    ax.set_ylabel("Prediction Error (True - Pred Modulus, kPa)", fontsize=11)
    ax.set_title("D. Sequential Prediction Shocks in Early Batches", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_dir = AUDIT_ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "fig_imputation_deception.pdf"
    png_path = out_dir / "fig_imputation_deception.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Generated {pdf_path} and {png_path}")

if __name__ == "__main__":
    generate_deception_figure()
