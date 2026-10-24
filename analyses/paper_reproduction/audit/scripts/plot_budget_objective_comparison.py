"""
Generate Publication Figure: Objective-Space Multi-Budget Discovery Comparison.
Compares Chen et al. Active Learning vs Space-Filling (Sobol / LHS / DOE)
across 3 budget milestones (25%, 50%, 100% of total experimental budget).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Path setup
AUDIT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from audit.seed_baselines import sample_sobol, sample_lhs, sample_d_optimal, is_feasible, TARGET_WINDOWS, BOUNDS_MIN, BOUNDS_MAX
from audit.pareto import compute_2d_hypervolume, extract_pareto_indices

def create_budget_comparison_figure():
    canonical_dir = AUDIT_ROOT / "artifacts" / "canonical"
    points_df = pd.read_parquet(canonical_dir / "points.parquet")
    optima_df = pd.read_parquet(canonical_dir / "optima.parquet")
    
    features = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]
    X = points_df[features].values
    y_mod = points_df["modulus"].values
    y_sr = points_df["stress_relaxation"].values
    
    # Train Digital Twin Oracle
    gb_mod = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_mod.fit(X, y_mod)
    gb_sr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_sr.fit(X, y_sr)

    # 3 Budget Milestones
    budgets = [42, 84, 168]
    budget_labels = ["25% Budget (N=42)", "50% Budget (N=84)", "100% Budget (N=168)"]
    
    # Precompute data for Chen AL
    al_data = {}
    for n in budgets:
        sub = points_df.iloc[:n]
        pts = sub[["modulus", "stress_relaxation"]].values
        hv = compute_2d_hypervolume(pts, (0.0, 0.0))
        # Pareto front
        p_idx = extract_pareto_indices(pts)
        al_data[n] = {
            "pts": pts,
            "hv": hv,
            "p_idx": p_idx,
            "batch_type": sub["batch_type"].values
        }

    # Precompute data for Sobol Space-Filling
    sobol_data = {}
    seed = 20260919
    for n in budgets:
        raw_pts = sample_sobol(n, seed=seed)
        pred_m = gb_mod.predict(raw_pts)
        pred_s = gb_sr.predict(raw_pts)
        pts = np.column_stack([pred_m, pred_s])
        hv = compute_2d_hypervolume(pts, (0.0, 0.0))
        p_idx = extract_pareto_indices(pts)
        sobol_data[n] = {
            "raw_pts": raw_pts,
            "pts": pts,
            "hv": hv,
            "p_idx": p_idx
        }

    # Style configuration
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
    })

    fig = plt.figure(figsize=(15, 11), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85], hspace=0.32, wspace=0.22)

    # Define target windows styling
    target_boxes = [
        {"name": "Healthy Skin", "mod": (1.35, 1.65), "sr": (22.5, 27.5), "color": "#2ca02c", "alpha": 0.25, "text_pos": (1.5, 30.5)},
        {"name": "Soft Tumor", "mod": (1.17, 1.43), "sr": (45.0, 55.0), "color": "#7b1fa2", "alpha": 0.25, "text_pos": (1.3, 58.0)},
        {"name": "Scarred Skin", "mod": (10.8, 13.2), "sr": (18.0, 22.0), "color": "#e65100", "alpha": 0.25, "text_pos": (12.0, 24.5)},
        {"name": "Stiff Tumor", "mod": (14.4, 17.6), "sr": (10.0, 14.0), "color": "#c62828", "alpha": 0.25, "text_pos": (16.0, 16.5)},
    ]

    def count_target_hits(pts):
        hits = 0
        for tw in target_boxes:
            m_low, m_high = tw["mod"]
            s_low, s_high = tw["sr"]
            in_w = np.any((pts[:, 0] >= m_low) & (pts[:, 0] <= m_high) & (pts[:, 1] >= s_low) & (pts[:, 1] <= s_high))
            if in_w:
                hits += 1
        return hits

    def draw_target_windows(ax):
        for tw in target_boxes:
            m_low, m_high = tw["mod"]
            s_low, s_high = tw["sr"]
            rect = patches.Rectangle(
                (m_low, s_low), m_high - m_low, s_high - s_low,
                linewidth=1.2, edgecolor=tw["color"], facecolor=tw["color"],
                alpha=tw["alpha"], linestyle="--", zorder=1
            )
            ax.add_patch(rect)
            ax.text(tw["text_pos"][0], tw["text_pos"][1], tw["name"],
                    fontsize=8, color=tw["color"], fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    def draw_pareto_step(ax, pts, p_idx, color="#1f77b4"):
        p_pts = pts[p_idx]
        # Sort by modulus ascending
        order = np.argsort(p_pts[:, 0])
        p_sorted = p_pts[order]
        # Step line
        ax.step(p_sorted[:, 0], p_sorted[:, 1], where="post", color=color, lw=2.0, alpha=0.85, label="Pareto Front", zorder=3)
        ax.scatter(p_sorted[:, 0], p_sorted[:, 1], color=color, s=45, edgecolor="black", lw=0.6, zorder=4)

    # -------------------------------------------------------------------------
    # ROW 1: Chen et al. Active Learning (Atlas BO) across 3 budgets
    # -------------------------------------------------------------------------
    for col_idx, n in enumerate(budgets):
        ax = fig.add_subplot(gs[0, col_idx])
        draw_target_windows(ax)
        
        pts = al_data[n]["pts"]
        p_idx = al_data[n]["p_idx"]
        hv = al_data[n]["hv"]
        targ_hits = count_target_hits(pts)

        # Plot points: distinguish closed loop vs pareto mapping
        b_type = al_data[n]["batch_type"]
        is_cl = (b_type == "closed_loop")
        
        ax.scatter(pts[is_cl, 0], pts[is_cl, 1], color="#1f77b4", alpha=0.65, s=36, edgecolor="none", label="Closed-Loop Points")
        if np.any(~is_cl):
            ax.scatter(pts[~is_cl, 0], pts[~is_cl, 1], color="#e377c2", alpha=0.75, s=40, marker="D", edgecolor="none", label="Pareto Campaign")

        draw_pareto_step(ax, pts, p_idx, color="#08519c")

        # Mark 7 reported optima if at N=168
        if n == 168:
            opt_m = optima_df["measured_modulus"].values
            opt_s = optima_df["measured_stress_relaxation"].values
            ax.scatter(opt_m, opt_s, color="gold", s=110, marker="*", edgecolor="black", lw=1.0, zorder=5, label="7 Reported Optima")

        ax.set_xlim(-0.5, 19.0)
        ax.set_ylim(-2.0, 78.0)
        ax.set_title(f"A{col_idx+1}. Chen Active Learning — {budget_labels[col_idx]}", fontweight="bold", fontsize=10.5)
        ax.set_ylabel(r"Stress Relaxation $\sigma_{100}$ (%)")
        if col_idx == 0:
            ax.set_ylabel(r"Stress Relaxation $\sigma_{100}$ (%)" + "\n" + r"[Chen et al. Active Learning]", fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Stat Tag
        info_text = f"Hypervolume: {hv:.1f}\nTargets Reached: {targ_hits}/4"
        ax.text(0.96, 0.94, info_text, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.8))

        if col_idx == 0:
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)

    # -------------------------------------------------------------------------
    # ROW 2: Space-Filling Design (Sobol Quasi-Random) across 3 budgets
    # -------------------------------------------------------------------------
    for col_idx, n in enumerate(budgets):
        ax = fig.add_subplot(gs[1, col_idx])
        draw_target_windows(ax)
        
        pts = sobol_data[n]["pts"]
        p_idx = sobol_data[n]["p_idx"]
        hv = sobol_data[n]["hv"]
        targ_hits = count_target_hits(pts)

        ax.scatter(pts[:, 0], pts[:, 1], color="#2ca02c", alpha=0.65, s=36, edgecolor="none", label="Sobol Samples (Oracle)")
        draw_pareto_step(ax, pts, p_idx, color="#006d2c")

        ax.set_xlim(-0.5, 19.0)
        ax.set_ylim(-2.0, 78.0)
        ax.set_title(f"B{col_idx+1}. Space-Filling (Sobol) — {budget_labels[col_idx]}", fontweight="bold", fontsize=10.5)
        if col_idx == 0:
            ax.set_ylabel(r"Stress Relaxation $\sigma_{100}$ (%)" + "\n" + r"[Space-Filling / Sobol DOE]", fontweight="bold")
        else:
            ax.set_ylabel(r"Stress Relaxation $\sigma_{100}$ (%)")
        ax.set_xlabel("Compressive Modulus $E$ (kPa)")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Stat Tag
        info_text = f"Hypervolume: {hv:.1f}\nTargets Reached: {targ_hits}/4"
        ax.text(0.96, 0.94, info_text, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.8))

        if col_idx == 0:
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)

    # -------------------------------------------------------------------------
    # ROW 3: Quantitative Synthesis Panels
    # Panel C1: Hypervolume vs Budget Curve
    # Panel C2: Targets Reached vs Budget Curve
    # Panel C3: Formulation Space PCA Exploration (Space-Filling vs AL)
    # -------------------------------------------------------------------------
    eval_budgets = [20, 32, 42, 64, 84, 100, 128, 168]
    
    # Compute trajectories across eval_budgets
    hv_al_curve = []
    hv_sob_curve = []
    hv_lhs_curve = []
    targ_al_curve = []
    targ_sob_curve = []
    targ_lhs_curve = []

    for b in eval_budgets:
        # AL
        sub = points_df.iloc[:b]
        p_al = sub[["modulus", "stress_relaxation"]].values
        hv_al_curve.append(compute_2d_hypervolume(p_al, (0, 0)))
        targ_al_curve.append(count_target_hits(p_al))
        
        # Sobol
        sb_raw = sample_sobol(b, seed=42)
        sb_pts = np.column_stack([gb_mod.predict(sb_raw), gb_sr.predict(sb_raw)])
        hv_sob_curve.append(compute_2d_hypervolume(sb_pts, (0, 0)))
        targ_sob_curve.append(count_target_hits(sb_pts))

        # LHS
        lh_raw = sample_lhs(b, seed=42)
        lh_pts = np.column_stack([gb_mod.predict(lh_raw), gb_sr.predict(lh_raw)])
        hv_lhs_curve.append(compute_2d_hypervolume(lh_pts, (0, 0)))
        targ_lhs_curve.append(count_target_hits(lh_pts))

    # Panel C1: Hypervolume
    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c1.plot(eval_budgets, hv_al_curve, marker="o", color="#1f77b4", lw=2.2, label="Chen Active Learning")
    ax_c1.plot(eval_budgets, hv_sob_curve, marker="s", color="#2ca02c", lw=2.2, linestyle="--", label="Sobol Quasi-Random")
    ax_c1.plot(eval_budgets, hv_lhs_curve, marker="^", color="#d95f02", lw=2.0, linestyle="-.", label="Latin Hypercube (LHS)")
    ax_c1.set_xlabel("Experimental Budget (Sample Count $N$)")
    ax_c1.set_ylabel("2D Pareto Hypervolume ($HV$)")
    ax_c1.set_title("C1. Hypervolume Acquisition vs Budget", fontweight="bold")
    ax_c1.grid(True, linestyle=":", alpha=0.5)
    ax_c1.legend(loc="lower right", fontsize=8)

    # Panel C2: Targets Reached
    ax_c2 = fig.add_subplot(gs[2, 1])
    ax_c2.plot(eval_budgets, targ_al_curve, marker="o", color="#1f77b4", lw=2.2, label="Chen Active Learning")
    ax_c2.plot(eval_budgets, targ_sob_curve, marker="s", color="#2ca02c", lw=2.2, linestyle="--", label="Sobol Quasi-Random")
    ax_c2.plot(eval_budgets, targ_lhs_curve, marker="^", color="#d95f02", lw=2.0, linestyle="-.", label="Latin Hypercube (LHS)")
    ax_c2.set_xlabel("Experimental Budget (Sample Count $N$)")
    ax_c2.set_ylabel("Target Tissues Discovered (Count / 4)")
    ax_c2.set_title("C2. Tissue Targets Hit vs Budget", fontweight="bold")
    ax_c2.set_yticks([0, 1, 2, 3, 4])
    ax_c2.grid(True, linestyle=":", alpha=0.5)
    ax_c2.legend(loc="lower right", fontsize=8)

    # Panel C3: PCA of 5D Formulation Space Exploration (N=168)
    ax_c3 = fig.add_subplot(gs[2, 2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    pca_al = pca.fit_transform(X_scaled)

    # Transform Sobol N=168
    sobol_raw_168 = sobol_data[168]["raw_pts"]
    sobol_scaled = scaler.transform(sobol_raw_168)
    pca_sobol = pca.transform(sobol_scaled)

    ax_c3.scatter(pca_sobol[:, 0], pca_sobol[:, 1], color="#2ca02c", alpha=0.5, s=30, label="Sobol (Uniform Coverage)")
    ax_c3.scatter(pca_al[:, 0], pca_al[:, 1], color="#d73027", alpha=0.8, s=35, marker="^", label="Chen AL (Clustered Rays)")
    ax_c3.set_xlabel(f"Formulation PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_c3.set_ylabel(f"Formulation PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_c3.set_title("C3. 5D Formulation Space: PCA Projection", fontweight="bold")
    ax_c3.grid(True, linestyle=":", alpha=0.5)
    ax_c3.legend(loc="upper right", fontsize=8)

    fig.suptitle("Objective-Space Discovery Across Experimental Budgets: Active Learning vs Space-Filling Designs",
                 fontsize=13.5, fontweight="bold", y=0.995)

    out_pdf = AUDIT_ROOT / "report" / "figures" / "fig_budget_objective_comparison.pdf"
    out_png = AUDIT_ROOT / "report" / "figures" / "fig_budget_objective_comparison.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated: {out_png}")
    print(f"✅ Generated: {out_pdf}")

if __name__ == "__main__":
    create_budget_comparison_figure()
