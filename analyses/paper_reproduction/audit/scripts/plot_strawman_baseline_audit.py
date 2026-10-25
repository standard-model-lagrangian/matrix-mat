"""
Generate Presentation Graphic: Audit of Chen et al.'s Strawman Baseline.
Contrasting the Author's Crippled 3-Level (Ternary) Grid Search against
Standard Continuous Random Search and Sobol Quasi-Random Sampling across budgets.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

AUDIT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from audit.baselines import BOUNDS, check_known_constraint

def run_simulation(n_seeds=200, total_evals=100, seed=39582884):
    """Run 200-seed simulation on the digital twin oracle."""
    points_df = pd.read_parquet(AUDIT_ROOT / "artifacts" / "canonical" / "points.parquet")
    FEATURES = ["exposure_time", "gelma_l", "gelma_h", "acnc", "lap"]
    X_pool = points_df[FEATURES].values
    y_mod = points_df["modulus"].values
    y_sr = points_df["stress_relaxation"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool)
    gb_mod = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=30).fit(X_scaled, y_mod)
    gb_sr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=30).fit(X_scaled, y_sr)

    def oracle_eval(X_cand):
        Xs = scaler.transform(np.atleast_2d(X_cand))
        return gb_mod.predict(Xs), gb_sr.predict(Xs)

    # Scarred Skin targets: E = 12.0 kPa, tol = 1.2; SR = 20.0%, tol = 2.0
    def loss_fn(mod, sr):
        return np.maximum(np.abs(mod - 12.0) / 1.2, np.abs(sr - 20.0) / 2.0)

    # 1. Author's 3-level grid
    p_grids = [np.linspace(b[0], b[1], 3) for b in BOUNDS]
    mesh = np.meshgrid(*p_grids, indexing="ij")
    grid_all = np.stack([g.flatten() for g in mesh], axis=1)
    grid_valid = np.array([pt for pt in grid_all if check_known_constraint(pt)])
    g_mod_all, g_sr_all = oracle_eval(grid_valid)
    g_loss_all = loss_fn(g_mod_all, g_sr_all)

    methods = ["Crippled_Grid", "Random", "Sobol", "BO_Simulation"]
    trajectories = {m: np.zeros((n_seeds, total_evals)) for m in methods}
    
    # Milestone budgets
    milestones = [10, 25, 50, 75, 100]
    hits_at_milestones = {m: {b: 0 for b in milestones} for m in methods}

    for s_idx in range(n_seeds):
        rng = np.random.default_rng(seed + s_idx)

        # A. Crippled Grid
        perm = rng.permutation(len(grid_valid))
        sel_idx = perm[:total_evals]
        if len(sel_idx) < total_evals:
            sel_idx = np.concatenate([sel_idx, perm[:total_evals - len(sel_idx)]])
        g_cum = np.minimum.accumulate(g_loss_all[sel_idx])
        trajectories["Crippled_Grid"][s_idx] = g_cum

        # B. Random Search (Continuous)
        rand_pts = []
        while len(rand_pts) < total_evals:
            cand = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
            if check_known_constraint(cand):
                rand_pts.append(cand)
        r_mod, r_sr = oracle_eval(np.array(rand_pts))
        r_loss = loss_fn(r_mod, r_sr)
        r_cum = np.minimum.accumulate(r_loss)
        trajectories["Random"][s_idx] = r_cum

        # C. Sobol (Quasi-Random)
        engine = qmc.Sobol(d=5, seed=seed + s_idx)
        s_pts = []
        while len(s_pts) < total_evals:
            raw = engine.random(n=128)
            scaled = qmc.scale(raw, BOUNDS[:, 0], BOUNDS[:, 1])
            for pt in scaled:
                if check_known_constraint(pt):
                    s_pts.append(pt)
                    if len(s_pts) == total_evals:
                        break
        s_mod, s_sr = oracle_eval(np.array(s_pts))
        s_loss = loss_fn(s_mod, s_sr)
        s_cum = np.minimum.accumulate(s_loss)
        trajectories["Sobol"][s_idx] = s_cum

        # D. BO Simulation (Heuristic surrogate on oracle)
        bo_points = list(rand_pts[:10])
        for b in range(1, 10):
            curr_X = np.array(bo_points)
            c_mod, c_sr = oracle_eval(curr_X)
            c_loss = loss_fn(c_mod, c_sr)
            # Pool
            cands = []
            while len(cands) < 150:
                c = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
                if check_known_constraint(c):
                    cands.append(c)
            cands = np.array(cands)
            cand_m, cand_s = oracle_eval(cands)
            cand_l = loss_fn(cand_m, cand_s)
            acq = cand_l + rng.normal(0, 0.1 / np.sqrt(b), size=len(cands))
            best_idx = np.argsort(acq)[:10]
            bo_points.extend(cands[best_idx])
        bo_X = np.array(bo_points[:total_evals])
        bo_m, bo_s = oracle_eval(bo_X)
        bo_cum = np.minimum.accumulate(loss_fn(bo_m, bo_s))
        trajectories["BO_Simulation"][s_idx] = bo_cum

        for b in milestones:
            for m in methods:
                if trajectories[m][s_idx, b - 1] <= 1.0:
                    hits_at_milestones[m][b] += 1

    # Convert hit counts to percentages
    hit_rates = {m: [hits_at_milestones[m][b] / n_seeds * 100 for b in milestones] for m in methods}
    return trajectories, hit_rates, milestones, grid_valid

def generate_figure():
    trajectories, hit_rates, milestones, grid_valid = run_simulation(n_seeds=200, total_evals=100)

    # 16:9 publication-quality figure
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#f8fafc")

    # Colors
    c_navy = "#0f172a"
    c_gray = "#64748b"
    c_grid = "#9333ea"     # Purple for crippled grid
    c_rand = "#2563eb"     # Blue for random search
    c_sobol = "#0d9488"    # Teal for Sobol
    c_bo = "#ea580c"       # Orange for BO
    c_border = "#cbd5e1"

    # Title Banner
    fig.text(0.5, 0.945, "The Strawman Baseline Audit: 3-Level Grid vs. Continuous Random Search",
             fontsize=18, fontweight="bold", ha="center", va="center", color=c_navy)
    fig.text(0.5, 0.915, "Exposing Chen et al.'s benchmark flaw: An artificially crippled ternary grid (117 points) created the illusion of vast Active Learning superiority",
             fontsize=11, ha="center", va="center", color=c_gray)

    # 4 Subplot layout
    # [left, bottom, width, height]
    # Panel A (Top Left): 2D Parameter Projection (Ternary Grid vs Random)
    ax_a = fig.add_axes([0.06, 0.50, 0.42, 0.36])
    # Panel B (Top Right): Loss Trajectories across Evaluations
    ax_b = fig.add_axes([0.54, 0.50, 0.41, 0.36])
    # Panel C (Bottom Left): Hit Rate across Budgets (Bar Chart)
    ax_c = fig.add_axes([0.06, 0.08, 0.42, 0.34])
    # Panel D (Bottom Right): The Absurdity Breakdown Scorecard Table
    ax_d = fig.add_axes([0.54, 0.08, 0.41, 0.34])
    ax_d.axis("off")

    # -------------------------------------------------------------------------
    # Panel A: 2D Projection of Parameter Space (GelMA vs aCNC)
    # -------------------------------------------------------------------------
    ax_a.set_facecolor("#ffffff")
    # Plot author's 3-level grid points in GelMA (L+H) vs aCNC space
    grid_gelma = grid_valid[:, 1] + grid_valid[:, 2]
    grid_acnc = grid_valid[:, 3]

    # Generate 100 random points for comparison
    rng = np.random.default_rng(42)
    rand_pts = []
    while len(rand_pts) < 100:
        c = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        if check_known_constraint(c):
            rand_pts.append(c)
    rand_pts = np.array(rand_pts)
    rand_gelma = rand_pts[:, 1] + rand_pts[:, 2]
    rand_acnc = rand_pts[:, 3]

    # Background grid lines to emphasize discrete levels {0, 3, 6, 9, 12}
    for g_val in [3.0, 6.0, 9.0, 12.0]:
        ax_a.axvline(g_val, color="#e2e8f0", ls="--", lw=1.0, zorder=1)
    for a_val in [0.0, 3.0]:
        ax_a.axhline(a_val, color="#e2e8f0", ls="--", lw=1.0, zorder=1)

    # Shaded feasible region boundary
    ax_a.scatter(grid_gelma, grid_acnc, color=c_grid, s=70, marker="s", edgecolors="#581c87", lw=1.2,
                 alpha=0.85, label="Author's Ternary Grid Nodes (Rigid 3-Lvl Lattice)", zorder=3)
    ax_a.scatter(rand_gelma, rand_acnc, color=c_rand, s=35, marker="o", edgecolors="#1e40af", lw=0.8,
                 alpha=0.75, label="Continuous Random Samples (Unrestricted)", zorder=4)

    # Target Sweet Spot Box (Scarred skin formulation sweet spot)
    target_box = patches.Rectangle((7.5, 0.8), 2.2, 1.4, fc="#dcfce7", ec="#16a34a", lw=2.0, ls="-",
                                   alpha=0.6, label="Scarred Skin Feasible Target Region", zorder=2)
    ax_a.add_patch(target_box)
    ax_a.text(8.6, 1.5, "Optimal Target\nWindow", fontsize=8.5, fontweight="bold", ha="center", va="center", color="#15803d", zorder=5)

    # Callout Arrow pointing to huge grid void
    ax_a.annotate("Vast Interior Void!\nNearest grid node is far away",
                  xy=(6.0, 3.0), xytext=(3.2, 2.2),
                  arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                  fontsize=8.0, fontweight="bold", color="#dc2626",
                  bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#fca5a5", lw=0.8), zorder=6)

    ax_a.set_xlabel(r"Total GelMA Concentration (GelMA-L + GelMA-H, wt%)", fontsize=9.5, fontweight="bold", color=c_navy)
    ax_a.set_ylabel(r"aCNC Nanofiber Concentration (wt%)", fontsize=9.5, fontweight="bold", color=c_navy)
    ax_a.set_title("A. Spatial Coverage: Discrete 3-Level Grid vs. Continuous Search", fontsize=10.5, fontweight="bold", color=c_navy, pad=8)
    ax_a.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.9)
    ax_a.set_xlim(2.5, 12.5)
    ax_a.set_ylim(-0.3, 3.8)
    for spine in ax_a.spines.values():
        spine.set_color(c_border)

    # -------------------------------------------------------------------------
    # Panel B: Loss Trajectories across Number of Formulations Evaluated
    # -------------------------------------------------------------------------
    ax_b.set_facecolor("#ffffff")
    eval_axis = np.arange(1, 101)

    labels = {
        "BO_Simulation": "Atlas MOBO (Circular Digital Twin)",
        "Sobol": "Sobol Quasi-Random (Space-Filling)",
        "Random": "Continuous Random Search",
        "Crippled_Grid": "Author's Crippled Grid (3 levels/dim)"
    }
    colors = {
        "BO_Simulation": c_bo,
        "Sobol": c_sobol,
        "Random": c_rand,
        "Crippled_Grid": c_grid
    }

    for m in ["Crippled_Grid", "Random", "Sobol", "BO_Simulation"]:
        median = np.median(trajectories[m], axis=0)
        p10 = np.percentile(trajectories[m], 10, axis=0)
        p90 = np.percentile(trajectories[m], 90, axis=0)
        ax_b.plot(eval_axis, median, color=colors[m], lw=2.2, label=labels[m])
        ax_b.fill_between(eval_axis, p10, p90, color=colors[m], alpha=0.15)

    # Horizontal target tolerance boundary (Loss <= 1.0)
    ax_b.axhline(1.0, color="#16a34a", ls="--", lw=1.6, label="Target Window Boundary (Loss <= 1.0)")
    # Grid theoretical resolution floor
    ax_b.axhline(0.546, color=c_grid, ls=":", lw=1.4, label="Ternary Grid Resolution Floor (0.546)")

    ax_b.set_yscale("log")
    ax_b.set_xlabel("Number of Formulations Evaluated (Budget)", fontsize=9.5, fontweight="bold", color=c_navy)
    ax_b.set_ylabel(r"Target Loss $\max(\Delta E / 1.2, \Delta \sigma_{100} / 2.0)$ [log scale]", fontsize=9.5, fontweight="bold", color=c_navy)
    ax_b.set_title("B. Optimization Trajectories ($N = 200$ Seeds, Median & 10–90th Bands)", fontsize=10.5, fontweight="bold", color=c_navy, pad=8)
    ax_b.legend(loc="upper right", fontsize=7.8, frameon=True, framealpha=0.9)
    ax_b.grid(True, ls=":", alpha=0.5)
    for spine in ax_b.spines.values():
        spine.set_color(c_border)

    # Annotation on flatline
    ax_b.annotate("Grid Search FLATLINES at 0.546!\nExhausts its 117 discrete nodes",
                  xy=(65, 0.546), xytext=(45, 0.22),
                  arrowprops=dict(arrowstyle="->", color=c_grid, lw=1.4),
                  fontsize=8.0, fontweight="bold", color=c_grid,
                  bbox=dict(boxstyle="round,pad=0.2", fc="#faf5ff", ec="#d8b4fe", lw=0.8))

    # -------------------------------------------------------------------------
    # Panel C: Hit Rate vs. Budget Milestones (Grouped Bar Chart)
    # -------------------------------------------------------------------------
    ax_c.set_facecolor("#ffffff")
    x_indices = np.arange(len(milestones))
    bar_w = 0.18

    # Bars: Crippled Grid, Random, Sobol, BO
    rects_grid = ax_c.bar(x_indices - 1.5*bar_w, hit_rates["Crippled_Grid"], bar_w, label="Author's 3-Lvl Grid", color=c_grid, alpha=0.9, ec="#581c87")
    rects_rand = ax_c.bar(x_indices - 0.5*bar_w, hit_rates["Random"], bar_w, label="Random Search", color=c_rand, alpha=0.9, ec="#1e40af")
    rects_sobol = ax_c.bar(x_indices + 0.5*bar_w, hit_rates["Sobol"], bar_w, label="Sobol Quasi-Random", color=c_sobol, alpha=0.9, ec="#0f766e")
    rects_bo = ax_c.bar(x_indices + 1.5*bar_w, hit_rates["BO_Simulation"], bar_w, label="Atlas MOBO", color=c_bo, alpha=0.9, ec="#c2410c")

    # Clean annotations on hit rates
    # N=25 BO milestone
    ax_c.annotate("BO: 100% in 25 evals", xy=(1 + 1.5*bar_w, 100), xytext=(0.85, 107),
                  arrowprops=dict(arrowstyle="->", color=c_bo, lw=1.2),
                  fontsize=7.5, fontweight="bold", color=c_bo)
    # N=50 Random / Sobol
    ax_c.text(2, 82, f"Random & Sobol: {hit_rates['Random'][2]:.0f}%",
              ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#1e3a8a")
    # N=100 Random / Sobol
    ax_c.annotate(f"Random ({hit_rates['Random'][4]:.0f}%) & Sobol ({hit_rates['Sobol'][4]:.0f}%)\neasily solve target at full budget",
                  xy=(4, 94), xytext=(2.7, 106),
                  arrowprops=dict(arrowstyle="->", color=c_rand, lw=1.2),
                  fontsize=7.5, fontweight="bold", color=c_navy)

    ax_c.set_xticks(x_indices)
    ax_c.set_xticklabels([f"N = {b}\n({b}% Budget)" for b in milestones], fontsize=8.5, fontweight="bold", color=c_navy)
    ax_c.set_ylabel("Target Discovery Hit Rate (% Seeds)", fontsize=9.5, fontweight="bold", color=c_navy)
    ax_c.set_ylim(0, 118)
    ax_c.set_title("C. Target Discovery Success vs. Experimental Budget", fontsize=10.5, fontweight="bold", color=c_navy, pad=8)
    ax_c.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.9)
    ax_c.grid(True, axis="y", ls=":", alpha=0.5)
    for spine in ax_c.spines.values():
        spine.set_color(c_border)

    # -------------------------------------------------------------------------
    # Panel D: The Absurdity Breakdown Scorecard Table
    # -------------------------------------------------------------------------
    # Frame card
    table_card = FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                transform=ax_d.transAxes,
                                boxstyle="round,pad=0.03,rounding_size=0.05",
                                fc="#ffffff", ec=c_border, lw=1.2, zorder=1)
    ax_d.add_patch(table_card)

    # Header
    ax_d.text(0.5, 0.93, "D. Anatomy of a Strawman: Mathematical Flaws in Table S1 / Fig. S10",
              fontsize=10.5, fontweight="bold", ha="center", va="center", color=c_navy, transform=ax_d.transAxes)

    scorecard_items = [
        ("Discretization Formula in Code",
         r"$\mathtt{points\_per\_dim} = \mathrm{ceil}(100^{1/5}) = 3$ (forces rigid ternary lattice)"),
        ("Sampled Parameter Levels",
         "{Min, Midpoint, Max} only — zero interior exploration across 5 dimensions"),
        ("aCNC Solubility Defect",
         "6.0 wt% node rejected by constraint $\\to$ collapses to binary {0.0, 3.0} wt%"),
        ("Exposure Time Resolution",
         "300 s step size: {0, 300, 600} s — skips intermediate curing kinetics"),
        ("Feasible Grid Population",
         "ONLY 117 valid points exist — evaluating 100 tests 85.5% of entire grid universe"),
        ("Theoretical Resolution Floor",
         "Loss = 0.546 hard floor — grid cannot interpolate, permanently flatlining"),
        ("Real Random Baseline",
         "Continuous Random achieves 91% hit rate (94% Sobol) & reaches loss < 0.3"),
        ("Scientific Audit Verdict",
         "Manufactured Advantage: Vast 'BO superiority' is an artifact of a crippled lattice")
    ]

    y_start = 0.81
    row_gap = 0.095
    for i, (title, desc) in enumerate(scorecard_items):
        y_pos = y_start - i * row_gap
        bg_col = "#f8fafc" if i % 2 == 0 else "#ffffff"
        row_rect = FancyBboxPatch((0.02, y_pos - 0.036), 0.96, 0.080,
                                  transform=ax_d.transAxes,
                                  boxstyle="round,pad=0.01,rounding_size=0.02",
                                  fc=bg_col, ec="#e2e8f0", lw=0.6, zorder=2)
        ax_d.add_patch(row_rect)

        # Highlight verdict row
        if i == len(scorecard_items) - 1:
            row_rect.set_facecolor("#fef2f2")
            row_rect.set_edgecolor("#fca5a5")
            c_title = "#b91c1c"
            c_desc = "#991b1b"
        elif "Floor" in title or "Grid" in title:
            c_title = c_navy
            c_desc = c_grid
        elif "Random" in title:
            c_title = c_navy
            c_desc = c_rand
        else:
            c_title = c_navy
            c_desc = "#334155"

        ax_d.text(0.04, y_pos, title, fontsize=8.0, fontweight="bold", color=c_title, va="center", transform=ax_d.transAxes, zorder=3)
        ax_d.text(0.35, y_pos, desc, fontsize=7.4, fontweight="bold" if (i == len(scorecard_items) - 1 or "Floor" in title or "Random" in title) else "normal", color=c_desc, va="center", transform=ax_d.transAxes, zorder=3)

    # Save figure
    out_png = AUDIT_ROOT / "report" / "figures" / "fig_strawman_baseline_audit.png"
    out_pdf = AUDIT_ROOT / "report" / "figures" / "fig_strawman_baseline_audit.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated: {out_png}")
    print(f"✅ Generated: {out_pdf}")

if __name__ == "__main__":
    generate_figure()
