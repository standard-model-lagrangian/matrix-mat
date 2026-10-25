"""
plot_low_vs_medium_budget_sobol_vs_grid.py

Publication-grade 16:9 presentation slide figure comparing:
  - Author's Crippled 3-Level / Binary Grid Search ("Weird Binary Search")
  - Sobol Quasi-Random Sequence (Best Performing Space-Filling Baseline)
Across two side-by-side budget regimes:
  - Low Budget: N = 25 evaluations (25% total budget)
  - Medium Budget: N = 50 evaluations (50% total budget)

Ultra-clean, minimalist presentation aesthetics with zero clutter and large fonts.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

AUDIT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AUDIT_ROOT / "src"))

from audit.baselines import BOUNDS, check_known_constraint

def get_pareto_envelope(mod_arr, sr_arr):
    """
    Computes empirical 2D trade-off boundary (lower boundary for stress relaxation
    given modulus, representing the stiffest and most elastic gels).
    """
    pts = np.column_stack([mod_arr, sr_arr])
    # Sort by modulus ascending
    order = np.argsort(pts[:, 0])
    s_pts = pts[order]
    pareto = [s_pts[0]]
    for pt in s_pts[1:]:
        if pt[1] < pareto[-1][1]:
            pareto.append(pt)
    return np.array(pareto)

def generate_figure():
    # -------------------------------------------------------------------------
    # 1. Load Canonical Data and Digital Twin Oracle
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2. Generate Author's 3-Level Grid (117 Valid Points)
    # -------------------------------------------------------------------------
    p_grids = [np.linspace(b[0], b[1], 3) for b in BOUNDS]
    mesh = np.meshgrid(*p_grids, indexing="ij")
    grid_all = np.stack([g.flatten() for g in mesh], axis=1)
    grid_valid = np.array([pt for pt in grid_all if check_known_constraint(pt)])
    g_mod_all, g_sr_all = oracle_eval(grid_valid)

    # -------------------------------------------------------------------------
    # 3. Multi-Seed Simulation for Robust Benchmark Stats (N = 100 Seeds)
    # -------------------------------------------------------------------------
    target_E, target_SR = 12.0, 20.0
    tol_E, tol_SR = 1.2, 2.0

    def loss(m, s):
        return np.maximum(np.abs(m - target_E) / tol_E, np.abs(s - target_SR) / tol_SR)

    n_sim_seeds = 100
    g_hits = {25: 0, 50: 0}
    s_hits = {25: 0, 50: 0}
    g_min_losses = {25: [], 50: []}
    s_min_losses = {25: [], 50: []}

    for s_idx in range(n_sim_seeds):
        rng = np.random.default_rng(1000 + s_idx)
        perm = rng.permutation(len(grid_valid))
        for b in [25, 50]:
            sel = perm[:b]
            l_val = np.min(loss(g_mod_all[sel], g_sr_all[sel]))
            g_min_losses[b].append(l_val)
            if l_val <= 1.0:
                g_hits[b] += 1

        sobol_gen = qmc.Sobol(d=5, scramble=True, seed=1000 + s_idx)
        raw = sobol_gen.random(400)
        scaled = BOUNDS[:, 0] + raw * (BOUNDS[:, 1] - BOUNDS[:, 0])
        valid_s = np.array([pt for pt in scaled if check_known_constraint(pt)])
        sm, ss = oracle_eval(valid_s[:50])
        for b in [25, 50]:
            l_val = np.min(loss(sm[:b], ss[:b]))
            s_min_losses[b].append(l_val)
            if l_val <= 1.0:
                s_hits[b] += 1

    # Representative seed for plotting clean scatter points
    rep_seed = 2026
    rng_rep = np.random.default_rng(rep_seed)
    perm_rep = rng_rep.permutation(len(grid_valid))
    g_pts_25 = grid_valid[perm_rep[:25]]
    g_pts_50 = grid_valid[perm_rep[:50]]
    gm_25, gs_25 = oracle_eval(g_pts_25)
    gm_50, gs_50 = oracle_eval(g_pts_50)

    sobol_rep = qmc.Sobol(d=5, scramble=True, seed=rep_seed)
    raw_rep = sobol_rep.random(400)
    scaled_rep = BOUNDS[:, 0] + raw_rep * (BOUNDS[:, 1] - BOUNDS[:, 0])
    valid_s_rep = np.array([pt for pt in scaled_rep if check_known_constraint(pt)])
    s_pts_25 = valid_s_rep[:25]
    s_pts_50 = valid_s_rep[:50]
    sm_25, ss_25 = oracle_eval(s_pts_25)
    sm_50, ss_50 = oracle_eval(s_pts_50)

    # -------------------------------------------------------------------------
    # 4. Create 16:9 Presentation Figure
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#f8fafc")

    c_navy = "#0f172a"
    c_gray = "#64748b"
    c_grid = "#9333ea"    # Purple for author's crippled binary grid
    c_sobol = "#0d9488"   # Teal for Sobol space-filling
    c_border = "#cbd5e1"

    # Title Banner
    fig.text(0.5, 0.950, "Side-by-Side Benchmark: Author's Binary Grid vs. Sobol Space-Filling",
             fontsize=18, fontweight="bold", ha="center", va="center", color=c_navy)
    fig.text(0.5, 0.920, "Physical Property Space ($E$ vs $\\sigma_{100}$) Exploration at Low ($N = 25$) and Medium ($N = 50$) Experimental Budgets",
             fontsize=11, ha="center", va="center", color=c_gray)

    # Target Windows Definition
    target_boxes = [
        {"name": "Scarred Skin\n(Primary Target)", "mod": (10.8, 13.2), "sr": (18.0, 22.0),
         "fc": "#fef08a", "ec": "#b45309", "lw": 2.2, "zorder": 2, "label_pos": (12.0, 25.0)},
        {"name": "Healthy Skin", "mod": (1.35, 1.65), "sr": (22.5, 27.5),
         "fc": "#dcfce7", "ec": "#15803d", "lw": 1.4, "zorder": 1, "label_pos": (1.5, 30.5)},
        {"name": "Soft Tumor", "mod": (1.17, 1.43), "sr": (45.0, 55.0),
         "fc": "#f3e8ff", "ec": "#7e22ce", "lw": 1.4, "zorder": 1, "label_pos": (1.3, 58.0)},
        {"name": "Stiff Tumor", "mod": (14.4, 17.6), "sr": (10.0, 14.0),
         "fc": "#fee2e2", "ec": "#b91c1c", "lw": 1.4, "zorder": 1, "label_pos": (16.0, 16.5)},
    ]

    # Two side-by-side subplots
    ax_left = fig.add_axes([0.06, 0.10, 0.42, 0.77])
    ax_right = fig.add_axes([0.53, 0.10, 0.42, 0.77])

    panels = [
        {
            "ax": ax_left,
            "title": "A. Low Budget: $N = 25$ Formulations (25% Budget)",
            "subtitle": "Author's binary search trapped at discrete corners; Sobol spans the entire trade-off",
            "gm": gm_25, "gs": gs_25,
            "sm": sm_25, "ss": ss_25,
            "b": 25,
            "g_hit_rate": g_hits[25] / n_sim_seeds * 100,
            "s_hit_rate": s_hits[25] / n_sim_seeds * 100,
            "g_med_loss": np.median(g_min_losses[25]),
            "s_med_loss": np.median(s_min_losses[25]),
            "callout_type": "left"
        },
        {
            "ax": ax_right,
            "title": "B. Medium Budget: $N = 50$ Formulations (50% Budget)",
            "subtitle": "Sobol blankets the Pareto front & lands in target; Grid leaves persistent empty voids",
            "gm": gm_50, "gs": gs_50,
            "sm": sm_50, "ss": ss_50,
            "b": 50,
            "g_hit_rate": g_hits[50] / n_sim_seeds * 100,
            "s_hit_rate": s_hits[50] / n_sim_seeds * 100,
            "g_med_loss": np.median(g_min_losses[50]),
            "s_med_loss": np.median(s_min_losses[50]),
            "callout_type": "right"
        }
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_facecolor("#ffffff")

        # Draw target windows
        for tw in target_boxes:
            m_low, m_high = tw["mod"]
            s_low, s_high = tw["sr"]
            rect = patches.Rectangle((m_low, s_low), m_high - m_low, s_high - s_low,
                                     fc=tw["fc"], ec=tw["ec"], lw=tw["lw"], ls="--",
                                     alpha=0.65, zorder=tw["zorder"])
            ax.add_patch(rect)
            ax.text(tw["label_pos"][0], tw["label_pos"][1], tw["name"],
                    fontsize=8.0, fontweight="bold", color=tw["ec"], ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=tw["ec"], lw=0.8, alpha=0.92),
                    zorder=tw["zorder"] + 3)

        # Plot Pareto envelope steps
        p_sobol = get_pareto_envelope(p["sm"], p["ss"])
        p_grid = get_pareto_envelope(p["gm"], p["gs"])

        ax.step(p_sobol[:, 0], p_sobol[:, 1], where="post", color=c_sobol, lw=1.8, ls="-", alpha=0.5, zorder=3)
        ax.step(p_grid[:, 0], p_grid[:, 1], where="post", color=c_grid, lw=1.8, ls="--", alpha=0.5, zorder=3)

        # Plot Author's Crippled Grid Points
        ax.scatter(p["gm"], p["gs"], color=c_grid, s=80, marker="s", edgecolors="#581c87", lw=1.4,
                   alpha=0.85, label=f"Author's 3-Level Grid (Binary aCNC) [N={p['b']}]", zorder=5)

        # Plot Sobol Quasi-Random Points
        ax.scatter(p["sm"], p["ss"], color=c_sobol, s=65, marker="o", edgecolors="#0f766e", lw=1.2,
                   alpha=0.85, label=f"Sobol Quasi-Random (Space-Filling) [N={p['b']}]", zorder=6)

        # Axis formatting
        ax.set_xlim(-0.5, 18.5)
        ax.set_ylim(-2.0, 75.0)
        ax.set_xlabel("Compressive Modulus $E$ (kPa)", fontsize=10.5, fontweight="bold", color=c_navy)
        ax.set_ylabel(r"Stress Relaxation $\sigma_{100}$ (%)", fontsize=10.5, fontweight="bold", color=c_navy)
        ax.set_title(p["title"], fontsize=12.0, fontweight="bold", color=c_navy, pad=10)
        ax.grid(True, ls=":", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(c_border)

        # Subtitle banner inside plot at top
        ax.text(0.5, 0.965, p["subtitle"], transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=c_gray,
                bbox=dict(boxstyle="round,pad=0.25", fc="#f8fafc", ec="#e2e8f0", lw=0.8), zorder=8)

        # Stat Card Box (Upper Right where no data points exist: E in [12, 18], SR in [45, 75])
        stat_card = FancyBboxPatch((0.55, 0.63), 0.42, 0.30,
                                   transform=ax.transAxes,
                                   boxstyle="round,pad=0.02,rounding_size=0.03",
                                   fc="#f8fafc", ec=c_border, lw=1.2, zorder=7)
        ax.add_patch(stat_card)

        # Title of stat card
        ax.text(0.760, 0.890, f"Benchmark Performance (N=100 Seeds)", transform=ax.transAxes,
                fontsize=8.2, fontweight="bold", ha="center", va="center", color=c_navy, zorder=8)

        # Metrics rows
        ax.text(0.57, 0.825, "Metric", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=c_gray, zorder=8)
        ax.text(0.80, 0.825, "Sobol", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=c_sobol, ha="center", zorder=8)
        ax.text(0.92, 0.825, "Grid", transform=ax.transAxes, fontsize=7.5, fontweight="bold", color=c_grid, ha="center", zorder=8)

        # Row 1: Scarred Skin Hit Rate
        ax.text(0.57, 0.765, "Target Hit Rate:", transform=ax.transAxes, fontsize=7.8, fontweight="bold", color=c_navy, zorder=8)
        ax.text(0.80, 0.765, f"{p['s_hit_rate']:.0f}%", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=c_sobol, ha="center", zorder=8)
        ax.text(0.92, 0.765, f"{p['g_hit_rate']:.0f}%", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=c_grid, ha="center", zorder=8)

        # Row 2: Median Target Loss
        ax.text(0.57, 0.710, "Median Min Loss:", transform=ax.transAxes, fontsize=7.8, fontweight="bold", color=c_navy, zorder=8)
        ax.text(0.80, 0.710, f"{p['s_med_loss']:.2f}", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=c_sobol, ha="center", zorder=8)
        ax.text(0.92, 0.710, f"{p['g_med_loss']:.2f}", transform=ax.transAxes, fontsize=8.0, fontweight="bold", color=c_grid, ha="center", zorder=8)

        # Row 3: Parameter coverage
        if p["b"] == 25:
            ax.text(0.760, 0.655, "aCNC: Sobol = Continuous  |  Grid = Binary {0, 3}%",
                    transform=ax.transAxes, fontsize=6.8, fontweight="bold", ha="center", va="center", color="#b45309", zorder=8)
        else:
            ax.text(0.760, 0.655, "Grid evaluates 43% of entire universe (50/117 pts)!",
                    transform=ax.transAxes, fontsize=6.8, fontweight="bold", ha="center", va="center", color="#b91c1c", zorder=8)

        # Annotations / Callouts
        if p["callout_type"] == "left":
            # Callout to empty void between 3 and 6 kPa
            ax.annotate("Vast Property Void!\nNo grid formulations between 3–6 kPa",
                        xy=(4.5, 30.0), xytext=(4.2, 40.0),
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
                        fontsize=8.0, fontweight="bold", color="#dc2626",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#fca5a5", lw=0.8),
                        zorder=9)
            # Callout to Sobol points along trade-off
            ax.annotate("Sobol outlines full trade-off envelope\nin just 25 experiments",
                        xy=(7.2, 19.5), xytext=(4.0, 7.5),
                        arrowprops=dict(arrowstyle="->", color=c_sobol, lw=1.5),
                        fontsize=8.0, fontweight="bold", color=c_sobol,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f0fdfa", ec="#99f6e4", lw=0.8),
                        zorder=9)
        else:
            # Callout to Sobol landing directly in Scarred Skin
            ax.annotate("Sobol lands squarely inside\nScarred Skin target window!",
                        xy=(12.0, 19.5), xytext=(9.2, 8.5),
                        arrowprops=dict(arrowstyle="->", color=c_sobol, lw=1.5),
                        fontsize=8.0, fontweight="bold", color="#15803d",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f0fdf4", ec="#86efac", lw=0.8),
                        zorder=9)
            # Callout to persistent grid clustering
            ax.annotate("Grid adds 25 more points, but STILL\ntrapped in same discrete clusters!",
                        xy=(3.4, 56.5), xytext=(1.8, 64.0),
                        arrowprops=dict(arrowstyle="->", color=c_grid, lw=1.5),
                        fontsize=8.0, fontweight="bold", color=c_grid,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#faf5ff", ec="#d8b4fe", lw=0.8),
                        zorder=9)

        # Legend at bottom left
        ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92, edgecolor=c_border)

    # Save figure
    out_png = AUDIT_ROOT / "report" / "figures" / "fig_low_vs_medium_budget_sobol_vs_grid.png"
    out_pdf = AUDIT_ROOT / "report" / "figures" / "fig_low_vs_medium_budget_sobol_vs_grid.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated: {out_png}")
    print(f"✅ Generated: {out_pdf}")

if __name__ == "__main__":
    generate_figure()
