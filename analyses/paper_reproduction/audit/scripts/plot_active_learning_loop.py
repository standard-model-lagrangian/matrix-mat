"""
Generate Presentation Graphic: Formal Active Learning & Bayesian Optimization Loop.
Designed for 16:9 slide presentation with publication-quality typography,
embedded mini-plots, mathematical notation, and biomaterials context.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle

AUDIT_ROOT = Path(__file__).resolve().parent.parent

def generate_active_learning_diagram():
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Typography & Colors
    c_bg_card = "#ffffff"
    c_border_card = "#cbd5e1"
    c_navy = "#0f172a"
    c_blue = "#2563eb"
    c_teal = "#0d9488"
    c_orange = "#ea580c"
    c_purple = "#7c3aed"
    c_green = "#16a34a"
    c_red = "#dc2626"
    c_gray_txt = "#475569"
    c_light_blue = "#eff6ff"
    c_light_orange = "#fff7ed"
    c_light_green = "#f0fdf4"

    # Title Banner
    ax.text(8.0, 8.45, "The Active Learning & Bayesian Optimization Framework",
            fontsize=19, fontweight="bold", ha="center", va="center", color=c_navy)
    ax.text(8.0, 8.05, r"Iterative closed-loop discovery: sequential surrogate modeling, acquisition guidance, and physical feedback",
            fontsize=11.5, ha="center", va="center", color=c_gray_txt)

    # 5 Step Cards coordinates
    card_w = 2.75
    card_h = 4.8
    y_card = 2.7
    x_cards = [0.65, 3.75, 6.85, 9.95, 13.05]

    steps_meta = [
        {
            "num": "1",
            "title": "Dataset Buffer",
            "formula": r"$\mathcal{D}_t = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^t$",
            "color": c_blue,
            "bg_color": "#f8fafc"
        },
        {
            "num": "2",
            "title": "Surrogate Model",
            "formula": r"$f(\mathbf{x}) \sim \mathcal{GP}(\mu_t, \sigma_t^2)$",
            "color": c_teal,
            "bg_color": "#f0fdfa"
        },
        {
            "num": "3",
            "title": "Acquisition Surface",
            "formula": r"$\alpha_t(\mathbf{x}) = \mathrm{Acq}(\mu_t, \sigma_t)$",
            "color": c_orange,
            "bg_color": "#fff7ed"
        },
        {
            "num": "4",
            "title": "Decision Policy",
            "formula": r"$\mathbf{x}_{t+1} = \arg\max_{\mathbf{x}} \alpha_t(\mathbf{x})$",
            "color": c_purple,
            "bg_color": "#faf5ff"
        },
        {
            "num": "5",
            "title": "Physical Synthesis",
            "formula": r"$\mathbf{y}_{t+1} = \mathbf{f}_{\mathrm{true}}(\mathbf{x}_{t+1}) + \boldsymbol{\epsilon}$",
            "color": c_green,
            "bg_color": "#f0fdf4"
        }
    ]

    # Draw step cards
    for idx, (xc, meta) in enumerate(zip(x_cards, steps_meta)):
        # Card shadow & body
        shadow = FancyBboxPatch((xc + 0.04, y_card - 0.04), card_w, card_h,
                                boxstyle="round,pad=0.08,rounding_size=0.18",
                                fc="#e2e8f0", ec="none", zorder=1)
        ax.add_patch(shadow)

        card = FancyBboxPatch((xc, y_card), card_w, card_h,
                              boxstyle="round,pad=0.08,rounding_size=0.18",
                              fc=c_bg_card, ec=c_border_card, lw=1.3, zorder=2)
        ax.add_patch(card)

        # Header accent bar
        header_bar = FancyBboxPatch((xc, y_card + card_h - 0.85), card_w, 0.85,
                                    boxstyle="round,pad=0.05,rounding_size=0.18",
                                    fc=meta["bg_color"], ec=meta["color"], lw=1.2, zorder=3)
        ax.add_patch(header_bar)

        # Step Number Badge
        circle_badge = patches.Circle((xc + 0.42, y_card + card_h - 0.42), 0.26,
                                      fc=meta["color"], ec="none", zorder=4)
        ax.add_patch(circle_badge)
        ax.text(xc + 0.42, y_card + card_h - 0.42, meta["num"],
                color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

        # Step Title & Math
        ax.text(xc + 0.82, y_card + card_h - 0.30, meta["title"],
                fontsize=11.5, fontweight="bold", ha="left", va="center", color=c_navy, zorder=5)
        ax.text(xc + 0.82, y_card + card_h - 0.58, meta["formula"],
                fontsize=9.5, ha="left", va="center", color=meta["color"], zorder=5)

    # Connecting Arrows between cards (Row 1)
    for i in range(4):
        x_start = x_cards[i] + card_w + 0.05
        x_end = x_cards[i+1] - 0.05
        y_arrow = y_card + card_h / 2.0
        arrow = FancyArrowPatch((x_start, y_arrow), (x_end, y_arrow),
                                arrowstyle="-|>", mutation_scale=16,
                                lw=2.2, color="#94a3b8", zorder=3)
        ax.add_patch(arrow)

    # -------------------------------------------------------------------------
    # Card 1 Internal Graphic: Data Table & 5D Formulation Vector
    # -------------------------------------------------------------------------
    xc1 = x_cards[0]
    ax.text(xc1 + card_w/2, y_card + 3.75, "Formulation & Response Log",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Mini Table layout
    table_x = xc1 + 0.18
    table_y = y_card + 1.35
    col_w = [1.38, 1.01]
    row_h = 0.44

    headers = [r"Inputs ($\mathbf{x} \in \mathcal{X}$)", r"Outputs ($\mathbf{y}$)"]
    ax.fill_between([table_x, table_x + col_w[0] + col_w[1]], [table_y + 4*row_h, table_y + 4*row_h],
                    [table_y + 3*row_h, table_y + 3*row_h], color="#e2e8f0", zorder=3)
    ax.text(table_x + col_w[0]/2, table_y + 3.5*row_h, headers[0], fontsize=7.8, fontweight="bold", ha="center", va="center", color=c_navy, zorder=4)
    ax.text(table_x + col_w[0] + col_w[1]/2, table_y + 3.5*row_h, headers[1], fontsize=7.8, fontweight="bold", ha="center", va="center", color=c_navy, zorder=4)

    sample_rows = [
        ("GelMA-L, -H (wt%)", "E = 1.5 kPa"),
        ("aCNC Nanofibers", r"$\sigma_{100} = 25\%$"),
        ("LAP, Exposure Time", "Target Match")
    ]
    for r_idx, (c1_txt, c2_txt) in enumerate(sample_rows):
        ry = table_y + (2 - r_idx) * row_h
        bg_c = "#f8fafc" if r_idx % 2 == 0 else "#ffffff"
        rect_row = patches.Rectangle((table_x, ry), col_w[0] + col_w[1], row_h,
                                     fc=bg_c, ec="#cbd5e1", lw=0.6, zorder=3)
        ax.add_patch(rect_row)
        ax.text(table_x + col_w[0]/2, ry + row_h/2, c1_txt, fontsize=7.5, ha="center", va="center", color=c_gray_txt, zorder=4)
        ax.text(table_x + col_w[0] + col_w[1]/2, ry + row_h/2, c2_txt, fontsize=7.5, fontweight="bold", ha="center", va="center", color=c_blue, zorder=4)

    ax.text(xc1 + card_w/2, y_card + 0.50, r"Chen et al.: Initialized with" + "\n" + r"$N=43$ Prior $\to 100$ Closed-Loop",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 2 Internal Graphic: Embedded 1D GP Slice Plot
    # -------------------------------------------------------------------------
    xc2 = x_cards[1]
    # Inset axes for GP - cleanly separated from header and bottom box
    ax_gp = fig.add_axes([0.246, 0.405, 0.148, 0.265])
    x_synth = np.linspace(0, 10, 100)
    y_true = np.sin(x_synth * 0.8) + 0.1 * x_synth
    # Sample points
    obs_x = np.array([1.2, 3.5, 7.2, 9.0])
    obs_y = np.sin(obs_x * 0.8) + 0.1 * obs_x
    
    # GP mean and epistemic variance
    mu = np.sin(x_synth * 0.8) * 0.85 + 0.1 * x_synth
    sigma = 0.55 * np.exp(-((x_synth - 5.5)/2.0)**2) + 0.18

    ax_gp.fill_between(x_synth, mu - 1.96*sigma, mu + 1.96*sigma, color="#2dd4bf", alpha=0.35, label=r"$\pm 1.96\sigma(x)$")
    ax_gp.plot(x_synth, mu, color=c_teal, lw=2.0, label=r"Mean $\mu(x)$")
    ax_gp.scatter(obs_x, obs_y, color=c_navy, s=30, zorder=5, label=r"Data $\mathcal{D}_t$")
    ax_gp.set_xticks([])
    ax_gp.set_yticks([])
    ax_gp.legend(loc="upper left", fontsize=7.2, frameon=True, framealpha=0.85, handlelength=1.2, borderpad=0.2)
    ax_gp.set_title("GP Posterior Belief", fontsize=8.5, fontweight="bold", color=c_navy, pad=3)
    for spine in ax_gp.spines.values():
        spine.set_color("#cbd5e1")

    ax.text(xc2 + card_w/2, y_card + 0.50, "Surrogate: SingleTaskGP\nKernel: Matérn-5/2 + ARD",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 3 Internal Graphic: Embedded Acquisition Utility Plot
    # -------------------------------------------------------------------------
    xc3 = x_cards[2]
    ax_acq = fig.add_axes([0.440, 0.405, 0.148, 0.265])
    # Acquisition function curve (high where sigma is high and mu is high)
    acq_curve = (mu + 1.8 * sigma)
    acq_curve = (acq_curve - acq_curve.min()) / (acq_curve.max() - acq_curve.min())
    x_opt_idx = np.argmax(acq_curve)
    x_opt_val = x_synth[x_opt_idx]

    ax_acq.plot(x_synth, acq_curve, color=c_orange, lw=2.2, label=r"$\alpha_t(x)$ Utility")
    ax_acq.axvline(x_opt_val, color=c_red, linestyle="--", lw=1.5, label="Max Utility")
    ax_acq.scatter([x_opt_val], [acq_curve[x_opt_idx]], color=c_red, s=45, zorder=5)
    ax_acq.set_xticks([])
    ax_acq.set_yticks([])
    ax_acq.legend(loc="upper left", fontsize=7.2, frameon=True, framealpha=0.85, handlelength=1.2, borderpad=0.2)
    ax_acq.set_title("Exploration vs Exploitation", fontsize=8.5, fontweight="bold", color=c_navy, pad=3)
    for spine in ax_acq.spines.values():
        spine.set_color("#cbd5e1")

    ax.text(xc3 + card_w/2, y_card + 0.50, "Atlas Multi-Objective Policy:\nTarget Window & Utopian HV",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 4 Internal Graphic: Optimal Proposal Vector & Feasibility Filter
    # -------------------------------------------------------------------------
    xc4 = x_cards[3]
    ax.text(xc4 + card_w/2, y_card + 3.75, "Candidate Selection",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Bullseye / Target Icon
    target_circ1 = patches.Circle((xc4 + card_w/2, y_card + 2.75), 0.65, fc="#f3e8ff", ec=c_purple, lw=1.5, zorder=3)
    target_circ2 = patches.Circle((xc4 + card_w/2, y_card + 2.75), 0.42, fc="#ffffff", ec=c_purple, lw=1.2, zorder=4)
    target_circ3 = patches.Circle((xc4 + card_w/2, y_card + 2.75), 0.18, fc=c_purple, ec="none", zorder=5)
    ax.add_patch(target_circ1)
    ax.add_patch(target_circ2)
    ax.add_patch(target_circ3)

    # Crosshairs
    ax.plot([xc4 + card_w/2 - 0.75, xc4 + card_w/2 + 0.75], [y_card + 2.75, y_card + 2.75], color=c_purple, lw=1.0, zorder=6)
    ax.plot([xc4 + card_w/2, xc4 + card_w/2], [y_card + 2.75 - 0.75, y_card + 2.75 + 0.75], color=c_purple, lw=1.0, zorder=6)

    ax.text(xc4 + card_w/2, y_card + 1.60, r"Recommended Batch: $B=4$" + "\n" + r"$\mathbf{x}^* = [\text{aCNC}, \text{GelMA}, \text{LAP}, t_{\mathrm{UV}}]$",
            fontsize=8.0, fontweight="bold", ha="center", va="center", color=c_purple, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec=c_purple, lw=0.8))

    ax.text(xc4 + card_w/2, y_card + 0.50, "Constraints Enforced:\nSolubility & Gelation Limits",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 5 Internal Graphic: Physical Wet-Lab Synthesis & Dual Endpoint
    # -------------------------------------------------------------------------
    xc5 = x_cards[4]
    ax.text(xc5 + card_w/2, y_card + 3.75, "Laboratory Execution",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Petri dish / hydrogel disk schematic
    dish_outer = patches.Circle((xc5 + card_w/2, y_card + 2.75), 0.65, fc="#e2e8f0", ec="#94a3b8", lw=1.2, zorder=3)
    dish_inner = patches.Circle((xc5 + card_w/2, y_card + 2.75), 0.52, fc="#dcfce7", ec=c_green, lw=1.5, zorder=4)
    ax.add_patch(dish_outer)
    ax.add_patch(dish_inner)
    ax.text(xc5 + card_w/2, y_card + 2.75, "Fibrous\nHydrogel", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#15803d", zorder=5)

    # Physical test indicators
    ax.text(xc5 + 0.30, y_card + 1.65, r"1. Mach-1 Compression $\to$ $E$",
            fontsize=7.8, fontweight="bold", ha="left", va="center", color=c_navy, zorder=5)
    ax.text(xc5 + 0.30, y_card + 1.30, r"2. DHR-1 Rheometer $\to$ $\sigma_{100}$",
            fontsize=7.8, fontweight="bold", ha="left", va="center", color=c_navy, zorder=5)

    ax.text(xc5 + card_w/2, y_card + 0.50, "Empirical Measurements:\nModulus & Stress Relaxation",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Curved Feedback Loop Banner (Bottom)
    # -------------------------------------------------------------------------
    # Arrow from Card 5 down, across bottom, and up to Card 1
    p_start = (x_cards[4] + card_w/2, y_card - 0.05)
    p_corner1 = (x_cards[4] + card_w/2, 1.35)
    p_corner2 = (x_cards[0] + card_w/2, 1.35)
    p_end = (x_cards[0] + card_w/2, y_card - 0.05)

    # Draw rounded path for feedback loop
    loop_arrow = FancyArrowPatch(p_corner1, p_end,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=15",
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=3.0, color=c_blue, zorder=3)
    ax.add_patch(loop_arrow)

    # Vertical drop from step 5 to feedback loop
    drop_arrow = FancyArrowPatch(p_start, p_corner1,
                                 arrowstyle="-", lw=3.0, color=c_blue, zorder=3)
    ax.add_patch(drop_arrow)

    # Bottom Banner Box for Feedback Formula
    fb_banner = FancyBboxPatch((3.5, 0.85), 9.0, 0.95,
                               boxstyle="round,pad=0.1,rounding_size=0.2",
                               fc="#ffffff", ec=c_blue, lw=1.8, zorder=4)
    ax.add_patch(fb_banner)

    ax.text(8.0, 1.45, r"Feedback & Dataset Augmentation Protocol",
            fontsize=10.5, fontweight="bold", ha="center", va="center", color=c_navy, zorder=5)
    ax.text(8.0, 1.10, r"$\mathcal{D}_{t+1} \longleftarrow \mathcal{D}_t \cup \{(\mathbf{x}_{t+1}, \mathbf{y}_{t+1})\}$" +
            "  —  Iterate until target tolerance achieved or budget exhausted",
            fontsize=10.0, ha="center", va="center", color=c_blue, fontweight="bold", zorder=5)

    # Small indicators on corner turns
    ax.scatter([p_corner1[0]], [p_corner1[1]], color=c_blue, s=40, zorder=4)

    out_png = AUDIT_ROOT / "report" / "figures" / "fig_active_learning_loop.png"
    out_pdf = AUDIT_ROOT / "report" / "figures" / "fig_active_learning_loop.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated: {out_png}")
    print(f"✅ Generated: {out_pdf}")

if __name__ == "__main__":
    generate_active_learning_diagram()
