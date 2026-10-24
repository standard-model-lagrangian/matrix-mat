"""
Generate Presentation Graphic: Abstract Forest Explorer Analogy of Active Learning.
Designed for 16:9 slide presentation with publication-quality typography,
hand-crafted vector illustrations, and intuitive step-by-step storytelling.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, PathPatch
import matplotlib.path as mpath

AUDIT_ROOT = Path(__file__).resolve().parent.parent

def draw_tree(ax, x, y, scale=0.35, color="#15803d"):
    """Draw a stylized pine tree at (x, y)."""
    # Trunk
    trunk = patches.Rectangle((x - 0.05*scale, y), 0.1*scale, 0.3*scale,
                              fc="#78350f", ec="none", zorder=4)
    ax.add_patch(trunk)
    # Foliage tiers
    p1 = Polygon([[x - 0.4*scale, y + 0.25*scale], [x + 0.4*scale, y + 0.25*scale], [x, y + 0.75*scale]],
                 fc=color, ec="none", zorder=5)
    p2 = Polygon([[x - 0.32*scale, y + 0.5*scale], [x + 0.32*scale, y + 0.5*scale], [x, y + 0.95*scale]],
                 fc=color, ec="none", zorder=5)
    p3 = Polygon([[x - 0.22*scale, y + 0.75*scale], [x + 0.22*scale, y + 0.75*scale], [x, y + 1.15*scale]],
                 fc=color, ec="none", zorder=5)
    ax.add_patch(p1)
    ax.add_patch(p2)
    ax.add_patch(p3)

def draw_cloud(ax, x, y, scale=0.4, color="#cbd5e1", alpha=0.85):
    """Draw a stylized puffy cloud / fog blob."""
    c1 = Circle((x, y), 0.35*scale, fc=color, ec="none", alpha=alpha, zorder=6)
    c2 = Circle((x + 0.3*scale, y + 0.1*scale), 0.4*scale, fc=color, ec="none", alpha=alpha, zorder=6)
    c3 = Circle((x + 0.65*scale, y - 0.05*scale), 0.32*scale, fc=color, ec="none", alpha=alpha, zorder=6)
    c4 = Circle((x + 0.3*scale, y - 0.15*scale), 0.35*scale, fc=color, ec="none", alpha=alpha, zorder=6)
    for c in [c1, c2, c3, c4]:
        ax.add_patch(c)

def generate_forest_analogy_diagram():
    fig = plt.figure(figsize=(16, 9), dpi=300, facecolor="#f8fafc")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Typography & Palette
    c_bg_card = "#ffffff"
    c_border_card = "#cbd5e1"
    c_navy = "#0f172a"
    c_blue = "#2563eb"
    c_teal = "#0d9488"
    c_orange = "#ea580c"
    c_purple = "#7c3aed"
    c_green = "#16a34a"
    c_gray_txt = "#475569"

    # Title Banner
    ax.text(8.0, 8.45, "Active Learning Explained: The Forest Explorer Analogy",
            fontsize=19, fontweight="bold", ha="center", va="center", color=c_navy)
    ax.text(8.0, 8.05, "How an intelligent explorer navigates an unknown forest to find a hidden treasure with minimum wasted steps",
            fontsize=11.5, ha="center", va="center", color=c_gray_txt)

    # 5 Step Cards coordinates
    card_w = 2.75
    card_h = 4.8
    y_card = 2.7
    x_cards = [0.65, 3.75, 6.85, 9.95, 13.05]

    steps_meta = [
        {
            "num": "1",
            "title": "Past Footprints",
            "subtitle": "The Memory / Data",
            "concept": "Where have I already walked?",
            "color": c_blue,
            "bg_color": "#eff6ff"
        },
        {
            "num": "2",
            "title": "Mental Map & Fog",
            "subtitle": "The Belief / Surrogate",
            "concept": "What do I think is out there?",
            "color": c_teal,
            "bg_color": "#f0fdfa"
        },
        {
            "num": "3",
            "title": "Curiosity vs. Hope",
            "subtitle": "The Strategy / Acquisition",
            "concept": "Where is it best to explore next?",
            "color": c_orange,
            "bg_color": "#fff7ed"
        },
        {
            "num": "4",
            "title": "Pick Destination",
            "subtitle": "The Decision / Argmax",
            "concept": "Locking onto the next ridge.",
            "color": c_purple,
            "bg_color": "#faf5ff"
        },
        {
            "num": "5",
            "title": "Take the Step",
            "subtitle": "The Reality / Experiment",
            "concept": "See what is actually there!",
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
        circle_badge = Circle((xc + 0.42, y_card + card_h - 0.42), 0.26,
                              fc=meta["color"], ec="none", zorder=4)
        ax.add_patch(circle_badge)
        ax.text(xc + 0.42, y_card + card_h - 0.42, meta["num"],
                color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)

        # Step Title & Subtitle
        ax.text(xc + 0.80, y_card + card_h - 0.30, meta["title"],
                fontsize=11.0, fontweight="bold", ha="left", va="center", color=c_navy, zorder=5)
        ax.text(xc + 0.80, y_card + card_h - 0.58, meta["subtitle"],
                fontsize=9.0, fontweight="semibold", ha="left", va="center", color=meta["color"], zorder=5)

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
    # Card 1: Past Footprints (Trail map with breadcrumbs)
    # -------------------------------------------------------------------------
    xc1 = x_cards[0]
    ax.text(xc1 + card_w/2, y_card + 3.75, "Trail Map & Waypoints",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Mini map background
    map_box = FancyBboxPatch((xc1 + 0.22, y_card + 1.25), card_w - 0.44, 2.30,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            fc="#f1f5f9", ec="#cbd5e1", lw=1.0, zorder=3)
    ax.add_patch(map_box)

    # Trees on map - positioned to leave center trail wide open
    draw_tree(ax, xc1 + 0.40, y_card + 2.55, scale=0.48, color="#16a34a")
    draw_tree(ax, xc1 + 2.38, y_card + 2.65, scale=0.48, color="#15803d")
    draw_tree(ax, xc1 + 0.45, y_card + 1.35, scale=0.48, color="#22c55e")
    draw_tree(ax, xc1 + 2.30, y_card + 1.35, scale=0.50, color="#166534")

    # Dotted trail of visited sites
    trail_pts = np.array([
        [xc1 + 0.85, y_card + 1.70],
        [xc1 + 1.30, y_card + 2.15],
        [xc1 + 1.65, y_card + 2.70]
    ])
    ax.plot(trail_pts[:, 0], trail_pts[:, 1], color=c_blue, ls="--", lw=1.8, marker="o",
            markersize=7, markerfacecolor="#ffffff", markeredgecolor=c_blue, markeredgewidth=1.8, zorder=6)

    # Labels for visited points - compact & perfectly positioned
    ax.text(trail_pts[0, 0] + 0.12, trail_pts[0, 1] - 0.05, "1. Valley (Low)", fontsize=6.8, fontweight="bold", ha="left", va="center", color=c_navy, zorder=7)
    ax.text(trail_pts[1, 0] - 0.12, trail_pts[1, 1] + 0.05, "2. Ridge (Dry)", fontsize=6.8, fontweight="bold", ha="right", va="center", color=c_navy, zorder=7)
    ax.text(trail_pts[2, 0] - 0.12, trail_pts[2, 1] + 0.05, "3. Stream (Hope)", fontsize=6.8, fontweight="bold", ha="right", va="center", color=c_blue, zorder=7)

    ax.text(xc1 + card_w/2, y_card + 0.50, "The Archive:\nLog of visited coordinates\nand what was found there",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 2: Mental Map & Fog of War (Surrogate Belief)
    # -------------------------------------------------------------------------
    xc2 = x_cards[1]
    ax.text(xc2 + card_w/2, y_card + 3.75, "Terrain Belief & Fog of War",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Inset frame for terrain cross-section
    terrain_box = FancyBboxPatch((xc2 + 0.22, y_card + 1.25), card_w - 0.44, 2.30,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#f8fafc", ec="#cbd5e1", lw=1.0, zorder=3)
    ax.add_patch(terrain_box)

    # Simulated terrain hill line - strictly bounded inside box
    x_local = np.linspace(xc2 + 0.28, xc2 + card_w - 0.28, 100)
    # Controlled waveform that stays gracefully within the frame
    y_terrain = y_card + 1.95 + 0.38 * np.sin((x_local - xc2 - 0.4) * 2.8) + 0.12 * np.cos((x_local - xc2) * 5.2)
    
    # Ground fill
    ax.fill_between(x_local, y_card + 1.25, y_terrain, color="#ccfbf1", alpha=0.75, zorder=4)
    ax.plot(x_local, y_terrain, color=c_teal, lw=2.2, zorder=5)

    # Visited spots (clear sun / checkmark placed cleanly ABOVE hill summits)
    pt1_idx, pt2_idx = 24, 92
    ax.scatter([x_local[pt1_idx], x_local[pt2_idx]], [y_terrain[pt1_idx], y_terrain[pt2_idx]],
               color=c_teal, s=45, zorder=7, edgecolors="white", lw=1.5)
    ax.text(x_local[pt1_idx], y_terrain[pt1_idx] + 0.12, "Known Peak", fontsize=7.0, fontweight="bold", ha="center", va="bottom", color="#0f766e", zorder=7)
    ax.text(x_local[pt2_idx] - 0.10, y_terrain[pt2_idx] + 0.12, "Known Ridge", fontsize=7.0, fontweight="bold", ha="right", va="bottom", color="#0f766e", zorder=7)

    # Fog clouds hovering over the uncharted middle valley
    draw_cloud(ax, xc2 + 1.15, y_card + 2.65, scale=0.70, color="#94a3b8", alpha=0.90)
    draw_cloud(ax, xc2 + 1.45, y_card + 2.80, scale=0.65, color="#64748b", alpha=0.85)
    ax.text(xc2 + 1.45, y_card + 2.70, "Unexplored Fog\n(High Mystery)",
            fontsize=7.2, fontweight="bold", ha="center", va="center", color="#ffffff", zorder=8)

    ax.text(xc2 + card_w/2, y_card + 0.50, "The Mental Model:\nGuessed landscape elevation\n+ thick fog in unexplored valleys",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 3: Curiosity vs Hope (The Acquisition / Compass)
    # -------------------------------------------------------------------------
    xc3 = x_cards[2]
    ax.text(xc3 + card_w/2, y_card + 3.75, "Weighing the Two Drives",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Balance scale box
    acq_box = FancyBboxPatch((xc3 + 0.22, y_card + 1.25), card_w - 0.44, 2.30,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            fc="#fff7ed", ec="#fdba74", lw=1.0, zorder=3)
    ax.add_patch(acq_box)

    cx3 = xc3 + card_w/2
    cy3 = y_card + 2.35
    # Fulcrum triangle
    fulcrum = Polygon([[cx3 - 0.22, cy3 - 0.45], [cx3 + 0.22, cy3 - 0.45], [cx3, cy3 - 0.10]],
                      fc="#9a3412", ec="none", zorder=4)
    ax.add_patch(fulcrum)
    # Beam
    ax.plot([cx3 - 0.95, cx3 + 0.95], [cy3 - 0.05, cy3 + 0.15], color="#c2410c", lw=3.0, zorder=5)

    # Left Pan: Exploitation (High Hope)
    pan_l = FancyBboxPatch((cx3 - 1.10, cy3 - 0.65), 0.68, 0.48,
                           boxstyle="round,pad=0.03,rounding_size=0.08",
                           fc="#ffedd5", ec=c_orange, lw=1.2, zorder=6)
    ax.add_patch(pan_l)
    ax.text(cx3 - 0.76, cy3 - 0.41, "Hope\n(Exploit)", fontsize=7.2, fontweight="bold", ha="center", va="center", color=c_orange, zorder=7)

    # Top formula banner
    ax.text(cx3, cy3 + 0.40, "Value = Hope + Curiosity", fontsize=7.5, fontweight="bold", ha="center", va="center", color="#9a3412", zorder=6)

    # Right Pan: Exploration (Curiosity in Fog)
    pan_r = FancyBboxPatch((cx3 + 0.42, cy3 - 0.45), 0.68, 0.48,
                           boxstyle="round,pad=0.03,rounding_size=0.08",
                           fc="#fed7aa", ec=c_orange, lw=1.2, zorder=6)
    ax.add_patch(pan_r)
    ax.text(cx3 + 0.76, cy3 - 0.21, "Curiosity\n(Explore)", fontsize=7.2, fontweight="bold", ha="center", va="center", color=c_orange, zorder=7)

    ax.text(xc3 + card_w/2, y_card + 0.50, "The Scoring Metric:\nBalances checking near good spots\nvs daring to enter uncharted fog",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 4: Pick Destination (The Decision / Pinpoint Target)
    # -------------------------------------------------------------------------
    xc4 = x_cards[3]
    ax.text(xc4 + card_w/2, y_card + 3.75, "Select Next Destination",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Decision box
    dec_box = FancyBboxPatch((xc4 + 0.22, y_card + 1.25), card_w - 0.44, 2.30,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            fc="#faf5ff", ec="#d8b4fe", lw=1.0, zorder=3)
    ax.add_patch(dec_box)

    cx4 = xc4 + card_w/2
    cy4 = y_card + 2.50

    # Crosshairs & Waypoint Marker Flag
    ax.plot([cx4 - 0.70, cx4 + 0.70], [cy4, cy4], color=c_purple, lw=1.0, ls=":", zorder=4)
    ax.plot([cx4, cx4], [cy4 - 0.70, cy4 + 0.70], color=c_purple, lw=1.0, ls=":", zorder=4)

    ring1 = Circle((cx4, cy4), 0.50, fc="none", ec=c_purple, lw=1.2, ls="--", zorder=4)
    ring2 = Circle((cx4, cy4), 0.22, fc="#f3e8ff", ec=c_purple, lw=1.5, zorder=5)
    center_pt = Circle((cx4, cy4), 0.08, fc=c_purple, ec="none", zorder=6)
    ax.add_patch(ring1)
    ax.add_patch(ring2)
    ax.add_patch(center_pt)

    # Flag Pole and Pennant
    ax.plot([cx4 + 0.15, cx4 + 0.15], [cy4 + 0.10, cy4 + 0.65], color="#4c1d95", lw=1.8, zorder=7)
    flag = Polygon([[cx4 + 0.15, cy4 + 0.65], [cx4 + 0.55, cy4 + 0.50], [cx4 + 0.15, cy4 + 0.35]],
                   fc="#9333ea", ec="none", zorder=8)
    ax.add_patch(flag)

    ax.text(cx4, y_card + 1.55, "Optimal Next Waypoint\n(Highest Combined Hope + Mystery)",
            fontsize=7.5, fontweight="bold", ha="center", va="center", color=c_purple, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=c_purple, lw=0.8))

    ax.text(xc4 + card_w/2, y_card + 0.50, "The Decision:\nPinpoints the single most\nrewarding coordinate to visit",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Card 5: Take the Step & Discover (Ground Truth Reality)
    # -------------------------------------------------------------------------
    xc5 = x_cards[4]
    ax.text(xc5 + card_w/2, y_card + 3.75, "Arrive & Clear the Fog",
            fontsize=9.0, fontweight="bold", ha="center", color=c_navy, zorder=4)

    # Discovery box
    disc_box = FancyBboxPatch((xc5 + 0.22, y_card + 1.25), card_w - 0.44, 2.30,
                             boxstyle="round,pad=0.05,rounding_size=0.12",
                             fc="#f0fdf4", ec="#86efac", lw=1.0, zorder=3)
    ax.add_patch(disc_box)

    cx5 = xc5 + card_w/2
    cy5 = y_card + 2.50

    # Magnifying glass / treasure discovery icon
    lens_c = Circle((cx5 - 0.10, cy5 + 0.10), 0.45, fc="#dcfce7", ec=c_green, lw=2.2, zorder=5)
    ax.add_patch(lens_c)
    # Lens handle
    ax.plot([cx5 + 0.22, cx5 + 0.55], [cy5 - 0.22, cy5 - 0.55], color="#166534", lw=4.0, solid_capstyle="round", zorder=6)

    # Gold star / gem inside lens
    ax.text(cx5 - 0.10, cy5 + 0.10, "★", fontsize=22, ha="center", va="center", color="#eab308", zorder=7)

    # Measurement result banner
    ax.text(cx5, y_card + 1.55, "Ground Truth Discovered!\n(Fog cleared | New data logged)",
            fontsize=7.5, fontweight="bold", ha="center", va="center", color=c_green, zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffffff", ec=c_green, lw=0.8))

    ax.text(xc5 + card_w/2, y_card + 0.50, "The Reality:\nMake the physical hike,\nclear local fog, and record truth",
            fontsize=8.0, ha="center", va="center", color=c_gray_txt, zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc="#f1f5f9", ec="#e2e8f0", lw=0.8))

    # -------------------------------------------------------------------------
    # Curved Feedback Loop Banner (Bottom)
    # -------------------------------------------------------------------------
    p_start = (x_cards[4] + card_w/2, y_card - 0.05)
    p_corner1 = (x_cards[4] + card_w/2, 1.35)
    p_end = (x_cards[0] + card_w/2, y_card - 0.05)

    loop_arrow = FancyArrowPatch(p_corner1, p_end,
                                 connectionstyle="angle,angleA=0,angleB=90,rad=15",
                                 arrowstyle="-|>", mutation_scale=18,
                                 lw=3.0, color=c_blue, zorder=3)
    ax.add_patch(loop_arrow)

    drop_arrow = FancyArrowPatch(p_start, p_corner1,
                                 arrowstyle="-", lw=3.0, color=c_blue, zorder=3)
    ax.add_patch(drop_arrow)

    # Bottom Banner Box for Feedback Loop
    fb_banner = FancyBboxPatch((2.2, 0.82), 11.6, 1.05,
                               boxstyle="round,pad=0.1,rounding_size=0.2",
                               fc="#ffffff", ec=c_blue, lw=1.8, zorder=4)
    ax.add_patch(fb_banner)

    ax.text(8.0, 1.48, "The Closed Learning Loop: Your Mental Map Updates After Every Single Hike",
            fontsize=11.0, fontweight="bold", ha="center", va="center", color=c_navy, zorder=5)
    ax.text(8.0, 1.10, "1. Clear away fog at visited spot   ➔   2. Re-estimate all mountain peaks   ➔   3. Next destination is even smarter!",
            fontsize=10.0, ha="center", va="center", color=c_blue, fontweight="bold", zorder=5)

    ax.scatter([p_corner1[0]], [p_corner1[1]], color=c_blue, s=40, zorder=4)

    # Output paths
    out_png = AUDIT_ROOT / "report" / "figures" / "fig_active_learning_forest_analogy.png"
    out_pdf = AUDIT_ROOT / "report" / "figures" / "fig_active_learning_forest_analogy.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated: {out_png}")
    print(f"✅ Generated: {out_pdf}")

if __name__ == "__main__":
    generate_forest_analogy_diagram()
