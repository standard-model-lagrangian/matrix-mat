#!/usr/bin/env python3
"""
Calculate and back-verify crosslink characteristics (nu_e and conversion alpha)
for peptide hydrogels (Sample X = 213, Sample Y = 105) across all rheology runs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
T_C = 4.0
T_K = 273.15 + T_C
R = 8.314462618 # J / (mol K)
RT = R * T_K # 2304.35 Pa m^3 / mol = J / mol

# Formulation parameters
# 150 mg/mL PEG-DBCO (Mn = 20,000 g/mol) mixed 1:1 with 18.0 mM peptide
# -> c_PEG = 75 mg/mL = 75 g/L -> C_PEG = 3.75 mM = 3.75 mol/m^3
C_PEG = 3.75 # mol/m^3
two_C_PEG = 2.0 * C_PEG # 7.50 mol/m^3

files = {
    'X1 (Fresh)': 'Rheology/t-sweep x, y 1 replicate 17:09/peptidegels_X1_4C_tsweep.csv',
    'X2 (Day-old A)': 'Rheology/peptidegels_X2_4C_tsweep_oldpep:peg.csv',
    'X3 (Day-old B)': 'Rheology/peptidegels_X3_4C_tsweep.csv',
    'X4 (Day-old C)': 'Rheology/peptidegels_X4_4C_tsweep.csv',
    'Y1 (Fresh Rest)': 'Rheology/t-sweep x, y 1 replicate 17:09/Y_4C_Rest.csv',
    'Y2 (Day-old A)': 'Rheology/peptidegels_Y2_4C_tsweep_oldpep:peg.csv',
    'Y3 (Day-old B)': 'Rheology/peptidegels_Y3_4C_tsweep.csv',
    'Y4 (Day-old C)': 'Rheology/peptidegels_Y4_4C_tsweep.csv'
}

def m_hill(t, G_inf, tau, n, t0):
    dt = np.maximum(0, t - t0)
    return G_inf * (dt**n) / (tau**n + dt**n + 1e-12)

data_dict = {}
results = []

for name, rel_path in files.items():
    df = pd.read_csv(rel_path).dropna(subset=['Time (action)(s)'])
    t = df['Time (action)(s)'].values
    gp = df['Shear modulus (elastic component)(Pa)'].values
    gpp = df['Shear modulus (viscous component)(Pa)'].values
    pa = df['Phase angle(°)'].values
    
    G_end = gp[-1]
    gpp_end = gpp[-1]
    tand_end = gpp_end / G_end
    
    # Fit Hill
    pH, _ = curve_fit(m_hill, t, gp, p0=[G_end*1.2, 350, 2.0, 100], 
                      bounds=([G_end*0.9, 50, 0.5, -200], [G_end*3, 2000, 5.0, 300]))
    G_inf = pH[0]
    tau = pH[1]
    n_exp = pH[2]
    t0 = pH[3]
    
    # Residual drift at end (last 60s)
    mask_last = t >= (t[-1] - 60)
    slope_last_min = np.polyfit(t[mask_last], gp[mask_last], 1)[0] * 60.0 # Pa/min
    drift_pct_min = (slope_last_min / G_end) * 100.0
    
    # Empirical crosslink density (mol/m^3 = mM)
    nu_e_end = G_end / RT
    nu_e_inf = G_inf / RT
    
    # Theoretical conversion alpha: alpha = nu_e / (2 * C_PEG)
    alpha_end = nu_e_end / two_C_PEG
    alpha_inf = nu_e_inf / two_C_PEG
    
    # Underestimation ratio
    understate_ratio = (nu_e_inf - nu_e_end) / nu_e_inf * 100.0
    
    sample_type = 'X (213)' if name.startswith('X') else 'Y (105)'
    
    entry = {
        'Run': name,
        'Sample': sample_type,
        'Duration_s': round(float(t[-1]), 1),
        'G_end_Pa': round(float(G_end), 1),
        'G_inf_Pa': round(float(G_inf), 1),
        'tau_s': round(float(tau), 1),
        'Hill_n': round(float(n_exp), 2),
        't0_s': round(float(t0), 1),
        'dG_dt_Pamin': round(float(slope_last_min), 2),
        'drift_pct_min': round(float(drift_pct_min), 2),
        'nu_e_end_mM': round(float(nu_e_end), 4),
        'nu_e_inf_mM': round(float(nu_e_inf), 4),
        'alpha_end_pct': round(float(alpha_end * 100.0), 2),
        'alpha_inf_pct': round(float(alpha_inf * 100.0), 2),
        'understate_pct': round(float(understate_ratio), 1),
        'tan_delta_end': round(float(tand_end), 4)
    }
    results.append(entry)
    data_dict[name] = {
        't': t, 'gp': gp, 'gpp': gpp, 'pa': pa, 
        'pH': pH, 'entry': entry
    }

res_df = pd.DataFrame(results)
print("=== CALCULATION RESULTS ===")
print(res_df[['Run', 'G_end_Pa', 'G_inf_Pa', 'drift_pct_min', 'nu_e_inf_mM', 'alpha_inf_pct', 'understate_pct']].to_string(index=False))

# Save summary JSON
with open('Rheology/analysis_plots/crosslink_characteristics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create 4-panel publication-grade figure
fig = plt.figure(figsize=(16, 12), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

# Color palettes
colors_X = {'X1 (Fresh)': '#94a3b8', 'X2 (Day-old A)': '#3b82f6', 'X3 (Day-old B)': '#1d4ed8', 'X4 (Day-old C)': '#dc2626'}
colors_Y = {'Y1 (Fresh Rest)': '#cbd5e1', 'Y2 (Day-old A)': '#ea580c', 'Y3 (Day-old B)': '#10b981', 'Y4 (Day-old C)': '#047857'}

# --- PANEL A: Kinetics & Hill Fits ---
ax0 = fig.add_subplot(gs[0, 0])
for name, d in data_dict.items():
    c = colors_X[name] if name.startswith('X') else colors_Y[name]
    ls = '--' if 'Fresh' in name else ('-.' if 'X4' in name or 'Y2' in name else '-')
    # Raw data
    ax0.plot(d['t'], d['gp'], color=c, lw=1.8, label=f"{name} (G'end={d['entry']['G_end_Pa']:.0f} Pa)")
    # Hill fit extension to 1400s
    t_ext = np.linspace(d['t'][0], 1400, 300)
    gp_fit = m_hill(t_ext, *d['pH'])
    ax0.plot(t_ext, gp_fit, color=c, lw=1.0, ls=':', alpha=0.6)
    # Asymptote dashed line
    ax0.axhline(d['entry']['G_inf_Pa'], color=c, ls=':', lw=0.8, alpha=0.4)

ax0.set_xlabel('Reaction Time (s)', fontsize=11, fontweight='bold')
ax0.set_ylabel("Storage Modulus G' (Pa)", fontsize=11, fontweight='bold')
ax0.set_title("A. Modulus Evolution G'(t) & Fitted Asymptotes G_inf", fontsize=12, fontweight='bold')
ax0.grid(True, ls='--', alpha=0.35)
ax0.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
ax0.set_xlim(0, 1400)
ax0.set_ylim(0, 1000)

# --- PANEL B: Empirical nu_e = G' / RT (Endpoint vs Fitted G_inf) ---
ax1 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(results))
width = 0.36

nu_end = [r['nu_e_end_mM'] for r in results]
nu_inf = [r['nu_e_inf_mM'] for r in results]
labels = [r['Run'] for r in results]
colors_bar = [colors_X[r['Run']] if r['Run'].startswith('X') else colors_Y[r['Run']] for r in results]

b1 = ax1.bar(x_pos - width/2, nu_end, width, label="Endpoint G' / RT", color=colors_bar, alpha=0.55, edgecolor='black', lw=0.8)
b2 = ax1.bar(x_pos + width/2, nu_inf, width, label="Fitted G_inf / RT (Empirical)", color=colors_bar, alpha=0.95, edgecolor='black', lw=1.2)

# Value annotations on top of b2
for i, v in enumerate(nu_inf):
    ax1.text(x_pos[i] + width/2, v + 0.008, f"{v:.3f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', rotation=45)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, rotation=35, ha='right', fontsize=8.5, fontweight='medium')
ax1.set_ylabel(r'Apparent Crosslink Density $\nu_e$ (mM)', fontsize=11, fontweight='bold')
ax1.set_title(r"B. Empirical $\nu_e = G' / RT$ (Endpoint vs. Fitted $G_\infty$)", fontsize=12, fontweight='bold')
ax1.axhline(0.207, color='#1d4ed8', ls='--', lw=1.2, label='Baseline X (X2,X3: ~0.207 mM)')
ax1.axhline(0.214, color='#047857', ls=':', lw=1.2, label='Baseline Y (Y3,Y4: ~0.214 mM)')
ax1.set_ylim(0, 0.46)
ax1.grid(True, axis='y', ls='--', alpha=0.35)
ax1.legend(fontsize=8, loc='upper left', framealpha=0.9)

# --- PANEL C: Theoretical Conversion alpha = nu_e / (2 * C_PEG) ---
ax2 = fig.add_subplot(gs[1, 0])
alpha_inf = [r['alpha_inf_pct'] for r in results]
alpha_end = [r['alpha_end_pct'] for r in results]

b3 = ax2.bar(x_pos - width/2, alpha_end, width, label="From Endpoint G'", color=colors_bar, alpha=0.55, edgecolor='black', lw=0.8)
b4 = ax2.bar(x_pos + width/2, alpha_inf, width, label="From Fitted G_inf (True alpha)", color=colors_bar, alpha=0.95, edgecolor='black', lw=1.2)

for i, v in enumerate(alpha_inf):
    ax2.text(x_pos[i] + width/2, v + 0.1, f"{v:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=8.5, fontweight='medium')
ax2.set_ylabel(r'Gelation Conversion $\alpha$ (%)', fontsize=11, fontweight='bold')
ax2.set_title(r"C. Theoretical Conversion $\alpha = \nu_e / (2 C_{\rm PEG})$", fontsize=12, fontweight='bold')
ax2.axhline(2.76, color='#1d4ed8', ls='--', lw=1.2, label='X Baseline (X2,X3: 2.76%)')
ax2.axhline(2.86, color='#047857', ls=':', lw=1.2, label='Y Baseline (Y3,Y4: 2.86%)')
ax2.set_ylim(0, 6.2)
ax2.grid(True, axis='y', ls='--', alpha=0.35)
ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.9)

# --- PANEL D: Mechanistic Failure Mode Discrimination ---
# Normalized kinetics: G'(t) / G_inf vs. (t - t0) / tau
# Proves kinetic master curve superposition (identical reaction mechanism)
ax3 = fig.add_subplot(gs[1, 1])

# Inset / secondary axis for tan delta
ax3_twin = ax3.twinx()
ax3_twin.set_ylabel(r'Loss Factor $\tan\delta = G^{\prime\prime}/G^\prime$', fontsize=10, color='#6b7280')
ax3_twin.tick_params(axis='y', labelcolor='#6b7280')
ax3_twin.set_ylim(0, 0.035)

for name in ['X2 (Day-old A)', 'X3 (Day-old B)', 'X4 (Day-old C)', 'Y2 (Day-old A)', 'Y3 (Day-old B)', 'Y4 (Day-old C)']:
    d = data_dict[name]
    pH = d['pH']
    G_inf_val = pH[0]
    tau_val = pH[1]
    t0_val = pH[3]
    
    t_norm = (d['t'] - t0_val) / tau_val
    gp_norm = d['gp'] / G_inf_val
    c = colors_X[name] if name.startswith('X') else colors_Y[name]
    
    mask = t_norm >= 0
    ax3.plot(t_norm[mask], gp_norm[mask], color=c, lw=2.0, label=f"{name}")

# Ideal Hill master curve with n = 1.90
t_master = np.linspace(0, 3.5, 200)
gp_master = (t_master**1.9) / (1.0 + t_master**1.9)
ax3.plot(t_master, gp_master, 'k--', lw=2.5, alpha=0.7, label=r'Universal Master Curve ($n \approx 1.9$)')

# Plot tan delta points on twin axis
for name in ['X2 (Day-old A)', 'X3 (Day-old B)', 'X4 (Day-old C)', 'Y2 (Day-old A)', 'Y3 (Day-old B)', 'Y4 (Day-old C)']:
    d = data_dict[name]
    c = colors_X[name] if name.startswith('X') else colors_Y[name]
    ax3_twin.scatter([3.2], [d['entry']['tan_delta_end']], color=c, s=50, zorder=5, marker='D')

ax3.set_xlabel(r'Normalized Reaction Time $(t - t_0) / \tau$', fontsize=11, fontweight='bold')
ax3.set_ylabel(r"Normalized Modulus $G'(t) / G_\infty$", fontsize=11, fontweight='bold')
ax3.set_title(r"D. Failure Mode Test: Master Curve Collapse & $\tan\delta$", fontsize=12, fontweight='bold')
ax3.grid(True, ls='--', alpha=0.35)
ax3.set_xlim(0, 3.5)
ax3.set_ylim(0, 1.05)
ax3.legend(fontsize=8, loc='lower right', framealpha=0.9)

# Overall title
fig.suptitle(r"Crosslink Characteristics & Network Density Verification: $\nu_e = G'/RT$ and $\nu_e = 2C_{\rm PEG}\alpha$" + "\n" +
             r"Formulation X (213: EKGEKGEKG) vs. Formulation Y (105: EKEKEKEKEK) at 4.0 °C", 
             fontsize=14, fontweight='bold', y=0.98)

out_plot = 'Rheology/analysis_plots/crosslink_characteristics_nu_alpha.png'
plt.savefig(out_plot, bbox_inches='tight')
plt.close()
print(f"Figure successfully saved to: {out_plot}")
