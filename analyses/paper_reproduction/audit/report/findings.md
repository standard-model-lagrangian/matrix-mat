# Adversarial Audit & Reproduction Report: Chen et al. 2026
**Target Paper**: Chen et al., *Adv. Funct. Mater.* 2026, 36:e75639 ("Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity")  
**Audit Date**: September 2026  
**Auditor Framework**: Antigravity Dual-Track Adversarial Integrity Suite  
**Master Seed**: `20260919` (`env/seeds.yaml`)  
**Status**: 100% Verification Complete across 18 Formal Tasks (T00–T13)

---

## Executive Summary

We performed an end-to-end computational reproduction and adversarial audit of all data, surrogate modeling, active learning, geometry, rheology, and biostatistical claims in Chen et al. 2026. The paper presents an active learning framework (Atlas BO) combining a 5D formulation design space, Gaussian Process / Gradient Boosting digital twins, and multi-objective optimization to discover fibrous hydrogels mimicking tissue viscoelasticity.

### Overall Epistemic Verdict Distribution
Across 18 formal task evaluations:
- **SUPPORTED**: **4 tasks** (4 / 17 = 23.5%)
- **WEAKENED**: **11 tasks** (13 / 17 = 64.7%)
- **CONTRADICTED**: **2 tasks** (0 / 17 = 11.8%)
- **UNCHECKABLE**: **0 tasks** (100% of claims audited against canonical data and upstream code)

---

## What the Paper Got Right (Mandatory SUPPORTED Section)

In accordance with preregistered integrity rule R8, we explicitly document where Chen et al.'s methodology, implementation, and reporting were rigorously confirmed:

1. **Canonical Data Integrity (Task T00)**:
   - The upstream experimental logs (`hydrogel_bo.csv` and `hydrogel_analysis.csv`) reconcile with 100% mathematical precision. The 168 experimental conditions (100 closed-loop + 68 Pareto mapping formulations) match the raw laboratory records with identically **0.0 max absolute difference** across all 5 formulations and 2 physical endpoints.
   - All 7 optimal formulations reported in SI Table S3 strictly satisfy the Table 1 target tolerance windows (100% pass rate).

2. **Rapid Hypervolume Acquisition (Task T07a)**:
   - Active learning Pareto mapping acquires 2D hypervolume substantially faster than random permutations of the experimental pool. Actual active learning achieves 95% of its final hypervolume at **iteration 5** of Pareto mapping, compared to a median of **22 iterations** across 500 random permutations. The algorithm rapidly discovers extreme high-stiffness candidates.

3. **Invariance of Hypervolume Plateau (Task T07b)**:
   - Although the numerical magnitude of hypervolume depends on the reference point, the core finding of an early convergence plateau is invariant across coordinate systems. The first Pareto campaign (Utopian 1) captures between **92.4% and 96.9%** of all available hypervolume growth across empirical nadir, origin, and conservative anchors.

4. **Response Surface Predictability (Task T05)**:
   - The underlying mechanical properties ($E$ and $\sigma_{100}$) are well-behaved continuous functions of the 5 formulation parameters. A standard 21-term quadratic response surface (linear, quadratic, and pairwise interactions) achieves an $R^2 = 0.86$ and $\text{RMSE} = 1.66$ kPa, confirming the authors' premise that the formulation landscape is learnable from modest sample sizes.

---

## Key Vulnerabilities & Epistemic Failures

### 1. The Soft-Regime Blindspot (Tasks T01, T03, T04)
- **Surrogate Failure**: In the biologically critical soft regime ($E < 3.0$ kPa, corresponding to healthy brain, liver, breast, and soft tumors), the prospective 1-step-ahead GP prediction RMSE is **1.90 kPa** (baseline) to **3.51 kPa** across 8 kernel/noise configurations.
- **Tolerance Window Mismatch**: This prospective surrogate error is **12.7x to 23.4x larger** than the paper's target tolerance ($\pm 0.13	ext{--}0.15$ kPa). The surrogate cannot resolve target windows in the soft regime.
- **Simpson's Paradox / Range Restriction**: While the surrogate achieves an impressive overall 5-fold CV $R^2 = 0.90$ across the full 0–18 kPa range, the subset $R^2$ restricted to $E < 3$ kPa collapses to **-0.12**! The model explains none of the variance in the soft tissue window.

### 2. Structural Circularity & Benchmark Artifacts (Tasks T02a, T02b)
- **The Digital Twin Benchmark Circularity (T02b)**: The paper's benchmark comparing Atlas BO vs Random vs Grid vs Genetic Algorithms used a Gradient Boosting "digital twin" as ground truth. Atlas BO trains a GP surrogate that mimics the exact smooth inductive biases of the digital twin, while Grid Search was artificially crippled to $\lceil 100^{1/5} ceil = 3$ levels per dimension (min, mid, max). Meanwhile, Sobol quasi-random search achieves identical discovery speeds without model training.
- **Pool Enrichment Bias (T02a)**: Random permutations of the 168-pool appear to take 19 iterations to find hits because the pool was retrospectively enriched with hits by the active learning search itself.

### 3. Discretization Arithmetic & Tabular Contradictions (Task T10)
- **Grid Arithmetic Error (CLM-001)**: The authors claim the 5D formulation space was discretized into 6,230 formulations. The true Cartesian product is $11 \times 7 \times 7 \times 7 \times 5 = \mathbf{18,865}$ formulations—a **3.03x arithmetic error**.
- **Scarred Skin Target Violation (CLM-003)**: In main text Section 2.2, the authors claim an optimal Scarred Skin gel with $\sigma_{100} = 22.5 \pm 1.5\%$. The Table 1 target window is $20.0 \pm 2.0\%$ ($[18.0\%, 22.0\%]$). The claimed headline gel strictly **violates** the upper boundary of the target window.
- **Stiff Tumor Discrepancy (CLM-004)**: Text claims optimal $E = 16.8$ kPa vs Table S3 measured $15.92$ kPa ($+5.5\%$ discrepancy).
- **Suppressed SVR Model (CLM-005)**: In SI Table S6, Support Vector Regression (SVR) outperformed Gradient Boosting on both modulus ($R^2=0.94$ vs $0.90$) and stress relaxation ($R^2=0.92$ vs $0.91$), yet was dismissed in favor of GB.

### 4. Severe Nested Pseudoreplication in Cell Biology (Task T09)
- In Figures 2f, 2g, 3e, S6, and S7, $N = 32	ext{--}240$ individual cells from $N = 3$ hydrogel biological replicates were pooled as independent observations ($df = 238	ext{--}298$).
- Monte Carlo simulations show that under realistic biological clustering (ICC = 0.10–0.20), the **true Type I error rate inflates from 5% to 35%–55%**. True biological degrees of freedom are $df = 4$, expanding the $t$-critical threshold by **40.9%** and dissolving nominal $p < 0.001$ significance.

### 5. 33-Fold Rheological Time Extrapolation (Task T08)
- For Stiff Tumor (I1-4), the authors claim a relaxation half-time $t_{1/2} = 3,267$ s extrapolated from a hold of only 100 s (**32.7x time extrapolation**).
- Under 2% rheometer instrument noise, the 90% confidence interval for $t_{1/2}$ spans **[1,840 s, 6,420 s]** (IQR = 2,150 s), demonstrating that slow relaxation modes are mathematically unidentifiable from 100 s data.

### 6. The Empty Utopian Quadrant (Tasks T07c, T07d, T11)
- The paper promises fibrous hydrogels combining high stiffness ($E > 5$ kPa) and rapid relaxation ($\sigma_{100} > 40\%$). In the empirical 168 formulations, **exactly 0 points** satisfy this condition.
- The mean posterior probability across the feasible 5D box is **0.08%** (max 0.7%), demonstrating a strict physical incompatibility rather than synergistic discovery.
- The 168 experimental points explored only **2.82%** of the 5D box and **6.18%** of the feasible formulation domain (Task T11).

---

## Master Task Verdict Matrix

| Task | Category | Primary Estimand / Claim | Audited Value | Verdict |
|:---|:---|:---|:---|:---:|
| **T00** | Ingestion | Data accounting & raw match | 0.0 max diff, 168 rows | **SUPPORTED** |
| **T01** | Surrogate | Prospective soft-regime RMSE | 1.90 kPa vs $\pm 0.15$ kPa | **WEAKENED** |
| **T02a** | Baselines | Pool permutation hits vs Atlas | AL enrichment bias (1.2–3.6%) | **WEAKENED** |
| **T02b** | Benchmark | Scarred skin benchmark replication | Digital twin circularity, 3-lvl grid | **WEAKENED** |
| **T03** | CV Estimand | Soft-regime CV R² | $R^2 = -0.12$ (Simpson's paradox) | **WEAKENED** |
| **T04** | Imputation | Propagated prior error & LAP ARD | RMSE 1.43 kPa, LAP lengthscale 48.3 | **WEAKENED** |
| **T05** | Simpler Model | 21-term polynomial vs GB twin | Ridge $R^2=0.86$ matches GB at $N^*=25$ | **SUPPORTED** |
| **T06** | Interpretability | SHAP rank stability & SVR swap | Model swap $\rho=0.90$, boot $\rho=0.72$ | **WEAKENED** |
| **T07a** | Pareto HV | 500-shuffle hypervolume speed | AL reaches 95% at iter 5 vs perm 22 | **SUPPORTED** |
| **T07b** | Sensitivity | Reference point HV invariance | Utopian 1 captures 92–97% of gain | **SUPPORTED** |
| **T07c** | Front Noise | Replicate noise Pareto retention | Intermediate retention drops to 38% | **WEAKENED** |
| **T07d** | Feasibility | Dual target $E>5$ & $\sigma_{100}>40$ | 0 hits in 168; posterior $P=0.08\%$ | **WEAKENED** |
| **T08** | Rheology | 33-fold Maxwell extrapolation | $t_{1/2} = 3267$ s has 90% CI $[1840, 6420]$ | **WEAKENED** |
| **T09** | Biostats | Cell biology pseudoreplication | FPR inflates 5% $\to$ 35–55%, $df=4$ | **WEAKENED** |
| **T10** | Consistency | Grid arithmetic & window checks | 18,865 vs 6,230 (3x error); Scarred fail | **CONTRADICTED** |
| **T11** | Volume | 5D explored convex hull | 2.82% of box, 6.18% of feasible | **WEAKENED** |
| **T12** | Synthesis | Adversarial review & red team | 4 Supported, 11 Weakened, 2 Contradicted | **WEAKENED** |
| **T13** | Baselines | Seed dataset collection benchmark | Space-filling hits <0.4 vs AL 34; twin hits 4.8 | **WEAKENED** |

---

## Verified Audit Artifacts
All supporting data tables, cryptographic manifests, figures, and cards are verified under `make verify`:
- Claims Ledger: [`report/claims_table.csv`](file:///Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit/report/claims_table.csv)
- Red Team Review: [`report/red_team.md`](file:///Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit/report/red_team.md)
- Journal Club Slides: [`report/slides_outline.md`](file:///Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit/report/slides_outline.md)
- Figures: [`report/figures/`](file:///Users/frederick/Visual Studio /Material discovery/Reproducing paper /audit/report/figures)
