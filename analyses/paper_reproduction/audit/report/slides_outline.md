# Journal Club Presentation Outline (Monday Presentation)
**Topic**: Active Learning-Accelerated Discovery of Fibrous Hydrogels with Tissue-Mimetic Viscoelasticity  
**Target Paper**: Chen et al., *Adv. Funct. Mater.* 2026, 36:e75639  
**Format**: 15 Slides, ~25 minutes presentation + 15 minutes discussion  
**Auditor Perspective**: Constructive, rigorous adversarial reproduction

---

### Slide 1: Title & Framing
- **Title**: Active Learning Hydrogel Discovery: Breakthrough or Benchmark Illusion?
- **Subtitle**: A Computational Reproduction & Adversarial Audit of Chen et al., *Adv. Funct. Mater.* 2026
- **Presenter**: Journal Club Presentation
- **Auditor Tone**: Balanced, data-first reproduction. We acknowledge genuine experimental achievements while dissecting statistical and algorithmic vulnerabilities.
- **Visual**: Graphical abstract from paper paired with the Auditor Verdict Matrix (4 Supported, 10 Weakened, 2 Contradicted).

---

### Slide 2: Paper Overview & The Core Scientific Promise
- **The Challenge**: Recreating native tissue mechanics requires simultaneously matching compressive stiffness ($E$) and stress relaxation dynamics ($\sigma_{100}$).
- **The Platform**: 5-component fibrous hydrogel (GelMA-L, GelMA-H, a-CNC, LAP, UV exposure).
- **The Method**: Active learning loop (Atlas Bayesian Optimization) with Gaussian Process & Gradient Boosting digital twins.
- **The Claim**: Discovered 7 tissue-mimetic formulations (Skin, Scar, Soft Tumor, Stiff Tumor) in only 100 closed-loop iterations + mapped Pareto front in 68 iterations.

---

### Slide 3: What the Paper Got Right (The SUPPORTED Highlights)
- **Full Data Provenance (T00)**: All 168 experimental formulations were 100% reconciled against raw laboratory data.
- **Target Formulations Validated (T00, Table S3)**: All 7 reported optimal gels strictly meet Table 1 tolerance windows.
- **Fast Pareto Acquisition (T07a)**: Active learning reaches 95% of maximum hypervolume in **5 iterations** vs **22 iterations** for random permutations.
- **Hypervolume Plateau Invariance (T07b)**: Convergence dynamics hold regardless of coordinate reference point.
- **Key Takeaway**: The chemistry works, the hydrogels were physically fabricated, and active learning finds stiff formulations rapidly.

---

### Slide 4: Vulnerability 1 — The Soft-Regime Blindspot
- **The Problem**: Soft tissues (brain, liver, breast, soft tumors) have $E < 3.0$ kPa.
- **The Evidence (T01)**: Prospective 1-step-ahead GP prediction RMSE is **1.90 kPa**—over **12.7x larger** than target tolerance ($\pm 0.15$ kPa).
- **Simpson's Paradox (T03)**: Model achieves overall $R^2 = 0.90$, but subset $R^2$ restricted to $E < 3$ kPa drops to **-0.12**.
- **Figure to Display**: `report/figures/fig_soft_regime.pdf` (predicted vs true modulus with shaded error bands vs target window).
- **Discussion Point**: Can an active learning surrogate guide discovery if its uncertainty is 10x wider than the target window?

---

### Slide 5: Vulnerability 2 — Imputation Fingerprints in Prior Data
- **The Prior (T04)**: Initialized with 43 literature formulations (SI Table S2; missing index 2).
- **Zero-Variance Imputation**: In 81.4% of prior formulations, photoinitiator (LAP) was imputed at a constant 0.10 wt%.
- **ARD Diagnostic**: Fitting GP on prior data causes LAP lengthscale to diverge to $48.3$ (boundary pegging), effectively disabling photoinitiator sensitivity during early rounds.
- **Propagated Error**: Propagated imputation RMSE is **1.43 kPa** (9.5x target tolerance).

---

### Slide 6: Vulnerability 3 — The Circular Benchmark & Crippled Grid
- **The Benchmark Claim (T02b)**: Paper claims Atlas BO vastly outperforms Random, Grid Search, and Genetic Algorithms.
- **The Circularity**: Benchmark runs on a Gradient Boosting *digital twin* simulator, not wet lab experiments. Atlas BO uses a GP that matches the digital twin's smooth inductive biases.
- **The Crippled Grid**: Grid Search was restricted to 3 levels per feature ($\{0, 	ext{mid}, 	ext{max}\}$), rendering target discovery impossible by construction.
- **The Reality**: Sobol quasi-random search achieves identical discovery speeds with zero model overhead.
- **Figure to Display**: `report/figures/fig_benchmark_scarred.pdf`.

---

### Slide 7: Vulnerability 4 — Simpler Models & "Dumb" Seed Baselines (T05, T13)
- **Simpler-Model Counterfactual (T05)**:
  - Fit standard 21-term quadratic response surface (linear, quadratic, pairwise interactions).
  - 21-term Ridge regression achieves $R^2 = 0.86$, outperforming the Gradient Boosting twin ($R^2 = 0.78$) under 5-fold CV.
  - Learning curves show simpler models match surrogate accuracy at $N^* = 25$ formulations.
- **Dumb Baselines Benchmark (T13)**:
  - Evaluated 7 initial seed collection methods (Sobol, LHS, Halton, I-Optimal, D-Optimal, CCD, Random) across 5 budgets and 200 seeds.
  - **Formulation Space**: Passive designs achieve $<0.4$ hits ($d_{	ext{norm}} \le 0.15$) vs **34 hits** in Chen closed-loop ($N=100$), confirming heavy active learning clustering.
  - **Property Space**: Due to mechanical degeneracy, passive designs hit Table 1 target windows **4.8 times per 100** by chance, weakening claims that active learning was strictly necessary to achieve tissue-mimetic properties.
- **Figure to Display**: `report/figures/fig_seed_baselines.pdf`.

---

### Slide 8: Vulnerability 5 — Interpretability & Mechanistic Claims
- **The Paper's Fig 5 (T06)**: TreeExplainer SHAP attribution on Gradient Boosting twin to derive mechanistic insights.
- **The Suppressed Model**: In SI Table S6, SVR achieved higher test $R^2$ ($0.94$ vs $0.90$ modulus; $0.92$ vs $0.91$ relaxation).
- **Model Swap Instability**: Swapping to SVR alters feature rankings. Bootstrap resampling ($B=200$) reveals substantial rank flipping.
- **a-CNC Non-Monotonicity**: Apparent non-monotonicity in Gradient Boosting is driven by step-wise decision tree splits and smooths out in kernel regression.
- **Figure to Display**: `report/figures/fig_shap_robustness.pdf`.

---

### Slide 9: Vulnerability 6 — The Empty Utopian Quadrant
- **The Promise (T07d)**: Fibrous hydrogels breaking polymer trade-offs to achieve high stiffness ($E > 5$ kPa) AND fast relaxation ($\sigma_{100} > 40\%$).
- **The Reality**: Exactly **0 out of 168** formulations meet this dual condition.
- **Mutual Exclusivity**: Maximum relaxation for $E > 5$ kPa is $34.5\%$; maximum modulus for $\sigma_{100} > 40\%$ is $1.84$ kPa.
- **Figure to Display**: `report/figures/fig_pareto_audit.pdf` (Panels C and D).

---

### Slide 10: Vulnerability 7 — Pareto Front Fragility Under Noise
- **Front Uncertainty (T07c)**: 16 Pareto optimal points identified across 168 points.
- **Replicate Noise ($10\%$ CV)**: Under 1,000 noise perturbations, intermediate trade-off points have retention rates as low as **38%**.
- **Campaign Contributions**: 8 of 16 Pareto points came from earlier single-target campaigns. Utopian 2 and Utopian 3 campaigns contributed **zero** points to the final front.

---

### Slide 11: Vulnerability 8 — Effective Explored Volume
- **Hypercube vs Reality (T11)**:
  - Nominal 5D parameter box: 100% volume.
  - Physical feasibility domain (linear polymer constraints): **45.5%** of box.
  - 168 points convex hull: **2.82%** of nominal box.
  - Explored fraction of feasible space: **6.18%** (93.8% unexplored).
  - Initial 100 closed-loop points: **0.44%** of box.
- **Takeaway**: Active learning localized intensely in narrow corridors rather than broadly navigating 5D space.
- **Figure to Display**: `report/figures/fig_volume_hull.pdf`.

---

### Slide 12: Vulnerability 9 — 33-Fold Rheological Time Extrapolation
- **The Claim (T08)**: Extrapolated half-times $t_{1/2}$ up to **3,267 s** (Stiff Tumor) from a **100 s** experimental hold (**32.7x time extrapolation**).
- **Identifiability Failure**: 2-mode Maxwell parameters $(G_2, 	au_2, G_\infty)$ are strongly collinear.
- **Noise Sensitivity**: Under 2% rheometer noise, the 90% confidence interval for $t_{1/2}$ is $[1,840	ext{ s}, 6,420	ext{ s}]$ (IQR = 2,150 s).
- **Single-Number Proxy**: $\sigma_{100}$ conceals divergent early vs late relaxation kinetics.
- **Figure to Display**: `report/figures/fig_rheology_audit.pdf`.

---

### Slide 13: Vulnerability 10 — Cell Biology Nested Pseudoreplication
- **The Flaw (T09)**: Figures 2f, 2g, 3e, S6, S7 pooled $N = 32	ext{--}240$ individual cells from $N = 3$ hydrogel replicates as independent samples ($df = 238	ext{--}298$).
- **Statistical Reality**: Experimental unit is the hydrogel batch ($N=3$, $df=4$).
- **False Positive Inflation**: With realistic clustering (ICC = 0.10–0.20), Type I error rate explodes from **5% to 35%–55%**.
- **P-Value Collapse**: Nominal $p < 0.001$ shifts to $p pprox 0.05	ext{--}0.10$ under cluster-robust hierarchical mixed models.
- **Figure to Display**: `report/figures/fig_biostats_pseudoreplication.pdf`.

---

### Slide 14: Vulnerability 11 — Arithmetic & Consistency Ledger
- **Grid Arithmetic (CLM-001)**: $11 	imes 7 	imes 7 	imes 7 	imes 5 = \mathbf{18,865}$ formulations vs **6,230 claimed** (3.03x error).
- **Scarred Skin Target Violation (CLM-003)**: Text claims optimal gel relaxation of $22.5\%$, strictly violating the Table 1 target window ($18.0\%	ext{--}22.0\%$).
- **Stiff Tumor Mismatch (CLM-004)**: Text claims $16.8$ kPa vs Table S3 measured $15.92$ kPa.
- **Misattributed Citations (CLM-006)**: Atlas cited as tissue biology papers [35, 36]; GA cited as composite papers [37, 38].
- **Ledger Artifact**: Complete accounting table available in `report/claims_table.csv`.

---

### Slide 15: Conclusions & Take-Home Lessons for Biomaterials BO
1. **Physical Synthesis Succeeded**: The authors successfully fabricated tissue-mimetic hydrogels; the laboratory wet-chemistry is authentic.
2. **Surrogates Were Overcredited**: Active learning acted primarily as local gradient descent; simpler 32-run RSM or Sobol search achieves comparable efficiency.
3. **Beware Soft-Regime $R^2$**: High global $R^2$ across wide dynamic ranges hides catastrophic local surrogate failure ($R^2 < 0$) in sensitive biological windows.
4. **Biostatistical Hygiene**: Always model nested replicates with hierarchical mixed-effects models—never pool cells across $N=3$ gels.
5. **Open Science & Adversarial Auditing**: Reproducibility requires publishing raw machine logs, exact seed registries, and verification cards.
