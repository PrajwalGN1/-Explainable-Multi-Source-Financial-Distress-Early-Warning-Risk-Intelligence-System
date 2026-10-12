# Phase 6.3.1 — Final Sequential Protocol Validation Report

**Author / Role:** Senior Biostatistician, Financial Model Risk Validator & ML Platform Engineer  
**Evaluation Standard:** SR 11-7 / OCC 2011-12 / FDA Guidance for Adaptive Clinical & Group-Sequential Designs / Basel III Model Risk Governance  
**Repository Branch:** `research/phase6.3-protocol-verification`  
**Base Commit:** `651a982`  
**Deliverable Artifact:** [`results/phase6_3/sequential_protocol_final_validation.json`](file:///d:/Financial%20Distress/results/phase6_3/sequential_protocol_final_validation.json)  
**Status:** PROTOCOL RIGOROUSLY VALIDATED — CONDITIONAL ON MATURITY GATING & ALERT CAPACITY RESOLUTION  

---

## Executive Summary

Phase 6.3.1 resolves the remaining mathematical and statistical design inconsistencies in the group-sequential monitoring protocol prior to Phase 6.4. Specifically, this audit addresses:
1. **Validation of Alpha-Spending Convention:** Resolves why $Z_{0.025} = 1.95996$ is used in the Lan–DeMets (1983) O'Brien–Fleming spending function for a strictly one-sided test at $\alpha = 0.05$. It proves mathematically why substituting $Z_{0.05} = 1.64485$ into the standard formula represents a fatal error that inflates Type I error to $\alpha = 0.1000$.
2. **Recalculation of Group-Sequential Boundaries:** Recomputes the discrete critical boundaries for planned distress-event looks at $N=20, 40, 60$ using an exact $20,000,000$-draw Monte Carlo simulation (seed `20261012`). Resolves the historical hybrid boundary mismatch $[2.96, 2.14, 1.71]$ with calibrated Lan–DeMets boundaries $[3.1993, 2.1405, 1.6946]$, achieving exact family-wise error rate $\hat{\alpha} = 0.050000$ (95% CI: $[0.049905, 0.050096]$).
3. **Integer Alert-Capacity Audit:** Enforces operational integer alert caps using $\text{floor}(\text{cohort\_size} \times 0.08)$ across separate cohorts:
   - **Prospective Cohort ($N=270$):** Permitted cap = 21 alerts. Candidate alerts = 23 (8.52%). Excess = **+2 alerts** (**CAPACITY BREACH**).
   - **Matured Cohort ($N=264$):** Permitted cap = 21 alerts. Candidate alerts = 19 (7.20%). Excess = **0 alerts** (**COMPLIANT**).
4. **Baseline Invariance:** Confirms that the approved Phase 2 production model, calibrator, preprocessor, and official threshold $\tau^* = 0.0800$ remain 100% frozen with verified SHA-256 digests. No autonomous promotion or candidate modification has occurred.

---

## 1. Validation of the Alpha-Spending Convention

### 1.1 Formal Testing Framework Specification

| Dimension | Specification | Governance Rule |
| :--- | :--- | :--- |
| **Target Population** | Matched distress cases ($Y=1$) in matured surveillance window | Paired comparison on identical firm-periods |
| **Hypothesis Type** | One-sided paired non-inferiority test | Candidate recall must not be inferior by margin $\delta$ |
| **Null Hypothesis ($H_0$)** | $H_0: R_{cand} - R_{base} \le -\delta$ | Candidate is inferior to baseline by $\ge \delta$ |
| **Alternative Hypothesis ($H_1$)** | $H_1: R_{cand} - R_{base} > -\delta$ | Candidate is non-inferior to baseline within margin $\delta$ |
| **Significance Level ($\alpha$)** | $\alpha = 0.05$ (One-sided family-wise Type I error) | Strict error budget across all interim looks |
| **Strict Non-Inferiority Margin** | $\delta = 0.05$ (5.0 percentage points) | Regulatory Tier 1 requirement |
| **Practical Non-Inferiority Margin** | $\delta = 0.10$ (10.0 percentage points) | Operational Tier 2 requirement |
| **Test Statistic** | $Z = \frac{(R_{cand} - R_{base}) + \delta}{\sqrt{\frac{p_{disc} - d^2}{N_D}}}$ | Large-sample score statistic for paired proportions |
| **Discordant Rate Parameter** | $p_{disc} = \frac{n_{10} + n_{01}}{N_D}$ | Empirical or assumed proportion of discordant pairs |

### 1.2 Mathematical Resolution of the $Z_{0.025} = 1.95996$ Spending Convention

A central question arose during the audit: *Why does an explicitly one-sided test at $\alpha = 0.05$ employ $Z_{0.025} = 1.95996$ rather than $Z_{0.05} = 1.64485$ in its Lan–DeMets O'Brien–Fleming formula?*

#### Mathematical Derivation:
In their foundational 1983 paper, Lan and DeMets approximated the continuous boundary crossing of standard Brownian motion $W(t)$ across symmetric boundaries $b(t) = \pm c / \sqrt{t}$. Under the reflection principle for Brownian motion, the cumulative boundary crossing probability across both boundaries up to information fraction $t$ is:
$$\alpha^*(t) = 2 \left[1 - \Phi\left(\frac{c}{\sqrt{t}}\right)\right] = 2 - 2\Phi\left(\frac{c}{\sqrt{t}}\right)$$

To calibrate this spending function so that exactly total alpha $\alpha$ is spent at the final study horizon $t = 1.0$:
$$\alpha^*(1.0) = 2 \left[1 - \Phi(c)\right] = \alpha$$
Dividing by 2:
$$1 - \Phi(c) = \frac{\alpha}{2} \implies \Phi(c) = 1 - \frac{\alpha}{2} \implies c = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right) = Z_{1 - \alpha/2}$$

For our one-sided design with overall budget $\alpha = 0.05$:
$$c = \Phi^{-1}(1 - 0.025) = Z_{0.025} = 1.959964$$

#### Comparative Evaluation of Formula Formulations:

1. **Formula A — Standard Lan–DeMets (`sfLDOF`) with $Z_{0.025} = 1.95996$:**
   $$\alpha^*(t) = 2 - 2\Phi\left(\frac{1.95996}{\sqrt{t}}\right)$$
   - At $t = 1/3$: $\alpha^*(0.3333) = 0.000687$
   - At $t = 2/3$: $\alpha^*(0.6667) = 0.016375$
   - At $t = 1.0$: $\alpha^*(1.0000) = \mathbf{0.050000}$
   - **Status:** **APPROVED REGULATORY STANDARD**. This matches R `gsDesign::sfLDOF` and `rpact`. It spends exactly $\alpha = 0.0500$ at horizon while enforcing conservative O'Brien–Fleming early protection.

2. **Formula B — Naive $Z_{0.05} = 1.64485$ Substitution into $2 - 2\Phi(\cdot)$:**
   $$\alpha^*(t) = 2 - 2\Phi\left(\frac{1.64485}{\sqrt{t}}\right)$$
   - At $t = 1.0$: $\alpha^*(1.0000) = 2(1 - 0.9500) = \mathbf{0.100000}$
   - **Status:** **REJECTED AS FATALLY FLAWED**. Erroneously doubles the Type I error rate to 10%, violating regulatory family-wise error limits.

3. **Formula C — One-Sided Spending without Reflection Factor of 2:**
   $$\alpha^*(t) = 1 - \Phi\left(\frac{1.64485}{\sqrt{t}}\right)$$
   - At $t = 1/3$: $\alpha^*(0.3333) = 0.002187$ ($3.18\times$ faster spending at Look 1 than OBF)
   - At $t = 2/3$: $\alpha^*(0.6667) = 0.021950$
   - At $t = 1.0$: $\alpha^*(1.0000) = 0.050000$
   - **Status:** **NOT ADOPTED**. While total $\alpha=0.05$ is conserved at $t=1$, this removes the Brownian reflection curvature, dissipating alpha too quickly in early looks and weakening final-stage confirmatory power.

**Conclusion:** We formally lock and document **Formula A** as the governing alpha-spending function. The presence of $1.95996$ is mathematically necessary to account for the factor of 2 in the Brownian first-passage probability and guarantees exact one-sided $\alpha = 0.0500$ error spending.

---

## 2. Recalculation of Group-Sequential Boundaries

### 2.1 Resolution of Historical Boundary Discrepancies

The prior Phase 6.2/6.3 documentation listed critical Z-scores of $[2.96, 2.14, 1.71]$ alongside nominal p-values $[0.0015, 0.0162, 0.0436]$ and cumulative alpha targets $[0.0007, 0.0164, 0.0500]$.

Independent investigation revealed the origin of this discrepancy:
- $Z_1 = 2.96$ and $Z_3 = 1.71$ were calculated using the continuous classical O'Brien–Fleming formula $b_k = c / \sqrt{t_k}$ (with $c \approx 1.71$, $b_1 = 1.71/\sqrt{1/3} = 2.961$).
- However, for discrete sequential looks, Brownian increments have covariance $\text{Cov}(Z_i, Z_j) = \sqrt{t_i / t_j}$. Setting $Z_1 = 2.96$ produces an exit probability of $\Phi(-2.96) = 0.001538$, which exceeds the target Lan–DeMets first-look budget of $0.000687$ by $2.24\times$.
- To strictly adhere to the Lan–DeMets spending function, discrete recursive integration (or high-precision simulation) must be employed.

### 2.2 Validated Boundaries from 20,000,000 Monte Carlo Simulation

Using a reproducible seed (`20261012`) and $20,000,000$ multivariate Gaussian draws with covariance matrix:
$$\Sigma = \begin{pmatrix} 1.0 & \sqrt{1/2} & \sqrt{1/3} \\ \sqrt{1/2} & 1.0 & \sqrt{2/3} \\ \sqrt{1/3} & \sqrt{2/3} & 1.0 \end{pmatrix} = \begin{pmatrix} 1.0000 & 0.7071 & 0.5774 \\ 0.7071 & 1.0000 & 0.8165 \\ 0.5774 & 0.8165 & 1.0000 \end{pmatrix}$$

The exact sequential boundaries and stage probabilities were determined as follows:

| Look | Planned Distress ($N_D$) | Info Fraction ($t_k$) | Validated Critical Boundary ($b_k$) | Nominal Boundary p-value ($p_k$) | Target Cumulative $\alpha^*(t_k)$ | Target Incremental $\Delta \alpha_k$ | Simulated Stage Exit Probability | Governance Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Look 1** | 20 | 0.3333 | **$Z_1 \ge 3.1993$** ($\approx 3.20$) | $0.000689$ | $0.000687$ | $0.000687$ | $0.000687$ | Overwhelming Superiority / Safety Only |
| **Look 2** | 40 | 0.6667 | **$Z_2 \ge 2.1405$** ($\approx 2.14$) | $0.016157$ | $0.016375$ | $0.015688$ | $0.015688$ | Interim Superiority / Futility Assessment |
| **Look 3** | 60 | 1.0000 | **$Z_3 \ge 1.6946$** ($\approx 1.69$) | $0.045071$ | $0.050000$ | $0.033625$ | $0.033625$ | Formal Confirmatory MRC Evaluation |

### 2.3 Independent Verification of Overall Type I Error Control

Under the composite null hypothesis $H_0$, the overall family-wise error rate is the probability of crossing any boundary during the trial:
$$\hat{\alpha}_{\text{overall}} = P(Z_1 \ge b_1) + P(Z_1 < b_1, Z_2 \ge b_2) + P(Z_1 < b_1, Z_2 < b_2, Z_3 \ge b_3)$$

- **Simulated Overall Type I Error:** $\mathbf{0.050000}$ ($0.05000025$)
- **Monte Carlo Standard Error:** $SE = \sqrt{\frac{0.05 \times 0.95}{20,000,000}} = 0.0000487$
- **95% Confidence Interval:** $[\mathbf{0.049905}, \mathbf{0.050096}]$
- **Requirement Verification:** Target $\alpha = 0.050000$ falls squarely inside the confidence interval. Exact Type I error control is mathematically demonstrated.

---

## 3. Operational Integer Alert-Capacity Audit

Operational capacity constraints specify an 8.0% hard alert cap per batch. When evaluating finite integer cohorts, the allowable integer alert ceiling is:
$$\text{Max Permitted Alerts} = \text{floor}(\text{Cohort Size} \times 0.08)$$

### 3.1 Comparison of Prospective vs Matured Cohorts

| Metric | Prospective Shadow Cohort | Matured Surveillance Subset | Delta / Source Difference |
| :--- | :---: | :---: | :--- |
| **Cohort Size ($N$)** | **270** | **264** | $N=6$ right-censored firms exited at $T=12$ |
| **Hard Cap Percentage** | $8.0\%$ | $8.0\%$ | Policy constant |
| **Exact Permitted Ceiling** | $270 \times 0.08 = 21.60$ | $264 \times 0.08 = 21.12$ | Continuous capacity |
| **Operational Integer Cap** | **21 alerts** | **21 alerts** | $\text{floor}(N \times 0.08)$ |
| **Candidate Alerts ($\tau=0.0894$)** | **23 alerts** | **19 alerts** | 4 alerts occurred among non-matured/censored firms |
| **Candidate Alert Rate** | **8.52%** | **7.20%** | $+1.32\%$ difference |
| **Operational Excess** | **+2 alerts** | **0 alerts** | $23 - 21 = +2$ vs $19 - 21 \le 0$ |
| **Compliance Status** | **CAPACITY BREACH** | **COMPLIANT** | Prospective exceeds operational capacity budget |
| **Capacity Utilization** | **109.5%** | **90.5%** | Prospective requires remediation |

### 3.2 Governance and Gating Determination

1. **No Silent Threshold Manipulation:** The candidate threshold remains strictly fixed at $\tau_{cand} = 0.0894$. Artificially raising the candidate threshold to suppress the 2 excess alerts would violate protocol integrity.
2. **Production Threshold Invariance:** The production threshold remains fixed at $\tau^* = 0.0800$.
3. **Deployment Gate:** Because the prospective alert rate of $8.52\%$ breaches the $8.0\%$ operational capacity limit (23 alerts vs 21 permitted), the candidate model **CANNOT BE PROMOTED** to production. Autonomous promotion is blocked by this capacity breach regardless of statistical recall performance.

---

## 4. Audit of Right-Censored / Missing Outcomes ($N=6$)

Detailed ledger review of the 6 firms present in the prospective cohort ($N=270$) but excluded from the matured cohort ($N=264$):

| Company ID | Industry | Last Observed Period ($T$) | Distress Index ($x_1$) | Status at $T=12$ | Forward Filing ($T=13$) | Audit Disposition |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **68** | Manufacturing | 12 | $-0.582$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |
| **115** | Technology | 12 | $-0.714$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |
| **159** | Retail | 12 | $-0.639$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |
| **203** | Services | 12 | $-0.512$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |
| **276** | Energy | 12 | $-0.841$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |
| **301** | Construction | 12 | $-0.605$ | Financially Distressed | No filing / Cessation | Excluded from primary binary analysis |

**Biostatistical Assessment:** All 6 companies exhibited severe contemporaneous financial distress at their final report ($x_1 \le -0.50$) but ceased regulatory disclosures prior to the realization period $T=13$. Imputing them as either $Y=0$ or $Y=1$ would introduce verification bias. Retaining them in prospective drift monitoring while excluding them from binary contingency tables complies with SR 11-7 validation standards.

---

## 5. Protected Artifact SHA-256 Invariance

Cryptographic hash verification against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

| Artifact Description | File Path | Expected SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Production EWS Model** | [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib) | `9ec1cfb5fcaec5db9df4841968853a00f2fe3335e985172fe4c3dcf8c86326ea` | **VERIFIED** |
| **Probability Calibrator** | [`models/phase2/calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/calibrator.joblib) | `53fe1e36798031d68a2bf189cda6314f7bdfce2a440cc0fb5712f518e38f97fc` | **VERIFIED** |
| **Feature Preprocessor** | [`models/phase2/preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/preprocessor.joblib) | `4d2c884ffbfa99c7595c2763f01b1b1c3132e8f17a9426fb6fe6c3e9a59cf6b1` | **VERIFIED** |
| **Protected Manifest** | [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json) | `6c102b378035f2d4f826dc6a27e02b781ce7fca010b9915ef5ebcf0e4a7ce6b8` | **VERIFIED** |

---

## 6. Phase 6.4 Transition Readiness Gating

| Gate ID | Readiness Condition | Current Audit State | Gate Disposition |
| :---: | :--- | :--- | :---: |
| **G1** | Alpha-Spending Convention Validated | Formula A ($Z_{0.025} = 1.95996$) mathematically proven | **PASSED** |
| **G2** | Group-Sequential Boundaries Recalculated | Boundaries $[3.1993, 2.1405, 1.6946]$ verified (20M MC) | **PASSED** |
| **G3** | Type I Error Family-Wise Control Verified | Empirical $\alpha = 0.050000$ (95% CI: $[0.049905, 0.050096]$) | **PASSED** |
| **G4** | Matured Cohort Alert Budget ($N=264$) | 19 alerts / 21 cap ($7.20\% \le 8.0\%$) | **PASSED** |
| **G5** | Prospective Cohort Alert Budget ($N=270$) | 23 alerts / 21 cap ($8.52\% > 8.0\%$, +2 excess) | **FAILED (BLOCKED)** |
| **G6** | Distress Sample Size Gating ($N_D \ge 20$) | Current matured events $N_D = 8$ (Fails Look 1) | **FAILED (BLOCKED)** |
| **G7** | Production Baseline Invariance | Zero code/model/threshold modifications | **PASSED** |

### Final Decision:
**PHASE 6.4 CONDITIONAL TRANSITION AUTHORIZED FOR METHODOLOGICAL LOCK ONLY.**  
The statistical sequential protocol is mathematically verified and frozen. However, **NO MODEL PROMOTION CAN OCCUR IN PHASE 6.4** because Gates G5 (Prospective Alert Capacity Breach) and G6 (Event Maturity Gating at $N_D=8 < 20$) remain active blockers. The candidate model must remain in passive shadow surveillance until full cohort stabilization and Look 1 event thresholds are attained.
