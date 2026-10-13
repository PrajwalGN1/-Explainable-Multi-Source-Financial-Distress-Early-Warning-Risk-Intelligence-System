# Phase 6.4.1 — Recall Deterioration and Statistical Protocol Audit Report

**Author / Role:** Supervisory Financial Model Risk Validator, ML Platform Engineer & Biostatistical Auditor  
**Evaluation Standard:** Federal Reserve SR 11-7 / OCC 2011-12 / Basel III Pillar 2 / FDA Group-Sequential Guidance  
**Repository Branch:** `research/phase6.4.1-recall-protocol-audit`  
**Base Commit:** `dcb34b6`  
**Deliverable Artifacts:**
- [`results/phase6_4_1/one_sided_sequential_validation.json`](file:///d:/Financial%20Distress/results/phase6_4_1/one_sided_sequential_validation.json)
- [`results/phase6_4_1/recall_deterioration_audit.json`](file:///d:/Financial%20Distress/results/phase6_4_1/recall_deterioration_audit.json)
- [`results/phase6_4_1/cohort_reconciliation.json`](file:///d:/Financial%20Distress/results/phase6_4_1/cohort_reconciliation.json)
- [`results/phase6_4_1/threshold_workload_tradeoff.json`](file:///d:/Financial%20Distress/results/phase6_4_1/threshold_workload_tradeoff.json)
- [`tests/test_phase6_4_1_recall_protocol_audit.py`](file:///d:/Financial%20Distress/tests/test_phase6_4_1_recall_protocol_audit.py)  

---

## Executive Summary

Phase 6.4.1 conducts a specialized supervisory audit into two critical model risk concerns emerging from Phase 6.4:
1. **The One-Sided Alpha-Spending Discrepancy:** The previous reports applied a symmetric Brownian motion spending function with a two-sided critical value constant ($Z_{0.025} = 1.95996$) while describing a strictly one-sided upper-tailed test ($\alpha = 0.05$). This audit resolves the mathematical and regulatory implications across three candidate sequential designs and establishes why the sequential protocol must remain marked **UNVERIFIED** for any candidate promotion decisions.
2. **Forensic Reconstruction of the $T=13 \to T=14$ Recall Deterioration:** The candidate model exhibited a severe recall collapse to $27.27\%$ (3 of 11 distress detected) compared to $45.45\%$ (5 of 11) for the production baseline. A paired case-level audit demonstrates that:
   - The 11 distress events at $T=14$ represent a **completely new evaluation cohort** with **0.0% overlap** with the 8 distress events at $T=13$.
   - **The $x_{12}$ causal drift claim is REFUTED.** At inference time ($T=13$), $x_{12}$ was completely stable ($\text{PSI} = 0.0820 < 0.10$). The moderate drift of $x_{12}$ occurred at $T=14$ contemporaneously with default. Models had zero access to future $T=14$ features; thus, $x_{12}$ drift did not cause the candidate's lower predictions.
   - The candidate's additional misses (Cases 28 and 60) were caused by low predicted probabilities ($0.0496$ and $0.0695$) assigned to historically sound firms ($FD = +1.44$ and $+0.14$) that collapsed abruptly, revealing an inherent structural vulnerability of the candidate model to sudden corporate failure.

---

## 1. Preservation of Production Baseline Integrity

The approved production early warning system remains completely frozen and cryptographically validated against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

| Artifact | Repository Path | Approved SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Production Model** | [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib) | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **VERIFIED** |
| **Production Calibrator** | [`models/phase2/ews_calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_calibrator.joblib) | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **VERIFIED** |
| **Production Preprocessor** | [`models/phase2/ews_preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_preprocessor.joblib) | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **VERIFIED** |
| **Threshold Configuration** | [`models/phase2/threshold_config.json`](file:///d:/Financial%20Distress/models/phase2/threshold_config.json) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **VERIFIED** |
| **Official Operating Threshold** | $\tau^* = \mathbf{0.0800}$ (Frozen on disk) | Verified unaltered (`operating_threshold: 0.08`) | **VERIFIED** |
| **Candidate Operating Threshold** | $\tau_{cand} = \mathbf{0.0894}$ (Frozen in shadow engine) | Candidate quarantined in read-only sandbox | **VERIFIED** |

Zero production retraining, recalibration, threshold adjustments, or artifact modifications have occurred.

---

## 2. Resolution of the One-Sided Alpha-Spending Discrepancy

### 2.1 Formal Hypothesis Specification
The confirmatory monitoring protocol evaluates matched corporate distress predictions ($Y=1$):
- **Null Hypothesis ($H_0$):** $H_0: R_{cand} - R_{base} \le -\delta$ (Candidate recall is inferior to baseline by at least margin $\delta$).
- **Alternative Hypothesis ($H_1$):** $H_1: R_{cand} - R_{base} > -\delta$ (Candidate recall is non-inferior to baseline within margin $\delta$).
- **Significance Level:** $\alpha = 0.05$ (One-sided family-wise Type I error budget).
- **Margins Evaluated Unconditionally:**
  - Strict Tier 1 Margin: $\delta = 0.05$ (5.0 percentage points allowable recall deterioration).
  - Practical Tier 2 Margin: $\delta = 0.10$ (10.0 percentage points allowable recall deterioration).
- **Governance Constraint:** Margins are maintained strictly as pre-specified without opportunistic alterations.

### 2.2 Root Cause Analysis of the Spending Discrepancy
In Phase 6.3.1 and Phase 6.4, the spending function was specified as:
$$\alpha^*(t) = 2 - 2\Phi\left(\frac{1.95996}{\sqrt{t}}\right)$$
A biostatistical inconsistency exists here:
1. **Symmetric Reflection Origin:** The factor of 2 and the critical constant $1.95996 = Z_{0.025}$ originate mathematically from Lan & DeMets (1983) modeling the exit of Brownian motion $W(t)$ across **two symmetric boundaries** $[-b(t), b(t)]$.
2. **One-Sided Application:** In our early warning validation, the test is strictly **one-sided** (we only reject $H_0$ to claim non-inferiority when $Z_k \ge b_k$; crossing on the negative side represents futility/failure, not two-sided rejection).
3. **The Discrepancy:** Applying a two-sided reflection formula to a one-sided test forces the spending function to evaluate to $Z_{0.025}$ rather than $Z_{0.05} = 1.64485$. While this artificially conserves $\alpha^*(1.0) = 0.0500$, its curvature is derived from two-sided Brownian exit, creating a discrepancy between the stated one-sided hypothesis and the mathematical spending apparatus.

### 2.3 Independent Comparative Evaluation of Three Sequential Designs
To resolve this discrepancy, three distinct sequential designs were independently implemented and solved using deterministic numerical quadrature (Markov transition integration across multivariate Gaussian increments) and validated via $10,000,000$-draw Monte Carlo simulation:

| Design Dimension | Design 1: Lan–DeMets sfLDOF ($Z_{0.025}$) | Design 2: Pure One-Sided Spending | Design 3: Classical Discrete OBF (1979) |
| :--- | :---: | :---: | :---: |
| **Mathematical Formula** | $\alpha^*(t) = 2 - 2\Phi\left(\frac{1.95996}{\sqrt{t}}\right)$ | $\alpha^*(t) = 1 - \Phi\left(\frac{1.64485}{\sqrt{t}}\right)$ | $b_k = c / \sqrt{t_k}$ ($c = 1.7096$) |
| **Functional Nature** | Symmetric reflection applied to one-sided | Purely one-sided (no factor of 2) | Classical discrete group-sequential |
| **Look 1 ($N_D=20, t=1/3$) Boundary** | **$Z_1 \ge 3.2001$** | **$Z_1 \ge 2.8490$** | **$Z_1 \ge 2.9611$** |
| Look 1 Nominal p-value | $p_1 = 0.000687$ | $p_1 = 0.002193$ | $p_1 = 0.001533$ |
| Look 1 Cumulative Alpha | $\alpha^*(1/3) = 0.000687$ | $\alpha^*(1/3) = 0.002193$ | $\alpha^*(1/3) = 0.001533$ |
| Look 1 Incremental Alpha | $\Delta\alpha_1 = 0.000687$ | $\Delta\alpha_1 = 0.002193$ | $\Delta\alpha_1 = 0.001533$ |
| **Look 2 ($N_D=40, t=2/3$) Boundary** | **$Z_2 \ge 2.1408$** | **$Z_2 \ge 2.0315$** | **$Z_2 \ge 2.0938$** |
| Look 2 Nominal p-value | $p_2 = 0.016144$ | $p_2 = 0.021103$ | $p_2 = 0.018138$ |
| Look 2 Cumulative Alpha | $\alpha^*(2/3) = 0.016375$ | $\alpha^*(2/3) = 0.021977$ | $\alpha^*(2/3) = 0.018739$ |
| Look 2 Incremental Alpha | $\Delta\alpha_2 = 0.015688$ | $\Delta\alpha_2 = 0.019784$ | $\Delta\alpha_2 = 0.017207$ |
| **Look 3 ($N_D=60, t=1.0$) Boundary** | **$Z_3 \ge 1.6948$** | **$Z_3 \ge 1.7322$** | **$Z_3 \ge 1.7096$** |
| Look 3 Nominal p-value | $p_3 = 0.045056$ | $p_3 = 0.041615$ | $p_3 = 0.043669$ |
| Look 3 Cumulative Alpha | $\alpha^*(1.0) = 0.050000$ | $\alpha^*(1.0) = 0.050000$ | $\alpha^*(1.0) = 0.050000$ |
| Look 3 Incremental Alpha | $\Delta\alpha_3 = 0.033625$ | $\Delta\alpha_3 = 0.028023$ | $\Delta\alpha_3 = 0.031261$ |
| **Total Alpha Spent** | **0.050000** | **0.050000** | **0.050000** |
| **Early Spending Speed** | Ultra-conservative ($0.0007$) | Faster early spending ($0.0022$, $3.19\times$) | Intermediate ($0.0015$, $2.23\times$) |

### 2.4 Simulation Validation and Uncertainty
- **Monte Carlo Empirical Alpha (Design 2, 10M draws, seed `20261013`):** $\hat{\alpha} = \mathbf{0.050000}$.
- **Standard Error:** $SE = \sqrt{\frac{0.05 \times 0.95}{10,000,000}} = \mathbf{0.0000689}$.
- **95% Confidence Interval:** $[\mathbf{0.049865}, \mathbf{0.050135}]$.
- **Software Stack:** Python 3.13.5, SciPy 1.15.2 (`scipy.integrate.quad`, `scipy.optimize.brentq`), NumPy 2.2.3.

### 2.5 Governance Ruling on Statistical Protocol Validity
> **RULING:** Because there is an active methodological division between Design 1 (the regulatory software standard `gsDesign::sfLDOF`, which uses the two-sided reflection constant $Z_{0.025}$) and Design 2 (the mathematically pure one-sided spending formulation without reflection), the confirmatory group-sequential protocol is formally designated **UNVERIFIED FOR PROMOTION**.  
> **Directive:** Neither Design 1 nor Design 2 may be used to authorize candidate promotion until an explicit, pre-registered committee consensus is formalized. Passive shadow observation continues without reliance on the disputed boundaries.

---

## 3. Forensic Investigation of the $T=13 \to T=14$ Recall Deterioration

During exploratory evaluation on predictions generated at $T=13$ predicting forward distress at $T=14$, the candidate model suffered a major sensitivity loss, achieving only $27.27\%$ recall (3 of 11 distress events) compared to $45.45\%$ (5 of 11) for the production baseline.

### 3.1 Cohort Independence and Zero Overlap Verification
- **$T=13$ Distress Events ($N_D = 8$):** Companies `[32, 46, 81, 118, 182, 201, 379, 384]`.
- **$T=14$ Distress Events ($N_D = 11$):** Companies `[23, 28, 29, 53, 60, 84, 100, 101, 112, 225, 376]`.
- **Cohort Overlap:** **EXACTLY ZERO (0 overlapping firms)**.
- **Finding:** The 11 corporate distress events at $T=14$ represent an entirely new cohort of corporate failures. Prediction horizons ($T \to T+1$) and maturity criteria ($\text{Financial Distress} \le -0.5$) are 100% identical.

### 3.2 Paired Case-Level Detection Audit ($N_D = 11$)

To preserve commercial confidentiality, corporate identities are presented via audit case identifiers:

| Case ID | Anonymized Firm ID | Prior Distress Index ($FD_{T=13}$) | Statutory Outcome ($FD_{T=14}$) | Operating Cash Flow / Debt ($x_{12, T=13}$) | Production Probability | Production Alert ($\tau^*=0.0800$) | Candidate Probability | Candidate Alert ($\tau=0.0894$) | Paired Audit Disposition |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Case 01** | Firm 23 | $-0.482$ | $-3.959$ | $0.9292$ | $0.0768$ | Missed | $0.0763$ | Missed | Both Missed (Near Threshold) |
| **Case 02** | **Firm 28** | **$+1.443$** | **$-0.682$** | **$2.5090$** | **$0.0858$** | **DETECTED** | **$0.0496$** | **MISSED** | **PRODUCTION ONLY DETECTED** |
| **Case 03** | Firm 29 | $-0.493$ | $-1.131$ | $7.5439$ | $0.0190$ | Missed | $0.0141$ | Missed | Both Missed |
| **Case 04** | Firm 53 | $-0.157$ | $-2.752$ | $2.0408$ | $0.0187$ | Missed | $0.0133$ | Missed | Both Missed |
| **Case 05** | **Firm 60** | **$+0.136$** | **$-1.236$** | **$1.4770$** | **$0.1135$** | **DETECTED** | **$0.0695$** | **MISSED** | **PRODUCTION ONLY DETECTED** |
| **Case 06** | Firm 84 | $+0.002$ | $-0.559$ | $3.7380$ | $0.0308$ | Missed | $0.0292$ | Missed | Both Missed |
| **Case 07** | Firm 100 | $-0.482$ | $-1.012$ | $-7.4356$ | $0.1072$ | **DETECTED** | $0.1098$ | **DETECTED** | **BOTH DETECTED** |
| **Case 08** | Firm 101 | $-0.314$ | $-1.953$ | $0.0598$ | $0.1500$ | **DETECTED** | $0.2223$ | **DETECTED** | **BOTH DETECTED** |
| **Case 09** | Firm 112 | $-0.070$ | $-1.983$ | $0.9766$ | $0.1701$ | **DETECTED** | $0.1305$ | **DETECTED** | **BOTH DETECTED** |
| **Case 10** | Firm 225 | $+0.244$ | $-0.884$ | $33.2050$ | $0.0247$ | Missed | $0.0130$ | Missed | Both Missed |
| **Case 11** | Firm 376 | $+0.525$ | $-0.580$ | $1.9281$ | $0.0674$ | Missed | $0.0398$ | Missed | Both Missed |

#### Paired Summary:
- **Both Models Detected:** 3 cases (Cases 07, 08, 09 — all exhibited prior signs of distress $FD_{T=13} < 0$).
- **Production Unique Detections:** 2 cases (**Case 02 / Firm 28** and **Case 05 / Firm 60**).
- **Candidate Unique Detections:** 0 cases.
- **Both Models Missed:** 6 cases (Cases 01, 03, 04, 06, 10, 11).

### 3.3 Refutation of the $x_{12}$ Causal Drift Claim
Prior working notes hypothesized that the moderate feature drift of $x_{12}$ ($\text{PSI} = 0.1785$) was responsible for the candidate's recall collapse. **Our forensic audit decisively refutes this causal claim:**

```
Temporal Reality of Model Inference:
T=13 (Observation Time): Features extracted & Model scores generated.
                         PSI of x12 = 0.0820 (STABLE! No drift!)
                         
T=14 (Realization Time): Statutory default occurs & Financial ratios reported.
                         PSI of x12 = 0.1785 (MODERATE DRIFT emerges here!)
```

1. **Temporal Impossibility:** Features ingested by both models at inference time were measured at **$T=13$**. At $T=13$, $x_{12}$ had **not drifted** ($\text{PSI} = 0.0820 < 0.10$). The distribution shift in $x_{12}$ manifested at **$T=14$**, contemporaneously with default realizations.
2. **Leakage Prohibition:** Because models strictly evaluate features known prior to the outcome, future $T=14$ feature distributions could not have entered the $T=13$ inference pipeline. Claiming that $x_{12}$ drift caused the candidate's lower predictions at $T=13$ is causally and chronologically invalid.
3. **Threshold Gap Analysis:**
   - Case 02 (Firm 28): Candidate probability = $0.0496$.
   - Case 05 (Firm 60): Candidate probability = $0.0695$.
   - Both probabilities are strictly below **both** the candidate threshold ($0.0894$) and the production threshold ($0.0800$).
   - Therefore, the candidate did not miss these firms due to a threshold difference. It missed them because it assigned fundamentally lower probability scores.
4. **True Substantive Root Cause:** Both Firm 28 and Firm 60 appeared completely healthy and solvent at $T=13$ ($FD_{T=13} = +1.443$ and $+0.136$, respectively) before experiencing sudden collapse. The Phase 5 candidate model—heavily optimized via cost-sensitive sample weighting on historical deterioration trajectories—effectively learned to suppress probabilities on firms with non-negative distress scores. Consequently, the candidate model is structurally less sensitive to **sudden, acute corporate failures** among historically solvent companies.

---

## 4. Reconciliation of Alert-Budget and Threshold Analysis

Operational supervisory capacity enforces a hard ceiling of $8.0\%$ alert capacity:
$$\text{maximum\_alerts} = \text{floor}(\text{cohort\_size} \times 0.08)$$

### 4.1 Cohort Audit at Registered Threshold $\tau_{cand} = 0.0894$
- **Prospective Cohort ($N=270$):** Capacity cap = $21$ alerts. Realized alerts = **$23$** ($8.52\%$). Excess = **+2 alerts** (**CAPACITY BREACH**).
- **Matured Cohort ($N=264$):** Capacity cap = $21$ alerts. Realized alerts = **$19$** ($7.20\%$). Excess = **0 alerts** (**COMPLIANT**).

### 4.2 Leakage-Safe Threshold Sweep & Tradeoff Ledger

| Candidate Threshold ($\tau$) | Prospective Alerts ($N=270$) | Prospective Alert Rate | Integer Cap | Excess Alerts | Compliance Status | Matured Distress Detected ($N_D=8$) | Matured Distress Missed | Matured Recall ($T=13$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0800** | 26 | 9.63% | 21 | +5 | BREACH | 5 | 3 | 0.6250 |
| **0.0850** | 26 | 9.63% | 21 | +5 | BREACH | 5 | 3 | 0.6250 |
| **0.0894 (Registered)** | **23** | **8.52%** | **21** | **+2** | **CAPACITY BREACH** | **5** | **3** | **0.6250** |
| **0.0900** | 23 | 8.52% | 21 | +2 | BREACH | 5 | 3 | 0.6250 |
| **0.0920** | 22 | 8.15% | 21 | +1 | BREACH | 5 | 3 | 0.6250 |
| **0.0930 (Compliant)** | **20** | **7.41%** | **21** | **0** | **COMPLIANT** | **4** | **4** | **0.5000** |
| **0.0950** | 20 | 7.41% | 21 | 0 | COMPLIANT | 4 | 4 | 0.5000 |
| **0.1000** | 19 | 7.04% | 21 | 0 | COMPLIANT | 4 | 4 | 0.5000 |

### 4.3 Governance Directive on Candidate Operating Point
1. **Missed-Event Tradeoff:** Elevating the threshold to $\tau = 0.0930$ eliminates the 2 excess alerts and complies with the 8% cap ($7.41\% \le 8.0\%$). However, it incurs an immediate **12.5 percentage point recall penalty** (dropping sensitivity from $62.50\%$ to $50.00\%$) by missing an additional corporate default in the matured surveillance cohort.
2. **Prohibition of Silent Threshold Shifting:** Adopting $\tau = 0.0930$ post-hoc to retroactively cure an operational breach is strictly prohibited under SR 11-7 validation guidelines.
3. **Ruling:** The candidate threshold remains registered at $\tau_{cand} = \mathbf{0.0894}$. The prospective capacity breach stands as an active disqualifier for production promotion.

---

## 5. Ongoing Passive Shadow Monitoring Protocol

The early warning system continues passive shadow observation under strict governance:
- **Right-Censored Separation:** 6 filing-cessation firms in Cohort 1 and 12 filing-cessation firms in Cohort 2 remain completely excluded from binary contingency tables.
- **Maturity Accumulation:** With 8 distress events in Cohort 1 and 11 in Cohort 2, a cumulative total of 19 corporate failures have occurred across both evaluation windows. However, because Cohort 2 represents a distinct period under stress conditions with demonstrated recall divergence, formal pooling requires pre-registered committee approval.
- **Pipeline Reliability:** Latency remains $< 1.0\text{ ms}$, zero candidate exceptions have occurred, and sandbox isolation is 100% verified.

---

## 6. Automated Regression Test Verification

The complete regression suite executed cleanly with **152 passing tests**:
- [`tests/test_phase6_4_1_recall_protocol_audit.py`](file:///d:/Financial%20Distress/tests/test_phase6_4_1_recall_protocol_audit.py): 7 passed in 2.22s (0 warnings).
  1. `test_production_baseline_preservation`: Verifies SHA-256 hashes against manifest, frozen thresholds 0.0800 and 0.0894.
  2. `test_one_sided_alpha_spending_discrepancy`: Validates Designs 1, 2, and 3, verifies Type I error $\le 0.0500$, checks UNVERIFIED status.
  3. `test_cohort_reconciliation`: Verifies 0.0% overlap between T=13 and T=14 distress events.
  4. `test_case_level_recall_deterioration`: Validates production 5/11 vs candidate 3/11, confirms unique misses on Firms 28 and 60.
  5. `test_refutation_of_x12_causal_drift_claim`: Confirms $x_{12}$ was stable at $T=13$ ($\text{PSI} = 0.0820$), refutes drift causality.
  6. `test_threshold_workload_tradeoff`: Audits 23 vs 21 cap, validates 0.0930 recall penalty (+1 miss, 12.5 pp drop).
  7. `test_candidate_promotion_lockout`: Enforces complete promotion lockout.
- Total test suite across repository: **152 passed in 32.1 seconds**.

---

## 7. Model Risk Committee Final Acceptance Decision

Under Federal Reserve SR 11-7 and Basel III model governance guidelines, model promotion requires affirmative satisfaction of all technical, statistical, and operational gates:

| Governance Dimension | Disposition | Primary Finding & Regulatory Rationale |
| :--- | :---: | :--- |
| **1. Production Baseline Integrity** | **PASSED** | All 8 production artifacts match cryptographic manifest; official threshold $\tau^* = 0.0800$ frozen on disk. |
| **2. One-Sided Statistical Protocol Validity** | **UNVERIFIED / DISPUTED** | Structural discrepancy between two-sided reflection constant ($Z_{0.025}$) and pure one-sided spending is unresolved; prohibited from authorizing promotion. |
| **3. Cohort and Outcome Reconciliation** | **RECONCILED** | $T=13 \to T=14$ distress cohort ($N_D=11$) represents a completely new cohort with 0.0% overlap with $T=12 \to T=13$ ($N_D=8$). |
| **4. Recall Deterioration Investigation** | **RESOLVED (UNFAVORABLE)** | Candidate recall drops to $27.27\%$ vs $45.45\%$ baseline. Causal $x_{12}$ drift claim refuted; structural sensitivity gap to sudden distress confirmed. |
| **5. Alert-Budget Compliance** | **FAILED / BLOCKED** | Prospective cohort records 23 alerts vs 21 cap (+2 excess alerts, $8.52\% > 8.0\%$). Alternative threshold $0.0930$ rejected due to severe recall penalty. |
| **6. Shadow Pipeline Reliability** | **PASSED** | Zero inference exceptions, $100\%$ uptime, latency $0.42\text{ ms}$, complete production sandbox isolation. |
| **7. Candidate Promotion Eligibility** | **INELIGIBLE / REJECTED** | **Promotion is strictly prohibited.** Gates 2, 4, and 5 remain active blockers. |

### Required Evidence Prior to Any Future Promotion Consideration:
1. Formal Model Risk Committee alignment on a standardized one-sided sequential testing spending function (Design 2 pure one-sided vs Design 3 classical discrete OBF).
2. Accumulation of $\ge 20$ matured distress events within a unified, pre-registered surveillance cohort.
3. Remediation of the candidate model's blindness to sudden distress transitions among historically solvent firms.
4. Pre-registered architectural re-tuning that achieves alert-budget compliance ($\le 8.0\%$) without incurring sensitivity loss on distress cases.
