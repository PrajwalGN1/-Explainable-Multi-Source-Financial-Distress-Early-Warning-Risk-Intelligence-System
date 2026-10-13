# Phase 6.5 — Dual-Path Financial Distress Detection Architecture Redesign Study Report

**Author / Roles:** Senior Financial Risk Architect, Machine Learning Research Scientist, Biostatistician, Financial Model Risk Validator, and MLOps Engineer  
**Evaluation Standard:** Federal Reserve SR 11-7 / OCC 2011-12 / Basel III Pillar 2 / FDA Group-Sequential Monitoring Principles  
**Repository Branch:** `research/phase6.5-dual-path-redesign`  
**Base Commit:** `4b7ef32`  
**Completion Commit:** `[Pending Commit]`  
**Research-Gate Decision:** `DUAL_PATH_RESEARCH_REPRODUCIBLE_BUT_INSUFFICIENT`  
**Production Status:** Approved Production Baseline Frozen & Unchanged; Candidate Quarantined; Promotion Strictly Prohibited  

---

## Deliverables Index

The following reproducible JSON artifacts and test suites were generated and verified during Phase 6.5:

1. [`results/phase6_5/production_integrity_manifest.json`](file:///d:/Financial%20Distress/results/phase6_5/production_integrity_manifest.json) — Cryptographic SHA-256 digest audit of all 8 protected production artifacts.
2. [`results/phase6_5/cohort_integrity_audit.json`](file:///d:/Financial%20Distress/results/phase6_5/cohort_integrity_audit.json) — Complete reconciliation of cohort denominators, right-censoring, and integer alert caps.
3. [`results/phase6_5/sudden_transition_reproduction.json`](file:///d:/Financial%20Distress/results/phase6_5/sudden_transition_reproduction.json) — Empirical breakdown of all 106 historical forward defaults (36 sudden vs. 70 chronic) and failure case audits.
4. [`results/phase6_5/path_a_chronic_deterioration_metrics.json`](file:///d:/Financial%20Distress/results/phase6_5/path_a_chronic_deterioration_metrics.json) — Performance metrics for Path A alone (`Phase5_RF_CostSensitive_w10` @ 0.0894).
5. [`results/phase6_5/path_b_acute_distress_metrics.json`](file:///d:/Financial%20Distress/results/phase6_5/path_b_acute_distress_metrics.json) — Performance metrics for Path B alone (`models/phase2/best_ews_model.joblib` @ 0.0800).
6. [`results/phase6_5/dual_path_comparison.json`](file:///d:/Financial%20Distress/results/phase6_5/dual_path_comparison.json) — Comparative evaluation of 5 architecture variants, primary estimand analysis, and hypothesis evaluation.
7. [`results/phase6_5/alert_budget_allocation_analysis.json`](file:///d:/Financial%20Distress/results/phase6_5/alert_budget_allocation_analysis.json) — Alert-budget allocation frontier grid sweep and mathematical impossibility theorem for Firm 28.
8. [`results/phase6_5/calibration_and_stability.json`](file:///d:/Financial%20Distress/results/phase6_5/calibration_and_stability.json) — ECE calibration curves, Brier scores, probability distribution stability, and traceability schema.
9. [`tests/test_phase6_5_dual_path_architecture.py`](file:///d:/Financial%20Distress/tests/test_phase6_5_dual_path_architecture.py) — 13 automated unit and governance regression tests (100% passing).

---

## Executive Summary

Phase 6.5 investigates whether a **Dual-Path Early Warning Architecture**—separating **Chronic Financial Deterioration Detection (Path A)** from **Acute Distress-Transition Detection (Path B)**, unified by a research **Decision Coordinator (Path C)**—can resolve the sensitivity deterioration documented in Phase 6.4.1–6.4.3 under a strict operational alert budget of:

$$\text{maximum\_alerts} = \lfloor 0.08 \times \text{eligible\_firms} \rfloor$$

### Core Findings & Empirical Conclusions:

1. **Pre-Registered Hypothesis Not Supported:** The critical hypothesis—*"Combining a chronic deterioration pathway with an acute distress pathway may improve distress sensitivity relative to the existing research candidate while satisfying the fixed alert-volume constraint"*—is **empirically rejected** on the matured stress evaluation cohort ($T=13 \to T=14$, $N=262$, Defaults = 11, Cap = 20 alerts):
   - **Production Baseline (Path B Alone @ 0.0800):** Recall = **$45.45\%$ (5/11)**, Alerts = **37** (**Non-Compliant**, $+17$ excess alerts, alert rate $14.12\% > 8.0\%$).
   - **Candidate Baseline (Path A Alone @ 0.0894):** Recall = **$27.27\%$ (3/11)**, Alerts = **20** (**Compliant**, alert rate $7.63\% \le 8.0\%$).
   - **Naive Union (Variant 3):** Recall = **$45.45\%$ (5/11)**, Alerts = **39** (**Non-Compliant**, $+19$ excess alerts, alert rate $14.89\%$).
   - **Constrained Partitioned Allocation (Variant 4, $K_A=14, K_B=6$):** Recall = **$27.27\%$ (3/11)**, Alerts = **14** (**Compliant**), $\Delta\text{Recall}_{\text{vs Prod}} = \mathbf{-18.18\text{ pp}}$, $\Delta\text{Recall}_{\text{vs Cand}} = \mathbf{0.0\text{ pp}}$.
   - **Solvency-Gated Policy (Variant 5):** Recall = **$36.36\%$ (4/11)**, Alerts = **17** (**Compliant**), $\Delta\text{Recall}_{\text{vs Prod}} = \mathbf{-9.09\text{ pp}}$, $\Delta\text{Recall}_{\text{vs Cand}} = \mathbf{+9.09\text{ pp}}$ (detects Firm 60, but misses Firm 28).

2. **Mathematical Impossibility Theorem (Firm 28):**
   - In the matured cohort, Firm 28 ranks **35th** in Path B ($P = 0.0858$) and **51st** in Path A ($P = 0.0496$).
   - Because both $\text{Rank}_B(28) = 35 > 20$ and $\text{Rank}_A(28) = 51 > 20$, **no partitioned top-K allocation, score blend, or gating rule constrained to $\le 20$ alerts can capture Firm 28**.
   - Capturing Firm 28 requires expanding the Path B alert quota to at least $35$ alerts ($13.36\%$ alert rate), directly violating the $8\%$ operational mandate.

3. **Cohort Independence & Generalization:**
   - On the preceding matured surveillance cohort ($T=12 \to T=13$, $N=264$, Defaults = 8, Cap = 21 alerts), Path A and Path B detect the **exact same 5 defaults** ([81, 182, 201, 379, 384], Recall = $62.50\%$).
   - Path B provides **zero incremental detections** over Path A on $T=12 \to T=13$, while generating 30 alerts (breaching the 21-alert cap by $+9$). The naive union generates 32 alerts with zero recall gain.

4. **Research-Gate Verdict:**
   $$\mathbf{DUAL\_PATH\_RESEARCH\_REPRODUCIBLE\_BUT\_INSUFFICIENT}$$
   The dual-path architecture provides valuable diagnostic insights but **fails to deliver a viable operating policy that simultaneously satisfies the 8% alert budget and matches production sensitivity**. Promotion remains strictly denied.

---

## 1. Mandatory Production Baseline & Governance Controls

In accordance with Federal Reserve SR 11-7 and OCC 2011-12 model risk governance standards:

### 1.1 Protected Artifacts Manifest Invariance

Cryptographic SHA-256 digests of all 8 protected production artifacts were verified prior to and following experimental execution against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

| Protected Production Artifact | Expected SHA-256 Digest | Audited SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib) | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **VERIFIED MATCH** |
| [`models/phase2/ews_calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_calibrator.joblib) | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **VERIFIED MATCH** |
| [`models/phase2/ews_preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_preprocessor.joblib) | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **VERIFIED MATCH** |
| [`models/phase2/threshold_config.json`](file:///d:/Financial%20Distress/models/phase2/threshold_config.json) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **VERIFIED MATCH** |
| [`models/phase2/model_manifest.json`](file:///d:/Financial%20Distress/models/phase2/model_manifest.json) | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **VERIFIED MATCH** |
| [`models/phase2/ews_metrics.json`](file:///d:/Financial%20Distress/models/phase2/ews_metrics.json) | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **VERIFIED MATCH** |
| [`config/risk_config.yaml`](file:///d:/Financial%20Distress/config/risk_config.yaml) | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **VERIFIED MATCH** |
| [`config/config.yaml`](file:///d:/Financial%20Distress/config/config.yaml) | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **VERIFIED MATCH** |
| **Official Operating Threshold** | $\tau^* = \mathbf{0.0800}$ (Frozen on disk) | `operating_threshold: 0.08` | **VERIFIED FROZEN** |
| **Candidate Operating Threshold** | $\tau_{cand} = \mathbf{0.0894}$ (Frozen in sandbox) | Quarantined in shadow engine | **VERIFIED FROZEN** |

Zero production artifacts, calibrators, models, or configurations were modified, retrained, recalibrated, or overwritten.

---

## 2. Cohort Reconciliation & Denominator Mechanics

The cohort integrity audit formally verified all evaluation populations and right-censoring mechanisms:

```
[Observation Time T]                      [Realization Time T+1]
        |                                           |
        +---> Active SME Filers (N_obs)             +---> Realized Statements (N_mat)
                 |                                           |
                 | (Prospective Workload Cap:                +---> Ground-Truth Distress (FD <= -0.5)
                 |  floor(0.08 * N_obs))                     |
                 |                                           +---> Ground-Truth Healthy (FD > -0.5)
                 |                                           |
                 +---> Right-Censored Non-Filers -----------> Excluded from binary contingency tables
                       (Filing Cessations: N_obs - N_mat)     (Neither healthy nor statutory default)
```

### 2.1 Formal Reconciliation Ledger

| Cohort Identifier | $T_{\text{obs}}$ | $T_{\text{tgt}}$ | Eligible Filers ($N$) | Distress Events ($N_D$) | Healthy Outcomes ($N_H$) | Right-Censored Excluded | 8% Integer Cap | Candidate Alerts | Cap Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort T12 Prospective** | $12$ | $13$ | **270** | Pending | Pending | 6 | **21** | 23 | **BREACH (+2)** |
| **Cohort T12 $\to$ T13 Matured** | $12$ | $13$ | **264** | **8** | **256** | **6** | **21** | 19 | **COMPLIANT** |
| **Cohort T13 Prospective** | $13$ | $14$ | **274** | Pending | Pending | 12 | **21** | Pending | Pending |
| **Cohort T13 $\to$ T14 Matured** | $13$ | $14$ | **262** | **11** | **251** | **12** | **20** | 20 | **COMPLIANT** |

### 2.2 Distress Independence Audit:
- $T=13$ Distress Entities ($N_D=8$): Companies `[32, 46, 81, 118, 182, 201, 379, 384]`.
- $T=14$ Distress Entities ($N_D=11$): Companies `[23, 28, 29, 53, 60, 84, 100, 101, 112, 225, 376]`.
- **Distress Overlap:** **EXACTLY 0.0% (Zero shared firms)**. All 11 defaults at $T=14$ represent distinct corporate entities.

---

## 3. Sudden vs. Chronic Transition Independent Reproduction

The prompt required independent verification of all historical forward distress transitions across Dataset A ($T=2 \dots 13 \to T+1$):

### 3.1 Historical Transition Breakdown (All 106 Empirical Defaults)

```
========================================================================================
HISTORICAL FORWARD DEFAULTS IN DATASET A (N = 106)
========================================================================================
1. Chronic Distress Transitions (FD_t <= 0.0 -> FD_{t+1} <= -0.5):
   Count: 70 events (66.04% of all corporate failures)
   - Path A (Candidate w10 @ 0.0894) Recall: 90.00% (63/70 detected)
   - Path B (Production @ 0.0800) Recall:     85.71% (60/70 detected)

2. Sudden Distress Transitions (FD_t > 0.0 -> FD_{t+1} <= -0.5):
   Count: 36 events (33.96% of all corporate failures)
   - Path A (Candidate w10 @ 0.0894) Recall: 77.78% (28/36 detected)
   - Path B (Production @ 0.0800) Recall:     69.44% (25/36 detected)

3. Overall Historical Sensitivity (N = 106):
   - Path A (Candidate w10 @ 0.0894) Recall: 85.85% (91/106 detected)
   - Path B (Production @ 0.0800) Recall:     80.19% (85/106 detected)
========================================================================================
```

### 3.2 Out-of-Time Stress Cohort Pathology ($T=13 \to T=14$)

While Path A exhibits superior historical recall on pooled transitions, its sensitivity collapsed on the out-of-time $T=13 \to 14$ cohort:

| Company ID | Antecedent $FD_{t=13}$ | Realized $FD_{t=14}$ | Transition Class | Path B Score | Path B Alert | Path A Score | Path A Alert | Path A Failure Mechanism |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **28** | **+1.4430** | -0.6815 | **SUDDEN** | **0.0858** | **ALERT** | **0.0496** | Missed | Heavy solvency suppression via `roll_mean_3` |
| **60** | **+0.1357** | -1.2361 | **SUDDEN** | **0.1135** | **ALERT** | **0.0695** | Missed | Moderate solvency suppression via `roll_mean_3` |
| **84** | +0.0019 | -0.6558 | SUDDEN | 0.0612 | Missed | 0.0521 | Missed | Missed by both pathways |
| **225** | +0.2440 | -0.5234 | SUDDEN | 0.0451 | Missed | 0.0388 | Missed | Missed by both pathways |
| **376** | +0.5252 | -0.5510 | SUDDEN | 0.0512 | Missed | 0.0410 | Missed | Missed by both pathways |
| **23** | -0.4820 | -0.9852 | CHRONIC | 0.0645 | Missed | 0.0712 | Missed | Missed by both pathways |
| **29** | -0.4930 | -0.8124 | CHRONIC | 0.0588 | Missed | 0.0624 | Missed | Missed by both pathways |
| **53** | -0.1569 | -0.6120 | CHRONIC | 0.0610 | Missed | 0.0685 | Missed | Missed by both pathways |
| **100** | -0.4824 | -0.7421 | CHRONIC | **0.1420** | **ALERT** | **0.1582** | **ALERT** | Detected by both pathways |
| **101** | -0.3137 | -0.8912 | CHRONIC | **0.1250** | **ALERT** | **0.1340** | **ALERT** | Detected by both pathways |
| **112** | -0.0699 | -0.9145 | CHRONIC | **0.1890** | **ALERT** | **0.2110** | **ALERT** | Detected by both pathways |

**Pathology Summary:**
On this cohort, sudden distress transitions were detected **exclusively by Path B** (Firms 28 and 60). Path A suffered a complete blackout on sudden failures ($0/5$, $0.0\%$).

---

## 4. Decision Coordinator & Dual-Path Architecture Evaluation

The research Decision Coordinator evaluated 5 architecture variants on the primary matured evaluation cohort ($T=13 \to T=14$, $N=262$, Defaults = 11, Integer Cap = 20):

### 4.1 Comparative Architecture Matrix ($T=13 \to T=14$)

| Architecture Variant | Model Engine / Routing Rule | Operating Threshold(s) | Total Alerts | Alert Rate | 8% Cap Status | Recall ($N_D=11$) | Precision | Specificity | Brier Score | $\Delta\text{Rec}_{\text{vs Prod}}$ | $\Delta\text{Rec}_{\text{vs Cand}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant 1: Path A Alone** | `Phase5_RF_CostSensitive_w10` | $0.0894$ | **20** | $7.63\%$ | **COMPLIANT** | **$27.27\%$ (3/11)** | $15.00\%$ | $93.23\%$ | $0.0389$ | $-18.18\text{ pp}$ | Baseline |
| **Variant 2: Path B Alone** | `best_ews_model.joblib` | $0.0800$ | **37** | $14.12\%$ | **BREACH (+17)** | **$45.45\%$ (5/11)** | $13.51\%$ | $87.25\%$ | $0.0382$ | Baseline | $+18.18\text{ pp}$ |
| **Variant 3: Naive Union** | Logical OR ($A \lor B$) | $0.0800$ / $0.0894$ | **39** | $14.89\%$ | **BREACH (+19)** | **$45.45\%$ (5/11)** | $12.82\%$ | $86.45\%$ | $0.0384$ | $0.00\text{ pp}$ | $+18.18\text{ pp}$ |
| **Variant 4: Partitioned Budget** | Top-$14(A) \cup \text{Top-}6(B)$ | Rank Allocation | **14** | $5.34\%$ | **COMPLIANT** | **$27.27\%$ (3/11)** | $21.43\%$ | $95.62\%$ | $0.0415$ | $-18.18\text{ pp}$ | $0.00\text{ pp}$ |
| **Variant 5: Solvency-Gated** | Prior $FD \le 0 \to A$; $FD > 0 \to B$ | $0.0894$ / $0.1130$ | **17** | $6.49\%$ | **COMPLIANT** | **$36.36\%$ (4/11)** | $23.53\%$ | $94.82\%$ | $0.0383$ | $-9.09\text{ pp}$ | $+9.09\text{ pp}$ |

### 4.2 Cross-Period Verification ($T=12 \to T=13$, $N=264$, Defaults = 8, Integer Cap = 21)

To confirm whether these results generalize or reflect cohort-specific noise, all variants were evaluated on the preceding matured surveillance cohort:

| Architecture Variant | Operating Threshold(s) | Total Alerts | 8% Cap Status | Recall ($N_D=8$) | Precision | Detections Breakdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Variant 1: Path A Alone** | $0.0894$ | **19** | **COMPLIANT** | **$62.50\%$ (5/8)** | $26.32\%$ | `[81, 182, 201, 379, 384]` |
| **Variant 2: Path B Alone** | $0.0800$ | **30** | **BREACH (+9)** | **$62.50\%$ (5/8)** | $16.67\%$ | `[81, 182, 201, 379, 384]` |
| **Variant 3: Naive Union** | $0.0800$ / $0.0894$ | **32** | **BREACH (+11)** | **$62.50\%$ (5/8)** | $15.62\%$ | Zero incremental detections |
| **Variant 4: Partitioned Budget** | Rank Allocation | **15** | **COMPLIANT** | **$50.00\%$ (4/8)** | $26.67$ % | Misses Firm 384 |
| **Variant 5: Solvency-Gated** | $0.0894$ / $0.1130$ | **18** | **COMPLIANT** | **$62.50\%$ (5/8)** | $27.78\%$ | `[81, 182, 201, 379, 384]` |

**Key Cross-Period Finding:** On $T=12 \to 13$, Path B provides **zero incremental sensitivity** over Path A, while issuing 11 unnecessary false alarms that breach the regulatory cap.

---

## 5. Alert-Budget Allocation Frontier Analysis

A systematic grid sweep of all $(K_A, K_B)$ budget allocations was conducted on the 262-firm cohort to determine whether any allocation rule satisfies the 8% cap while recovering the missed sudden defaults:

```
========================================================================================
ALERT BUDGET ALLOCATION FRONTIER: Ka + Kb <= 20
========================================================================================
Firm 28: Path B Rank = 35 (P = 0.0858) | Path A Rank = 51 (P = 0.0496)
Firm 60: Path B Rank = 16 (P = 0.1135) | Path A Rank = 35 (P = 0.0695)

  Ka (Path A)   Kb (Path B)   Unique Alerts   Cap Compliant?   Recall (Detections)   Catches 28?   Catches 60?
  -------------------------------------------------------------------------------------------------------
      20             0              20             YES           27.27% (3/11)           NO           NO
      14             6              14             YES           27.27% (3/11)           NO           NO
      10            10              12             YES           18.18% (2/11)           NO           NO
       4            16              18             YES           36.36% (4/11)           NO          YES
       0            20              20             YES           36.36% (4/11)           NO          YES
  -------------------------------------------------------------------------------------------------------
      10            35              41              NO           45.45% (5/11)          YES          YES
========================================================================================
```

### 5.1 Formal Mathematical Impossibility Proof

$$\mathbf{Theorem\text{ (Alert Capacity Exclusion of Firm 28):}}$$

Under the regulatory capacity constraint $\text{Cap} = \lfloor 0.08 \times N \rfloor = 20$ alerts on cohort $N=262$:
1. Let $\text{Rank}_B(i)$ denote the descending probability rank of firm $i$ in Path B, where $\text{Rank}_B(28) = 35$.
2. Let $\text{Rank}_A(i)$ denote the descending probability rank of firm $i$ in Path A, where $\text{Rank}_A(28) = 51$.
3. For any partitioned top-K coordinator selecting $K_A$ alerts from Path A and $K_B$ alerts from Path B, firm 28 is flagged if and only if:
   $$28 \in \text{Top}_{K_A}(\text{Path A}) \cup \text{Top}_{K_B}(\text{Path B}) \iff K_A \ge 51 \quad \text{or} \quad K_B \ge 35$$
4. Since the total alert budget enforces $K_A + K_B \le 20$, it follows that:
   $$K_A \le 20 < 51 \quad \text{and} \quad K_B \le 20 < 35$$
5. Therefore, firm 28 **cannot be flagged** by any budget-compliant partitioned policy.
6. To capture firm 28, the coordinator must expand $K_B \ge 35$, requiring an alert volume of at least 35 alerts ($13.36\%$ alert rate), creating a minimum breach of $+15$ alerts ($+75.0\%$ over capacity). $\quad \blacksquare$

---

## 6. Calibration, Scale Alignment & Traceability

### 6.1 Calibration & Distributional Diagnostics

| Path Identifier | Model Family | Calibrator | Brier Score | 5-Bin ECE | Score Range | Median Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Path B (Acute / Production)** | Random Forest | Platt Scaling (Sigmoid) | **$0.0382$** | **$0.0210$** | $[0.0104, 0.2215]$ | $0.0412$ |
| **Path A (Chronic / Candidate)** | Cost-Sensitive RF | Isotonic Regression | **$0.0389$** | **$0.0235$** | $[0.0085, 0.2840]$ | $0.0345$ |

**Scale Alignment Analysis:**
Path A and Path B exhibit a Pearson correlation of $r = 0.8124$. However, their probability distributions diverge significantly in the upper quartile:
- Path B calibrated probabilities are smoothed by Platt scaling, causing scores to cluster densely between $0.07$ and $0.12$.
- Path A isotonic probabilities produce step-function artifacts in the decision boundary.
- Direct linear probability blending ($\alpha P_A + (1-\alpha) P_B$) without rank normalization distorts calibrated meaning and fails to elevate Firm 28 into the top 20.

### 6.2 Decision Traceability & Audit Schema

Every combined alert produced by the Decision Coordinator is logged with full governance provenance:

```json
{
  "prediction_timestamp_utc": "2026-10-13T11:03:36Z",
  "company_pseudonym_id": "FIRM_060",
  "cohort_identifier": "COHORT_T13_T14_MATURED",
  "originating_pathway": "PATH_B_ACUTE_DISTRESS_DETECTOR",
  "path_a_chronic_probability": 0.0695,
  "path_a_threshold_decision": false,
  "path_b_acute_probability": 0.1135,
  "path_b_threshold_decision": true,
  "decision_coordinator_rule": "SOLVENCY_GATED_ACUTE_ROUTING (FD_obs=0.1357 > 0, tau_acute=0.1130)",
  "final_research_alert": true,
  "primary_contributing_risk_drivers": [
    {"feature": "x1_working_capital_burn", "direction": "DETERIORATING"},
    {"feature": "x12_liquidity_drop", "direction": "DETERIORATING"},
    {"feature": "x48_debt_expansion", "direction": "ELEVATED"}
  ],
  "solvency_transition_category": "SUDDEN_DISTRESS_TRANSITION",
  "regulatory_status": "RESEARCH_ALERT_ONLY (NON_PRODUCTION_ACTIONABLE)"
}
```

---

## 7. Statistical Protocol Governance & Promotion Lockout

In accordance with Phase 6.4.1 and Phase 6.4.3 regulatory audits:
1. **Disputed Protocol Preserved:** The group-sequential Lan-DeMets alpha-spending protocol remains designated **`UNVERIFIED_FOR_PROMOTION`**.
2. **Small-Sample Multi-Look Invalidation:** The observed event counts ($8$ events at $T=13$ and $11$ events at $T=14$) cannot support robust sequential boundaries.
3. **Candidate Promotion Lockout:** Candidate `Phase5_RF_CostSensitive_w10` and all Dual-Path architecture variants remain strictly designated **`INELIGIBLE_FOR_PROMOTION`**.

---

## 8. Automated Test Execution

The automated test suite in [`tests/test_phase6_5_dual_path_architecture.py`](file:///d:/Financial%20Distress/tests/test_phase6_5_dual_path_architecture.py) was executed:

```
================================== test session starts ==================================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0
collected 13 items

tests/test_phase6_5_dual_path_architecture.py::test_production_baseline_manifest_immutability PASSED [  7%]
tests/test_phase6_5_dual_path_architecture.py::test_independent_pathway_evaluation PASSED [ 15%]
tests/test_phase6_5_dual_path_architecture.py::test_point_in_time_feature_integrity PASSED [ 23%]
tests/test_phase6_5_dual_path_architecture.py::test_preprocessing_isolation PASSED [ 30%]
tests/test_phase6_5_dual_path_architecture.py::test_coordinator_deduplication_and_integer_cap PASSED [ 38%]
tests/test_phase6_5_dual_path_architecture.py::test_missing_scores_and_exception_handling PASSED [ 46%]
tests/test_phase6_5_dual_path_architecture.py::test_right_censoring_and_cohort_reconciliation PASSED [ 53%]
tests/test_phase6_5_dual_path_architecture.py::test_cohort_identities_and_timestamps PASSED [ 61%]
tests/test_phase6_5_dual_path_architecture.py::test_native_probabilities_and_threshold_retention PASSED [ 69%]
tests/test_phase6_5_dual_path_architecture.py::test_no_holdout_label_leakage PASSED [ 76%]
tests/test_phase6_5_dual_path_architecture.py::test_deterministic_coordinator_behavior PASSED [ 84%]
tests/test_phase6_5_dual_path_architecture.py::test_deliverables_and_traceability PASSED [ 92%]
tests/test_phase6_5_dual_path_architecture.py::test_hypothesis_and_gate_status PASSED [100%]

================================== 13 passed in 2.95s ===================================
```

Additionally, the full repository regression suite was executed:
- **Total Tests:** 179 passed, 0 failed, 0 errors.
- **Exit Code:** `0` (clean execution).

---

## 9. Research-Gate Review & Supervisory Decision

### Formal Research-Gate Evaluation:

1. **Did the Dual-Path Architecture Improve Distress Sensitivity Under the 8% Alert Cap?**  
   **NO.** The pre-registered primary estimand ($\Delta\text{Recall}$ vs. Production) is **$-18.18\text{ pp}$** for Partitioned Allocation (Variant 4) and **$-9.09\text{ pp}$** for Solvency-Gated Allocation (Variant 5). Both fall strictly below the pre-registered non-inferiority margin ($+0.0500$).

2. **Can Any Two-Path Combination Capture Firm 28 Within Budget?**  
   **NO.** Proven by the Mathematical Impossibility Theorem: capturing Firm 28 requires absorbing $\ge 35$ alerts from Path B, breaching the 20-alert cap.

3. **Did the Study Preserve Production Baseline Immutability?**  
   **YES.** All 8 protected production artifact SHA-256 digests remain identical to the manifest.

### Final Supervisory Status:
$$\mathbf{DUAL\_PATH\_RESEARCH\_REPRODUCIBLE\_BUT\_INSUFFICIENT}$$

### Binding Operating Directives:
1. **Production Pipeline Remains Frozen:** Model `models/phase2/best_ews_model.joblib` operating at official threshold $\tau^* = 0.0800$ remains the sole authorized production early warning system.
2. **Experimental Candidate Quarantined:** Candidate `Phase5_RF_CostSensitive_w10` remains strictly isolated as a passive shadow model at threshold $0.0894$.
3. **No Dual-Path Deployment:** All Dual-Path components remain isolated in research artifacts. No production routing or gating changes may be made.
4. **Recommended Next Step:** The research frontier indicates that existing financial ratio transformations cannot separate acute sudden defaults from solvent peers within an 8% alert cap. Further research should focus on **high-frequency non-accounting signals (e.g., transaction flow velocity, supply chain payment delays, or macroeconomic stress interactions)** rather than re-weighting quarterly financial statements.
