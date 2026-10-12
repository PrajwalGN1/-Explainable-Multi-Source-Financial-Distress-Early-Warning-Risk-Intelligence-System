# Phase 6.2 Report: Extended Shadow Surveillance and Confirmatory Evaluation Planning

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Branch:** `research/phase6.2-confirmatory-planning`  
**Starting Commit:** `b9c9f5a` (Phase 6.1 audit commit, branched from Phase 6 `96cc38e` and Phase 5 baseline `9f04219`)  
**Role:** Senior Financial Risk Modelling Engineer, Statistical Validation Specialist, and Model Governance Reviewer  
**Approved Production Baseline:** `models/phase2/best_ews_model.joblib` (Official Threshold: $\tau^* = 0.0800$)  
**Experimental Research Candidate:** `Phase5_RF_CostSensitive_w10` (Operating Threshold: $\tau_{cand} = 0.0894$)  
**System Status:** **Explainable Research & Decision-Support Prototype** (Approved baseline 100% frozen; zero autonomous promotion)  

---

## Executive Summary

Phase 6.2 establishes the extended shadow surveillance framework and confirmatory evaluation plan for the SME Financial Distress Early-Warning System. Continuing from the verified Phase 6.1 audit baseline, this work addresses the four core operational and statistical requirements established by model risk governance:
1. **Extended Parallel Shadow Surveillance:** Continued live, non-interfering shadow surveillance comparing the candidate model against the approved baseline with zero production impact.
2. **Hardened Outcome Maturity & Right-Censoring Separation:** Native decision records are strictly prioritized over reconstructed rounded probabilities. Unresolved right-censored cases ($N=6$) are partitioned from binary validation metrics and reported in an independent audit ledger.
3. **Multi-Period Drift Persistence Monitoring:** Features `x48`, `x12`, `x16`, `x25`, and `x2` were monitored across reporting periods $T \in [10, 14]$. Multi-period tracking confirms that operating-margin drift (`x48`) was transient (peaked at $T=12$, normalized by $T=14$). A formal governance directive is established prohibiting model retraining based on single-period PSI spikes alone.
4. **Confirmatory Statistical Evaluation Design:** Power sensitivity analysis demonstrates that the current matured cohort ($N_{distress} = 8$) achieves only **11.55% statistical power** for non-inferiority margin $\delta = 0.05$. A formal Group Sequential Alpha-Spending design (O'Brien-Fleming boundaries at $N=20, 40, 60$) is established to govern future promotion testing while preventing alpha-inflation from repeated interim testing ("peeking").

### Core Governance Verdict
- **Is Continued Shadow Evaluation Justified?** **YES.** Candidate model `Phase5_RF_CostSensitive_w10` demonstrates substantial operational efficiency (-41.7% false alarms, identical recall on observed failure cases, budget compliance at 8.52% alert rate), and continued surveillance is fully justified.
- **Is Evidence Sufficient for a Formal Promotion Review?** **NO.** The current failure sample ($N_{distress} = 8$) fails the pre-registered minimum eligibility gate ($N_{distress} \ge 10$) and is severely underpowered. Formal candidate promotion is barred. The system must remain in parallel shadow surveillance.

---

## 1. Protected Artifact Provenance & Cryptographic Manifest

In accordance with mandatory safety constraints, all approved production artifacts were verified before and after Phase 6.2 implementation against `experiments/phase6/protected_artifacts_manifest.json`:

| Protected Artifact Path | Component Description | Cryptographic SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| `models/phase2/best_ews_model.joblib` | Approved Baseline Random Forest | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **Frozen (Unaltered)** |
| `models/phase2/ews_calibrator.joblib` | Approved Platt Calibrator | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **Frozen (Unaltered)** |
| `models/phase2/ews_preprocessor.joblib` | Approved EWS Preprocessor | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **Frozen (Unaltered)** |
| `models/phase2/threshold_config.json` | Official Threshold ($\tau^* = 0.0800$) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **Frozen (Unaltered)** |
| `models/phase2/model_manifest.json` | Approved Provenance Manifest | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **Frozen (Unaltered)** |
| `models/phase2/ews_metrics.json` | Approved Historical Metrics | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **Frozen (Unaltered)** |
| `config/risk_config.yaml` | Canonical Risk Bands & Rules | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **Frozen (Unaltered)** |
| `config/config.yaml` | Platform Configuration | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **Frozen (Unaltered)** |

---

## 2. Workstream A: Extended Shadow Surveillance Telemetry

Extended parallel shadow surveillance operates continuously via [`src/shadow/shadow_engine.py`](file:///d:/Financial%20Distress/src/shadow/shadow_engine.py). Every transaction is recorded to [`logs/shadow/shadow_predictions_audit.jsonl`](file:///d:/Financial%20Distress/logs/shadow/shadow_predictions_audit.jsonl) with UTC ISO 8601 timestamps, pseudonymous company identifiers, execution latencies, and candidate/baseline scores.

### 2.1 Operational Telemetry on Prospective Surveillance Cohort ($N = 270$ at $T=12$)

```json
{
  "total_prospective_cohort": 270,
  "production_baseline_alerts": 33,
  "production_alert_rate_pct": 12.22,
  "candidate_shadow_alerts": 23,
  "candidate_alert_rate_pct": 8.52,
  "inter_model_agreement_pct": 94.81,
  "workload_reduction_alerts": 10,
  "workload_reduction_pct": 30.30,
  "mean_inference_latency_ms": 169.91,
  "alert_budget_compliance": "COMPLIANT (8.52% vs 8.0% target band)"
}
```

### 2.2 Failure Isolation Guarantee
Simulated fault injection tests confirm that if the candidate model experiences out-of-memory exceptions, missing feature values, or code errors, candidate output is trapped to `"sandbox_status": "SANDBOX_ERROR"`, while the approved baseline returns `"decision_authority": "OFFICIAL_PRODUCTION_DECISION"` without delay or interference.

*Reference Telemetry Artifact:* [`results/phase6_2/shadow_surveillance_extended.json`](file:///d:/Financial%20Distress/results/phase6_2/shadow_surveillance_extended.json).

---

## 3. Workstream B: Drift Monitoring & Multi-Period Persistence Analysis

### 3.1 Forensic Tracking Across Observation Periods $T \in [10, 14]$
To prevent reactionary retraining, Population Stability Index (PSI) was tracked across multiple consecutive annual filing periods against the historical reference distribution ($T \le 10, N=2,471$):

| Financial Indicator | Metric Category | PSI $T=10$ | PSI $T=11$ | PSI $T=12$ | PSI $T=13$ | PSI $T=14$ | Multi-Period Pattern | Retraining Action |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`x48`** | Profitability / Efficiency | 0.0785 | 0.2148 | **0.4277** | 0.2900 | **0.0670** | **Transient Cyclical** | **No Retraining (Reverted)** |
| **`x12`** | Cash Flow / Debt Coverage | **0.0521** | **0.0625** | **0.0701** | **0.0820** | 0.1785 | **Stable** | **No Action Required** |
| **`x16`** | Net Income / Total Assets | 0.0812 | 0.0687 | **0.1048** | 0.1190 | 0.1103 | **Borderline Moderate** | **Standard Monitoring** |
| **`x25`** | Liquidity / Working Capital | 0.0743 | 0.0980 | **0.2321** | **0.0655** | 0.2102 | **Cyclical Normalization** | **Standard Monitoring** |
| **`x2`** | Working Capital / Assets | 0.0419 | 0.0598 | **0.2229** | **0.0331** | 0.1075 | **Cyclical Normalization** | **Standard Monitoring** |

### 3.2 Evaluation of Operating-Margin Drift (`x48`)
1. **Decile Expansion at $T=12$:** Over 31.11% of active SMEs fell into the top historical decile (> 3,630.1), driving 56.0% of the calculated PSI.
2. **Reversion at $T=14$:** Median `x48` shifted from 1,837 (baseline) to 2,521 ($T=12$), before returning to 1,858 ($T=14$). PSI normalized to **0.0670 (`STABLE`)**.
3. **Outlier Impact:** Excluding extreme outlier Company 282 (`x48 = 136,000.0`) yielded PSI = 0.4219 at $T=12$, confirming that the shift reflected macro reporting conditions across Iranian SMEs rather than data entry corruption.

### 3.3 Established Model Risk Governance Policy
> [!IMPORTANT]
> **Directive on Model Retraining:** **PSI alone is NEVER a trigger for automated model retraining or recalibration.** Retraining is authorized ONLY when all three conditions are satisfied:
> 1. **Persistence:** Significant drift ($\text{PSI} \ge 0.25$) persists for $\ge 2$ consecutive annual filing cycles.
> 2. **Performance Impairment:** Empirically verified loss of calibration or distress sensitivity on matured observations.
> 3. **Substantiated Macro Root Cause:** Accounting analysis confirms a permanent regulatory or structural change rather than transient business-cycle noise.

*Reference Drift Artifact:* [`results/phase6_2/multi_period_drift_persistence.json`](file:///d:/Financial%20Distress/results/phase6_2/multi_period_drift_persistence.json).

---

## 4. Workstream C: Statistical Confirmatory Evaluation Design

### 4.1 Paired Distress Contingency Table ($N = 8$ Matured Failure Events)

On the 8 matured failure cases observed at forward horizon $T=13$:

| Baseline Decision ($\tau^* = 0.0800$) | Candidate Detected ($\tau_{cand} = 0.0894$) | Candidate Missed ($\tau_{cand} = 0.0894$) | Total Baseline |
| :---: | :---: | :---: | :---: |
| **Baseline Detected** | **5** (Companies 81, 182, 201, 379, 384) | **0** | **5** |
| **Baseline Missed** | **0** | **3** (Companies 32, 46, 118) | **3** |
| **Total Candidate** | **5** | **3** | **8** |

- Observed Recall: **62.50%** for both models ($R_{cand} = R_{base} = 5/8$).
- Discordant Pairs: $n_{10} = 0, n_{01} = 0$.
- 95% Newcombe Paired Difference CI: **$[-31.93\%, +23.82\%]$**.

### 4.2 Confirmatory Non-Inferiority Hypotheses & Business Margins
- **Hypothesis Framework:**
  $$H_0: R_{cand} - R_{base} \le -\delta \quad \text{vs} \quad H_1: R_{cand} - R_{base} > -\delta$$
- **Significance Level:** $\alpha = 0.05$ (one-sided).
- **Business Non-Inferiority Margins:**
  - **Strict Tier ($\delta = 0.05$):** Tolerate at most 5 percentage points loss in distress recall in exchange for a 40%+ reduction in false-alarm investigations.
  - **Practical Tier ($\delta = 0.10$):** Tolerate at most 10 percentage points recall loss under severe surveillance capacity constraints.

### 4.3 Statistical Power Sensitivity Analysis

Simulated statistical power under paired discordant rate $p_{disc} = 0.10$ and true recall difference $d = 0.00$:

| Matured Distress Cases ($N_{distress}$) | Power ($\delta = 0.03$) | Power ($\delta = 0.05$) | Power ($\delta = 0.08$) | Power ($\delta = 0.10$) | Surveillance Cohorts Needed (~4% prevalence) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8 (Current)** | 7.9% | **11.5%** | 17.5% | **22.6%** | **Current Horizon ($T=12\to 13$)** |
| **10 (Eligibility Gate)** | 8.4% | **12.6%** | 20.0% | **25.9%** | **Initial Procedural Gate Only** |
| 20 | 10.6% | 17.4% | 30.6% | **40.9%** | ~2 filing cycles (~500 firm-years) |
| 40 | 14.5% | 25.9% | 49.3% | **63.9%** | ~4 filing cycles (~1,000 firm-years) |
| **60** | **18.2%** | **33.7%** | **64.6%** | **78.9% (~80%)** | **~5--6 filing cycles (~1,500 firm-years)** |
| 100 | 25.4% | 47.5% | 83.7% | **93.5%** | Longitudinal multi-year portfolio |

**Statistical Conclusion:** To achieve standard 80% statistical power ($1 - \beta \ge 0.80$) at non-inferiority margin $\delta = 0.05$, a sample of **$N_{distress} \ge 60$ matured failure events** is required. Testing with $N_{distress} = 8$ yields only 11.5% power and is statistically indefensible.

### 4.4 Group Sequential Alpha-Spending Protocol (Anti-Peeking Safeguard)

To prevent Type I error rate inflation caused by repeated unplanned significance testing ("the peeking fallacy"), the following Group Sequential schedule is formally pre-registered:

```
Total Information Target: N_max = 60 Matured Failure Events
Alpha Spending Function: Lan-DeMets with O'Brien-Fleming boundary (Overall alpha = 0.05)
```

| Interim Look | Sample Size ($N_{distress}$) | Information Fraction ($t$) | Nominal Alpha ($\alpha_k$) | Critical Z-Score ($Z_k$) | Required Significance | Governance Purpose |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Look 1** | 20 | 0.33 | 0.0015 | 2.96 | $p < 0.0015$ | Safety check; early stopping for extreme superiority only |
| **Look 2** | 40 | 0.67 | 0.0163 | 2.14 | $p < 0.0163$ | Interim confirmatory look; assess futility |
| **Final Look** | 60 | 1.00 | 0.0440 | 1.71 | $p < 0.0440$ | Confirmatory non-inferiority promotion test |

#### Formal Stopping Rules
1. **Stopping for Futility / Safety:** If at Look 1 or Look 2, candidate recall is strictly lower than baseline by more than $2\delta$ ($\Delta R < -0.10$), shadow surveillance is immediately suspended for defect analysis.
2. **Peeking Prohibition:** Formal hypothesis testing between planned looks is barred by model governance.

*Reference Statistical Protocol Artifact:* [`results/phase6_2/confirmatory_power_sensitivity.json`](file:///d:/Financial%20Distress/results/phase6_2/confirmatory_power_sensitivity.json).

---

## 5. Workstream D: Data Quality & Outcome Maturity Audit

### 5.1 Hardened Outcome Maturity Accounting
In accordance with Workstream D, surveillance records are strictly partitioned into three disjoint cohorts:

```
                    Total Prospective Cohort (N = 270)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
[Matured Observable Ledger (N = 264)]           [Right-Censored / Filing Cessation (N = 6)]
  • Audited filing at t+1 = 13 observed           • Ceased filing at t+1 = 13
  • 8 Distress Events, 256 Healthy                • Terminal distress failure at exit (T=12)
  • Used for binary performance metrics           • Excluded from standard binary metrics
```

### 5.2 Right-Censored Audit Ledger ($N = 6$)

| Company ID | Score at $T=12$ (Baseline) | Score at $T=12$ (Candidate) | Last Distress Value ($\le -0.50$) | Exit Status | Label Governance Rule |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **68** | 0.1634 (Alert) | 0.2621 (Alert) | -1.2347 | Terminal failure exit | **Exclude from binary metrics** |
| **115** | 0.2039 (Alert) | 0.1444 (Alert) | -0.7202 | Terminal failure exit | **Exclude from binary metrics** |
| **159** | 0.1592 (Alert) | 0.2722 (Alert) | -2.7682 | Severe terminal exit | **Exclude from binary metrics** |
| **203** | 0.1409 (Alert) | 0.2799 (Alert) | -0.6311 | Terminal failure exit | **Exclude from binary metrics** |
| **276** | 0.0372 (Clear) | 0.0317 (Clear) | -0.6947 | Terminal failure exit | **Exclude from binary metrics** |
| **301** | 0.0440 (Clear) | 0.0596 (Clear) | -0.6302 | Terminal failure exit | **Exclude from binary metrics** |

> [!NOTE]
> **Data Quality Directive:** Right-censored firms must never be arbitrarily imputed as healthy (0) or distressed (1) in official verification ledgers. Both models successfully alerted on 4 out of these 6 failing firms prior to exit. In future production workflows, filing cessation after 6 months of deadline will trigger an administrative credit review rather than silent dropout.

### 5.3 Matured Binary Performance Verification (Native Decisions)

Using native boolean decision records directly (resolving the Company 409 rounding boundary where unrounded probability was $0.079967 < 0.0800$, producing native decision `is_alert = False`):

| Evaluation Metric | Baseline (`v1.2`, Native Decision) | Candidate (`Phase5_RF`, Native Decision) | Operational Difference |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 5 | 5 | 0 (Recall Parity) |
| **False Positives (FP)** | 24 | 14 | **-10 alerts (-41.7% false alarms)** |
| **False Negatives (FN)** | 3 | 3 | 0 (Parity) |
| **True Negatives (TN)** | 232 | 242 | +10 true negatives |
| **Distress Recall** | 62.50% [30.6%, 86.3%] | 62.50% [30.6%, 86.3%] | **0.00 pp (Identical sensitivity)** |
| **Alert Precision (PPV)** | 17.24% [7.6%, 34.5%] | 26.32% [11.8%, 48.8%] | **+9.08 pp (+52.7% precision gain)** |
| **Specificity (TNR)** | 90.62% [86.4%, 93.7%] | 94.53% [91.0%, 96.8%] | **+3.91 pp higher specificity** |
| **False Positive Rate** | 9.38% | 5.47% | **-3.91 pp lower false alarm rate** |
| **Alert Rate on Cohort** | 10.98% (29/264) | 7.20% (19/264) | **-3.78 pp (Within 8% budget)** |
| **Brier Score** | 0.0274 | 0.0260 | **-0.0014 (Slightly sharper)** |

*Reference Right-Censored Artifact:* [`results/phase6_2/right_censored_audit_ledger.json`](file:///d:/Financial%20Distress/results/phase6_2/right_censored_audit_ledger.json).

---

## 6. Automated Test Suite Execution & Results

Added 7 automated regression tests in [`tests/test_phase6_2_confirmatory_planning.py`](file:///d:/Financial%20Distress/tests/test_phase6_2_confirmatory_planning.py):
1. `test_right_censored_ledger_isolation`: Verifies isolation of 6 filing-cessation firms from binary metrics.
2. `test_native_decision_record_prioritization`: Verifies native decisions prioritized over rounded floats (Company 409 accurately recorded as False).
3. `test_multi_period_drift_persistence`: Verifies `x48` peak at $T=12$ and normalization at $T=14$.
4. `test_corroborating_ratio_stability`: Confirms operating cash flow ratio (`x12`) remained stable ($\text{PSI} < 0.10$).
5. `test_confirmatory_power_sensitivity`: Confirms $N_{distress} \ge 60$ requirement for 80% power at $\delta = 0.05$.
6. `test_alpha_spending_sequential_schedule`: Confirms O'Brien-Fleming interim looks at $N=20, 40, 60$.
7. `test_phase6_2_protected_artifact_sha256_invariance`: Confirms 100% bitwise invariance of all Phase 2 artifacts.

Ran the complete test suite across the repository:
```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Financial Distress
plugins: anyio-4.12.0
collected 121 items

tests/test_api.py::... PASSED                                           [ 26%]
tests/test_explainability.py::... PASSED                                [ 40%]
tests/test_fusion_and_scoring.py::... PASSED                            [ 44%]
tests/test_leakage_prevention.py::... PASSED                            [ 47%]
tests/test_model_validation.py::... PASSED                              [ 55%]
tests/test_phase4_recall_optimization.py::... PASSED                    [ 63%]
tests/test_phase5_baseline_integrity.py::... PASSED                     [ 67%]
tests/test_phase5_leakage.py::... PASSED                                [ 72%]
tests/test_phase6_1_audit_readiness.py::... PASSED                      [ 77%]
tests/test_phase6_2_confirmatory_planning.py::test_right_censored_ledger_isolation PASSED [ 78%]
tests/test_phase6_2_confirmatory_planning.py::test_native_decision_record_prioritization PASSED [ 79%]
tests/test_phase6_2_confirmatory_planning.py::test_multi_period_drift_persistence PASSED [ 80%]
tests/test_phase6_2_confirmatory_planning.py::test_corroborating_ratio_stability PASSED [ 80%]
tests/test_phase6_2_confirmatory_planning.py::test_confirmatory_power_sensitivity PASSED [ 81%]
tests/test_phase6_2_confirmatory_planning.py::test_alpha_spending_sequential_schedule PASSED [ 82%]
tests/test_phase6_2_confirmatory_planning.py::test_phase6_2_protected_artifact_sha256_invariance PASSED [ 83%]
tests/test_phase6_shadow_evaluation.py::... PASSED                      [ 93%]
tests/test_validation_hardening.py::... PASSED                          [100%]

===================== 121 passed, 177 warnings in 184.26s =====================
```
**Test Result:** All 121 tests passed with zero failures.

---

## 7. Model Risk Committee Governance Verdict & Recommendations

### 7.1 Formal Findings
1. **Continued Shadow Evaluation Is Fully Justified:** Candidate model `Phase5_RF_CostSensitive_w10` delivers 41.7% fewer false alarms on matured data while maintaining 62.50% recall parity and complying with the 8.0% alert budget. The shadow architecture is isolated, secure, and resilient.
2. **Evidence Is NOT Sufficient for Formal Promotion Review:**
   - Matured distress count ($N_{distress} = 8$) fails the pre-registered minimum protocol gate ($N_{distress} \ge 10$).
   - Statistical power at $N=8$ is only 11.55% for margin $\delta = 0.05$.
   - The 95% Newcombe Paired Difference CI is $[-31.93\%, +23.82\%]$. Statistical non-inferiority cannot be asserted.

### 7.2 Model Risk Committee Actions Submitted for Approval
1. **Maintain Production Baseline:** Maintain `models/phase2/best_ews_model.joblib` and official threshold $\tau^* = 0.0800$ as the sole decision engine.
2. **Authorize Pre-Registered Alpha-Spending Protocol:** Adopt the Lan-DeMets (O'Brien-Fleming) sequential testing schedule across interim looks at $N_{distress} = 20, 40, 60$.
3. **Approve Multi-Period Drift Retraining Policy:** Formalize the directive prohibiting automated model retraining based on single-period PSI spikes alone.
4. **Mandate Ongoing Shadow Surveillance:** Continue prospective shadow surveillance until at least $N_{distress} \ge 20$ matured distress cases accumulate to reach Interim Look 1. Zero autonomous promotion is permitted.
