# Phase 6.4.2 — Candidate Sensitivity Diagnostics & Leakage-Safe Redesign Study

**Author / Role:** Supervisory Financial Model Risk Validator, ML Platform Engineer & Biostatistical Auditor  
**Evaluation Standard:** Federal Reserve SR 11-7 / OCC 2011-12 / Basel III Pillar 2 / FDA Group-Sequential Guidance  
**Repository Branch:** `research/phase6.4.2-candidate-sensitivity-study`  
**Base Commit:** `8192c80`  
**Deliverable Artifacts:**
- [`results/phase6_4_2/failure_case_diagnostics.json`](file:///d:/Financial%20Distress/results/phase6_4_2/failure_case_diagnostics.json)
- [`results/phase6_4_2/subgroup_recall_analysis.json`](file:///d:/Financial%20Distress/results/phase6_4_2/subgroup_recall_analysis.json)
- [`results/phase6_4_2/research_model_comparison.json`](file:///d:/Financial%20Distress/results/phase6_4_2/research_model_comparison.json)
- [`results/phase6_4_2/alert_budget_tradeoff.json`](file:///d:/Financial%20Distress/results/phase6_4_2/alert_budget_tradeoff.json)
- [`tests/test_phase6_4_2_candidate_sensitivity.py`](file:///d:/Financial%20Distress/tests/test_phase6_4_2_candidate_sensitivity.py)  

---

## Executive Summary

Phase 6.4.2 executes an in-depth supervisory model risk diagnostic investigation into the sensitivity divergence observed between the approved production baseline early warning system (`models/phase2/best_ews_model.joblib`) and the experimental shadow candidate (`Phase5_RF_CostSensitive_w10`). Specifically, this study resolves why the candidate model suffered a sensitivity collapse to $27.27\%$ (3 of 11 distress detected) compared to $45.45\%$ (5 of 11) for the production baseline during the $T=13 \to T=14$ surveillance evaluation.

### Core Audit Findings:
1. **Forensic Dissection of Failure Cases (Firms 28 and 60):**
   - Both firms appeared solvent and robustly healthy at prediction time $T=13$ ($FD_{T=13} = +1.4430$ and $+0.1357$, respectively) before experiencing sudden statutory collapse at $T=14$ ($FD_{T=14} = -0.6815$ and $-1.2361$).
   - **Failure Mechanism:** The candidate model assigned calibrated probabilities of $0.0496$ (Firm 28) and $0.0695$ (Firm 60). Because both scores are strictly below **both** the candidate threshold ($0.0894$) and the production threshold ($0.0800$), the failure is mathematically proven to be an intrinsic **probability-ranking failure**, NOT a threshold interaction. Lowering the candidate threshold to $0.0800$ still misses both firms.
   - **Feature Attribution Mechanism:** TreeSHAP analysis reveals that the candidate model's rolling backward stability features (e.g., $x_{16}$ roll mean 3, $x_{25}$ roll mean 3, $x_{10}$ roll mean 2) exerted massive risk-suppressing contributions (up to $-0.0246$ SHAP delta), because historically solvent metrics channeled these firms into negative leaves.
2. **Subgroup Recall Diagnostics:**
   - **Prior Borderline / Distressed Firms ($FD_{T-1} \le 0.0$):** Candidate recall matches production recall exactly ($50.0\%$, 3 of 6 detected) with superior precision ($0.3750$ vs $0.3000$).
   - **Prior Solvent / Healthy Firms ($FD_{T-1} > 0.0$):** Candidate recall collapses completely to **$0.0\%$ (0 of 5 detected)**, whereas the production baseline detects **$40.0\%$ (2 of 5 detected)**. The candidate model's sensitivity deficit is isolated entirely to **sudden corporate transitions**.
   - **Sector Concentration:** Sector 25 represents an acute stress cluster accounting for 5 of 11 distress events ($45.5\%$). Production detected 2 of 5; candidate detected 0 of 5.
3. **Research-Only Model Alternatives & Weight Sensitivity:**
   - Across cost-sensitive weight variations $w \in [1.0, 2.0, 5.0, 10.0]$, healthy-to-distressed recall remains strictly $0/5$ ($0.0\%$). Only $w=10$ at $0.0894$ complies with the operational 8% alert cap ($20 \le 20$).
   - Score-blending ensembles ($\alpha \ge 0.4$) restore detection of Firm 60, but generate 31 to 35 alerts ($11.8\%$ to $13.4\%$), heavily breaching the 8% capacity ceiling.
   - No evaluated alternative simultaneously meets the 8% alert budget AND matches baseline sensitivity on sudden corporate collapse.
4. **Final Governance Decision:**
   - Candidate promotion remains **STRICTLY REJECTED**. The approved production model remains frozen and authoritative.

---

## 1. Immutable Governance Controls

The approved production early warning system remains completely frozen and cryptographically verified against [`experiments/phase6/protected_artifacts_manifest.json`](file:///d:/Financial%20Distress/experiments/phase6/protected_artifacts_manifest.json):

| Artifact | Repository Path | Cryptographic SHA-256 Digest | Governance Disposition |
| :--- | :--- | :--- | :---: |
| **Production Model** | [`models/phase2/best_ews_model.joblib`](file:///d:/Financial%20Distress/models/phase2/best_ews_model.joblib) | `fcec88142b59f55f915e7a7301bc15bc583be9f3d925f91eeff59530db7fa6d3` | **VERIFIED / IMMUTABLE** |
| **Production Calibrator** | [`models/phase2/ews_calibrator.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_calibrator.joblib) | `774a411b14376591b99c68fbd5e1d19b5926bafeea431596bda2a162e6a79d1e` | **VERIFIED / IMMUTABLE** |
| **Production Preprocessor** | [`models/phase2/ews_preprocessor.joblib`](file:///d:/Financial%20Distress/models/phase2/ews_preprocessor.joblib) | `2342eb7f7b94657524a047d18b8760be55d82ea4045eae093b67b33edbc1659f` | **VERIFIED / IMMUTABLE** |
| **Threshold Configuration** | [`models/phase2/threshold_config.json`](file:///d:/Financial%20Distress/models/phase2/threshold_config.json) | `e60ed3b1f35e23012f096b492855c367006eb48f6cd877c92c107ea6eb96d136` | **VERIFIED / IMMUTABLE** |
| **Model Manifest** | [`models/phase2/model_manifest.json`](file:///d:/Financial%20Distress/models/phase2/model_manifest.json) | `aada1e00afdc3b07bf73eaa1b59755173b5e4e800aa3a5a2c988a8ddbee1a86c` | **VERIFIED / IMMUTABLE** |
| **EWS Performance Metrics** | [`models/phase2/ews_metrics.json`](file:///d:/Financial%20Distress/models/phase2/ews_metrics.json) | `5f9a2506114bfd690840401be24bb3f83d3684aab746204b942a0c8986e4ddd1` | **VERIFIED / IMMUTABLE** |
| **Risk Engine Rules** | [`config/risk_config.yaml`](file:///d:/Financial%20Distress/config/risk_config.yaml) | `9975b94587bf641146b0566cbaa0e5fcb3a280ce831015e2b2f45fc926ef08e8` | **VERIFIED / IMMUTABLE** |
| **Core System Config** | [`config/config.yaml`](file:///d:/Financial%20Distress/config/config.yaml) | `77f228b100a9989da85ca8c8da916e07949f2a93744181f07624f73e58d83864` | **VERIFIED / IMMUTABLE** |
| **Official Operating Threshold** | $\tau^* = \mathbf{0.0800}$ (Frozen on disk) | Verified unaltered (`operating_threshold: 0.08`) | **VERIFIED / FROZEN** |
| **Candidate Operating Threshold** | $\tau_{cand} = \mathbf{0.0894}$ (Frozen in shadow engine) | Candidate quarantined in read-only sandbox | **VERIFIED / FROZEN** |

Zero production retraining, recalibration, threshold adjustments, or artifact modifications were performed.

---

## 2. Reconstructing the Failure Cases (Firms 28 and 60)

A paired case-level audit of the two additional missed defaults from the Phase 6.4.1 surveillance cohort was executed:

### 2.1 Reconstructed Financial Telemetry Ledger

| Forensic Parameter | Case 01: Anonymized Firm 28 | Case 02: Anonymized Firm 60 |
| :--- | :---: | :---: |
| **Industry Sector Code ($x_{80}$)** | Sector 25 | Sector 25 |
| **Firm Size Proxy ($x_1$)** | $1.1032$ (Median Tercile) | $1.1309$ (Median Tercile) |
| **Prediction Timestamp** | $T=13$ (Strictly point-in-time) | $T=13$ (Strictly point-in-time) |
| **Statutory Outcome Timestamp** | $T=14$ | $T=14$ |
| **Antecedent Distress Score ($FD_{T=13}$)** | $\mathbf{+1.4430}$ (Strong Solvency) | $\mathbf{+0.1357}$ (Solvent) |
| **Realized Distress Score ($FD_{T=14}$)** | $\mathbf{-0.6815}$ (Statutory Default) | $\mathbf{-1.2361}$ (Severe Default) |
| **Transition Classification** | **Sudden Corporate Shock** | **Sudden Corporate Shock** |
| **Operating Cash Flow / Debt ($x_{12, T=13}$)** | $2.5090$ | $1.4770$ |
| **Leverage Ratio ($x_{2, T=13}$)** | $0.0322$ | $0.0228$ |
| **Working Capital / Assets ($x_{5, T=13}$)** | $0.0169$ | $0.0688$ |
| **Retained Earnings / Assets ($x_{16, T=13}$)** | $0.0356$ | $0.0501$ |
| **Production Raw Probability** | $0.4640$ | $0.5198$ |
| **Production Calibrated Probability** | $\mathbf{0.0858}$ | $\mathbf{0.1135}$ |
| **Production Alert Decision ($\tau^*=0.0800$)** | **DETECTED (Alert Issued)** | **DETECTED (Alert Issued)** |
| **Production Cohort Rank ($N=262$)** | Rank 35 (Top 12.98%) | Rank 16 (Top 5.73%) |
| **Candidate Raw Probability** | $0.1912$ | $0.2817$ |
| **Candidate Calibrated Probability** | $\mathbf{0.0496}$ | $\mathbf{0.0695}$ |
| **Candidate Alert Decision ($\tau=0.0894$)** | **MISSED (Suppressed Risk)** | **MISSED (Suppressed Risk)** |
| **Candidate Cohort Rank ($N=262$)** | Rank 51 (Top 19.08%) | Rank 35 (Top 12.98%) |
| **Rank Slippage ($\Delta\text{Rank}$)** | $\mathbf{+16\text{ positions}}$ | $\mathbf{+19\text{ positions}}$ |
| **Below Candidate Threshold ($0.0894$)** | **TRUE** | **TRUE** |
| **Below Production Threshold ($0.0800$)** | **TRUE** | **TRUE** |
| **Failure Mechanism Classification** | **PROBABILITY_RANKING_FAILURE** | **PROBABILITY_RANKING_FAILURE** |

### 2.2 Mathematical Proof of Probability-Ranking Domination
- For Firm 28: $P_{cand} = 0.0496 < 0.0800 < 0.0894$.
- For Firm 60: $P_{cand} = 0.0695 < 0.0800 < 0.0894$.
- **Finding:** Even if the candidate model's operating threshold were lowered to the production threshold ($\tau = 0.0800$), **both firms would still be missed.** The candidate model's failure on these firms is not an artifact of threshold calibration; it is driven by a fundamental drop in predicted probability ranking ($16$ and $19$ positions lower in the cohort).

### 2.3 TreeSHAP Feature Attribution Diagnostics

```
Firm 28 SHAP Decomposition:
  Production Model Top Positive (Risk-Increasing) Attributions:
    + x5  (Working Capital / Assets)      : SHAP = +0.0217 (Level weakness detected)
    + x16 (Retained Earnings / Assets)    : SHAP = +0.0209 (Level weakness detected)
    + x2_roll_mean_3 (Leverage rolling)   : SHAP = +0.0207
  Candidate Model Top Negative (Risk-Suppressing) Attributions:
    - x16_roll_mean_3 (Retained Earnings) : SHAP = -0.0246 (Healthy history suppressed risk!)
    - x25 (Sales / Assets Level)          : SHAP = -0.0173
    - x52_roll_mean_3                     : SHAP = -0.0163
    - x10_roll_mean_2                     : SHAP = -0.0147

Firm 60 SHAP Decomposition:
  Production Model Top Positive Attributions:
    + x12 (Operating Cash Flow / Debt)    : SHAP = +0.0233
    + x16 (Retained Earnings / Assets)    : SHAP = +0.0172
    + x2_roll_mean_3 (Leverage rolling)   : SHAP = +0.0164
  Candidate Model Top Negative Attributions:
    - x10_roll_mean_2                     : SHAP = -0.0161 (Backward solvency suppressed risk!)
    - x3_roll_mean_2                      : SHAP = -0.0149
    - x10 level                           : SHAP = -0.0144
    - x2_roll_mean_2                      : SHAP = -0.0104
```

**Diagnostic Interpretation:** The production model relies on level balance-sheet fragility indicators ($x_5, x_{16}, x_2, x_{12}$) that detect underlying vulnerability even in solvent firms. Conversely, the candidate model incorporates 26 point-in-time deterioration features; because Firms 28 and 60 had strong backward rolling ratios, the model's trees aggressively channeled them into negative leaves, resulting in probability suppression.

### 2.4 Data-Quality & Leakage Verification
- Missing values at inference time ($T=13$): **0.0% (Zero missingness)**.
- Future information peeking: **Zero**. Features are constructed strictly point-in-time using backward history ($T \le 13$).
- Confidentiality: Identifiable corporate names are excluded, adhering to supervisory audit standards.

---

## 3. Subgroup Recall & Sensitivity Diagnostics

To isolate the structural boundary conditions of the candidate model, the matured evaluation cohort ($N=262, N_D=11$) was segmented across financial health, industry sector, and firm size.

### 3.1 Subgroup Performance Ledger

| Diagnostic Subgroup | Cohort Share ($N$) | Distress Events ($N_D$) | Production Alerts | Production Recall | Production Precision | Candidate Alerts | Candidate Recall | Candidate Precision | Sensitivity Gap ($\Delta R$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prior Solvent / Healthy ($FD > 0$)** | 241 | 5 | 27 | **40.0% (2/5)** | 0.0741 | 12 | **0.0% (0/5)** | 0.0000 | **-40.0 pp (Collapse)** |
| **Prior Distressed / Borderline ($FD \le 0$)** | 21 | 6 | 10 | **50.0% (3/6)** | 0.3000 | 8 | **50.0% (3/6)** | 0.3750 | **0.0 pp (Parity)** |
| **Sector 25 (Primary Stress Cluster)** | 28 | 5 | 5 | **40.0% (2/5)** | 0.4000 | 2 | **0.0% (0/5)** | 0.0000 | **-40.0 pp (Collapse)** |
| **Other Industry Sectors ($x_{80} \ne 25$)** | 234 | 6 | 32 | **50.0% (3/6)** | 0.0938 | 18 | **50.0% (3/6)** | 0.1667 | **0.0 pp (Parity)** |
| **Small Firms ($x_1 \le 1.10$)** | 87 | 6 | 18 | **50.0% (3/6)** | 0.1667 | 10 | **50.0% (3/6)** | 0.3000 | **0.0 pp (Parity)** |
| **Medium Firms ($1.10 < x_1 \le 1.41$)** | 87 | 3 | 12 | **66.7% (2/3)** | 0.1667 | 7 | **0.0% (0/3)** | 0.0000 | **-66.7 pp (Collapse)** |
| **Large Firms ($x_1 > 1.41$)** | 88 | 2 | 7 | **0.0% (0/2)** | 0.0000 | 3 | **0.0% (0/2)** | 0.0000 | **0.0 pp** |
| **Entire Matured Cohort** | **262** | **11** | **37** | **45.45% (5/11)** | **0.1351** | **20** | **27.27% (3/11)** | **0.1500** | **-18.18 pp** |

### 3.2 Key Analytical Takeaways:
1. **Dichotomous Failure Profile:** The candidate model's recall deficit is **$100\%$ concentrated in historically solvent firms experiencing sudden collapse ($FD > 0$)**. On firms that showed pre-existing distress ($FD \le 0$), the candidate performs with identical sensitivity ($50.0\%$) and superior precision ($37.5\%$ vs $30.0\%$).
2. **Sector 25 Vulnerability:** Sector 25 represents an acute stress cluster accounting for $45.5\%$ of defaults. Because defaults in this sector occurred abruptly, the candidate model caught $0$ of $5$.

---

## 4. Evaluation of Research-Only Model Alternatives

In accordance with Section 4 directives, model redesign alternatives were investigated strictly within isolated research artifacts under company-isolated 5-Fold GroupKFold validation ($T \le 10$) and out-of-time evaluation ($T=13 \to T=14$).

### 4.1 Comparative Model Performance Ledger ($N=262, N_D=11$)

| Candidate Architecture | Weighting / Sample Policy | Operating Threshold ($\tau$) | Alerts Issued | Alert Rate | 8% Cap Compliance ($N \le 20$) | Distress Recall | Distress Precision | ROC-AUC | PR-AUC | Brier Score | Sudden Recall ($FD > 0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Approved Baseline Production** | Balanced Class Weight | $\mathbf{0.0800}$ | **37** | **14.12%** | **NON-COMPLIANT** | **45.45% (5/11)** | $0.1351$ | **0.7961** | **0.1816** | **0.0382** | **40.0% (2/5)** |
| **Phase5_RF_CostSensitive_w10 (Reg.)** | Cost Weight $w=10$ | $\mathbf{0.0894}$ | **20** | **7.63%** | **COMPLIANT** | **27.27% (3/11)** | **0.1500** | **0.7521** | **0.1410** | **0.0389** | **0.0% (0/5)** |
| Phase5_RF_CostSensitive_w10 (Study) | Cost Weight $w=10$ | $0.0800$ | 29 | 11.07% | NON-COMPLIANT | 27.27% (3/11) | $0.1034$ | 0.7521 | 0.1410 | 0.0389 | 0.0% (0/5) |
| Phase5_RF_Weight_w1 (Unweighted) | Standard RF ($w=1$) | $0.0894$ | 26 | 9.92% | NON-COMPLIANT | 27.27% (3/11) | $0.1154$ | 0.7403 | 0.1362 | 0.0403 | 0.0% (0/5) |
| Phase5_RF_Weight_w2 | Cost Weight $w=2$ | $0.0894$ | 26 | 9.92% | NON-COMPLIANT | 36.36% (4/11) | $0.1538$ | 0.7414 | 0.1388 | 0.0399 | 0.0% (0/5) |
| Phase5_RF_Weight_w5 | Cost Weight $w=5$ | $0.0894$ | 26 | 9.92% | NON-COMPLIANT | 36.36% (4/11) | $0.1538$ | 0.7588 | 0.1466 | 0.0391 | 0.0% (0/5) |
| Phase5_RF_Deterioration_ROS15 | Random Over-Sampling 15% | $0.0894$ | 52 | 19.85% | NON-COMPLIANT | 45.45% (5/11) | $0.0962$ | 0.7820 | 0.1415 | 0.0431 | 20.0% (1/5) |
| **Ensemble Blend $\alpha=0.50$** | $0.5 P_{base} + 0.5 P_{cand}$ | $0.0800$ | **33** | **12.60%** | **NON-COMPLIANT** | **36.36% (4/11)** | $0.1212$ | 0.7845 | 0.1654 | 0.0381 | **20.0% (1/5)** |
| **Ensemble Blend $\alpha=0.60$** | $0.6 P_{base} + 0.4 P_{cand}$ | $0.0800$ | **35** | **13.36%** | **NON-COMPLIANT** | **36.36% (4/11)** | $0.1143$ | 0.7882 | 0.1698 | 0.0381 | **20.0% (1/5)** |
| **Ensemble Blend $\alpha=0.80$** | $0.8 P_{base} + 0.2 P_{cand}$ | $0.0800$ | **35** | **13.36%** | **NON-COMPLIANT** | **36.36% (4/11)** | $0.1143$ | 0.7931 | 0.1772 | 0.0382 | **20.0% (1/5)** |

### 4.2 Research Model Findings:
1. **Weight Sensitivity Ineffectiveness:** Varying the cost-sensitive penalty $w \in [1, 2, 5, 10]$ on deterioration features does **NOT** resolve blindness to sudden default. Across all weights, recall on historically solvent firms remains strictly $0/5$ ($0.0\%$).
2. **Ensemble Blends:** Blending baseline and candidate probabilities recovers Firm 60 at $\alpha \ge 0.4$, but pushes alert volume to 31–35 alerts ($11.8\%$–$13.4\%$), breaching the 8% capacity limit.
3. **Impossibility on Existing Architectures:** No single standalone or blended model simultaneously achieves $\ge 45\%$ recall AND $\le 8\%$ alert volume on this cohort.

---

## 5. Operational Alert-Budget Trade-Off Ledger

Under supervisory guidelines, operational capacity enforces an absolute integer ceiling:
$$\text{maximum\_alerts} = \text{floor}(\text{cohort\_size} \times 0.08)$$
For $N=262$, the maximum allowable workload is **$\mathbf{20\text{ alerts}}$**.

### 5.1 Sensitivity vs. Alert Capacity Frontier

```
Operational Frontier (T=13 -> T=14, N=262, Cap = 20 Alerts):

Model                       Alerts  Alert Rate  Cap Compliant?  Recall    Healthy Recall
-----------------------------------------------------------------------------------------
Approved Baseline (tau=0.080)   37     14.12%        NO (Excess=+17)  45.45%     40.0% (2/5)
Approved Baseline (tau=0.100)   22      8.40%        NO (Excess=+2)   36.36%     20.0% (1/5)
Approved Baseline (tau=0.110)   18      6.87%        YES (Compliant)  27.27%     20.0% (1/5)
Approved Baseline (tau=0.114)   15      5.73%        YES (Compliant)  18.18%      0.0% (0/5)
-----------------------------------------------------------------------------------------
Candidate Model   (tau=0.080)   29     11.07%        NO (Excess=+9)   27.27%      0.0% (0/5)
Candidate Model   (tau=0.0894)  20      7.63%        YES (Compliant)  27.27%      0.0% (0/5)
Candidate Model   (tau=0.093)   19      7.25%        YES (Compliant)  27.27%      0.0% (0/5)
-----------------------------------------------------------------------------------------
Ensemble Blend    (tau=0.080)   33     12.60%        NO (Excess=+13)  36.36%     20.0% (1/5)
Ensemble Blend    (tau=0.095)   20      7.63%        YES (Compliant)  27.27%      0.0% (0/5)
```

### 5.2 Regulatory Evaluation of Threshold Manipulation:
1. **The Baseline Compliance Penalty:** To force the baseline production model to comply with the 20-alert cap, the threshold must be inflated from $0.0800$ to $\ge 0.1100$. This incurs an immediate **$18.18\text{ percentage point}$ recall collapse** (dropping from $45.45\%$ to $27.27\%$) and misses Firm 28.
2. **Prohibition of Post-Hoc Inflation:** Under SR 11-7 validation principles, adjusting thresholds post-hoc to retroactively force operational compliance is prohibited. The official threshold remains $\tau^* = \mathbf{0.0800}$.

---

## 6. Statistical Governance & Sample Size Caveats

1. **Protocol Status Remains UNVERIFIED:** As established in Phase 6.4.1, the mathematical discrepancy between two-sided reflection formulas ($Z_{0.025} = 1.95996$) and pure one-sided alpha-spending ($Z_{0.05} = 1.64485$) remains unresolved by external consensus. Therefore, the sequential testing boundaries remain marked **UNVERIFIED** and may not be used for candidate promotion.
2. **Small-Sample Event Uncertainty:**
   - Active surveillance window ($T=12 \to T=13$): $N_D = 8$ distress events.
   - Stress surveillance window ($T=13 \to T=14$): $N_D = 11$ distress events.
   - Total independent distress events across both cohorts: $N_D = 19$.
   - A sample of 11 distress events is statistically underpowered for confirmatory non-inferiority claims ($95\%$ Wilson binomial CI for candidate recall $[0.098, 0.566]$). All candidate comparisons are exploratory.

---

## 7. Model Risk Committee Final Acceptance Decision

Under Federal Reserve SR 11-7 and Basel III model governance guidelines, the seven required governance dimensions are formally disposed:

| Governance Dimension | Status | Primary Audit Finding & Rationale |
| :--- | :---: | :--- |
| **1. Production Baseline Integrity** | **PASSED** | All 8 protected artifacts match cryptographic manifest digests; official threshold $\tau^* = 0.0800$ frozen on disk. |
| **2. Failure-Case Explanation** | **RESOLVED** | Forensic reconstruction confirms misses on Firms 28 and 60 represent a probability-ranking collapse below both thresholds ($0.0496$ and $0.0695$), driven by rolling solvency feature suppression. |
| **3. Leakage Prevention** | **VERIFIED** | All features use strictly backward point-in-time information ($T \le 13$); zero future $T=14$ leakage confirmed. |
| **4. Research-Model Performance** | **EVALUATED** | Weight sensitivity ($w=1, 2, 5, 10$) and score blending evaluated. Deterioration features alone cannot detect sudden transitions from solvent states. |
| **5. Alert-Budget Feasibility** | **AUDITED** | Candidate at $0.0894$ meets 8% cap ($20 \le 20$), but at the cost of sensitivity collapse ($27.27\%$). Baseline achieves $45.45\%$ recall but breaches cap (37 alerts). |
| **6. Statistical Protocol Validity** | **UNVERIFIED / DISPUTED** | Methodological spending discrepancy remains unresolved; sequential boundaries prohibited from authorizing promotion. |
| **7. Candidate Promotion Eligibility** | **INELIGIBLE / REJECTED** | **Promotion is strictly denied.** The candidate model is disqualified due to severe sensitivity collapse on sudden defaults. |

### Strategic Roadmap for Future Architectural Redesign:
1. **Dual-Path Routing Architecture:** A future redesign should decouple chronic deterioration detection (Phase 5 feature engine) from sudden balance-sheet insolvency detection (Phase 2 baseline engine) via a formal hierarchical gating network.
2. **Pre-Registered Alert Budget Allocation:** Formalize a partitioned capacity budget (e.g., $4\%$ allocated to chronic deterioration alerts, $4\%$ to acute solvency shock alerts).
3. **Formal Committee Consensus on Group-Sequential Design:** Secure independent statistical consensus between Design 2 (pure one-sided spending) and Design 3 (discrete O'Brien-Fleming) prior to any future confirmatory evaluation.
