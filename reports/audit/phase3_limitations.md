# Phase 3 Audit: Explainability Limitations & Methodological Disclosures

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Branch:** `upgrade/v1.3-explainability`  
**Audit Date:** October 12, 2026  
**Auditor Roles:** Senior Data Scientist, Financial Risk Analytics Specialist, ML Research Auditor  

---

## 1. Executive Summary

While Phase 3 establishes native, mathematically verified TreeSHAP explanations with exact additivity, the system operates as a decision-support and surveillance prototype. This document establishes four mandatory methodological boundaries that credit risk officers, model risk auditors, and financial reviewers must observe.

---

## 2. Mandatory Methodological Disclosures

### 2.1 Model Attribution is NOT Real-World Causation
> [!WARNING]
> **Core Statistical Principle:**
> TreeSHAP values quantify how the specific mathematical rules within the trained `RandomForestClassifier` respond to changes in input variables. They represent **statistical associations conditional on the tree ensemble's learned split structure**, NOT empirical real-world causation.
>
> Specifically:
> - A positive SHAP value ($\phi_j > 0$) indicates that feature $j$ pushed the model's distress probability upward relative to the expectation. It does **not** prove that changing that metric in the real company will prevent insolvency.
> - A feature with zero attribution ($\phi_j = 0$) might still be economically vital in corporate finance, but redundant to the decision trees due to correlated collinear features.
> - Credit analysts must never present SHAP values to SME borrowers or regulators as "proof of the underlying cause of distress."

### 2.2 Anonymized Feature Masking in Dataset A
- In Dataset A (Tehran Stock Exchange SME panel), original financial indicators are anonymized as `x1` through `x83`.
- No official mapping to standard accounting taxonomy (e.g. Current Ratio, EBITDA Margin, Debt-to-Equity) has been verified.
- **Strict Policy:**
  - The system displays verified feature identifiers (e.g. `x12`, `x2_roll_mean_2`) and mathematical transformation definitions.
  - Fabricating balance sheet narratives (e.g. claiming that `x12` represents "short-term liquidity insolvency") is strictly prohibited.
  - Feature categories in the dashboard are labeled as `"Statistical/Unmapped Feature"` or derived temporal transformations.

### 2.3 Non-Linearity of Platt Probability Calibration
- Native TreeSHAP is additive with respect to the Random Forest positive-class output space ($p_{\text{RF}} \in [0, 1]$):
  $$\sum_{j=1}^{163} \phi_j(\mathbf{x}) + \mathbb{E}[p_{\text{RF}}] = p_{\text{RF}}(\mathbf{x})$$
- However, the operational probability used for alerting is produced by downstream Platt sigmoid calibration:
  $$p_{\text{cal}} = \sigma(a \cdot p_{\text{RF}} + b)$$
- Because $\sigma(\cdot)$ is non-linear, the sum of raw SHAP values **does not equal** $p_{\text{cal}} - \mathbb{E}[p_{\text{cal}}]$.
- **Architectural Guardrail:**
  - The explainer and API explicitly document that SHAP values are expressed in $p_{\text{RF}}$ space.
  - Feature contributions are not linearly summed with the calibrated probability $p_{\text{cal}}$ or the 0–100 composite risk score.

### 2.4 Retrospective Validation Does Not Prove Live Production Readiness
- All validation experiments (ROC-AUC 0.7783, Average Precision 0.1946) were conducted retrospectively on a single historical annual partition ($T = 13$, 2012–2014 period).
- Small positive sample size ($N_{\text{pos}} = 11$ distress events out of 262 firms) produces wide confidence intervals (Recall 95% CI $[9.8\%, 56.6\%]$).
- A retrospective early-warning prototype cannot guarantee identical out-of-period recall in modern economic regimes without:
  1. Prospective out-of-time shadow deployment.
  2. Automated data drift and concept drift monitoring.
  3. Credit committee human oversight and manual override mechanisms.

---

## 3. Summary of Decision-Support Boundaries

| Capability | Supported by Phase 3 | NOT Supported / Prohibited |
| :--- | :---: | :---: |
| **Model Verification** | Mathematical additivity of tree ensemble | Claiming mathematical additivity on calibrated probability |
| **Feature Association** | Identifying which features drive RF predictions | Inferring causal business failure mechanisms |
| **Temporal Tracking** | Decomposing period-over-period attribution shifts | Interpreting feature deltas as proven corporate recovery |
| **Feature Labels** | Displaying masked IDs and temporal constructions | Fabricating balance sheet accounting definitions |
| **Decision Scope** | Prioritizing high-risk firms for human audit | Autonomous loan rejection or automated adverse action |
