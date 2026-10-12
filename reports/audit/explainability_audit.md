# Explainable AI (XAI) & Interpretability Audit Report

**Project:** Explainable SME Financial Distress Early-Warning & Risk Intelligence System  
**Audit Scope:** PHASE 7 — Audit Explainable AI Implementation  
**Evidence Source:** `src/explainability/shap_explainer.py`, `src/features/feature_registry.py`, `src/pipeline.py`, `app.py`, `reports/explainability/global_importance.csv`, and precomputed intelligence database.

---

## 1. Executive Summary: Explainability Audit

A core requirement of an institutional financial distress early-warning system is **trustworthy, scientifically defensible explainability**. Credit analysts, relationship managers, and risk auditors must understand *why* an enterprise is flagged, *which specific financial indicators* drove the risk increase, and *whether deterioration is temporary or structural*.

This audit evaluated the technical implementation of Explainable AI in the project and discovered two critical defects:
1. **The SHAP Implementation is Compromised (Heuristic Fallback):** The documentation, README, and dashboard claim to use **TreeSHAP**. In reality, because the selected primary model is a `RandomForestClassifier` and the external `shap` package is not installed, the explainer falls back to an ad-hoc formula:
   $$\text{Attribution}_i = x_i \times \frac{\text{Gini Importance}_i}{\sum \text{Gini Importance}}$$
   **This is mathematically not Shapley values, not TreeSHAP, and violates additive efficiency and scale-invariance.** Features with large unstandardized numerical scales dominate the reported explanations.
2. **Semantic Fabrication of Masked Indicators:** For Dataset A, where attributes `x1` through `x83` are unmasked, anonymized variables without published accounting definitions, the `FeatureRegistry` assigns canonical financial meanings (`x48` as "Profitability", `x25` as "Liquidity", `x64` as "Leverage", `x81` as "Solvency") based merely on target correlation. The dashboard then displays automated narratives citing these fabricated accounting labels.

---

## 2. Technical Audit of `ModelExplainer` (`src/explainability/shap_explainer.py`)

### 2.1 The Native Booster vs. Heuristic Branch
In `src/explainability/shap_explainer.py` (lines 34–43):

```python
if hasattr(self.model, "get_booster"):
  booster = self.model.get_booster()
  dmat = xgb.DMatrix(
      feature_row.reshape(1, -1), feature_names=self.feature_names
  )
  contribs = booster.predict(dmat, pred_contribs=True)[0]
  feature_contribs = contribs[:-1]
  expected_val = float(contribs[-1])
else:
  feature_contribs = feature_row * self._norm_importances
  expected_val = float(getattr(self.model, "intercept_", [0.0])[0])
```

#### Forensic Findings:
1. **XGBoost Has Native TreeSHAP:** When the model is an XGBoost classifier, `get_booster()` exists, and `booster.predict(..., pred_contribs=True)` calculates exact Lundberg et al. (2020) TreeSHAP values directly inside the XGBoost C++ engine without requiring the `shap` Python library.
2. **Random Forest Triggers the Fallback:** In `models/ews/ews_metrics.json`, the model selected as the best EWS model is:
   `"best_model_name": "random_forest"`.
3. A scikit-learn `RandomForestClassifier` does not possess `get_booster()`. Therefore, for the primary model chosen by the pipeline, **the code unconditionally executes the `else` branch**:
   `feature_contribs = feature_row * self._norm_importances`.

---

### 2.2 Mathematical Flaws of the Fallback Formulation

The fallback calculation `feature_contribs = feature_row * self._norm_importances` exhibits severe mathematical problems:

1. **Violation of Additive Efficiency (Efficiency Axiom):**  
   True Shapley values satisfy:
   $$\sum_{i=1}^M \phi_i(x) = f(x) - \mathbb{E}[f(X)]$$
   The sum of feature attributions must equal the model prediction minus the base expected value. The fallback formulation does not satisfy this; its sum is arbitrary and has no probabilistic or margin interpretation.
2. **Scale Distortion:**  
   If feature $x_A$ is measured in thousands (e.g., $x_A = 5,000$) with importance $0.02$, its reported impact is $5,000 \times 0.02 = 100$. If feature $x_B$ is a normalized ratio (e.g., $x_B = 0.35$) with importance $0.20$, its reported impact is $0.35 \times 0.20 = 0.07$. The unstandardized feature appears 1,400 times more influential than the ratio, purely due to measurement scale.
3. **Misidentification of Protective Factors:**  
   In the fallback, any feature with a negative value ($x_i < 0$) is automatically classified as a "protective factor" because its product with non-negative Gini importance is negative. Conversely, any positive feature ($x_i > 0$) is classified as a "risk driver". In reality, an indicator with a negative value could be a severe risk driver (e.g., negative net worth or negative working capital).

---

## 3. Audit of Longitudinal & Period-over-Period Delta Explanations

### 3.1 Method: `explain_risk_change`
In `src/explainability/shap_explainer.py` (lines 83–120), the method decomposes why a firm's estimated risk changed between period $t-1$ and period $t$:

```python
d_contribs = current_contribs - prev_contribs
d_values = current_row - prev_row
pos_shift_idx = np.where(d_contribs > 0.001)[0]
```

#### Evaluation:
- **Conceptually Strong:** Decomposing the shift in feature contributions between sequential reporting cycles is an excellent operational feature for credit risk monitoring. It answers the analyst's question: *"What changed this quarter that caused the rating to downgrade?"*
- **Execution Limitation:** Because `current_contribs` and `prev_contribs` are computed using the flawed fallback heuristic, the delta attributions reflect shifts in unscaled feature values rather than true Shapley marginal changes.

---

## 4. Audit of Feature Meaning & Interpretability Integrity

### 4.1 Masked Features in Dataset A (`x1` through `x83`)
In `src/features/feature_registry.py`:
- `x81` is registered as: "Financial/Economic Health — high positive correlation with distress index".
- `x48` is registered as: "Profitability/Efficiency — positive correlation with solvency score".
- `x25` is registered as: "Liquidity/Working Capital — cash/operating stability indicator".
- `x64` is registered as: "Financial Vulnerability/Cost — elevates distress risk".

#### The Problem:
- In the original dataset documentation (Tehran Stock Exchange panel by S. Ebrahimi), the specific accounting formulas for `x1` through `x83` are **masked**. Only `x80` is documented (as a 37-class industry code).
- Assigning accounting identities ("Profitability", "Liquidity", "Working Capital") based solely on statistical correlation confuses correlation with semantic identity.
- In View 6 of the dashboard ("Why Did Risk Change?"), the UI generates automated explanations such as:
  *"Risk increased by +3.2 pp, driven primarily by severe deterioration in Liquidity/Working Capital (x25)."*
- **Audit Conclusion:** The dashboard is presenting speculative conjectures as verified financial statement line items.

---

## 5. Audit Against Key Surveillance Questions

| Core Surveillance Question | Current System Capability | Evaluation & Trustworthiness |
| :--- | :---: | :--- |
| **Why is this company flagged?** | **Partially capable** | Displays top 5 "risk drivers". However, drivers are determined by unscaled heuristic products, not true TreeSHAP values. |
| **Which indicators contribute most to the current prediction?** | **Unreliable** | Unstandardized features with large numerical magnitudes dominate the ranking. |
| **Why has the estimated risk changed since the previous observation?** | **Partially capable** | Delta engine tracks feature shifts, but attributions are distorted by the underlying heuristic. |
| **Which changes are new vs. already present?** | **Capable** | Longitudinal trajectory correctly tracks consecutive deterioration streaks and multi-period acceleration. |
| **Is deterioration persistent or temporary?** | **Capable** | Trajectory engine classifies patterns into "Persistent Deterioration", "Accelerating Deterioration", and "Stable Fluctuations". |
| **Are explanations presented as proof of causation?** | **Compliant** | Code and UI maintain descriptive correlation terminology and include appropriate academic research disclaimers. |

---

## 6. Recommended Upgrade Plan for Explainability

### Upgrade Path A: Use XGBoost with Verified Native TreeSHAP (Recommended)
1. In `src/models/ews_model.py`, select **XGBoost** as the production model architecture. XGBoost's validation performance (`ROC-AUC = 0.8824`, `PR-AUC = 0.2050`) is competitive with Random Forest (`ROC-AUC = 0.8828`).
2. XGBoost executes **native C++ TreeSHAP** (`booster.predict(dmat, pred_contribs=True)`), computing mathematically exact Shapley attributions in sub-milliseconds with zero external dependencies.

### Upgrade Path B: Install `shap` Library for Random Forest
1. Add `shap>=0.44.0` to `requirements.txt`.
2. Wrap `shap.TreeExplainer(rf_model)` inside `ModelExplainer` to compute exact TreeSHAP values for the Random Forest model.

### Transparent Labeling of Masked Features
1. Update `FeatureRegistry` and dashboard views to replace fabricated accounting terms with data-driven statistical factor labels:
   - Instead of "Liquidity/Working Capital (x25)", display:
     `Key Predictive Indicator x25 (Rank 1 Statistical Factor, Negative Direction)`.
   - Include a transparent tooltip: *"Attribute masked in dataset source; exhibits strong empirical association with solvency retention."*
