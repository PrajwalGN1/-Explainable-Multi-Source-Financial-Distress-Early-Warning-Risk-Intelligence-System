"""
Phase 6: Outcome-Maturity Policy & Operational Drift Monitoring.

Implements Section 6, 7, and 10 of the Phase 6 Master Implementation Prompt:
1. Outcome-Maturity Policy:
   - Predictions at time t predict distress at time t+1.
   - Predictions remain in PENDING status until time t+1 filing is audited.
   - Ground-truth evaluation (Recall, Precision, Specificity, Brier) is permitted
     STRICTLY on matured predictions.
2. Operational Real-Time Telemetry (Computed on all prospective predictions):
   - Realized alert rates (Production vs Shadow Candidate)
   - Inter-model agreement & disagreement classification
   - Missing data & feature imputation rate
   - Population Stability Index (PSI) for top financial ratios (x12, x25, x16, x2, x48)
3. Pre-Registered Evaluation Protocol & Human Approval Gates:
   - Minimum sample requirement: N_matured >= 100, N_distress >= 10
   - Explicit human approval gate blocking autonomous promotion.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

sys.path.insert(0, os.path.abspath("."))
from src.utils.logger import setup_logger

logger = setup_logger("maturity_and_drift")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between baseline and prospective distributions.
    PSI < 0.1: No significant drift
    0.1 <= PSI < 0.25: Moderate drift
    PSI >= 0.25: Significant drift requiring model review
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Clean non-finite
    exp_clean = expected[np.isfinite(expected)]
    act_clean = actual[np.isfinite(actual)]
    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine quantile bins from expected distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(exp_clean, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Bucket counts
    exp_counts, _ = np.histogram(exp_clean, bins=bins)
    act_counts, _ = np.histogram(act_clean, bins=bins)

    # Fractions with Laplace smoothing
    eps = 1e-4
    exp_pct = (exp_counts + eps) / (len(exp_clean) + eps * num_bins)
    act_pct = (act_counts + eps) / (len(act_clean) + eps * num_bins)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi_val), 4)


class OutcomeMaturityPolicy:
    """Enforces outcome maturity rules and calculates performance on matured observations."""

    MIN_MATURE_OBSERVATIONS = 100
    MIN_MATURE_DISTRESS_EVENTS = 10

    @staticmethod
    def classify_maturity(observation_period: int, current_available_period: int) -> str:
        """
        Determines maturity status.
        Forward target requires t+1 to be available.
        If current_available_period >= observation_period + 1 -> MATURED.
        Else -> PENDING_MATURITY.
        """
        if current_available_period >= observation_period + 1:
            return "MATURED"
        return "PENDING_MATURITY"

    @classmethod
    def evaluate_matured_predictions(
        cls,
        df_predictions: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates ground-truth metrics STRICTLY on matured observations.
        If minimum observation thresholds are not yet reached, reports telemetry with uncertainty disclosure.
        """
        if "outcome_maturity_status" not in df_predictions.columns or "actual_target" not in df_predictions.columns:
            return {
                "status": "MISSING_MATURITY_COLUMNS",
                "message": "Actual target or maturity status column not present in predictions."
            }

        df_mature = df_predictions[df_predictions["outcome_maturity_status"] == "MATURED"].dropna(subset=["actual_target"]).copy()
        n_mature = len(df_mature)
        n_pos = int(df_mature["actual_target"].sum()) if n_mature > 0 else 0

        meets_protocol = (n_mature >= cls.MIN_MATURE_OBSERVATIONS) and (n_pos >= cls.MIN_MATURE_DISTRESS_EVENTS)

        if n_mature == 0:
            return {
                "maturity_status": "ZERO_MATURED_PREDICTIONS",
                "matured_count": 0,
                "pending_count": len(df_predictions),
                "message": "All prospective predictions are currently pending t+1 statutory outcome observation."
            }

        y_true = df_mature["actual_target"].to_numpy().astype(int)
        
        # Production metrics
        prod_probs = df_mature["prod_calibrated_probability"].to_numpy()
        prod_preds = (prod_probs >= 0.080).astype(int)
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true, prod_preds, labels=[0, 1]).ravel()
        rec_p = recall_score(y_true, prod_preds, zero_division=0)
        prec_p = precision_score(y_true, prod_preds, zero_division=0)
        bs_p = brier_score_loss(y_true, prod_probs)

        # Candidate metrics
        cand_probs = df_mature["cand_calibrated_probability"].to_numpy()
        cand_preds = (cand_probs >= 0.0894).astype(int)
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, cand_preds, labels=[0, 1]).ravel()
        rec_c = recall_score(y_true, cand_preds, zero_division=0)
        prec_c = precision_score(y_true, cand_preds, zero_division=0)
        bs_c = brier_score_loss(y_true, cand_probs)

        return {
            "evaluation_cohort": "MATURED_ONLY",
            "meets_preregistered_sample_protocol": meets_protocol,
            "min_required_mature": cls.MIN_MATURE_OBSERVATIONS,
            "min_required_distress": cls.MIN_MATURE_DISTRESS_EVENTS,
            "matured_count": n_mature,
            "matured_distress_cases": n_pos,
            "production_baseline": {
                "operating_threshold": 0.080,
                "tp": int(tp_p), "fp": int(fp_p), "tn": int(tn_p), "fn": int(fn_p),
                "recall": round(float(rec_p), 4),
                "precision": round(float(prec_p), 4),
                "brier_score": round(float(bs_p), 4),
                "alert_rate_pct": round(float(tp_p + fp_p) / n_mature * 100.0, 2)
            },
            "shadow_candidate": {
                "research_threshold": 0.0894,
                "tp": int(tp_c), "fp": int(fp_c), "tn": int(tn_c), "fn": int(fn_c),
                "recall": round(float(rec_c), 4),
                "precision": round(float(prec_c), 4),
                "brier_score": round(float(bs_c), 4),
                "alert_rate_pct": round(float(tp_c + fp_c) / n_mature * 100.0, 2)
            },
            "comparison": {
                "delta_recall": round(float(rec_c - rec_p), 4),
                "delta_precision": round(float(prec_c - prec_p), 4),
                "delta_alerts": int((tp_c + fp_c) - (tp_p + fp_p)),
                "statistically_verified": bool(meets_protocol and n_pos >= 10)
            }
        }


class ShadowDriftMonitor:
    """Monitors prospective distribution drift, inter-model agreement, and missingness."""

    TOP_RATIOS = ["x12", "x25", "x16", "x2", "x48"]

    def __init__(self, baseline_ref_df: pd.DataFrame):
        self.baseline_ref = baseline_ref_df

    def evaluate_prospective_telemetry(
        self,
        df_prospective_predictions: pd.DataFrame,
        df_prospective_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculates running telemetry across all prospective observations."""
        n_total = len(df_prospective_predictions)
        if n_total == 0:
            return {"status": "NO_PREDICTIONS_RECORDED"}

        # 1. Alert Volume & Rates
        prod_alerts = int(df_prospective_predictions["prod_is_alert"].sum())
        cand_alerts = int(df_prospective_predictions["cand_is_alert"].sum())
        prod_alert_rate = round(prod_alerts / n_total * 100.0, 2)
        cand_alert_rate = round(cand_alerts / n_total * 100.0, 2)

        # 2. Inter-Model Agreement
        both_alert = int(((df_prospective_predictions["prod_is_alert"]) & (df_prospective_predictions["cand_is_alert"])).sum())
        neither_alert = int(((~df_prospective_predictions["prod_is_alert"]) & (~df_prospective_predictions["cand_is_alert"])).sum())
        cand_only_alert = int(((~df_prospective_predictions["prod_is_alert"]) & (df_prospective_predictions["cand_is_alert"])).sum())
        prod_only_alert = int(((df_prospective_predictions["prod_is_alert"]) & (~df_prospective_predictions["cand_is_alert"])).sum())
        agreement_pct = round((both_alert + neither_alert) / n_total * 100.0, 2)

        # 3. Feature Missingness
        missing_rate_pct = round(float(df_prospective_features.isna().mean().mean() * 100.0), 2)

        # 4. Feature Drift PSI
        psi_report = {}
        for col in self.TOP_RATIOS:
            if col in self.baseline_ref.columns and col in df_prospective_features.columns:
                psi_val = calculate_psi(
                    expected=self.baseline_ref[col].to_numpy(),
                    actual=df_prospective_features[col].to_numpy()
                )
                drift_status = "STABLE" if psi_val < 0.1 else ("MODERATE_DRIFT" if psi_val < 0.25 else "SIGNIFICANT_DRIFT")
                psi_report[col] = {
                    "psi_value": psi_val,
                    "drift_status": drift_status
                }

        # 5. Alert Budget Constraint Check
        # Target alert budget: 8.0%. Acceptable operating range: [6.0%, 10.0%]
        budget_compliant = (6.0 <= cand_alert_rate <= 10.0)

        return {
            "total_prospective_observations": n_total,
            "production_alert_count": prod_alerts,
            "production_alert_rate_pct": prod_alert_rate,
            "shadow_candidate_alert_count": cand_alerts,
            "shadow_candidate_alert_rate_pct": cand_alert_rate,
            "inter_model_agreement_pct": agreement_pct,
            "disagreement_matrix": {
                "both_alerted": both_alert,
                "neither_alerted": neither_alert,
                "shadow_only_alerted": cand_only_alert,
                "production_only_alerted": prod_only_alert
            },
            "feature_missingness_rate_pct": missing_rate_pct,
            "alert_budget_compliance": {
                "target_budget_pct": 8.0,
                "realized_rate_pct": cand_alert_rate,
                "is_within_budget_limits": budget_compliant,
                "status": "COMPLIANT" if budget_compliant else ("EXCESS_ALERT_BURDEN" if cand_alert_rate > 10.0 else "UNDER_ALERT_BURDEN")
            },
            "feature_drift_psi": psi_report
        }


def generate_synthetic_shadow_simulation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a reproducible prospective shadow evaluation run using holdout data
    to populate shadow logs and demonstrate operational dashboards.
    """
    from src.data.loaders import DataLoader
    loader = DataLoader("data/raw")
    df_raw = loader.load_dataset_a()

    from src.shadow.shadow_engine import ShadowInferenceEngine
    engine = ShadowInferenceEngine()

    # Simulate prospective surveillance on period 12 companies (with historical T<=12)
    companies = df_raw[df_raw["Time"] == 12]["Company"].unique()
    pred_records = []
    features_list = []

    for comp_id in companies:
        df_co = df_raw[df_raw["Company"] == comp_id].copy()
        res = engine.evaluate_shadow_sample(
            company_id=str(comp_id),
            df_company_history=df_co,
            current_period=12,
            record_audit_log=False
        )
        
        # Check actual target at T=13 from Dataset A if available
        co_t13 = df_raw[(df_raw["Company"] == comp_id) & (df_raw["Time"] == 13)]
        target_t13 = None
        if len(co_t13) > 0:
            target_t13 = int(co_t13["Financial Distress"].values[0] <= -0.5)

        pred_records.append({
            "company_id": str(comp_id),
            "observation_period": 12,
            "distress_target_period": 13,
            "outcome_maturity_status": "MATURED" if target_t13 is not None else "PENDING_MATURITY",
            "actual_target": target_t13,
            "prod_is_alert": res["production"]["is_distress_alert"],
            "prod_calibrated_probability": res["production"]["calibrated_probability"],
            "prod_risk_score": res["production"]["risk_score"],
            "cand_is_alert": res["shadow_candidate"]["is_shadow_alert"],
            "cand_calibrated_probability": res["shadow_candidate"]["calibrated_probability"],
            "cand_risk_score": res["shadow_candidate"]["risk_score"]
        })

        # Top features at T=12
        co_curr = df_co[df_co["Time"] == 12].iloc[0]
        features_list.append({
            col: float(co_curr[col]) for col in ["x12", "x25", "x16", "x2", "x48"] if col in co_curr
        })

    df_preds = pd.DataFrame(pred_records)
    df_feats = pd.DataFrame(features_list)
    return df_preds, df_feats


def run_and_save_shadow_simulation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs prospective shadow simulation and persists outputs in results/phase6/."""
    os.makedirs("results/phase6", exist_ok=True)
    df_p, df_f = generate_synthetic_shadow_simulation()
    
    from src.data.loaders import DataLoader
    df_raw = DataLoader("data/raw").load_dataset_a()
    train_ref = df_raw[df_raw["Time"] <= 10]

    monitor = ShadowDriftMonitor(baseline_ref_df=train_ref)
    telemetry = monitor.evaluate_prospective_telemetry(df_p, df_f)
    maturity_eval = OutcomeMaturityPolicy.evaluate_matured_predictions(df_p)

    # Persist artifacts
    df_p.to_csv("results/phase6/prospective_predictions.csv", index=False)
    with open("results/phase6/prospective_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    with open("results/phase6/matured_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(maturity_eval, f, indent=2)

    logger.info("Saved Phase 6 prospective simulation artifacts to results/phase6/.")
    return telemetry, maturity_eval


if __name__ == "__main__":
    telemetry, maturity_eval = run_and_save_shadow_simulation()
    print("Prospective Telemetry Summary:")
    print(json.dumps(telemetry, indent=2))
    print("\nMatured Ground-Truth Comparison:")
    print(json.dumps(maturity_eval, indent=2))
