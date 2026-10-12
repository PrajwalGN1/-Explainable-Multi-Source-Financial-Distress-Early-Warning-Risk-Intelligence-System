"""
Phase 4: Operating Threshold Optimization on Validation Data (T=11, 12).

Evaluates candidate decision thresholds exclusively on the validation cohort (N=517, N_pos=14):
- Keeps approved model frozen (ZERO model retraining)
- Sweeps thresholds from 0.010 to 0.200
- Evaluates recall, precision, specificity, F1, F2, alert count, and alert rate
- Computes Wilson score 95% confidence intervals
- Identifies optimal operating points using validation-only criteria
- Strictly preserves T=13 test set untouched
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("phase4_threshold_sweep")


def wilson_ci(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total) / denom
    return (max(0.0, round(float(center - margin), 4)), min(1.0, round(float(center + margin), 4)))


def run_threshold_sweep() -> Dict[str, Any]:
    os.makedirs("results/phase4", exist_ok=True)
    os.makedirs("experiments/phase4", exist_ok=True)

    # 1. Load Data
    logger.info("Loading Dataset A via DataLoader...")
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()

    # 2. Target Formulation & Feature Engineering
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(
        df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
    )
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    df_valid = df_feats.dropna(subset=["forward_target"]).copy()
    df_valid["forward_target"] = df_valid["forward_target"].astype(int)

    # Strict split: Validation only (T = 11, 12)
    val_df = df_valid[(df_valid["Time"] > 10) & (df_valid["Time"] <= 12)].copy()
    y_val = val_df["forward_target"].to_numpy()
    n_val = len(val_df)
    n_pos_val = int(sum(y_val))
    val_prevalence = float(np.mean(y_val))

    logger.info(f"Validation cohort: {n_val} rows, {n_pos_val} positive events ({val_prevalence*100:.2f}%)")

    # 3. Load Frozen Phase 2 Approved Model Artifacts (ZERO Retraining)
    preprocessor = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    model = joblib.load("models/phase2/best_ews_model.joblib")
    calibrator = joblib.load("models/phase2/ews_calibrator.joblib")

    X_val, _ = preprocessor.transform(val_df)
    raw_val_probs = model.predict_proba(X_val)[:, 1]
    cal_val_probs = calibrator.predict_proba(X_val)[:, 1]

    # 4. Comprehensive Threshold Sweep Grid
    threshold_grid = [
        0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050,
        0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095,
        0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.170, 0.200
    ]

    sweep_records = []
    for tau in threshold_grid:
        y_pred = (cal_val_probs >= tau).astype(int)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_val, y_pred, zero_division=0))
        rec = float(recall_score(y_val, y_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        alerts = int(tp + fp)
        alert_rate = float(alerts / n_val) * 100.0

        rec_ci = wilson_ci(tp, n_pos_val)
        prec_ci = wilson_ci(tp, alerts)

        sweep_records.append({
            "threshold": tau,
            "total_val": n_val,
            "distress_events": n_pos_val,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "recall": round(rec, 4),
            "recall_ci_low": rec_ci[0],
            "recall_ci_high": rec_ci[1],
            "precision": round(prec, 4),
            "precision_ci_low": prec_ci[0],
            "precision_ci_high": prec_ci[1],
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
            "f2": round(f2, 4),
            "alerts_count": alerts,
            "alert_rate_pct": round(alert_rate, 2),
            "false_positive_to_true_positive_ratio": round(fp / tp, 2) if tp > 0 else float("inf")
        })

    df_sweep = pd.DataFrame(sweep_records)
    df_sweep.to_csv("results/phase4/validation_threshold_sweep.csv", index=False)
    logger.info("Saved validation threshold sweep table to results/phase4/validation_threshold_sweep.csv")

    # 5. Documented Validation-Only Policy Operating Points
    # Baseline Governed Operating Point (Current reference: tau = 0.080)
    baseline_point = df_sweep[df_sweep["threshold"] == 0.080].iloc[0].to_dict()

    # Optimal F2 Policy: Maximizes F2 on validation data
    best_f2_row = df_sweep.loc[df_sweep["f2"].idxmax()].to_dict()

    # High-Recall Balanced Policy: First threshold achieving >= 70% recall on validation
    high_rec_candidates = df_sweep[df_sweep["recall"] >= 0.70]
    high_rec_point = high_rec_candidates.iloc[-1].to_dict() if len(high_rec_candidates) > 0 else best_f2_row

    # Constrained Alert Budget Policy: Maximizes recall subject to Alert Rate <= 10.0%
    budget_candidates = df_sweep[df_sweep["alert_rate_pct"] <= 10.0]
    budget_point = budget_candidates.loc[budget_candidates["recall"].idxmax()].to_dict() if len(budget_candidates) > 0 else baseline_point

    candidate_operating_points = {
        "validation_dataset": {
            "periods": "T=11, 12",
            "sample_size": n_val,
            "positive_events": n_pos_val,
            "prevalence_pct": round(val_prevalence * 100.0, 2)
        },
        "operating_points": {
            "baseline_governed": {
                "threshold": baseline_point["threshold"],
                "rationale": "Current approved operational policy (tau* = 0.080)",
                "val_recall": baseline_point["recall"],
                "val_precision": baseline_point["precision"],
                "val_f2": baseline_point["f2"],
                "val_alert_rate_pct": baseline_point["alert_rate_pct"],
                "val_alerts_count": baseline_point["alerts_count"],
                "val_tp": baseline_point["tp"],
                "val_fp": baseline_point["fp"],
                "val_fn": baseline_point["fn"]
            },
            "validation_optimal_f2": {
                "threshold": best_f2_row["threshold"],
                "rationale": "Empirical maximum F2-score on validation cohort",
                "val_recall": best_f2_row["recall"],
                "val_precision": best_f2_row["precision"],
                "val_f2": best_f2_row["f2"],
                "val_alert_rate_pct": best_f2_row["alert_rate_pct"],
                "val_alerts_count": best_f2_row["alerts_count"],
                "val_tp": best_f2_row["tp"],
                "val_fp": best_f2_row["fp"],
                "val_fn": best_f2_row["fn"]
            },
            "high_recall_policy_70pct": {
                "threshold": high_rec_point["threshold"],
                "rationale": "Highest threshold achieving >= 70% recall on validation cohort",
                "val_recall": high_rec_point["recall"],
                "val_precision": high_rec_point["precision"],
                "val_f2": high_rec_point["f2"],
                "val_alert_rate_pct": high_rec_point["alert_rate_pct"],
                "val_alerts_count": high_rec_point["alerts_count"],
                "val_tp": high_rec_point["tp"],
                "val_fp": high_rec_point["fp"],
                "val_fn": high_rec_point["fn"]
            },
            "constrained_alert_budget_10pct": {
                "threshold": budget_point["threshold"],
                "rationale": "Highest recall achievable with alert rate strictly <= 10.0%",
                "val_recall": budget_point["recall"],
                "val_precision": budget_point["precision"],
                "val_f2": budget_point["f2"],
                "val_alert_rate_pct": budget_point["alert_rate_pct"],
                "val_alerts_count": budget_point["alerts_count"],
                "val_tp": budget_point["tp"],
                "val_fp": budget_point["fp"],
                "val_fn": budget_point["fn"]
            }
        }
    }

    with open("results/phase4/validation_candidate_operating_points.json", "w", encoding="utf-8") as f:
        json.dump(candidate_operating_points, f, indent=2)

    logger.info("Saved candidate operating points to results/phase4/validation_candidate_operating_points.json")
    return {
        "sweep_df": df_sweep,
        "candidate_operating_points": candidate_operating_points
    }


if __name__ == "__main__":
    run_threshold_sweep()
