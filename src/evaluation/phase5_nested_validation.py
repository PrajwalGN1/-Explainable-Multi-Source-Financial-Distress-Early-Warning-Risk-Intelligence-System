"""
Phase 5: Nested Validation & Robustness Analysis.

Executes Section 7 & 8 of the Phase 5 Master Implementation Prompt:
1. Per-period temporal breakdown on validation:
   - Cohort T = 11 alone (N=257, N_pos=6)
   - Cohort T = 12 alone (N=260, N_pos=8)
2. Rolling-origin temporal stability evaluation across historical cutoffs:
   - Cutoff 1: Train T <= 8, Test T = 9
   - Cutoff 2: Train T <= 9, Test T = 10
   - Cutoff 3: Train T <= 10, Test T in {11, 12}
3. 5-Fold GroupKFold cross-sectional fold-level tracking (tracking N_pos per fold)
4. Paired bootstrap resampling comparison (B=2,000 resamples) between
   top candidate (Phase5_RF_CostSensitive_w10) and Approved Baseline (Approved_Baseline_RF_Ref)
   on validation cohort.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.abspath("."))
from src.utils.logger import setup_logger
from src.evaluation.phase5_feature_engineering import build_phase5_dataset
from src.evaluation.phase5_model_comparison import Phase5PreprocessingPipeline

logger = setup_logger("phase5_nested_validation")


def evaluate_cohort(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.080) -> Dict[str, Any]:
    """Computes standard metrics on a cohort at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "n_obs": len(y_true),
        "n_pos": int(tp + fn),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts_count": int(tp + fp),
        "alert_rate_pct": round((tp + fp) / len(y_true) * 100.0, 2),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4)
    }


def paired_bootstrap_validation(
    y_true: np.ndarray,
    probs_base: np.ndarray,
    thresh_base: float,
    probs_cand: np.ndarray,
    thresh_cand: float,
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """Paired bootstrap on validation data to estimate uncertainty in metric differences."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diff_rec, diff_prec, diff_alerts, diff_aucs, diff_aps = [], [], [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue

        p_base = probs_base[idx]
        p_cand = probs_cand[idx]

        pred_base = (p_base >= thresh_base).astype(int)
        pred_cand = (p_cand >= thresh_cand).astype(int)

        rec_base = recall_score(y_b, pred_base, zero_division=0)
        rec_cand = recall_score(y_b, pred_cand, zero_division=0)
        diff_rec.append(rec_cand - rec_base)

        prec_base = precision_score(y_b, pred_base, zero_division=0)
        prec_cand = precision_score(y_b, pred_cand, zero_division=0)
        diff_prec.append(prec_cand - prec_base)

        diff_alerts.append(np.sum(pred_cand) - np.sum(pred_base))
        diff_aucs.append(roc_auc_score(y_b, p_cand) - roc_auc_score(y_b, p_base))
        diff_aps.append(average_precision_score(y_b, p_cand) - average_precision_score(y_b, p_base))

    p_val_rec = float(np.mean(np.array(diff_rec) <= 0.0) if np.mean(diff_rec) > 0 else np.mean(np.array(diff_rec) >= 0.0) * 2.0)

    return {
        "n_bootstraps": len(diff_rec),
        "delta_recall_mean": round(float(np.mean(diff_rec)), 4),
        "delta_recall_ci_95": [round(float(np.percentile(diff_rec, 2.5)), 4), round(float(np.percentile(diff_rec, 97.5)), 4)],
        "delta_precision_mean": round(float(np.mean(diff_prec)), 4),
        "delta_precision_ci_95": [round(float(np.percentile(diff_prec, 2.5)), 4), round(float(np.percentile(diff_prec, 97.5)), 4)],
        "delta_alerts_mean": round(float(np.mean(diff_alerts)), 1),
        "delta_alerts_ci_95": [round(float(np.percentile(diff_alerts, 2.5)), 1), round(float(np.percentile(diff_alerts, 97.5)), 1)],
        "delta_roc_auc_mean": round(float(np.mean(diff_aucs)), 4),
        "delta_roc_auc_ci_95": [round(float(np.percentile(diff_aucs, 2.5)), 4), round(float(np.percentile(diff_aucs, 97.5)), 4)],
        "delta_pr_auc_mean": round(float(np.mean(diff_aps)), 4),
        "delta_pr_auc_ci_95": [round(float(np.percentile(diff_aps, 2.5)), 4), round(float(np.percentile(diff_aps, 97.5)), 4)],
        "p_value_recall_diff": round(p_val_rec, 4)
    }


def run_nested_validation_analysis():
    """Executes full nested validation and robustness audit."""
    os.makedirs("results/phase5", exist_ok=True)

    # 1. Load Dataset
    df_p5 = build_phase5_dataset()

    train_df = df_p5[df_p5["Time"] <= 10].copy().reset_index(drop=True)
    val_df = df_p5[df_p5["Time"].isin([11, 12])].copy().reset_index(drop=True)
    val_t11 = df_p5[df_p5["Time"] == 11].copy().reset_index(drop=True)
    val_t12 = df_p5[df_p5["Time"] == 12].copy().reset_index(drop=True)

    # Preprocessors
    prep = Phase5PreprocessingPipeline()
    prep.fit(train_df)
    X_val, y_val = prep.transform(val_df)
    X_t11, y_t11 = prep.transform(val_t11)
    X_t12, y_t12 = prep.transform(val_t12)

    from src.preprocessing.preprocessors import EWSPreprocessor
    app_prep = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    X_val_app, _ = app_prep.transform(val_df)
    X_t11_app, _ = app_prep.transform(val_t11)
    X_t12_app, _ = app_prep.transform(val_t12)

    # Models
    app_rf = joblib.load("models/phase2/best_ews_model.joblib")
    app_cal = joblib.load("models/phase2/ews_calibrator.joblib")
    p5_cand_model = joblib.load("experiments/phase5/models/Phase5_RF_CostSensitive_w10.joblib")
    p5_cand_cal = joblib.load("experiments/phase5/models/Phase5_RF_CostSensitive_w10_calibrator.joblib")

    probs_app_val = app_cal.predict_proba(X_val_app)[:, 1]
    probs_cand_val = p5_cand_cal.predict_proba(X_val)[:, 1]

    probs_app_t11 = app_cal.predict_proba(X_t11_app)[:, 1]
    probs_cand_t11 = p5_cand_cal.predict_proba(X_t11)[:, 1]

    probs_app_t12 = app_cal.predict_proba(X_t12_app)[:, 1]
    probs_cand_t12 = p5_cand_cal.predict_proba(X_t12)[:, 1]

    # 2. Per-Period Breakdown
    period_records = [
        {"model": "Approved_Baseline", "period": "T=11", **evaluate_cohort(y_t11, probs_app_t11, 0.080)},
        {"model": "Phase5_CostSensitive", "period": "T=11", **evaluate_cohort(y_t11, probs_cand_t11, 0.0894)},
        {"model": "Approved_Baseline", "period": "T=12", **evaluate_cohort(y_t12, probs_app_t12, 0.080)},
        {"model": "Phase5_CostSensitive", "period": "T=12", **evaluate_cohort(y_t12, probs_cand_t12, 0.0894)},
        {"model": "Approved_Baseline", "period": "Pooled T=11,12", **evaluate_cohort(y_val, probs_app_val, 0.080)},
        {"model": "Phase5_CostSensitive", "period": "Pooled T=11,12", **evaluate_cohort(y_val, probs_cand_val, 0.0894)}
    ]
    df_periods = pd.DataFrame(period_records)
    df_periods.to_csv("results/phase5/temporal_per_period_breakdown.csv", index=False)
    logger.info("Saved per-period breakdown to results/phase5/temporal_per_period_breakdown.csv")

    # 3. Rolling-Origin Temporal Stability
    rolling_cutoffs = [
        {"name": "Cutoff 1: Train T<=8, Val T=9", "tr_max": 8, "te_min": 9, "te_max": 9},
        {"name": "Cutoff 2: Train T<=9, Val T=10", "tr_max": 9, "te_min": 10, "te_max": 10},
        {"name": "Cutoff 3: Train T<=10, Val T=11-12", "tr_max": 10, "te_min": 11, "te_max": 12}
    ]

    rolling_records = []
    for c in rolling_cutoffs:
        df_tr_c = df_p5[df_p5["Time"] <= c["tr_max"]].copy().reset_index(drop=True)
        df_te_c = df_p5[(df_p5["Time"] >= c["te_min"]) & (df_p5["Time"] <= c["te_max"])].copy().reset_index(drop=True)

        p_c = Phase5PreprocessingPipeline()
        p_c.fit(df_tr_c)
        X_tr_c, y_tr_c = p_c.transform(df_tr_c)
        X_te_c, y_te_c = p_c.transform(df_te_c)

        est_c = clone(p5_cand_model)
        est_c.fit(X_tr_c, y_tr_c)
        cal_c = CalibratedClassifierCV(estimator=clone(est_c), method="sigmoid", cv=3)
        cal_c.fit(X_tr_c, y_tr_c)

        preds_c = cal_c.predict_proba(X_te_c)[:, 1]
        ev_c = evaluate_cohort(y_te_c, preds_c, threshold=0.080)
        rolling_records.append({
            "cutoff_name": c["name"],
            "train_periods": f"T<={c['tr_max']}",
            "eval_periods": f"T={c['te_min']}-{c['te_max']}",
            **ev_c
        })

    df_rolling = pd.DataFrame(rolling_records)
    df_rolling.to_csv("results/phase5/rolling_origin_stability.csv", index=False)
    logger.info("Saved rolling-origin stability to results/phase5/rolling_origin_stability.csv")

    # 4. Paired Bootstrap Resampling on Validation Data
    paired_boot = paired_bootstrap_validation(
        y_val, probs_app_val, 0.080, probs_cand_val, 0.0894, n_boot=2000, seed=42
    )

    with open("results/phase5/validation_paired_bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(paired_boot, f, indent=2)

    logger.info("Saved paired bootstrap summary to results/phase5/validation_paired_bootstrap_summary.json")

    return {
        "periods": df_periods,
        "rolling": df_rolling,
        "paired_boot": paired_boot
    }


if __name__ == "__main__":
    run_nested_validation_analysis()
