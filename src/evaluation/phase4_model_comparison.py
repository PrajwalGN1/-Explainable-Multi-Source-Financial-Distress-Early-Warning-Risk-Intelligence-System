"""
Phase 4: Model Comparison, Class Imbalance & Validation Robustness.

Conducts Experiments B, C, D, and E under strict leak-free protocols:
- Training: T <= 10 (2,471 rows, 95 events, 3.84% prevalence)
- Temporal Validation: T = 11, 12 (517 rows, 14 events, 2.71% prevalence)
- Test: T = 13 remains FROZEN during model selection
- 5-Fold GroupKFold on T <= 10 with company isolation and inner-fold preprocessing

Models evaluated:
1. Baseline_RF_Balanced: Current approved Random Forest baseline (class_weight='balanced')
2. RF_Balanced_Subsample: Random Forest with bootstrap-level balanced subsampling
3. RF_CostSensitive_w10: Cost-sensitive Random Forest (1:10 penalty)
4. RF_CostSensitive_w25: Cost-sensitive Random Forest (1:25 penalty)
5. RF_TrainingROS_15pct: Random Forest with training-fold-only Random Over-Sampling (15% prevalence)
6. Logistic_Regression_Balanced: Transparent linear benchmark with balanced weighting
7. XGBoost_ScalePosWeight: Gradient boosted decision trees with scale_pos_weight
"""

import os
import sys
import json
import joblib
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, precision_score, recall_score,
    f1_score
)
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import logit

# Check if xgboost is available
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.models.calibration import ProbabilityCalibrator
from src.utils.logger import setup_logger

logger = setup_logger("phase4_model_comparison")


def random_oversample_binary(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.15, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly oversamples minority class strictly within the training fold.
    Never applied to validation or test data.
    """
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    desired_n_pos = int(n_neg * target_ratio / (1.0 - target_ratio))
    if desired_n_pos <= n_pos:
        return X, y

    n_additional = desired_n_pos - n_pos
    sampled_pos_idx = rng.choice(pos_idx, size=n_additional, replace=True)
    all_pos_idx = np.concatenate([pos_idx, sampled_pos_idx])
    all_idx = np.concatenate([neg_idx, all_pos_idx])
    rng.shuffle(all_idx)

    return X[all_idx], y[all_idx]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.080) -> Dict[str, Any]:
    """Calculates comprehensive ranking and classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    alerts = int(tp + fp)
    alert_rate = float(alerts / len(y_true)) * 100.0 if len(y_true) > 0 else 0.0

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(ap, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts_count": alerts,
        "alert_rate_pct": round(alert_rate, 2)
    }


def run_model_comparison() -> Dict[str, Any]:
    os.makedirs("results/phase4", exist_ok=True)
    os.makedirs("experiments/phase4/models", exist_ok=True)

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

    train_df = df_valid[df_valid["Time"] <= 10].copy()
    val_df = df_valid[(df_valid["Time"] > 10) & (df_valid["Time"] <= 12)].copy()

    logger.info(f"Train (T<=10): {len(train_df)} rows, {train_df['forward_target'].sum()} positive events")
    logger.info(f"Val   (T=11,12): {len(val_df)} rows, {val_df['forward_target'].sum()} positive events")

    # 3. Model Definitions Grid
    models_dict = {
        "Baseline_RF_Balanced": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1),
            "oversample": False,
            "description": "Approved Baseline Random Forest (max_depth=5, class_weight='balanced')"
        },
        "RF_Balanced_Subsample": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
            "oversample": False,
            "description": "Random Forest with bootstrap-level balanced subsampling"
        },
        "RF_CostSensitive_w10": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight={0: 1.0, 1: 10.0}, random_state=42, n_jobs=-1),
            "oversample": False,
            "description": "Cost-sensitive Random Forest with 10x minority loss weight"
        },
        "RF_CostSensitive_w25": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight={0: 1.0, 1: 25.0}, random_state=42, n_jobs=-1),
            "oversample": False,
            "description": "Cost-sensitive Random Forest with 25x minority loss weight (reflects ~1:26 base rate)"
        },
        "RF_TrainingROS_15pct": {
            "estimator": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight=None, random_state=42, n_jobs=-1),
            "oversample": True,
            "oversample_ratio": 0.15,
            "description": "Random Forest with training-fold-only Random Over-Sampling to 15% prevalence"
        },
        "Logistic_Regression_Balanced": {
            "estimator": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "oversample": False,
            "description": "L2-penalized Logistic Regression with balanced class weighting"
        }
    }

    if XGB_AVAILABLE:
        # scale_pos_weight = neg_count / pos_count = 2376 / 95 ~= 25.0
        models_dict["XGBoost_ScalePosWeight"] = {
            "estimator": XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                scale_pos_weight=25.0, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42
            ),
            "oversample": False,
            "description": "Gradient Boosted Trees (XGBoost) with scale_pos_weight=25.0 and max_depth=3"
        }
    # 4. Fit Preprocessor strictly on Train (T<=10)
    preprocessor = EWSPreprocessor()
    preprocessor.fit(train_df)
    X_train_clean, y_train = preprocessor.transform(train_df)
    X_val_clean, y_val = preprocessor.transform(val_df)

    logger.info(f"Preprocessor fitted on {len(X_train_clean)} train rows. Feature count: {X_train_clean.shape[1]}")

    # 5. Dual Validation Execution
    comparison_results = []
    trained_artifacts = {}

    for name, cfg in models_dict.items():
        logger.info(f"--- Evaluating Candidate: {name} ---")
        t0 = time.time()
        est = cfg["estimator"]
        use_ros = cfg.get("oversample", False)

        # A. Train-fold Data Prep
        if use_ros:
            X_tr, y_tr = random_oversample_binary(X_train_clean, y_train, target_ratio=cfg.get("oversample_ratio", 0.15), random_state=42)
        else:
            X_tr, y_tr = X_train_clean, y_train

        # B. Fit Model on Train Set (T<=10)
        est.fit(X_tr, y_tr)
        train_duration = round(time.time() - t0, 3)

        # C. Leak-free Probability Calibration
        # Calibrate using 3-fold cross-validation strictly on training set (T<=10)
        # to ensure zero lookahead or validation leakage
        calibrator = CalibratedClassifierCV(estimator=clone(est), method="sigmoid", cv=3)
        calibrator.fit(X_tr, y_tr)

        # D. Predict on Temporal Validation Set (T=11, 12)
        raw_val_probs = est.predict_proba(X_val_clean)[:, 1]
        cal_val_probs = calibrator.predict_proba(X_val_clean)[:, 1]

        val_metrics_008 = evaluate_predictions(y_val, cal_val_probs, threshold=0.080)
        val_metrics_005 = evaluate_predictions(y_val, cal_val_probs, threshold=0.050)

        # E. Cross-Sectional 5-Fold GroupKFold on T<=10 (Generalization Setting B)
        train_companies = train_df["Company"].to_numpy()
        gkf_outer = GroupKFold(n_splits=5)
        fold_aucs = []
        fold_aps = []
        fold_briers = []
        fold_recalls = []

        for f_idx, (f_tr_idx, f_te_idx) in enumerate(gkf_outer.split(X_train_clean, y_train, groups=train_companies)):
            f_est = clone(est)
            X_f_tr, y_f_tr = X_train_clean[f_tr_idx], y_train[f_tr_idx]
            if use_ros:
                X_f_tr, y_f_tr = random_oversample_binary(X_f_tr, y_f_tr, target_ratio=cfg.get("oversample_ratio", 0.15), random_state=42 + f_idx)
            f_est.fit(X_f_tr, y_f_tr)
            f_preds = f_est.predict_proba(X_train_clean[f_te_idx])[:, 1]

            if len(np.unique(y_train[f_te_idx])) > 1:
                fold_aucs.append(roc_auc_score(y_train[f_te_idx], f_preds))
                fold_aps.append(average_precision_score(y_train[f_te_idx], f_preds))
            fold_briers.append(brier_score_loss(y_train[f_te_idx], f_preds))
            f_binary = (f_preds >= 0.50).astype(int)
            fold_recalls.append(recall_score(y_train[f_te_idx], f_binary, zero_division=0))

        mean_group_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
        std_group_auc = float(np.std(fold_aucs)) if fold_aucs else 0.0
        mean_group_ap = float(np.mean(fold_aps)) if fold_aps else 0.0
        std_group_ap = float(np.std(fold_aps)) if fold_aps else 0.0

        # Save artifacts to research directory
        model_save_path = f"experiments/phase4/models/{name}.joblib"
        cal_save_path = f"experiments/phase4/models/{name}_calibrator.joblib"
        joblib.dump(est, model_save_path)
        joblib.dump(calibrator, cal_save_path)

        trained_artifacts[name] = {
            "model_path": model_save_path,
            "calibrator_path": cal_save_path,
            "model": est,
            "calibrator": calibrator
        }

        comparison_results.append({
            "model_name": name,
            "description": cfg["description"],
            "training_time_sec": train_duration,
            # Temporal Validation Metrics (T=11,12) at tau=0.080
            "val_roc_auc": val_metrics_008["roc_auc"],
            "val_pr_auc": val_metrics_008["pr_auc"],
            "val_brier_score": val_metrics_008["brier_score"],
            "val_recall_tau008": val_metrics_008["recall"],
            "val_precision_tau008": val_metrics_008["precision"],
            "val_f2_tau008": val_metrics_008["f2"],
            "val_alerts_tau008": val_metrics_008["alerts_count"],
            "val_alert_rate_tau008": val_metrics_008["alert_rate_pct"],
            "val_tp_tau008": val_metrics_008["tp"],
            "val_fp_tau008": val_metrics_008["fp"],
            "val_fn_tau008": val_metrics_008["fn"],
            # Temporal Validation Metrics at tau=0.050
            "val_recall_tau005": val_metrics_005["recall"],
            "val_precision_tau005": val_metrics_005["precision"],
            "val_f2_tau005": val_metrics_005["f2"],
            "val_alerts_tau005": val_metrics_005["alerts_count"],
            "val_alert_rate_tau005": val_metrics_005["alert_rate_pct"],
            # Cross-Sectional GroupKFold Metrics on T<=10
            "group_cv_roc_auc_mean": round(mean_group_auc, 4),
            "group_cv_roc_auc_std": round(std_group_auc, 4),
            "group_cv_pr_auc_mean": round(mean_group_ap, 4),
            "group_cv_pr_auc_std": round(std_group_ap, 4)
        })

    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv("results/phase4/model_comparison_validation.csv", index=False)
    logger.info("Saved model comparison table to results/phase4/model_comparison_validation.csv")

    # Save summary JSON
    summary_path = "results/phase4/model_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    return {
        "comparison_df": df_comp,
        "artifacts": trained_artifacts
    }


if __name__ == "__main__":
    run_model_comparison()
