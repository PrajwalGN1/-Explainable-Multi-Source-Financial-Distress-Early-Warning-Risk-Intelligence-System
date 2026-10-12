"""
Phase 4: Experiment D — Probability Calibration Analysis.

Evaluates probability calibration quality across candidate models:
1. Baseline_RF_Balanced
2. RF_CostSensitive_w10
3. RF_TrainingROS_15pct

Compares:
- Platt Scaling (Sigmoid / Logistic calibration)
- Isotonic Regression (Non-parametric step calibration)
- Uncalibrated diagnostic baseline (Raw tree voting probability)

Metrics computed:
- Brier Score
- Reference Brier Score (Climatology / Base rate)
- Brier Skill Score (BSS = 1 - BS / BS_ref)
- Log Loss (Binary Cross-Entropy)
- Expected Calibration Error (ECE, 10 bins)
- Maximum Calibration Error (MCE)
- Calibration Slope & Intercept (Cox calibration via logistic regression)
- Reliability Curve coordinates (observed vs predicted per bin)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from scipy.special import logit, expit

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("phase4_calibration_analysis")


def calculate_ece_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float, List[Dict[str, float]]]:
    """Calculate Expected Calibration Error (ECE), MCE, and bin statistics."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include upper boundary on the last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            abs_err = abs(bin_acc - bin_conf)
            ece += (bin_count / len(y_prob)) * abs_err
            mce = max(mce, abs_err)
            bin_stats.append({
                "bin_idx": i,
                "bin_lower": round(float(bin_lower), 3),
                "bin_upper": round(float(bin_upper), 3),
                "count": int(bin_count),
                "confidence": round(float(bin_conf), 4),
                "accuracy": round(float(bin_acc), 4),
                "abs_error": round(float(abs_err), 4)
            })

    return round(float(ece), 4), round(float(mce), 4), bin_stats


def calculate_cox_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Estimate Cox calibration intercept and slope via logistic regression of y on logit(p).
    Ideal: intercept = 0.0, slope = 1.0.
    """
    # Clip probabilities to avoid logit singularity
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logits = logit(p_clipped).reshape(-1, 1)

    try:
        # Fit unpenalized (or very weakly penalized) logistic regression
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500)
        lr.fit(logits, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception as e:
        logger.warning(f"Cox calibration fitting encountered: {e}. Falling back to L2 penalty.")
        lr = LogisticRegression(C=1e3, solver="lbfgs", max_iter=500)
        lr.fit(logits, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])

    return round(intercept, 4), round(slope, 4)


def run_calibration_analysis():
    """Execute complete Experiment D calibration audit."""
    os.makedirs("results/phase4", exist_ok=True)
    os.makedirs("reports/phase4", exist_ok=True)

    # 1. Load dataset & engineer temporal features
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(
        df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
    )
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    df_features = df_feats.dropna(subset=["forward_target"]).copy()
    df_features["forward_target"] = df_features["forward_target"].astype(int)

    train_df = df_features[df_features["Time"] <= 10].copy().reset_index(drop=True)
    val_df = df_features[df_features["Time"].isin([11, 12])].copy().reset_index(drop=True)

    # 2. Fit Preprocessor strictly on Train (T<=10)
    preprocessor = EWSPreprocessor()
    preprocessor.fit(train_df)
    X_train_clean, y_train = preprocessor.transform(train_df)
    X_val_clean, y_val = preprocessor.transform(val_df)

    val_prevalence = float(np.mean(y_val))
    bs_ref = float(val_prevalence * (1.0 - val_prevalence))
    logger.info(f"Val size: {len(y_val)}, positives: {int(np.sum(y_val))}, prevalence: {val_prevalence:.4f}, BS_ref: {bs_ref:.4f}")

    # 3. Candidate models to evaluate
    from sklearn.ensemble import RandomForestClassifier
    from src.evaluation.phase4_model_comparison import random_oversample_binary

    candidates = {
        "Baseline_RF_Balanced": {
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, class_weight="balanced", random_state=42
            ),
            "use_ros": False
        },
        "RF_CostSensitive_w10": {
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, class_weight={0: 1.0, 1: 10.0}, random_state=42
            ),
            "use_ros": False
        },
        "RF_TrainingROS_15pct": {
            "model": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            "use_ros": True,
            "ros_ratio": 0.15
        }
    }

    results = []

    for name, cfg in candidates.items():
        logger.info(f"Auditing Calibration for Model: {name}")
        est = cfg["model"]
        use_ros = cfg["use_ros"]

        if use_ros:
            X_tr, y_tr = random_oversample_binary(X_train_clean, y_train, target_ratio=cfg["ros_ratio"], random_state=42)
        else:
            X_tr, y_tr = X_train_clean, y_train

        # Fit base model on Train
        est.fit(X_tr, y_tr)

        # Raw validation predictions
        raw_val_p = est.predict_proba(X_val_clean)[:, 1]

        # Fit Platt Calibrator strictly on Train (3-fold inner cross-validation)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_train_preds = np.zeros(len(y_tr))
        for tr_idx, cal_idx in skf.split(X_tr, y_tr):
            fold_m = clone_estimator(est)
            fold_m.fit(X_tr[tr_idx], y_tr[tr_idx])
            cv_train_preds[cal_idx] = fold_m.predict_proba(X_tr[cal_idx])[:, 1]

        # Fit Platt (Sigmoid) on CV train predictions
        platt_lr = LogisticRegression(solver="lbfgs", max_iter=500)
        platt_lr.fit(cv_train_preds.reshape(-1, 1), y_tr)
        platt_val_p = platt_lr.predict_proba(raw_val_p.reshape(-1, 1))[:, 1]

        # Fit Isotonic on CV train predictions
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(cv_train_preds, y_tr)
        iso_val_p = iso.predict(raw_val_p)

        # Evaluate each variant
        variants = [
            ("Uncalibrated_Raw", raw_val_p),
            ("Platt_Scaling_Sigmoid", platt_val_p),
            ("Isotonic_Regression", iso_val_p)
        ]

        for method_name, p_preds in variants:
            bs = brier_score_loss(y_val, p_preds)
            bss = 1.0 - (bs / bs_ref) if bs_ref > 0 else 0.0
            
            # Log loss requires clipping
            eps = 1e-6
            p_safe = np.clip(p_preds, eps, 1.0 - eps)
            ll = log_loss(y_val, p_safe)
            
            ece, mce, bin_stats = calculate_ece_mce(y_val, p_preds, n_bins=10)
            intercept, slope = calculate_cox_calibration(y_val, p_preds)
            roc_auc = roc_auc_score(y_val, p_preds)
            pr_auc = average_precision_score(y_val, p_preds)

            results.append({
                "model_name": name,
                "calibration_method": method_name,
                "val_roc_auc": round(float(roc_auc), 4),
                "val_pr_auc": round(float(pr_auc), 4),
                "brier_score": round(float(bs), 4),
                "reference_brier": round(float(bs_ref), 4),
                "brier_skill_score": round(float(bss), 4),
                "log_loss": round(float(ll), 4),
                "ece_10bins": ece,
                "mce": mce,
                "cox_intercept": intercept,
                "cox_slope": slope,
                "mean_pred_prob": round(float(np.mean(p_preds)), 4),
                "val_prevalence": round(val_prevalence, 4),
                "bin_details": bin_stats
            })

    df_cal = pd.DataFrame([{k: v for k, v in r.items() if k != "bin_details"} for r in results])
    df_cal.to_csv("results/phase4/calibration_analysis_comparison.csv", index=False)
    logger.info("Saved calibration analysis table to results/phase4/calibration_analysis_comparison.csv")

    with open("results/phase4/calibration_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved calibration analysis summary to results/phase4/calibration_analysis_summary.json")
    return df_cal


def clone_estimator(est):
    """Deep clone an estimator."""
    from sklearn.base import clone
    return clone(est)


if __name__ == "__main__":
    run_calibration_analysis()
