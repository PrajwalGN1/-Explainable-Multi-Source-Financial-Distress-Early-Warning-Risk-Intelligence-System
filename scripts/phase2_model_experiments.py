"""
Phase 2 Model Experiments & Empirical Validation Runner.
Implements:
- Experiment A: Existing stacking model with original synthetic proxies
- Experiment B: Clean Temporal EWS (Logistic Regression, Random Forest, XGBoost)
- Experiment C: Leak-free Out-Of-Fold Stacking Meta-Model
- Probability Calibration (Platt Scaling)
- Operating Threshold Optimization (Validation fold only)
- Untouched Holdout Test Evaluation (T=13)
"""

import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score
)

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.features.financial_features import FinancialHealthEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("phase2_experiments")


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = (5.0 * prec * rec) / (4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    youden_j = rec + spec - 1.0
    
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_prob)
    
    total = len(y_true)
    flagged = int(tp + fp)
    flagged_pct = (flagged / total) * 100.0 if total > 0 else 0.0
    
    return {
        "threshold": round(float(threshold), 4),
        "total": int(total),
        "distress_events": int(sum(y_true)),
        "base_rate_pct": round(float(np.mean(y_true) * 100.0), 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "specificity": round(float(spec), 4),
        "false_negative_rate": round(float(fnr), 4),
        "false_positive_rate": round(float(fpr), 4),
        "youden_j": round(float(youden_j), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "portfolio_flagged_count": int(flagged),
        "portfolio_flagged_pct": round(float(flagged_pct), 2)
    }


def run_experiments():
    logger.info("=" * 70)
    logger.info("PHASE 2 MODEL VALIDATION & EMPIRICAL RESEARCH RUNNER")
    logger.info("=" * 70)
    
    # 1. Load Data
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()
    
    # 2. Construct Forward Target & Generate Temporal Features
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5)
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    
    df_valid = df_feats.dropna(subset=["forward_target"]).copy()
    df_valid["forward_target"] = df_valid["forward_target"].astype(int)
    
    # Splits
    train_df = df_valid[df_valid["Time"] <= 10].copy()
    val_df = df_valid[(df_valid["Time"] > 10) & (df_valid["Time"] <= 12)].copy()
    test_df = df_valid[df_valid["Time"] > 12].copy()
    
    logger.info(f"Train split (T<=10): {len(train_df)} rows, {train_df['forward_target'].sum()} events ({train_df['forward_target'].mean()*100:.2f}%)")
    logger.info(f"Val split (T=11,12): {len(val_df)} rows, {val_df['forward_target'].sum()} events ({val_df['forward_target'].mean()*100:.2f}%)")
    logger.info(f"Test split (T=13):   {len(test_df)} rows, {test_df['forward_target'].sum()} events ({test_df['forward_target'].mean()*100:.2f}%)")
    
    # 3. Fit Preprocessor strictly on Train split
    preprocessor = EWSPreprocessor(target_col="forward_target")
    preprocessor.fit(train_df)
    
    X_train, y_train = preprocessor.transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
    
    # =========================================================================
    # PART 1: CANDIDATE MODEL TRAINING & COMPARISON (Dataset A Clean)
    # =========================================================================
    logger.info("\n--- PART 1: RETRAINING AND COMPARING BASELINE CANDIDATES ---")
    
    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Logistic Regression (Unweighted)": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest (Balanced, max_depth=5)": RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Random Forest (Unweighted, max_depth=5)": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        "Random Forest (Balanced, max_depth=8)": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost (Weighted, max_depth=4)": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "XGBoost (Unweighted, max_depth=4)": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="logloss", n_jobs=-1
        )
    }
    
    val_candidate_results = []
    fitted_models = {}
    
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        fitted_models[name] = clf
        
        val_probs = clf.predict_proba(X_val)[:, 1]
        m = compute_classification_metrics(y_val, val_probs, threshold=0.50)
        m["Model"] = name
        
        # Also compute at threshold = 0.08 for realistic operational comparison
        m_op = compute_classification_metrics(y_val, val_probs, threshold=0.08)
        m["Recall@0.08"] = m_op["recall"]
        m["Precision@0.08"] = m_op["precision"]
        m["F1@0.08"] = m_op["f1"]
        m["F2@0.08"] = m_op["f2"]
        
        val_candidate_results.append(m)
        logger.info(f"{name:42s} | ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Brier: {m['brier_score']:.4f} | Rec@0.08: {m['Recall@0.08']:.4f}")
        
    df_candidates = pd.DataFrame(val_candidate_results)
    
    # =========================================================================
    # PART 2: SYNTHETIC PROXY CONTAMINATION AUDIT (EXPERIMENTS A, B, C)
    # =========================================================================
    logger.info("\n--- PART 2: SYNTHETIC PROXY CONTAMINATION & STACKING REFACTORING ---")
    
    # Primary model (RF Balanced max_depth=5)
    rf_primary = fitted_models["Random Forest (Balanced, max_depth=5)"]
    
    # Baseline Platt calibrator fitted on val
    calibrator_baseline = CalibratedClassifierCV(estimator=rf_primary, method="sigmoid", cv="prefit")
    calibrator_baseline.fit(X_val, y_val)
    
    val_cal_probs_rf = calibrator_baseline.predict_proba(X_val)[:, 1]
    test_cal_probs_rf = calibrator_baseline.predict_proba(X_test)[:, 1]
    
    health_engine = FinancialHealthEngine()
    
    # Experiment A: Existing Stacking with Synthetic Proxies
    val_health_risks = []
    val_b_proxies = []
    val_anomaly_proxies = []
    for _, row in val_df.iterrows():
        h = health_engine.compute_for_dataset_a_row(row)
        val_health_risks.append(h["financial_health_risk"])
        val_b_proxies.append(float(np.clip(h["financial_health_risk"] * 0.85 + (row.get("x64", 0) > 0.5) * 0.15, 0.0, 1.0)))
        val_anomaly_proxies.append(float(np.clip(abs(row.get("x81", 0)) * 0.2 + (row.get("x64", 0) > 1.0) * 0.3, 0.0, 1.0)))
        
    oof_proxy_meta_val = np.column_stack([
        val_cal_probs_rf,
        np.array(val_b_proxies),
        np.array(val_anomaly_proxies),
        np.array(val_health_risks)
    ])
    
    meta_model_exp_a = LogisticRegression(penalty="l2", C=1.0, random_state=42)
    meta_model_exp_a.fit(oof_proxy_meta_val, y_val)
    exp_a_coefs = dict(zip(["ews_prob", "proxy_bankrupt", "proxy_anomaly", "health_risk"], meta_model_exp_a.coef_[0]))
    logger.info(f"Experiment A (Distorted Proxy Meta-Model) Coefficients: {exp_a_coefs}")
    
    # Test set for Experiment A
    test_health_risks = []
    test_b_proxies = []
    test_anomaly_proxies = []
    for _, row in test_df.iterrows():
        h = health_engine.compute_for_dataset_a_row(row)
        test_health_risks.append(h["financial_health_risk"])
        test_b_proxies.append(float(np.clip(h["financial_health_risk"] * 0.85 + (row.get("x64", 0) > 0.5) * 0.15, 0.0, 1.0)))
        test_anomaly_proxies.append(float(np.clip(abs(row.get("x81", 0)) * 0.2 + (row.get("x64", 0) > 1.0) * 0.3, 0.0, 1.0)))
        
    test_meta_exp_a = np.column_stack([
        test_cal_probs_rf,
        np.array(test_b_proxies),
        np.array(test_anomaly_proxies),
        np.array(test_health_risks)
    ])
    exp_a_test_probs = meta_model_exp_a.predict_proba(test_meta_exp_a)[:, 1]
    exp_a_test_metrics = compute_classification_metrics(y_test, exp_a_test_probs, threshold=0.50)
    logger.info(f"Experiment A Test ROC-AUC: {exp_a_test_metrics['roc_auc']:.4f} | PR-AUC: {exp_a_test_metrics['pr_auc']:.4f}")
    
    # Experiment B: Primary Clean Model (No Proxies)
    exp_b_test_probs = test_cal_probs_rf
    exp_b_test_metrics = compute_classification_metrics(y_test, exp_b_test_probs, threshold=0.50)
    logger.info(f"Experiment B Test ROC-AUC: {exp_b_test_metrics['roc_auc']:.4f} | PR-AUC: {exp_b_test_metrics['pr_auc']:.4f}")
    
    # Experiment C: Properly Constructed Stacking using 5-Fold Group Out-Of-Fold CV on Training Set (T<=10)
    # Stacking ensemble of RF + XGBoost + Logistic Regression
    logger.info("Constructing Experiment C: Out-Of-Fold Meta-Model strictly on Train split (T<=10)...")
    gkf = GroupKFold(n_splits=5)
    train_companies = train_df["Company"].values
    
    oof_preds_rf = np.zeros(len(train_df))
    oof_preds_xgb = np.zeros(len(train_df))
    oof_preds_lr = np.zeros(len(train_df))
    
    for tr_idx, hld_idx in gkf.split(X_train, y_train, groups=train_companies):
        X_tr_f, y_tr_f = X_train[tr_idx], y_train[tr_idx]
        X_hld_f = X_train[hld_idx]
        
        # Fit fold base models
        m_rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1)
        m_rf.fit(X_tr_f, y_tr_f)
        oof_preds_rf[hld_idx] = m_rf.predict_proba(X_hld_f)[:, 1]
        
        m_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss", n_jobs=-1)
        m_xgb.fit(X_tr_f, y_tr_f)
        oof_preds_xgb[hld_idx] = m_xgb.predict_proba(X_hld_f)[:, 1]
        
        m_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        m_lr.fit(X_tr_f, y_tr_f)
        oof_preds_lr[hld_idx] = m_lr.predict_proba(X_hld_f)[:, 1]
        
    oof_train_meta = np.column_stack([oof_preds_rf, oof_preds_xgb, oof_preds_lr])
    
    # Non-negative constrained meta-learner (or Ridge / LogisticRegression with L2)
    meta_model_exp_c = LogisticRegression(penalty="l2", C=0.5, random_state=42)
    meta_model_exp_c.fit(oof_train_meta, y_train)
    exp_c_coefs = dict(zip(["rf_oof", "xgb_oof", "lr_oof"], meta_model_exp_c.coef_[0]))
    logger.info(f"Experiment C OOF Stacking Coefficients: {exp_c_coefs}, Intercept: {meta_model_exp_c.intercept_[0]:.4f}")
    
    # Generate meta-features for val and test using full models fitted on entire train set
    m_rf_full = fitted_models["Random Forest (Balanced, max_depth=5)"]
    m_xgb_full = fitted_models["XGBoost (Weighted, max_depth=4)"]
    m_lr_full = fitted_models["Logistic Regression (Balanced)"]
    
    val_meta_c = np.column_stack([
        m_rf_full.predict_proba(X_val)[:, 1],
        m_xgb_full.predict_proba(X_val)[:, 1],
        m_lr_full.predict_proba(X_val)[:, 1]
    ])
    val_probs_meta_c = meta_model_exp_c.predict_proba(val_meta_c)[:, 1]
    val_meta_c_metrics = compute_classification_metrics(y_val, val_probs_meta_c, threshold=0.50)
    
    test_meta_c = np.column_stack([
        m_rf_full.predict_proba(X_test)[:, 1],
        m_xgb_full.predict_proba(X_test)[:, 1],
        m_lr_full.predict_proba(X_test)[:, 1]
    ])
    test_probs_meta_c = meta_model_exp_c.predict_proba(test_meta_c)[:, 1]
    exp_c_test_metrics = compute_classification_metrics(y_test, test_probs_meta_c, threshold=0.50)
    
    logger.info(f"Experiment C Val ROC-AUC: {val_meta_c_metrics['roc_auc']:.4f} | PR-AUC: {val_meta_c_metrics['pr_auc']:.4f}")
    logger.info(f"Experiment C Test ROC-AUC: {exp_c_test_metrics['roc_auc']:.4f} | PR-AUC: {exp_c_test_metrics['pr_auc']:.4f}")
    
    proxy_comparison_df = pd.DataFrame([
        {
            "Experiment": "Experiment A (Stacking with Distorted Proxies)",
            "Description": "Original meta-model on x64/x81 formulas + val_a fitted",
            "Val_ROC_AUC": round(float(roc_auc_score(y_val, meta_model_exp_a.predict_proba(oof_proxy_meta_val)[:, 1])), 4),
            "Test_ROC_AUC": exp_a_test_metrics["roc_auc"],
            "Test_PR_AUC": exp_a_test_metrics["pr_auc"],
            "Test_Brier": exp_a_test_metrics["brier_score"]
        },
        {
            "Experiment": "Experiment B (Clean Temporal EWS - Random Forest)",
            "Description": "Single primary model without any synthetic proxies",
            "Val_ROC_AUC": round(float(roc_auc_score(y_val, val_cal_probs_rf)), 4),
            "Test_ROC_AUC": exp_b_test_metrics["roc_auc"],
            "Test_PR_AUC": exp_b_test_metrics["pr_auc"],
            "Test_Brier": exp_b_test_metrics["brier_score"]
        },
        {
            "Experiment": "Experiment C (Proper Leak-Free OOF Stacking)",
            "Description": "OOF on Train (T<=10) of RF + XGB + LR, evaluated on Val/Test",
            "Val_ROC_AUC": val_meta_c_metrics["roc_auc"],
            "Test_ROC_AUC": exp_c_test_metrics["roc_auc"],
            "Test_PR_AUC": exp_c_test_metrics["pr_auc"],
            "Test_Brier": exp_c_test_metrics["brier_score"]
        }
    ])
    
    # =========================================================================
    # PART 3: PROBABILITY CALIBRATION ANALYSIS (PLATT SCALING)
    # =========================================================================
    logger.info("\n--- PART 3: PROBABILITY CALIBRATION EVALUATION ---")
    
    raw_val_probs_rf = rf_primary.predict_proba(X_val)[:, 1]
    raw_test_probs_rf = rf_primary.predict_proba(X_test)[:, 1]
    
    brier_val_raw = brier_score_loss(y_val, raw_val_probs_rf)
    brier_val_cal = brier_score_loss(y_val, val_cal_probs_rf)
    brier_test_raw = brier_score_loss(y_test, raw_test_probs_rf)
    brier_test_cal = brier_score_loss(y_test, test_cal_probs_rf)
    
    logger.info(f"Val Brier: Raw = {brier_val_raw:.4f} -> Calibrated = {brier_val_cal:.4f} ({(brier_val_raw - brier_val_cal)/brier_val_raw*100:+.2f}%)")
    logger.info(f"Test Brier: Raw = {brier_test_raw:.4f} -> Calibrated = {brier_test_cal:.4f} ({(brier_test_raw - brier_test_cal)/brier_test_raw*100:+.2f}%)")
    
    prob_true_val, prob_pred_val = calibration_curve(y_val, val_cal_probs_rf, n_bins=5, strategy="uniform")
    calibration_diag = {
        "val_brier_raw": round(float(brier_val_raw), 4),
        "val_brier_calibrated": round(float(brier_val_cal), 4),
        "test_brier_raw": round(float(brier_test_raw), 4),
        "test_brier_calibrated": round(float(brier_test_cal), 4),
        "val_calibrated_curve": {
            "prob_true": [round(float(x), 4) for x in prob_true_val],
            "prob_pred": [round(float(x), 4) for x in prob_pred_val]
        }
    }
    
    # =========================================================================
    # PART 4: OPERATING DECISION THRESHOLD OPTIMIZATION (VALIDATION FOLD ONLY)
    # =========================================================================
    logger.info("\n--- PART 4: OPERATING THRESHOLD OPTIMIZATION (VALIDATION FOLD ONLY) ---")
    
    candidate_thresholds = [0.05, 0.08, 0.10, 0.13, 0.17, 0.50]
    val_threshold_table = []
    
    for tau in candidate_thresholds:
        row_metrics = compute_classification_metrics(y_val, val_cal_probs_rf, threshold=tau)
        val_threshold_table.append(row_metrics)
        logger.info(
            f"Threshold {tau:4.2f} | TP: {row_metrics['tp']:2d} | FP: {row_metrics['fp']:2d} | FN: {row_metrics['fn']:2d} | TN: {row_metrics['tn']:3d} | "
            f"Rec: {row_metrics['recall']*100:5.1f}% | Prec: {row_metrics['precision']*100:5.1f}% | F1: {row_metrics['f1']:.4f} | F2: {row_metrics['f2']:.4f} | "
            f"Flagged: {row_metrics['portfolio_flagged_pct']:4.1f}% ({row_metrics['portfolio_flagged_count']}/{row_metrics['total']})"
        )
        
    df_val_thresholds = pd.DataFrame(val_threshold_table)
    
    # Objective selection of optimal threshold tau* on VALIDATION FOLD ONLY
    # Target: Operational Recall >= 60% with manageable false-positive burden (maximizing F2-score)
    # Filter candidates with Recall >= 0.60
    candidates_ge_60 = [m for m in val_threshold_table if m["recall"] >= 0.60]
    if candidates_ge_60:
        optimal_val_row = max(candidates_ge_60, key=lambda x: (x["f2"], x["youden_j"]))
    else:
        optimal_val_row = max(val_threshold_table, key=lambda x: x["f2"])
        
    tau_star = optimal_val_row["threshold"]
    logger.info(f"\nOPTIMAL OPERATING THRESHOLD SELECTED ON VALIDATION SET: tau* = {tau_star:.4f}")
    logger.info(f"Selection Rationale: Maximizes F2-score ({optimal_val_row['f2']:.4f}) while meeting operational recall target >= 60% ({optimal_val_row['recall']*100:.1f}%), flagging {optimal_val_row['portfolio_flagged_pct']:.1f}% of portfolio.")
    
    # =========================================================================
    # PART 5: FINAL EVALUATION ON UNTOUCHED TEST SET (T=13)
    # =========================================================================
    logger.info("\n--- PART 5: EVALUATION ON UNTOUCHED TEST SET (T=13) ---")
    
    # Evaluate at baseline naive threshold 0.50
    test_metrics_naive_050 = compute_classification_metrics(y_test, test_cal_probs_rf, threshold=0.50)
    logger.info(f"Test @ Naive 0.50 | TP: {test_metrics_naive_050['tp']} | FP: {test_metrics_naive_050['fp']} | FN: {test_metrics_naive_050['fn']} | TN: {test_metrics_naive_050['tn']} | Rec: {test_metrics_naive_050['recall']*100:.1f}%")
    
    # Evaluate at frozen optimal threshold tau*
    test_metrics_optimal_tau = compute_classification_metrics(y_test, test_cal_probs_rf, threshold=tau_star)
    logger.info(
        f"Test @ Frozen tau*={tau_star:.2f} | TP: {test_metrics_optimal_tau['tp']:2d} | FP: {test_metrics_optimal_tau['fp']:2d} | FN: {test_metrics_optimal_tau['fn']:2d} | TN: {test_metrics_optimal_tau['tn']:3d} | "
        f"Rec: {test_metrics_optimal_tau['recall']*100:5.1f}% | Prec: {test_metrics_optimal_tau['precision']*100:5.1f}% | F1: {test_metrics_optimal_tau['f1']:.4f} | F2: {test_metrics_optimal_tau['f2']:.4f} | "
        f"Spec: {test_metrics_optimal_tau['specificity']*100:5.1f}% | Flagged: {test_metrics_optimal_tau['portfolio_flagged_pct']:4.1f}%"
    )
    
    # Evaluate all candidate thresholds on test set for honest reporting & full trade-off disclosure
    test_all_thresholds = [compute_classification_metrics(y_test, test_cal_probs_rf, threshold=tau) for tau in candidate_thresholds]
    df_test_thresholds = pd.DataFrame(test_all_thresholds)
    
    # Compute 95% Wilson Score Confidence Interval for Test Recall
    # Wilson interval for p = TP / (TP + FN) with n = 11
    def wilson_ci(pos_hits, n_total, z=1.96):
        if n_total == 0:
            return (0.0, 0.0)
        p = pos_hits / n_total
        denom = 1.0 + (z**2) / n_total
        center = (p + (z**2) / (2.0 * n_total)) / denom
        diff = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n_total)) / n_total) / denom
        return (max(0.0, round(center - diff, 4)), min(1.0, round(center + diff, 4)))
        
    test_recall_ci = wilson_ci(test_metrics_optimal_tau["tp"], test_metrics_optimal_tau["distress_events"])
    logger.info(f"Test Recall 95% Wilson Confidence Interval: [{test_recall_ci[0]*100:.1f}%, {test_recall_ci[1]*100:.1f}%]")
    
    # =========================================================================
    # PART 6: PERSIST PHASE 2 ARTIFACTS
    # =========================================================================
    logger.info("\n--- PART 6: SAVING PHASE 2 MODEL ARTIFACTS ---")
    
    os.makedirs("models/phase2", exist_ok=True)
    os.makedirs("reports/phase2", exist_ok=True)
    os.makedirs("reports/audit", exist_ok=True)
    
    # Save Model Artifacts
    joblib.dump(rf_primary, "models/phase2/best_ews_model.joblib")
    preprocessor.save("models/phase2/ews_preprocessor.joblib")
    joblib.dump(calibrator_baseline, "models/phase2/ews_calibrator.joblib")
    
    threshold_config = {
        "operating_threshold": float(tau_star),
        "selection_metric": "F2-Score (Recall >= 60% constraint)",
        "selection_fold": "Validation Fold (T=11,12)",
        "candidate_thresholds": candidate_thresholds,
        "critical_alert_threshold": 0.170,
        "high_alert_threshold": 0.130,
        "watch_score_threshold": 8.0,
        "high_score_threshold": 13.0,
        "critical_score_threshold": 17.0
    }
    with open("models/phase2/threshold_config.json", "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
        
    ews_metrics = {
        "model_name": "Random Forest (Balanced, max_depth=5)",
        "validation_metrics_optimal_tau": optimal_val_row,
        "test_metrics_naive_050": test_metrics_naive_050,
        "test_metrics_optimal_tau": test_metrics_optimal_tau,
        "test_recall_95ci": test_recall_ci,
        "calibration": calibration_diag,
        "feature_count": len(preprocessor.feature_names)
    }
    with open("models/phase2/ews_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ews_metrics, f, indent=2)
        
    model_manifest = {
        "phase": "Phase 2 Model Validation & Stacking Refactoring",
        "primary_model_type": "RandomForestClassifier",
        "preprocessor_type": "EWSPreprocessor",
        "calibrator_type": "CalibratedClassifierCV (Platt / Sigmoid)",
        "dataset_a_train_periods": "1 <= T <= 10 (N=2471)",
        "dataset_a_val_periods": "11 <= T <= 12 (N=517)",
        "dataset_a_test_periods": "T == 13 (N=262)",
        "status": "Validated",
        "synthetic_proxies_removed": True,
        "operating_threshold": float(tau_star),
        "test_roc_auc": test_metrics_optimal_tau["roc_auc"],
        "test_pr_auc": test_metrics_optimal_tau["pr_auc"],
        "test_recall": test_metrics_optimal_tau["recall"],
        "test_precision": test_metrics_optimal_tau["precision"]
    }
    with open("models/phase2/model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(model_manifest, f, indent=2)
        
    # Save CSV tables
    df_candidates.to_csv("reports/phase2/candidate_model_comparisons.csv", index=False)
    proxy_comparison_df.to_csv("reports/phase2/proxy_contamination_experiments.csv", index=False)
    df_val_thresholds.to_csv("reports/phase2/validation_threshold_analysis.csv", index=False)
    df_test_thresholds.to_csv("reports/phase2/test_threshold_analysis.csv", index=False)
    
    logger.info("Phase 2 empirical experiment run completed successfully!")
    return {
        "candidates": df_candidates,
        "proxy_comparison": proxy_comparison_df,
        "val_thresholds": df_val_thresholds,
        "test_thresholds": df_test_thresholds,
        "tau_star": tau_star,
        "test_metrics_optimal_tau": test_metrics_optimal_tau
    }


if __name__ == "__main__":
    run_experiments()
