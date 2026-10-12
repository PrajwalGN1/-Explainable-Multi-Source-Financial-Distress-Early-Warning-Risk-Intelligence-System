"""
Phase 4: Final Test Protocol & Robust Validation.

Executes Section 12 of the Phase 4 Master Implementation Prompt:
- Evaluates frozen candidate models against the approved baseline on the final holdout test cohort (T=13, N=262, N_pos=11).
- Strictly preserves approved artifacts in models/phase2/ (no modification).
- Compares:
  1. Approved Baseline: RF (class_weight='balanced', Platt cal) at tau=0.080
  2. Research Operating Policy: Approved RF at tau=0.050
  3. Research Candidate A: RF_CostSensitive_w10 at tau=0.080
  4. Research Candidate B: RF_CostSensitive_w10 at tau=0.050
  5. Research Candidate C: RF_TrainingROS_15pct at tau=0.080
  6. Research Candidate D: RF_TrainingROS_15pct at tau=0.050
- Computes Wilson 95% CIs and paired bootstrap resampling (2,000 iterations) for differences (Delta Recall, Delta Precision, Delta Alerts).
- Discloses the small sample limitation (11 distress cases in test).
"""

import os
import sys
import json
import hashlib
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)

sys.path.insert(0, os.path.abspath("."))

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("phase4_recall_validation")


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def wilson_ci(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total) / denom
    return (max(0.0, round(float(center - margin), 4)), min(1.0, round(float(center + margin), 4)))


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute full suite of classification and calibration metrics at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)
    n_total = len(y_true)

    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # F2 score
    beta = 2.0
    f2 = float((1.0 + beta**2) * prec * rec / ((beta**2 * prec) + rec)) if ((beta**2 * prec) + rec) > 0 else 0.0

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    bs = float(brier_score_loss(y_true, y_prob))

    rec_ci = wilson_ci(tp, n_pos)
    prec_ci = wilson_ci(tp, int(tp + fp))

    return {
        "threshold": threshold,
        "n_obs": n_total,
        "n_pos": n_pos,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts_count": int(tp + fp),
        "alert_rate_pct": round(float(tp + fp) / n_total * 100.0, 2),
        "recall": round(rec, 4),
        "recall_ci_95": rec_ci,
        "precision": round(prec, 4),
        "precision_ci_95": prec_ci,
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(bs, 4)
    }


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    baseline_probs: np.ndarray,
    baseline_thresh: float,
    candidate_probs: np.ndarray,
    candidate_thresh: float,
    n_boot: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """Perform paired bootstrap resampling to estimate uncertainty in metric differences."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    diff_recalls = []
    diff_precisions = []
    diff_alerts = []
    diff_roc_aucs = []

    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        y_b = y_true[indices]
        if len(np.unique(y_b)) < 2:
            continue
            
        p_base = baseline_probs[indices]
        p_cand = candidate_probs[indices]
        
        pred_base = (p_base >= baseline_thresh).astype(int)
        pred_cand = (p_cand >= candidate_thresh).astype(int)
        
        rec_base = recall_score(y_b, pred_base, zero_division=0)
        rec_cand = recall_score(y_b, pred_cand, zero_division=0)
        diff_recalls.append(rec_cand - rec_base)
        
        prec_base = precision_score(y_b, pred_base, zero_division=0)
        prec_cand = precision_score(y_b, pred_cand, zero_division=0)
        diff_precisions.append(prec_cand - prec_base)
        
        alerts_base = np.sum(pred_base)
        alerts_cand = np.sum(pred_cand)
        diff_alerts.append(alerts_cand - alerts_base)
        
        diff_roc_aucs.append(roc_auc_score(y_b, p_cand) - roc_auc_score(y_b, p_base))

    return {
        "n_bootstraps": len(diff_recalls),
        "delta_recall_mean": round(float(np.mean(diff_recalls)), 4),
        "delta_recall_ci_95": [round(float(np.percentile(diff_recalls, 2.5)), 4), round(float(np.percentile(diff_recalls, 97.5)), 4)],
        "delta_precision_mean": round(float(np.mean(diff_precisions)), 4),
        "delta_precision_ci_95": [round(float(np.percentile(diff_precisions, 2.5)), 4), round(float(np.percentile(diff_precisions, 97.5)), 4)],
        "delta_alerts_mean": round(float(np.mean(diff_alerts)), 1),
        "delta_alerts_ci_95": [round(float(np.percentile(diff_alerts, 2.5)), 1), round(float(np.percentile(diff_alerts, 97.5)), 1)],
        "delta_roc_auc_mean": round(float(np.mean(diff_roc_aucs)), 4),
        "delta_roc_auc_ci_95": [round(float(np.percentile(diff_roc_aucs, 2.5)), 4), round(float(np.percentile(diff_roc_aucs, 97.5)), 4)],
        "p_value_recall_different_from_zero": round(float(np.mean(np.array(diff_recalls) <= 0.0) if np.mean(diff_recalls) > 0 else np.mean(np.array(diff_recalls) >= 0.0) * 2.0), 4)
    }


def run_final_test_evaluation():
    """Execute formal test-set evaluation protocol comparing baseline and research candidates."""
    os.makedirs("results/phase4", exist_ok=True)
    os.makedirs("reports/phase4", exist_ok=True)

    # 1. Record Artifact Hashes
    approved_rf_path = "models/phase2/best_ews_model.joblib"
    approved_cal_path = "models/phase2/ews_calibrator.joblib"
    approved_prep_path = "models/phase2/ews_preprocessor.joblib"

    hashes = {
        "approved_rf_model": compute_file_hash(approved_rf_path),
        "approved_calibrator": compute_file_hash(approved_cal_path),
        "approved_preprocessor": compute_file_hash(approved_prep_path),
        "phase4_w10_model": compute_file_hash("experiments/phase4/models/RF_CostSensitive_w10.joblib"),
        "phase4_w10_calibrator": compute_file_hash("experiments/phase4/models/RF_CostSensitive_w10_calibrator.joblib"),
        "phase4_ros_model": compute_file_hash("experiments/phase4/models/RF_TrainingROS_15pct.joblib"),
        "phase4_ros_calibrator": compute_file_hash("experiments/phase4/models/RF_TrainingROS_15pct_calibrator.joblib"),
    }
    logger.info("Recorded artifact hashes for immutability verification.")

    # 2. Load Data & Filter to Test Cohort (T=13)
    loader = DataLoader("data/raw")
    df_a = loader.load_dataset_a()
    engine = TemporalFeatureEngine()
    df_target = engine.construct_forward_target(
        df_a, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5
    )
    df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
    df_clean = df_feats.dropna(subset=["forward_target"]).copy()
    df_clean["forward_target"] = df_clean["forward_target"].astype(int)

    test_df = df_clean[df_clean["Time"] == 13].copy().reset_index(drop=True)
    logger.info(f"Test cohort T=13 loaded: {len(test_df)} observations, {int(test_df['forward_target'].sum())} actual distress cases.")

    # 3. Transform via Approved Preprocessor
    approved_preprocessor = EWSPreprocessor.load(approved_prep_path)
    X_test_clean, y_test = approved_preprocessor.transform(test_df)

    # 4. Generate Predictions from Approved Baseline
    approved_rf = joblib.load(approved_rf_path)
    approved_cal = joblib.load(approved_cal_path)

    raw_baseline_probs = approved_rf.predict_proba(X_test_clean)[:, 1]
    cal_baseline_probs = approved_cal.predict_proba(X_test_clean)[:, 1]

    # Baseline Evaluation at tau=0.080
    base_eval_008 = evaluate_metrics(y_test, cal_baseline_probs, threshold=0.080)
    base_eval_008["model_name"] = "Approved_Baseline_RF"
    base_eval_008["policy"] = "Production_Approved (tau=0.080)"

    # Baseline at Candidate Threshold tau=0.050
    base_eval_005 = evaluate_metrics(y_test, cal_baseline_probs, threshold=0.050)
    base_eval_005["model_name"] = "Approved_Baseline_RF"
    base_eval_005["policy"] = "Research_Threshold_Sweep (tau=0.050)"

    # 5. Evaluate Candidate Models
    test_evaluations = [base_eval_008, base_eval_005]
    pairwise_comparisons = {}

    candidates_to_test = [
        ("RF_CostSensitive_w10", "experiments/phase4/models/RF_CostSensitive_w10.joblib", "experiments/phase4/models/RF_CostSensitive_w10_calibrator.joblib"),
        ("RF_TrainingROS_15pct", "experiments/phase4/models/RF_TrainingROS_15pct.joblib", "experiments/phase4/models/RF_TrainingROS_15pct_calibrator.joblib")
    ]

    for cand_name, mod_p, cal_p in candidates_to_test:
        if os.path.exists(mod_p) and os.path.exists(cal_p):
            c_mod = joblib.load(mod_p)
            c_cal = joblib.load(cal_p)
            c_probs = c_cal.predict_proba(X_test_clean)[:, 1]

            # At tau=0.080
            eval_008 = evaluate_metrics(y_test, c_probs, threshold=0.080)
            eval_008["model_name"] = cand_name
            eval_008["policy"] = f"Candidate Model (tau=0.080)"
            test_evaluations.append(eval_008)

            # At tau=0.050
            eval_005 = evaluate_metrics(y_test, c_probs, threshold=0.050)
            eval_005["model_name"] = cand_name
            eval_005["policy"] = f"Candidate Model (tau=0.050)"
            test_evaluations.append(eval_005)

            # Paired bootstrap vs Approved Baseline (tau=0.080)
            comp_key_008 = f"{cand_name}_tau008_vs_Baseline_tau008"
            pairwise_comparisons[comp_key_008] = paired_bootstrap_comparison(
                y_test, cal_baseline_probs, 0.080, c_probs, 0.080
            )

            comp_key_005 = f"{cand_name}_tau005_vs_Baseline_tau008"
            pairwise_comparisons[comp_key_005] = paired_bootstrap_comparison(
                y_test, cal_baseline_probs, 0.080, c_probs, 0.050
            )

    # Also compute paired bootstrap for Baseline tau=0.050 vs Baseline tau=0.080
    pairwise_comparisons["Baseline_tau005_vs_Baseline_tau008"] = paired_bootstrap_comparison(
        y_test, cal_baseline_probs, 0.080, cal_baseline_probs, 0.050
    )

    df_test_eval = pd.DataFrame(test_evaluations)
    df_test_eval.to_csv("results/phase4/test_final_model_comparison.csv", index=False)
    logger.info("Saved final test model comparison table to results/phase4/test_final_model_comparison.csv")

    full_output = {
        "artifact_hashes": hashes,
        "test_cohort_size": len(test_df),
        "test_distress_cases": int(np.sum(y_test)),
        "test_evaluations": test_evaluations,
        "pairwise_bootstrap_comparisons": pairwise_comparisons
    }

    with open("results/phase4/test_final_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    logger.info("Saved test final evaluation summary to results/phase4/test_final_evaluation_summary.json")
    return full_output


if __name__ == "__main__":
    run_final_test_evaluation()
