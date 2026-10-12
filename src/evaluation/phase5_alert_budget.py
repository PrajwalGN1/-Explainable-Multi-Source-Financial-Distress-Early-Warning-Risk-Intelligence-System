"""
Phase 5: Alert-Budget Optimization & Policy Evaluation.

Evaluates operational decision-making under predefined alert-budget constraints:
- Scenario 1: Strict Workload Alert Budget (~5% portfolio alert rate)
- Scenario 2: Baseline Reference Alert Budget (~8% portfolio alert rate, matching approved policy)
- Scenario 3: Moderate Sensitivity Alert Budget (~10% portfolio alert rate)
- Scenario 4: High-Recall Investigative Alert Budget (~15% portfolio alert rate)

All operating points are selected and evaluated STRICTLY on validation data (T in {11, 12}).
For each model and budget scenario, calculates:
- Exact threshold tau_{budget} achieving target review volume
- Confusion matrix (TP, FP, TN, FN)
- Distress Recall, Precision, Specificity, F1, F2
- Total alerts, Alert Rate, Review Workload (FP/TP ratio)
- Wilson 95% confidence intervals for Recall and Precision
- Calibration diagnostics (Brier Score, BSS, Log Loss)
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

sys.path.insert(0, os.path.abspath("."))
from src.utils.logger import setup_logger

logger = setup_logger("phase5_alert_budget")


def wilson_ci(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2.0 * total)) / denom
    margin = z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * total)) / total) / denom
    return (max(0.0, round(float(center - margin), 4)), min(1.0, round(float(center + margin), 4)))


def find_threshold_for_alert_budget(
    y_prob: np.ndarray,
    target_alert_rate: float
) -> float:
    """
    Finds the probability threshold tau that achieves approximately target_alert_rate.
    target_alert_rate is in percentage (e.g., 5.0 for 5%).
    Threshold is quantile corresponding to (100 - target_alert_rate) percentile.
    """
    percentile = max(0.0, min(100.0, 100.0 - target_alert_rate))
    tau = float(np.percentile(y_prob, percentile))
    return round(tau, 4)


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    budget_label: str = "custom",
    target_budget_pct: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluates full classification, workload, and calibration metrics at a threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_obs = len(y_true)
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)
    alerts = int(tp + fp)
    alert_rate = (alerts / n_obs) * 100.0 if n_obs > 0 else 0.0

    rec = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # F2 score (recall weighted twice as heavily as precision)
    beta = 2.0
    f2 = float((1.0 + beta**2) * prec * rec / ((beta**2 * prec) + rec)) if ((beta**2 * prec) + rec) > 0 else 0.0

    # Review workload ratio: False Positives per True Positive
    fp_per_tp = round(float(fp / tp), 2) if tp > 0 else float("inf")

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    # Reference Brier (climatology)
    base_rate = float(np.mean(y_true))
    brier_ref = base_rate * (1.0 - base_rate)
    bss = 1.0 - (brier / brier_ref) if brier_ref > 0 else 0.0

    eps = 1e-6
    p_safe = np.clip(y_prob, eps, 1.0 - eps)
    ll = float(log_loss(y_true, p_safe))

    rec_ci = wilson_ci(tp, n_pos)
    prec_ci = wilson_ci(tp, alerts)

    return {
        "budget_scenario": budget_label,
        "target_alert_rate_pct": target_budget_pct,
        "threshold": threshold,
        "n_obs": n_obs,
        "n_pos": n_pos,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alerts_count": alerts,
        "realized_alert_rate_pct": round(alert_rate, 2),
        "recall": round(rec, 4),
        "recall_ci_95": rec_ci,
        "precision": round(prec, 4),
        "precision_ci_95": prec_ci,
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "fp_per_tp_ratio": fp_per_tp,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "brier_skill_score": round(bss, 4),
        "log_loss": round(ll, 4)
    }


def evaluate_alert_budgets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    budgets: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates model across standard alert budgets: 5%, 8%, 10%, 15%
    plus standard operating threshold tau=0.080 as anchor.
    """
    if budgets is None:
        budgets = [5.0, 8.0, 10.0, 15.0]

    budget_rows = []

    # 1. Standard Anchor at tau=0.080
    anchor_eval = evaluate_at_threshold(y_true, y_prob, threshold=0.080, budget_label="Fixed_Tau_0.080", target_budget_pct=None)
    anchor_eval["model_name"] = model_name
    budget_rows.append(anchor_eval)

    # 2. Operating Budget Scenarios
    for b in budgets:
        tau_b = find_threshold_for_alert_budget(y_prob, target_alert_rate=b)
        b_label = f"Budget_{int(b)}pct"
        eval_b = evaluate_at_threshold(y_true, y_prob, threshold=tau_b, budget_label=b_label, target_budget_pct=b)
        eval_b["model_name"] = model_name
        budget_rows.append(eval_b)

    return budget_rows
