"""
Phase 5: Candidate Model Comparison & Alert-Budget Evaluation.

Evaluates research candidates against the approved baseline on common validation cohorts:
1. Approved_Baseline_RF_Ref: Approved Phase 2 production model (zero retraining)
2. Phase4_Hypothesis_RF_ROS15: Hypothesis from Phase 4 on baseline features
3. Phase5_RF_Deterioration_ROS10: RF on Phase 5 deterioration features with ROS (10%)
4. Phase5_RF_Deterioration_ROS15: RF on Phase 5 deterioration features with ROS (15%)
5. Phase5_RF_CostSensitive_w10: RF on Phase 5 deterioration features with 1:10 loss weight
6. Phase5_XGBoost_ScalePosWeight: XGBoost on Phase 5 deterioration features
7. Phase5_Logistic_Balanced: L2 Logistic Regression benchmark on Phase 5 deterioration features
8. Phase5_Interpretable_Rule_Benchmark: Deterministic multi-ratio deterioration rule

Dual Validation Protocol:
- Setting A: Temporal Validation (T in {11, 12}, N=517, N_pos=14)
- Setting B: 5-Fold GroupKFold on T <= 10 with company isolation (N=2,471, N_pos=95)
- All alert budgets evaluated: 5%, 8%, 10%, 15% and fixed tau=0.080
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    recall_score, precision_score
)

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

sys.path.insert(0, os.path.abspath("."))
from src.utils.logger import setup_logger
from src.preprocessing.preprocessors import EWSPreprocessor
from src.evaluation.phase5_feature_engineering import build_phase5_dataset
from src.evaluation.phase5_alert_budget import evaluate_alert_budgets, evaluate_at_threshold

logger = setup_logger("phase5_model_comparison")


def random_oversample_binary(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Random Over-Sampling exclusively to the positive class in training data."""
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    target_pos = int(np.ceil((target_ratio * n_neg) / (1.0 - target_ratio)))
    if target_pos <= n_pos:
        return X, y

    rng = np.random.RandomState(random_state)
    add_count = target_pos - n_pos
    extra_pos_idx = rng.choice(pos_idx, size=add_count, replace=True)

    combined_idx = np.concatenate([neg_idx, pos_idx, extra_pos_idx])
    rng.shuffle(combined_idx)
    return X[combined_idx], y[combined_idx]


class Phase5PreprocessingPipeline:
    """Inner-fold imputer and scaler for Phase 5 features with strict isolation."""

    def __init__(self, drop_cols: Optional[List[str]] = None):
        self.drop_cols = drop_cols or [
            "Company", "Time", "Financial Distress", "current_distress",
            "forward_target", "forward_target_2period"
        ]
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, df_train: pd.DataFrame):
        X = df_train.drop(columns=[c for c in self.drop_cols if c in df_train.columns], errors="ignore")
        self.feature_names = list(X.columns)
        X_imp = self.imputer.fit_transform(X)
        self.scaler.fit(X_imp)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted on training fold first.")
        X = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        y = df["forward_target"].to_numpy() if "forward_target" in df.columns else np.zeros(len(df))
        return X_scaled, y


class RuleBasedDeteriorationBenchmark:
    """Interpretable, deterministic multi-ratio financial deterioration scoring rule."""

    def __init__(self):
        self.fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes heuristic distress likelihood from normalized indicator spreads.
        Uses logistic sigmoid over unweighted deterioration indices.
        """
        # Sum of top features: X is standardized/robust-scaled.
        # Higher positive scores in distress indicators increase distress risk.
        z = np.mean(X[:, :10], axis=1)
        prob = 1.0 / (1.0 + np.exp(-z))
        # Ensure output shape matches (N, 2)
        return np.column_stack([1.0 - prob, prob])


def run_phase5_model_comparison() -> Dict[str, Any]:
    """Executes complete Phase 5 model comparison and alert-budget evaluation."""
    os.makedirs("results/phase5", exist_ok=True)
    os.makedirs("experiments/phase5/models", exist_ok=True)

    # 1. Load Phase 5 Dataset
    logger.info("Building Phase 5 feature-engineered dataset...")
    df_p5 = build_phase5_dataset()

    train_df = df_p5[df_p5["Time"] <= 10].copy().reset_index(drop=True)
    val_df = df_p5[df_p5["Time"].isin([11, 12])].copy().reset_index(drop=True)

    logger.info(f"Train (T<=10): {len(train_df)} rows, {int(train_df['forward_target'].sum())} distress cases.")
    logger.info(f"Val (T=11,12): {len(val_df)} rows, {int(val_df['forward_target'].sum())} distress cases.")

    # 2. Fit Phase 5 Preprocessor strictly on Train (T<=10)
    prep_p5 = Phase5PreprocessingPipeline()
    prep_p5.fit(train_df)
    X_train_p5, y_train_p5 = prep_p5.transform(train_df)
    X_val_p5, y_val_p5 = prep_p5.transform(val_df)
    logger.info(f"Phase 5 Preprocessor fitted with {len(prep_p5.feature_names)} features.")

    # 3. Fit Baseline Preprocessor on Train for baseline feature reference
    prep_base = EWSPreprocessor()
    prep_base.fit(train_df)
    X_train_base, y_train_base = prep_base.transform(train_df)
    X_val_base, y_val_base = prep_base.transform(val_df)

    # 4. Load Approved Baseline Model & Calibrator
    approved_rf = joblib.load("models/phase2/best_ews_model.joblib")
    approved_cal = joblib.load("models/phase2/ews_calibrator.joblib")

    # Predict Approved Baseline on Val (using its own preprocessor)
    approved_prep = EWSPreprocessor.load("models/phase2/ews_preprocessor.joblib")
    X_val_approved, y_val_approved = approved_prep.transform(val_df)
    approved_val_probs = approved_cal.predict_proba(X_val_approved)[:, 1]

    # 5. Define Candidate Models
    models_registry = {
        "Phase4_Hypothesis_RF_ROS15": {
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            "feature_set": "baseline",
            "ros_ratio": 0.15,
            "description": "Phase 4 Hypothesis: Random Forest on baseline features with 15% training ROS"
        },
        "Phase5_RF_Deterioration_ROS10": {
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            "feature_set": "phase5_deterioration",
            "ros_ratio": 0.10,
            "description": "Random Forest on Phase 5 deterioration features with 10% training ROS"
        },
        "Phase5_RF_Deterioration_ROS15": {
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42
            ),
            "feature_set": "phase5_deterioration",
            "ros_ratio": 0.15,
            "description": "Random Forest on Phase 5 deterioration features with 15% training ROS"
        },
        "Phase5_RF_CostSensitive_w10": {
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5,
                class_weight={0: 1.0, 1: 10.0}, random_state=42
            ),
            "feature_set": "phase5_deterioration",
            "ros_ratio": None,
            "description": "Random Forest on Phase 5 deterioration features with 1:10 minority class penalty"
        },
        "Phase5_Logistic_Balanced": {
            "estimator": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "feature_set": "phase5_deterioration",
            "ros_ratio": None,
            "description": "L2-penalized Logistic Regression benchmark on Phase 5 deterioration features"
        },
        "Phase5_Interpretable_Rule_Benchmark": {
            "estimator": RuleBasedDeteriorationBenchmark(),
            "feature_set": "phase5_deterioration",
            "ros_ratio": None,
            "description": "Transparent composite rule benchmark tracking multi-ratio deterioration"
        }
    }

    if XGB_AVAILABLE:
        models_registry["Phase5_XGBoost_ScalePosWeight"] = {
            "estimator": XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, scale_pos_weight=25.0,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
            ),
            "feature_set": "phase5_deterioration",
            "ros_ratio": None,
            "description": "Gradient Boosted Trees (XGBoost) with scale_pos_weight=25.0 on Phase 5 features"
        }

    all_budget_evaluations = []
    model_summary_records = []
    train_companies = train_df["Company"].to_numpy()

    # First, record Approved Baseline across alert budgets
    logger.info("Evaluating Approved Baseline Reference across alert budgets...")
    baseline_budgets = evaluate_alert_budgets(y_val_approved, approved_val_probs, model_name="Approved_Baseline_RF_Ref")
    all_budget_evaluations.extend(baseline_budgets)

    # 6. Train and Evaluate Each Candidate
    for name, cfg in models_registry.items():
        logger.info(f"--- Training & Evaluating Candidate: {name} ---")
        t0 = time.time()
        est = cfg["estimator"]
        use_ros = cfg["ros_ratio"] is not None

        # Select feature matrix
        if cfg["feature_set"] == "baseline":
            X_tr_raw, y_tr = X_train_base, y_train_base
            X_v = X_val_base
        else:
            X_tr_raw, y_tr = X_train_p5, y_train_p5
            X_v = X_val_p5

        # Inner-fold Training Resampling
        if use_ros:
            X_tr, y_tr_res = random_oversample_binary(
                X_tr_raw, y_tr, target_ratio=cfg["ros_ratio"], random_state=42
            )
        else:
            X_tr, y_tr_res = X_tr_raw, y_tr

        # Fit model
        est.fit(X_tr, y_tr_res)
        fit_duration = round(time.time() - t0, 3)

        # Leak-Free Inner Platt Calibration (3-fold CV on Train T<=10)
        if isinstance(est, RuleBasedDeteriorationBenchmark):
            cal_val_p = est.predict_proba(X_v)[:, 1]
        else:
            calibrator = CalibratedClassifierCV(estimator=clone(est), method="sigmoid", cv=3)
            calibrator.fit(X_tr, y_tr_res)
            cal_val_p = calibrator.predict_proba(X_v)[:, 1]

            # Save experimental model and calibrator
            joblib.dump(est, f"experiments/phase5/models/{name}.joblib")
            joblib.dump(calibrator, f"experiments/phase5/models/{name}_calibrator.joblib")

        # Evaluate across Alert Budgets
        budget_evals = evaluate_alert_budgets(y_val_p5, cal_val_p, model_name=name)
        all_budget_evaluations.extend(budget_evals)

        # 7. Cross-Sectional 5-Fold GroupKFold on T <= 10 (Company Isolation)
        gkf = GroupKFold(n_splits=5)
        fold_aucs, fold_aps, fold_briers = [], [], []

        if not isinstance(est, RuleBasedDeteriorationBenchmark):
            for f_idx, (f_tr_idx, f_te_idx) in enumerate(gkf.split(X_tr_raw, y_tr, groups=train_companies)):
                f_est = clone(est)
                X_f_tr, y_f_tr = X_tr_raw[f_tr_idx], y_tr[f_tr_idx]
                if use_ros:
                    X_f_tr, y_f_tr = random_oversample_binary(
                        X_f_tr, y_f_tr, target_ratio=cfg["ros_ratio"], random_state=42 + f_idx
                    )
                f_est.fit(X_f_tr, y_f_tr)
                f_preds = f_est.predict_proba(X_tr_raw[f_te_idx])[:, 1]
                if len(np.unique(y_tr[f_te_idx])) > 1:
                    fold_aucs.append(roc_auc_score(y_tr[f_te_idx], f_preds))
                    fold_aps.append(average_precision_score(y_tr[f_te_idx], f_preds))
                fold_briers.append(brier_score_loss(y_tr[f_te_idx], f_preds))

        group_auc_mean = round(float(np.mean(fold_aucs)), 4) if fold_aucs else 0.5
        group_auc_std = round(float(np.std(fold_aucs)), 4) if fold_aucs else 0.0
        group_ap_mean = round(float(np.mean(fold_aps)), 4) if fold_aps else 0.0
        group_ap_std = round(float(np.std(fold_aps)), 4) if fold_aps else 0.0

        model_summary_records.append({
            "model_name": name,
            "description": cfg["description"],
            "training_duration_sec": fit_duration,
            "group_cv_roc_auc_mean": group_auc_mean,
            "group_cv_roc_auc_std": group_auc_std,
            "group_cv_pr_auc_mean": group_ap_mean,
            "group_cv_pr_auc_std": group_ap_std
        })

    # Save Results
    df_budgets = pd.DataFrame(all_budget_evaluations)
    df_budgets.to_csv("results/phase5/alert_budget_comparison_validation.csv", index=False)
    logger.info("Saved alert budget comparison to results/phase5/alert_budget_comparison_validation.csv")

    with open("results/phase5/alert_budget_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_budget_evaluations, f, indent=2)

    df_models = pd.DataFrame(model_summary_records)
    df_models.to_csv("results/phase5/model_cv_summary.csv", index=False)
    logger.info("Saved model CV summary to results/phase5/model_cv_summary.csv")

    return {
        "budget_evaluations": df_budgets,
        "model_summaries": df_models
    }


if __name__ == "__main__":
    run_phase5_model_comparison()
