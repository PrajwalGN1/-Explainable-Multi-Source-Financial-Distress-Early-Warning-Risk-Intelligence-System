import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.models.calibration import ProbabilityCalibrator
from src.utils.logger import setup_logger

logger = setup_logger("ews_model")

class TemporalEWSModel:
    """Manages training, time-aware validation, and evaluation for the Temporal EWS Model."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = EWSPreprocessor(target_col="forward_target")
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.best_model = None
        self.best_model_name = ""
        self.evaluation_results: Dict[str, any] = {}

    def prepare_data(self, df_raw: pd.DataFrame, train_max_time: int = 10, val_max_time: int = 12) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineers temporal features and splits strictly by Time periods to prevent temporal leakage.
        Train: Time <= 10
        Validation: 11 <= Time <= 12
        Holdout Test: 13 <= Time <= 14
        """
        engine = TemporalFeatureEngine()
        df_target = engine.construct_forward_target(df_raw, id_col="Company", time_col="Time", raw_target_col="Financial Distress", threshold=-0.5)
        df_feats = engine.generate_temporal_features(df_target, id_col="Company", time_col="Time")
        
        # Drop rows where forward target is NaN (terminal period or no next period)
        df_valid = df_feats.dropna(subset=["forward_target"]).copy()
        df_valid["forward_target"] = df_valid["forward_target"].astype(int)
        
        train_df = df_valid[df_valid["Time"] <= train_max_time].copy()
        val_df = df_valid[(df_valid["Time"] > train_max_time) & (df_valid["Time"] <= val_max_time)].copy()
        test_df = df_valid[df_valid["Time"] > val_max_time].copy()
        
        logger.info(f"Time-aware split: Train={len(train_df)} (Time 1-{train_max_time}), Val={len(val_df)} (Time {train_max_time+1}-{val_max_time}), Test={len(test_df)} (Time {val_max_time+1}+)")
        logger.info(f"Distress rates: Train={train_df['forward_target'].mean()*100:.2f}%, Val={val_df['forward_target'].mean()*100:.2f}%, Test={test_df['forward_target'].mean()*100:.2f}%")
        return train_df, val_df, test_df

    def train_and_compare(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, any]:
        """Trains Logistic Regression, Random Forest, and XGBoost; compares on validation set."""
        self.preprocessor.fit(train_df)
        X_train, y_train = self.preprocessor.transform(train_df)
        X_val, y_val = self.preprocessor.transform(val_df)
        X_test, y_test = self.preprocessor.transform(test_df)
        
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
        
        candidates = {
            "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_state),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=self.random_state, n_jobs=-1),
            "xgboost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=self.random_state, eval_metric="logloss", n_jobs=-1)
        }
        
        results = {}
        for name, clf in candidates.items():
            logger.info(f"Training EWS candidate: {name}...")
            clf.fit(X_train, y_train)
            
            # Evaluate on validation
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            
            val_metrics = self._compute_metrics(y_val, val_preds, val_probs)
            results[name] = {"model": clf, "val_metrics": val_metrics}
            logger.info(f"{name} Val ROC-AUC: {val_metrics['roc_auc']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            
        # Select best model (prioritizing PR-AUC and ROC-AUC for early warning)
        best_name = max(results.keys(), key=lambda k: results[k]["val_metrics"]["roc_auc"] + results[k]["val_metrics"]["recall"])
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]
        logger.info(f"Selected best EWS model: {best_name}")
        
        # Fit calibrator on validation fold
        self.calibrator = ProbabilityCalibrator(method="sigmoid")
        self.calibrator.fit(self.best_model, X_val, y_val)
        
        # Test evaluation
        test_raw_probs = self.best_model.predict_proba(X_test)[:, 1]
        test_cal_probs = self.calibrator.predict_proba(X_test)[:, 1]
        test_preds = (test_cal_probs >= 0.5).astype(int)
        test_metrics = self._compute_metrics(y_test, test_preds, test_cal_probs)
        cal_eval = self.calibrator.evaluate_calibration(y_test, test_raw_probs, test_cal_probs)
        
        self.evaluation_results = {
            "best_model_name": best_name,
            "candidate_comparisons": {k: v["val_metrics"] for k, v in results.items()},
            "test_metrics": test_metrics,
            "calibration_evaluation": cal_eval,
            "feature_count": len(self.preprocessor.feature_names)
        }
        return self.evaluation_results

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "specificity": round(float(specificity), 4),
            "false_negative_rate": round(float(fn_rate), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }

    def predict_risk(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (raw_probabilities, calibrated_probabilities)."""
        X, _ = self.preprocessor.transform(df)
        raw_prob = self.best_model.predict_proba(X)[:, 1]
        cal_prob = self.calibrator.predict_proba(X)[:, 1] if self.calibrator else raw_prob
        return raw_prob, cal_prob

    def save(self, model_dir: str = "models/ews"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(model_dir, "best_ews_model.joblib"))
        self.preprocessor.save(os.path.join(model_dir, "ews_preprocessor.joblib"))
        if self.calibrator:
            self.calibrator.save(os.path.join(model_dir, "ews_calibrator.joblib"))
        with open(os.path.join(model_dir, "ews_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2)
        logger.info(f"EWS model artifacts successfully saved to {model_dir}")

    @staticmethod
    def load(model_dir: str = "models/ews") -> 'TemporalEWSModel':
        instance = TemporalEWSModel()
        instance.best_model = joblib.load(os.path.join(model_dir, "best_ews_model.joblib"))
        instance.preprocessor = EWSPreprocessor.load(os.path.join(model_dir, "ews_preprocessor.joblib"))
        if os.path.exists(os.path.join(model_dir, "ews_calibrator.joblib")):
            instance.calibrator = ProbabilityCalibrator.load(os.path.join(model_dir, "ews_calibrator.joblib"))
        if os.path.exists(os.path.join(model_dir, "ews_metrics.json")):
            with open(os.path.join(model_dir, "ews_metrics.json"), "r", encoding="utf-8") as f:
                instance.evaluation_results = json.load(f)
        return instance
