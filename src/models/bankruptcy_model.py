import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
)
from src.preprocessing.preprocessors import BankruptcyPreprocessor
from src.models.calibration import ProbabilityCalibrator
from src.utils.logger import setup_logger

logger = setup_logger("bankruptcy_model")

class BankruptcyModel:
    """Manages training and evaluation of the cross-sectional Bankruptcy Model on Dataset B."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = BankruptcyPreprocessor()
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.best_model = None
        self.best_model_name = ""
        self.evaluation_results: Dict[str, any] = {}

    def train_and_evaluate(self, df_raw: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, any]:
        """
        Splits data with stratification to maintain the 3.2% class distribution across folds.
        Train (70%), Validation (10%), Holdout Test (20%).
        """
        # Split train+val vs test
        train_val_df, test_df = train_test_split(
            df_raw, test_size=test_size, stratify=df_raw["Bankrupt?"], random_state=self.random_state
        )
        # Split train vs val
        val_ratio = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, stratify=train_val_df["Bankrupt?"], random_state=self.random_state
        )
        
        logger.info(f"Dataset B split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        self.preprocessor.fit(train_df)
        X_train, y_train = self.preprocessor.transform(train_df)
        X_val, y_val = self.preprocessor.transform(val_df)
        X_test, y_test = self.preprocessor.transform(test_df)
        
        pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
        
        candidates = {
            "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.random_state),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=self.random_state, n_jobs=-1),
            "xgboost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=self.random_state, eval_metric="logloss", n_jobs=-1)
        }
        
        results = {}
        for name, clf in candidates.items():
            logger.info(f"Training Bankruptcy candidate: {name}...")
            clf.fit(X_train, y_train)
            
            val_probs = clf.predict_proba(X_val)[:, 1]
            val_preds = (val_probs >= 0.5).astype(int)
            val_metrics = self._compute_metrics(y_val, val_preds, val_probs)
            results[name] = {"model": clf, "val_metrics": val_metrics}
            logger.info(f"{name} Val ROC-AUC: {val_metrics['roc_auc']:.4f}, Recall: {val_metrics['recall']:.4f}")
            
        best_name = max(results.keys(), key=lambda k: results[k]["val_metrics"]["roc_auc"])
        self.best_model_name = best_name
        self.best_model = results[best_name]["model"]
        logger.info(f"Selected best Bankruptcy model: {best_name}")
        
        # Calibration on validation set
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
        
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
            "specificity": round(float(specificity), 4),
            "false_negative_rate": round(float(fn_rate), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }

    def predict_bankruptcy_probability(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, _ = self.preprocessor.transform(df)
        raw_prob = self.best_model.predict_proba(X)[:, 1]
        cal_prob = self.calibrator.predict_proba(X)[:, 1] if self.calibrator else raw_prob
        return raw_prob, cal_prob

    def save(self, model_dir: str = "models/bankruptcy"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(model_dir, "best_bankruptcy_model.joblib"))
        self.preprocessor.save(os.path.join(model_dir, "bankruptcy_preprocessor.joblib"))
        if self.calibrator:
            self.calibrator.save(os.path.join(model_dir, "bankruptcy_calibrator.joblib"))
        with open(os.path.join(model_dir, "bankruptcy_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, indent=2)
        logger.info(f"Bankruptcy model artifacts successfully saved to {model_dir}")

    @staticmethod
    def load(model_dir: str = "models/bankruptcy") -> 'BankruptcyModel':
        instance = BankruptcyModel()
        instance.best_model = joblib.load(os.path.join(model_dir, "best_bankruptcy_model.joblib"))
        instance.preprocessor = BankruptcyPreprocessor.load(os.path.join(model_dir, "bankruptcy_preprocessor.joblib"))
        if os.path.exists(os.path.join(model_dir, "bankruptcy_calibrator.joblib")):
            instance.calibrator = ProbabilityCalibrator.load(os.path.join(model_dir, "bankruptcy_calibrator.joblib"))
        if os.path.exists(os.path.join(model_dir, "bankruptcy_metrics.json")):
            with open(os.path.join(model_dir, "bankruptcy_metrics.json"), "r", encoding="utf-8") as f:
                instance.evaluation_results = json.load(f)
        return instance
