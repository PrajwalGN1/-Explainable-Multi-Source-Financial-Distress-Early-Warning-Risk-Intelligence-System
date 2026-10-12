import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger("meta_model")

class StackingMetaModel:
    """Stacking meta-learner trained strictly on out-of-fold predictions to prevent leakage."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.meta_clf = LogisticRegression(penalty="l2", C=1.0, random_state=random_state)
        self.feature_names = ["ews_prob", "bankruptcy_prob", "anomaly_score", "financial_health_risk"]
        self.fitted = False

    def fit(self, oof_features: np.ndarray, y_true: np.ndarray) -> 'StackingMetaModel':
        """Fits meta-model on out-of-fold validation predictions."""
        logger.info(f"Fitting Stacking Meta-Model on {len(y_true)} out-of-fold samples...")
        self.meta_clf.fit(oof_features, y_true)
        self.fitted = True
        
        coefs = dict(zip(self.feature_names, [float(c) for c in self.meta_clf.coef_[0]]))
        intercept = float(self.meta_clf.intercept_[0])
        logger.info(f"Meta-Model fitted. Coefficients: {coefs}, Intercept: {intercept:.4f}")
        return self

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Meta-model not fitted.")
        return self.meta_clf.predict_proba(meta_features)[:, 1]

    def get_coefficients(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        return dict(zip(self.feature_names, [round(float(c), 4) for c in self.meta_clf.coef_[0]]))

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'StackingMetaModel':
        return joblib.load(filepath)
