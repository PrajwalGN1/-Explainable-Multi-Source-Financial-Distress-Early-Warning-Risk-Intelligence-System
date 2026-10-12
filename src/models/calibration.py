import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger("calibration")

class ProbabilityCalibrator:
    """Evaluates and fits probability calibration via Platt scaling (sigmoid) or Isotonic regression."""
    
    def __init__(self, method: str = "sigmoid"):
        self.method = method # 'sigmoid' or 'isotonic'
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit(self, base_estimator, X_val: np.ndarray, y_val: np.ndarray) -> 'ProbabilityCalibrator':
        """Calibrates base estimator using validation set (avoids target leakage)."""
        logger.info(f"Fitting {self.method} probability calibration on {len(y_val)} validation samples...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.method,
            cv="prefit"
        )
        self.calibrated_model.fit(X_val, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated_model is None:
            raise RuntimeError("Calibrator not fitted yet.")
        return self.calibrated_model.predict_proba(X)

    @staticmethod
    def evaluate_calibration(y_true: np.ndarray, raw_probs: np.ndarray, cal_probs: np.ndarray, n_bins: int = 10) -> Dict[str, any]:
        """Calculates Brier score and reliability curve coordinates."""
        brier_raw = float(brier_score_loss(y_true, raw_probs))
        brier_cal = float(brier_score_loss(y_true, cal_probs))
        
        prob_true_raw, prob_pred_raw = calibration_curve(y_true, raw_probs, n_bins=n_bins, strategy="uniform")
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, cal_probs, n_bins=n_bins, strategy="uniform")
        
        return {
            "brier_score_raw": round(brier_raw, 4),
            "brier_score_calibrated": round(brier_cal, 4),
            "brier_improvement_pct": round(((brier_raw - brier_cal) / (brier_raw + 1e-8)) * 100, 2),
            "raw_curve": {
                "prob_true": [float(x) for x in prob_true_raw],
                "prob_pred": [float(x) for x in prob_pred_raw]
            },
            "calibrated_curve": {
                "prob_true": [float(x) for x in prob_true_cal],
                "prob_pred": [float(x) for x in prob_pred_cal]
            }
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'ProbabilityCalibrator':
        return joblib.load(filepath)
