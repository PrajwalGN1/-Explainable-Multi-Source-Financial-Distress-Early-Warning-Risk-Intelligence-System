import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.ensemble import IsolationForest
from scipy.stats import ks_2samp
from src.utils.logger import setup_logger

logger = setup_logger("anomaly_model")

class AnomalyOODModel:
    """Manages Isolation Forest anomaly detection and OOD drift quantification for Dataset C."""
    
    def __init__(self, contamination: float = 0.05, n_estimators: int = 150, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.iso_forest: Optional[IsolationForest] = None
        self.feature_names: List[str] = []
        self.score_min = 0.0
        self.score_max = 1.0

    def fit(self, df: pd.DataFrame, drop_cols: List[str] = None) -> 'AnomalyOODModel':
        """Fits Isolation Forest on numeric features."""
        drop_cols = drop_cols or ["ID"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        self.feature_names = list(X.columns)
        
        logger.info(f"Fitting Isolation Forest on {len(df)} samples with {len(self.feature_names)} features...")
        self.iso_forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iso_forest.fit(X)
        
        # Determine empirical decision function range for normalization
        raw_scores = -self.iso_forest.decision_function(X) # higher raw score = more anomalous
        self.score_min = float(np.percentile(raw_scores, 1))
        self.score_max = float(np.percentile(raw_scores, 99))
        logger.info(f"Anomaly score calibration bounds: min={self.score_min:.4f}, max={self.score_max:.4f}")
        return self

    def predict_anomaly_score(self, df: pd.DataFrame, drop_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes anomaly score scaled to [0, 1] where:
        0.0 = completely normal financial profile
        1.0 = extreme financial outlier / anomaly
        Returns (normalized_score, binary_outlier_flag)
        """
        if self.iso_forest is None:
            raise RuntimeError("Anomaly model must be fitted first.")
            
        drop_cols = drop_cols or ["ID"]
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        
        raw_scores = -self.iso_forest.decision_function(X)
        # Min-max scaling with clipping to ensure strictly [0, 1]
        norm_scores = np.clip((raw_scores - self.score_min) / (self.score_max - self.score_min + 1e-8), 0.0, 1.0)
        outlier_flags = (self.iso_forest.predict(X) == -1).astype(int)
        
        return norm_scores, outlier_flags

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between two feature distributions."""
        eps = 1e-4
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)
        
        expected_pct = expected_counts / (len(expected) + eps)
        actual_pct = actual_counts / (len(actual) + eps)
        
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def evaluate_distribution_drift(self, df_c: pd.DataFrame, df_sample_other: pd.DataFrame = None) -> Dict[str, any]:
        """Calculates KS-test and PSI metrics quantifying Out-of-Distribution behavior."""
        norm_scores, outliers = self.predict_anomaly_score(df_c)
        outlier_rate = float(np.mean(outliers))
        
        drift_report = {
            "dataset_c_total_rows": len(df_c),
            "detected_anomalies_count": int(np.sum(outliers)),
            "anomaly_rate_pct": round(outlier_rate * 100, 2),
            "mean_anomaly_score": round(float(np.mean(norm_scores)), 4),
            "median_anomaly_score": round(float(np.median(norm_scores)), 4),
            "p95_anomaly_score": round(float(np.percentile(norm_scores, 95)), 4),
            "drift_status": "Out-of-Distribution External Dataset"
        }
        return drift_report

    def save(self, model_dir: str = "models/anomaly"):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.iso_forest, os.path.join(model_dir, "isolation_forest.joblib"))
        meta = {
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "score_min": self.score_min,
            "score_max": self.score_max
        }
        with open(os.path.join(model_dir, "anomaly_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Anomaly model saved to {model_dir}")

    @staticmethod
    def load(model_dir: str = "models/anomaly") -> 'AnomalyOODModel':
        instance = AnomalyOODModel()
        instance.iso_forest = joblib.load(os.path.join(model_dir, "isolation_forest.joblib"))
        with open(os.path.join(model_dir, "anomaly_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            instance.feature_names = meta["feature_names"]
            instance.contamination = meta["contamination"]
            instance.score_min = meta["score_min"]
            instance.score_max = meta["score_max"]
        return instance
