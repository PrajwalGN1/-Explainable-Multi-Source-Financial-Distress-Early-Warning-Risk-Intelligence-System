import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.fusion.meta_model import StackingMetaModel
from src.models.calibration import ProbabilityCalibrator
from src.config.settings import get_risk_config
from src.utils.logger import setup_logger

logger = setup_logger("risk_fusion")

class RiskFusionEngine:
    """Combines temporal EWS, bankruptcy, anomaly, and financial health evidence into calibrated final risk."""
    
    def __init__(self, risk_config: Dict[str, Any] = None):
        self.risk_config = risk_config or get_risk_config()
        self.categories_cfg = self.risk_config.get("risk_categories", {})
        self.baseline_weights = {
            "ews": 0.50,
            "bankruptcy": 0.25,
            "anomaly": 0.15,
            "health_risk": 0.10
        }
        self.meta_model: Optional[StackingMetaModel] = None
        self.calibrator: Optional[ProbabilityCalibrator] = None

    def fuse_scores_baseline(self, ews_prob: float, bankruptcy_prob: float, 
                             anomaly_score: float, health_risk: float) -> float:
        """Level 1 transparent linear weighted fusion."""
        w = self.baseline_weights
        fused = (
            w["ews"] * ews_prob +
            w["bankruptcy"] * bankruptcy_prob +
            w["anomaly"] * anomaly_score +
            w["health_risk"] * health_risk
        )
        return float(np.clip(fused, 0.0, 1.0))

    def fuse(self, ews_prob: float, bankruptcy_prob: float, 
             anomaly_score: float, health_risk: float, use_meta_model: bool = True) -> float:
        """Calculates final risk probability using meta-model (if available) or baseline weighted fusion."""
        if use_meta_model and self.meta_model is not None and self.meta_model.fitted:
            features = np.array([[ews_prob, bankruptcy_prob, anomaly_score, health_risk]])
            prob = float(self.meta_model.predict_proba(features)[0])
        else:
            prob = self.fuse_scores_baseline(ews_prob, bankruptcy_prob, anomaly_score, health_risk)
            
        return float(np.clip(prob, 0.0, 1.0))

    def compute_risk_score(self, final_probability: float) -> float:
        """Converts probability to standardized 0-100 Risk Score."""
        return round(float(np.clip(final_probability * 100.0, 0.0, 100.0)), 1)

    def determine_category(self, risk_score: float) -> Dict[str, str]:
        """
        Maps numerical risk score (0-100) to configured qualitative risk category.
        Enforces canonical, contiguous boundary intervals:
          - Healthy:  [0.0, 4.0]
          - Low Risk: (4.0, 8.0]
          - Watch:    (8.0, 13.0]
          - High Risk:(13.0, 17.0]
          - Critical: (17.0, 100.0]
        Guarantees zero gaps, zero overlaps, and robust invalid/out-of-bounds handling.
        """
        if risk_score is None or (isinstance(risk_score, (int, float)) and np.isnan(risk_score)):
            logger.warning(f"Invalid risk score provided: {risk_score}. Defaulting to Watch category.")
            return {
                "category_id": "watch",
                "label": "Watch",
                "color": "#f59e0b",
                "description": "Indeterminate risk score; flagged for manual review."
            }

        try:
            score = float(np.clip(float(risk_score), 0.0, 100.0))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert risk score {risk_score} to float. Defaulting to Watch.")
            return {
                "category_id": "watch",
                "label": "Watch",
                "color": "#f59e0b",
                "description": "Indeterminate risk score; flagged for manual review."
            }

        # Check configured categories in sequential order
        order = ["healthy", "low", "watch", "high", "critical"]
        for idx, key in enumerate(order):
            if key in self.categories_cfg:
                conf = self.categories_cfg[key]
                c_max = float(conf.get("max", 100.0))
                if score <= c_max or idx == len(order) - 1:
                    return {
                        "category_id": key,
                        "label": conf["label"],
                        "color": conf["color"],
                        "description": conf["description"]
                    }

        # Fallback if categories_cfg is missing keys
        if score > 17.0:
            return {"category_id": "critical", "label": "Critical", "color": "#ef4444", "description": "Severe distress indicators."}
        elif score > 13.0:
            return {"category_id": "high", "label": "High Risk", "color": "#f97316", "description": "Substantial deterioration."}
        elif score > 8.0:
            return {"category_id": "watch", "label": "Watch", "color": "#f59e0b", "description": "Early warning indicators."}
        elif score > 4.0:
            return {"category_id": "low", "label": "Low Risk", "color": "#3b82f6", "description": "Low deterioration risk."}
        return {"category_id": "healthy", "label": "Healthy", "color": "#10b981", "description": "Strong financial position."}

    def assess_company(self, ews_prob: float, bankruptcy_prob: float, anomaly_score: float, 
                       financial_health_score: float, use_meta_model: bool = True) -> Dict[str, Any]:
        """Produces complete multi-evidence risk intelligence object."""
        health_risk = max(0.0, min(1.0, 1.0 - (financial_health_score / 100.0)))
        final_prob = self.fuse(ews_prob, bankruptcy_prob, anomaly_score, health_risk, use_meta_model=use_meta_model)
        score = self.compute_risk_score(final_prob)
        cat_info = self.determine_category(score)
        
        return {
            "final_risk_probability": round(final_prob, 4),
            "final_risk_probability_pct": round(final_prob * 100.0, 1),
            "risk_score": score,
            "risk_category": cat_info["label"],
            "risk_color": cat_info["color"],
            "risk_description": cat_info["description"],
            "evidence_breakdown": {
                "ews_probability": round(ews_prob, 4),
                "bankruptcy_probability": round(bankruptcy_prob, 4),
                "anomaly_score": round(anomaly_score, 4),
                "financial_health_score": round(financial_health_score, 1),
                "financial_health_risk": round(health_risk, 4)
            }
        }

    def save(self, fusion_dir: str = "models/fusion"):
        os.makedirs(fusion_dir, exist_ok=True)
        if self.meta_model:
            self.meta_model.save(os.path.join(fusion_dir, "meta_model.joblib"))
        meta = {
            "baseline_weights": self.baseline_weights,
            "has_meta_model": self.meta_model is not None and self.meta_model.fitted
        }
        with open(os.path.join(fusion_dir, "fusion_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Risk fusion artifacts saved to {fusion_dir}")

    @staticmethod
    def load(fusion_dir: str = "models/fusion") -> 'RiskFusionEngine':
        instance = RiskFusionEngine()
        meta_path = os.path.join(fusion_dir, "meta_model.joblib")
        if os.path.exists(meta_path):
            instance.meta_model = StackingMetaModel.load(meta_path)
        return instance
