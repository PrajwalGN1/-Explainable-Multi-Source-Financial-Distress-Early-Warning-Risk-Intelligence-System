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
        """Maps numerical risk score to configured qualitative risk category."""
        for key, conf in self.categories_cfg.items():
            if conf["min"] <= risk_score <= conf["max"]:
                return {
                    "category_id": key,
                    "label": conf["label"],
                    "color": conf["color"],
                    "description": conf["description"]
                }
        # Fallback boundary
        if risk_score > 80.0:
            return {"category_id": "critical", "label": "Critical", "color": "#ef4444", "description": "Severe distress indicators."}
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
