import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import xgboost as xgb
from src.features.feature_registry import FeatureRegistry
from src.utils.logger import setup_logger

logger = setup_logger("shap_explainer")

class ModelExplainer:
    """Provides local and global explainability via native TreeSHAP and delta decomposition."""
    
    def __init__(self, model, feature_names: List[str], registry: FeatureRegistry = None):
        self.model = model
        self.feature_names = feature_names
        self.registry = registry or FeatureRegistry()
        
        # Pre-cache registry metadata for all feature names to eliminate redundant lookups
        self._meta_cache = {}
        for name in self.feature_names:
            self._meta_cache[name] = self.registry.get(name)
            
        # Precompute importance weights for fallback
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            self._norm_importances = importances / (np.sum(importances) + 1e-8)
        elif hasattr(self.model, "coef_"):
            self._norm_importances = self.model.coef_[0]
        else:
            self._norm_importances = np.ones(len(self.feature_names)) / len(self.feature_names)

    def explain_instance_shap(self, feature_row: np.ndarray, top_n: int = 5) -> Dict[str, Any]:
        """Calculates local contributions and separates top risk drivers from protective factors."""
        if hasattr(self.model, "get_booster"):
            booster = self.model.get_booster()
            dmat = xgb.DMatrix(feature_row.reshape(1, -1), feature_names=self.feature_names)
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            feature_contribs = contribs[:-1]
            expected_val = float(contribs[-1])
        else:
            feature_contribs = feature_row * self._norm_importances
            expected_val = float(getattr(self.model, "intercept_", [0.0])[0])
            
        # Top positive risk drivers (pushes risk up)
        pos_indices = np.where(feature_contribs > 0)[0]
        neg_indices = np.where(feature_contribs < 0)[0]
        
        # Sort top indices
        top_pos = pos_indices[np.argsort(-feature_contribs[pos_indices])[:top_n]] if len(pos_indices) > 0 else []
        top_neg = neg_indices[np.argsort(feature_contribs[neg_indices])[:top_n]] if len(neg_indices) > 0 else []
        
        top_risk_drivers = []
        for idx in top_pos:
            fname = self.feature_names[idx]
            meta = self._meta_cache[fname]
            top_risk_drivers.append({
                "feature": fname,
                "category": meta["category"],
                "interpretation": meta["interpretation"],
                "shap_impact": round(float(feature_contribs[idx]), 4),
                "feature_value": round(float(feature_row[idx]), 4)
            })
            
        top_protective_factors = []
        for idx in top_neg:
            fname = self.feature_names[idx]
            meta = self._meta_cache[fname]
            top_protective_factors.append({
                "feature": fname,
                "category": meta["category"],
                "interpretation": meta["interpretation"],
                "shap_impact": round(float(feature_contribs[idx]), 4),
                "feature_value": round(float(feature_row[idx]), 4)
            })
            
        return {
            "expected_value": round(expected_val, 4),
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "feature_contribs_array": feature_contribs
        }

    def explain_risk_change(self, current_row: np.ndarray, prev_row: np.ndarray,
                           current_prob: float, prev_prob: float, 
                           current_contribs: np.ndarray = None, prev_contribs: np.ndarray = None,
                           top_n: int = 4) -> Dict[str, Any]:
        """Decomposes risk change between periods via direct vector subtraction."""
        delta_prob = current_prob - prev_prob
        delta_points = round(delta_prob * 100.0, 1)
        
        if current_contribs is None:
            current_contribs = current_row * self._norm_importances
        if prev_contribs is None:
            prev_contribs = prev_row * self._norm_importances
            
        d_contribs = current_contribs - prev_contribs
        d_values = current_row - prev_row
        
        # Find top deteriorating shifts (d_contribs > 0)
        pos_shift_idx = np.where(d_contribs > 0.001)[0]
        neg_shift_idx = np.where(d_contribs < -0.001)[0]
        
        top_pos = pos_shift_idx[np.argsort(-d_contribs[pos_shift_idx])[:top_n]] if len(pos_shift_idx) > 0 else []
        top_neg = neg_shift_idx[np.argsort(d_contribs[neg_shift_idx])[:top_n]] if len(neg_shift_idx) > 0 else []
        
        top_drivers_of_increase = []
        for idx in top_pos:
            fname = self.feature_names[idx]
            meta = self._meta_cache[fname]
            top_drivers_of_increase.append({
                "feature": fname,
                "category": meta["category"],
                "prev_value": round(float(prev_row[idx]), 3),
                "curr_value": round(float(current_row[idx]), 3),
                "value_change": round(float(d_values[idx]), 3),
                "shap_delta": round(float(d_contribs[idx]), 4),
                "interpretation": meta["interpretation"],
                "direction": "Worsened"
            })
            
        top_drivers_of_recovery = []
        for idx in top_neg:
            fname = self.feature_names[idx]
            meta = self._meta_cache[fname]
            top_drivers_of_recovery.append({
                "feature": fname,
                "category": meta["category"],
                "prev_value": round(float(prev_row[idx]), 3),
                "curr_value": round(float(current_row[idx]), 3),
                "value_change": round(float(d_values[idx]), 3),
                "shap_delta": round(float(d_contribs[idx]), 4),
                "interpretation": meta["interpretation"],
                "direction": "Improved"
            })
            
        summary_text = (
            f"Risk probability increased by {delta_points:+.1f} percentage points (from {prev_prob*100:.1f}% to {current_prob*100:.1f}%). "
            if delta_prob > 0 else
            f"Risk probability decreased by {abs(delta_points):.1f} percentage points (from {prev_prob*100:.1f}% to {current_prob*100:.1f}%). "
        )
        
        return {
            "risk_change_points": delta_points,
            "previous_risk_pct": round(prev_prob * 100.0, 1),
            "current_risk_pct": round(current_prob * 100.0, 1),
            "summary": summary_text,
            "top_drivers_of_increase": top_drivers_of_increase,
            "top_drivers_of_recovery": top_drivers_of_recovery
        }

    def get_global_importance(self, top_n: int = 25) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names))
            
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        
        df_imp["category"] = df_imp["feature"].apply(lambda f: self._meta_cache[f]["category"])
        df_imp["interpretation"] = df_imp["feature"].apply(lambda f: self._meta_cache[f]["interpretation"])
        return df_imp.head(top_n)
