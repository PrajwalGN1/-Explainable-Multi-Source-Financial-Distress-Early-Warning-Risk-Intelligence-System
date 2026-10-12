"""
Temporal Risk Attribution and Period-over-Period Explanation Service.

Decomposes shifts in model risk between reporting periods (t and t-1) for a specific company:
- Distinguishes 3 distinct phenomena:
    1. Prediction change (Delta p_cal and Delta p_RF)
    2. Input feature value change (Delta x_scaled and Delta x_raw)
    3. Attribution shift (Delta phi_j in TreeSHAP space)
- Explicit non-causal warnings and data availability validations
- Strict guardrails against cross-model or incompatible comparison
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.explainability.tree_shap_explainer import NativeTreeSHAPExplainer, get_tree_shap_explainer
from src.features.temporal_features import TemporalFeatureEngine
from src.utils.logger import setup_logger

logger = setup_logger("temporal_attribution")


class TemporalAttributionService:
    """
    Decomposes temporal risk changes between consecutive annual reporting periods.
    """

    def __init__(self, explainer: Optional[NativeTreeSHAPExplainer] = None):
        self.explainer = explainer or get_tree_shap_explainer()

    def explain_period_transition(
        self,
        prev_transformed_row: np.ndarray,
        curr_transformed_row: np.ndarray,
        prev_period: int,
        curr_period: int,
        company_id: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Decomposes risk change between period t-1 and period t using identical model and explainer.
        """
        if not self.explainer.is_available:
            return {
                "status": "unavailable",
                "warning": "TreeSHAP explainer is unavailable.",
                "company_id": str(company_id),
                "current_period": curr_period,
                "previous_period": prev_period
            }

        # Local explanations for both periods
        exp_prev = self.explainer.explain_transformed_row(prev_transformed_row, top_n=len(self.explainer.feature_names))
        exp_curr = self.explainer.explain_transformed_row(curr_transformed_row, top_n=len(self.explainer.feature_names))

        p_cal_prev = exp_prev["calibrated_distress_probability"]
        p_cal_curr = exp_curr["calibrated_distress_probability"]
        delta_p_cal = p_cal_curr - p_cal_prev
        delta_p_cal_points = round(delta_p_cal * 100.0, 2)

        p_rf_prev = exp_prev["raw_random_forest_probability"]
        p_rf_curr = exp_curr["raw_random_forest_probability"]
        delta_p_rf = p_rf_curr - p_rf_prev

        # Determine risk categories based on canonical thresholds
        cat_prev = self._derive_risk_category(p_cal_prev)
        cat_curr = self._derive_risk_category(p_cal_curr)

        # Feature level shifts: Delta phi and Delta x
        feat_names = self.explainer.feature_names
        prev_x = prev_transformed_row.flatten()
        curr_x = curr_transformed_row.flatten()

        shaps_prev = np.array([exp_prev["feature_attributions_all"][fn] for fn in feat_names])
        shaps_curr = np.array([exp_curr["feature_attributions_all"][fn] for fn in feat_names])

        delta_shaps = shaps_curr - shaps_prev
        delta_x = curr_x - prev_x

        # Top drivers of increased risk (delta_shaps > 0)
        pos_shift_indices = np.where(delta_shaps > 0.0001)[0]
        sorted_pos = pos_shift_indices[np.argsort(-delta_shaps[pos_shift_indices])] if len(pos_shift_indices) > 0 else []

        # Top drivers of recovery / reduced risk (delta_shaps < 0)
        neg_shift_indices = np.where(delta_shaps < -0.0001)[0]
        sorted_neg = neg_shift_indices[np.argsort(delta_shaps[neg_shift_indices])] if len(neg_shift_indices) > 0 else []

        top_drivers_of_increase = []
        for idx in sorted_pos[:top_n]:
            fn = feat_names[idx]
            meta = self.explainer._meta_cache[fn]
            top_drivers_of_increase.append({
                "feature": fn,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "prev_feature_value_scaled": round(float(prev_x[idx]), 4),
                "curr_feature_value_scaled": round(float(curr_x[idx]), 4),
                "feature_value_change": round(float(delta_x[idx]), 4),
                "prev_shap_impact": round(float(shaps_prev[idx]), 4),
                "curr_shap_impact": round(float(shaps_curr[idx]), 4),
                "shap_delta": round(float(delta_shaps[idx]), 4),
                "shift_direction": "Increased Risk Attribution"
            })

        top_drivers_of_recovery = []
        for idx in sorted_neg[:top_n]:
            fn = feat_names[idx]
            meta = self.explainer._meta_cache[fn]
            top_drivers_of_recovery.append({
                "feature": fn,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "prev_feature_value_scaled": round(float(prev_x[idx]), 4),
                "curr_feature_value_scaled": round(float(curr_x[idx]), 4),
                "feature_value_change": round(float(delta_x[idx]), 4),
                "prev_shap_impact": round(float(shaps_prev[idx]), 4),
                "curr_shap_impact": round(float(shaps_curr[idx]), 4),
                "shap_delta": round(float(delta_shaps[idx]), 4),
                "shift_direction": "Reduced Risk Attribution (Recovery)"
            })

        # Summary narrative
        if delta_p_cal > 0.005:
            summary = (
                f"Calibrated distress probability increased by {delta_p_cal_points:+.2f} percentage points "
                f"(from {p_cal_prev*100:.2f}% in Period {prev_period} to {p_cal_curr*100:.2f}% in Period {curr_period}). "
                f"Risk category transitioned from {cat_prev} to {cat_curr}."
            )
        elif delta_p_cal < -0.005:
            summary = (
                f"Calibrated distress probability decreased by {abs(delta_p_cal_points):.2f} percentage points "
                f"(from {p_cal_prev*100:.2f}% in Period {prev_period} to {p_cal_curr*100:.2f}% in Period {curr_period}). "
                f"Risk category transitioned from {cat_prev} to {cat_curr}."
            )
        else:
            summary = (
                f"Calibrated distress probability remained essentially stable ({delta_p_cal_points:+.2f} percentage points, "
                f"Period {prev_period}: {p_cal_prev*100:.2f}%, Period {curr_period}: {p_cal_curr*100:.2f}%)."
            )

        return {
            "status": "success",
            "company_id": str(company_id),
            "previous_period": prev_period,
            "current_period": curr_period,
            "model_version": self.explainer.model_version,
            "explanation_method": self.explainer.explanation_method,
            "output_space": self.explainer.output_space,
            "previous_calibrated_prob": round(p_cal_prev, 4),
            "current_calibrated_prob": round(p_cal_curr, 4),
            "calibrated_prob_delta": round(delta_p_cal, 4),
            "calibrated_prob_delta_points": delta_p_cal_points,
            "previous_raw_rf_prob": round(p_rf_prev, 4),
            "current_raw_rf_prob": round(p_rf_curr, 4),
            "raw_rf_prob_delta": round(delta_p_rf, 4),
            "previous_risk_category": cat_prev,
            "current_risk_category": cat_curr,
            "summary": summary,
            "top_drivers_of_increase": top_drivers_of_increase,
            "top_drivers_of_recovery": top_drivers_of_recovery,
            "disclaimer": (
                "Temporal attribution decomposes shifts in model output between periods under identical model assumptions. "
                "A change in SHAP contribution indicates that the model responded to feature shifts, but does not prove real-world causality."
            )
        }

    def _derive_risk_category(self, cal_prob: float) -> str:
        """Derives canonical risk category from calibrated probability."""
        score = cal_prob * 100.0
        if score < 8.0:
            return "Healthy"
        elif score < 13.0:
            return "Watch"
        elif score < 17.0:
            return "High"
        else:
            return "Critical"
