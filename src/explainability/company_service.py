"""
Company Explanation Service for FastAPI and Dashboard Integration.

Manages on-demand and cached local/temporal/global explanations:
- Pre-caches company features for all 422 SMEs in Dataset A
- Produces live native TreeSHAP explanations in <70ms
- Produces period-over-period temporal attribution deltas
- Serves cached global population importance summaries
"""

import os
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.loaders import DataLoader
from src.features.temporal_features import TemporalFeatureEngine
from src.explainability.tree_shap_explainer import NativeTreeSHAPExplainer, get_tree_shap_explainer
from src.explainability.temporal_attribution import TemporalAttributionService
from src.fusion.risk_fusion import RiskFusionEngine
from src.utils.logger import setup_logger

logger = setup_logger("company_explanation_service")


class CompanyExplanationService:
    """
    High-performance, thread-safe service providing local, temporal, and global
    TreeSHAP risk intelligence for companies.
    """

    def __init__(
        self,
        raw_data_path: str = "data/raw/Financial Distress.csv",
        explainer: Optional[NativeTreeSHAPExplainer] = None
    ):
        self.raw_data_path = raw_data_path
        self.explainer = explainer or get_tree_shap_explainer()
        self.temporal_service = TemporalAttributionService(explainer=self.explainer)
        self._fusion_engine = RiskFusionEngine()

        self._lock = threading.Lock()
        self._company_features_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self._company_raw_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._company_periods_cache: Dict[str, List[int]] = {}
        self._is_initialized = False

    def initialize_cache(self) -> None:
        """Pre-processes all 422 companies in memory for sub-millisecond lookups."""
        if self._is_initialized:
            return

        with self._lock:
            if self._is_initialized:
                return

            if not os.path.exists(self.raw_data_path):
                logger.warning(f"Raw dataset not found at {self.raw_data_path}; company cache not initialized.")
                return

            logger.info("Initializing in-memory company temporal features cache...")
            loader = DataLoader()
            df_raw = loader.load_dataset_a()

            engine = TemporalFeatureEngine()
            # Engineer temporal features for all companies
            df_feats = engine.generate_temporal_features(df_raw, id_col="Company", time_col="Time")

            # Transform through preprocessor
            X_all, _ = self.explainer._preprocessor.transform(df_feats)

            for idx, row in df_feats.iterrows():
                cid = str(int(row["Company"]))
                period = int(row["Time"])

                if cid not in self._company_features_cache:
                    self._company_features_cache[cid] = {}
                    self._company_raw_cache[cid] = {}
                    self._company_periods_cache[cid] = []

                self._company_features_cache[cid][period] = X_all[idx]
                self._company_raw_cache[cid][period] = {
                    "Company": cid,
                    "Time": period,
                    "Financial Distress": row.get("Financial Distress", None)
                }
                self._company_periods_cache[cid].append(period)

            # Sort periods
            for cid in self._company_periods_cache:
                self._company_periods_cache[cid].sort()

            self._is_initialized = True
            logger.info(f"Initialized company cache for {len(self._company_features_cache)} companies.")

    def get_local_explanation(
        self,
        company_id: str,
        period: Optional[int] = None,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieves or computes a local TreeSHAP explanation for a specific company and period.
        """
        if not self._is_initialized:
            self.initialize_cache()

        cid = str(company_id)
        if cid not in self._company_features_cache:
            raise KeyError(f"Company ID {company_id} not found in database.")

        available_periods = self._company_periods_cache[cid]
        if not available_periods:
            raise ValueError(f"No reporting periods available for Company {company_id}.")

        selected_period = period if period is not None else available_periods[-1]
        if selected_period not in self._company_features_cache[cid]:
            raise KeyError(
                f"Period {selected_period} not found for Company {company_id}. "
                f"Available periods: {available_periods}"
            )

        row_transformed = self._company_features_cache[cid][selected_period]
        explanation = self.explainer.explain_transformed_row(row_transformed, top_n=top_n)

        # Enrich with company metadata
        explanation["company_id"] = cid
        explanation["reporting_period"] = selected_period
        explanation["available_periods"] = available_periods

        # Operational risk score & category (canonical RiskFusionEngine)
        cal_prob = explanation["calibrated_distress_probability"]
        score = self._fusion_engine.compute_risk_score(cal_prob)
        cat_info = self._fusion_engine.determine_category(score)
        explanation["risk_score"] = score
        explanation["risk_category"] = cat_info["label"]
        explanation["risk_color"] = cat_info["color"]
        explanation["risk_description"] = cat_info["description"]

        return explanation

    def get_temporal_explanation(
        self,
        company_id: str,
        curr_period: Optional[int] = None,
        prev_period: Optional[int] = None,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Decomposes risk change between two consecutive periods for a company.
        """
        if not self._is_initialized:
            self.initialize_cache()

        cid = str(company_id)
        if cid not in self._company_features_cache:
            raise KeyError(f"Company ID {company_id} not found in database.")

        available_periods = self._company_periods_cache[cid]
        if len(available_periods) < 2:
            return {
                "status": "insufficient_history",
                "company_id": cid,
                "message": f"Company {cid} has only {len(available_periods)} period(s). At least 2 periods are required for temporal attribution.",
                "available_periods": available_periods
            }

        # Resolve periods
        if curr_period is None:
            c_p = available_periods[-1]
            p_p = available_periods[-2]
        else:
            c_p = curr_period
            if prev_period is not None:
                p_p = prev_period
            else:
                curr_idx = available_periods.index(c_p) if c_p in available_periods else -1
                if curr_idx <= 0:
                    raise ValueError(f"Cannot determine prior period for Period {c_p}. Available: {available_periods}")
                p_p = available_periods[curr_idx - 1]

        if c_p not in self._company_features_cache[cid]:
            raise KeyError(f"Current period {c_p} not found for Company {cid}.")
        if p_p not in self._company_features_cache[cid]:
            raise KeyError(f"Previous period {p_p} not found for Company {cid}.")

        row_prev = self._company_features_cache[cid][p_p]
        row_curr = self._company_features_cache[cid][c_p]

        res = self.temporal_service.explain_period_transition(
            prev_transformed_row=row_prev,
            curr_transformed_row=row_curr,
            prev_period=p_p,
            curr_period=c_p,
            company_id=cid,
            top_n=top_n
        )
        res["available_periods"] = available_periods
        return res

    def explain_custom_payload(
        self,
        raw_features: Dict[str, float],
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Explains an arbitrary raw input dictionary through preprocessor and TreeSHAP.
        """
        df_input = pd.DataFrame([raw_features])
        X_trans, _ = self.explainer._preprocessor.transform(df_input)
        exp = self.explainer.explain_transformed_row(X_trans[0], top_n=top_n)

        cal_prob = exp["calibrated_distress_probability"]
        score = self._fusion_engine.compute_risk_score(cal_prob)
        cat_info = self._fusion_engine.determine_category(score)
        exp["risk_score"] = score
        exp["risk_category"] = cat_info["label"]
        exp["risk_color"] = cat_info["color"]
        exp["risk_description"] = cat_info["description"]
        return exp

    def get_global_explanation(self, population: str = "test") -> Dict[str, Any]:
        """
        Returns precomputed global feature importance from disk.
        """
        summary_path = "reports/phase3/global_explanations/global_summary.json"
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Global explanations not found at {summary_path}. Run generate_phase3_global_explanations.py first.")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        csv_file = (
            "reports/phase3/global_explanations/global_importance_test.csv"
            if population.lower() == "test" else
            "reports/phase3/global_explanations/global_importance_val.csv"
        )
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            summary["selected_population_rankings"] = df.to_dict(orient="records")
        return summary

    def get_explanation_info(self) -> Dict[str, Any]:
        """
        Returns system explanation capabilities, model version, and output space.
        """
        # Test additivity on zero row
        dummy = np.zeros(len(self.explainer.feature_names))
        valid, recon, actual, disc = self.explainer.validate_additivity(dummy)

        return {
            "status": "ready" if self.explainer.is_available else "unavailable",
            "model_version": self.explainer.model_version,
            "explanation_method": self.explainer.explanation_method,
            "output_space": self.explainer.output_space,
            "feature_count": len(self.explainer.feature_names),
            "operating_threshold": self.explainer.operating_threshold,
            "expected_value_base": round(self.explainer.expected_value, 4),
            "additivity_verified": valid,
            "additivity_tolerance": 1e-4,
            "additivity_sample_discrepancy": disc,
            "disclaimer": (
                "Native TreeSHAP explains the Random Forest output space directly. "
                "Downstream Platt calibration is non-linear and preserves rank monotonicity, "
                "but calibrated probabilities are not linearly additive with raw TreeSHAP values. "
                "All features named x1 through x83 are masked financial indicators."
            )
        }


# Global singleton instance
_COMPANY_SERVICE_INSTANCE: Optional[CompanyExplanationService] = None
_COMPANY_SERVICE_LOCK = threading.Lock()


def get_company_explanation_service() -> CompanyExplanationService:
    global _COMPANY_SERVICE_INSTANCE
    if _COMPANY_SERVICE_INSTANCE is not None:
        return _COMPANY_SERVICE_INSTANCE
    with _COMPANY_SERVICE_LOCK:
        if _COMPANY_SERVICE_INSTANCE is None:
            _COMPANY_SERVICE_INSTANCE = CompanyExplanationService()
        return _COMPANY_SERVICE_INSTANCE
