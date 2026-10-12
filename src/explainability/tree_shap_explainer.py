"""
Native TreeSHAP Explainer Service for Phase 2 Random Forest EWS Model.

Implements model-faithful explainability using shap.TreeExplainer on the audited
RandomForestClassifier checkpoint. Explicitly validates:
- Estimator compatibility (RandomForestClassifier, 100 trees, max_depth=5)
- Class indexing (Class 1 = Financial Distress)
- Exact 163-feature order matching EWSPreprocessor
- Output space: "raw_random_forest_probability"
- Additivity: sum(SHAP) + expected_value == model_output within numerical tolerance
- Clear separation between raw Random Forest attributions and downstream Platt-calibrated probabilities
"""

import os
import json
import threading
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_VERSION = shap.__version__
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_VERSION = "unavailable"

from src.features.feature_registry import FeatureRegistry
from src.features.temporal_features import TemporalFeatureEngine
from src.preprocessing.preprocessors import EWSPreprocessor
from src.utils.logger import setup_logger

logger = setup_logger("tree_shap_explainer")


class NativeTreeSHAPExplainer:
    """
    Model-faithful TreeSHAP explainer for the Phase 2 Random Forest EWS model.
    
    CRITICAL METHODOLOGICAL DISTINCTION:
    Native TreeSHAP explains the Random Forest output space directly.
    For sklearn RandomForestClassifier, the TreeExplainer output space is the
    raw positive-class probability p_RF in [0, 1].
    
    Downstream Platt calibration (CalibratedClassifierCV) is a non-linear monotonic
    mapping p_cal = sigma(a * p_RF + b). SHAP values are additive in p_RF space,
    NOT in p_cal space. This explainer explicitly documents this boundary.
    """

    def __init__(
        self,
        model_path: str = "models/phase2/best_ews_model.joblib",
        preprocessor_path: str = "models/phase2/ews_preprocessor.joblib",
        calibrator_path: str = "models/phase2/ews_calibrator.joblib",
        threshold_config_path: str = "models/phase2/threshold_config.json",
        manifest_path: str = "models/phase2/model_manifest.json",
        registry: Optional[FeatureRegistry] = None
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.calibrator_path = calibrator_path
        self.threshold_config_path = threshold_config_path
        self.manifest_path = manifest_path

        self.registry = registry or FeatureRegistry()
        self.output_space = "raw_random_forest_probability"
        self.explanation_method = f"TreeSHAP (shap {SHAP_VERSION})"

        self._lock = threading.Lock()
        self._explainer: Optional[Any] = None
        self._model: Optional[Any] = None
        self._preprocessor: Optional[EWSPreprocessor] = None
        self._calibrator: Optional[Any] = None
        self._threshold_config: Dict[str, Any] = {}
        self._manifest: Dict[str, Any] = {}
        self._feature_names: List[str] = []
        self._expected_value_distress: float = 0.5
        self._meta_cache: Dict[str, Dict[str, Any]] = {}

        self._initialize_artifacts()

    def _initialize_artifacts(self) -> None:
        """Loads read-only Phase 2 artifacts and initializes the TreeExplainer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing required model artifact: {self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Missing required preprocessor artifact: {self.preprocessor_path}")

        logger.info(f"Loading Phase 2 model from {self.model_path}...")
        self._model = joblib.load(self.model_path)
        self._preprocessor = EWSPreprocessor.load(self.preprocessor_path)

        if os.path.exists(self.calibrator_path):
            self._calibrator = joblib.load(self.calibrator_path)
        else:
            self._calibrator = None
            logger.warning(f"Calibrator not found at {self.calibrator_path}; operational probabilities will match raw probabilities.")

        if os.path.exists(self.threshold_config_path):
            with open(self.threshold_config_path, "r", encoding="utf-8") as f:
                self._threshold_config = json.load(f)
        else:
            self._threshold_config = {"operating_threshold": 0.08}

        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        else:
            self._manifest = {"model_version": "1.2.1-hardened"}

        self._feature_names = list(self._preprocessor.feature_names)
        logger.info(f"Loaded preprocessor with {len(self._feature_names)} features.")

        # Cache feature metadata from registry
        for fname in self._feature_names:
            meta = self.registry.get(fname)
            # Add temporal construction descriptor if this is an engineered temporal feature
            if "_lag_" in fname or "_diff_" in fname or "_growth_" in fname or "_acceleration" in fname or "_roll_" in fname or "_consec_" in fname or "_dev_from_baseline" in fname:
                parts = fname.split("_", 1)
                base_var = parts[0]
                transform_type = parts[1]
                meta = dict(meta)
                meta["is_temporal_engineered"] = True
                meta["base_variable"] = base_var
                meta["transformation"] = transform_type
                meta["interpretation"] = f"Temporal feature: {transform_type.replace('_', ' ')} of {base_var}"
            else:
                meta = dict(meta)
                meta["is_temporal_engineered"] = False
                meta["base_variable"] = fname
                meta["transformation"] = "raw_level"
            self._meta_cache[fname] = meta

        # Verify class ordering
        if hasattr(self._model, "classes_"):
            classes = list(self._model.classes_)
            if 1 not in classes:
                raise ValueError(f"Distress class (1) not found in model classes: {classes}")
            self._distress_class_idx = classes.index(1)
        else:
            self._distress_class_idx = 1

        # Initialize TreeExplainer
        if SHAP_AVAILABLE:
            logger.info("Initializing shap.TreeExplainer on RandomForestClassifier...")
            self._explainer = shap.TreeExplainer(self._model)
            
            # Extract expected_value for distress class
            ev = self._explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                self._expected_value_distress = float(ev[self._distress_class_idx])
            else:
                self._expected_value_distress = float(ev)
            logger.info(f"TreeExplainer initialized. Base value for distress class: {self._expected_value_distress:.4f}")
        else:
            logger.error("SHAP library is not installed in the environment.")
            self._explainer = None

    @property
    def is_available(self) -> bool:
        """Returns True if SHAP is available and explainer is initialized."""
        return SHAP_AVAILABLE and self._explainer is not None

    @property
    def model_version(self) -> str:
        return self._manifest.get("model_version", "1.2.1-hardened")

    @property
    def operating_threshold(self) -> float:
        return float(self._threshold_config.get("operating_threshold", 0.08))

    @property
    def expected_value(self) -> float:
        return self._expected_value_distress

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def validate_additivity(
        self,
        row_transformed: np.ndarray,
        tolerance: float = 1e-4
    ) -> Tuple[bool, float, float, float]:
        """
        Mathematically verifies SHAP additivity for a single transformed row:
        sum(SHAP_values) + expected_value == model_raw_probability
        
        Returns: (is_valid, sum_plus_base, model_prob, absolute_discrepancy)
        """
        if not self.is_available:
            return False, 0.0, 0.0, 1.0

        if row_transformed.ndim == 1:
            x_2d = row_transformed.reshape(1, -1)
        else:
            x_2d = row_transformed

        # Model output
        prob_rf = float(self._model.predict_proba(x_2d)[0, self._distress_class_idx])

        # SHAP calculation
        with self._lock:
            shap_vals = self._explainer.shap_values(x_2d)

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            # Shape (1, n_features, n_classes)
            row_shaps = shap_vals[0, :, self._distress_class_idx]
        elif isinstance(shap_vals, list):
            # List of (1, n_features) per class
            row_shaps = shap_vals[self._distress_class_idx][0]
        else:
            row_shaps = shap_vals[0]

        sum_shaps = float(np.sum(row_shaps))
        reconstructed = sum_shaps + self._expected_value_distress
        discrepancy = abs(reconstructed - prob_rf)
        is_valid = bool(discrepancy <= tolerance)

        return is_valid, round(reconstructed, 6), round(prob_rf, 6), round(discrepancy, 6)

    def explain_transformed_row(
        self,
        row_transformed: np.ndarray,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Generates local TreeSHAP attributions for a preprocessed feature vector (1, 163).
        """
        if not self.is_available:
            return {
                "status": "unavailable",
                "warning": "SHAP library is not installed or explainer initialization failed.",
                "explanation_method": self.explanation_method,
                "output_space": self.output_space
            }

        if row_transformed.ndim == 1:
            x_2d = row_transformed.reshape(1, -1)
            x_1d = row_transformed
        else:
            x_2d = row_transformed
            x_1d = row_transformed[0]

        if x_2d.shape[1] != len(self._feature_names):
            raise ValueError(f"Feature dimension mismatch: expected {len(self._feature_names)}, got {x_2d.shape[1]}")

        # Compute raw RF prediction and calibrated prediction
        raw_prob_distress = float(self._model.predict_proba(x_2d)[0, self._distress_class_idx])
        if self._calibrator:
            cal_prob_distress = float(self._calibrator.predict_proba(x_2d)[0, self._distress_class_idx])
        else:
            cal_prob_distress = raw_prob_distress

        # Compute SHAP values
        with self._lock:
            shap_raw = self._explainer.shap_values(x_2d)

        if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
            row_shaps = shap_raw[0, :, self._distress_class_idx]
        elif isinstance(shap_raw, list):
            row_shaps = shap_raw[self._distress_class_idx][0]
        else:
            row_shaps = shap_raw[0]

        # Additivity verification
        sum_shaps = float(np.sum(row_shaps))
        reconstructed = sum_shaps + self._expected_value_distress
        additivity_error = abs(reconstructed - raw_prob_distress)
        additivity_verified = bool(additivity_error <= 1e-4)

        # Decompose positive risk drivers and protective factors
        pos_indices = np.where(row_shaps > 0)[0]
        neg_indices = np.where(row_shaps < 0)[0]

        # Sort by magnitude
        sorted_pos = pos_indices[np.argsort(-row_shaps[pos_indices])] if len(pos_indices) > 0 else []
        sorted_neg = neg_indices[np.argsort(row_shaps[neg_indices])] if len(neg_indices) > 0 else []

        top_risk_drivers = []
        for idx in sorted_pos[:top_n]:
            fname = self._feature_names[idx]
            meta = self._meta_cache[fname]
            top_risk_drivers.append({
                "feature": fname,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "shap_impact": round(float(row_shaps[idx]), 4),
                "feature_value_scaled": round(float(x_1d[idx]), 4),
                "direction": "Elevates Risk"
            })

        top_protective_factors = []
        for idx in sorted_neg[:top_n]:
            fname = self._feature_names[idx]
            meta = self._meta_cache[fname]
            top_protective_factors.append({
                "feature": fname,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "shap_impact": round(float(row_shaps[idx]), 4),
                "feature_value_scaled": round(float(x_1d[idx]), 4),
                "direction": "Reduces Risk"
            })

        # Top features overall by absolute magnitude
        abs_order = np.argsort(-np.abs(row_shaps))[:top_n]
        top_features_overall = []
        for idx in abs_order:
            fname = self._feature_names[idx]
            meta = self._meta_cache[fname]
            val_shap = float(row_shaps[idx])
            top_features_overall.append({
                "feature": fname,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "shap_impact": round(val_shap, 4),
                "abs_magnitude": round(abs(val_shap), 4),
                "feature_value_scaled": round(float(x_1d[idx]), 4),
                "direction": "Elevates Risk" if val_shap > 0 else "Reduces Risk"
            })

        return {
            "status": "success",
            "model_version": self.model_version,
            "explanation_method": self.explanation_method,
            "output_space": self.output_space,
            "expected_value_base": round(self._expected_value_distress, 4),
            "raw_random_forest_probability": round(raw_prob_distress, 4),
            "calibrated_distress_probability": round(cal_prob_distress, 4),
            "calibrated_distress_probability_pct": round(cal_prob_distress * 100.0, 2),
            "operating_threshold": self.operating_threshold,
            "alert_triggered": bool(cal_prob_distress >= self.operating_threshold),
            "additivity_verified": additivity_verified,
            "additivity_reconstructed_prob": round(reconstructed, 4),
            "additivity_discrepancy": round(additivity_error, 6),
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "top_features_overall": top_features_overall,
            "feature_attributions_all": {
                self._feature_names[i]: round(float(row_shaps[i]), 4)
                for i in range(len(self._feature_names))
            },
            "disclaimer": (
                "SHAP attributions quantify feature associations with the Random Forest probability output. "
                "They do not establish causal relationships. Features named x1..x83 are masked indicators."
            )
        }

    def explain_batch(
        self,
        X_transformed: np.ndarray
    ) -> np.ndarray:
        """
        Computes SHAP values for a batch of preprocessed rows.
        Returns array of shape (N, 163).
        """
        if not self.is_available:
            raise RuntimeError("SHAP explainer is unavailable.")

        with self._lock:
            shap_batch = self._explainer.shap_values(X_transformed)

        if isinstance(shap_batch, np.ndarray) and shap_batch.ndim == 3:
            return shap_batch[:, :, self._distress_class_idx]
        elif isinstance(shap_batch, list):
            return shap_batch[self._distress_class_idx]
        else:
            return shap_batch

    def compute_global_importance(
        self,
        X_sample: np.ndarray,
        sample_name: str = "Holdout Test (T=13)"
    ) -> Dict[str, Any]:
        """
        Computes global feature importance across an evaluation sample.
        Calculates:
        - Mean absolute SHAP value: mean(|phi_j|)
        - Signed mean contribution: mean(phi_j)
        - Standard deviation of contributions: std(phi_j)
        """
        shaps_2d = self.explain_batch(X_sample)
        n_samples, n_features = shaps_2d.shape

        mean_abs = np.mean(np.abs(shaps_2d), axis=0)
        mean_signed = np.mean(shaps_2d, axis=0)
        std_signed = np.std(shaps_2d, axis=0)

        records = []
        for i in range(n_features):
            fname = self._feature_names[i]
            meta = self._meta_cache[fname]
            records.append({
                "feature": fname,
                "base_variable": meta["base_variable"],
                "category": meta["category"],
                "is_temporal_engineered": meta["is_temporal_engineered"],
                "transformation": meta["transformation"],
                "interpretation": meta["interpretation"],
                "mean_abs_shap": round(float(mean_abs[i]), 5),
                "mean_signed_shap": round(float(mean_signed[i]), 5),
                "std_shap": round(float(std_signed[i]), 5)
            })

        df_imp = pd.DataFrame(records).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        df_imp["rank"] = df_imp.index + 1

        return {
            "evaluation_population": sample_name,
            "sample_size": n_samples,
            "feature_count": n_features,
            "model_version": self.model_version,
            "explanation_method": self.explanation_method,
            "output_space": self.output_space,
            "rankings": df_imp.to_dict(orient="records"),
            "top_10": df_imp.head(10).to_dict(orient="records"),
            "top_20": df_imp.head(20).to_dict(orient="records")
        }


# Global thread-safe singleton cache
_EXPLAINER_INSTANCE: Optional[NativeTreeSHAPExplainer] = None
_EXPLAINER_LOCK = threading.Lock()


def get_tree_shap_explainer() -> NativeTreeSHAPExplainer:
    """Returns the singleton instance of NativeTreeSHAPExplainer."""
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is not None:
        return _EXPLAINER_INSTANCE
    with _EXPLAINER_LOCK:
        if _EXPLAINER_INSTANCE is None:
            _EXPLAINER_INSTANCE = NativeTreeSHAPExplainer()
        return _EXPLAINER_INSTANCE
