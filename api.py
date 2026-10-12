import os
import json
import secrets
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, Header
from pydantic import BaseModel, Field

from src.fusion.risk_fusion import RiskFusionEngine
from src.features.financial_features import FinancialHealthEngine
from src.models.ews_model import TemporalEWSModel
from src.models.bankruptcy_model import BankruptcyModel
from src.models.anomaly_model import AnomalyOODModel
from src.config.settings import get_risk_config
from src.utils.logger import setup_logger

logger = setup_logger("api")

import threading

# Load cached intelligence DB if available
PROCESSED_DB_PATH = "data/processed/companies_risk_intelligence.json"
WATCHLIST_PATH = "data/processed/watchlist.csv"

# Thread-safe in-memory singleton cache
_DB_CACHE: Optional[Dict[str, Any]] = None
_DB_CACHE_LOCK = threading.Lock()
_DB_LOAD_COUNT: int = 0

def get_db(force_reload: bool = False, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Thread-safe in-memory singleton database loader.
    Loads the 27 MB JSON intelligence database once per process under normal operation.
    Subsequent requests reuse the loaded representation with sub-millisecond response time.
    Supports explicit invalidation/reload via force_reload=True or clear_db_cache().
    """
    global _DB_CACHE, _DB_LOAD_COUNT
    path = db_path or PROCESSED_DB_PATH

    # Fast path: cache hit without lock overhead (standard path only)
    if _DB_CACHE is not None and not force_reload and db_path is None:
        return _DB_CACHE

    with _DB_CACHE_LOCK:
        # Double-check inside lock
        if _DB_CACHE is not None and not force_reload and db_path is None:
            return _DB_CACHE

        if not os.path.exists(path):
            logger.warning(f"Intelligence database not found at {path}. Returning empty database representation.")
            empty_db = {"companies": {}, "metadata": {}}
            if db_path is None:
                _DB_CACHE = empty_db
            return empty_db

        try:
            logger.info(f"Loading intelligence database into memory from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict) or "companies" not in loaded:
                logger.error(f"Malformed database file at {path}: expected dict containing 'companies' key.")
                empty_db = {"companies": {}, "metadata": {}}
                if db_path is None:
                    _DB_CACHE = empty_db
                return empty_db

            _DB_LOAD_COUNT += 1
            if db_path is None:
                _DB_CACHE = loaded
            logger.info(f"Intelligence database loaded successfully ({len(loaded.get('companies', {}))} companies, load_count={_DB_LOAD_COUNT}).")
            return loaded
        except json.JSONDecodeError as jde:
            logger.error(f"JSON decode error reading {path}: {jde}")
            empty_db = {"companies": {}, "metadata": {}}
            if db_path is None:
                _DB_CACHE = empty_db
            return empty_db
        except Exception as exc:
            logger.error(f"Unexpected error reading database from {path}: {exc}")
            raise

def clear_db_cache() -> None:
    """Explicitly invalidates the in-memory database cache."""
    global _DB_CACHE
    with _DB_CACHE_LOCK:
        _DB_CACHE = None

def get_db_load_count() -> int:
    """Returns number of times the database was read from disk into memory."""
    return _DB_LOAD_COUNT

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Pre-warms the in-memory database cache during application startup."""
    try:
        get_db()
    except Exception as e:
        logger.warning(f"Could not preload intelligence DB on startup: {e}")
    yield

app = FastAPI(
    title="Financial Distress Risk Intelligence API",
    description="Multi-Source Early Warning & Risk Analytics Engine",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic Schemas
class RiskPredictionRequest(BaseModel):
    company_id: Optional[str] = Field("TEST_FIRM", description="Unique corporate identifier")
    ews_probability: float = Field(..., ge=0.0, le=1.0, description="Forward distress probability from Temporal EWS")
    bankruptcy_probability: float = Field(0.05, ge=0.0, le=1.0, description="Bankruptcy probability from static ratios")
    anomaly_score: float = Field(0.10, ge=0.0, le=1.0, description="Out-of-distribution anomaly score")
    financial_health_score: float = Field(75.0, ge=0.0, le=100.0, description="Composite financial health index")

class RiskPredictionResponse(BaseModel):
    company_id: str
    final_risk_probability: float
    final_risk_probability_pct: float
    risk_score: float
    risk_category: str
    risk_color: str
    risk_description: str
    evidence_breakdown: Dict[str, float]
    recommendations: List[str]

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Financial Distress Early-Warning Risk Intelligence API",
        "version": "1.0.0"
    }

@app.get("/model-info", tags=["Models"])
def model_info():
    """Returns technical metadata, architectures, and evaluation metrics for all 3 data sources."""
    ews_metrics = {}
    bankruptcy_metrics = {}
    if os.path.exists("models/ews/ews_metrics.json"):
        with open("models/ews/ews_metrics.json", "r") as f:
            ews_metrics = json.load(f)
    if os.path.exists("models/bankruptcy/bankruptcy_metrics.json"):
        with open("models/bankruptcy/bankruptcy_metrics.json", "r") as f:
            bankruptcy_metrics = json.load(f)
            
    phase2_metrics = {}
    if os.path.exists("models/phase2/ews_metrics.json"):
        with open("models/phase2/ews_metrics.json", "r") as f:
            phase2_metrics = json.load(f)
            
    manifest = {}
    if os.path.exists("models/phase2/model_manifest.json"):
        with open("models/phase2/model_manifest.json", "r") as f:
            manifest = json.load(f)
            
    info = {
        "temporal_ews_model": {
            "dataset": "Dataset A (Financial Distress panel, Tehran Stock Exchange)",
            "prediction_horizon": "t+1 Forward Distress",
            "validation_strategy": "Time-Aware Temporal Split (Periods 1-10 Train, 11-12 Val, 13-14 Test)",
            "metrics": ews_metrics.get("test_metrics", {})
        },
        "bankruptcy_model": {
            "dataset": "Dataset B (Taiwan Economic Journal cross-sectional)",
            "target": "Bankrupt? (Binary)",
            "validation_strategy": "Stratified Holdout Split",
            "metrics": bankruptcy_metrics.get("test_metrics", {})
        },
        "external_anomaly_layer": {
            "dataset": "Dataset C (Polish Companies Bankruptcy test set)",
            "algorithm": "Isolation Forest (Unsupervised)",
            "purpose": "Out-of-Distribution Drift & Anomaly Quantification"
        },
        "fusion_engine": {
            "architecture": "Two-Level Fusion (Weighted Linear Baseline + Stacking Meta-Learner)",
            "leakage_protection": "Trained exclusively on out-of-fold validation splits"
        }
    }
    if phase2_metrics:
        info["phase2_validated_model"] = phase2_metrics
    if manifest:
        info["model_manifest"] = {
            "model_version": manifest.get("model_version", "1.2.1-hardened"),
            "operating_threshold": manifest.get("operating_threshold", 0.08),
            "experiment_identifier": manifest.get("experiment_identifier", "EXP-PHASE2-REPRO-001"),
            "dataset_sha256": manifest.get("dataset_sha256", ""),
            "status": manifest.get("status", "Hardened and Reproducibility Verified")
        }
    return info



@app.post("/predict", response_model=RiskPredictionResponse, tags=["Scoring"])
def predict_risk(payload: RiskPredictionRequest):
    """Calculates calibrated final risk probability and composite risk score."""
    try:
        fusion_engine = RiskFusionEngine()
        meta_path = "models/fusion/meta_model.joblib"
        if os.path.exists(meta_path):
            from src.fusion.meta_model import StackingMetaModel
            fusion_engine.meta_model = StackingMetaModel.load(meta_path)
            
        assessment = fusion_engine.assess_company(
            ews_prob=payload.ews_probability,
            bankruptcy_prob=payload.bankruptcy_probability,
            anomaly_score=payload.anomaly_score,
            financial_health_score=payload.financial_health_score,
            use_meta_model=True
        )
        
        from src.risk_engine.recommendations import RecommendationEngine
        rec_engine = RecommendationEngine()
        recs = rec_engine.generate_recommendations(assessment["risk_category"], [])
        
        return RiskPredictionResponse(
            company_id=payload.company_id,
            final_risk_probability=assessment["final_risk_probability"],
            final_risk_probability_pct=assessment["final_risk_probability_pct"],
            risk_score=assessment["risk_score"],
            risk_category=assessment["risk_category"],
            risk_color=assessment["risk_color"],
            risk_description=assessment["risk_description"],
            evidence_breakdown=assessment["evidence_breakdown"],
            recommendations=recs["recommended_actions"]
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/company/{company_id}", tags=["Company Intelligence"])
def get_company_profile(company_id: str):
    """Returns longitudinal risk profile, trajectory, early warnings, and SHAP drivers for a specific firm."""
    db = get_db()
    companies = db.get("companies", {})
    if company_id not in companies:
        raise HTTPException(status_code=404, detail=f"Company ID {company_id} not found in database.")
    return companies[company_id]

@app.get("/watchlist", tags=["Portfolio Surveillance"])
def get_watchlist(limit: int = Query(50, ge=1, le=500)):
    """Returns prioritized corporate distress surveillance watchlist."""
    if os.path.exists(WATCHLIST_PATH):
        df = pd.read_csv(WATCHLIST_PATH)
        df = df.replace({np.nan: None})
        return df.head(limit).to_dict(orient="records")
    db = get_db()
    return db.get("watchlist", [])[:limit]

@app.get("/risk-summary", tags=["Portfolio Surveillance"])
def get_risk_summary():
    """Returns macro distribution and key surveillance KPIs across monitored corporate universe."""
    db = get_db()
    return db.get("metadata", {
        "total_companies": 0,
        "high_risk_count": 0,
        "critical_count": 0,
        "mean_risk_score": 0.0
    })

@app.post("/admin/reload-db", tags=["System"])
def reload_database(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """
    Explicitly invalidates and reloads the cached in-memory intelligence database.
    Requires a valid 'X-Admin-Key' request header matching server 'ADMIN_API_KEY'.
    Fails closed (HTTP 503) if ADMIN_API_KEY is not configured in the server environment.
    Rejects missing header with HTTP 401 and invalid key with HTTP 403.
    """
    expected_key = os.getenv("ADMIN_API_KEY")
    if not expected_key:
        logger.error("Admin reload rejected: ADMIN_API_KEY environment variable is not configured on the server.")
        raise HTTPException(
            status_code=503,
            detail="Administrative reload operation is disabled: server ADMIN_API_KEY is not configured."
        )

    if not x_admin_key:
        logger.warning("Admin reload rejected: missing X-Admin-Key header.")
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required: missing X-Admin-Key header."
        )

    if not secrets.compare_digest(x_admin_key, expected_key):
        logger.warning("Admin reload rejected: invalid admin credentials provided.")
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid admin credentials."
        )

    db = get_db(force_reload=True)
    count = len(db.get("companies", {}))
    return {
        "status": "reloaded",
        "companies_loaded": count,
        "load_count": get_db_load_count()
    }


# ---------------------------------------------------------
# Phase 3: Explainability & Risk Intelligence Endpoints
# ---------------------------------------------------------

@app.get("/explanations/info", tags=["Explainability"])
def get_explanation_info():
    """Returns TreeSHAP capability status, model version, output space, and additivity verification."""
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()
    return srv.get_explanation_info()


@app.get("/explanations/local/{company_id}", tags=["Explainability"])
def get_local_explanation(
    company_id: str,
    period: Optional[int] = Query(None, ge=1, le=14, description="Reporting period (defaults to company's latest)"),
    top_n: int = Query(10, ge=1, le=50, description="Top positive and negative risk drivers to return")
):
    """
    Returns model-faithful local TreeSHAP attribution for a specific SME firm.
    Explains the Random Forest probability output space with exact additivity.
    """
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()
    try:
        return srv.get_local_explanation(company_id=company_id, period=period, top_n=top_n)
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke).strip("'"))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Local explanation failed for {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")


@app.post("/explanations/local", tags=["Explainability"])
def explain_custom_payload(
    payload: Dict[str, float],
    top_n: int = Query(10, ge=1, le=50, description="Top positive and negative drivers to return")
):
    """
    Produces calibrated prediction and TreeSHAP attribution for an arbitrary feature dictionary.
    """
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()
    try:
        return srv.explain_custom_payload(raw_features=payload, top_n=top_n)
    except Exception as e:
        logger.error(f"Custom payload explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to explain custom payload: {str(e)}")


@app.get("/explanations/temporal/{company_id}", tags=["Explainability"])
def get_temporal_explanation(
    company_id: str,
    curr_period: Optional[int] = Query(None, ge=1, le=14, description="Current evaluation period"),
    prev_period: Optional[int] = Query(None, ge=1, le=14, description="Prior comparison period"),
    top_n: int = Query(5, ge=1, le=20, description="Top feature shifts to return")
):
    """
    Decomposes risk change between consecutive reporting periods for an SME.
    Distinguishes prediction change, feature value change, and attribution shift.
    """
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()
    try:
        res = srv.get_temporal_explanation(
            company_id=company_id,
            curr_period=curr_period,
            prev_period=prev_period,
            top_n=top_n
        )
        if res.get("status") == "insufficient_history":
            raise HTTPException(status_code=400, detail=res["message"])
        return res
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke).strip("'"))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Temporal explanation failed for {company_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate temporal explanation: {str(e)}")


@app.get("/explanations/global", tags=["Explainability"])
def get_global_explanation(
    population: str = Query("test", description="Reference population ('test' or 'val')")
):
    """
    Returns global feature attribution summaries and mean absolute SHAP rankings.
    """
    from src.explainability.company_service import get_company_explanation_service
    srv = get_company_explanation_service()
    try:
        return srv.get_global_explanation(population=population)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        logger.error(f"Global explanation retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve global explanation: {str(e)}")


