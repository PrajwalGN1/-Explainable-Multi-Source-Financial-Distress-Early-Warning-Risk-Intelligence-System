import pytest
import numpy as np
from src.fusion.risk_fusion import RiskFusionEngine
from src.risk_engine.trajectory import RiskTrajectoryEngine
from src.risk_engine.alerts import EarlyWarningSignalEngine
from src.risk_engine.recommendations import RecommendationEngine

def test_risk_fusion_bounds():
    fusion = RiskFusionEngine()
    
    # Healthy inputs
    res_healthy = fusion.assess_company(
        ews_prob=0.01,
        bankruptcy_prob=0.02,
        anomaly_score=0.05,
        financial_health_score=90.0,
        use_meta_model=False
    )
    assert 0.0 <= res_healthy["final_risk_probability"] <= 1.0
    assert 0.0 <= res_healthy["risk_score"] <= 100.0
    assert res_healthy["risk_category"] in ["Healthy", "Low Risk", "Watch", "High Risk", "Critical"]
    
    # High risk inputs
    res_critical = fusion.assess_company(
        ews_prob=0.85,
        bankruptcy_prob=0.90,
        anomaly_score=0.75,
        financial_health_score=15.0,
        use_meta_model=False
    )
    assert 0.0 <= res_critical["final_risk_probability"] <= 1.0
    assert 0.0 <= res_critical["risk_score"] <= 100.0
    assert res_critical["risk_score"] > res_healthy["risk_score"]

def test_trajectory_engine():
    traj_engine = RiskTrajectoryEngine()
    
    time_periods = [1, 2, 3]
    risk_probs = [0.05, 0.08, 0.14]
    
    res = traj_engine.analyze_trajectory(time_periods, risk_probs)
    assert res["trend"] == "Increasing"
    assert res["risk_change_pct_points"] == 6.0
    assert res["consecutive_increases"] == 2

def test_early_warning_alerts():
    alert_engine = EarlyWarningSignalEngine()
    
    trajectory_info = {
        "trend": "Increasing",
        "risk_change_pct_points": 18.0,
        "pattern": "Accelerating Deterioration"
    }
    health_info = {
        "liquidity_score": 10.0,
        "profitability_score": 5.0,
        "financial_health_score": 25.0
    }
    
    signals = alert_engine.generate_signals(
        company_id="TEST_FIRM",
        current_period=4,
        final_prob=0.75,
        trajectory_info=trajectory_info,
        health_info=health_info
    )
    assert len(signals) > 0
    signal_types = [s["signal_type"] for s in signals]
    assert any("Distress" in st or "Deterioration" in st or "Liquidity" in st for st in signal_types)

def test_recommendation_engine():
    rec_engine = RecommendationEngine()
    
    critical_recs = rec_engine.generate_recommendations("Critical", [{"signal_type": "Severe Liquidity Compression"}])
    assert len(critical_recs["recommended_actions"]) >= 3
    assert any("freeze" in a.lower() or "senior" in a.lower() for a in critical_recs["recommended_actions"])
    
    healthy_recs = rec_engine.generate_recommendations("Healthy", [])
    assert len(healthy_recs["recommended_actions"]) >= 1
