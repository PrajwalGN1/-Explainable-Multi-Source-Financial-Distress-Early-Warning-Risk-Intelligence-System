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

def test_risk_category_boundary_conditions():
    """Verifies contiguous, gap-free, overlap-free mapping across all risk category thresholds."""
    fusion = RiskFusionEngine()
    
    # Boundary 4.0 (Healthy vs Low Risk)
    assert fusion.determine_category(0.0)["label"] == "Healthy"
    assert fusion.determine_category(3.999)["label"] == "Healthy"
    assert fusion.determine_category(4.0)["label"] == "Healthy"
    assert fusion.determine_category(4.001)["label"] == "Low Risk"
    
    # Boundary 8.0 (Low Risk vs Watch)
    assert fusion.determine_category(7.999)["label"] == "Low Risk"
    assert fusion.determine_category(8.0)["label"] == "Low Risk"
    assert fusion.determine_category(8.001)["label"] == "Watch"
    
    # Boundary 13.0 (Watch vs High Risk)
    assert fusion.determine_category(12.999)["label"] == "Watch"
    assert fusion.determine_category(13.0)["label"] == "Watch"
    assert fusion.determine_category(13.001)["label"] == "High Risk"
    
    # Boundary 17.0 (High Risk vs Critical)
    assert fusion.determine_category(16.999)["label"] == "High Risk"
    assert fusion.determine_category(17.0)["label"] == "High Risk"
    assert fusion.determine_category(17.001)["label"] == "Critical"
    
    # Firm #364 critical score check
    assert fusion.determine_category(19.1)["label"] == "Critical"
    assert fusion.determine_category(100.0)["label"] == "Critical"
    
    # Edge cases and bounds clipping
    assert fusion.determine_category(-5.0)["label"] == "Healthy"
    assert fusion.determine_category(105.0)["label"] == "Critical"
    
    # Missing / Invalid handling
    assert fusion.determine_category(None)["label"] == "Watch"
    assert fusion.determine_category(np.nan)["label"] == "Watch"
    assert fusion.determine_category("invalid")["label"] == "Watch"

def test_configured_early_warning_alerts_firing():
    """Verifies that EarlyWarningSignalEngine Rule 1 fires deterministically at calibrated thresholds."""
    alert_engine = EarlyWarningSignalEngine()
    
    # 1. Critical threshold: p=0.18 >= 0.170 should trigger CRITICAL alert
    crit_signals = alert_engine.generate_signals(
        company_id=364,
        current_period=13,
        final_prob=0.18,
        trajectory_info={},
        health_info={}
    )
    assert any(s["signal_type"] == "Critical Distress Risk Level" and s["severity"] == "CRITICAL" for s in crit_signals)
    
    # 2. High threshold: p=0.14 >= 0.130 should trigger HIGH alert
    high_signals = alert_engine.generate_signals(
        company_id=200,
        current_period=13,
        final_prob=0.14,
        trajectory_info={},
        health_info={}
    )
    assert any(s["signal_type"] == "Elevated Distress Warning" and s["severity"] == "HIGH" for s in high_signals)
    assert not any(s["severity"] == "CRITICAL" for s in high_signals)
    
    # 3. Below threshold: p=0.10 < 0.130 should NOT trigger Rule 1
    safe_signals = alert_engine.generate_signals(
        company_id=100,
        current_period=13,
        final_prob=0.10,
        trajectory_info={},
        health_info={}
    )
    assert not any("Distress" in s["signal_type"] for s in safe_signals)

def test_alert_engine_safe_handling_of_edge_cases():
    """Verifies that EarlyWarningSignalEngine gracefully handles missing or corrupt inputs."""
    alert_engine = EarlyWarningSignalEngine()
    
    # Missing probability (None)
    res_none = alert_engine.generate_signals(company_id=1, current_period=1, final_prob=None)
    assert isinstance(res_none, list)
    
    # NaN probability
    res_nan = alert_engine.generate_signals(company_id=1, current_period=1, final_prob=np.nan)
    assert isinstance(res_nan, list)
    
    # None trajectory and health info
    res_no_hist = alert_engine.generate_signals(
        company_id=1,
        current_period=1,
        final_prob=0.05,
        trajectory_info=None,
        health_info=None
    )
    assert isinstance(res_no_hist, list)

def test_dashboard_chart_bands_synchronized():
    """Verifies that charts.py loads canonical bands matching risk_config.yaml."""
    from dashboard.components.charts import get_canonical_risk_bands, create_gauge_chart, create_timeline_chart
    bands = get_canonical_risk_bands()
    assert len(bands) == 5
    assert bands[0]["label"] == "Healthy" and bands[0]["max"] == 4.0
    assert bands[1]["label"] == "Low Risk" and bands[1]["max"] == 8.0
    assert bands[2]["label"] == "Watch" and bands[2]["max"] == 13.0
    assert bands[3]["label"] == "High Risk" and bands[3]["max"] == 17.0
    assert bands[4]["label"] == "Critical" and bands[4]["max"] == 100.0
    
    # Verify gauge chart steps
    gauge = create_gauge_chart(19.1, "Critical", "#ef4444")
    steps = gauge.data[0].gauge.steps
    assert len(steps) == 5
    assert steps[0].range == (0.0, 4.0)
    assert steps[4].range == (17.0, 100.0)
    # Score 19.1 must fall in step 4 (Critical)
    assert steps[4].range[0] <= 19.1 <= steps[4].range[1]

