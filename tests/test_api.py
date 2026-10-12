import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "temporal_ews_model" in data
    assert "bankruptcy_model" in data
    assert "fusion_engine" in data

def test_predict_endpoint_valid():
    payload = {
        "company_id": "TEST_UNIT_999",
        "ews_probability": 0.45,
        "bankruptcy_probability": 0.30,
        "anomaly_score": 0.20,
        "financial_health_score": 60.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["company_id"] == "TEST_UNIT_999"
    assert 0.0 <= data["final_risk_probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert data["risk_category"] in ["Healthy", "Low Risk", "Watch", "High Risk", "Critical"]
    assert len(data["recommendations"]) > 0

def test_predict_endpoint_invalid_inputs():
    # Negative probability should trigger Pydantic validation error (422)
    payload = {
        "company_id": "INVALID",
        "ews_probability": -0.5,
        "bankruptcy_probability": 0.30,
        "anomaly_score": 0.20,
        "financial_health_score": 60.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_company_profile_endpoint():
    response = client.get("/company/364")
    assert response.status_code == 200
    data = response.json()
    assert data["company_id"] == "364"
    assert "latest_assessment" in data

def test_company_profile_not_found():
    response = client.get("/company/NON_EXISTENT_COMPANY_99999")
    assert response.status_code == 404

def test_watchlist_endpoint():
    response = client.get("/watchlist?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5

def test_risk_summary_endpoint():
    response = client.get("/risk-summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_companies" in data
