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

def test_api_database_singleton_caching(tmp_path):
    """Verifies that the 27MB database is cached in-memory and not reloaded on repeated requests."""
    from api import get_db, get_db_load_count
    
    # 1. Direct function calls return identical singleton object reference
    db1 = get_db()
    initial_count = get_db_load_count()
    db2 = get_db()
    assert db1 is db2
    assert get_db_load_count() == initial_count  # No additional disk read
    
    # 2. Repeated API endpoint requests do not increment load count
    count_before = get_db_load_count()
    resp1 = client.get("/company/364")
    assert resp1.status_code == 200
    resp2 = client.get("/company/364")
    assert resp2.status_code == 200
    assert get_db_load_count() == count_before  # Reused in-memory cache

def test_admin_reload_missing_server_config(monkeypatch):
    """Verifies fail-closed behavior: 503 Service Unavailable when server ADMIN_API_KEY is not set."""
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    response = client.post("/admin/reload-db", headers={"X-Admin-Key": "some-arbitrary-key"})
    assert response.status_code == 503
    assert "ADMIN_API_KEY is not configured" in response.json()["detail"]

def test_admin_reload_missing_request_header(monkeypatch):
    """Verifies that missing X-Admin-Key header is rejected with 401 Unauthorized."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-key-777")
    response = client.post("/admin/reload-db")
    assert response.status_code == 401
    assert "missing X-Admin-Key" in response.json()["detail"]

def test_admin_reload_invalid_key(monkeypatch):
    """Verifies that incorrect key is rejected with 403 Forbidden."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-key-777")
    response = client.post("/admin/reload-db", headers={"X-Admin-Key": "incorrect-password"})
    assert response.status_code == 403
    assert "invalid admin credentials" in response.json()["detail"]

def test_admin_reload_valid_key(monkeypatch):
    """Verifies that authorized requests with matching X-Admin-Key successfully reload the cache."""
    from api import get_db_load_count
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-key-777")
    count_before = get_db_load_count()
    response = client.post("/admin/reload-db", headers={"X-Admin-Key": "test-secure-admin-key-777"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reloaded"
    assert data["load_count"] == count_before + 1
    assert data["companies_loaded"] > 0


def test_api_database_missing_file_handling():
    """Verifies that missing database file returns safe empty schema without crashing."""
    from api import get_db
    result = get_db(db_path="non_existent_path_to_db_12345.json")
    assert isinstance(result, dict)
    assert "companies" in result
    assert result["companies"] == {}

def test_api_database_malformed_file_handling(tmp_path):
    """Verifies that malformed/corrupted JSON file returns safe empty schema without crashing."""
    from api import get_db
    corrupt_file = tmp_path / "corrupt_db.json"
    corrupt_file.write_text("{ this is definitely not valid json ...", encoding="utf-8")
    
    result = get_db(db_path=str(corrupt_file))
    assert isinstance(result, dict)
    assert result.get("companies") == {}

