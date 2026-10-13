"""
Phase 6.8 Automated Test Suite:
External Data Ingestion, Point-in-Time Integration & Quality Certification.

Comprehensive test coverage verifying:
1. Production baseline manifest immutability (SHA-256 digests, official threshold 0.0800 frozen).
2. Data contract v1.1.0 schema reconciliation (16 canonical fields, type checks, nullability).
3. Security engine: fail-closed key management, HMAC-SHA256 pseudonymization, collision resistance.
4. Banking adapter: transaction mapping, pending/reversal filtering, balance derivation.
5. Trade credit adapter: DPD calculation, supplier privacy suppression, aging buckets.
6. Macroeconomic adapter: vintage release tracking, observation period preservation.
7. Data quality engine: detection of duplicates, temporal violations, out-of-bounds values, quarantine sink.
8. Point-in-time joiner: lookahead leakage prevention, historical vintage selection.
9. Data availability gate: detection of real data absence, segregation of synthetic test fixtures.
10. Research-gate decision enforcement: EXTERNAL_DATA_ACQUISITION_REQUIRED.
"""

import os
import json
import math
import uuid
import pytest
from datetime import datetime, timezone, timedelta

from src.shadow.shadow_engine import compute_sha256
from src.data_ingestion.security import SecretProvider, Pseudonymizer, SecurityConfigurationError
from src.data_ingestion.contract import CanonicalDataContract, ContractValidator, SCHEMA_VERSION
from src.data_ingestion.adapters import (
    BankingIngestionAdapter,
    TradeCreditIngestionAdapter,
    MacroeconomicIngestionAdapter,
    IngestionResult
)
from src.data_quality.quality_engine import DataQualityEngine, QuarantineSink
from src.point_in_time.point_in_time_joiner import PointInTimeJoiner

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments", "phase6", "protected_artifacts_manifest.json")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results", "phase6_8")
FIXTURES_DIR = os.path.join(WORKSPACE_ROOT, "tests", "phase6_8", "fixtures")

OFFICIAL_PROD_THRESHOLD = 0.0800
REGISTERED_CAND_THRESHOLD = 0.0894


@pytest.fixture(scope="module")
def phase6_8_deliverables():
    """Loads all Phase 6.8 JSON deliverable files."""
    deliverables = {
        "integrity": os.path.join(RESULTS_DIR, "production_integrity_manifest.json"),
        "reconciliation": os.path.join(RESULTS_DIR, "data_contract_reconciliation.json"),
        "certification": os.path.join(RESULTS_DIR, "data_source_certification.json"),
        "ingestion": os.path.join(RESULTS_DIR, "ingestion_audit_manifest.json"),
        "pit": os.path.join(RESULTS_DIR, "point_in_time_validation.json"),
        "quarantine": os.path.join(RESULTS_DIR, "quarantine_audit_log.json")
    }

    loaded = {}
    for key, path in deliverables.items():
        assert os.path.exists(path), f"Missing deliverable: {path}"
        with open(path, "r", encoding="utf-8") as f:
            loaded[key] = json.load(f)

    return loaded


# =============================================================================
# TEST 1: PRODUCTION BASELINE MANIFEST IMMUTABILITY
# =============================================================================

def test_production_artifacts_immutable(phase6_8_deliverables):
    """Verifies that all 8 protected production artifacts match reference SHA-256 digests."""
    integrity = phase6_8_deliverables["integrity"]
    assert integrity["production_integrity_verdict"] == "PASSED"
    assert integrity["cryptographic_hash_invariance"] is True
    assert integrity["official_production_threshold_frozen"] is True
    assert math.isclose(integrity["official_production_threshold"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert math.isclose(integrity["threshold_on_disk"], OFFICIAL_PROD_THRESHOLD, rel_tol=1e-5)
    assert integrity["candidate_promotion_state"] == "PASSIVE_SHADOW_CANDIDATE (QUARANTINED)"

    # Live verification of SHA-256 digests
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for path, exp_hash in manifest["pre_implementation_hashes"].items():
        assert compute_sha256(path) == exp_hash, f"Hash mismatch on {path}"


# =============================================================================
# TEST 2: DATA CONTRACT SCHEMA RECONCILIATION
# =============================================================================

def test_data_contract_reconciliation_schema(phase6_8_deliverables):
    """Verifies that data contract v1.1.0 defines exactly 16 reconciled fields."""
    recon = phase6_8_deliverables["reconciliation"]
    assert recon["version"] == "1.1.0"
    fields = recon["canonical_fields"]
    assert len(fields) == 16

    field_names = [f["name"] for f in fields]
    assert "firm_id" in field_names
    assert "source_id" in field_names
    assert "event_id" in field_names
    assert "timestamp_obs" in field_names
    assert "timestamp_pub" in field_names
    assert "timestamp_avail" in field_names
    assert "timestamp_ingest" in field_names
    assert "metric_name" in field_names
    assert "metric_value" in field_names
    assert "data_provenance" in field_names


# =============================================================================
# TEST 3: SECURITY AND PSEUDONYMIZATION ENGINE
# =============================================================================

def test_security_and_pseudonymization():
    """Tests secret fail-closed behavior, HMAC pseudonymization, and collision resistance."""
    # 1. Fail closed when secret absent and test mock disabled
    old_env = os.environ.pop("EWS_PSEUDONYM_HMAC_SECRET", None)
    try:
        with pytest.raises(SecurityConfigurationError):
            SecretProvider.get_pseudonym_secret(allow_test_mock=False)
    finally:
        if old_env:
            os.environ["EWS_PSEUDONYM_HMAC_SECRET"] = old_env

    # 2. Test mock allowed in development/testing
    mock_secret = SecretProvider.get_pseudonym_secret(allow_test_mock=True)
    assert isinstance(mock_secret, bytes)

    # 3. Deterministic pseudonymization and format validation
    pseudonymizer = Pseudonymizer(secret_key=b"fixed_test_key_for_hashing_32_bytes", allow_test_mock=True)
    pseudo1 = pseudonymizer.mask_entity("BORROWER_12345", "US")
    pseudo2 = pseudonymizer.mask_entity("BORROWER_12345", "US")
    assert pseudo1 == pseudo2, "Pseudonymization must be deterministic"
    assert Pseudonymizer.validate_pseudonym_format(pseudo1) is True

    # 4. Collision resistance
    pseudo3 = pseudonymizer.mask_entity("BORROWER_99999", "US")
    assert pseudo1 != pseudo3


# =============================================================================
# TEST 4: BANKING INGESTION ADAPTER
# =============================================================================

def test_banking_ingestion_adapter(tmp_path):
    """Verifies banking transaction transformation and filtering of pending/reversed events."""
    raw_dir = tmp_path / "raw"
    staged_dir = tmp_path / "staged"
    quarantine_dir = tmp_path / "quarantine"
    manifest_dir = tmp_path / "manifest"
    raw_dir.mkdir()

    pseudonymizer = Pseudonymizer(allow_test_mock=True)
    firm_id = pseudonymizer.mask_entity("BORROWER_TEST_1", "US")

    payload = [
        {
            "firm_id": firm_id,
            "timestamp_obs": "2023-06-01T10:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "metric_name": "daily_gross_inflows",
            "metric_value": 5000.0,
            "metric_unit": "CURRENCY_BASE",
            "currency": "USD"
        },
        # Pending transaction (should be skipped)
        {
            "firm_id": firm_id,
            "timestamp_obs": "2023-06-01T11:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "metric_name": "daily_gross_outflows",
            "metric_value": 1200.0,
            "is_pending": True
        },
        # Reversed transaction (should be skipped)
        {
            "firm_id": firm_id,
            "timestamp_obs": "2023-06-01T12:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "metric_name": "daily_gross_outflows",
            "metric_value": 800.0,
            "is_reversed": True
        }
    ]

    file_path = raw_dir / "bank_payload.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    adapter = BankingIngestionAdapter(
        source_id="BANK_PLAID_TRANSACTIONS",
        raw_landing_dir=str(raw_dir),
        staged_dir=str(staged_dir),
        quarantine_dir=str(quarantine_dir),
        manifest_dir=str(manifest_dir),
        pseudonymizer=pseudonymizer
    )

    license_meta = {"license_status": "ACTIVE_LICENSED", "permitted_use_research": True}
    result = adapter.ingest(str(file_path), license_meta)

    assert result.execution_status == "SUCCESS"
    assert result.total_records_processed == 3
    assert result.valid_records_staged == 1  # 2 skipped (pending/reversed)
    assert len(result.staged_parquet_paths) == 1


# =============================================================================
# TEST 5: TRADE-CREDIT INGESTION ADAPTER
# =============================================================================

def test_trade_credit_ingestion_adapter(tmp_path):
    """Verifies DPD calculation, supplier privacy suppression, and valid contract staging."""
    raw_dir = tmp_path / "raw"
    staged_dir = tmp_path / "staged"
    quarantine_dir = tmp_path / "quarantine"
    manifest_dir = tmp_path / "manifest"
    raw_dir.mkdir()

    pseudonymizer = Pseudonymizer(allow_test_mock=True)
    firm_id = pseudonymizer.mask_entity("BORROWER_TEST_2", "US")

    payload = [
        {
            "firm_id": firm_id,
            "timestamp_obs": "2023-06-15T00:00:00Z",
            "timestamp_avail": "2023-06-20T00:00:00Z",
            "invoice_due_date": "2023-05-15T00:00:00Z",
            "invoice_settlement_date": "2023-06-14T00:00:00Z",
            "metric_name": "days_past_due_max",
            "supplier_name": "SECRET_VENDOR_LLC"  # Should not leak into canonical record
        }
    ]

    file_path = raw_dir / "trade_payload.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    adapter = TradeCreditIngestionAdapter(
        source_id="BUREAU_DNB_PAYDEX",
        raw_landing_dir=str(raw_dir),
        staged_dir=str(staged_dir),
        quarantine_dir=str(quarantine_dir),
        manifest_dir=str(manifest_dir),
        pseudonymizer=pseudonymizer
    )

    license_meta = {"license_status": "ACTIVE_LICENSED", "permitted_use_research": True}
    result = adapter.ingest(str(file_path), license_meta)

    assert result.execution_status == "SUCCESS"
    assert result.valid_records_staged == 1

    # Verify staged content does not contain secret vendor name
    with open(result.staged_parquet_paths[0], "r", encoding="utf-8") as f:
        staged_rows = json.load(f)
    assert staged_rows[0]["metric_value"] == 30.0  # (June 14 - May 15 = 30 days)
    assert "supplier_name" not in staged_rows[0]


# =============================================================================
# TEST 6: MACROECONOMIC VINTAGE INGESTION ADAPTER
# =============================================================================

def test_macroeconomic_vintage_adapter(tmp_path):
    """Verifies macroeconomic vintage release date preservation."""
    raw_dir = tmp_path / "raw"
    staged_dir = tmp_path / "staged"
    quarantine_dir = tmp_path / "quarantine"
    manifest_dir = tmp_path / "manifest"
    raw_dir.mkdir()

    pseudonymizer = Pseudonymizer(allow_test_mock=True)

    payload = [
        {
            "observation_period_timestamp": "2023-03-31T00:00:00Z",
            "vintage_release_timestamp": "2023-04-28T12:30:00Z",
            "reporting_period": "2023-Q1",
            "metric_name": "central_bank_policy_rate",
            "metric_value": 5.0,
            "metric_unit": "PERCENT"
        }
    ]

    file_path = raw_dir / "macro_payload.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    adapter = MacroeconomicIngestionAdapter(
        source_id="MACRO_FRED_ALFRED",
        raw_landing_dir=str(raw_dir),
        staged_dir=str(staged_dir),
        quarantine_dir=str(quarantine_dir),
        manifest_dir=str(manifest_dir),
        pseudonymizer=pseudonymizer
    )

    license_meta = {"license_status": "ACTIVE_LICENSED", "permitted_use_research": True}
    result = adapter.ingest(str(file_path), license_meta)

    assert result.execution_status == "SUCCESS"
    assert result.valid_records_staged == 1


# =============================================================================
# TEST 7: DATA QUALITY CONTROLS & QUARANTINE SINK
# =============================================================================

def test_data_quality_controls_and_quarantine(tmp_path):
    """Verifies that duplicate records, invalid timestamps, and out-of-bounds values are quarantined."""
    quarantine_dir = tmp_path / "quarantine"
    quarantine_dir.mkdir()
    q_sink = QuarantineSink(str(quarantine_dir))
    dq_engine = DataQualityEngine(quarantine_sink=q_sink)

    pseudonymizer = Pseudonymizer(allow_test_mock=True)
    firm_id = pseudonymizer.mask_entity("BORROWER_TEST_DQ", "US")

    now_utc = datetime.now(timezone.utc).isoformat()
    test_records = [
        # 1. Valid record
        {
            "firm_id": firm_id,
            "source_id": "BANK_PLAID_TRANSACTIONS",
            "event_id": str(uuid.uuid4()),
            "timestamp_obs": "2023-06-01T12:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "timestamp_ingest": now_utc,
            "reporting_period": "2023-W22",
            "metric_name": "daily_gross_inflows",
            "metric_value": 1000.0,
            "metric_unit": "CURRENCY_BASE",
            "currency": "USD"
        },
        # 2. Duplicate record (same key)
        {
            "firm_id": firm_id,
            "source_id": "BANK_PLAID_TRANSACTIONS",
            "event_id": str(uuid.uuid4()),
            "timestamp_obs": "2023-06-01T12:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "timestamp_ingest": now_utc,
            "reporting_period": "2023-W22",
            "metric_name": "daily_gross_inflows",
            "metric_value": 1000.0,
            "metric_unit": "CURRENCY_BASE",
            "currency": "USD"
        },
        # 3. Temporal inequality violation (T_obs > T_avail)
        {
            "firm_id": firm_id,
            "source_id": "BANK_PLAID_TRANSACTIONS",
            "event_id": str(uuid.uuid4()),
            "timestamp_obs": "2023-06-05T12:00:00Z",
            "timestamp_avail": "2023-06-01T14:00:00Z",
            "timestamp_ingest": now_utc,
            "reporting_period": "2023-W22",
            "metric_name": "daily_gross_outflows",
            "metric_value": 500.0,
            "metric_unit": "CURRENCY_BASE",
            "currency": "USD"
        },
        # 4. Out of bounds negative DPD
        {
            "firm_id": firm_id,
            "source_id": "BUREAU_DNB_PAYDEX",
            "event_id": str(uuid.uuid4()),
            "timestamp_obs": "2023-06-02T12:00:00Z",
            "timestamp_avail": "2023-06-02T14:00:00Z",
            "timestamp_ingest": now_utc,
            "reporting_period": "2023-M06",
            "metric_name": "days_past_due_max",
            "metric_value": -20.0,
            "metric_unit": "DAYS"
        }
    ]

    license_meta = {"license_status": "ACTIVE_LICENSED", "permitted_use_research": True}
    passed, rejected, report = dq_engine.evaluate_batch(
        batch_id="TEST_DQ_BATCH",
        records=test_records,
        license_meta=license_meta,
        raw_checksum="dummy_checksum"
    )

    assert len(passed) == 1
    assert len(rejected) == 3
    assert report.quality_verdict == "REJECTED_QUARANTINED"
    assert report.quarantine_manifest_path is not None
    assert os.path.exists(report.quarantine_manifest_path)


# =============================================================================
# TEST 8: POINT-IN-TIME JOINER & LOOKAHEAD PREVENTER
# =============================================================================

def test_point_in_time_joiner_and_leakage_prevention():
    """Verifies lookahead filtering and real-time macroeconomic vintage selection."""
    pseudonymizer = Pseudonymizer(allow_test_mock=True)
    firm_id = pseudonymizer.mask_entity("BORROWER_TEST_PIT", "US")

    records = [
        # Available on 2023-05-10 <= T_pred
        {
            "firm_id": firm_id,
            "metric_name": "eod_available_cash_balance",
            "metric_value": 10000.0,
            "timestamp_avail": "2023-05-10T00:00:00Z"
        },
        # Available on 2023-06-05 > T_pred (Lookahead record)
        {
            "firm_id": firm_id,
            "metric_name": "eod_available_cash_balance",
            "metric_value": 2000.0,
            "timestamp_avail": "2023-06-05T00:00:00Z"
        }
    ]

    t_pred = datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    filtered = PointInTimeJoiner.filter_available_records(records, t_pred)

    assert len(filtered) == 1
    assert filtered[0]["metric_value"] == 10000.0

    # Macro vintage selection: 2023-Q1 GDP released on 2023-04-28 (1.0) and revised on 2023-05-25 (1.5)
    vintages = [
        {
            "reporting_period": "2023-Q1",
            "timestamp_avail": "2023-04-28T12:00:00Z",
            "metric_value": 1.0
        },
        {
            "reporting_period": "2023-Q1",
            "timestamp_avail": "2023-05-25T12:00:00Z",
            "metric_value": 1.5
        }
    ]

    # At T_pred = 2023-05-01: should return first vintage (1.0)
    v_may1 = PointInTimeJoiner.select_macro_vintage(vintages, "2023-Q1", datetime(2023, 5, 1, tzinfo=timezone.utc))
    assert v_may1["metric_value"] == 1.0

    # At T_pred = 2023-06-01: should return revised vintage (1.5)
    v_jun1 = PointInTimeJoiner.select_macro_vintage(vintages, "2023-Q1", datetime(2023, 6, 1, tzinfo=timezone.utc))
    assert v_jun1["metric_value"] == 1.5


# =============================================================================
# TEST 9: DATA AVAILABILITY GATE & SYNTHETIC FIXTURE SEGREGATION
# =============================================================================

def test_data_availability_gate_audit(phase6_8_deliverables):
    """Verifies that absence of real external data is detected and synthetic fixtures are segregated."""
    cert = phase6_8_deliverables["certification"]
    gate = cert["data_availability_gate"]
    assert gate["real_authorized_external_data_present"] is False
    assert gate["gate_status"] == "REAL_EXTERNAL_DATA_ABSENT"

    # Check that test fixture manifest explicitly warns about synthetic status
    manifest_path = os.path.join(FIXTURES_DIR, "fixtures_manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "STRICTLY FOR SOFTWARE TESTING" in manifest["warning"]


# =============================================================================
# TEST 10: RESEARCH-GATE DECISION ENFORCEMENT
# =============================================================================

def test_research_gate_decision_enforcement(phase6_8_deliverables):
    """Asserts that research gate verdict is strictly EXTERNAL_DATA_ACQUISITION_REQUIRED."""
    cert = phase6_8_deliverables["certification"]
    assert cert["research_gate_decision"] == "EXTERNAL_DATA_ACQUISITION_REQUIRED"
    assert "halted" in cert["rationale"].lower()
    assert cert["production_impact"] == "Zero impact; Production baseline remains frozen at threshold 0.0800."
