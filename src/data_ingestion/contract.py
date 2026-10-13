"""
Canonical Point-in-Time Enterprise Financial Risk Data Contract (v1.1.0).

Reconciliation Notice (Phase 6.7 to Phase 6.8):
Phase 6.7 documentation referenced a '15-field schema', while the schema dictionary
in data_contract.json defined 16 distinct keys.
In v1.1.0, this is formally reconciled:
- 15 fields represent columnar primitive/scalar attributes:
  (firm_id, source_id, event_id, timestamp_obs, timestamp_pub, timestamp_avail,
   timestamp_ingest, reporting_period, metric_name, metric_value, metric_unit,
   currency, scale_factor, is_imputed, missingness_code).
- 1 field represents the structured cryptographic lineage metadata: (data_provenance).
Total Canonical Field Count = 16.
"""

import re
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

SCHEMA_VERSION = "1.1.0"

APPROVED_SOURCES = {
    "BANK_PLAID_TRANSACTIONS",
    "BANK_CODAT_LEDGER",
    "BUREAU_DNB_PAYDEX",
    "BUREAU_EXPERIAN_COMM",
    "MACRO_FRED_ALFRED",
    "INTERNAL_ACCOUNTING_DATASET_A",
    "SYNTHETIC_TEST_FIXTURE_DO_NOT_USE_FOR_MODELING"
}

APPROVED_UNITS = {
    "CURRENCY_BASE",
    "PERCENT",
    "PERCENT_BPS",
    "RATIO",
    "COUNT",
    "DAYS",
    "FLAG",
    "INDEX_BASE100"
}

APPROVED_MISSINGNESS = {
    "PRESENT",
    "NOT_REPORTED",
    "NOT_APPLICABLE",
    "CORRUPTED",
    "RIGHT_CENSORED"
}

FIRM_ID_REGEX = re.compile(r"^FIRM_HMAC_[a-f0-9]{64}$")
METRIC_NAME_REGEX = re.compile(r"^[a-z0-9_]+$")


def parse_iso8601_utc(ts: Any) -> datetime:
    """Parses an ISO 8601 timestamp string or datetime into UTC datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    if isinstance(ts, str):
        # Handle trailing Z
        clean_ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise ValueError(f"Cannot parse timestamp of type {type(ts)}: {ts}")


class ContractValidationError(Exception):
    """Raised when a data record violates the canonical contract schema."""
    pass


class CanonicalDataContract:
    """
    Specification and validator for Point-in-Time Data Contract v1.1.0.
    """

    VERSION = SCHEMA_VERSION
    FIELD_COUNT = 16

    @classmethod
    def get_schema_specification(cls) -> Dict[str, Any]:
        """Returns the formal contract schema definition."""
        return {
            "title": "Point-in-Time Enterprise Financial Risk Data Contract",
            "schema_version": cls.VERSION,
            "canonical_field_count": cls.FIELD_COUNT,
            "reconciliation": {
                "scalar_attribute_fields_count": 15,
                "metadata_provenance_fields_count": 1,
                "total_reconciled_fields": 16,
                "explanation": "Resolves Phase 6.7 15 vs 16 field count discrepancy by codifying 15 scalar columns + 1 lineage object column."
            },
            "fields": {
                "firm_id": {"type": "STRING", "required": True, "format": "^FIRM_HMAC_[a-f0-9]{64}$"},
                "source_id": {"type": "STRING", "required": True, "enum": sorted(list(APPROVED_SOURCES))},
                "event_id": {"type": "STRING", "required": True, "description": "Unique deterministic event UUID or SHA256"},
                "timestamp_obs": {"type": "TIMESTAMP_UTC", "required": True, "description": "Economic observation timestamp"},
                "timestamp_pub": {"type": "TIMESTAMP_UTC", "required": False, "nullable": True, "description": "Publisher release timestamp"},
                "timestamp_avail": {"type": "TIMESTAMP_UTC", "required": True, "description": "Point-in-time system queryability timestamp"},
                "timestamp_ingest": {"type": "TIMESTAMP_UTC", "required": True, "description": "Physical ingestion timestamp"},
                "reporting_period": {"type": "STRING", "required": True, "description": "Accounting or economic interval token"},
                "metric_name": {"type": "STRING", "required": True, "format": "^[a-z0-9_]+$"},
                "metric_value": {"type": "FLOAT", "required": True, "description": "Numeric measurement"},
                "metric_unit": {"type": "STRING", "required": True, "enum": sorted(list(APPROVED_UNITS))},
                "currency": {"type": "STRING", "required": False, "nullable": True, "description": "ISO 4217 currency code if monetary"},
                "scale_factor": {"type": "FLOAT", "required": True, "default": 1.0},
                "is_imputed": {"type": "BOOLEAN", "required": True, "default": False},
                "missingness_code": {"type": "STRING", "required": True, "enum": sorted(list(APPROVED_MISSINGNESS))},
                "data_provenance": {
                    "type": "OBJECT",
                    "required": True,
                    "properties": ["schema_version", "source_batch_id", "raw_checksum_sha256", "ingestion_pipeline_commit"]
                }
            }
        }


class ContractValidator:
    """Validates records against the canonical schema and point-in-time ordering invariants."""

    @staticmethod
    def validate_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates a single record dictionary against CanonicalDataContract v1.1.0.

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message_or_none)
        """
        # 1. Required fields presence
        required_fields = [
            "firm_id", "source_id", "event_id",
            "timestamp_obs", "timestamp_avail", "timestamp_ingest",
            "reporting_period", "metric_name", "metric_value", "metric_unit",
            "scale_factor", "is_imputed", "missingness_code", "data_provenance"
        ]
        for f in required_fields:
            if f not in record:
                return False, f"Missing mandatory field '{f}'."

        # 2. Pseudonym format
        firm_id = str(record["firm_id"])
        if not FIRM_ID_REGEX.match(firm_id):
            return False, f"Invalid pseudonym format: '{firm_id}'. Must match ^FIRM_HMAC_[a-f0-9]{{64}}$."

        # 3. Source ID enum
        source_id = str(record["source_id"])
        if source_id not in APPROVED_SOURCES:
            return False, f"Unregistered source_id '{source_id}'."

        # 4. Metric name format
        metric_name = str(record["metric_name"])
        if not METRIC_NAME_REGEX.match(metric_name):
            return False, f"Invalid metric_name '{metric_name}'. Must be lowercase snake_case."

        # 5. Metric value numeric check
        try:
            val = float(record["metric_value"])
        except (ValueError, TypeError):
            return False, f"metric_value '{record['metric_value']}' cannot be converted to float."

        # 6. Metric unit enum
        metric_unit = str(record["metric_unit"])
        if metric_unit not in APPROVED_UNITS:
            return False, f"Invalid metric_unit '{metric_unit}'."

        # 7. Currency check for monetary metrics
        if metric_unit == "CURRENCY_BASE":
            currency = record.get("currency")
            if not currency or not isinstance(currency, str) or len(currency) != 3:
                return False, f"Monetary metric requires valid 3-letter ISO 4217 currency; got '{currency}'."

        # 8. Missingness code enum
        missingness = str(record["missingness_code"])
        if missingness not in APPROVED_MISSINGNESS:
            return False, f"Invalid missingness_code '{missingness}'."

        # 9. Provenance object structure
        prov = record.get("data_provenance")
        if not isinstance(prov, dict):
            return False, "data_provenance must be an object."
        for p_key in ["schema_version", "source_batch_id", "raw_checksum_sha256", "ingestion_pipeline_commit"]:
            if p_key not in prov or not prov[p_key]:
                return False, f"data_provenance missing required property '{p_key}'."

        # 10. Temporal ordering invariants
        try:
            t_obs = parse_iso8601_utc(record["timestamp_obs"])
            t_avail = parse_iso8601_utc(record["timestamp_avail"])
            t_ingest = parse_iso8601_utc(record["timestamp_ingest"])

            # T_obs <= T_avail
            if t_obs > t_avail:
                return False, f"Temporal violation: timestamp_obs ({t_obs}) > timestamp_avail ({t_avail})."

            # If timestamp_pub provided: T_obs <= T_pub <= T_avail
            t_pub_raw = record.get("timestamp_pub")
            if t_pub_raw is not None:
                t_pub = parse_iso8601_utc(t_pub_raw)
                if t_obs > t_pub:
                    return False, f"Temporal violation: timestamp_obs ({t_obs}) > timestamp_pub ({t_pub})."
                if t_pub > t_avail:
                    return False, f"Temporal violation: timestamp_pub ({t_pub}) > timestamp_avail ({t_avail})."

            # T_avail <= T_ingest for normal completed ingestion
            if t_avail > t_ingest:
                return False, f"Temporal violation: timestamp_avail ({t_avail}) > timestamp_ingest ({t_ingest})."

        except Exception as e:
            return False, f"Timestamp parsing or sequence error: {str(e)}"

        return True, None
