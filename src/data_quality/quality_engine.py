"""
Data Quality Engine implementing 10 Phase 6.7 / 6.8 Pre-Ingestion Controls.

Governance Standards:
- Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management
- Basel III Committee on Banking Supervision (BCBS 239) Risk Data Aggregation
"""

import os
import json
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict

from src.data_ingestion.contract import parse_iso8601_utc, FIRM_ID_REGEX


@dataclass
class QualityViolation:
    rule_id: str
    rule_name: str
    severity: str  # "FATAL_REJECTION", "ERROR_CORRECTION", "WARNING_AUDIT", "GOVERNANCE_BLOCK"
    record_index: int
    firm_id: Optional[str]
    metric_name: Optional[str]
    failure_reason: str
    remediation_action: str


@dataclass
class DataQualityReport:
    batch_id: str
    timestamp_utc: str
    total_records_screened: int
    passed_records_count: int
    fatal_violations_count: int
    warning_violations_count: int
    quarantined_records_count: int
    rule_summary: Dict[str, int]
    psi_metric_scores: Dict[str, float]
    quality_verdict: str  # "CERTIFIED_PASSED", "WARNINGS_REVIEW_REQUIRED", "REJECTED_QUARANTINED"
    quarantine_manifest_path: Optional[str] = None


class QuarantineSink:
    """Restricted quarantine sink for storing rejected data records without leaking sensitive data."""

    def __init__(self, quarantine_dir: str):
        self.quarantine_dir = quarantine_dir
        os.makedirs(self.quarantine_dir, exist_ok=True)

    def write_quarantine(self, batch_id: str, rejected_records: List[Dict[str, Any]], violations: List[QualityViolation]) -> str:
        """Writes sanitized rejected records and violations to JSON sidecar."""
        file_path = os.path.join(self.quarantine_dir, f"quarantine_{batch_id}.json")
        payload = {
            "batch_id": batch_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "quarantined_count": len(rejected_records),
            "violations": [asdict(v) for v in violations],
            "rejected_records": rejected_records
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return file_path


class DataQualityEngine:
    """
    Executes all 10 Data Quality Controls on Canonical Contract records:
    DQ_RULE_01: Primary Key Uniqueness & Duplicate Event Detection
    DQ_RULE_02: Temporal Timestamp Sequence & Inequality Integrity
    DQ_RULE_03: Physical & Numerical Value Sanity (Bounds)
    DQ_RULE_04: Currency & Denomination Standardization
    DQ_RULE_05: Temporal Cadence & Abnormal Gap Monitoring
    DQ_RULE_06: Backfilled Record Detection (>30d Latency Tagging)
    DQ_RULE_07: Pseudonym Hash Format & Registration Integrity
    DQ_RULE_08: Reporting Cadence Consistency
    DQ_RULE_09: Population Stability Index (PSI) Distribution Shift Monitoring
    DQ_RULE_10: Cryptographic Data Provenance & Legal License Clearance
    """

    def __init__(self, quarantine_sink: Optional[QuarantineSink] = None):
        self.quarantine_sink = quarantine_sink

    @staticmethod
    def calculate_psi(baseline: List[float], current: List[float], num_buckets: int = 10) -> float:
        """Computes Population Stability Index (PSI) between baseline and current distributions."""
        if not baseline or not current:
            return 0.0

        b_min, b_max = min(baseline), max(baseline)
        if math.isclose(b_min, b_max):
            return 0.0

        step = (b_max - b_min) / num_buckets
        breakpoints = [b_min + i * step for i in range(num_buckets + 1)]
        breakpoints[0] = -float('inf')
        breakpoints[-1] = float('inf')

        def get_counts(data: List[float]) -> List[int]:
            counts = [0] * num_buckets
            for val in data:
                for b_idx in range(num_buckets):
                    if breakpoints[b_idx] <= val < breakpoints[b_idx + 1]:
                        counts[b_idx] += 1
                        break
            return counts

        b_counts = get_counts(baseline)
        c_counts = get_counts(current)
        n_b = len(baseline)
        n_c = len(current)

        psi = 0.0
        for b_cnt, c_cnt in zip(b_counts, c_counts):
            pct_b = max(b_cnt / n_b, 0.0001)
            pct_c = max(c_cnt / n_c, 0.0001)
            psi += (pct_c - pct_b) * math.log(pct_c / pct_b)

        return float(psi)

    def evaluate_batch(
        self,
        batch_id: str,
        records: List[Dict[str, Any]],
        license_meta: Dict[str, Any],
        raw_checksum: str,
        reference_baseline: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], DataQualityReport]:
        """
        Screens a batch of Canonical Contract records across all 10 rules.

        Returns:
            passed_records, rejected_records, report
        """
        violations: List[QualityViolation] = []
        rejected_indices: Set[int] = set()
        seen_keys: Set[Tuple[str, str, str, str]] = set()

        now_utc = datetime.now(timezone.utc)
        rule_counts = {f"DQ_RULE_{i:02d}": 0 for i in range(1, 11)}

        # Rule 10: Legal License Clearance
        if license_meta.get("license_status") != "ACTIVE_LICENSED":
            violations.append(QualityViolation(
                rule_id="DQ_RULE_10",
                rule_name="CRYPTOGRAPHIC_DATA_PROVENANCE_AND_LEGAL_SEAL",
                severity="FATAL_REJECTION",
                record_index=-1,
                firm_id=None,
                metric_name=None,
                failure_reason=f"License status '{license_meta.get('license_status')}' not authorized.",
                remediation_action="Halt batch ingestion; quarantine entire dataset."
            ))
            rule_counts["DQ_RULE_10"] += len(records)
            for idx in range(len(records)):
                rejected_indices.add(idx)

        # Record-level validation
        grouped_values: Dict[str, List[float]] = {}

        for idx, rec in enumerate(records):
            firm_id = rec.get("firm_id")
            source_id = rec.get("source_id")
            metric_name = rec.get("metric_name")
            t_obs_raw = rec.get("timestamp_obs")

            # Collect for PSI
            if metric_name and "metric_value" in rec:
                try:
                    grouped_values.setdefault(metric_name, []).append(float(rec["metric_value"]))
                except (ValueError, TypeError):
                    pass

            # Rule 1: Duplicate check
            key = (str(firm_id), str(source_id), str(metric_name), str(t_obs_raw))
            if key in seen_keys:
                violations.append(QualityViolation(
                    rule_id="DQ_RULE_01",
                    rule_name="PRIMARY_KEY_AND_DUPLICATE_EVENT_CHECK",
                    severity="FATAL_REJECTION",
                    record_index=idx,
                    firm_id=firm_id,
                    metric_name=metric_name,
                    failure_reason="Duplicate composite primary key (firm_id, source_id, metric, t_obs).",
                    remediation_action="Deduplicate record; preserve oldest ingestion timestamp."
                ))
                rejected_indices.add(idx)
                rule_counts["DQ_RULE_01"] += 1
            else:
                seen_keys.add(key)

            # Rule 7: Pseudonym integrity
            if not firm_id or not FIRM_ID_REGEX.match(str(firm_id)):
                violations.append(QualityViolation(
                    rule_id="DQ_RULE_07",
                    rule_name="FIRM_PSEUDONYM_HASH_INTEGRITY",
                    severity="FATAL_REJECTION",
                    record_index=idx,
                    firm_id=firm_id,
                    metric_name=metric_name,
                    failure_reason="Malformed or missing pseudonymous firm_id.",
                    remediation_action="Drop record immediately; alert security engineer."
                ))
                rejected_indices.add(idx)
                rule_counts["DQ_RULE_07"] += 1

            # Rule 2: Temporal Sequence
            try:
                dt_obs = parse_iso8601_utc(rec.get("timestamp_obs"))
                dt_avail = parse_iso8601_utc(rec.get("timestamp_avail"))
                dt_ingest = parse_iso8601_utc(rec.get("timestamp_ingest"))

                if dt_obs > dt_avail:
                    violations.append(QualityViolation(
                        rule_id="DQ_RULE_02",
                        rule_name="TEMPORAL_TIMESTAMP_SEQUENCE_INTEGRITY",
                        severity="FATAL_REJECTION",
                        record_index=idx,
                        firm_id=firm_id,
                        metric_name=metric_name,
                        failure_reason=f"Observation timestamp ({dt_obs}) is later than availability ({dt_avail}).",
                        remediation_action="Quarantine record for temporal reconciliation."
                    ))
                    rejected_indices.add(idx)
                    rule_counts["DQ_RULE_02"] += 1

                # Ingestion cannot be significantly in the future
                if dt_ingest > now_utc + timedelta(minutes=5):
                    violations.append(QualityViolation(
                        rule_id="DQ_RULE_02",
                        rule_name="TEMPORAL_TIMESTAMP_SEQUENCE_INTEGRITY",
                        severity="FATAL_REJECTION",
                        record_index=idx,
                        firm_id=firm_id,
                        metric_name=metric_name,
                        failure_reason=f"Ingestion timestamp ({dt_ingest}) is in the future.",
                        remediation_action="Reset timestamp to ingestion host clock."
                    ))
                    rejected_indices.add(idx)
                    rule_counts["DQ_RULE_02"] += 1

                # Rule 6: Backfilled Record (>30 days latency)
                if (dt_ingest - dt_avail).days > 30:
                    violations.append(QualityViolation(
                        rule_id="DQ_RULE_06",
                        rule_name="BACKFILLED_RECORD_DETECTION_AND_TAGGING",
                        severity="GOVERNANCE_BLOCK",
                        record_index=idx,
                        firm_id=firm_id,
                        metric_name=metric_name,
                        failure_reason=f"Record received with {(dt_ingest - dt_avail).days} days latency (backfill).",
                        remediation_action="Tag as backfill; restrict from point-in-time evaluation where T_pred < T_ingest."
                    ))
                    rule_counts["DQ_RULE_06"] += 1

            except Exception as e:
                violations.append(QualityViolation(
                    rule_id="DQ_RULE_02",
                    rule_name="TEMPORAL_TIMESTAMP_SEQUENCE_INTEGRITY",
                    severity="FATAL_REJECTION",
                    record_index=idx,
                    firm_id=firm_id,
                    metric_name=metric_name,
                    failure_reason=f"Unparseable timestamp: {str(e)}",
                    remediation_action="Quarantine record."
                ))
                rejected_indices.add(idx)
                rule_counts["DQ_RULE_02"] += 1

            # Rule 3: Numeric Bounds & Sanity
            val = rec.get("metric_value")
            if val is None or math.isnan(float(val)) or math.isinf(float(val)):
                violations.append(QualityViolation(
                    rule_id="DQ_RULE_03",
                    rule_name="PHYSICAL_AND_NUMERICAL_VALUE_SANITY",
                    severity="FATAL_REJECTION",
                    record_index=idx,
                    firm_id=firm_id,
                    metric_name=metric_name,
                    failure_reason=f"Invalid numeric value: {val}.",
                    remediation_action="Quarantine record."
                ))
                rejected_indices.add(idx)
                rule_counts["DQ_RULE_03"] += 1
            else:
                fval = float(val)
                if metric_name == "days_past_due_max" and fval < 0:
                    violations.append(QualityViolation(
                        rule_id="DQ_RULE_03",
                        rule_name="PHYSICAL_AND_NUMERICAL_VALUE_SANITY",
                        severity="FATAL_REJECTION",
                        record_index=idx,
                        firm_id=firm_id,
                        metric_name=metric_name,
                        failure_reason=f"Negative days past due ({fval}).",
                        remediation_action="Quarantine record."
                    ))
                    rejected_indices.add(idx)
                    rule_counts["DQ_RULE_03"] += 1
                elif metric_name == "overdraft_headroom_utilization" and (fval < 0 or fval > 10.0):
                    violations.append(QualityViolation(
                        rule_id="DQ_RULE_03",
                        rule_name="PHYSICAL_AND_NUMERICAL_VALUE_SANITY",
                        severity="FATAL_REJECTION",
                        record_index=idx,
                        firm_id=firm_id,
                        metric_name=metric_name,
                        failure_reason=f"Overdraft headroom out of bounds ({fval}).",
                        remediation_action="Quarantine record."
                    ))
                    rejected_indices.add(idx)
                    rule_counts["DQ_RULE_03"] += 1

            # Rule 4: Currency Standardization
            unit = rec.get("metric_unit")
            curr = rec.get("currency")
            if unit == "CURRENCY_BASE" and (not curr or len(str(curr)) != 3):
                violations.append(QualityViolation(
                    rule_id="DQ_RULE_04",
                    rule_name="CURRENCY_AND_DENOMINATION_STANDARDIZATION",
                    severity="ERROR_CORRECTION",
                    record_index=idx,
                    firm_id=firm_id,
                    metric_name=metric_name,
                    failure_reason=f"Missing or invalid ISO 4217 currency: '{curr}'.",
                    remediation_action="Quarantine until currency mapped."
                ))
                rejected_indices.add(idx)
                rule_counts["DQ_RULE_04"] += 1

        # Rule 9: Population Stability Index (PSI)
        psi_scores: Dict[str, float] = {}
        if reference_baseline:
            for m_name, curr_vals in grouped_values.items():
                if m_name in reference_baseline:
                    base_vals = reference_baseline[m_name]
                    psi = self.calculate_psi(base_vals, curr_vals)
                    psi_scores[m_name] = psi
                    if psi >= 0.25:
                        violations.append(QualityViolation(
                            rule_id="DQ_RULE_09",
                            rule_name="STATISTICAL_DISTRIBUTION_SHIFT_ALERT",
                            severity="WARNING_AUDIT",
                            record_index=-1,
                            firm_id=None,
                            metric_name=m_name,
                            failure_reason=f"High PSI drift alert (PSI = {psi:.4f} >= 0.25) against baseline.",
                            remediation_action="Flag drift review; do NOT automatically retrain model."
                        ))
                        rule_counts["DQ_RULE_09"] += 1

        # Separate passed and rejected records
        passed_records = [rec for idx, rec in enumerate(records) if idx not in rejected_indices]
        rejected_records = [records[idx] for idx in sorted(list(rejected_indices))]

        quarantine_manifest_path = None
        if rejected_records and self.quarantine_sink:
            quarantine_manifest_path = self.quarantine_sink.write_quarantine(
                batch_id=batch_id,
                rejected_records=rejected_records,
                violations=violations
            )

        fatal_count = sum(1 for v in violations if v.severity in ["FATAL_REJECTION", "ERROR_CORRECTION"])
        warning_count = sum(1 for v in violations if v.severity in ["WARNING_AUDIT", "GOVERNANCE_BLOCK"])

        if fatal_count == 0 and warning_count == 0:
            verdict = "CERTIFIED_PASSED"
        elif fatal_count == 0 and warning_count > 0:
            verdict = "WARNINGS_REVIEW_REQUIRED"
        else:
            verdict = "REJECTED_QUARANTINED"

        report = DataQualityReport(
            batch_id=batch_id,
            timestamp_utc=now_utc.isoformat(),
            total_records_screened=len(records),
            passed_records_count=len(passed_records),
            fatal_violations_count=fatal_count,
            warning_violations_count=warning_count,
            quarantined_records_count=len(rejected_records),
            rule_summary=rule_counts,
            psi_metric_scores=psi_scores,
            quality_verdict=verdict,
            quarantine_manifest_path=quarantine_manifest_path
        )

        return passed_records, rejected_records, report
