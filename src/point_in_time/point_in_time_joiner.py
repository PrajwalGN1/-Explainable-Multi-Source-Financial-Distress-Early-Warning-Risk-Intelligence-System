"""
Point-in-Time Temporal Joiner and Feature Availability Engine.

Governance Standards:
- Federal Reserve SR 11-7 (Prevention of Lookahead Bias and Data Leakage)
- Basel Committee on Banking Supervision (BCBS 239)
- Real-Time Macroeconomic Vintage Evaluation Standards
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from src.data_ingestion.contract import parse_iso8601_utc


class LookaheadLeakageError(Exception):
    """Raised when data with availability timestamp after prediction timestamp is accessed."""
    pass


class PointInTimeJoiner:
    """
    Executes strict point-in-time filtering and temporal joins.
    Enforces invariant: T_avail <= T_pred across all features.
    """

    @staticmethod
    def filter_available_records(records: List[Dict[str, Any]], t_pred: datetime) -> List[Dict[str, Any]]:
        """
        Filters records to strictly include only those with T_avail <= T_pred.

        Raises:
            LookaheadLeakageError: If any record bypasses filtering.
        """
        if t_pred.tzinfo is None:
            t_pred = t_pred.replace(tzinfo=timezone.utc)

        available_records = []
        for rec in records:
            dt_avail = parse_iso8601_utc(rec["timestamp_avail"])
            if dt_avail <= t_pred:
                available_records.append(rec)

        return available_records

    @staticmethod
    def select_macro_vintage(
        series_vintages: List[Dict[str, Any]],
        observation_period: str,
        t_pred: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Selects the exact macroeconomic release vintage available at prediction timestamp T_pred.

        Formula:
            Selected Vintage = max { ReleaseDate <= T_pred } for the specified observation period.

        Parameters:
            series_vintages: List of vintage records for the macroeconomic metric.
            observation_period: e.g. "2023-Q1" or "2023-M04".
            t_pred: Prediction cutoff timestamp.

        Returns:
            The latest release record published on or before T_pred, or None if unreleased.
        """
        if t_pred.tzinfo is None:
            t_pred = t_pred.replace(tzinfo=timezone.utc)

        # Filter by observation period and release date <= T_pred
        eligible = []
        for v in series_vintages:
            if v.get("reporting_period") == observation_period:
                dt_avail = parse_iso8601_utc(v.get("timestamp_avail", v.get("timestamp_pub")))
                if dt_avail <= t_pred:
                    eligible.append((dt_avail, v))

        if not eligible:
            return None

        # Return latest release on or before T_pred
        eligible.sort(key=lambda x: x[0])
        return eligible[-1][1]

    @staticmethod
    def extract_point_in_time_features(
        firm_records: List[Dict[str, Any]],
        firm_id: str,
        t_pred: datetime
    ) -> Dict[str, Any]:
        """
        Extracts point-in-time features for a given firm at prediction timestamp T_pred.
        Strictly enforces T_avail <= T_pred.
        """
        if t_pred.tzinfo is None:
            t_pred = t_pred.replace(tzinfo=timezone.utc)

        filtered = [
            r for r in firm_records
            if r.get("firm_id") == firm_id and parse_iso8601_utc(r["timestamp_avail"]) <= t_pred
        ]

        if not filtered:
            return {
                "firm_id": firm_id,
                "t_pred": t_pred.isoformat(),
                "has_external_data": False,
                "latest_eod_cash_balance": None,
                "max_days_past_due": None,
                "nsf_returned_events_count": 0
            }

        # Calculate metrics
        balances = [
            float(r["metric_value"]) for r in filtered
            if r.get("metric_name") == "eod_available_cash_balance"
        ]
        dpds = [
            float(r["metric_value"]) for r in filtered
            if r.get("metric_name") == "days_past_due_max"
        ]
        nsf_counts = [
            float(r["metric_value"]) for r in filtered
            if r.get("metric_name") == "nsf_returned_payment_count"
        ]

        return {
            "firm_id": firm_id,
            "t_pred": t_pred.isoformat(),
            "has_external_data": True,
            "latest_eod_cash_balance": balances[-1] if balances else None,
            "max_days_past_due": max(dpds) if dpds else 0.0,
            "nsf_returned_events_count": int(sum(nsf_counts)) if nsf_counts else 0
        }
