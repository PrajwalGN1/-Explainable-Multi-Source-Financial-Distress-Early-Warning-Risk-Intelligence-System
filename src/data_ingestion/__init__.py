"""
Data Ingestion Package for Explainable Financial Distress Risk Intelligence System.
Phase 6.8 External Data Ingestion, Point-in-Time Integration & Quality Certification.
"""

from src.data_ingestion.security import SecretProvider, Pseudonymizer
from src.data_ingestion.contract import CanonicalDataContract, ContractValidator
from src.data_ingestion.adapters import (
    BaseIngestionAdapter,
    BankingIngestionAdapter,
    TradeCreditIngestionAdapter,
    MacroeconomicIngestionAdapter,
    IngestionResult
)

__all__ = [
    "SecretProvider",
    "Pseudonymizer",
    "CanonicalDataContract",
    "ContractValidator",
    "BaseIngestionAdapter",
    "BankingIngestionAdapter",
    "TradeCreditIngestionAdapter",
    "MacroeconomicIngestionAdapter",
    "IngestionResult"
]
