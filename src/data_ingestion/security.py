"""
Security and Pseudonymous Identity Module for External Financial Data Ingestion.

Governance Standards:
- Federal Reserve SR 11-7 / OCC 2011-12 Model Risk Management
- Gramm-Leach-Bliley Act (GLBA) / Fair Credit Reporting Act (FCRA)
- General Data Protection Regulation (GDPR) Article 4(5) & 32 (Pseudonymisation)
- NIST SP 800-57 Part 1 Rev. 5 (Recommendation for Key Management)
"""

import os
import re
import hmac
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Test mock secret used strictly for local unit testing and development fixtures
# NEVER used in production or confirmatory evaluation workflows
TEST_ONLY_MOCK_SECRET = b"TEST_ONLY_MOCK_SECRET_KEY_NOT_FOR_PRODUCTION_USE_0000"
PSEUDONYM_REGEX = re.compile(r"^FIRM_HMAC_[a-f0-9]{64}$")


class SecurityConfigurationError(Exception):
    """Raised when security prerequisites (e.g. cryptographic keys) are absent or invalid."""
    pass


class SecretProvider:
    """
    Interface for retrieving cryptographic keys and secrets.
    Fails closed in production environments if keys are missing.
    """

    @staticmethod
    def get_pseudonym_secret(allow_test_mock: bool = False) -> bytes:
        """
        Retrieves the 256-bit HMAC secret key for borrower pseudonymization.

        Parameters:
            allow_test_mock: If True, falls back to clearly labeled test mock secret
                             strictly for unit test execution.

        Returns:
            bytes: The HMAC secret key.

        Raises:
            SecurityConfigurationError: If no key is configured and allow_test_mock is False.
        """
        env_secret = os.environ.get("EWS_PSEUDONYM_HMAC_SECRET")
        if env_secret:
            return env_secret.encode("utf-8")

        if allow_test_mock:
            logger.warning(
                "SECURITY WARNING: Using clearly labeled TEST_ONLY_MOCK_SECRET. "
                "This secret is prohibited in production or confirmatory evaluation."
            )
            return TEST_ONLY_MOCK_SECRET

        raise SecurityConfigurationError(
            "Missing mandatory environment variable 'EWS_PSEUDONYM_HMAC_SECRET' or HSM connection. "
            "Failing closed in compliance with enterprise data governance."
        )


class Pseudonymizer:
    """
    One-way deterministic cryptographic pseudonymization engine for commercial entities.
    Prevents unauthorized re-identification while preserving deterministic linkage.
    """

    def __init__(self, secret_key: Optional[bytes] = None, allow_test_mock: bool = False):
        if secret_key is not None:
            self._secret_key = secret_key
        else:
            self._secret_key = SecretProvider.get_pseudonym_secret(allow_test_mock=allow_test_mock)

    def mask_entity(self, tax_id: str, country_code: str = "US") -> str:
        """
        Generates a deterministic, irreversible pseudonymous identifier for an entity.

        Formula:
            Pseudonym = FIRM_HMAC_ + HMAC-SHA256(SecretKey, CleanTaxID || CleanCountry)

        Parameters:
            tax_id: Legal tax registration number or business identifier.
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            str: Normalized pseudonymous firm identifier matching ^FIRM_HMAC_[a-f0-9]{64}$.
        """
        if not tax_id or not isinstance(tax_id, str):
            raise ValueError("Entity tax_id must be a non-empty string.")

        clean_tax_id = tax_id.strip().upper()
        clean_country = country_code.strip().upper() if country_code else "UNKNOWN"

        message = f"{clean_tax_id}|{clean_country}".encode("utf-8")
        digest = hmac.new(self._secret_key, message, hashlib.sha256).hexdigest()

        return f"FIRM_HMAC_{digest}"

    @staticmethod
    def validate_pseudonym_format(firm_id: str) -> bool:
        """
        Verifies that a given firm_id strictly conforms to the approved pseudonym format.
        """
        if not firm_id or not isinstance(firm_id, str):
            return False
        return bool(PSEUDONYM_REGEX.match(firm_id))
