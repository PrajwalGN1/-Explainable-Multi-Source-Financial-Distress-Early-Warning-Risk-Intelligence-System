"""
Phase 5: Robustness Analysis Entry Point.

Wraps nested validation and cross-sectional group stability analysis.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.evaluation.phase5_nested_validation import run_nested_validation_analysis

if __name__ == "__main__":
    run_nested_validation_analysis()
