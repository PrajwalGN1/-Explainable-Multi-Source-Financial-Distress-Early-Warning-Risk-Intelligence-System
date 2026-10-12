import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("data_validator")

class DataValidator:
    @staticmethod
    def validate_dataset_a(df: pd.DataFrame, id_col="Company", time_col="Time", target_col="Financial Distress") -> Dict[str, any]:
        """Validates temporal structure, monotonic time, and target validity for Dataset A."""
        results = {"valid": True, "errors": [], "warnings": []}
        
        # Check required columns
        for col in [id_col, time_col, target_col]:
            if col not in df.columns:
                results["valid"] = False
                results["errors"].append(f"Missing required column: {col}")
        
        if not results["valid"]:
            return results
            
        # Check infinite values
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            results["warnings"].append(f"Found {inf_count} infinite values in numeric columns.")
            
        # Check time monotonicity per company
        time_monotonic = True
        gaps_detected = 0
        for company_id, group in df.groupby(id_col):
            times = group[time_col].values
            if not np.all(np.diff(times) > 0):
                time_monotonic = False
                results["errors"].append(f"Company {company_id} has non-increasing time periods: {times}")
            # Check for gaps > 1
            if np.any(np.diff(times) > 1):
                gaps_detected += 1
                
        if not time_monotonic:
            results["valid"] = False
            
        results["time_monotonic"] = time_monotonic
        results["companies_with_gaps"] = gaps_detected
        results["total_companies"] = df[id_col].nunique()
        results["min_time"] = int(df[time_col].min())
        results["max_time"] = int(df[time_col].max())
        
        logger.info(f"Dataset A validation complete: Valid={results['valid']}, Companies={results['total_companies']}, Gaps={gaps_detected}")
        return results

    @staticmethod
    def validate_dataset_b(df: pd.DataFrame, target_col="Bankrupt?") -> Dict[str, any]:
        """Validates Dataset B schema, binary target, and values."""
        results = {"valid": True, "errors": [], "warnings": []}
        clean_target = target_col.strip()
        matched_target = None
        for col in df.columns:
            if col.strip() == clean_target:
                matched_target = col
                break
                
        if not matched_target:
            results["valid"] = False
            results["errors"].append(f"Target column '{target_col}' not found in Dataset B columns.")
            return results
            
        target_vals = set(df[matched_target].unique())
        if not target_vals.issubset({0, 1}):
            results["valid"] = False
            results["errors"].append(f"Target column contains non-binary values: {target_vals}")
            
        results["target_distribution"] = df[matched_target].value_counts().to_dict()
        results["num_features"] = len(df.columns) - 1
        logger.info(f"Dataset B validation complete: Valid={results['valid']}, Features={results['num_features']}")
        return results

    @staticmethod
    def validate_dataset_c(df: pd.DataFrame, id_col="ID") -> Dict[str, any]:
        """Validates Dataset C external unlabelled schema."""
        results = {"valid": True, "errors": [], "warnings": []}
        if id_col not in df.columns:
            results["warnings"].append(f"ID column '{id_col}' not found. Defaulting to index.")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results["num_numeric"] = len(numeric_cols)
        results["total_rows"] = len(df)
        logger.info(f"Dataset C validation complete: Rows={results['total_rows']}, NumericCols={results['num_numeric']}")
        return results
