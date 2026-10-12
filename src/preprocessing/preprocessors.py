import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger("preprocessors")

class EWSPreprocessor:
    """Preprocessor for Dataset A with temporal leakage protection."""
    
    def __init__(self, target_col: str = "forward_target", drop_cols: List[str] = None):
        self.target_col = target_col
        self.drop_cols = drop_cols or ["Company", "Time", "Financial Distress", "current_distress", "forward_target", "forward_target_2period"]
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'EWSPreprocessor':
        """Fits imputation and scaling strictly on training periods."""
        X_train = df_train.drop(columns=[c for c in self.drop_cols if c in df_train.columns], errors="ignore")
        self.feature_names = list(X_train.columns)
        
        X_imp = self.imputer.fit_transform(X_train)
        self.scaler.fit(X_imp)
        self.fitted = True
        logger.info(f"EWSPreprocessor fitted on {len(df_train)} samples with {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms features using fitted statistics."""
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        X = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")
        # Ensure identical column alignment
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        
        y = None
        if self.target_col in df.columns and not df[self.target_col].isna().all():
            y = df[self.target_col].values
            
        return X_scaled, y

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'EWSPreprocessor':
        return joblib.load(filepath)


class BankruptcyPreprocessor:
    """Preprocessor for Dataset B (cross-sectional bankruptcy data)."""
    
    def __init__(self, target_col: str = "Bankrupt?"):
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names: List[str] = []
        self.constant_cols = ["Net Income Flag"]
        self.fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'BankruptcyPreprocessor':
        X_train = df_train.drop(columns=[self.target_col] + self.constant_cols, errors="ignore")
        self.feature_names = list(X_train.columns)
        
        X_imp = self.imputer.fit_transform(X_train)
        self.scaler.fit(X_imp)
        self.fitted = True
        logger.info(f"BankruptcyPreprocessor fitted on {len(df_train)} samples with {len(self.feature_names)} features.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
            
        X = df.drop(columns=[self.target_col] + self.constant_cols, errors="ignore")
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]
        
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        
        y = None
        if self.target_col in df.columns:
            y = df[self.target_col].values
            
        return X_scaled, y

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'BankruptcyPreprocessor':
        return joblib.load(filepath)
