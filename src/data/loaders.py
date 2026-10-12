import os
import pandas as pd
from typing import Dict, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")

class DataLoader:
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        
    def load_dataset_a(self, filename: str = "Financial Distress.csv") -> pd.DataFrame:
        """Loads Dataset A (Financial Distress temporal panel dataset)."""
        fpath = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset A not found at: {fpath}")
        logger.info(f"Loading Dataset A from {fpath}...")
        df = pd.read_csv(fpath)
        logger.info(f"Dataset A loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def load_dataset_b(self, filename: str = "data.csv") -> pd.DataFrame:
        """Loads Dataset B (Taiwan Economic Journal bankruptcy dataset)."""
        fpath = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset B not found at: {fpath}")
        logger.info(f"Loading Dataset B from {fpath}...")
        df = pd.read_csv(fpath)
        # Strip leading/trailing whitespaces in column names if any
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"Dataset B loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def load_dataset_c(self, filename: str = "bankruptcy_Test_X.csv") -> pd.DataFrame:
        """Loads Dataset C (External unlabelled Polish bankruptcy test dataset)."""
        fpath = os.path.join(self.raw_data_dir, filename)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Dataset C not found at: {fpath}")
        logger.info(f"Loading Dataset C from {fpath}...")
        df = pd.read_csv(fpath)
        logger.info(f"Dataset C loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            self.load_dataset_a(),
            self.load_dataset_b(),
            self.load_dataset_c()
        )
