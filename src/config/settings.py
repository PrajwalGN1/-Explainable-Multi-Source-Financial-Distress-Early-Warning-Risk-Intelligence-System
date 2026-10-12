import os
import yaml
from typing import Dict, Any

def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_config(base_dir: str = ".") -> Dict[str, Any]:
    config_path = os.path.join(base_dir, "config", "config.yaml")
    return load_yaml(config_path)

def get_risk_config(base_dir: str = ".") -> Dict[str, Any]:
    risk_config_path = os.path.join(base_dir, "config", "risk_config.yaml")
    return load_yaml(risk_config_path)
