import os
import yaml
from typing import Dict, Any, Optional

def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _resolve_config_path(subpath: str, base_dir: Optional[str] = None) -> str:
    if base_dir is not None:
        return os.path.join(base_dir, subpath)
    if os.path.exists(subpath):
        return subpath
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, subpath)

def get_config(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml(_resolve_config_path(os.path.join("config", "config.yaml"), base_dir))

def get_risk_config(base_dir: Optional[str] = None) -> Dict[str, Any]:
    return load_yaml(_resolve_config_path(os.path.join("config", "risk_config.yaml"), base_dir))
