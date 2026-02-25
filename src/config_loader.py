"""
config_loader.py
-----------------
Loads and provides the project configuration from config/config.yaml.
"""
import os
import yaml
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config" / "config.yaml"


def load_config(path: str | None = None) -> dict:
    """Load YAML config file and return as dict."""
    target = Path(path) if path else _CONFIG_PATH
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    with open(target, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_project_root() -> Path:
    return _ROOT
