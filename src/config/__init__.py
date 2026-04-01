"""Configuration loading."""

from pathlib import Path
import yaml


def load_config(config_path: str = "src/config/config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
