"""Configuration loading."""

from pathlib import Path
import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

REQUIRED_KEYS = {"embedding", "citation_aggregation", "umap", "data", "output"}


def load_config(config_path: str | None = None) -> dict:
    """Load pipeline configuration from YAML file."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")
    return config
