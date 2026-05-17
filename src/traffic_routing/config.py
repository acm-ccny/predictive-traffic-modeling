"""Project paths and environment loading."""
from __future__ import annotations

from pathlib import Path

# src/traffic_routing/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DATASETS_DIR = PROJECT_ROOT / "data" / "ml_datasets"

REQUIRED_ML_DATASETS = (
    "routing_nodes.csv",
    "routing_edges.csv",
    "congestion_ml.csv",
)


def load_project_env() -> None:
    """Load ``.env`` from the repository root when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


load_project_env()
