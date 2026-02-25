"""
Path resolution utilities for the RDT Trading System.

Provides consistent project root resolution regardless of CWD.
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Resolution order:
    1. RDT_ROOT environment variable
    2. Walk up from this file to find the project root
    """
    env_root = os.environ.get('RDT_ROOT')
    if env_root:
        root = Path(env_root)
        if root.exists():
            return root

    # Walk up from this file (utils/paths.py -> utils -> project_root)
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / 'data'


def get_models_dir() -> Path:
    """Get the models directory."""
    return get_project_root() / 'models'


def get_logs_dir() -> Path:
    """Get the logs directory."""
    return get_project_root() / 'logs'
