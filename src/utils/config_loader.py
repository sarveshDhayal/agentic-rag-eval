"""
Configuration loader utility for agentic-rag-eval.
Loads config.yaml, resolves environment variables from .env,
and ensures deterministic random seeds.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict
import yaml

# Project root directory (parent of 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env(env_path: Path | None = None) -> None:
    """
    Lightweight .env loader that does not crash if python-dotenv is missing.
    Reads KEY=VALUE lines and populates os.environ if not already set.
    """
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"

    if not env_path.is_file():
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        print(f"[Warning] Failed to parse .env file: {e}")


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Loads config.yaml, applies fixed random seed, and returns configuration dict.
    """
    load_env()

    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply global seed for reproducibility
    seed = config.get("project", {}).get("seed", 42)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    return config


def get_project_path(relative_path: str | Path) -> Path:
    """
    Resolves a relative path defined in config.yaml against the project root.
    """
    return (PROJECT_ROOT / relative_path).resolve()


if __name__ == "__main__":
    cfg = load_config()
    print("Project Config Loaded Successfully:")
    print(f"  Title: {cfg['project']['title']}")
    print(f"  Seed:  {cfg['project']['seed']}")
    print(f"  Embedding: {cfg['embedding']['model_name']}")
    print(f"  Generator: {cfg['llm']['generator']['model']}")
    print(f"  Judge:     {cfg['llm']['judge']['model']}")
