"""Configuration: config.yaml, .env and the data directory."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    """Read at call time so tests can point DATA_DIR at a temp folder."""
    path = Path(os.getenv("DATA_DIR") or BASE_DIR / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path
