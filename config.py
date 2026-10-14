"""Central configuration, loaded from environment variables / a .env file."""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    app_name: str
    env: str
    log_level: str
    raw_data_dir: Path
    processed_data_dir: Path
    anthropic_api_key: str


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else BASE_DIR / p


def load_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "nyaya-sahitya"),
        env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        raw_data_dir=_path("RAW_DATA_DIR", "data/raw"),
        processed_data_dir=_path("PROCESSED_DATA_DIR", "data/processed"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )


settings = load_settings()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(settings.app_name)
