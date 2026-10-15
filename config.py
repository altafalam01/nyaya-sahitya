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
    chunks_data_dir: Path
    index_dir: Path
    embed_model: str
    embed_batch_size: int
    rerank_model: str
    dense_candidates: int
    bm25_candidates: int
    rerank_top: int
    anthropic_api_key: str
    llm_provider: str
    gemini_api_key: str
    gemini_model: str
    ollama_url: str
    ollama_model: str
    answer_model: str
    answer_max_tokens: int
    answer_top_k: int
    cors_origins: list[str]
    api_url: str
    user_agent: str
    download_delay_seconds: float


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
        chunks_data_dir=_path("CHUNKS_DATA_DIR", "data/chunks"),
        index_dir=_path("CHROMA_DIR", "data/index"),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "16")),
        rerank_model=os.getenv("RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
        dense_candidates=int(os.getenv("DENSE_CANDIDATES", "50")),
        bm25_candidates=int(os.getenv("BM25_CANDIDATES", "50")),
        rerank_top=int(os.getenv("RERANK_TOP", "20")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        answer_model=os.getenv("ANSWER_MODEL", "claude-sonnet-5-5"),
        answer_max_tokens=int(os.getenv("ANSWER_MAX_TOKENS", "1500")),
        answer_top_k=int(os.getenv("ANSWER_TOP_K", "6")),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",") if o.strip()],
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        user_agent=os.getenv("USER_AGENT", "NyayaSahityaBot/0.1 (legal-research; non-commercial)"),
        download_delay_seconds=float(os.getenv("DOWNLOAD_DELAY_SECONDS", "5")),
    )


settings = load_settings()


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(settings.app_name)
