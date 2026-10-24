"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openrouter_api_key: str | None
    openrouter_base_url: str
    openai_base_url: str

    embedding_model: str
    jev_model: str
    llm_provider: str  # "openai" or "openrouter"
    llm_model: str
    openai_stt_model: str
    stt_language: str | None  # ISO 639-1; None = auto-detect
    openai_tts_model: str
    openai_tts_voice: str

    @property
    def llm_api_key(self) -> str | None:
        return (
            self.openai_api_key
            if self.llm_provider == "openai"
            else self.openrouter_api_key
        )

    @property
    def llm_base_url(self) -> str:
        return (
            self.openai_base_url
            if self.llm_provider == "openai"
            else self.openrouter_base_url
        )

    vector_store_path: Path
    metadata_db_path: Path
    uploads_path: Path
    extracted_path: Path

    top_k: int
    max_retrieval_attempts: int

    enable_answer_verification: bool
    debug_mode: bool

    log_level: str


def load_settings() -> Settings:
    provider = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    if provider not in {"openai", "openrouter"}:
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'openrouter', got {provider!r}"
        )
    default_llm_model = "gpt-4o-mini" if provider == "openai" else "openai/gpt-4o-mini"

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        jev_model=os.getenv("JEV_MODEL", "typesafe/jev-router"),
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", default_llm_model),
        openai_stt_model=os.getenv("OPENAI_STT_MODEL", "whisper-1"),
        stt_language=(os.getenv("STT_LANGUAGE", "en").strip() or None),
        openai_tts_model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        openai_tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
        vector_store_path=_path("VECTOR_STORE_PATH", "data/indexes"),
        metadata_db_path=_path("METADATA_DB_PATH", "data/metadata/tutor.db"),
        uploads_path=_path("UPLOADS_PATH", "data/uploads"),
        extracted_path=_path("EXTRACTED_PATH", "data/extracted"),
        top_k=_int("TOP_K", 5),
        max_retrieval_attempts=_int("MAX_RETRIEVAL_ATTEMPTS", 2),
        enable_answer_verification=_bool("ENABLE_ANSWER_VERIFICATION", False),
        debug_mode=_bool("DEBUG_MODE", False),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


settings = load_settings()
