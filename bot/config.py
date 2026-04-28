"""Environment-driven configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(v: str | None, default: int) -> int:
    try:
        return int(v) if v not in (None, "") else default
    except ValueError:
        return default


def _float(v: str | None, default: float) -> float:
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def _csv(v: str | None) -> list[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


@dataclass
class Config:
    bot_token: str
    api_key: str
    tavily_api_key: str
    base_url: str
    allowed_user_ids: set[int]
    default_model: str
    available_models: list[str]
    default_temperature: float
    default_top_p: float
    default_repeat_penalty: float
    default_max_tokens: int
    web_search_default: bool
    thinking_default: bool
    prompt_cache_default: bool
    max_history_messages: int
    summary_trigger_messages: int
    summary_trigger_tokens: int
    summary_model: str
    summary_keep_recent_messages: int
    summary_max_tokens: int
    summary_context_max_tokens: int
    prompt_max_input_tokens: int
    prompt_keep_recent_messages: int
    stream_default: bool
    embedding_model: str
    rag_enabled_default: bool
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_top_k: int
    rag_min_score: float
    tools_enabled: list[str]
    tool_timeout_seconds: float
    tool_max_calls_per_request: int
    data_dir: Path
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN 未设置，请先在 .env 里填入")

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置，请先在 .env 里填入")

        allowed_raw = _csv(os.getenv("ALLOWED_USER_IDS"))
        allowed_ids: set[int] = set()
        for x in allowed_raw:
            try:
                allowed_ids.add(int(x))
            except ValueError:
                pass

        cfg = cls(
            bot_token=token,
            api_key=api_key,
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/"),
            allowed_user_ids=allowed_ids,
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini").strip(),
            available_models=_csv(os.getenv("AVAILABLE_MODELS")),
            default_temperature=_float(os.getenv("DEFAULT_TEMPERATURE"), 0.7),
            default_top_p=_float(os.getenv("DEFAULT_TOP_P"), 1.0),
            default_repeat_penalty=_float(os.getenv("DEFAULT_REPEAT_PENALTY"), 1.0),
            default_max_tokens=_int(os.getenv("DEFAULT_MAX_TOKENS"), 2048),
            web_search_default=_bool(os.getenv("WEB_SEARCH_DEFAULT"), False),
            thinking_default=_bool(os.getenv("THINKING_DEFAULT"), False),
            prompt_cache_default=_bool(os.getenv("PROMPT_CACHE_DEFAULT"), False),
            max_history_messages=_int(os.getenv("MAX_HISTORY_MESSAGES"), 20),
            summary_trigger_messages=_int(os.getenv("SUMMARY_TRIGGER_MESSAGES"), 16),
            summary_trigger_tokens=_int(os.getenv("SUMMARY_TRIGGER_TOKENS"), 6000),
            summary_model=os.getenv("SUMMARY_MODEL", "").strip(),
            summary_keep_recent_messages=_int(os.getenv("SUMMARY_KEEP_RECENT_MESSAGES"), 8),
            summary_max_tokens=_int(os.getenv("SUMMARY_MAX_TOKENS"), 512),
            summary_context_max_tokens=_int(os.getenv("SUMMARY_CONTEXT_MAX_TOKENS"), 1200),
            prompt_max_input_tokens=_int(os.getenv("PROMPT_MAX_INPUT_TOKENS"), 10000),
            prompt_keep_recent_messages=_int(os.getenv("PROMPT_KEEP_RECENT_MESSAGES"), 10),
            stream_default=_bool(os.getenv("STREAM_DEFAULT"), True),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            rag_enabled_default=_bool(os.getenv("RAG_ENABLED_DEFAULT"), True),
            rag_chunk_size=_int(os.getenv("RAG_CHUNK_SIZE"), 1200),
            rag_chunk_overlap=_int(os.getenv("RAG_CHUNK_OVERLAP"), 200),
            rag_top_k=_int(os.getenv("RAG_TOP_K"), 4),
            rag_min_score=_float(os.getenv("RAG_MIN_SCORE"), 0.35),
            tools_enabled=_csv(os.getenv("TOOLS_ENABLED")),
            tool_timeout_seconds=_float(os.getenv("TOOL_TIMEOUT_SECONDS"), 20.0),
            tool_max_calls_per_request=_int(os.getenv("TOOL_MAX_CALLS_PER_REQUEST"), 2),
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
        # Avoid generating summaries that are immediately clipped when injected into prompt.
        if cfg.summary_context_max_tokens > 0:
            cfg.summary_context_max_tokens = max(cfg.summary_context_max_tokens, cfg.summary_max_tokens)
        return cfg
