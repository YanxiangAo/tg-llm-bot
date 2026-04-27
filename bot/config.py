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
    max_history_messages: int
    stream_default: bool
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

        return cls(
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
            max_history_messages=_int(os.getenv("MAX_HISTORY_MESSAGES"), 20),
            stream_default=_bool(os.getenv("STREAM_DEFAULT"), True),
            data_dir=Path(os.getenv("DATA_DIR", "/data")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        )
