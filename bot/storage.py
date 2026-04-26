"""Per-user state persistence backed by a JSON file."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class UserState:
    model: str
    system_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "history": self.history,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_requests": self.total_requests,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], defaults: "UserState") -> "UserState":
        return cls(
            model=d.get("model", defaults.model),
            system_prompt=d.get("system_prompt", defaults.system_prompt),
            temperature=float(d.get("temperature", defaults.temperature)),
            top_p=float(d.get("top_p", defaults.top_p)),
            max_tokens=int(d.get("max_tokens", defaults.max_tokens)),
            stream=bool(d.get("stream", defaults.stream)),
            history=list(d.get("history", [])),
            total_prompt_tokens=int(d.get("total_prompt_tokens", 0)),
            total_completion_tokens=int(d.get("total_completion_tokens", 0)),
            total_requests=int(d.get("total_requests", 0)),
            created_at=float(d.get("created_at", time.time())),
        )


class Storage:
    """Asyncio-safe JSON-file backed storage for per-user state."""

    def __init__(self, data_dir: Path, defaults: UserState):
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "users.json"
        self._defaults = defaults
        self._lock = asyncio.Lock()
        self._users: dict[int, UserState] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
        except Exception as e:
            log.exception("Failed to load %s: %s; starting empty", self._file, e)
            return
        for k, v in raw.items():
            try:
                self._users[int(k)] = UserState.from_dict(v, self._defaults)
            except Exception as e:
                log.warning("skip user %s: %s", k, e)

    async def _persist(self) -> None:
        data = {str(uid): st.to_dict() for uid, st in self._users.items()}
        tmp = self._file.with_suffix(".tmp")
        await asyncio.to_thread(
            tmp.write_text, json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
        )
        await asyncio.to_thread(tmp.replace, self._file)

    async def get(self, user_id: int) -> UserState:
        async with self._lock:
            st = self._users.get(user_id)
            if st is None:
                st = copy.deepcopy(self._defaults)
                self._users[user_id] = st
                await self._persist()
            return st

    async def update(self, user_id: int, **fields: Any) -> UserState:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            for k, v in fields.items():
                if hasattr(st, k):
                    setattr(st, k, v)
            self._users[user_id] = st
            await self._persist()
            return st

    async def append_history(self, user_id: int, messages: list[dict[str, Any]], cap: int) -> None:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            st.history.extend(messages)
            if len(st.history) > cap:
                st.history = st.history[-cap:]
            self._users[user_id] = st
            await self._persist()

    async def add_usage(self, user_id: int, prompt: int, completion: int) -> None:
        async with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return
            st.total_prompt_tokens += int(prompt or 0)
            st.total_completion_tokens += int(completion or 0)
            st.total_requests += 1
            await self._persist()

    async def reset_history(self, user_id: int) -> None:
        async with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return
            st.history = []
            await self._persist()
