"""Per-user state persistence backed by a JSON file."""
from __future__ import annotations

import asyncio
import copy
import uuid
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
    repeat_penalty: float
    max_tokens: int
    stream: bool
    web_search: bool
    thinking: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_session_id: str = ""
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
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
            "web_search": self.web_search,
            "thinking": self.thinking,
            "history": self.history,
            "sessions": self.sessions,
            "active_session_id": self.active_session_id,
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
            repeat_penalty=float(d.get("repeat_penalty", defaults.repeat_penalty)),
            max_tokens=int(d.get("max_tokens", defaults.max_tokens)),
            stream=bool(d.get("stream", defaults.stream)),
            web_search=bool(d.get("web_search", defaults.web_search)),
            thinking=bool(d.get("thinking", defaults.thinking)),
            history=list(d.get("history", [])),
            sessions=dict(d.get("sessions", {})),
            active_session_id=str(d.get("active_session_id", "")),
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

    _SESSION_CFG_KEYS = (
        "system_prompt",
        "temperature",
        "top_p",
        "repeat_penalty",
        "max_tokens",
        "stream",
        "web_search",
        "thinking",
    )

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
                st = UserState.from_dict(v, self._defaults)
                self._ensure_sessions(st)
                self._sync_legacy_history(st)
                self._users[int(k)] = st
            except Exception as e:
                log.warning("skip user %s: %s", k, e)

    def _new_session(self, title: str = "新会话") -> tuple[str, dict[str, Any]]:
        sid = uuid.uuid4().hex[:8]
        now = time.time()
        return sid, {
            "id": sid,
            "title": title,
            "history": [],
            "created_at": now,
            "updated_at": now,
        }

    def _copy_cfg_from_state(self, st: UserState) -> dict[str, Any]:
        return {k: getattr(st, k) for k in self._SESSION_CFG_KEYS}

    def _hydrate_session_cfg(self, st: UserState, sess: dict[str, Any]) -> None:
        # Backward compatibility: old sessions may not have per-session config fields.
        for k in self._SESSION_CFG_KEYS:
            if k not in sess:
                sess[k] = getattr(st, k)

    def _sync_active_session(self, st: UserState) -> None:
        self._ensure_sessions(st)
        sess = st.sessions[st.active_session_id]
        self._hydrate_session_cfg(st, sess)
        st.history = list(sess.get("history", []))
        for k in self._SESSION_CFG_KEYS:
            setattr(st, k, sess.get(k))

    def _ensure_sessions(self, st: UserState) -> None:
        if not isinstance(st.sessions, dict):
            st.sessions = {}
        if not st.sessions:
            sid, sess = self._new_session("新会话")
            st.sessions[sid] = sess
            st.active_session_id = sid
        for sess in st.sessions.values():
            if isinstance(sess, dict):
                self._hydrate_session_cfg(st, sess)
        if st.active_session_id not in st.sessions:
            st.active_session_id = next(iter(st.sessions.keys()))

    def _sync_legacy_history(self, st: UserState) -> None:
        self._ensure_sessions(st)
        active = st.sessions[st.active_session_id]
        self._hydrate_session_cfg(st, active)
        if st.history and not active.get("history"):
            active["history"] = list(st.history)
            active["updated_at"] = time.time()
        self._sync_active_session(st)

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
                self._ensure_sessions(st)
                self._sync_legacy_history(st)
                self._users[user_id] = st
                await self._persist()
            else:
                self._ensure_sessions(st)
                self._sync_legacy_history(st)
            return st

    async def update(self, user_id: int, **fields: Any) -> UserState:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            self._ensure_sessions(st)
            sess = st.sessions[st.active_session_id]
            self._hydrate_session_cfg(st, sess)
            for k, v in fields.items():
                if hasattr(st, k):
                    setattr(st, k, v)
                if k in self._SESSION_CFG_KEYS:
                    sess[k] = v
            if "history" in fields:
                sess["history"] = list(fields["history"] or [])
                sess["updated_at"] = time.time()
            if any(k in self._SESSION_CFG_KEYS for k in fields):
                sess["updated_at"] = time.time()
            self._sync_legacy_history(st)
            self._users[user_id] = st
            await self._persist()
            return st

    async def append_history(
        self,
        user_id: int,
        messages: list[dict[str, Any]],
        cap: int,
        title_hint: str | None = None,
    ) -> None:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            self._ensure_sessions(st)
            sess = st.sessions[st.active_session_id]
            hist = list(sess.get("history", []))
            hist.extend(messages)
            if len(hist) > cap:
                hist = hist[-cap:]
            sess["history"] = hist
            sess["updated_at"] = time.time()
            if title_hint and (not sess.get("title") or sess.get("title") == "新会话"):
                sess["title"] = title_hint
            st.history = list(hist)
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
            self._ensure_sessions(st)
            sess = st.sessions[st.active_session_id]
            sess["history"] = []
            sess["updated_at"] = time.time()
            st.history = []
            await self._persist()

    async def list_sessions(self, user_id: int) -> list[dict[str, Any]]:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            self._ensure_sessions(st)
            self._sync_legacy_history(st)
            self._users[user_id] = st
            sessions = list(st.sessions.values())
            sessions.sort(key=lambda x: float(x.get("updated_at", 0.0)), reverse=True)
            return sessions

    async def create_session(self, user_id: int, title: str = "新会话") -> dict[str, Any]:
        async with self._lock:
            st = self._users.get(user_id) or copy.deepcopy(self._defaults)
            self._ensure_sessions(st)
            sid, sess = self._new_session(title)
            # New sessions inherit current config for convenience.
            sess.update(self._copy_cfg_from_state(st))
            st.sessions[sid] = sess
            st.active_session_id = sid
            self._sync_active_session(st)
            self._users[user_id] = st
            await self._persist()
            return sess

    async def switch_session(self, user_id: int, session_id: str) -> bool:
        async with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return False
            self._ensure_sessions(st)
            if session_id not in st.sessions:
                return False
            st.active_session_id = session_id
            self._sync_active_session(st)
            await self._persist()
            return True

    async def delete_session(self, user_id: int, session_id: str) -> tuple[bool, str]:
        async with self._lock:
            st = self._users.get(user_id)
            if st is None:
                return False, ""
            self._ensure_sessions(st)
            if session_id not in st.sessions:
                return False, st.active_session_id
            del st.sessions[session_id]
            if not st.sessions:
                sid, sess = self._new_session("新会话")
                st.sessions[sid] = sess
            if st.active_session_id == session_id:
                st.active_session_id = next(iter(st.sessions.keys()))
            self._sync_active_session(st)
            await self._persist()
            return True, st.active_session_id
