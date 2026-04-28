"""Tool abstraction layer for model tool-calling."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolContext:
    user_id: int


class ToolProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> dict[str, Any]: ...

    @property
    def enabled(self) -> bool: ...

    async def execute(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]: ...


class ToolManager:
    def __init__(self, *, timeout_seconds: float, max_calls_per_request: int):
        self._timeout_seconds = max(1.0, float(timeout_seconds))
        self._max_calls = max(1, int(max_calls_per_request))
        self._providers: dict[str, ToolProvider] = {}
        self._enabled_names: set[str] | None = None

    def register(self, provider: ToolProvider) -> None:
        self._providers[provider.name] = provider

    def set_enabled_names(self, names: list[str]) -> None:
        normalized = {x.strip() for x in names if x.strip()}
        self._enabled_names = normalized if normalized else None

    def available_schemas(self, wanted_names: list[str] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        names = wanted_names or list(self._providers.keys())
        for name in names:
            p = self._providers.get(name)
            if not p or not p.enabled:
                continue
            if self._enabled_names is not None and name not in self._enabled_names:
                continue
            out.append(p.schema)
        return out

    def max_calls(self) -> int:
        return self._max_calls

    async def execute(self, *, name: str, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        p = self._providers.get(name)
        if not p:
            return {"error": f"unknown tool: {name}"}
        if not p.enabled:
            return {"error": f"tool disabled: {name}"}
        if self._enabled_names is not None and name not in self._enabled_names:
            return {"error": f"tool not enabled by config: {name}"}
        try:
            return await asyncio.wait_for(
                p.execute(args=args, context=context),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {"error": f"tool timeout: {name}"}
        except Exception as e:
            return {"error": f"tool failed: {name}: {e}"}

