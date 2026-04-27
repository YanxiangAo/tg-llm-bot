"""Async Tavily search client."""
from __future__ import annotations

from typing import Any

import httpx


class TavilyClient:
    def __init__(self, api_key: str):
        self._api_key = api_key.strip()
        self._url = "https://api.tavily.com/search"

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    async def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Tavily API key is not configured")

        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max(1, min(int(max_results or 5), 10)),
            "include_answer": True,
            "include_raw_content": False,
        }
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=15.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data if isinstance(data, dict) else {"results": []}

