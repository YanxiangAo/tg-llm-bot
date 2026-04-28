"""Tavily tool provider for ToolManager."""
from __future__ import annotations

from typing import Any

from .base import ToolContext
from .tavily import TavilyClient


class TavilyWebSearchTool:
    def __init__(self, api_key: str):
        self._client = TavilyClient(api_key)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for up-to-date information and return snippets with URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keywords."},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum results to return (1-10).",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @property
    def enabled(self) -> bool:
        return self._client.enabled

    async def execute(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "missing query"}
        max_results = int(args.get("max_results", 5) or 5)
        return await self._client.search(query=query, max_results=max_results)

