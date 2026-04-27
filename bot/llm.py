"""Thin async wrapper around the OpenAI-compatible Chat Completions API."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

from .tools.tavily import TavilyClient

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, base_url: str, tavily_api_key: str = ""):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0),
            max_retries=1,
        )
        self._base_url = base_url
        self._tavily = TavilyClient(tavily_api_key)
        self._web_search_support: dict[str, bool | None] = {}
        self._thinking_support: dict[str, bool | None] = {}
        self._tool_calling_support: dict[str, bool | None] = {}
        self._web_tool_schema = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for up-to-date information and return short snippets with URLs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keywords."},
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return (1-10).",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    @staticmethod
    def _is_web_search_unsupported_error(exc: Exception) -> bool:
        text = str(exc).lower()
        web_hint = any(k in text for k in ("web_search", "web search", "search"))
        unsupported_hint = any(
            k in text for k in ("unknown", "unsupported", "not supported", "invalid", "unrecognized")
        )
        return web_hint and unsupported_hint

    def get_web_search_support(self, model: str) -> bool | None:
        """Return cached web-search support for a model.

        True/False are discovered from real requests; None means unknown.
        """
        return self._web_search_support.get(model)

    @staticmethod
    def _is_thinking_unsupported_error(exc: Exception) -> bool:
        text = str(exc).lower()
        thinking_hint = any(
            k in text
            for k in (
                "thinking",
                "reasoning",
                "enable_thinking",
            )
        )
        unsupported_hint = any(
            k in text for k in ("unknown", "unsupported", "not supported", "invalid", "unrecognized")
        )
        return thinking_hint and unsupported_hint

    def get_thinking_support(self, model: str) -> bool | None:
        """Return cached thinking-mode support for a model.

        True/False are discovered from real requests; None means unknown.
        """
        return self._thinking_support.get(model)

    @staticmethod
    def _is_tool_calling_unsupported_error(exc: Exception) -> bool:
        text = str(exc).lower()
        tool_hint = any(k in text for k in ("tool", "tools", "tool_call", "function", "function_call"))
        unsupported_hint = any(
            k in text for k in ("unknown", "unsupported", "not supported", "invalid", "unrecognized")
        )
        return tool_hint and unsupported_hint

    def _base_extra(self, model: str, repeat_penalty: float, web_search: bool, thinking: bool) -> dict[str, Any]:
        extra_body: dict[str, Any] = {"repeat_penalty": repeat_penalty}
        support = self._web_search_support.get(model)
        if web_search and support is not False:
            extra_body["web_search"] = True
        thinking_support = self._thinking_support.get(model)
        if thinking and thinking_support is not False:
            extra_body["enable_thinking"] = True
        elif not thinking and thinking_support is True:
            extra_body["enable_thinking"] = False
        return extra_body

    async def _create_non_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        extra_body: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=False,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return await self._client.chat.completions.create(**kwargs)

    async def _maybe_search_with_tavily(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        extra_body: dict[str, Any],
        web_search: bool,
    ) -> Any | None:
        if not (web_search and self._tavily.enabled):
            return None
        tool_support = self._tool_calling_support.get(model)
        if tool_support is False:
            return None
        try:
            resp = await self._create_non_stream(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
                tools=[self._web_tool_schema],
            )
            self._tool_calling_support[model] = True
        except Exception as e:
            if self._is_tool_calling_unsupported_error(e):
                self._tool_calling_support[model] = False
                return None
            raise

        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []
        if not tool_calls:
            return resp

        tool_messages = list(messages)
        tool_messages.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls],
            }
        )
        for call in tool_calls[:2]:
            fn = getattr(call, "function", None)
            if not fn or getattr(fn, "name", "") != "web_search":
                continue
            args_raw = getattr(fn, "arguments", "") or "{}"
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
            query = str(args.get("query", "")).strip()
            if not query:
                result = {"error": "missing query"}
            else:
                max_results = int(args.get("max_results", 5) or 5)
                try:
                    result = await self._tavily.search(query=query, max_results=max_results)
                except Exception as e:
                    result = {"error": f"tavily search failed: {e}"}
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(call, "id", ""),
                    "name": "web_search",
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        final = await self._create_non_stream(
            model=model,
            messages=tool_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
            tools=None,
        )
        return final

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.models.list()
            ids = sorted({m.id for m in resp.data})
            return ids
        except Exception as e:
            log.warning("list_models failed (%s); returning empty", e)
            return []

    async def chat_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        max_tokens: int,
        web_search: bool,
        thinking: bool,
    ) -> tuple[str, int, int]:
        """Non-streaming chat. Returns (text, prompt_tokens, completion_tokens)."""
        extra_body = self._base_extra(model, repeat_penalty, web_search, thinking)
        try:
            tool_resp = await self._maybe_search_with_tavily(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
                web_search=web_search,
            )
            if tool_resp is not None:
                resp = tool_resp
            else:
                resp = await self._create_non_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                    tools=None,
                )
            if "web_search" in extra_body:
                self._web_search_support[model] = True
            if "enable_thinking" in extra_body:
                self._thinking_support[model] = True
        except Exception as e:
            if "web_search" in extra_body and self._is_web_search_unsupported_error(e):
                self._web_search_support[model] = False
                fallback_extra = {"repeat_penalty": repeat_penalty}
                if "enable_thinking" in extra_body:
                    fallback_extra["enable_thinking"] = extra_body["enable_thinking"]
                resp = await self._create_non_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=fallback_extra,
                    tools=None,
                )
            elif "enable_thinking" in extra_body and self._is_thinking_unsupported_error(e):
                self._thinking_support[model] = False
                fallback_extra = {"repeat_penalty": repeat_penalty}
                if "web_search" in extra_body:
                    fallback_extra["web_search"] = True
                resp = await self._create_non_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=fallback_extra,
                    tools=None,
                )
            else:
                raise
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        return text, pt, ct

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        max_tokens: int,
        web_search: bool,
        thinking: bool,
    ) -> AsyncIterator[tuple[str, int, int]]:
        """Streaming chat. Yields (delta_text, prompt_tokens, completion_tokens).

        Token counts are 0 for intermediate chunks and only populated on the final chunk
        when the upstream supports `stream_options.include_usage`.
        """
        if web_search and self._tavily.enabled:
            text, pt, ct = await self.chat_once(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens,
                web_search=web_search,
                thinking=thinking,
            )
            if not text:
                yield "", pt, ct
                return
            step = 180
            for i in range(0, len(text), step):
                part = text[i : i + step]
                is_last = i + step >= len(text)
                yield part, (pt if is_last else 0), (ct if is_last else 0)
            return

        extra_body = self._base_extra(model, repeat_penalty, web_search, thinking)
        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
                stream=True,
                stream_options={"include_usage": True},
            )
            if "web_search" in extra_body:
                self._web_search_support[model] = True
            if "enable_thinking" in extra_body:
                self._thinking_support[model] = True
        except Exception as e:
            if "web_search" in extra_body and self._is_web_search_unsupported_error(e):
                self._web_search_support[model] = False
                fallback_extra = {"repeat_penalty": repeat_penalty}
                if "enable_thinking" in extra_body:
                    fallback_extra["enable_thinking"] = extra_body["enable_thinking"]
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=fallback_extra,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            elif "enable_thinking" in extra_body and self._is_thinking_unsupported_error(e):
                self._thinking_support[model] = False
                fallback_extra = {"repeat_penalty": repeat_penalty}
                if "web_search" in extra_body:
                    fallback_extra["web_search"] = True
                stream = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body=fallback_extra,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            else:
                raise
        async for chunk in stream:
            delta = ""
            if chunk.choices:
                d = chunk.choices[0].delta
                delta = (getattr(d, "content", "") or "") if d is not None else ""
            usage = getattr(chunk, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            yield delta, pt, ct
