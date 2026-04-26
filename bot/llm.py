"""Thin async wrapper around the OpenAI-compatible Chat Completions API."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, base_url: str):
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0),
            max_retries=1,
        )
        self._base_url = base_url

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
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Non-streaming chat. Returns (text, prompt_tokens, completion_tokens)."""
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
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
        max_tokens: int,
    ) -> AsyncIterator[tuple[str, int, int]]:
        """Streaming chat. Yields (delta_text, prompt_tokens, completion_tokens).

        Token counts are 0 for intermediate chunks and only populated on the final chunk
        when the upstream supports `stream_options.include_usage`.
        """
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            delta = ""
            if chunk.choices:
                d = chunk.choices[0].delta
                delta = (getattr(d, "content", "") or "") if d is not None else ""
            usage = getattr(chunk, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
            ct = getattr(usage, "completion_tokens", 0) or 0 if usage else 0
            yield delta, pt, ct
