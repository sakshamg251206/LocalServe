"""Upstream = the model server behind the gateway (llama-server's OpenAI-compatible API)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any, Protocol

import httpx


class UpstreamError(Exception):
    """Upstream answered with a non-200 status; the gateway passes status and body through."""

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"upstream status {status}")
        self.status = status
        self.body = body


class Upstream(Protocol):
    def stream_chat(self, body: dict[str, Any]) -> AsyncGenerator[str, None]:
        """Yield the `data:` payload of each SSE event (JSON text), excluding `[DONE]`."""
        ...

    async def complete_chat(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def list_models(self) -> dict[str, Any]: ...


class HttpUpstream:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        # One pooled client for the gateway's lifetime: reusing keep-alive connections avoids a
        # TCP handshake per request, which would otherwise show up as gateway overhead.
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def stream_chat(self, body: dict[str, Any]) -> AsyncGenerator[str, None]:
        # Leaving this `async with` early (client disconnect, timeout) closes the upstream
        # connection; llama-server notices the closed socket and stops generating for that slot.
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise UpstreamError(r.status_code, await r.aread())
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                yield data
            raise httpx.RemoteProtocolError("upstream stream ended without [DONE]")

    async def complete_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        r = await self._client.post("/v1/chat/completions", json=body)
        if r.status_code != 200:
            raise UpstreamError(r.status_code, r.content)
        result: dict[str, Any] = json.loads(r.content)
        return result

    async def list_models(self) -> dict[str, Any]:
        r = await self._client.get("/v1/models")
        if r.status_code != 200:
            raise UpstreamError(r.status_code, r.content)
        result: dict[str, Any] = r.json()
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
