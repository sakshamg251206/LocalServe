"""Shared test fakes. Everything here runs offline: no model, no network beyond 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI

from localserve.gateway.upstream import UpstreamError


def chunk(content: str) -> dict[str, Any]:
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}}]}


class FakeUpstream:
    """Async streaming upstream with configurable delays; counts started/completed/cancelled streams."""

    def __init__(
        self,
        tokens: tuple[str, ...] = ("Hello", " world", "!"),
        first_delay: float = 0.0,
        token_delay: float = 0.0,
        fail_status: int | None = None,
    ) -> None:
        self.tokens = tokens
        self.first_delay = first_delay
        self.token_delay = token_delay
        self.fail_status = fail_status
        self.started = 0
        self.completed = 0
        self.cancelled = 0
        self.bodies: list[dict[str, Any]] = []

    async def stream_chat(self, body: dict[str, Any]) -> AsyncGenerator[str, None]:
        self.started += 1
        self.bodies.append(body)
        if self.fail_status is not None:
            raise UpstreamError(self.fail_status, b'{"error":"fake failure"}')
        finished = False
        try:
            await asyncio.sleep(self.first_delay)
            for i, tok in enumerate(self.tokens):
                if i:
                    await asyncio.sleep(self.token_delay)
                yield json.dumps(chunk(tok))
            finished = True
            self.completed += 1
        finally:
            if not finished:
                self.cancelled += 1

    async def complete_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.started += 1
        self.bodies.append(body)
        if self.fail_status is not None:
            raise UpstreamError(self.fail_status, b'{"error":"fake failure"}')
        await asyncio.sleep(self.first_delay + self.token_delay * (len(self.tokens) - 1))
        self.completed += 1
        return {
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(self.tokens)}}],
        }

    async def list_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[str]:
    """Run `app` under uvicorn on an ephemeral 127.0.0.1 port; yield its base URL."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task
        sock.close()


async def wait_until(pred: Any, timeout: float = 2.0) -> bool:
    """Poll `pred()` until true or timeout (test helper for async side effects)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.01)
    return bool(pred())
