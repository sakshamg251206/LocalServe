"""One streaming chat request, timed from the client's side."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from localserve.gateway.app import has_token


@dataclass
class RequestResult:
    idx: int
    t_sched: float  # scheduled send time, seconds from run start
    t_sent: float  # actual send time (t_sent - t_sched = client lag; should be ~ms)
    status: int  # HTTP status; 0 = client timeout or connection error
    error: str | None
    ttft_s: float | None  # send -> first generated token
    # Gaps between consecutive token-bearing SSE events: the stall a reader sees. Usually one token
    # per event, but llama-server sometimes holds a token back and sends it merged with the next.
    itl_s: list[float] = field(default_factory=list)
    e2e_s: float = 0.0
    n_events: int = 0  # token-bearing SSE events
    completion_tokens: int | None = None  # authoritative, from the final `usage` chunk
    prompt_tokens: int | None = None

    @property
    def n_tokens(self) -> int:
        return self.completion_tokens if self.completion_tokens is not None else self.n_events

    @property
    def tpot_s(self) -> float | None:
        """Time per output token after the first: (e2e - ttft) / (tokens - 1)."""
        if self.ttft_s is None or self.n_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.n_tokens - 1)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


async def send_chat(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    timeout_s: float,
    idx: int = 0,
    t_sched: float = 0.0,
    t_run0: float | None = None,
) -> RequestResult:
    t0 = time.perf_counter()
    res = RequestResult(
        idx=idx,
        t_sched=t_sched,
        t_sent=t0 - t_run0 if t_run0 is not None else t_sched,
        status=0,
        error=None,
        ttft_s=None,
    )
    last: float | None = None
    try:
        async with asyncio.timeout(timeout_s), client.stream("POST", url, json=body) as r:
            res.status = r.status_code
            if r.status_code != 200:
                res.error = (await r.aread()).decode(errors="replace")[:200]
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                now = time.perf_counter()
                if has_token(data):
                    if last is None:
                        res.ttft_s = now - t0
                    else:
                        res.itl_s.append(now - last)
                    last = now
                    res.n_events += 1
                if '"usage"' in data or '"error"' in data:
                    obj = json.loads(data)
                    if obj.get("error"):
                        res.error = str(obj["error"])[:200]
                    if obj.get("usage"):
                        res.prompt_tokens = obj["usage"].get("prompt_tokens")
                        res.completion_tokens = obj["usage"].get("completion_tokens")
            else:
                res.error = res.error or "stream ended without [DONE]"
            if (
                body.get("ignore_eos")
                and res.completion_tokens is not None
                and res.completion_tokens != body.get("max_tokens")
            ):
                res.error = res.error or "completion token count differs from requested fixed length"
    except TimeoutError:
        res.status, res.error = 0, "client_timeout"
    except httpx.HTTPError as e:
        res.status, res.error = 0, type(e).__name__
    finally:
        res.e2e_s = time.perf_counter() - t0
    return res
