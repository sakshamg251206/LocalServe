"""OpenAI-compatible gateway in front of llama-server.

Request path (streaming):
  admit (or 429) -> wait for a concurrency slot -> open upstream stream -> wait for first token
  [all under `first_token_timeout_s`, else 504] -> send 200 + relay SSE events until done
  [under `request_timeout_s` total, else a final error event] -> release slot.
Admission and first-token happen *before* response headers are sent, so overload and stalls
surface as real HTTP status codes instead of a 200 with an error buried in the stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .metrics import GatewayMetrics, RequestRecord
from .queue import Admission, QueueFull, Slot
from .upstream import HttpUpstream, Upstream, UpstreamError


@dataclass(frozen=True)
class GatewaySettings:
    upstream_url: str = "http://127.0.0.1:8080"
    max_concurrent: int = 4  # match llama-server --parallel; extra requests wait here, not there
    max_queue: int = 64
    first_token_timeout_s: float = 30.0
    request_timeout_s: float = 300.0

    @classmethod
    def from_env(cls) -> GatewaySettings:
        e = os.environ
        d = cls()
        return cls(
            upstream_url=e.get("LOCALSERVE_UPSTREAM", d.upstream_url),
            max_concurrent=int(e.get("LOCALSERVE_MAX_CONCURRENT", d.max_concurrent)),
            max_queue=int(e.get("LOCALSERVE_MAX_QUEUE", d.max_queue)),
            first_token_timeout_s=float(e.get("LOCALSERVE_FIRST_TOKEN_TIMEOUT", d.first_token_timeout_s)),
            request_timeout_s=float(e.get("LOCALSERVE_REQUEST_TIMEOUT", d.request_timeout_s)),
        )


def _error(status: int, kind: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"type": kind, "message": message}}, status_code=status, headers=headers)


def _sse(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def has_token(data: str) -> bool:
    """True if an SSE chunk carries generated output (llama-server may first send a role-only delta)."""
    try:
        choices = json.loads(data).get("choices") or []
    except (ValueError, AttributeError):
        return False
    for ch in choices:
        delta = ch.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
            return True
    return False


async def _until_first_token(stream: AsyncIterator[str]) -> list[str]:
    """Pull events up to and including the first token-bearing one (or until the stream ends)."""
    buf: list[str] = []
    async for data in stream:
        buf.append(data)
        if has_token(data):
            break
    return buf


class ClientDisconnected(Exception):
    pass


async def _wait_disconnect(request: Request) -> None:
    # The body is already read, so the next ASGI message is http.disconnect when the client hangs up.
    while (await request.receive())["type"] != "http.disconnect":
        pass


async def _unless_disconnected[T](request: Request, coro: Coroutine[Any, Any, T]) -> T:
    """Await `coro`, but cancel it and raise ClientDisconnected if the client hangs up first.

    Only needed before response headers are sent (queued / waiting for first token). Once
    streaming, a failed write closes the relay generator, which cancels upstream by itself.
    """
    work = asyncio.ensure_future(coro)
    gone = asyncio.ensure_future(_wait_disconnect(request))
    try:
        await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        work.cancel()
        raise
    finally:
        gone.cancel()
    if work.done():
        return work.result()
    work.cancel()
    with suppress(BaseException):
        await work
    raise ClientDisconnected


def create_app(upstream: Upstream, settings: GatewaySettings) -> FastAPI:
    admission = Admission(settings.max_concurrent, settings.max_queue)
    metrics = GatewayMetrics(admission)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if isinstance(upstream, HttpUpstream):
            await upstream.aclose()

    app = FastAPI(title="localserve gateway", lifespan=lifespan)
    app.state.admission = admission
    app.state.metrics = metrics

    @app.get("/metrics")
    async def get_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "inflight": admission.inflight, "queued": admission.waiting}

    @app.get("/v1/models")
    async def models() -> Response:
        try:
            return JSONResponse(await upstream.list_models())
        except UpstreamError as e:
            return Response(e.body, status_code=e.status, media_type="application/json")
        except httpx.HTTPError as e:
            return _error(502, "upstream_unavailable", str(e))

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        try:
            body = json.loads(await request.body())
        except ValueError:
            body = None
        if not isinstance(body, dict):
            rec = RequestRecord(stream=False, outcome="bad_request", status=400)
            metrics.finish(rec)
            return _error(400, "invalid_request", "body must be a JSON object")

        rec = RequestRecord(stream=bool(body.get("stream")), model=body.get("model"))
        try:
            slot = admission.admit()
        except QueueFull:
            rec.outcome, rec.status = "rejected", 429
            metrics.finish(rec)
            return _error(429, "queue_full", "server at capacity, retry later", {"Retry-After": "1"})

        if rec.stream:
            return await _start_stream(request, body, slot, rec)
        return await _complete(request, body, slot, rec)

    async def _wait_slot(slot: Slot, rec: RequestRecord) -> None:
        await slot.acquire()
        rec.queue_wait_s = rec.elapsed()

    async def _complete(request: Request, body: dict[str, Any], slot: Slot, rec: RequestRecord) -> Response:
        async def work() -> dict[str, Any]:
            async with asyncio.timeout(settings.request_timeout_s):
                await _wait_slot(slot, rec)
                return await upstream.complete_chat(body)

        try:
            result = await _unless_disconnected(request, work())
            rec.ttft_s = rec.elapsed()  # non-streaming: first byte == whole answer
            return JSONResponse(result)
        except TimeoutError:
            rec.outcome, rec.status = "timeout", 504
            return _error(504, "timeout", "request timed out")
        except ClientDisconnected:
            rec.outcome, rec.status = "cancelled", 499
            return Response(status_code=499)
        except UpstreamError as e:
            rec.outcome, rec.status = "upstream_error", e.status
            return Response(e.body, status_code=e.status, media_type="application/json")
        except httpx.HTTPError as e:
            rec.outcome, rec.status = "upstream_error", 502
            return _error(502, "upstream_unavailable", str(e))
        finally:
            slot.release()
            metrics.finish(rec)

    async def _start_stream(
        request: Request, body: dict[str, Any], slot: Slot, rec: RequestRecord
    ) -> Response:
        stream = upstream.stream_chat(body)

        async def first_token() -> list[str]:
            async with asyncio.timeout(settings.first_token_timeout_s):
                await _wait_slot(slot, rec)
                return await _until_first_token(stream)

        try:
            head = await _unless_disconnected(request, first_token())
        except BaseException as e:
            await stream.aclose()
            slot.release()
            if isinstance(e, TimeoutError):
                rec.outcome, rec.status = "timeout", 504
                resp: Response = _error(504, "timeout", "no first token within deadline")
            elif isinstance(e, UpstreamError):
                rec.outcome, rec.status = "upstream_error", e.status
                resp = Response(e.body, status_code=e.status, media_type="application/json")
            elif isinstance(e, httpx.HTTPError):
                rec.outcome, rec.status = "upstream_error", 502
                resp = _error(502, "upstream_unavailable", str(e))
            elif isinstance(e, ClientDisconnected):
                rec.outcome, rec.status = "cancelled", 499
                resp = Response(status_code=499)
            else:  # cancellation or a bug: account for it, then re-raise
                rec.outcome, rec.status = "cancelled", 499
                metrics.finish(rec)
                raise
            metrics.finish(rec)
            return resp

        if head and has_token(head[-1]):
            rec.ttft_s = rec.elapsed()
        return StreamingResponse(_relay(head, stream, slot, rec), media_type="text/event-stream")

    async def _relay(
        head: list[str], stream: AsyncGenerator[str, None], slot: Slot, rec: RequestRecord
    ) -> AsyncIterator[bytes]:
        deadline = asyncio.get_running_loop().time() + settings.request_timeout_s - rec.elapsed()
        finished = False
        try:
            for data in head:
                rec.chunks += has_token(data)
                yield _sse(data)
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    # Per-event wait_for rather than asyncio.timeout(): a timeout scope must not
                    # span a `yield` in an async generator (the cancel could land in the caller).
                    data = await asyncio.wait_for(anext(stream), timeout=max(remaining, 0.0))
                except StopAsyncIteration:
                    break
                rec.chunks += has_token(data)
                yield _sse(data)
            yield b"data: [DONE]\n\n"
            finished = True
        except TimeoutError:
            rec.outcome = "timeout"
            yield _sse(json.dumps({"error": {"type": "timeout", "message": "request timed out"}}))
        except (UpstreamError, httpx.HTTPError) as e:
            rec.outcome = "upstream_error"
            yield _sse(json.dumps({"error": {"type": "upstream_error", "message": str(e)}}))
        finally:
            # Reached on normal end, timeout, or client disconnect (generator closed/cancelled).
            if not finished and rec.outcome == "ok":
                rec.outcome = "cancelled"
            await stream.aclose()
            slot.release()
            metrics.finish(rec)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = GatewaySettings.from_env()
    app = create_app(HttpUpstream(settings.upstream_url, settings.request_timeout_s), settings)
    uvicorn.run(
        app,
        host=os.environ.get("LOCALSERVE_HOST", "127.0.0.1"),
        port=int(os.environ.get("LOCALSERVE_PORT", "8000")),
        log_level="warning",  # access logs off: our JSON line per request replaces them
    )


if __name__ == "__main__":
    main()
