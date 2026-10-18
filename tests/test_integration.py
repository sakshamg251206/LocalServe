"""Real llama-server behind the real gateway. Skipped unless the llama.cpp build and a model exist."""

import asyncio
import json
import socket
from pathlib import Path

import httpx
import pytest

from localserve.gateway.app import GatewaySettings, create_app
from localserve.gateway.upstream import HttpUpstream
from localserve.manifest import LLAMA_BIN, ROOT
from localserve.server import llama_server

from .conftest import serve, wait_until

MODEL = ROOT / "models" / "qwen3-1.7b-Q4_K_M.gguf"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (LLAMA_BIN / "llama-server").exists() or not MODEL.exists(), reason="no llama.cpp/model"
    ),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def test_stream_tools_and_cancel_against_llama_server(tmp_path: Path) -> None:
    port = free_port()
    with llama_server(MODEL, "it", parallel=2, ctx_per_slot=1024, port=port, log=tmp_path / "llama.log"):
        upstream = HttpUpstream(f"http://127.0.0.1:{port}", 60)
        app = create_app(upstream, GatewaySettings(upstream_url=f"http://127.0.0.1:{port}", max_concurrent=2))
        body = {
            "model": "it",
            "messages": [{"role": "user", "content": "Count to five."}],
            "max_tokens": 16,
            "stream": True,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with serve(app) as url, httpx.AsyncClient(base_url=url, timeout=60) as c:
            r = await c.post("/v1/chat/completions", json=body)
            events = [x[6:] for x in r.text.splitlines() if x.startswith("data: ")]
            assert events[-1] == "[DONE]"
            assert sum(1 for e in events[:-1] if json.loads(e)["choices"][0]["delta"].get("content")) >= 10

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            r = await c.post(
                "/v1/chat/completions",
                json={
                    **body,
                    "stream": False,
                    "max_tokens": 100,
                    "ignore_eos": False,
                    "tools": tools,
                    "messages": [{"role": "user", "content": "Weather in Paris?"}],
                },
            )
            call = r.json()["choices"][0]["message"]["tool_calls"][0]["function"]
            assert call["name"] == "get_weather" and json.loads(call["arguments"])["city"] == "Paris"

            async with c.stream("POST", "/v1/chat/completions", json={**body, "max_tokens": 1000}) as s:
                async for line in s.aiter_lines():
                    if '"content"' in line:
                        break
            assert await wait_until(lambda: app.state.admission.admitted == 0, timeout=5)
            await asyncio.sleep(0.5)
            slots = httpx.get(f"http://127.0.0.1:{port}/slots").json()
            assert not any(sl["is_processing"] for sl in slots)  # upstream generation actually stopped
        await upstream.aclose()
