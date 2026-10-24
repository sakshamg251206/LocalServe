import json

import httpx

from localserve.gateway.app import GatewaySettings, create_app

from .conftest import FakeUpstream, serve

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def sse_data(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


async def test_stream_relays_tokens_in_order_then_done() -> None:
    fake = FakeUpstream(tokens=("a", "b", "c", "d"), token_delay=0.01)
    async with serve(create_app(fake, GatewaySettings())) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse_data(r.text)
    assert events[-1] == "[DONE]"
    assert [json.loads(e)["choices"][0]["delta"]["content"] for e in events[:-1]] == ["a", "b", "c", "d"]
    assert fake.bodies[0]["stream"] is True  # body forwarded unchanged


async def test_non_stream_returns_json() -> None:
    fake = FakeUpstream(tokens=("x", "y"))
    async with serve(create_app(fake, GatewaySettings())) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "xy"


async def test_upstream_error_status_is_passed_through() -> None:
    fake = FakeUpstream(fail_status=400)
    async with serve(create_app(fake, GatewaySettings())) as url, httpx.AsyncClient(base_url=url) as c:
        r1 = await c.post("/v1/chat/completions", json={**BODY, "stream": True})
        r2 = await c.post("/v1/chat/completions", json=BODY)
    assert (r1.status_code, r2.status_code) == (400, 400)


async def test_invalid_json_is_400() -> None:
    async with (
        serve(create_app(FakeUpstream(), GatewaySettings())) as url,
        httpx.AsyncClient(base_url=url) as c,
    ):
        r1 = await c.post("/v1/chat/completions", content=b"{not json")
        r2 = await c.post("/v1/chat/completions", json=[1, 2])
    assert (r1.status_code, r2.status_code) == (400, 400)


async def test_models_endpoint_proxies_upstream() -> None:
    async with (
        serve(create_app(FakeUpstream(), GatewaySettings())) as url,
        httpx.AsyncClient(base_url=url) as c,
    ):
        r = await c.get("/v1/models")
    assert r.json()["data"][0]["id"] == "fake-model"


async def test_total_deadline_applies_before_first_token() -> None:
    fake = FakeUpstream(first_delay=0.3)
    settings = GatewaySettings(first_token_timeout_s=1, request_timeout_s=0.05)
    app = create_app(fake, settings)
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 504
    assert app.state.admission.admitted == 0
