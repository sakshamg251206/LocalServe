import contextlib

import httpx

from localserve.gateway.app import GatewaySettings, create_app

from .conftest import FakeUpstream, serve, wait_until

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


async def test_client_disconnect_cancels_upstream_and_frees_slot() -> None:
    fake = FakeUpstream(tokens=tuple(str(i) for i in range(200)), token_delay=0.02)
    app = create_app(fake, GatewaySettings(max_concurrent=1, max_queue=0))
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        async with c.stream("POST", "/v1/chat/completions", json=BODY) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    break  # got first token; hang up
        assert await wait_until(lambda: fake.cancelled == 1)
        assert await wait_until(lambda: app.state.admission.admitted == 0)
        assert fake.completed == 0
        # slot is reusable
        r2 = await c.post("/v1/chat/completions", json={**BODY, "stream": False})
        assert r2.status_code == 200


async def test_first_token_timeout_is_504_and_cancels_upstream() -> None:
    fake = FakeUpstream(first_delay=2.0)
    app = create_app(fake, GatewaySettings(first_token_timeout_s=0.1))
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 504
    assert fake.cancelled == 1
    assert app.state.admission.admitted == 0


async def test_total_timeout_mid_stream_ends_with_error_event() -> None:
    fake = FakeUpstream(tokens=tuple("abcdefghij"), token_delay=0.1)
    app = create_app(fake, GatewaySettings(request_timeout_s=0.35))
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200  # headers were already sent
    assert '"type": "timeout"' in r.text
    assert "[DONE]" not in r.text
    assert await wait_until(lambda: fake.cancelled == 1)
    assert app.state.admission.admitted == 0


async def test_non_stream_timeout_is_504() -> None:
    fake = FakeUpstream(first_delay=2.0)
    app = create_app(fake, GatewaySettings(request_timeout_s=0.1))
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        r = await c.post("/v1/chat/completions", json={**BODY, "stream": False})
    assert r.status_code == 504
    assert app.state.admission.admitted == 0


async def test_disconnect_before_first_token_cancels_upstream_and_frees_slot() -> None:
    fake = FakeUpstream(first_delay=5.0)
    app = create_app(fake, GatewaySettings(max_concurrent=1, max_queue=0))
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        with contextlib.suppress(httpx.TimeoutException):
            await c.post("/v1/chat/completions", json=BODY, timeout=0.2)
        # freed long before the fake's 5 s "prefill" would have finished
        assert await wait_until(lambda: app.state.admission.admitted == 0, timeout=1.0)
        assert fake.cancelled == 1
        m = (await c.get("/metrics")).text
    assert 'localserve_requests_total{outcome="cancelled"} 1.0' in m
