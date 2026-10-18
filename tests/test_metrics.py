import asyncio
import json
import logging

import httpx
import pytest

from localserve.gateway.app import GatewaySettings, create_app

from .conftest import FakeUpstream, serve

BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    raise KeyError(name)


async def test_metrics_counts_outcomes_and_latencies(caplog: pytest.LogCaptureFixture) -> None:
    fake = FakeUpstream(tokens=("a", "b", "c"), first_delay=0.3)
    app = create_app(fake, GatewaySettings(max_concurrent=1, max_queue=1))
    caplog.set_level(logging.INFO, logger="localserve.gateway")
    async with serve(app) as url, httpx.AsyncClient(base_url=url, timeout=5) as c:
        rs = await asyncio.gather(*(c.post("/v1/chat/completions", json=BODY) for _ in range(3)))
        m = (await c.get("/metrics")).text
    assert sorted(r.status_code for r in rs) == [200, 200, 429]
    assert metric(m, 'localserve_requests_total{outcome="ok"}') == 2
    assert metric(m, 'localserve_requests_total{outcome="rejected"}') == 1
    assert metric(m, "localserve_ttft_seconds_count") == 2
    assert metric(m, "localserve_output_chunks_total") == 6
    assert metric(m, "localserve_inflight") == 0
    # the second admitted request waited ~0.3 s for the single slot
    assert metric(m, "localserve_queue_wait_seconds_sum") >= 0.25

    logs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "localserve.gateway"]
    assert sorted(entry["outcome"] for entry in logs) == ["ok", "ok", "rejected"]
    ok = [e for e in logs if e["outcome"] == "ok"]
    assert all(e["ttft_s"] >= 0.3 and e["chunks"] == 3 for e in ok)
    assert len({e["request_id"] for e in logs}) == 3
