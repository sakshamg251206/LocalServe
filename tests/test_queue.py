import asyncio

import httpx
import pytest

from localserve.gateway.app import GatewaySettings, create_app
from localserve.gateway.queue import Admission, QueueFull

from .conftest import FakeUpstream, serve


async def test_admits_up_to_capacity_then_rejects() -> None:
    adm = Admission(max_concurrent=2, max_queue=1)
    slots = [adm.admit() for _ in range(3)]
    with pytest.raises(QueueFull):
        adm.admit()
    slots[0].release()
    adm.admit()  # capacity freed


async def test_concurrency_is_bounded() -> None:
    adm = Admission(max_concurrent=2, max_queue=5)
    slots = [adm.admit() for _ in range(3)]
    await slots[0].acquire()
    await slots[1].acquire()
    third = asyncio.create_task(slots[2].acquire())
    await asyncio.sleep(0.05)
    assert not third.done()
    assert (adm.inflight, adm.waiting) == (2, 1)
    slots[0].release()
    await asyncio.wait_for(third, 1)
    assert (adm.inflight, adm.waiting) == (2, 0)


async def test_release_is_idempotent_and_works_before_acquire() -> None:
    adm = Admission(max_concurrent=1, max_queue=0)
    s = adm.admit()
    s.release()
    s.release()
    assert adm.admitted == 0
    s2 = adm.admit()
    await s2.acquire()
    s2.release()
    s2.release()
    assert (adm.admitted, adm.inflight) == (0, 0)
    await asyncio.wait_for(adm.admit().acquire(), 1)  # semaphore not over/under-released


async def test_gateway_returns_429_when_queue_full() -> None:
    fake = FakeUpstream(first_delay=0.5)
    app = create_app(fake, GatewaySettings(max_concurrent=1, max_queue=1))
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    async with serve(app) as url, httpx.AsyncClient(base_url=url, timeout=5) as c:
        rs = await asyncio.gather(*(c.post("/v1/chat/completions", json=body) for _ in range(3)))
    codes = sorted(r.status_code for r in rs)
    assert codes == [200, 200, 429]
    rejected = next(r for r in rs if r.status_code == 429)
    assert rejected.headers["retry-after"] == "1"
    assert fake.started == 2
