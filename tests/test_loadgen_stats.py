import asyncio
import math

import httpx
import numpy as np
import pytest

from localserve.gateway.app import GatewaySettings, create_app
from localserve.loadgen.arrivals import poisson_arrivals
from localserve.loadgen.client import RequestResult, send_chat
from localserve.loadgen.run import LoadSpec, chat_prompts, run_load
from localserve.loadgen.stats import pct, summarize_run

from .conftest import FakeUpstream, serve


def test_poisson_mean_and_cv() -> None:
    rate, duration = 10.0, 2000.0
    t = poisson_arrivals(rate, duration, np.random.default_rng(0))
    gaps = np.diff(t)
    assert t.min() >= 0 and t.max() < duration and np.all(gaps > 0)
    # n ~ Poisson(rate*duration): mean 20000, sd ~141 -> 4 sd tolerance
    assert abs(len(t) - rate * duration) < 4 * math.sqrt(rate * duration)
    # mean gap 1/rate; sd of the mean ~ (1/rate)/sqrt(n) = 0.0007 -> 1% tolerance is >10 sd
    assert gaps.mean() == pytest.approx(1 / rate, rel=0.01)
    # exponential gaps have coefficient of variation 1 (a fixed-interval "closed" schedule has 0)
    assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.03)


def test_poisson_is_seeded_and_handles_short_windows() -> None:
    a = poisson_arrivals(3.0, 50.0, np.random.default_rng(7))
    b = poisson_arrivals(3.0, 50.0, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert len(poisson_arrivals(0.01, 1.0, np.random.default_rng(0))) <= 1
    with pytest.raises(ValueError):
        poisson_arrivals(0.0, 10.0, np.random.default_rng(0))


@pytest.mark.parametrize("n", [1, 2, 7, 100, 1001])
def test_pct_matches_numpy_nearest_rank(n: int) -> None:
    x = np.random.default_rng(n).lognormal(size=n)
    for q in (50, 90, 95, 99):
        assert pct(x, q) == np.percentile(x, q, method="inverted_cdf")


def test_pct_hand_case_and_inf() -> None:
    x = [5.0, 1.0, 4.0, 2.0, 3.0]
    # nearest rank: ceil(q/100 * n)-th smallest
    assert pct(x, 50) == 3.0
    assert pct(x, 95) == 5.0
    assert pct(x, 20) == 1.0
    assert pct([1.0, 2.0, math.inf, math.inf], 95) == math.inf
    assert pct([1.0, 2.0, 3.0, math.inf], 50) == 2.0
    assert math.isnan(pct([], 50))


def _res(t: float, ok: bool, ttft: float | None, status: int = 200, tokens: int = 10) -> RequestResult:
    return RequestResult(
        idx=0,
        t_sched=t,
        t_sent=t,
        status=status if ok else (status or 0),
        error=None if ok else "x",
        ttft_s=ttft,
        itl_s=[0.02] * (tokens - 1) if ok else [],
        e2e_s=(ttft or 0) + 0.02 * (tokens - 1),
        n_events=tokens - 1 if ok else 0,  # one event carried two tokens
        completion_tokens=tokens if ok else None,
        prompt_tokens=200,
    )


def test_summary_excludes_warmup_and_counts_failures_against_slo() -> None:
    rs = [_res(1.0, True, 9.9)]  # warm-up: excluded
    rs += [_res(10 + i, True, 0.1 * (i + 1)) for i in range(18)]  # ttft 0.1..1.8
    rs += [_res(30.0, False, None, status=429), _res(31.0, False, None, status=0)]
    s = summarize_run(rs, warmup_s=5, duration_s=40)
    assert s["n_scheduled"] == 20
    assert (s["n_ok"], s["n_rejected"], s["n_failed"]) == (18, 1, 1)
    assert s["error_rate"] == pytest.approx(2 / 20)
    assert s["offered_rps"] == pytest.approx(20 / 40)
    assert s["output_tok_s"] == pytest.approx(18 * 10 / 40)
    ttft_ok = [0.1 * (i + 1) for i in range(18)]
    assert s["ttft_p95_s"] == pytest.approx(pct(ttft_ok, 95))
    # with 2 of 20 failed (inf), nearest-rank p95 = 19th smallest = inf
    assert s["ttft_p95_slo_s"] == math.inf
    assert s["itl_p50_s"] == pytest.approx(0.02)
    assert s["tpot_p50_s"] == pytest.approx(0.02)  # (e2e - ttft) / (10 - 1)
    assert s["output_tokens_mean"] == 10  # from usage, not the 9 events


async def test_client_measures_ttft_itl_tokens_against_fake() -> None:
    fake = FakeUpstream(tokens=tuple("abcdef"), first_delay=0.2, token_delay=0.05)
    async with serve(create_app(fake, GatewaySettings())) as url, httpx.AsyncClient(base_url=url) as c:
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
        r = await send_chat(c, "/v1/chat/completions", body, timeout_s=5)
    assert r.ok and r.status == 200
    assert r.n_events == 6 and r.n_tokens == 6 and len(r.itl_s) == 5  # fake sends no usage chunk
    assert r.tpot_s is not None and 0.04 <= r.tpot_s < 0.12
    assert r.ttft_s is not None and 0.2 <= r.ttft_s < 0.35
    assert all(0.04 <= g < 0.12 for g in r.itl_s)
    assert r.e2e_s >= r.ttft_s + 0.25


async def test_client_records_429_and_timeout() -> None:
    fake = FakeUpstream(first_delay=1.0)
    app = create_app(fake, GatewaySettings(max_concurrent=1, max_queue=0))
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}
    async with serve(app) as url, httpx.AsyncClient(base_url=url) as c:
        slow = await send_chat(c, "/v1/chat/completions", body, timeout_s=0.2)
        assert (slow.ok, slow.status, slow.error) == (False, 0, "client_timeout")
        # the timed-out client hung up; the gateway frees its slot, then we fill it again
        busy = asyncio.create_task(send_chat(c, "/v1/chat/completions", body, timeout_s=5))
        await asyncio.sleep(0.1)
        rej = await send_chat(c, "/v1/chat/completions", body, timeout_s=5)
        assert (rej.ok, rej.status) == (False, 429)
        assert (await busy).ok


async def test_run_load_open_loop_against_fake() -> None:
    fake = FakeUpstream(tokens=tuple("abcd"), first_delay=0.05, token_delay=0.01)
    async with serve(create_app(fake, GatewaySettings(max_concurrent=64))) as url:
        spec = LoadSpec(base_url=url, model="m", rate=20.0, duration_s=1.5, warmup_s=0.5, seed=3)
        results, resources = await run_load(spec)
    expected = poisson_arrivals(20.0, 2.0, np.random.default_rng(3))
    assert len(results) == len(expected)
    assert np.allclose([r.t_sched for r in results], expected)
    assert all(r.ok for r in results)
    # open loop: send times follow the schedule (not completions) to within event-loop jitter
    assert max(r.t_sent - r.t_sched for r in results) < 0.05
    s = summarize_run(results, spec.warmup_s, spec.duration_s)
    assert s["n_scheduled"] == sum(1 for t in expected if t >= 0.5)
    assert resources["mem_available_min_bytes"] > 0


def test_chat_prompts_unique_and_reproducible() -> None:
    a = chat_prompts(50, words=150, seed=1)
    assert len(set(a)) == 50  # unique from the first token: no accidental shared-prefix cache hits
    assert len({p.split()[0] + p.split()[1] for p in a}) == 50
    assert a == chat_prompts(50, words=150, seed=1)
    assert all(140 <= len(p.split()) <= 175 for p in a)
