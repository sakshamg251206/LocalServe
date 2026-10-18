"""Drive one open-loop load run against an OpenAI-compatible endpoint.

CLI: uv run python -m localserve.loadgen.run --url http://127.0.0.1:8000 --rate 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import psutil

from .arrivals import poisson_arrivals
from .client import RequestResult, send_chat
from .stats import summarize_run

_VOCAB = (  # noqa: SIM905
    "account order delivery refund invoice payment shipment customer support billing address "
    "warehouse product return policy exchange delay tracking package courier update status "
    "issue service request response manager team schedule meeting report quarter revenue "
    "budget forecast client contract renewal discount price quote estimate approval review "
    "feedback survey rating network outage server latency database backup restore migration "
    "release deploy rollback bug ticket priority escalation resolution workaround patch "
    "version license subscription upgrade monday tuesday morning evening week month urgent "
    "normal minor major broken missing late early phone email chat portal website mobile app "
    "login password reset access permission error "
).split()


def chat_prompts(n: int, words: int, seed: int) -> list[str]:
    """`n` synthetic chat prompts of `words` words, each unique from its second word on.

    Uniqueness matters: llama-server reuses a slot's KV cache for a shared prompt *prefix*
    (cache_prompt), so identical prompt openings would make prefill — and TTFT — look cheaper
    than real traffic. The shared-prefix case is measured separately (RAG experiment).
    """
    rng = np.random.default_rng(seed)
    header = ["summarize", "the", "customer", "notes", "below", "in", "three", "short", "bullet", "points:"]
    out = []
    for i in range(n):
        tag = f"{i:05d}-{rng.integers(16**6):06x}:"
        body = rng.choice(_VOCAB, size=words - len(header) - 2)
        out.append(" ".join(["Ticket", tag, *header, *body]))
    return out


@dataclass(frozen=True)
class LoadSpec:
    base_url: str
    model: str
    rate: float  # mean arrivals per second
    duration_s: float = 60.0  # measured window
    warmup_s: float = 15.0  # sent but excluded from stats
    max_tokens: int = 128
    prompt_words: int = 150  # ~200 tokens after Qwen3's chat template (measured per run)
    request_timeout_s: float = 60.0
    seed: int = 0
    path: str = "/v1/chat/completions"

    def body(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Fixed output length: every config generates exactly max_tokens, so speed differences
            # are not confounded by one quant writing longer answers than another.
            "ignore_eos": True,
            # Qwen3 non-thinking mode (needs llama-server --jinja); output content is irrelevant here.
            "chat_template_kwargs": {"enable_thinking": False},
        }


async def _sample_memory(pid: int | None, out: dict[str, float], stop: asyncio.Event) -> None:
    proc = psutil.Process(pid) if pid else None
    out["rss_peak_bytes"] = 0.0
    out["mem_available_min_bytes"] = float(psutil.virtual_memory().available)
    while not stop.is_set():
        if proc is not None:
            try:
                out["rss_peak_bytes"] = max(out["rss_peak_bytes"], float(proc.memory_info().rss))
            except psutil.Error:
                proc = None
        out["mem_available_min_bytes"] = min(
            out["mem_available_min_bytes"], float(psutil.virtual_memory().available)
        )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.5)


async def run_load(
    spec: LoadSpec, server_pid: int | None = None
) -> tuple[list[RequestResult], dict[str, float]]:
    rng = np.random.default_rng(spec.seed)
    offsets = poisson_arrivals(spec.rate, spec.warmup_s + spec.duration_s, rng)
    prompts = chat_prompts(len(offsets), spec.prompt_words, spec.seed)
    resources: dict[str, float] = {}
    stop = asyncio.Event()
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(base_url=spec.base_url, timeout=None, limits=limits) as client:
        sampler = asyncio.create_task(_sample_memory(server_pid, resources, stop))
        t0 = time.perf_counter()
        tasks = []
        for i, off in enumerate(offsets):
            delay = t0 + float(off) - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            coro = send_chat(
                client, spec.path, spec.body(prompts[i]), spec.request_timeout_s, i, float(off), t0
            )
            tasks.append(asyncio.create_task(coro))
        results = list(await asyncio.gather(*tasks))  # drain: each bounded by request_timeout_s
        stop.set()
        await sampler
    return results, resources


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="local")
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=15.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pid", type=int, help="server pid for RSS sampling")
    a = ap.parse_args()
    spec = LoadSpec(a.url, a.model, a.rate, a.duration, a.warmup, a.max_tokens, seed=a.seed)
    results, res = asyncio.run(run_load(spec, a.pid))
    print(json.dumps({**summarize_run(results, spec.warmup_s, spec.duration_s), **res}, indent=2))


if __name__ == "__main__":
    main()
