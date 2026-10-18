"""Per-run summary statistics.

Percentiles use the nearest-rank definition (numpy method="inverted_cdf"): the p-th percentile
is the ceil(p/100 * n)-th smallest observation. It always returns a real observation, and it
stays correct when failed requests are counted as infinite latency (interpolation gives NaN).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .client import RequestResult


def pct(x: Sequence[float] | np.ndarray, q: float) -> float:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return math.nan
    return float(np.percentile(a, q, method="inverted_cdf"))


def summarize_run(results: Sequence[RequestResult], warmup_s: float, duration_s: float) -> dict[str, Any]:
    """Summarize requests *scheduled* in [warmup_s, warmup_s + duration_s).

    Selecting by scheduled arrival (not completion) keeps the measured window an unbiased slice of
    the offered load. Rejections (429), timeouts and errors count as SLO misses: `ttft_p95_slo_s`
    treats them as infinite TTFT, so a server cannot pass the SLO by dropping its slow requests.
    """
    win = [r for r in results if warmup_s <= r.t_sched < warmup_s + duration_s]
    ok = [r for r in win if r.ok]
    ttft_ok = [r.ttft_s for r in ok if r.ttft_s is not None]
    ttft_all = [r.ttft_s if (r.ok and r.ttft_s is not None) else math.inf for r in win]
    itl = [g for r in ok for g in r.itl_s]
    e2e = [r.e2e_s for r in ok]
    tpot = [t for r in ok if (t := r.tpot_s) is not None]
    prompt_toks = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
    n = len(win)
    return {
        "n_scheduled": n,
        "n_ok": len(ok),
        "n_rejected": sum(1 for r in win if r.status == 429),
        "n_failed": sum(1 for r in win if not r.ok and r.status != 429),
        "error_rate": (n - len(ok)) / n if n else math.nan,
        "offered_rps": n / duration_s,
        "goodput_rps": len(ok) / duration_s,
        "output_tok_s": sum(r.n_tokens for r in ok) / duration_s,
        "ttft_p50_s": pct(ttft_ok, 50),
        "ttft_p95_s": pct(ttft_ok, 95),
        "ttft_p99_s": pct(ttft_ok, 99),
        "ttft_p95_slo_s": pct(ttft_all, 95),
        "itl_p50_s": pct(itl, 50),
        "itl_p95_s": pct(itl, 95),
        "tpot_p50_s": pct(tpot, 50),
        "tpot_p95_s": pct(tpot, 95),
        "e2e_p50_s": pct(e2e, 50),
        "e2e_p95_s": pct(e2e, 95),
        "output_tokens_mean": float(np.mean([r.n_tokens for r in ok])) if ok else math.nan,
        "prompt_tokens_mean": float(np.mean(prompt_toks)) if prompt_toks else math.nan,
        "max_send_lag_s": max((r.t_sent - r.t_sched for r in win), default=math.nan),
    }
