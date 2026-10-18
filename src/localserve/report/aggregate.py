"""Turn raw per-request JSONL into the numbers the report shows. Pure functions; no plotting."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from localserve.capacity import bootstrap_ci, capacity
from localserve.loadgen.client import RequestResult
from localserve.loadgen.stats import in_window, pct, summarize_run, ttft_for_slo

_RESULT_FIELDS = {f.name for f in fields(RequestResult)}


@dataclass
class Point:
    """One measured run: metadata + every request."""

    meta: dict[str, Any]
    results: list[RequestResult]

    @property
    def window(self) -> list[RequestResult]:
        spec = self.meta["spec"]
        return in_window(self.results, spec["warmup_s"], spec["duration_s"])

    def summary(self) -> dict[str, Any]:
        spec = self.meta["spec"]
        return summarize_run(self.results, spec["warmup_s"], spec["duration_s"])


def read_point(path: Path) -> Point:
    lines = path.read_text().splitlines()
    meta = json.loads(lines[0])["meta"]
    results = [
        RequestResult(**{k: v for k, v in json.loads(x).items() if k in _RESULT_FIELDS}) for x in lines[1:]
    ]
    return Point(meta, results)


def read_phase(raw: Path, phase: str) -> list[Point]:
    return [read_point(p) for p in sorted((raw / phase).glob("*.jsonl")) if p.name != "failures.jsonl"]


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    if not xs:
        return math.nan, math.nan
    return float(np.mean(xs)), (statistics.stdev(xs) if len(xs) > 1 else math.nan)


def pool(points: Sequence[Point]) -> dict[str, Any]:
    """Aggregate the repeated runs of one (config, rate). Latency percentiles pool all requests from
    all runs; throughput is averaged per run; per-run p95 min/max shows run-to-run spread."""
    wins = [p.window for p in points]
    allreq = [r for w in wins for r in w]
    ok = [r for r in allreq if r.ok]
    per_run = [p.summary() for p in points]
    goodput = _mean_sd([s["goodput_rps"] for s in per_run])
    toks = _mean_sd([s["output_tok_s"] for s in per_run])
    run_p95 = [pct(ttft_for_slo(w), 95) for w in wins]
    return {
        "n_runs": len(points),
        "n_requests": len(allreq),
        "n_ok": len(ok),
        "error_rate": (len(allreq) - len(ok)) / len(allreq) if allreq else math.nan,
        "offered_rps_mean": float(np.mean([s["offered_rps"] for s in per_run])),
        "goodput_rps_mean": goodput[0],
        "goodput_rps_sd": goodput[1],
        "output_tok_s_mean": toks[0],
        "output_tok_s_sd": toks[1],
        "ttft_p95_slo_s": pct(ttft_for_slo(allreq), 95),
        "ttft_p95_slo_run_min_s": min(run_p95) if run_p95 else math.nan,
        "ttft_p95_slo_run_max_s": max(run_p95) if run_p95 else math.nan,
        "ttft_p50_s": pct([r.ttft_s for r in ok if r.ttft_s is not None], 50),
        "ttft_p95_s": pct([r.ttft_s for r in ok if r.ttft_s is not None], 95),
        "itl_p50_s": pct([g for r in ok for g in r.itl_s], 50),
        "itl_p95_s": pct([g for r in ok for g in r.itl_s], 95),
        "tpot_p50_s": pct([t for r in ok if (t := r.tpot_s) is not None], 50),
        "e2e_p95_s": pct([r.e2e_s for r in ok], 95),
        "prompt_tokens_mean": float(np.mean(pt))
        if (pt := [r.prompt_tokens for r in ok if r.prompt_tokens])
        else math.nan,
        "swap_out_bytes_max": max(p.meta.get("swap_out_bytes", 0) for p in points),
        "footprint_peak_bytes_max": max(p.meta.get("phys_footprint_peak_bytes", 0) for p in points),
        "rss_peak_bytes_max": max(p.meta.get("rss_peak_bytes", 0) for p in points),
        "mem_available_min_bytes": min(p.meta.get("mem_available_min_bytes", math.inf) for p in points),
    }


def group(points: Iterable[Point], keys: Sequence[str]) -> dict[tuple[Any, ...], list[Point]]:
    out: dict[tuple[Any, ...], list[Point]] = defaultdict(list)
    for p in points:
        out[tuple(p.meta[k] for k in keys)].append(p)
    return dict(out)


def capacity_rows(points: Sequence[Point], slo_s: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (curve rows per quant/parallel/rate, capacity rows per quant/parallel)."""
    curve = []
    for (q, n, r), pts in sorted(group(points, ("quant", "parallel", "rate")).items()):
        curve.append({"quant": q, "parallel": n, "rate": r, **pool(pts)})
    caps = []
    for (q, n), _ in sorted(group(points, ("quant", "parallel")).items()):
        rows = [c for c in curve if c["quant"] == q and c["parallel"] == n]
        cap = capacity({c["rate"]: c["ttft_p95_slo_s"] for c in rows}, slo_s)
        caps.append({"quant": q, "parallel": n, "capacity_rps": cap.rate, "bound": cap.bound,
                     "min_runs": min(c["n_runs"] for c in rows)})  # fmt: skip
    return curve, caps


def overhead_serial(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Paired A/B on an idle server: per pair, gateway minus direct. Median + bootstrap 95% CI."""
    by_pair: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r["status"] == 200 and r["error"] is None:
            by_pair[r["pair"]][r["target"]] = r
    pairs = [v for v in by_pair.values() if len(v) == 2]
    out: dict[str, Any] = {"n_pairs": len(pairs)}
    for metric in ("ttft_s", "e2e_s"):
        d = [p["gateway"][metric] - p["direct"][metric] for p in pairs]
        med, lo, hi = bootstrap_ci(d, np.median) if d else (math.nan,) * 3
        out[metric] = {
            "direct_median": float(np.median([p["direct"][metric] for p in pairs])) if pairs else math.nan,
            "gateway_median": float(np.median([p["gateway"][metric] for p in pairs])) if pairs else math.nan,
            "diff_median": med,
            "diff_ci95": [lo, hi],
        }
    return out


def overhead_openloop(points: Sequence[Point]) -> list[dict[str, Any]]:
    rows = []
    for (r, target), pts in sorted(group(points, ("rate", "target")).items()):
        rows.append({"rate": r, "target": target, **pool(pts)})
    return rows


def bench_row(bench_json: dict[str, Any]) -> dict[str, Any]:
    """llama-bench JSON -> prompt-processing and generation tok/s (mean, sd over 5 repetitions)."""
    out: dict[str, Any] = {}
    for e in bench_json["results"]:
        kind = "pp" if e["n_gen"] == 0 else "tg"
        out[f"{kind}_tok_s"] = e["avg_ts"]
        out[f"{kind}_tok_s_sd"] = e["stddev_ts"]
        out["model_size_bytes"] = e["model_size"]
    return out
