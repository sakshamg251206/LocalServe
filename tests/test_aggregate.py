import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from localserve.loadgen.client import RequestResult
from localserve.report.aggregate import (
    Point,
    bench_row,
    capacity_rows,
    overhead_serial,
    pool,
    read_phase,
)

SPEC = {"warmup_s": 0.0, "duration_s": 10.0}


def req(ttft: float | None, ok: bool = True, tokens: int = 5) -> RequestResult:
    return RequestResult(
        idx=0,
        t_sched=1.0,
        t_sent=1.0,
        status=200 if ok else 0,
        error=None if ok else "client_timeout",
        ttft_s=ttft,
        itl_s=[0.02] * (tokens - 1) if ok else [],
        e2e_s=(ttft or 0) + 0.02 * (tokens - 1),
        n_events=tokens,
        completion_tokens=tokens if ok else None,
    )


def point(quant: str, rate: float, ttfts: Sequence[float | None], rep: int = 0) -> Point:
    meta = {"quant": quant, "parallel": 4, "rate": rate, "rep": rep, "spec": SPEC, "swap_out_bytes": rep}
    return Point(meta, [req(t, ok=t is not None) for t in ttfts])


def test_pool_pools_requests_and_reports_spread() -> None:
    p1 = point("Q4", 0.1, [0.1, 0.2, 0.3, 0.4], rep=0)
    p2 = point("Q4", 0.1, [0.5, 0.6, 0.7, None], rep=1)  # one failure -> inf
    s = pool([p1, p2])
    assert (s["n_runs"], s["n_requests"], s["n_ok"]) == (2, 8, 7)
    assert s["error_rate"] == pytest.approx(1 / 8)
    assert s["ttft_p95_slo_s"] == math.inf  # nearest rank: 8th of 8
    assert s["ttft_p95_slo_run_min_s"] == pytest.approx(0.4)
    assert s["ttft_p95_slo_run_max_s"] == math.inf
    assert s["ttft_p50_s"] == pytest.approx(0.4)  # successes only: 4th of 7
    assert s["goodput_rps_mean"] == pytest.approx((4 + 3) / 2 / 10)
    assert s["output_tok_s_mean"] == pytest.approx((20 + 15) / 2 / 10)
    assert s["swap_out_bytes_max"] == 1


def test_capacity_rows_end_to_end() -> None:
    pts = [
        point("Q4", 0.1, [0.2] * 20),
        point("Q4", 0.2, [0.5] * 20),
        point("Q4", 0.3, [0.5] * 18 + [3.0, 3.0]),  # p95 = 3.0 -> fails 1.0 s SLO
        point("Q8", 0.1, [1.5] * 20),  # fails at the lowest rate
    ]
    curve, caps = capacity_rows(pts, slo_s=1.0)
    assert len(curve) == 4
    assert {(c["quant"], c["capacity_rps"], c["bound"]) for c in caps} == {
        ("Q4", 0.2, "exact"),
        ("Q8", None, "below_grid"),
    }


def test_overhead_serial_paired_difference() -> None:
    rows = []
    for i in range(50):
        base = 0.3 + 0.01 * (i % 7)  # pair-to-pair variation cancels in the difference
        rows.append(
            {"pair": i, "target": "direct", "status": 200, "error": None, "ttft_s": base, "e2e_s": base}
        )
        g = base + 0.003
        rows.append({"pair": i, "target": "gateway", "status": 200, "error": None, "ttft_s": g, "e2e_s": g})
    rows.append({"pair": 99, "target": "direct", "status": 0, "error": "x", "ttft_s": None, "e2e_s": 1.0})
    o = overhead_serial(rows)
    assert o["n_pairs"] == 50
    assert o["ttft_s"]["diff_median"] == pytest.approx(0.003)
    lo, hi = o["ttft_s"]["diff_ci95"]
    assert lo == pytest.approx(0.003) and hi == pytest.approx(0.003)


def test_bench_row() -> None:
    j = {
        "results": [
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 900.0, "stddev_ts": 5.0, "model_size": 123},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 50.0, "stddev_ts": 0.5, "model_size": 123},
        ]
    }
    assert bench_row(j) == {
        "pp_tok_s": 900.0,
        "pp_tok_s_sd": 5.0,
        "tg_tok_s": 50.0,
        "tg_tok_s_sd": 0.5,
        "model_size_bytes": 123,
    }


def test_read_phase_roundtrip(tmp_path: Path) -> None:
    d = tmp_path / "capacity"
    d.mkdir()
    r = req(0.2)
    lines = [json.dumps({"meta": {"quant": "Q4", "spec": SPEC}}), json.dumps({**r.__dict__, "extra": 1})]
    (d / "a.jsonl").write_text("\n".join(lines) + "\n")
    (d / "failures.jsonl").write_text('{"error": "x"}\n')
    [p] = read_phase(tmp_path, "capacity")
    assert p.results[0].ttft_s == 0.2 and p.meta["quant"] == "Q4"
