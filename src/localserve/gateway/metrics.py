"""Per-request record -> Prometheus metrics + one structured (JSON) log line."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from .queue import Admission

log = logging.getLogger("localserve.gateway")

# Latency buckets from 5 ms to 60 s: TTFT for small local models spans tens of ms (idle) to
# many seconds (overloaded), so the histogram needs resolution at both ends.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0, 30.0, 60.0)

OUTCOMES = ("ok", "rejected", "timeout", "cancelled", "upstream_error", "bad_request")


@dataclass
class RequestRecord:
    stream: bool
    model: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    t_arrival: float = field(default_factory=time.perf_counter)
    outcome: str = "ok"
    status: int = 200
    queue_wait_s: float | None = None
    ttft_s: float | None = None  # arrival -> first upstream chunk (includes queue wait)
    e2e_s: float | None = None
    chunks: int = 0  # SSE content events relayed; llama-server sends ~1 token per event

    def elapsed(self) -> float:
        return time.perf_counter() - self.t_arrival


class GatewayMetrics:
    def __init__(self, admission: Admission) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("localserve_requests", "Requests by outcome", ["outcome"], registry=r)
        for o in OUTCOMES:
            self.requests.labels(o)  # export zeros so dashboards see every series
        self.ttft = Histogram(
            "localserve_ttft_seconds", "Arrival to first token", buckets=_BUCKETS, registry=r
        )
        self.e2e = Histogram("localserve_e2e_seconds", "Arrival to last byte", buckets=_BUCKETS, registry=r)
        self.queue_wait = Histogram(
            "localserve_queue_wait_seconds", "Time waiting for a slot", buckets=_BUCKETS, registry=r
        )
        self.chunks = Counter("localserve_output_chunks", "Streamed content events", registry=r)
        Gauge("localserve_inflight", "Requests upstream now", registry=r).set_function(
            lambda: admission.inflight
        )
        Gauge("localserve_queued", "Requests waiting for a slot", registry=r).set_function(
            lambda: admission.waiting
        )

    def finish(self, rec: RequestRecord) -> None:
        rec.e2e_s = rec.elapsed()
        self.requests.labels(rec.outcome).inc()
        if rec.outcome == "ok":
            self.e2e.observe(rec.e2e_s)
            if rec.ttft_s is not None:
                self.ttft.observe(rec.ttft_s)
        if rec.queue_wait_s is not None:
            self.queue_wait.observe(rec.queue_wait_s)
        self.chunks.inc(rec.chunks)
        log.info(json.dumps({k: v for k, v in asdict(rec).items() if k != "t_arrival"}))

    def render(self) -> bytes:
        return generate_latest(self.registry)
