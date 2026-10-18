"""Capacity at an SLO, and the interval estimators used in the report."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Capacity:
    rate: float | None  # req/s
    # exact: the next grid rate fails. lower_bound: no grid rate failed (true capacity is higher).
    # below_grid: even the lowest rate failed.
    bound: Literal["exact", "lower_bound", "below_grid"]


def capacity(p95_by_rate: Mapping[float, float], slo_s: float) -> Capacity:
    """Highest grid rate r such that r *and every lower grid rate* meet p95 TTFT <= slo_s.

    Requiring all lower rates to pass makes the estimate conservative: a higher rate that passes
    only by noise after a failure is not counted.
    """
    best: float | None = None
    for r in sorted(p95_by_rate):
        if not p95_by_rate[r] <= slo_s:  # inf/nan fail
            return Capacity(best, "exact" if best is not None else "below_grid")
        best = r
    return Capacity(best, "lower_bound")


def bootstrap_ci(
    x: np.ndarray | list[float],
    stat: Callable[[np.ndarray], float | np.floating],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile-bootstrap (1 - alpha) CI: resample x with replacement, recompute `stat`.

    Assumes observations are independent; for paired A/B timings pass the per-pair differences.
    """
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(a[rng.integers(0, a.size, a.size)]) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(a)), float(lo), float(hi)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than Wald for small n or p near 0/1)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
