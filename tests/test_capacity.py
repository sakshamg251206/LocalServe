import math

import numpy as np
import pytest

from localserve.capacity import Capacity, bootstrap_ci, capacity, wilson_ci


def test_capacity_is_highest_rate_with_all_lower_rates_passing() -> None:
    p95 = {0.1: 0.3, 0.2: 0.5, 0.3: 0.9, 0.4: 1.4, 0.5: 0.8}  # 0.5 passes only by noise
    assert capacity(p95, slo_s=1.0) == Capacity(rate=0.3, bound="exact")


def test_capacity_lower_bound_when_grid_never_fails() -> None:
    assert capacity({0.1: 0.2, 0.2: 0.4}, slo_s=1.0) == Capacity(rate=0.2, bound="lower_bound")


def test_capacity_below_grid_when_lowest_rate_fails() -> None:
    assert capacity({0.1: 1.2, 0.2: 2.0}, slo_s=1.0) == Capacity(rate=None, bound="below_grid")


def test_capacity_boundary_is_inclusive_and_inf_fails() -> None:
    assert capacity({0.1: 1.0, 0.2: math.inf}, slo_s=1.0) == Capacity(rate=0.1, bound="exact")


def test_bootstrap_ci_covers_known_median_shift() -> None:
    rng = np.random.default_rng(0)
    d = 0.004 + rng.normal(0, 0.002, size=200)  # true median 4 ms
    est, lo, hi = bootstrap_ci(d, np.median, seed=1)
    assert lo < 0.004 < hi
    assert est == pytest.approx(np.median(d))
    assert hi - lo < 0.002
    assert bootstrap_ci(d, np.median, seed=1) == (est, lo, hi)  # seeded


def test_wilson_ci_known_values() -> None:
    # Wilson 95% interval for 8/10 (textbook value approx [0.490, 0.943])
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    lo0, hi0 = wilson_ci(0, 20)
    assert lo0 == 0.0 and 0 < hi0 < 0.2  # unlike the Wald interval, not degenerate at 0
