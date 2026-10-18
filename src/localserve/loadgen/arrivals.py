"""Open-loop arrival schedules.

Open loop = requests are sent at pre-computed times regardless of whether earlier ones have
finished. A closed loop (N clients, each waiting for its reply before sending again) slows its
own sending when the server slows down, which hides overload ("coordinated omission"). Real
users don't wait for each other, so an open-loop schedule is the honest way to find capacity.

A Poisson process with rate λ has independent exponential gaps with mean 1/λ: the standard model
for many independent users each arriving at random.
"""

from __future__ import annotations

import math

import numpy as np


def poisson_arrivals(rate: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Arrival offsets (seconds, sorted) in [0, duration_s) of a Poisson process with `rate`/s."""
    if rate <= 0 or duration_s <= 0:
        raise ValueError("rate and duration_s must be positive")
    expected = rate * duration_s
    n = int(expected + 10 * math.sqrt(expected) + 10)  # overshoot so one draw almost always suffices
    t = np.cumsum(rng.exponential(1.0 / rate, n))
    while t[-1] < duration_s:
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / rate, n))])
    return t[t < duration_s]
