"""Slice sampling helpers."""

from __future__ import annotations

import numpy as np


def slice_sample_scalar(
    x0: float,
    logpdf,
    rng: np.random.Generator,
    w: float = 1.0,
    m: int = 25,
) -> float:
    """Stepping-out slice sampler for unconstrained scalar targets."""
    logy = logpdf(x0) - rng.exponential(1.0)
    u = rng.random()
    L = x0 - w * u
    R = L + w
    j = int(rng.integers(0, m))
    k = m - 1 - j
    while j > 0 and np.isfinite(logpdf(L)) and logpdf(L) > logy:
        L -= w
        j -= 1
    while k > 0 and np.isfinite(logpdf(R)) and logpdf(R) > logy:
        R += w
        k -= 1

    while True:
        x1 = rng.uniform(L, R)
        if logpdf(x1) >= logy:
            return float(x1)
        if x1 < x0:
            L = x1
        else:
            R = x1


def slice_sample_positive(
    x0: float,
    logpdf,
    rng: np.random.Generator,
    w: float = 0.3,
    m: int = 30,
    lower: float = 1e-8,
) -> float:
    """Slice sample a scalar constrained to (lower, inf)."""
    x0 = max(x0, lower)

    def lp(x: float) -> float:
        if x <= lower:
            return -np.inf
        return float(logpdf(x))

    logy = lp(x0) - rng.exponential(1.0)
    u = rng.random()
    L = max(lower, x0 - w * u)
    R = x0 + w * (1.0 - u)
    j = int(rng.integers(0, m))
    k = m - 1 - j

    while j > 0 and lp(L) > logy and L > lower:
        L = max(lower, L - w)
        j -= 1
    while k > 0 and lp(R) > logy:
        R += w
        k -= 1

    while True:
        x1 = rng.uniform(L, R)
        if lp(x1) >= logy:
            return float(x1)
        if x1 < x0:
            L = x1
        else:
            R = x1
