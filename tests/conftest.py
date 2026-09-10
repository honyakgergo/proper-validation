"""Shared fixtures.

Every generator here is seeded. A validator whose own test suite is flaky has
no business telling anyone else their results are not reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20240101)


def make_ar1(n: int, phi: float, seed: int = 0, scale: float = 1.0, mean: float = 0.0):
    """AR(1) series with autocorrelation ``phi**k`` at lag k.

    Burns in 500 observations so the process starts stationary rather than at
    its unconditional mean, which would bias the early autocovariances.
    """
    gen = np.random.default_rng(seed)
    burn = 500
    innovations = gen.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + innovations[t]
    x = x[burn:]
    # Scale to unit innovation variance -> unconditional sd sqrt(1/(1-phi^2)).
    return mean + scale * x


def make_returns(n: int, sharpe_per_period: float, seed: int = 0, sd: float = 0.01):
    """IID normal returns with a specified *true* per-period Sharpe ratio."""
    gen = np.random.default_rng(seed)
    return sharpe_per_period * sd + sd * gen.standard_normal(n)


@pytest.fixture
def iid_returns() -> np.ndarray:
    return make_returns(1000, 0.05, seed=11)


@pytest.fixture
def trial_matrix() -> np.ndarray:
    """``(T, N)`` matrix of pure-noise trial returns - the mined-noise setting."""
    gen = np.random.default_rng(99)
    return 0.01 * gen.standard_normal((750, 200))
