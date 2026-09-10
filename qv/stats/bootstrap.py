"""Stationary bootstrap (Politis and Romano 1994) with automatic block length.

The IID bootstrap is wrong for return series: it destroys the serial
dependence and volatility clustering that drive most of the interesting risk,
and its intervals come out too narrow as a result. The stationary bootstrap
resamples geometrically-distributed blocks instead, wrapping circularly, which
preserves short-range dependence while keeping the resampled series stationary.

Block length is chosen automatically by the Politis and White (2004) rule with
the Patton, Politis and White (2009) correction, so the caller is not asked to
invent a number they have no basis for.

This is the default source of every confidence interval in the package.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.stats.moments import as_returns_array, automatic_lag_bandwidth
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Estimate

__all__ = [
    "flat_top_lag_window",
    "politis_white_block_length",
    "stationary_bootstrap_indices",
    "BootstrapResult",
    "stationary_bootstrap",
]


def flat_top_lag_window(s: np.ndarray | float) -> np.ndarray:
    """Trapezoidal (flat-top) lag window of Politis and Romano.

    ``1`` for ``|s| <= 1/2``, tapering linearly to ``0`` at ``|s| = 1``. The
    flat top is what gives the resulting spectral estimate its small bias,
    which is the whole reason the automatic block-length rule works.
    """
    a = np.abs(np.asarray(s, dtype=float))
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def _autocovariances(x: np.ndarray, max_lag: int) -> np.ndarray:
    """``gamma_0 .. gamma_max_lag``, biased (divide-by-n) estimator."""
    n = x.size
    centred = x - x.mean()
    return np.array(
        [float(np.dot(centred[k:], centred[: n - k])) / n for k in range(max_lag + 1)]
    )


def politis_white_block_length(returns) -> float:
    """Optimal mean block length for the stationary bootstrap.

    Implements Politis and White (2004) as corrected by Patton, Politis and
    White (2009). In outline: find the lag beyond which the autocorrelation is
    indistinguishable from zero, use twice that as a bandwidth, form flat-top
    kernel estimates of the spectral density at zero and of its curvature, then
    combine them into the block length that minimises asymptotic MSE.

    Returns a float in ``[1, min(3*sqrt(n), n/3)]``. Falls back to 1.0 (which
    makes the stationary bootstrap degenerate to the IID bootstrap) only for a
    series with no estimable variance, where no amount of blocking helps.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"block-length selection needs at least 4 observations, got {n}")

    if float(np.var(x)) <= 0:
        return 1.0

    # Bandwidth: the lag beyond which dependence is indistinguishable from
    # zero. Shared with the Lo annualisation adjustment, which needs to make
    # exactly the same judgement call.
    big_m = automatic_lag_bandwidth(x)

    lags = np.arange(big_m + 1)
    weights = flat_top_lag_window(lags / big_m)
    g = _autocovariances(x, big_m)

    # Two-sided sums: lag 0 counted once, every other lag twice, since the
    # autocovariance sequence is even.
    two_sided = np.concatenate(([1.0], np.full(big_m, 2.0)))
    g_hat = float(np.sum(two_sided * weights * lags * g))
    g_zero = float(np.sum(two_sided * weights * g))

    if g_zero == 0.0 or g_hat == 0.0:
        return 1.0

    d_hat = 2.0 * g_zero**2
    b_opt = ((2.0 * g_hat**2) / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)

    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    return float(min(max(b_opt, 1.0), b_max))


def stationary_bootstrap_indices(
    n: int, block_length: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``(n_boot, n)`` array of circular-wrapped resampling indices.

    Each step either advances one position (probability ``1 - 1/b``) or jumps
    to a fresh uniform position (probability ``1/b``), giving geometrically
    distributed blocks with mean length ``b``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if block_length < 1:
        raise ValueError(f"block_length must be at least 1, got {block_length}")

    p = 1.0 / block_length
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    if n == 1:
        return idx

    starts_new = rng.random((n_boot, n - 1)) < p
    fresh = rng.integers(0, n, size=(n_boot, n - 1))
    for t in range(1, n):
        idx[:, t] = np.where(starts_new[:, t - 1], fresh[:, t - 1], (idx[:, t - 1] + 1) % n)
    return idx


@dataclass(frozen=True)
class BootstrapResult:
    """A bootstrap distribution plus the interval read off it."""

    observed: float
    ci_low: float
    ci_high: float
    distribution: np.ndarray
    block_length: float
    n_boot: int
    n_obs: int
    confidence: float
    reliable: bool
    note: str | None = None

    @property
    def standard_error(self) -> float:
        return float(np.std(self.distribution, ddof=1))

    def percentile_of(self, value: float) -> float:
        """Fraction of the bootstrap distribution at or below ``value``."""
        return float(np.mean(self.distribution <= value))

    def to_estimate(self, method: str) -> Estimate:
        return Estimate(
            value=self.observed,
            ci_low=self.ci_low,
            ci_high=self.ci_high,
            method=method,
            n_obs=self.n_obs,
            reliable=self.reliable,
            note=self.note,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "standard_error": self.standard_error,
            "block_length": self.block_length,
            "n_boot": self.n_boot,
            "n_obs": self.n_obs,
            "confidence": self.confidence,
            "reliable": self.reliable,
            "note": self.note,
        }


def stationary_bootstrap(
    returns,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    block_length: float | None = None,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapResult:
    """Percentile bootstrap interval for any statistic of a return series.

    ``statistic`` is called once per resample with a 1-D array and must return
    a scalar. Resamples for which it returns a non-finite value (a
    zero-variance draw, say) are dropped from the distribution rather than
    poisoning the percentiles.

    ``seed`` defaults to 0 rather than None: every number this package prints
    must be reproducible from the report alone.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"bootstrap needs at least 4 observations, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    b = politis_white_block_length(x) if block_length is None else float(block_length)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, b, n_boot, rng)

    draws = np.array([statistic(x[row]) for row in idx], dtype=float)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("every bootstrap replication produced a non-finite statistic")

    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])

    reliable = n >= MIN_OBS_FOR_ASYMPTOTICS
    notes: list[str] = []
    if not reliable:
        notes.append(
            f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} at which bootstrap "
            "intervals become trustworthy; the resamples cannot manufacture information "
            "the sample does not contain"
        )
    if finite.size < draws.size:
        notes.append(f"{draws.size - finite.size} of {draws.size} replications were non-finite")

    return BootstrapResult(
        observed=float(statistic(x)),
        ci_low=float(lo),
        ci_high=float(hi),
        distribution=finite,
        block_length=b,
        n_boot=n_boot,
        n_obs=n,
        confidence=confidence,
        reliable=reliable,
        note="; ".join(notes) if notes else None,
    )
