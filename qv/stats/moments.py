"""Sample moments and Lo's autocorrelation-adjusted annualisation.

Everything here takes a plain 1-D array of periodic returns and returns floats.
No pandas, no state, no I/O - which is what makes the known-answer tests worth
anything.

A note on conventions, because they are the usual source of silent error:

* ``skewness`` is the method-of-moments estimator (``scipy.stats.skew`` with
  ``bias=True``), not the sample-corrected one.
* ``kurtosis`` is **raw**, so a normal distribution gives 3.0, not 0.0. The
  Sharpe standard-error formula subtracts 3 itself, and passing an excess
  kurtosis there is a mistake that quietly shrinks every error bar.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "as_returns_array",
    "skewness",
    "kurtosis",
    "standardised_moments",
    "autocorrelations",
    "as_trial_matrix",
    "columnwise_sharpe",
    "max_estimable_lag",
    "automatic_lag_bandwidth",
    "lo_annualisation_factor",
    "naive_annualisation_factor",
]


def as_returns_array(returns) -> np.ndarray:
    """Coerce to a finite 1-D float array, dropping NaNs.

    Accepts anything array-like, including a pandas Series, without importing
    pandas. Raises on 2-D input rather than silently flattening it, because a
    silently flattened trial matrix produces a plausible and wrong Sharpe.
    """
    arr = np.asarray(getattr(returns, "to_numpy", lambda: returns)(), dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D series of returns, got shape {arr.shape}")
    return arr[np.isfinite(arr)]


def _central_moment(x: np.ndarray, order: int) -> float:
    return float(np.mean((x - x.mean()) ** order))


def skewness(returns) -> float:
    """Method-of-moments skewness. Returns 0.0 for a degenerate (zero-variance) series."""
    x = as_returns_array(returns)
    if x.size < 3:
        raise ValueError(f"skewness needs at least 3 observations, got {x.size}")
    m2 = _central_moment(x, 2)
    if m2 <= 0:
        return 0.0
    return _central_moment(x, 3) / m2**1.5


def kurtosis(returns) -> float:
    """Raw (non-excess) method-of-moments kurtosis. Normal distribution -> 3.0."""
    x = as_returns_array(returns)
    if x.size < 4:
        raise ValueError(f"kurtosis needs at least 4 observations, got {x.size}")
    m2 = _central_moment(x, 2)
    if m2 <= 0:
        return 3.0
    return _central_moment(x, 4) / m2**2


def standardised_moments(returns) -> tuple[float, float]:
    """``(skewness, raw kurtosis)`` in one pass over the data."""
    x = as_returns_array(returns)
    return skewness(x), kurtosis(x)


def autocorrelations(returns, max_lag: int) -> np.ndarray:
    """Sample autocorrelations ``rho_1 .. rho_max_lag``.

    Uses the biased (divide-by-n) estimator, which is standard in time-series
    work and keeps the autocovariance sequence positive semi-definite - the
    unbiased version can produce an autocorrelation above 1 at long lags on
    short samples, which then makes Lo's adjustment nonsense.
    """
    x = as_returns_array(returns)
    n = x.size
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    if max_lag >= n:
        raise ValueError(f"max_lag={max_lag} needs more than {max_lag} observations, got {n}")
    if max_lag == 0:
        return np.empty(0)

    centred = x - x.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0:
        return np.zeros(max_lag)
    return np.array(
        [float(np.dot(centred[k:], centred[:-k])) / denom for k in range(1, max_lag + 1)]
    )


def as_trial_matrix(trial_returns) -> np.ndarray:
    """Coerce to a 2-D ``(T, N)`` float array of per-trial return columns."""
    matrix = np.asarray(
        getattr(trial_returns, "to_numpy", lambda: trial_returns)(), dtype=float
    )
    if matrix.ndim != 2:
        raise ValueError(f"trial matrix must be 2-D (T x N), got shape {matrix.shape}")
    return matrix


def columnwise_sharpe(trial_returns, rf_per_period: float = 0.0) -> np.ndarray:
    """Per-period Sharpe of every column, with ``nan`` for degenerate columns.

    Testing ``std > 0`` is not enough. The sample standard deviation of a
    column of *identical* floats comes back around ``1e-16`` rather than
    exactly zero, so a strategy that never traded produces a Sharpe near
    ``1e14`` - which then wins every trial selection it is entered into and
    silently poisons the maximum. Checking that the column actually varies
    catches it exactly and without a magic tolerance.
    """
    matrix = as_trial_matrix(trial_returns)
    if matrix.shape[0] < 2:
        raise ValueError(f"need at least 2 rows to compute a Sharpe, got {matrix.shape[0]}")

    means = np.mean(matrix, axis=0) - rf_per_period
    sds = np.std(matrix, axis=0, ddof=1)
    degenerate = (np.ptp(matrix, axis=0) == 0) | ~(sds > 0) | ~np.isfinite(sds)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(degenerate, np.nan, means / np.where(degenerate, 1.0, sds))


def max_estimable_lag(n: int) -> int:
    """The furthest lag this many observations can say anything about.

    ``ceil(sqrt(n)) + K_N``, the ceiling of the Politis-White bandwidth search.
    Beyond it there are too few overlapping pairs for an autocorrelation
    estimate to carry information, so a lag sum reaching past this point is
    asserting something the data cannot speak to.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    k_n = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    return min(int(math.ceil(math.sqrt(n))) + k_n, n - 1)


def automatic_lag_bandwidth(returns) -> int:
    """How many lags this sample can actually support, data-driven.

    Implements the bandwidth search of Politis and White (2004) as corrected by
    Patton, Politis and White (2009): walk out until ``K_N`` consecutive
    autocorrelations all fall inside the band where they are indistinguishable
    from zero, then take twice that lag.

    This matters more than it looks. A sample autocorrelation has a standard
    error of roughly ``1/sqrt(n)``, so estimating 251 of them from 1000
    observations and then weighting them by up to 251 - which a literal reading
    of Lo eq. 9 with ``q=252`` demands - accumulates far more noise than
    signal. Truncating instead asserts that autocorrelation has died out beyond
    the chosen lag, which is both the standard assumption in HAC estimation and
    obviously better than averaging noise.

    Shared with :mod:`qv.stats.bootstrap`, which needs the same bandwidth for
    its flat-top kernel sums.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"bandwidth selection needs at least 4 observations, got {n}")

    log10n = math.log10(n)
    k_n = max(5, int(math.ceil(math.sqrt(log10n))))
    m_max = max_estimable_lag(n)
    threshold = 2.0 * math.sqrt(log10n / n)

    rho = autocorrelations(x, m_max)  # rho[k - 1] is the lag-k autocorrelation
    m_hat = m_max
    for m in range(1, m_max + 1):
        window = rho[m : min(m + k_n, m_max)]
        if window.size == 0 or bool(np.all(np.abs(window) < threshold)):
            m_hat = m
            break

    return max(min(2 * m_hat, m_max), 1)


def naive_annualisation_factor(q: int) -> float:
    """The ``sqrt(q)`` everyone uses. Correct only for serially independent returns."""
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}")
    return math.sqrt(q)


def lo_annualisation_factor(returns, q: int, max_lag: int | None = None) -> float:
    """Lo (2002) eq. 9: the honest replacement for ``sqrt(q)``.

    ``eta(q) = q / sqrt( q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k )``

    Multiply a per-period Sharpe by this to annualise it. With zero
    autocorrelation it collapses exactly to ``sqrt(q)``; with the positive
    autocorrelation typical of overlapping or slow-moving signals it is
    materially smaller, which is why naive annualisation flatters so many
    published Sharpes.

    The lag sum is truncated at :func:`automatic_lag_bandwidth` rather than
    running all the way to ``q - 1``. Without that, annualising a 1000-day
    backtest with ``q=252`` would weight 251 noisy autocorrelation estimates by
    up to 251 apiece and report a haircut driven almost entirely by sampling
    error. Pass ``max_lag`` to override the automatic choice.

    Raises ``ValueError`` if the implied q-period variance is non-positive.
    An untruncated autocovariance sequence is positive semi-definite so this
    cannot happen in population, but truncation can drive the partial sum
    negative, and a negative variance has no meaningful square root.
    """
    x = as_returns_array(returns)
    if q < 1:
        raise ValueError(f"q must be at least 1, got {q}")
    if q == 1:
        return 1.0

    cap = min(q - 1, x.size - 1)
    cap = min(cap, max_lag if max_lag is not None else automatic_lag_bandwidth(x))
    if cap < 1:
        return naive_annualisation_factor(q)

    rho = autocorrelations(x, cap)
    weights = np.array([q - k for k in range(1, cap + 1)], dtype=float)
    denom = q + 2.0 * float(np.dot(weights, rho))
    if denom <= 0:
        raise ValueError(
            "autocorrelation-adjusted q-period variance is non-positive "
            f"({denom:.4g}); the series is too strongly negatively autocorrelated "
            "for Lo's adjustment to apply"
        )
    return q / math.sqrt(denom)
