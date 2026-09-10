"""Descriptive performance metrics.

The numbers every backtest reports, computed once and consistently so the
strategy and its benchmark are always measured the same way. Nothing here
attempts inference - these are summaries, and the rest of the package exists
to ask whether they mean anything.

Sortino is included because it is what a strategy with negative skew reports
when the Sharpe is unflattering, so an auditor should see both side by side.
Maximum drawdown is computed on the compounded path rather than the cumulative
sum, since that is the loss an investor would actually have lived through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.stats.moments import as_returns_array
from qv.stats.sharpe import sharpe_ratio

__all__ = [
    "annualised_return",
    "annualised_volatility",
    "downside_deviation",
    "sortino_ratio",
    "max_drawdown",
    "PerformanceSummary",
    "summarise_performance",
]


def annualised_return(returns, periods_per_year: int = 252) -> float:
    """Geometric annualised return from periodic simple returns.

    Compounded, not the arithmetic mean scaled up. The two differ by roughly
    half the variance, which for a volatile strategy is not a rounding error -
    and the compounded figure is the one an investor experiences.

    Returns ``nan`` if the equity path goes non-positive, since there is no
    meaningful annualised return through a total loss.
    """
    x = as_returns_array(returns)
    if x.size == 0:
        return math.nan
    growth = float(np.prod(1.0 + x))
    if growth <= 0:
        return math.nan
    return growth ** (periods_per_year / x.size) - 1.0


def annualised_volatility(returns, periods_per_year: int = 252) -> float:
    """Standard deviation of periodic returns, scaled by ``sqrt(q)``.

    The naive scaling is used deliberately here: this is the descriptive
    number conventionally reported, and the autocorrelation-adjusted treatment
    belongs to the Sharpe, where the report shows both.
    """
    x = as_returns_array(returns)
    if x.size < 2:
        return math.nan
    return float(np.std(x, ddof=1)) * math.sqrt(periods_per_year)


def downside_deviation(returns, target: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised standard deviation of returns below ``target``.

    Divides by the full sample size rather than the count of downside periods,
    which is the convention in the Sortino literature: a strategy with few
    losing periods should be rewarded for that, and dividing by the downside
    count alone would erase the advantage.
    """
    x = as_returns_array(returns)
    if x.size < 2:
        return math.nan
    shortfall = np.minimum(x - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2))) * math.sqrt(periods_per_year)


def sortino_ratio(returns, target: float = 0.0, periods_per_year: int = 252) -> float:
    """Excess return over ``target`` divided by downside deviation, annualised."""
    x = as_returns_array(returns)
    if x.size < 2:
        return math.nan
    dd = downside_deviation(x, target, periods_per_year)
    if not math.isfinite(dd) or dd <= 0:
        return math.nan
    return (float(np.mean(x) - target) * periods_per_year) / dd


def max_drawdown(returns) -> tuple[float, int, int]:
    """Worst peak-to-trough loss on the compounded path.

    Returns ``(depth, peak_index, trough_index)`` with ``depth`` negative.
    Compounded rather than cumulative-sum, because a 50% loss followed by a
    50% gain is not break-even and an auditor should see the real hole.
    """
    x = as_returns_array(returns)
    if x.size == 0:
        return math.nan, 0, 0

    # The path starts at 1.0, *before* the first return. Without that leading
    # point the running peak begins at the value after period one, so a
    # strategy that loses 30% immediately and then recovers reports a drawdown
    # of zero - the one it actually had is invisible.
    equity = np.concatenate([[1.0], np.cumprod(1.0 + x)])
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity / running_peak - 1.0

    trough_at = int(np.argmin(drawdown))
    if trough_at == 0:
        return 0.0, 0, 0
    peak_at = int(np.argmax(equity[: trough_at + 1]))
    # Back to return-series indices: equity index i is the state after return
    # i - 1, and a peak at the starting point maps to the first return.
    return float(drawdown[trough_at]), max(peak_at - 1, 0), trough_at - 1


@dataclass(frozen=True)
class PerformanceSummary:
    """The conventional metrics, for a strategy or a benchmark."""

    name: str
    n_obs: int
    periods_per_year: int
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_length: int
    hit_rate: float
    best_period: float
    worst_period: float
    skewness: float
    kurtosis: float

    @property
    def calmar(self) -> float:
        """Annualised return over the depth of the worst drawdown."""
        if not math.isfinite(self.max_drawdown) or self.max_drawdown >= 0:
            return math.nan
        return self.annualised_return / abs(self.max_drawdown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_obs": self.n_obs,
            "periods_per_year": self.periods_per_year,
            "total_return": self.total_return,
            "annualised_return": self.annualised_return,
            "annualised_volatility": self.annualised_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_length": self.max_drawdown_length,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "best_period": self.best_period,
            "worst_period": self.worst_period,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def summarise_performance(
    returns,
    periods_per_year: int = 252,
    name: str = "strategy",
    rf_per_period: float = 0.0,
) -> PerformanceSummary:
    """Every conventional metric in one pass, for side-by-side comparison.

    ``rf_per_period`` shifts *only* the risk-adjusted ratios. Sharpe and Sortino
    are defined on returns in excess of cash, so a strategy that sits a third of
    the time in an uninvested book must not be credited with the bill rate it
    never earned. Total return, volatility, drawdown and Calmar stay on total
    returns, because those are what an investor actually lived through.
    """
    from qv.stats.moments import standardised_moments

    x = as_returns_array(returns)
    if x.size < 2:
        raise ValueError(f"performance summary needs at least 2 observations, got {x.size}")

    depth, peak, trough = max_drawdown(x)
    skew, kurt = (standardised_moments(x) if x.size >= 4 else (math.nan, math.nan))

    return PerformanceSummary(
        name=name,
        n_obs=int(x.size),
        periods_per_year=periods_per_year,
        total_return=float(np.prod(1.0 + x) - 1.0),
        annualised_return=annualised_return(x, periods_per_year),
        annualised_volatility=annualised_volatility(x, periods_per_year),
        sharpe=sharpe_ratio(x, rf_per_period) * math.sqrt(periods_per_year),
        sortino=sortino_ratio(x, rf_per_period, periods_per_year),
        max_drawdown=depth,
        max_drawdown_length=trough - peak,
        hit_rate=float(np.mean(x > 0)),
        best_period=float(np.max(x)),
        worst_period=float(np.min(x)),
        skewness=skew,
        kurtosis=kurt,
    )
