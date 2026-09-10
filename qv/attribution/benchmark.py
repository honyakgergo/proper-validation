"""Volatility-matched buy and hold.

The plain-language companion to the factor regression, and frequently more
persuasive than it. Take the benchmark, lever it up or down until its
volatility matches the strategy, and compare. If holding a scaled index does
as well, the strategy has to justify its complexity on some other grounds.

Kept alongside the factor regression rather than folded into it because they
answer the same question in registers that different readers trust: one gives
a t-statistic, the other gives a number anyone can check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.stats.moments import as_returns_array
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity

__all__ = ["BenchmarkComparison", "vol_matched_benchmark"]


@dataclass(frozen=True)
class BenchmarkComparison:
    """Strategy against a volatility-matched version of a benchmark."""

    strategy_sharpe: float
    benchmark_sharpe: float
    scaled_benchmark_sharpe: float
    leverage: float
    strategy_volatility: float
    benchmark_volatility: float
    correlation: float
    beta: float
    n_obs: int
    benchmark_name: str

    @property
    def excess_sharpe(self) -> float:
        """How much Sharpe the strategy adds over the scaled benchmark."""
        return self.strategy_sharpe - self.scaled_benchmark_sharpe

    @property
    def beats_benchmark(self) -> bool:
        return self.excess_sharpe > 0

    @property
    def is_levered_beta(self) -> bool:
        """High correlation to the benchmark and no Sharpe advantage over it."""
        return abs(self.correlation) > 0.7 and self.excess_sharpe <= 0

    @property
    def severity(self) -> Severity:
        if self.is_levered_beta:
            return Severity.CRITICAL
        if not self.beats_benchmark:
            return Severity.HIGH
        if abs(self.correlation) > 0.7:
            return Severity.MEDIUM
        return Severity.INFO

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "strategy_sharpe": self.strategy_sharpe,
            "benchmark_sharpe": self.benchmark_sharpe,
            "scaled_benchmark_sharpe": self.scaled_benchmark_sharpe,
            "excess_sharpe": self.excess_sharpe,
            "leverage": self.leverage,
            "strategy_volatility": self.strategy_volatility,
            "benchmark_volatility": self.benchmark_volatility,
            "correlation": self.correlation,
            "beta": self.beta,
            "beats_benchmark": self.beats_benchmark,
            "is_levered_beta": self.is_levered_beta,
            "severity": self.severity.name.lower(),
            "n_obs": self.n_obs,
        }


def vol_matched_benchmark(
    returns, benchmark_returns, benchmark_name: str = "benchmark"
) -> BenchmarkComparison:
    """Compare a strategy against the benchmark levered to the same volatility.

    Scaling a return series by a constant leaves its Sharpe unchanged, so
    ``scaled_benchmark_sharpe`` always equals ``benchmark_sharpe``. That is not
    redundancy - reporting the leverage is the point. "This strategy matches
    2.4x SPY" is a sentence a reader can evaluate immediately, and it makes the
    comparison concrete in a way a bare Sharpe difference does not.
    """
    x = as_returns_array(returns)
    b = as_returns_array(benchmark_returns)
    if x.size != b.size:
        raise ValueError(
            f"strategy has {x.size} observations but benchmark has {b.size}; "
            "they must describe the same time axis"
        )
    if x.size < 4:
        raise ValueError(f"benchmark comparison needs at least 4 observations, got {x.size}")

    strategy_vol = float(np.std(x, ddof=1))
    benchmark_vol = float(np.std(b, ddof=1))

    if benchmark_vol <= 0:
        raise ValueError("the benchmark has no variance, so it cannot be volatility-matched")

    leverage = strategy_vol / benchmark_vol
    scaled = b * leverage

    if strategy_vol > 0:
        correlation = float(np.corrcoef(x, b)[0, 1])
        beta = float(np.cov(x, b, ddof=1)[0, 1] / (benchmark_vol**2))
    else:
        correlation, beta = math.nan, 0.0

    return BenchmarkComparison(
        strategy_sharpe=sharpe_ratio(x),
        benchmark_sharpe=sharpe_ratio(b),
        scaled_benchmark_sharpe=sharpe_ratio(scaled),
        leverage=leverage,
        strategy_volatility=strategy_vol,
        benchmark_volatility=benchmark_vol,
        correlation=correlation,
        beta=beta,
        n_obs=x.size,
        benchmark_name=benchmark_name,
    )
