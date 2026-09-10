"""How much of the realised result was the path it happened to take?

The report states a maximum drawdown as a single number. That is exactly the
defect this package raises against other people as ``STAT-NO-CONFIDENCE-
INTERVAL``: maximum drawdown is an extreme-value statistic with enormous
sampling variability, and one realisation of it says much less than it appears
to. A strategy that drew down 48% might plausibly have drawn down 30% or 70%
on a different draw of the same process.

Three views, because no single one is trustworthy on its own:

* **Realised** - what actually happened, on one path.
* **Stationary bootstrap** - resamples the actual returns, so fat tails and
  short-range dependence survive. It is a *lower bound on severity*: block
  resampling shatters the long sequences that create deep drawdowns. The 2008
  decline unfolded over roughly seventeen months against a typical block
  length of a few days, so the bootstrap simply cannot reassemble it.
* **Matched random walk** - what a strategy with this drift and volatility
  would expect over this many periods. Assumes IID normal returns, so it
  understates too, for the opposite reason: no volatility clustering and no
  fat tails.

Both baselines understate, and knowing that is the point. When the realised
drawdown sits far outside both, it was a *sequence* rather than a draw - the
strategy was exposed to something the resampling cannot recreate, and the
number in the backtest is not a general property of the strategy.

This is deliberately backward-looking. Projecting equity forward from a fitted
distribution remains a flagged defect (``MC-FORWARD-PROJECTION``); this asks
only how much confidence the realised sample supports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.findings import make_finding
from qv.stats.bootstrap import politis_white_block_length, stationary_bootstrap_indices
from qv.stats.moments import as_returns_array
from qv.stats.performance import max_drawdown
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Finding, Severity

__all__ = [
    "RiskDistribution",
    "RiskSimulation",
    "matched_random_walk_drawdowns",
    "simulate_risk",
]


def _drawdown_of(equity_returns: np.ndarray) -> float:
    """Maximum drawdown of one path, on the compounded curve from 1.0."""
    equity = np.concatenate([[1.0], np.cumprod(1.0 + equity_returns)])
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


@dataclass(frozen=True)
class RiskDistribution:
    """One statistic: what happened, and what the resamples say about it."""

    name: str
    realised: float
    draws: np.ndarray
    lower_is_worse: bool = False
    baseline_draws: np.ndarray | None = None
    baseline_label: str | None = None

    @property
    def ci_low(self) -> float:
        return float(np.quantile(self.draws, 0.05))

    @property
    def ci_high(self) -> float:
        return float(np.quantile(self.draws, 0.95))

    @property
    def median(self) -> float:
        return float(np.median(self.draws))

    @property
    def percentile(self) -> float:
        """Where the realised value sits in the resampled distribution."""
        return float(np.mean(self.draws <= self.realised))

    @property
    def baseline_median(self) -> float | None:
        if self.baseline_draws is None or self.baseline_draws.size == 0:
            return None
        return float(np.median(self.baseline_draws))

    @property
    def baseline_percentile(self) -> float | None:
        if self.baseline_draws is None or self.baseline_draws.size == 0:
            return None
        return float(np.mean(self.baseline_draws <= self.realised))

    @property
    def outside_resamples(self) -> bool:
        """Is the realised value worse than 95% of the resampled paths?"""
        if self.lower_is_worse:
            return self.realised < self.ci_low
        return self.realised > self.ci_high

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "realised": self.realised,
            "median": self.median,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "percentile": self.percentile,
            "baseline_median": self.baseline_median,
            "baseline_percentile": self.baseline_percentile,
            "baseline_label": self.baseline_label,
            "outside_resamples": self.outside_resamples,
            "n_draws": int(self.draws.size),
        }


@dataclass(frozen=True)
class RiskSimulation:
    """Resampled distributions of the headline risk and return statistics."""

    total_return: RiskDistribution
    sharpe: RiskDistribution
    max_drawdown: RiskDistribution
    n_obs: int
    n_boot: int
    block_length: float
    periods_per_year: int
    reliable: bool
    note: str | None = None

    @property
    def drawdown_path_dependent(self) -> bool:
        """Was the worst drawdown a sequence rather than a draw?

        True when the realised drawdown is deeper than 95% of block-resampled
        paths. Because block resampling already understates severity, this is a
        conservative test: it fires only when the ordering of returns, not
        their distribution, produced the loss.
        """
        return self.max_drawdown.outside_resamples

    @property
    def drawdown_severity(self) -> Severity:
        if not self.drawdown_path_dependent:
            return Severity.INFO
        realised = abs(self.max_drawdown.realised)
        typical = abs(self.max_drawdown.median)
        if typical <= 0:
            return Severity.MEDIUM
        ratio = realised / typical
        if ratio >= 2.0:
            return Severity.HIGH
        return Severity.MEDIUM

    def to_finding(self) -> Finding | None:
        """A finding when the reported drawdown understates the risk."""
        dd = self.max_drawdown
        if not self.drawdown_path_dependent:
            return None
        baseline = ""
        if dd.baseline_median is not None:
            baseline = (
                f" A random walk with the same drift and volatility would typically have "
                f"drawn down {dd.baseline_median:.1%}."
            )
        return make_finding(
            "RISK-DRAWDOWN-PATH-DEPENDENT",
            detail=(
                f"The realised maximum drawdown of {dd.realised:.1%} is deeper than 95% of "
                f"block-resampled paths from the same returns, which centre on "
                f"{dd.median:.1%} (5th to 95th percentile {dd.ci_low:.1%} to "
                f"{dd.ci_high:.1%}). The loss came from the *order* of the returns, not "
                f"their distribution - so it is not a number the resampling can reproduce, "
                f"and not one a shorter or reordered sample would have revealed.{baseline}"
            ),
            severity=self.drawdown_severity,
            evidence=self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return.to_dict(),
            "sharpe": self.sharpe.to_dict(),
            "max_drawdown": self.max_drawdown.to_dict(),
            "drawdown_path_dependent": self.drawdown_path_dependent,
            "drawdown_severity": self.drawdown_severity.name.lower(),
            "n_obs": self.n_obs,
            "n_boot": self.n_boot,
            "block_length": self.block_length,
            "periods_per_year": self.periods_per_year,
            "reliable": self.reliable,
            "note": self.note,
        }


def matched_random_walk_drawdowns(
    mean: float, sd: float, n_obs: int, n_sims: int = 2000, seed: int | None = 0
) -> np.ndarray:
    """Maximum drawdowns of IID normal paths with the given drift and volatility.

    The baseline that makes a realised drawdown interpretable: what depth
    should a strategy with *this* Sharpe over *this* many periods expect,
    before anything specific to the strategy is considered?

    Simulated rather than taken from the Magdon-Ismail and Atiya closed form,
    which is a series expansion that is easy to get subtly wrong and hard to
    check. A simulation is unambiguous and costs milliseconds.

    Assumes IID normal returns, so it has no volatility clustering and no fat
    tails and will therefore understate real drawdowns. That is a stated
    limitation, not a hidden one.
    """
    if n_obs < 2:
        raise ValueError(f"need at least 2 periods, got {n_obs}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    if sd < 0:
        raise ValueError(f"sd must be non-negative, got {sd}")

    rng = np.random.default_rng(seed)
    paths = mean + sd * rng.standard_normal((n_sims, n_obs))
    equity = np.cumprod(1.0 + paths, axis=1)
    equity = np.hstack([np.ones((n_sims, 1)), equity])
    peaks = np.maximum.accumulate(equity, axis=1)
    return (equity / peaks - 1.0).min(axis=1)


def simulate_risk(
    returns,
    periods_per_year: int = 252,
    n_boot: int = 2000,
    n_baseline: int = 2000,
    block_length: float | None = None,
    seed: int | None = 0,
    rf_per_period: float = 0.0,
) -> RiskSimulation:
    """Resample the return series and recompute return, Sharpe and drawdown.

    One set of stationary-bootstrap indices drives all three statistics, so the
    distributions are mutually consistent - a resample that produced a good
    Sharpe is the same resample that produced its drawdown - and the work is
    done once rather than three times.

    ``rf_per_period`` applies to the Sharpe panel only. The return and drawdown
    panels describe the path an investor lived through, which is a total-return
    object; the Sharpe is a risk-adjusted ratio, which is an excess-return one.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 20:
        raise ValueError(f"risk simulation needs at least 20 observations, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    b = politis_white_block_length(x) if block_length is None else float(block_length)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, b, n_boot, rng)

    returns_draws = np.empty(n_boot)
    sharpe_draws = np.empty(n_boot)
    drawdown_draws = np.empty(n_boot)
    root = math.sqrt(periods_per_year)

    for i, row in enumerate(idx):
        sample = x[row]
        sd = float(np.std(sample, ddof=1))
        returns_draws[i] = float(np.prod(1.0 + sample) - 1.0)
        sharpe_draws[i] = (
            (float(np.mean(sample)) - rf_per_period) / sd * root if sd > 0 else np.nan
        )
        drawdown_draws[i] = _drawdown_of(sample)

    finite = np.isfinite(sharpe_draws)
    realised_sd = float(np.std(x, ddof=1))

    baseline = matched_random_walk_drawdowns(
        mean=float(np.mean(x)), sd=realised_sd, n_obs=n, n_sims=n_baseline, seed=seed
    )

    reliable = n >= MIN_OBS_FOR_ASYMPTOTICS
    notes = []
    if not reliable:
        notes.append(
            f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} at which resampled "
            "intervals become trustworthy"
        )
    notes.append(
        f"block resampling with a mean block of {b:.1f} periods cannot reassemble drawdowns "
        "that unfolded over much longer horizons, so the resampled distribution understates "
        "how deep a drawdown this strategy can produce"
    )

    return RiskSimulation(
        total_return=RiskDistribution(
            name="total return",
            realised=float(np.prod(1.0 + x) - 1.0),
            draws=returns_draws,
        ),
        sharpe=RiskDistribution(
            name="annualised Sharpe",
            realised=(
                (float(np.mean(x)) - rf_per_period) / realised_sd * root
                if realised_sd > 0
                else math.nan
            ),
            draws=sharpe_draws[finite],
        ),
        max_drawdown=RiskDistribution(
            name="maximum drawdown",
            realised=max_drawdown(x)[0],
            draws=drawdown_draws,
            lower_is_worse=True,
            baseline_draws=baseline,
            baseline_label="matched random walk",
        ),
        n_obs=n,
        n_boot=n_boot,
        block_length=b,
        periods_per_year=periods_per_year,
        reliable=reliable,
        note="; ".join(notes),
    )
