"""Does the result survive dropping part of the sample?

Two questions, deliberately distinct:

* **Top-N-day dependence** - is the P&L concentrated in a handful of days? A
  strategy whose entire edge lives in the best 1% of days is a lottery ticket
  with a Sharpe ratio attached, and the next such day may not arrive.
* **Rolling-origin sensitivity** - does the result depend on when you started
  measuring? This subsumes the calendar and decade splits that a regime
  analysis would otherwise duplicate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as _st

from qv.stats.moments import as_returns_array
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity

__all__ = [
    "expected_profit_share",
    "TopDayResult",
    "top_day_dependence",
    "RollingOriginResult",
    "rolling_origin_sensitivity",
]


def expected_profit_share(sharpe: float, fraction: float) -> float:
    """Share of total profit the best ``fraction`` of periods would supply if
    returns were normal with this Sharpe.

    The baseline that makes the raw share interpretable. For a normal series,
    ``E[sum of the top q share] = n*mu + sigma*n*phi(z)`` with
    ``z = Phi_inv(1 - q)``, so as a share of total profit ``n*mu`` this is

    ``q + phi(z) / SR``

    The second term is the crux, and it is what makes an unbaselined version of
    this test useless: it blows up as the Sharpe falls. A daily strategy with a
    per-period Sharpe of 0.03 - a perfectly ordinary annualised 0.5 - *should*
    show its best 1% of days accounting for essentially all of its profit.
    Flagging that as concentration risk would fire on almost every real
    strategy and mean nothing.
    """
    if not math.isfinite(sharpe) or sharpe <= 0:
        return math.nan
    z = float(_st.norm.ppf(1.0 - fraction))
    return fraction + float(_st.norm.pdf(z)) / sharpe


@dataclass(frozen=True)
class TopDayResult:
    """How much of the edge survives removing the best periods.

    Read ``concentration_ratios``, not ``sharpes``. The raw post-drop Sharpe
    depends overwhelmingly on the strategy's overall Sharpe rather than on any
    genuine concentration; the ratio against a normal baseline is what isolates
    the concentration itself.
    """

    full_sharpe: float
    drop_fractions: tuple[float, ...]
    sharpes: tuple[float, ...]
    total_return_shares: tuple[float, ...]
    expected_shares: tuple[float, ...]
    n_obs: int

    @property
    def concentration_ratios(self) -> tuple[float, ...]:
        """Observed profit share divided by the normal-baseline share.

        About 1.0 means the P&L is as concentrated as a normal series of this
        Sharpe would be - which is to say, not a finding. Above about 1.5 means
        genuinely lumpy returns driven by a few outliers.
        """
        return tuple(
            obs / exp if math.isfinite(exp) and exp > 0 else math.nan
            for obs, exp in zip(self.total_return_shares, self.expected_shares)
        )

    @property
    def worst_concentration_ratio(self) -> float:
        finite = [r for r in self.concentration_ratios if math.isfinite(r)]
        return max(finite) if finite else math.nan

    @property
    def sharpe_after_dropping_top_1pct(self) -> float:
        return self._at(0.01)

    @property
    def sharpe_after_dropping_top_5pct(self) -> float:
        return self._at(0.05)

    def _at(self, fraction: float) -> float:
        for f, s in zip(self.drop_fractions, self.sharpes):
            if abs(f - fraction) < 1e-12:
                return s
        raise KeyError(f"{fraction} was not among the drop fractions tested")

    @property
    def abnormally_concentrated(self) -> bool:
        """Is the profit lumpier than this Sharpe alone would explain?"""
        worst = self.worst_concentration_ratio
        return math.isfinite(worst) and worst > 1.5

    @property
    def severity(self) -> Severity:
        if self.full_sharpe <= 0:
            return Severity.INFO
        # Calibrated against simulation: ordinary normal returns give a ratio of
        # 1.0 with a spread of about 0.4 across seeds, so 1.5 is comfortably
        # outside noise and 2.0 - the best periods supplying twice the profit
        # share the Sharpe alone implies - is a real concentration problem.
        worst = self.worst_concentration_ratio
        if not math.isfinite(worst) or worst < 1.5:
            return Severity.INFO
        if worst < 2.0:
            return Severity.MEDIUM
        if worst < 3.5:
            return Severity.HIGH
        return Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_sharpe": self.full_sharpe,
            "drop_fractions": list(self.drop_fractions),
            "sharpes": list(self.sharpes),
            "total_return_shares": list(self.total_return_shares),
            "expected_shares": list(self.expected_shares),
            "concentration_ratios": list(self.concentration_ratios),
            "worst_concentration_ratio": self.worst_concentration_ratio,
            "sharpe_after_dropping_top_1pct": self.sharpe_after_dropping_top_1pct,
            "sharpe_after_dropping_top_5pct": self.sharpe_after_dropping_top_5pct,
            "abnormally_concentrated": self.abnormally_concentrated,
            "severity": self.severity.name.lower(),
            "n_obs": self.n_obs,
        }


def top_day_dependence(
    returns, drop_fractions: tuple[float, ...] = (0.01, 0.05)
) -> TopDayResult:
    """Recompute the Sharpe with the best periods removed.

    ``total_return_shares`` reports what fraction of the cumulative sum of
    returns those best periods accounted for - a strategy where the best 1% of
    days supplied 80% of the profit is telling you something the headline
    Sharpe hides.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 20:
        raise ValueError(f"top-day dependence needs at least 20 observations, got {n}")
    for f in drop_fractions:
        if not 0.0 < f < 1.0:
            raise ValueError(f"drop fractions must be strictly between 0 and 1, got {f}")

    order = np.argsort(x)[::-1]  # best first
    total = float(np.sum(x))
    full_sharpe = sharpe_ratio(x)

    sharpes, shares, expected = [], [], []
    for f in drop_fractions:
        k = max(1, int(round(f * n)))
        if n - k < 4:
            raise ValueError(
                f"dropping {f:.1%} of {n} observations leaves {n - k}, too few for a Sharpe"
            )
        dropped = order[:k]
        kept = np.delete(x, dropped)
        sharpes.append(sharpe_ratio(kept))
        shares.append(float(np.sum(x[dropped]) / total) if total != 0 else math.nan)
        expected.append(expected_profit_share(full_sharpe, k / n))

    return TopDayResult(
        full_sharpe=full_sharpe,
        drop_fractions=tuple(drop_fractions),
        sharpes=tuple(sharpes),
        total_return_shares=tuple(shares),
        expected_shares=tuple(expected),
        n_obs=n,
    )


@dataclass(frozen=True)
class RollingOriginResult:
    """Sharpe as a function of where the sample is taken to begin."""

    start_indices: tuple[int, ...]
    start_fractions: tuple[float, ...]
    sharpes: tuple[float, ...]
    full_sharpe: float
    n_obs: int

    @property
    def min_sharpe(self) -> float:
        return min(self.sharpes)

    @property
    def max_sharpe(self) -> float:
        return max(self.sharpes)

    @property
    def spread(self) -> float:
        return self.max_sharpe - self.min_sharpe

    @property
    def fraction_positive(self) -> float:
        return float(np.mean([s > 0 for s in self.sharpes]))

    @property
    def sign_stable(self) -> bool:
        """Does every start date agree on the sign of the edge?"""
        return self.fraction_positive in (0.0, 1.0)

    @property
    def relative_stability(self) -> float:
        """Smallest Sharpe divided by the largest, for a positive edge.

        A sign flip is the loud version of start-date dependence, but not the
        only one that matters. An edge that runs from 0.01 to 0.25 depending on
        where you start never changes sign and is still almost entirely a
        property of one window. Returns ``nan`` when the largest Sharpe is not
        positive, since a ratio of losses says nothing useful.
        """
        if self.max_sharpe <= 0:
            return math.nan
        return self.min_sharpe / self.max_sharpe

    @property
    def start_date_dependent(self) -> bool:
        """Either a sign flip, or a swing too large to call the edge stable."""
        if not self.sign_stable:
            return True
        stability = self.relative_stability
        return math.isfinite(stability) and stability < 0.25

    @property
    def severity(self) -> Severity:
        if not self.sign_stable:
            if self.fraction_positive >= 0.8:
                return Severity.MEDIUM
            if self.fraction_positive >= 0.5:
                return Severity.HIGH
            return Severity.CRITICAL
        stability = self.relative_stability
        if math.isfinite(stability) and stability < 0.25:
            return Severity.HIGH
        return Severity.INFO

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_indices": list(self.start_indices),
            "start_fractions": list(self.start_fractions),
            "sharpes": list(self.sharpes),
            "full_sharpe": self.full_sharpe,
            "min_sharpe": self.min_sharpe,
            "max_sharpe": self.max_sharpe,
            "spread": self.spread,
            "fraction_positive": self.fraction_positive,
            "sign_stable": self.sign_stable,
            "relative_stability": self.relative_stability,
            "start_date_dependent": self.start_date_dependent,
            "severity": self.severity.name.lower(),
            "n_obs": self.n_obs,
        }


def rolling_origin_sensitivity(
    returns, n_starts: int = 20, max_start_fraction: float = 0.5, min_obs: int = 30
) -> RollingOriginResult:
    """Recompute the Sharpe from progressively later start dates.

    Every window runs to the end of the sample, so this answers "would I have
    concluded the same thing if I had started measuring later?" - which is the
    question a decade-by-decade split answers worse, since a fixed calendar
    bucket also changes the amount of data.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < min_obs + 4:
        raise ValueError(
            f"rolling-origin analysis needs more than {min_obs + 4} observations, got {n}"
        )
    if n_starts < 2:
        raise ValueError(f"n_starts must be at least 2, got {n_starts}")
    if not 0.0 <= max_start_fraction < 1.0:
        raise ValueError(
            f"max_start_fraction must be in [0, 1), got {max_start_fraction}"
        )

    latest = min(int(n * max_start_fraction), n - min_obs)
    if latest < 0:
        latest = 0
    starts = np.unique(np.linspace(0, latest, n_starts).astype(int))

    sharpes = tuple(sharpe_ratio(x[s:]) for s in starts)
    return RollingOriginResult(
        start_indices=tuple(int(s) for s in starts),
        start_fractions=tuple(float(s / n) for s in starts),
        sharpes=sharpes,
        full_sharpe=sharpe_ratio(x),
        n_obs=n,
    )
