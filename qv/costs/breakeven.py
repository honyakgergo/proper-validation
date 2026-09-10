"""Break-even cost: the headline number.

At what per-trade cost in basis points does this strategy stop making money?
Compare that against what trading it would actually cost and you have, in one
number, the most clarifying thing a backtest audit can say. A strategy whose
break-even is 3 bps in a market that charges 5 does not have a small problem.

The zero-Sharpe case has a closed form. A Sharpe is zero exactly when the mean
net return is zero, and mean net return falls linearly in the cost rate, so

``breakeven_bps = 10000 * mean(gross) / mean(turnover)``

with no search required. Break-even against a non-zero target Sharpe does need
a solve, because the cost charge also changes the standard deviation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.costs.models import (
    ASSET_CLASS_LABELS,
    TYPICAL_COST_BPS,
    FixedBpsCost,
    apply_costs,
    turnover_series,
)
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity

__all__ = ["BreakEvenResult", "break_even_cost", "sharpe_vs_cost_curve"]


@dataclass(frozen=True)
class BreakEvenResult:
    """Break-even cost and its verdict against a realistic charge."""

    break_even_bps: float
    gross_sharpe: float
    mean_turnover: float
    annual_turnover: float | None
    target_sharpe: float
    asset_class: str | None = None
    realistic_bps: tuple[float, float] | None = None
    note: str | None = None

    @property
    def survives_realistic_costs(self) -> bool | None:
        """Does the break-even clear the top of the realistic cost range?

        ``None`` when no asset class was named - the tool will not invent a
        cost estimate on the researcher's behalf.
        """
        if self.realistic_bps is None:
            return None
        return self.break_even_bps > self.realistic_bps[1]

    @property
    def margin(self) -> float | None:
        """Break-even divided by the upper realistic cost. Below 1.0 is fatal."""
        if self.realistic_bps is None or self.realistic_bps[1] <= 0:
            return None
        return self.break_even_bps / self.realistic_bps[1]

    @property
    def severity(self) -> Severity:
        margin = self.margin
        if margin is None:
            return Severity.INFO
        if margin >= 3.0:
            return Severity.INFO
        if margin >= 1.5:
            return Severity.MEDIUM
        if margin >= 1.0:
            return Severity.HIGH
        return Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "break_even_bps": self.break_even_bps,
            "gross_sharpe": self.gross_sharpe,
            "mean_turnover": self.mean_turnover,
            "annual_turnover": self.annual_turnover,
            "target_sharpe": self.target_sharpe,
            "asset_class": self.asset_class,
            "asset_class_label": (
                None if self.asset_class is None
                else ASSET_CLASS_LABELS.get(self.asset_class, self.asset_class)
            ),
            "realistic_bps": list(self.realistic_bps) if self.realistic_bps else None,
            "survives_realistic_costs": self.survives_realistic_costs,
            "margin": self.margin,
            "severity": self.severity.name.lower(),
            "note": self.note,
        }


def break_even_cost(
    gross_returns,
    positions,
    target_sharpe: float = 0.0,
    asset_class: str | None = None,
    periods_per_year: int | None = None,
    max_bps: float = 10_000.0,
) -> BreakEvenResult:
    """Cost in basis points at which the strategy reaches ``target_sharpe``.

    ``asset_class`` may be any key of
    :data:`qv.costs.models.TYPICAL_COST_BPS`; supplying it adds the comparison
    that makes the number mean something. Leaving it out is honest rather than
    unhelpful - the tool will not guess what the researcher trades.

    Returns ``inf`` for a strategy that never trades and ``0.0`` for one whose
    gross Sharpe already fails the target, in both cases with a note rather
    than a misleading finite number.
    """
    gross = np.asarray(
        getattr(gross_returns, "to_numpy", lambda: gross_returns)(), dtype=float
    )
    if gross.ndim != 1:
        raise ValueError(f"gross_returns must be 1-D, got shape {gross.shape}")
    if max_bps <= 0:
        raise ValueError(f"max_bps must be positive, got {max_bps}")

    turnover = turnover_series(positions)
    if turnover.size != gross.size:
        raise ValueError(
            f"positions have {turnover.size} periods but returns have {gross.size}; "
            "they must describe the same time axis"
        )

    mean_turnover = float(np.mean(turnover))
    gross_sharpe = sharpe_ratio(gross)
    annual_turnover = None if periods_per_year is None else mean_turnover * periods_per_year

    realistic = None
    if asset_class is not None:
        if asset_class not in TYPICAL_COST_BPS:
            raise ValueError(
                f"unknown asset_class {asset_class!r}; "
                f"expected one of {sorted(TYPICAL_COST_BPS)}"
            )
        realistic = TYPICAL_COST_BPS[asset_class]

    def _result(bps: float, note: str | None) -> BreakEvenResult:
        return BreakEvenResult(
            break_even_bps=bps,
            gross_sharpe=gross_sharpe,
            mean_turnover=mean_turnover,
            annual_turnover=annual_turnover,
            target_sharpe=target_sharpe,
            asset_class=asset_class,
            realistic_bps=realistic,
            note=note,
        )

    if mean_turnover <= 0:
        return _result(
            math.inf,
            "the strategy never trades, so no cost can erode it; check the position "
            "series is what you think it is",
        )
    if not math.isfinite(gross_sharpe):
        return _result(math.nan, "gross returns have no variance, so the Sharpe is undefined")
    if gross_sharpe <= target_sharpe:
        return _result(
            0.0,
            f"the gross Sharpe of {gross_sharpe:.3f} already fails the target of "
            f"{target_sharpe:.3f}; costs are not the binding problem",
        )

    if target_sharpe == 0.0:
        # Exact: Sharpe is zero exactly when the mean net return is zero.
        return _result(10_000.0 * float(np.mean(gross)) / mean_turnover, None)

    # Non-zero target: the cost charge shifts the standard deviation too, so
    # bisect. Sharpe is continuous and decreasing in the cost rate.
    def net_sharpe(bps: float) -> float:
        return sharpe_ratio(apply_costs(gross, positions, FixedBpsCost(bps=bps)))

    lo, hi = 0.0, 1.0
    while net_sharpe(hi) > target_sharpe:
        hi *= 2.0
        if hi > max_bps:
            return _result(
                math.inf,
                f"Sharpe still exceeds the target at {max_bps:g} bps; the strategy is "
                "effectively cost-insensitive over any plausible range",
            )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if net_sharpe(mid) > target_sharpe:
            lo = mid
        else:
            hi = mid
    return _result(0.5 * (lo + hi), None)


def sharpe_vs_cost_curve(
    gross_returns, positions, max_bps: float | None = None, n_points: int = 60
) -> tuple[np.ndarray, np.ndarray]:
    """``(cost_bps, net_sharpe)`` arrays for the break-even chart.

    Spans zero to ``max_bps``, defaulting to 1.5x the zero-Sharpe break-even so
    the crossing is visible with some context either side.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")

    if max_bps is None:
        be = break_even_cost(gross_returns, positions).break_even_bps
        max_bps = 50.0 if not math.isfinite(be) or be <= 0 else 1.5 * be
    if max_bps <= 0:
        raise ValueError(f"max_bps must be positive, got {max_bps}")

    grid = np.linspace(0.0, max_bps, n_points)
    sharpes = np.array(
        [
            sharpe_ratio(apply_costs(gross_returns, positions, FixedBpsCost(bps=float(b))))
            for b in grid
        ]
    )
    return grid, sharpes
