"""Turnover and transaction-cost models.

Costs are where most paper strategies die, and they are almost never derived -
they are assumed, usually as a round number chosen after the fact. This module
computes turnover from the actual position series and applies explicit cost
models to it, so the report can state what the strategy costs rather than what
its author hoped it would cost.

Square-root market impact is deliberately absent, and the reason is narrower
than "no data". Impact is order size measured against available liquidity, so
it needs both terms. Volume is fetched - it is one of ``CANONICAL_COLUMNS`` -
but it stops at the data-quality scan and never reaches ``AuditInputs``, and
the intended trading size is only asked for in the interview, where the answer
is recorded for a reader to judge rather than modelled. Supplying one term
without the other gives a model that quietly degenerates into a constant,
which is the fixed-bps model already here wearing a more impressive name.
Break-even cost in basis points carries the argument without either: it says
what the strategy can afford to pay, and leaves the reader to decide whether
their size can trade inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "TYPICAL_COST_BPS",
    "ASSET_CLASS_LABELS",
    "as_position_matrix",
    "turnover_series",
    "average_turnover",
    "CostModel",
    "FixedBpsCost",
    "SpreadProportionalCost",
    "apply_costs",
]

#: Display names, so the report says "US large-cap ETFs" rather than the
#: lowercased identifier "us large cap etf".
ASSET_CLASS_LABELS: dict[str, str] = {
    "us_large_cap_etf": "US large-cap ETFs",
    "us_large_cap_equity": "US large-cap equities",
    "us_small_cap_equity": "US small-cap equities",
    "international_developed_equity": "international developed equities",
    "emerging_market_equity": "emerging-market equities",
}

#: Indicative round-trip cost in basis points of traded notional, for context
#: in the report only. These are order-of-magnitude figures for a small
#: institutional account in normal conditions, not quotes - a strategy whose
#: break-even sits inside these ranges is in trouble regardless of the exact
#: number.
TYPICAL_COST_BPS: dict[str, tuple[float, float]] = {
    "us_large_cap_etf": (1.0, 3.0),
    "us_large_cap_equity": (2.0, 5.0),
    "us_small_cap_equity": (10.0, 30.0),
    "international_developed_equity": (5.0, 15.0),
    "emerging_market_equity": (20.0, 50.0),
}


def as_position_matrix(positions) -> np.ndarray:
    """Coerce positions to a 2-D ``(T, n_assets)`` array of portfolio weights.

    A 1-D series is treated as a single asset, which is the common case for a
    timing strategy on one instrument.
    """
    matrix = np.asarray(
        getattr(positions, "to_numpy", lambda: positions)(), dtype=float
    )
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"positions must be 1-D or 2-D, got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"need at least 2 position observations, got {matrix.shape[0]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("positions contain non-finite values")
    return matrix


def turnover_series(positions, initial_position: float = 0.0) -> np.ndarray:
    """Traded notional per period, as a fraction of portfolio value.

    ``turnover_t = sum_i |w_{i,t} - w_{i,t-1}|`` - the *total* notional
    exchanged, counting both the sell and the buy legs of a rotation. A
    strategy that moves fully from one asset to another therefore shows a
    turnover of 2.0, not 1.0, and pays cost on both legs, which is what
    actually happens.

    The first period is measured against ``initial_position``, so the cost of
    establishing the book is charged rather than quietly forgiven.
    """
    matrix = as_position_matrix(positions)
    previous = np.vstack([np.full((1, matrix.shape[1]), initial_position), matrix[:-1]])
    return np.abs(matrix - previous).sum(axis=1)


def average_turnover(positions, periods_per_year: int | None = None) -> float:
    """Mean per-period turnover, annualised when ``periods_per_year`` is given."""
    mean = float(np.mean(turnover_series(positions)))
    return mean if periods_per_year is None else mean * periods_per_year


@dataclass(frozen=True)
class CostModel:
    """Base class. Subclasses turn a turnover series into a cost series."""

    name: str

    def cost_series(self, turnover: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class FixedBpsCost(CostModel):
    """A flat charge in basis points on every unit of traded notional.

    The simplest defensible model, and the one that makes break-even cost
    directly interpretable: the answer comes out in the same units as the
    input.
    """

    bps: float = 5.0
    name: str = "fixed bps"

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError(f"bps must be non-negative, got {self.bps}")

    def cost_series(self, turnover: np.ndarray) -> np.ndarray:
        return np.asarray(turnover, dtype=float) * self.bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "bps": self.bps}


@dataclass(frozen=True)
class SpreadProportionalCost(CostModel):
    """Half the quoted spread per leg, plus a fixed commission.

    Crossing the spread costs you half of it on entry and half on exit, so a
    ``spread_bps`` of 4 charges 2 bps per unit of traded notional. The
    commission is added on top and is not spread-dependent.
    """

    spread_bps: float = 4.0
    commission_bps: float = 0.5
    name: str = "spread-proportional"

    def __post_init__(self) -> None:
        if self.spread_bps < 0:
            raise ValueError(f"spread_bps must be non-negative, got {self.spread_bps}")
        if self.commission_bps < 0:
            raise ValueError(f"commission_bps must be non-negative, got {self.commission_bps}")

    @property
    def effective_bps(self) -> float:
        return 0.5 * self.spread_bps + self.commission_bps

    def cost_series(self, turnover: np.ndarray) -> np.ndarray:
        return np.asarray(turnover, dtype=float) * self.effective_bps / 10_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spread_bps": self.spread_bps,
            "commission_bps": self.commission_bps,
            "effective_bps": self.effective_bps,
        }


def apply_costs(gross_returns, positions, model: CostModel) -> np.ndarray:
    """Net returns after charging ``model`` against the position series."""
    gross = np.asarray(
        getattr(gross_returns, "to_numpy", lambda: gross_returns)(), dtype=float
    )
    if gross.ndim != 1:
        raise ValueError(f"gross_returns must be 1-D, got shape {gross.shape}")

    turnover = turnover_series(positions)
    if turnover.size != gross.size:
        raise ValueError(
            f"positions have {turnover.size} periods but returns have {gross.size}; "
            "they must describe the same time axis"
        )
    return gross - model.cost_series(turnover)
