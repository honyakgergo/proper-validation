"""Running a parameter search, so the audit is handed the whole search.

Deflated Sharpe, PBO and the parameter-plateau test all need something the
researcher usually throws away: the returns of every configuration examined,
not just the winner. `n_trials` is the most under-reported number in
backtesting precisely because nobody keeps the losers.

Given a strategy stated as :class:`qv.adapter.PositionsStrategy` and the axes
of the grid that was searched, this module reconstructs all three inputs by
re-running the search:

* the ``(T, N)`` **trial matrix** that PBO and the empirical max-Sharpe null
  need,
* the n-dimensional **parameter surface** the plateau test needs, one axis per
  parameter, and
* the flat and multi-dimensional index of the configuration actually chosen.

Two things it deliberately does not do. It does not choose the winner: the
chosen configuration is declared, because a researcher who picked a point on
theoretical grounds deserves to be judged on the point they picked rather than
on the grid maximum. And it does not decide `n_trials` for you - the grid it
reruns is a *lower bound* on the search, since it cannot see the
configurations that were abandoned before they were written down.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["GridResult", "parameter_grid", "run_parameter_grid"]


@dataclass(frozen=True)
class GridResult:
    """Every configuration examined, evaluated."""

    #: ``(T, N)`` returns, one column per configuration, in `grid` order.
    trial_returns: np.ndarray
    #: The configurations, as dicts, in the order of the matrix columns.
    grid: list[dict[str, Any]]
    #: Annualised Sharpe at each grid point, shaped one axis per parameter.
    parameter_scores: np.ndarray
    #: Position of the *declared* configuration in `parameter_scores`.
    chosen_index: tuple[int, ...]
    #: Its position in `grid` and in the trial matrix columns.
    chosen_column: int
    axes: dict[str, list[Any]]

    @property
    def n_trials(self) -> int:
        return len(self.grid)

    @property
    def chosen_score(self) -> float:
        return float(self.parameter_scores[self.chosen_index])

    @property
    def best_score(self) -> float:
        return float(np.nanmax(self.parameter_scores))

    def summary(self) -> str:
        """One line for a driver script to print."""
        best = self.grid[int(np.nanargmax(self.parameter_scores))]
        return (
            f"chosen {self.chosen_score:.3f} | best {self.best_score:.3f} {best} "
            f"| median {float(np.nanmedian(self.parameter_scores)):.3f}"
        )


def parameter_grid(axes: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Every combination of the declared axes, in a fixed, reproducible order.

    The order is the Cartesian product taken over the axes as given, which is
    what makes ``reshape`` in :func:`run_parameter_grid` line the columns of
    the trial matrix up with the axes of the parameter surface. Changing it
    would silently transpose the surface.
    """
    if not axes:
        raise ValueError("a parameter grid needs at least one axis")
    for name, values in axes.items():
        if len(list(values)) == 0:
            raise ValueError(f"axis {name!r} has no values")
    names = list(axes)
    return [
        dict(zip(names, combination))
        for combination in product(*[list(axes[name]) for name in names])
    ]


def run_parameter_grid(
    positions_of: Callable[..., Any],
    prices: pd.DataFrame,
    axes: Mapping[str, Sequence[Any]],
    chosen: Mapping[str, Any],
    periods_per_year: int = 252,
    **fixed: Any,
) -> GridResult:
    """Evaluate every configuration in ``axes``, and locate ``chosen`` in it.

    ``positions_of`` is a :class:`qv.adapter.PositionsStrategy` -
    ``(prices, **params) -> positions``. ``fixed`` holds parameters that were
    not searched and are passed to every call.

    The chosen configuration must be *in* the grid. A declared configuration
    that was not among the ones examined means either the grid or the
    declaration is wrong, and quietly auditing the nearest neighbour would
    misreport which point is being judged.
    """
    grid = parameter_grid(axes)
    names = list(axes)

    missing = [name for name in names if name not in chosen]
    if missing:
        raise ValueError(
            f"the chosen configuration does not declare {missing}, which the grid "
            f"searches over; it must name every axis"
        )
    declared = {name: chosen[name] for name in names}
    if declared not in grid:
        raise ValueError(
            f"the chosen configuration {declared} is not in the grid of "
            f"{len(grid)} configurations searched; one of the two is wrong"
        )

    columns = []
    for params in grid:
        positions = positions_of(prices, **params, **fixed)
        asset_returns = prices.pct_change().fillna(0.0)
        aligned = pd.DataFrame(positions).reindex_like(asset_returns).fillna(0.0)
        columns.append((aligned * asset_returns).sum(axis=1).to_numpy())

    trial_returns = np.column_stack(columns)

    # A column of identical values has a standard deviation of about 1e-16
    # rather than 0, which would give it a Sharpe of 1e14 and let it win any
    # comparison. Screen it out rather than letting a flat configuration take
    # the grid maximum.
    sds = trial_returns.std(axis=0, ddof=1)
    varies = sds > 1e-12
    sharpes = np.where(
        varies, trial_returns.mean(axis=0) / np.where(varies, sds, 1.0), np.nan
    ) * np.sqrt(periods_per_year)

    shape = tuple(len(list(axes[name])) for name in names)
    parameter_scores = sharpes.reshape(shape)

    chosen_index = tuple(
        list(axes[name]).index(chosen[name]) for name in names
    )
    chosen_column = grid.index(declared)

    return GridResult(
        trial_returns=trial_returns,
        grid=grid,
        parameter_scores=parameter_scores,
        chosen_index=chosen_index,
        chosen_column=chosen_column,
        axes={name: list(axes[name]) for name in names},
    )
