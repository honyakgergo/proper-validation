"""Is the chosen parameter a plateau or a spike on a cliff?

A strategy whose performance collapses when a lookback moves from 20 to 21 was
not discovered, it was fitted. A genuine effect should degrade gracefully as
parameters move away from the chosen point, producing a plateau; noise-fitting
produces an isolated spike.

Takes an already-evaluated parameter grid rather than running one, so it works
with the researcher's own search results and never needs to re-execute their
code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from qv.types import Severity

__all__ = ["PlateauResult", "plateau_ratio"]


@dataclass(frozen=True)
class PlateauResult:
    """How the chosen parameter point compares with its neighbours."""

    chosen_index: tuple[int, ...]
    chosen_score: float
    neighbour_scores: tuple[float, ...]
    neighbour_mean: float
    grid_mean: float
    grid_max: float
    ratio: float
    radius: int
    n_neighbours: int
    is_grid_maximum: bool

    @property
    def is_spike(self) -> bool:
        """Do the neighbours retain less than half the chosen point's score?"""
        return math.isfinite(self.ratio) and self.ratio < 0.5

    @property
    def severity(self) -> Severity:
        if not math.isfinite(self.ratio):
            return Severity.INFO
        if self.ratio >= 0.8:
            return Severity.INFO
        if self.ratio >= 0.5:
            return Severity.MEDIUM
        if self.ratio >= 0.25:
            return Severity.HIGH
        return Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_index": list(self.chosen_index),
            "chosen_score": self.chosen_score,
            "neighbour_mean": self.neighbour_mean,
            "grid_mean": self.grid_mean,
            "grid_max": self.grid_max,
            "ratio": self.ratio,
            "radius": self.radius,
            "n_neighbours": self.n_neighbours,
            "is_grid_maximum": self.is_grid_maximum,
            "is_spike": self.is_spike,
            "severity": self.severity.name.lower(),
        }


def plateau_ratio(
    scores, chosen_index: tuple[int, ...] | int | None = None, radius: int = 1
) -> PlateauResult:
    """Mean neighbourhood score divided by the chosen point's score.

    ``scores`` is an array of any dimensionality - one axis per parameter -
    holding the performance metric (usually a Sharpe) at each grid point.
    ``chosen_index`` defaults to the grid maximum, which is what a search would
    have selected.

    A ratio near 1.0 is a plateau. Below 0.5 the chosen point is a spike its
    own neighbours do not support, which is the signature of a fitted
    parameter. The ratio is ``nan`` when the chosen score is not positive,
    since a proportion of a non-positive number does not mean anything.
    """
    grid = np.asarray(scores, dtype=float)
    if grid.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    if grid.size < 3:
        raise ValueError(f"a parameter grid needs at least 3 points, got {grid.size}")
    if radius < 1:
        raise ValueError(f"radius must be at least 1, got {radius}")
    if not np.any(np.isfinite(grid)):
        raise ValueError("scores contain no finite values")

    if chosen_index is None:
        chosen = np.unravel_index(int(np.nanargmax(grid)), grid.shape)
    elif isinstance(chosen_index, (int, np.integer)):
        chosen = (int(chosen_index),)
    else:
        chosen = tuple(int(i) for i in chosen_index)

    if len(chosen) != grid.ndim:
        raise ValueError(
            f"chosen_index has {len(chosen)} entries but the grid has {grid.ndim} dimensions"
        )
    for axis, (i, size) in enumerate(zip(chosen, grid.shape)):
        if not 0 <= i < size:
            raise ValueError(f"chosen_index {i} is outside axis {axis} of length {size}")

    offsets = [r for r in product(*[range(-radius, radius + 1)] * grid.ndim) if any(r)]
    neighbours = []
    for offset in offsets:
        idx = tuple(i + d for i, d in zip(chosen, offset))
        if all(0 <= i < size for i, size in zip(idx, grid.shape)) and np.isfinite(grid[idx]):
            neighbours.append(float(grid[idx]))

    if not neighbours:
        raise ValueError("the chosen point has no finite neighbours within the given radius")

    chosen_score = float(grid[chosen])
    neighbour_mean = float(np.mean(neighbours))
    ratio = neighbour_mean / chosen_score if chosen_score > 0 else math.nan

    return PlateauResult(
        chosen_index=chosen,
        chosen_score=chosen_score,
        neighbour_scores=tuple(neighbours),
        neighbour_mean=neighbour_mean,
        grid_mean=float(np.nanmean(grid)),
        grid_max=float(np.nanmax(grid)),
        ratio=ratio,
        radius=radius,
        n_neighbours=len(neighbours),
        is_grid_maximum=bool(chosen_score >= np.nanmax(grid)),
    )
