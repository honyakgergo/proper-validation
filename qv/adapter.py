"""Turning a DataFrame strategy into the callable the audit can re-run.

The engine hands a strategy a bare ``(T, n_features)`` array, because the whole
point of the behavioural test is that it controls every number the strategy
sees. Most real strategies want a labelled frame back: a ``DatetimeIndex`` to
find the month ends, column names to pick one sleeve out of a universe.
Rebuilding one is three lines, and getting those three lines wrong in one
specific way voids the test silently - close over the original frame instead of
the array you were handed, and the corruption never reaches the strategy, which
then passes every time and has earned nothing.

So this closes over the *labels* only. There is no parameter that takes values,
and the trap is therefore not expressible through this function.

Two contracts live here, and the second is derived from the first:

* :class:`PositionsStrategy` - ``(prices, **params) -> positions``. What a
  researcher actually writes, and what a parameter sweep needs, since a sweep
  has to vary the parameters.
* ``strategy(data) -> signals`` - what :func:`qv.leakage.perturbation
  .perturbation_test` needs, with the parameters already bound. This is the
  narrower contract, because the behavioural test audits one configuration:
  the one that was reported.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = ["PositionsStrategy", "frame_adapter"]


@runtime_checkable
class PositionsStrategy(Protocol):
    """A strategy stated precisely enough to re-run.

    Takes a wide price frame - a ``DatetimeIndex``, one column per instrument -
    plus its parameters as keywords, and returns a frame of positions on the
    same index. Both real strategies in this repository already satisfy it
    without modification, which is the point: the contract describes how people
    write this code, rather than asking them to rewrite it.

    Two requirements the type cannot express and the behavioural test will
    catch anyway: it must be deterministic, and it must read nothing but the
    frame it is given. No file access, no module-level cache that survives
    between calls, no network.
    """

    def __call__(self, prices: pd.DataFrame, **params: Any) -> pd.DataFrame: ...


def frame_adapter(
    positions_of: Callable[..., Any],
    index: Any,
    columns: Any,
    **params: Any,
) -> Callable[[np.ndarray], np.ndarray]:
    """Adapt ``positions_of(frame, **params)`` to ``strategy(array) -> (T, k)``.

    ``index`` and ``columns`` are copied out of the frame the research ran on.
    A trading calendar and a universe are both known in advance, so holding
    them fixed is not hindsight - but it does mean the behavioural test clears
    the strategy's use of *prices* and says nothing about its use of the
    calendar. That limitation is real and is stated in the report rather than
    papered over.

    Every *number* the strategy sees comes from the array the engine supplies.

    The shape check is not defensive clutter. Labels that no longer describe
    the data would silently relabel the columns, so the audit would test a
    different strategy from the one that was run - and it would come back
    clean, for entirely the wrong reason.
    """
    index = pd.Index(index).copy()
    columns = pd.Index(columns).copy()
    if len(index) == 0:
        raise ValueError("frame_adapter needs a non-empty index")
    if len(columns) == 0:
        raise ValueError("frame_adapter needs at least one column label")
    if columns.has_duplicates:
        raise ValueError(f"column labels must be unique, got {list(columns)}")
    expected = (len(index), len(columns))

    def strategy(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data, dtype=float)
        if array.shape != expected:
            raise ValueError(
                f"adapter was built for data of shape {expected} but was given "
                f"{array.shape}; the labels no longer describe the data"
            )
        frame = pd.DataFrame(array, index=index, columns=columns)
        return np.asarray(positions_of(frame, **params), dtype=float)

    return strategy
