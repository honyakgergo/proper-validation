"""Cross-sectional momentum over thirty S&P 500 names.

Ordinary construction, deliberately: 12-1 month momentum, hold the top decile-ish
equal weight, rebalance monthly, positions traded from the session after the signal.
Nothing here is meant to be clever. The example exists to show what happens when a
perfectly reasonable strategy is run on a universe assembled from the index as it
looks *today*.

Read only the `prices` frame handed in. An adapter that closes over the original
DataFrame means the leakage test's corruption never reaches the strategy, and it
then reports a clean bill of health it has not earned.
"""

from __future__ import annotations

import pandas as pd


def positions(prices: pd.DataFrame, **params) -> pd.DataFrame:
    """Weights to hold, one row per row of `prices`.

    Row t is the book held *into* t+1, so it may use data up to and including row
    t and nothing after it. The `.shift(1)` at the end is what stops a position
    earning the return that produced it.

    The 12-1 construction skips the most recent month, which is standard: the
    short-horizon reversal that dominates the last few weeks runs against the
    medium-horizon momentum the signal is trying to capture.
    """
    lookback = int(params.get("lookback", 252))
    skip = int(params.get("skip", 21))
    top_n = int(params.get("top_n", 6))
    rebal = int(params.get("rebalance", 21))

    lagged = prices.shift(skip)
    score = lagged / lagged.shift(lookback) - 1.0

    ranks = score.rank(axis=1, ascending=False, na_option="bottom")
    book = (ranks <= top_n).astype(float) / top_n
    book[score.isna().all(axis=1)] = 0.0

    # Hold between rebalances rather than drifting every session.
    held = book.iloc[::rebal].reindex(book.index).ffill().fillna(0.0)
    return held.shift(1).fillna(0.0)
