"""Short-term reversal across the NASDAQ-100's long-standing members.

The anomaly is old and well documented: over horizons of a few days, stocks
that have just fallen tend to bounce and stocks that have just risen tend to
give some back. Lehmann (1990) and Jegadeesh (1990) both measured it on US
equities; it is usually read as compensation for supplying liquidity to
whoever was forced to trade.

The construction here is the long-only version a retail researcher would
actually build: rank the universe on its trailing few-day return, hold the
worst performers equal weight, and re-rank every week.

Read only the `prices` frame handed in. An adapter that closes over the
original DataFrame means the leakage test's corruption never reaches the
strategy, and it then reports a clean bill of health it has not earned.
"""

from __future__ import annotations

import pandas as pd


def positions(prices: pd.DataFrame, **params) -> pd.DataFrame:
    """Weights to hold, one row per row of `prices`.

    Row t is the book held *into* t+1. The signal uses the return up to and
    including t, and the `.shift(1)` at the end is what stops a position
    earning the very move that selected it - which for a reversal strategy is
    not a technicality but the whole question, since the effect being harvested
    lives in exactly those next few sessions.
    """
    lookback = int(params.get("lookback", 5))
    top_n = int(params.get("top_n", 8))
    rebal = int(params.get("rebalance", 5))

    recent = prices / prices.shift(lookback) - 1.0

    # Buy the losers: ascending rank puts the worst recent performer first.
    ranks = recent.rank(axis=1, ascending=True, na_option="bottom")
    book = (ranks <= top_n).astype(float) / top_n
    book[recent.isna().all(axis=1)] = 0.0

    held = book.iloc[::rebal].reindex(book.index).ffill().fillna(0.0)
    return held.shift(1).fillna(0.0)
