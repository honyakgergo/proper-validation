"""A cross-sectional momentum strategy on US sector ETFs - a *test fixture*.

This lived in `case_studies/` as a worked example until the examples were cut
to three genuinely different strategies, at which point it was the odd one out:
a plainer version of the dual-momentum rotation sitting beside it. What the test
suite actually needs from it is a clean, popular construction whose
`momentum_score` can be monkeypatched into leaking, so the perturbation test has
something honest to clear and something dishonest to catch. That is a fixture's
job, so it lives with the fixtures.

Written the way someone would actually write it, not as a strawman. This is
the standard construction from the momentum literature: rank a universe by
trailing return skipping the most recent month, hold the top few equal
weighted, rebalance monthly.

It is deliberately *honest*. Every signal is computed from data up to the
rebalance date and traded from the next session, and there is no full-sample
statistic anywhere. If this were leaky the audit would only be finding my bug,
and the interesting question is what an audit says about a construction that is
genuinely clean and genuinely popular.

Pure functions over DataFrames. No I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["SECTOR_ETFS", "momentum_score", "momentum_positions", "strategy_returns"]

#: The nine original SPDR sector funds. Chosen because they all trade from
#: 1998, so the universe is fixed for the whole sample and there is no
#: survivorship question to argue about - a rare luxury.
SECTOR_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")


def momentum_score(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Trailing return over ``lookback`` sessions, ending ``skip`` sessions ago.

    The skip is not decoration. Short-horizon returns reverse, so the standard
    momentum signal omits the most recent month; including it measurably
    weakens the effect. A 252/21 pair is the familiar "12-1 month" version.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2 sessions, got {lookback}")
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    recent = prices.shift(skip)
    return recent / recent.shift(lookback) - 1.0


def momentum_positions(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 3,
) -> pd.DataFrame:
    """Equal-weight the top ``top_n`` names by momentum, rebalanced monthly.

    Two details carry the honesty of the whole thing:

    * Weights are only allowed to change on the last session of each month and
      are held flat in between. A strategy that silently rebalances daily is a
      different and far more expensive strategy.
    * The result is shifted one session. The ranking observed at the close of
      the rebalance date is traded at the next open, so no position earns the
      return that produced it.
    """
    if top_n < 1 or top_n > prices.shape[1]:
        raise ValueError(
            f"top_n must be between 1 and {prices.shape[1]}, got {top_n}"
        )

    scores = momentum_score(prices, lookback, skip)
    ranks = scores.rank(axis=1, ascending=False, na_option="bottom")
    target = (ranks <= top_n).astype(float) / top_n
    # A row with no usable score should be flat rather than arbitrarily long.
    target[scores.isna().all(axis=1)] = 0.0

    month = pd.Series(prices.index.to_period("M"), index=prices.index)
    is_rebalance = month.ne(month.shift(-1)).to_numpy()
    mask = pd.DataFrame(
        np.repeat(is_rebalance[:, None], target.shape[1], axis=1),
        index=target.index,
        columns=target.columns,
    )
    held = target.where(mask).ffill().fillna(0.0)

    return held.shift(1).fillna(0.0)


def strategy_returns(
    prices: pd.DataFrame, positions: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Portfolio returns and the per-asset return frame that produced them."""
    asset_returns = prices.pct_change().fillna(0.0)
    aligned = positions.reindex_like(asset_returns).fillna(0.0)
    return (aligned * asset_returns).sum(axis=1), asset_returns


# The parameter grid these strategies searched is declared in
# `research_manifest.yaml` under `search.axes`, and re-run by
# `qv.trials.run_parameter_grid`. It used to be a `parameter_grid`/`run_grid`
# pair here as well; two copies of the same list invites them to disagree about
# what was actually searched, and the manifest is the one a reader can check.
