"""Dual momentum with volatility targeting.

A deliberately more serious strategy than the plain sector rotation next door,
built from three effects that are separately documented in the literature
rather than discovered by searching this data:

* **Relative momentum** - rank a cross-section by trailing return and hold the
  leaders. Jegadeesh and Titman (1993); Moskowitz, Ooi and Pedersen (2012).
* **Absolute momentum** - only hold a leader if it also beats cash on the same
  lookback, otherwise sit in bonds. Antonacci (2014). This is what turns a
  long-only rotation into something that can step aside in a bear market, and
  it is the component that changes the drawdown profile rather than the return.
* **Volatility targeting** - scale gross exposure by the ratio of a target
  volatility to recent realised volatility. Moreira and Muir (2017). Levers
  down into turbulence and up into calm.

The universe adds bonds and gold to the nine sector funds, because an absolute
momentum filter needs somewhere to go. Without a defensive leg the filter can
only move to cash, which throws away most of its benefit.

Everything is computed from trailing windows and traded from the session after
the signal, so the construction is honest. Whether it *works* is the audit's
question, not this file's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "RISK_ASSETS",
    "DEFENSIVE_ASSETS",
    "UNIVERSE",
    "momentum_score",
    "dual_momentum_positions",
    "strategy_returns",
]

#: The nine original SPDR sector funds, all trading continuously since 1998.
RISK_ASSETS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")

#: Where the absolute-momentum filter retreats to. IEF is intermediate
#: Treasuries and GLD is gold; both trade from 2004, which is why the sample
#: starts in 2005.
DEFENSIVE_ASSETS = ("IEF", "GLD")

UNIVERSE = RISK_ASSETS + DEFENSIVE_ASSETS


def momentum_score(prices: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Trailing return over ``lookback`` sessions, ending ``skip`` sessions ago.

    The skip omits the most recent month because short-horizon returns reverse;
    including it measurably weakens the effect.
    """
    if lookback < 2:
        raise ValueError(f"lookback must be at least 2 sessions, got {lookback}")
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    recent = prices.shift(skip)
    return recent / recent.shift(lookback) - 1.0


def realised_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Trailing annualised volatility, excluding the current session."""
    return returns.shift(1).rolling(window).std() * np.sqrt(252)


def dual_momentum_positions(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 3,
    vol_target: float | None = 0.10,
    vol_window: int = 63,
    max_leverage: float = 1.5,
    risk_assets: tuple[str, ...] = RISK_ASSETS,
    defensive_assets: tuple[str, ...] = DEFENSIVE_ASSETS,
) -> pd.DataFrame:
    """Weights for the full universe, rebalanced monthly.

    Three stages, in order:

    1. Rank the risk assets by momentum and take the top ``top_n``.
    2. Drop any of them whose momentum is negative - the absolute filter - and
       put the freed weight into the best-performing defensive asset.
    3. Scale the whole book so trailing volatility meets ``vol_target``,
       capped at ``max_leverage``.

    Set ``vol_target=None`` to leave the book unscaled, which is how the grid
    isolates what the volatility overlay contributes.
    """
    if top_n < 1 or top_n > len(risk_assets):
        raise ValueError(f"top_n must be between 1 and {len(risk_assets)}, got {top_n}")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be positive, got {max_leverage}")

    scores = momentum_score(prices, lookback, skip)
    risk_scores = scores[list(risk_assets)]

    ranks = risk_scores.rank(axis=1, ascending=False, na_option="bottom")
    selected = (ranks <= top_n) & risk_scores.notna()

    # Absolute momentum: a selected asset only earns its weight if its own
    # trailing return is positive. This is the piece that lets the strategy
    # step aside rather than rotate into the least-bad sector.
    held = selected & (risk_scores > 0)

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[list(risk_assets)] = held.astype(float) / top_n

    # Whatever the filter refused goes to the strongest defensive asset, or to
    # cash if neither is positive.
    defensive_scores = scores[list(defensive_assets)]
    # Before the lookback window fills, every score is NaN and idxmax has
    # nothing to choose between. Filling with -inf keeps the column selection
    # well defined; the `defensive_ok` gate below then zeroes those rows, so
    # the strategy simply holds nothing until it has enough history.
    best_defensive = defensive_scores.fillna(-np.inf).idxmax(axis=1)
    defensive_ok = defensive_scores.max(axis=1).fillna(-np.inf) > 0
    freed = 1.0 - weights[list(risk_assets)].sum(axis=1)
    for asset in defensive_assets:
        weights[asset] = np.where(
            defensive_ok & (best_defensive == asset), freed, 0.0
        )

    if vol_target is not None:
        # Scale on the book the weights imply, measured on trailing data only.
        asset_returns = prices.pct_change().fillna(0.0)
        gross = (weights * asset_returns).sum(axis=1)
        trailing = realised_volatility(gross, vol_window)
        scale = (vol_target / trailing).clip(upper=max_leverage)
        weights = weights.mul(scale.fillna(1.0).clip(lower=0.0), axis=0)

    month = pd.Series(prices.index.to_period("M"), index=prices.index)
    is_rebalance = month.ne(month.shift(-1)).to_numpy()
    mask = pd.DataFrame(
        np.repeat(is_rebalance[:, None], weights.shape[1], axis=1),
        index=weights.index,
        columns=weights.columns,
    )
    held_weights = weights.where(mask).ffill().fillna(0.0)

    # Traded from the next session, so no weight earns the return that set it.
    return held_weights.shift(1).fillna(0.0)


def strategy_returns(
    prices: pd.DataFrame, positions: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Portfolio returns and the per-asset return frame behind them."""
    asset_returns = prices.pct_change().fillna(0.0)
    aligned = positions.reindex_like(asset_returns).fillna(0.0)
    return (aligned * asset_returns).sum(axis=1), asset_returns


# The parameter grid these strategies searched is declared in
# `research_manifest.yaml` under `search.axes`, and re-run by
# `qv.trials.run_parameter_grid`. It used to be a `parameter_grid`/`run_grid`
# pair here as well; two copies of the same list invites them to disagree about
# what was actually searched, and the manifest is the one a reader can check.
