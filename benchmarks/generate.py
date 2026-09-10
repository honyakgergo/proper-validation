"""Synthetic strategies whose true edge is known by construction.

This is the most important thing in the repository. Without it, "my tool
detects overfitting" is a claim; with it, the claim becomes a measurement that
a reviewer can reproduce.

Every label below has a ground truth fixed by how it was built, and
``genuine_weak`` exists specifically to keep the tool honest in the other
direction: a validator that condemns everything is not a validator, it is a
pessimism generator, and the only way to prove otherwise is to plant a real
edge and require that it survive.

The generators take a seed and are pure. Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["LabelledStrategy", "GENERATORS", "generate", "generate_all", "LABELS"]


@dataclass
class LabelledStrategy:
    """A synthetic strategy plus the truth about it."""

    label: str
    returns: np.ndarray
    has_edge: bool
    description: str
    positions: np.ndarray | None = None
    asset_returns: np.ndarray | None = None
    asset_class: str | None = None
    trial_returns: np.ndarray | None = None
    n_trials: int | None = None
    strategy: Callable[[np.ndarray], np.ndarray] | None = None
    strategy_data: np.ndarray | None = None
    expected_findings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return int(self.returns.size)


def _market(n: int, rng: np.random.Generator, mu: float = 0.0003, sd: float = 0.011) -> np.ndarray:
    """A plausible daily equity index: mild drift, volatility clustering, fat tails."""
    vol = np.zeros(n)
    vol[0] = sd
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        vol[t] = np.sqrt(0.000002 + 0.08 * (vol[t - 1] * shocks[t - 1]) ** 2 + 0.90 * vol[t - 1] ** 2)
    return mu + vol * rng.standard_t(df=6, size=n) / np.sqrt(6 / 4)


def null_mined(n: int = 2000, n_rules: int = 400, seed: int = 0) -> LabelledStrategy:
    """Mine many random rules on one market and keep the best. True edge: zero.

    The flagship demonstration. Every rule is a random sign pattern with no
    predictive content whatsoever, so the winner's entire performance is
    selection.
    """
    rng = np.random.default_rng(seed)
    market = _market(n, rng)
    signals = rng.choice([-1.0, 1.0], size=(n, n_rules))
    # Hold each position for a few days so the rules look like real strategies
    # rather than daily coin flips.
    hold = 5
    for start in range(0, n, hold):
        signals[start : start + hold] = signals[start]
    trials = signals * market[:, None]

    sharpes = trials.mean(axis=0) / trials.std(axis=0, ddof=1)
    best = int(np.argmax(sharpes))
    return LabelledStrategy(
        label="null_mined",
        returns=trials[:, best],
        positions=signals[:, best],
        asset_returns=market,
        trial_returns=trials,
        n_trials=n_rules,
        has_edge=False,
        description=f"best of {n_rules} random sign rules on a simulated index",
        expected_findings=("SELECT-DEFLATED-SHARPE-FAILS",),
        metadata={"winning_rule": best, "in_sample_sharpe": float(sharpes[best])},
    )


def null_pure(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """A single strategy that is pure noise, honestly reported. True edge: zero."""
    rng = np.random.default_rng(seed + 1000)
    market = _market(n, rng)
    positions = np.repeat(rng.choice([0.0, 1.0], size=n // 5 + 1), 5)[:n]
    return LabelledStrategy(
        label="null_pure",
        returns=positions * market,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        has_edge=False,
        description="one random-timing strategy, no search",
        expected_findings=(),
    )


def leaky_shift(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """Signal reads tomorrow's return. True edge: zero (it is unimplementable)."""
    rng = np.random.default_rng(seed + 2000)
    market = _market(n, rng)

    def strategy(data: np.ndarray) -> np.ndarray:
        x = data[:, 0]
        out = np.zeros_like(x)
        out[:-1] = np.sign(x[1:])  # shift(-1): the classic
        return out

    data = market[:, None]
    positions = strategy(data)

    # The signal at t predicts the return at t+1, so the backtest books it
    # against the return at t+1 - which is exactly the shape of the real bug:
    # `signal = close.shift(-1) > close` scored against the forward return. The
    # position knows the very thing it is scored on, so the equity curve is
    # essentially the absolute value of the market.
    returns = np.zeros_like(market)
    returns[:-1] = positions[:-1] * market[1:]

    return LabelledStrategy(
        label="leaky_shift",
        returns=returns,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        strategy=strategy,
        strategy_data=data,
        has_edge=False,
        description="position takes the sign of the next period return",
        expected_findings=("LEAK-BEHAVIOURAL",),
    )


def leaky_scaler(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """Threshold set from the full-sample mean and standard deviation.

    A milder leak than a negative shift and far more common: the signal uses a
    z-score computed over the whole sample, so early periods know the future
    distribution.
    """
    rng = np.random.default_rng(seed + 3000)
    market = _market(n, rng)

    def strategy(data: np.ndarray) -> np.ndarray:
        x = data[:, 0]
        # Full-sample standardisation: legitimate-looking, and a leak.
        z = (x - x.mean()) / x.std()
        out = np.zeros_like(x)
        out[1:] = np.where(z[:-1] < -1.0, 1.0, 0.0)
        return out

    data = market[:, None]
    positions = strategy(data)
    return LabelledStrategy(
        label="leaky_scaler",
        returns=positions * market,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        strategy=strategy,
        strategy_data=data,
        has_edge=False,
        description="mean-reversion threshold from a full-sample z-score",
        expected_findings=("LEAK-BEHAVIOURAL",),
    )


def regime_fluke(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """Works in one window and nowhere else. True edge: near zero overall."""
    rng = np.random.default_rng(seed + 4000)
    market = _market(n, rng)
    positions = np.repeat(rng.choice([0.0, 1.0], size=n // 5 + 1), 5)[:n]
    returns = positions * market
    window = slice(int(n * 0.25), int(n * 0.45))
    returns = returns.copy()
    returns[window] += 0.006
    return LabelledStrategy(
        label="regime_fluke",
        returns=returns,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        has_edge=False,
        description="an edge confined to one 20% window of the sample",
        expected_findings=("ROBUST-START-DATE-SENSITIVE",),
    )


def cost_fragile(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """A real gross edge that dies above a few basis points. True net edge: zero."""
    rng = np.random.default_rng(seed + 5000)
    market = _market(n, rng, mu=0.0)
    # Trade nearly every day for a small genuine gross edge, so break-even
    # lands a couple of basis points below what the asset class charges.
    positions = rng.choice([0.0, 1.0], size=n)
    gross = positions * market + positions * 0.00007
    return LabelledStrategy(
        label="cost_fragile",
        returns=gross,
        positions=positions,
        asset_returns=market,
        asset_class="us_large_cap_equity",
        n_trials=1,
        has_edge=False,
        description="genuine gross edge, roughly daily turnover, dies inside realistic costs",
        expected_findings=("COST-BELOW-REALISTIC",),
    )


def levered_beta(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """Simply long the market at leverage. True edge: zero alpha."""
    rng = np.random.default_rng(seed + 6000)
    market = _market(n, rng, mu=0.0004)
    positions = np.full(n, 1.6)
    return LabelledStrategy(
        label="levered_beta",
        returns=positions * market,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        has_edge=False,
        description="constant 1.6x long the index, no timing at all",
        # Not ROBUST-NO-TIMING-SKILL: exposure here is literally constant, so
        # the matched-exposure test is correctly skipped rather than shuffling
        # an unchanging series and reporting a p-value near 1 as if it meant
        # something. Attribution is the test that catches this one.
        expected_findings=("ATTR-LEVERED-BETA",),
        metadata={"benchmark": "market"},
    )


def lottery_ticket(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """All the profit in a handful of days. True edge: not repeatable."""
    rng = np.random.default_rng(seed + 7000)
    market = _market(n, rng, mu=0.0)
    positions = np.repeat(rng.choice([0.0, 1.0], size=n // 5 + 1), 5)[:n]
    returns = positions * market
    spikes = rng.choice(n, size=max(3, n // 250), replace=False)
    returns = returns.copy()
    returns[spikes] += 0.10
    return LabelledStrategy(
        label="lottery_ticket",
        returns=returns,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        has_edge=False,
        description="six enormous days supply the entire profit",
        expected_findings=("ROBUST-TOP-DAY-DEPENDENT",),
    )


def genuine_weak(n: int = 2000, seed: int = 0) -> LabelledStrategy:
    """A small, real, persistent edge. **This one must pass.**

    The row that proves the tool is not merely a pessimism generator. The edge
    is deliberately modest - an annualised Sharpe near 1 - because a validator
    that only clears implausibly good strategies is no more useful than one
    that clears nothing.
    """
    rng = np.random.default_rng(seed + 8000)

    # A slow, persistent latent state that genuinely predicts the next return.
    # The position is formed from the state as of t-1, so the strategy is
    # implementable, and the predictability is real rather than a premium for
    # being invested - which matters, because a strategy that merely collects a
    # premium for exposure has no *timing* skill and the matched-exposure test
    # would be right to say so.
    state = np.zeros(n)
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        state[t] = 0.97 * state[t - 1] + shocks[t]
    state /= state.std()

    base = _market(n, rng, mu=0.0)
    predictable = 0.0035 * state
    market = base + np.concatenate([[0.0], predictable[:-1]])

    positions = np.zeros(n)
    positions[1:] = (state[:-1] > 0.25).astype(float)
    returns = positions * market

    return LabelledStrategy(
        label="genuine_weak",
        returns=returns,
        positions=positions,
        asset_returns=market,
        n_trials=1,
        has_edge=True,
        description=(
            "a real, modest, low-turnover timing edge - the signal genuinely predicts "
            "the next return - that must survive the audit"
        ),
        expected_findings=(),
    )


GENERATORS: dict[str, Callable[..., LabelledStrategy]] = {
    "null_mined": null_mined,
    "null_pure": null_pure,
    "leaky_shift": leaky_shift,
    "leaky_scaler": leaky_scaler,
    "regime_fluke": regime_fluke,
    "cost_fragile": cost_fragile,
    "levered_beta": levered_beta,
    "lottery_ticket": lottery_ticket,
    "genuine_weak": genuine_weak,
}

LABELS = tuple(GENERATORS)


def generate(label: str, seed: int = 0, **kwargs) -> LabelledStrategy:
    """Build one labelled strategy by name."""
    if label not in GENERATORS:
        raise KeyError(f"unknown label {label!r}; expected one of {LABELS}")
    return GENERATORS[label](seed=seed, **kwargs)


def generate_all(seed: int = 0, **kwargs) -> list[LabelledStrategy]:
    """One replication of every label."""
    return [generate(label, seed=seed, **kwargs) for label in LABELS]
