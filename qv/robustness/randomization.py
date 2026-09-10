"""Randomisation tests: does the signal carry information?

Two tests, two questions.

* **Sign-flip permutation** - flip the sign of randomly chosen blocks of
  returns. Under the null that the strategy has no edge, the observed Sharpe
  should be unremarkable against this distribution. Blocks rather than
  individual observations, so serial dependence survives the permutation.
* **Matched-exposure random entry** - replace the timing with random timing
  that has the *same* average exposure and roughly the same turnover, and see
  whether the real signal beats it. This is what catches a strategy whose
  apparent edge is really just being long a rising market.

A plain timing-shuffle test is deliberately absent: it is the matched-exposure
test with the exposure left uncontrolled, which makes it strictly less
informative about the case that actually matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.stats.moments import as_returns_array
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity

__all__ = [
    "RandomisationResult",
    "sign_flip_test",
    "matched_exposure_test",
]


@dataclass(frozen=True)
class RandomisationResult:
    """Observed statistic against a simulated null distribution."""

    observed: float
    null_distribution: np.ndarray
    n_sims: int
    test: str
    description: str

    @property
    def percentile(self) -> float:
        return float(np.mean(self.null_distribution <= self.observed))

    @property
    def p_value(self) -> float:
        """One-sided p-value with the ``(hits + 1) / (n + 1)`` correction."""
        hits = int(np.sum(self.null_distribution >= self.observed))
        return (hits + 1) / (self.null_distribution.size + 1)

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null_distribution))

    @property
    def significant_at_5pct(self) -> bool:
        return self.p_value < 0.05

    @property
    def severity(self) -> Severity:
        if self.p_value < 0.01:
            return Severity.INFO
        if self.p_value < 0.05:
            return Severity.LOW
        if self.p_value < 0.20:
            return Severity.MEDIUM
        return Severity.HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "description": self.description,
            "observed": self.observed,
            "null_mean": self.null_mean,
            "percentile": self.percentile,
            "p_value": self.p_value,
            "significant_at_5pct": self.significant_at_5pct,
            "severity": self.severity.name.lower(),
            "n_sims": self.n_sims,
        }


def sign_flip_test(
    returns, n_sims: int = 2000, block_length: int = 21, seed: int | None = 0
) -> RandomisationResult:
    """Flip the sign of random blocks of returns and recompute the Sharpe.

    Under the null of no edge, positive and negative runs are exchangeable, so
    the observed Sharpe should sit inside this distribution. Flipping whole
    blocks rather than single observations preserves volatility clustering -
    an observation-level flip would destroy it and produce a null that is too
    narrow, overstating significance.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 20:
        raise ValueError(f"sign-flip test needs at least 20 observations, got {n}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")
    if block_length < 1:
        raise ValueError(f"block_length must be at least 1, got {block_length}")

    n_blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)

    draws = np.empty(n_sims)
    for i in range(n_sims):
        signs = rng.choice([-1.0, 1.0], size=n_blocks)
        expanded = np.repeat(signs, block_length)[:n]
        draws[i] = sharpe_ratio(x * expanded)

    finite = draws[np.isfinite(draws)]
    return RandomisationResult(
        observed=sharpe_ratio(x),
        null_distribution=finite,
        n_sims=int(finite.size),
        test="sign-flip",
        description=(
            f"signs of {block_length}-period blocks randomly flipped, preserving "
            "volatility clustering"
        ),
    )


def matched_exposure_test(
    asset_returns,
    positions,
    n_sims: int = 2000,
    seed: int | None = 0,
) -> RandomisationResult:
    """Compare the strategy against random timing at the same exposure.

    Each replication permutes the position series in blocks, which preserves
    both the average exposure and the holding-period structure while
    destroying the *timing* - so what remains is whatever the strategy earns
    simply by being in the market that much.

    This is the test that exposes levered beta. A strategy that is 70% long on
    average during a bull market will show a fine Sharpe; if random timing at
    70% average exposure shows the same Sharpe, the signal contributed nothing.

    ``asset_returns`` is the return of the traded instrument, not of the
    strategy. Strategy return in period ``t`` is ``position[t] * asset[t]``.
    """
    asset = as_returns_array(asset_returns)
    pos = np.asarray(getattr(positions, "to_numpy", lambda: positions)(), dtype=float)
    if pos.ndim != 1:
        raise ValueError(f"positions must be 1-D for this test, got shape {pos.shape}")
    if pos.shape != asset.shape:
        raise ValueError(
            f"positions have {pos.shape} but asset returns have {asset.shape}; "
            "they must describe the same time axis"
        )
    if asset.size < 20:
        raise ValueError(f"matched-exposure test needs at least 20 observations, got {asset.size}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")

    n = asset.size
    # Block size from the average holding period, so shuffled books trade at a
    # comparable rate to the real one rather than far more often.
    changes = int(np.sum(np.abs(np.diff(pos)) > 0))
    block = max(1, min(n // 4, n // max(changes, 1)))
    n_blocks = int(np.ceil(n / block))
    padded = np.concatenate([pos, np.zeros(n_blocks * block - n)])
    blocks = padded.reshape(n_blocks, block)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_sims)
    for i in range(n_sims):
        shuffled = blocks[rng.permutation(n_blocks)].ravel()[:n]
        draws[i] = sharpe_ratio(shuffled * asset)

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("every replication produced a non-finite Sharpe")

    return RandomisationResult(
        observed=sharpe_ratio(pos * asset),
        null_distribution=finite,
        n_sims=int(finite.size),
        test="matched-exposure random entry",
        description=(
            f"position blocks of {block} periods randomly reordered, preserving average "
            f"exposure ({float(np.mean(pos)):.3f}) and holding-period structure"
        ),
    )
