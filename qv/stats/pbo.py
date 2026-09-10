"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Bailey, Borwein, Lopez de Prado and Zhu (2015).

Where the Deflated Sharpe asks whether the *winner* clears a selection-adjusted
bar, PBO asks a different and complementary question: is the *selection
procedure itself* any good? Split the sample into ``S`` blocks, take every way
of choosing half of them as in-sample, pick the best trial in-sample, and see
where it lands out-of-sample. If the in-sample winner is a coin flip
out-of-sample, the search has been fitting noise no matter how good its winner
looks.

``PBO = P(the in-sample best ranks in the bottom half out-of-sample)``. Above
0.5 means the procedure is worse than useless - it actively selects strategies
that underperform.

The implementation is exact rather than sampled: all ``C(S, S/2)``
combinations are evaluated. It stays fast by reducing each block to its count,
sum and sum of squares, so a combination Sharpe is a matrix product rather than
a pass over the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

__all__ = ["PBOResult", "combinatorially_symmetric_cv"]

#: Cap on floats held at once while evaluating combinations, so a wide trial
#: matrix cannot quietly allocate gigabytes.
_CHUNK_CELLS = 4_000_000


@dataclass(frozen=True)
class PBOResult:
    """Outcome of the CSCV procedure."""

    pbo: float
    logits: np.ndarray
    is_best_sharpe: np.ndarray
    oos_of_is_best: np.ndarray
    degradation_slope: float
    degradation_intercept: float
    probability_of_loss: float
    n_splits: int
    n_combinations: int
    n_trials: int
    n_obs: int
    note: str | None = None

    @property
    def median_oos_rank(self) -> float:
        """Median relative rank of the in-sample winner out-of-sample.

        0.5 is a coin flip. Anything at or below it means the search has no
        demonstrated ability to pick a strategy that keeps working.
        """
        return float(np.median(1.0 / (1.0 + np.exp(-self.logits))))

    @property
    def overfit(self) -> bool:
        return self.pbo > 0.5

    @property
    def degradation_slope_is_diagnostic(self) -> bool:
        """Whether the degradation slope can be read as evidence at all.

        Almost always ``False``, and saying so is the point. CSCV partitions
        one fixed sample, so a period where the winner did unusually well
        in-sample is *by construction* absent from its out-of-sample half. That
        induces a negative slope even for a strategy with a genuine, stable
        edge - in testing, a planted real edge produces a slope near -1.0 while
        pure noise produces about -0.5, so the sign and even the ordering are
        the opposite of intuitive.

        Read :attr:`pbo` and :attr:`probability_of_loss` instead. The slope and
        the scatter behind it are kept because the scatter plot is genuinely
        informative to look at, not because the fitted coefficient is.
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "pbo": self.pbo,
            "overfit": self.overfit,
            "median_oos_rank": self.median_oos_rank,
            "degradation_slope": self.degradation_slope,
            "degradation_intercept": self.degradation_intercept,
            "probability_of_loss": self.probability_of_loss,
            "n_splits": self.n_splits,
            "n_combinations": self.n_combinations,
            "n_trials": self.n_trials,
            "n_obs": self.n_obs,
            "note": self.note,
        }


def _block_statistics(matrix: np.ndarray, n_splits: int):
    """Reduce each row-block to (count, sum, sum of squares) per trial.

    Any Sharpe over a union of blocks is then recoverable from these, which is
    what makes evaluating thousands of combinations cheap.
    """
    n_obs = matrix.shape[0]
    edges = np.linspace(0, n_obs, n_splits + 1).astype(int)
    counts = np.empty(n_splits)
    sums = np.empty((n_splits, matrix.shape[1]))
    sumsqs = np.empty((n_splits, matrix.shape[1]))
    for s in range(n_splits):
        block = matrix[edges[s] : edges[s + 1]]
        counts[s] = block.shape[0]
        sums[s] = block.sum(axis=0)
        sumsqs[s] = (block**2).sum(axis=0)
    return counts, sums, sumsqs


def _sharpes_from_moments(n: np.ndarray, total: np.ndarray, total_sq: np.ndarray) -> np.ndarray:
    """Vectorised Sharpe from block-aggregated moments. ``ddof=1``.

    The variance floor is relative to the mean square rather than an absolute
    ``> 0``: cancellation in ``E[x^2] - n*mean^2`` leaves a constant block with
    a variance around ``1e-33`` instead of zero, and the resulting Sharpe near
    ``1e14`` would win every combination it appeared in.
    """
    n_col = n[:, None]
    mean = total / n_col
    variance = (total_sq - n_col * mean**2) / (n_col - 1.0)
    floor = 1e-20 * np.maximum(total_sq / n_col, 1e-300)
    valid = variance > floor
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(valid, mean / np.sqrt(np.where(valid, variance, 1.0)), np.nan)


def combinatorially_symmetric_cv(
    trial_returns, n_splits: int = 16
) -> PBOResult:
    """Run CSCV on a ``(T, N)`` matrix of trial returns.

    ``n_splits`` must be even; 16 gives 12,870 combinations, which is the value
    used throughout the source paper and a reasonable default. Larger values
    cost combinatorially more for little extra resolution.

    Requires at least 2 trials, and enough observations that each block holds
    at least two rows - a Sharpe over a single row is undefined.
    """
    matrix = np.asarray(
        getattr(trial_returns, "to_numpy", lambda: trial_returns)(), dtype=float
    )
    if matrix.ndim != 2:
        raise ValueError(f"trial matrix must be 2-D (T x N), got shape {matrix.shape}")
    n_obs, n_trials = matrix.shape
    if n_trials < 2:
        raise ValueError(f"CSCV needs at least 2 trials, got {n_trials}")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even number of at least 2, got {n_splits}")
    if n_obs < 2 * n_splits:
        raise ValueError(
            f"CSCV with {n_splits} splits needs at least {2 * n_splits} observations "
            f"so every block holds at least 2 rows, got {n_obs}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("trial matrix contains non-finite values")

    counts, sums, sumsqs = _block_statistics(matrix, n_splits)
    total_count, total_sum, total_sumsq = counts.sum(), sums.sum(axis=0), sumsqs.sum(axis=0)

    combos = list(combinations(range(n_splits), n_splits // 2))
    n_combos = len(combos)
    membership = np.zeros((n_combos, n_splits))
    for i, combo in enumerate(combos):
        membership[i, list(combo)] = 1.0

    logits = np.empty(n_combos)
    is_best = np.empty(n_combos)
    oos_best = np.empty(n_combos)

    chunk = max(1, _CHUNK_CELLS // max(n_trials, 1))
    for start in range(0, n_combos, chunk):
        block = membership[start : start + chunk]

        n_is = block @ counts
        sharpe_is = _sharpes_from_moments(n_is, block @ sums, block @ sumsqs)
        sharpe_oos = _sharpes_from_moments(
            total_count - n_is,
            total_sum - (block @ sums),
            total_sumsq - (block @ sumsqs),
        )

        # A trial with no in-sample variance cannot be the winner.
        best = np.nanargmax(np.where(np.isfinite(sharpe_is), sharpe_is, -np.inf), axis=1)
        rows = np.arange(block.shape[0])
        chosen_oos = sharpe_oos[rows, best]

        # Relative rank of the winner among all trials out-of-sample. Trials
        # whose OOS Sharpe is undefined are excluded from the denominator.
        valid = np.isfinite(sharpe_oos)
        n_valid = valid.sum(axis=1)
        beaten = (np.where(valid, sharpe_oos, np.inf) < chosen_oos[:, None]).sum(axis=1)
        omega = (beaten + 1.0) / (n_valid + 1.0)
        omega = np.clip(omega, 1e-12, 1 - 1e-12)

        sl = slice(start, start + block.shape[0])
        logits[sl] = np.log(omega / (1.0 - omega))
        is_best[sl] = sharpe_is[rows, best]
        oos_best[sl] = chosen_oos

    usable = np.isfinite(logits) & np.isfinite(is_best) & np.isfinite(oos_best)
    if not usable.any():
        raise ValueError("no combination produced a usable in-sample and out-of-sample Sharpe")

    logits, is_best, oos_best = logits[usable], is_best[usable], oos_best[usable]

    # Performance degradation: OOS Sharpe of the winner against its IS Sharpe.
    # See PBOResult.degradation_slope for why the sign of this is not the
    # diagnostic it looks like.
    if np.ptp(is_best) > 0:
        slope, intercept = np.polyfit(is_best, oos_best, 1)
    else:
        slope, intercept = math.nan, float(np.mean(oos_best))

    dropped = int((~usable).sum())
    return PBOResult(
        pbo=float(np.mean(logits <= 0.0)),
        logits=logits,
        is_best_sharpe=is_best,
        oos_of_is_best=oos_best,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        probability_of_loss=float(np.mean(oos_best < 0.0)),
        n_splits=n_splits,
        n_combinations=int(logits.size),
        n_trials=n_trials,
        n_obs=n_obs,
        note=None if dropped == 0 else f"{dropped} combinations were unusable and dropped",
    )
