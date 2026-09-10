"""Empirical distribution of the maximum Sharpe under the null.

The Deflated Sharpe Ratio prices selection bias with a closed form that makes
two assumptions worth checking: that trial Sharpes are normally distributed,
and that the trials are **independent**. Neither is usually true. A parameter
grid produces heavily overlapping strategies, and returns have fat tails.

So simulate the null directly instead. Demean every trial to impose zero edge,
resample the rows jointly with the stationary bootstrap - which preserves both
the serial dependence within each trial and the cross-sectional correlation
between them - and record the maximum Sharpe across trials in each replication.

**This is a validity check, not a second verdict.** The Deflated Sharpe stays
the headline number. What this adds is the ability to notice when the closed
form is being asked to do something it cannot, and to draw the null
distribution that makes selection bias legible at a glance.

Read the two ways it can disagree:

* empirical **below** analytic - the trials were correlated, so there were
  fewer effective bets than ``n_trials`` suggests and DSR over-penalises.
* empirical **above** analytic - tails are fatter than normal, so DSR
  under-penalises and the real bar is higher than it claims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtri_exp

from qv.stats.bootstrap import politis_white_block_length, stationary_bootstrap_indices
from qv.stats.deflated import expected_max_sharpe
from qv.stats.moments import columnwise_sharpe

__all__ = [
    "MaxSharpeNullResult",
    "empirical_max_sharpe_null",
    "gaussian_max_sharpe_null",
]

#: Ratio of empirical to analytic expected maximum outside which the closed
#: form is treated as not describing this search. Ten per cent either way is
#: comfortably beyond simulation noise at the default replication count.
_AGREEMENT_TOLERANCE = 0.10


@dataclass(frozen=True)
class MaxSharpeNullResult:
    """Where the observed Sharpe sits against a simulated null of the maximum."""

    observed_sharpe: float
    null_distribution: np.ndarray
    analytic_expected_max: float
    n_trials: int
    n_sims: int
    method: str
    block_length: float | None = None
    note: str | None = None

    @property
    def empirical_expected_max(self) -> float:
        return float(np.mean(self.null_distribution))

    @property
    def percentile(self) -> float:
        """Fraction of null maxima the observed Sharpe beats."""
        return float(np.mean(self.null_distribution <= self.observed_sharpe))

    @property
    def p_value(self) -> float:
        """Probability of a maximum this large from a search with no edge.

        Uses the ``(hits + 1) / (n + 1)`` correction so a p-value is never
        exactly zero: no finite simulation can rule out a larger draw.
        """
        hits = int(np.sum(self.null_distribution >= self.observed_sharpe))
        return (hits + 1) / (self.null_distribution.size + 1)

    @property
    def agreement_ratio(self) -> float:
        """Empirical expected maximum divided by the analytic one."""
        if self.analytic_expected_max == 0:
            return math.nan
        return self.empirical_expected_max / self.analytic_expected_max

    @property
    def analytic_disagrees(self) -> bool:
        """Is the closed form materially wrong about this search?"""
        ratio = self.agreement_ratio
        return math.isfinite(ratio) and abs(ratio - 1.0) > _AGREEMENT_TOLERANCE

    @property
    def disagreement_direction(self) -> str | None:
        """Which way the closed form errs, in plain terms."""
        if not self.analytic_disagrees:
            return None
        if self.agreement_ratio < 1.0:
            return (
                "trials were correlated, so the effective number of independent bets is "
                "below n_trials and the analytic Deflated Sharpe over-penalises"
            )
        return (
            "trial Sharpes have fatter tails than the normal approximation assumes, so "
            "the analytic Deflated Sharpe under-penalises and the real bar is higher"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_sharpe": self.observed_sharpe,
            "analytic_expected_max": self.analytic_expected_max,
            "empirical_expected_max": self.empirical_expected_max,
            "agreement_ratio": self.agreement_ratio,
            "analytic_disagrees": self.analytic_disagrees,
            "disagreement_direction": self.disagreement_direction,
            "percentile": self.percentile,
            "p_value": self.p_value,
            "n_trials": self.n_trials,
            "n_sims": self.n_sims,
            "method": self.method,
            "block_length": self.block_length,
            "note": self.note,
        }


def _as_trial_matrix(trial_returns) -> np.ndarray:
    matrix = np.asarray(
        getattr(trial_returns, "to_numpy", lambda: trial_returns)(), dtype=float
    )
    if matrix.ndim != 2:
        raise ValueError(f"trial matrix must be 2-D (T x N), got shape {matrix.shape}")
    if matrix.shape[0] < 4:
        raise ValueError(f"trial matrix needs at least 4 rows, got {matrix.shape[0]}")
    if matrix.shape[1] < 2:
        raise ValueError(f"trial matrix needs at least 2 columns, got {matrix.shape[1]}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("trial matrix contains non-finite values")
    return matrix


def empirical_max_sharpe_null(
    trial_returns,
    observed_sharpe: float | None = None,
    n_sims: int = 2000,
    block_length: float | None = None,
    seed: int | None = 0,
) -> MaxSharpeNullResult:
    """Simulate the maximum Sharpe across trials under a zero-edge null.

    ``trial_returns`` is the ``(T, N)`` matrix of per-period returns, one
    column per configuration the search examined. Columns are demeaned to
    impose the null; **rows are resampled jointly**, which is what preserves
    the correlation between trials that the analytic formula ignores.

    ``observed_sharpe`` defaults to the best trial in the matrix, which is the
    usual case: the reported strategy *is* the winner of the search.
    """
    matrix = _as_trial_matrix(trial_returns)
    n_obs, n_trials = matrix.shape
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")

    # Impose the null: every trial has exactly zero mean return.
    centred = matrix - matrix.mean(axis=0, keepdims=True)

    if observed_sharpe is None:
        observed_sharpe = float(np.nanmax(columnwise_sharpe(matrix)))

    # One block length for the whole panel, from the equal-weighted composite:
    # the trials share a common time axis, so they share a dependence structure.
    b = politis_white_block_length(centred.mean(axis=1)) if block_length is None else float(
        block_length
    )

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n_obs, b, n_sims, rng)

    maxima = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        sharpes = columnwise_sharpe(centred[idx[i]])
        maxima[i] = np.nanmax(sharpes) if np.any(np.isfinite(sharpes)) else np.nan

    finite = maxima[np.isfinite(maxima)]
    if finite.size == 0:
        raise ValueError("every replication produced a non-finite maximum Sharpe")

    observed_trial_sharpes = columnwise_sharpe(matrix)
    trial_std = float(np.std(observed_trial_sharpes[np.isfinite(observed_trial_sharpes)], ddof=1))

    return MaxSharpeNullResult(
        observed_sharpe=float(observed_sharpe),
        null_distribution=finite,
        analytic_expected_max=expected_max_sharpe(n_trials, trial_std),
        n_trials=n_trials,
        n_sims=int(finite.size),
        method="stationary bootstrap of demeaned trials, rows resampled jointly",
        block_length=b,
        note=(
            None
            if finite.size == n_sims
            else f"{n_sims - finite.size} of {n_sims} replications were non-finite"
        ),
    )



def gaussian_max_sharpe_null(
    observed_sharpe: float,
    n_trials: int,
    trial_sharpe_std: float,
    n_sims: int = 20_000,
    seed: int | None = 0,
) -> MaxSharpeNullResult:
    """Fallback when no trial matrix exists, only ``n_trials`` and a dispersion.

    Models ``n_trials`` independent normal Sharpes per replication and keeps the
    maximum. This reproduces the assumptions behind the analytic formula rather
    than testing them, so it cannot detect correlation or fat tails - it exists
    to draw the null-distribution chart when the matrix is unavailable, and its
    agreement with the closed form is a check on the implementation, not on the
    data.

    The maximum is drawn **directly** rather than by simulating the trials and
    taking their maximum. The two are the same distribution - the maximum of
    ``N`` iid standard normals has CDF ``Phi(x)**N``, so it can be sampled
    exactly as ``Phi^-1(U**(1/N))`` - but the second form allocates an
    ``(n_sims, n_trials)`` array, and this function is reached precisely when
    the trial count is large. At the 20,000 replications used by default, a
    search of 100,000 configurations asked for a single 14.9 GiB array, which
    was enough to get the test suite OOM-killed on a 16 GB CI runner. Drawing
    the maximum is O(n_sims) instead of O(n_sims * n_trials).

    ``ndtri_exp`` evaluates ``Phi^-1(exp(y))`` from the *log* probability, which
    matters here: for a large ``N``, ``U**(1/N)`` rounds to 1.0 in float64 and
    the upper tail - the only part of this distribution anyone reads - would be
    quantised away.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if trial_sharpe_std < 0:
        raise ValueError(f"trial_sharpe_std must be non-negative, got {trial_sharpe_std}")
    if n_sims < 1:
        raise ValueError(f"n_sims must be at least 1, got {n_sims}")

    rng = np.random.default_rng(seed)
    # log(U)/N for U uniform on (0, 1], drawn as a negative exponential so that
    # log(0) never arises. Clipped just below zero because ndtri_exp(0) is
    # +inf: U == 1 is a draw the generator can legitimately produce.
    log_u_over_n = np.minimum(-rng.standard_exponential(n_sims) / n_trials, -np.finfo(float).tiny)
    maxima = trial_sharpe_std * ndtri_exp(log_u_over_n)

    return MaxSharpeNullResult(
        observed_sharpe=float(observed_sharpe),
        null_distribution=maxima,
        analytic_expected_max=expected_max_sharpe(n_trials, trial_sharpe_std),
        n_trials=n_trials,
        n_sims=n_sims,
        method="independent Gaussian trial Sharpes (no trial matrix available)",
        block_length=None,
        note=(
            "assumes independent, normally distributed trial Sharpes - the same "
            "assumptions as the analytic formula, so this cannot validate them"
        ),
    )
