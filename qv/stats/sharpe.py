"""Sharpe ratio, its standard error, and honest annualisation.

The headline of most backtests is an annualised Sharpe produced by multiplying
a periodic Sharpe by ``sqrt(252)`` and reporting no error bar at all. This
module produces three numbers instead of one - periodic, naively annualised,
and autocorrelation-adjusted - each with an interval, so the report can show
the haircut rather than assert it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as _st

from qv.stats.moments import (
    as_returns_array,
    as_trial_matrix,
    columnwise_sharpe,
    max_estimable_lag,
    lo_annualisation_factor,
    naive_annualisation_factor,
    standardised_moments,
)
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Estimate

__all__ = [
    "sharpe_ratio",
    "sharpe_standard_error",
    "sharpe_tstat",
    "columnwise_sharpe_tstat",
    "SharpeResult",
    "analyse_sharpe",
]


def sharpe_ratio(returns, rf_per_period: float = 0.0) -> float:
    """Per-period Sharpe ratio. Not annualised - annualisation is a separate decision.

    Uses the sample standard deviation with ``ddof=1``. Returns ``nan`` for a
    zero-variance series rather than raising, since a constant-return strategy
    is a legitimate (if suspicious) thing to hand this tool.
    """
    x = as_returns_array(returns)
    if x.size < 2:
        raise ValueError(f"Sharpe needs at least 2 observations, got {x.size}")
    sd = float(np.std(x, ddof=1))
    if sd <= 0:
        return math.nan
    return float(np.mean(x) - rf_per_period) / sd


def sharpe_standard_error(returns, rf_per_period: float = 0.0) -> float:
    """Standard error of the periodic Sharpe (Lo 2002; Mertens 2002).

    ``SE = sqrt( (1 + SR^2/2 - g3*SR + ((g4 - 3)/4)*SR^2) / n )``

    where ``g3`` is skewness and ``g4`` is **raw** kurtosis. The higher-moment
    terms matter: fat tails and negative skew - the signature of a short
    volatility strategy - widen this interval well beyond the ``1/sqrt(n)``
    that a normality assumption gives.

    The bracketed variance term can go negative for extreme moment
    combinations. When it does we fall back to the Gaussian term
    ``1 + SR^2/2``, which is the same expression with the higher moments set to
    their normal values, rather than returning a nan that propagates silently.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"Sharpe standard error needs at least 4 observations, got {n}")

    sr = sharpe_ratio(x, rf_per_period)
    if not math.isfinite(sr):
        return math.nan

    g3, g4 = standardised_moments(x)
    variance_term = 1.0 + 0.5 * sr**2 - g3 * sr + 0.25 * (g4 - 3.0) * sr**2
    if variance_term <= 0:
        variance_term = 1.0 + 0.5 * sr**2
    return math.sqrt(variance_term / n)


def sharpe_tstat(returns, rf_per_period: float = 0.0) -> float:
    """Sharpe divided by its standard error. Roughly standard normal under the null."""
    se = sharpe_standard_error(returns, rf_per_period)
    if not math.isfinite(se) or se <= 0:
        return math.nan
    return sharpe_ratio(returns, rf_per_period) / se


def columnwise_sharpe_tstat(trial_returns, rf_per_period: float = 0.0) -> np.ndarray:
    """:func:`sharpe_tstat` for every column, with ``nan`` for degenerate ones.

    A per-column loop rather than a vectorised rewrite of the standard error.
    The formula has a skewness term, a kurtosis term and a fallback for a
    negative variance term, and a second copy of it that drifts from this one
    would put the audited result and the family it is compared against on
    different footings - which is exactly the bug this function was written to
    remove.

    Degenerate columns are screened by :func:`columnwise_sharpe` first. A column
    of identical floats has a sample standard deviation around ``1e-16``, not
    zero, so it would otherwise arrive here with a Sharpe near ``1e14``, take
    rank 1 in any family it joins, and drag every other adjusted p-value with
    it.
    """
    matrix = as_trial_matrix(trial_returns)
    if matrix.shape[0] < 4:
        raise ValueError(
            f"Sharpe standard error needs at least 4 observations, got {matrix.shape[0]}"
        )
    sharpes = columnwise_sharpe(matrix, rf_per_period)
    out = np.full(matrix.shape[1], math.nan)
    for j, sr in enumerate(sharpes):
        if math.isfinite(sr):
            out[j] = sharpe_tstat(matrix[:, j], rf_per_period)
    return out


@dataclass(frozen=True)
class SharpeResult:
    """Everything the report needs about the Sharpe, in one object.

    ``annualised_naive`` is deliberately retained: it is what a tool that
    ignores serial dependence would print, and the gap between it and
    ``annualised_adjusted`` is one of the report's most useful numbers.
    """

    n_obs: int
    periods_per_year: int
    periodic: Estimate
    annualised_naive: Estimate
    annualised_adjusted: Estimate
    factor_naive: float
    factor_lo: float
    skewness: float
    kurtosis: float
    tstat: float
    autocorrelation_note: str | None = None

    @property
    def autocorrelation_haircut(self) -> float:
        """Fraction of the naive annualised Sharpe destroyed by the adjustment.

        ``0.0`` means naive annualisation was fine; ``0.38`` means the honest
        number is 38% lower. Negative values are possible and mean the returns
        are negatively autocorrelated, which *raises* the annualised Sharpe.
        """
        if self.factor_naive == 0:
            return math.nan
        return 1.0 - (self.factor_lo / self.factor_naive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "periods_per_year": self.periods_per_year,
            "periodic": self.periodic.to_dict(),
            "annualised_naive": self.annualised_naive.to_dict(),
            "annualised_adjusted": self.annualised_adjusted.to_dict(),
            "factor_naive": self.factor_naive,
            "factor_lo": self.factor_lo,
            "autocorrelation_haircut": self.autocorrelation_haircut,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "tstat": self.tstat,
            "autocorrelation_note": self.autocorrelation_note,
        }


def analyse_sharpe(
    returns,
    periods_per_year: int,
    rf_per_period: float = 0.0,
    confidence: float = 0.95,
) -> SharpeResult:
    """Full Sharpe workup: periodic, naive annual, and Lo-adjusted annual.

    Intervals here are the *analytic* ones. The report leads with the bootstrap
    interval from :mod:`qv.stats.bootstrap` instead; these are shown beside it
    to expose the difference. Below ``MIN_OBS_FOR_ASYMPTOTICS`` observations
    every estimate is flagged ``reliable=False`` with an explanatory note.
    """
    x = as_returns_array(returns)
    n = x.size
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be at least 1, got {periods_per_year}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    sr = sharpe_ratio(x, rf_per_period)
    se = sharpe_standard_error(x, rf_per_period)
    g3, g4 = standardised_moments(x)
    z = float(_st.norm.ppf(0.5 + confidence / 2.0))

    reliable = n >= MIN_OBS_FOR_ASYMPTOTICS
    note = (
        None
        if reliable
        else (
            f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} needed for the "
            "asymptotic standard error to mean anything; treat this interval as "
            "indicative only"
        )
    )

    factor_naive = naive_annualisation_factor(periods_per_year)
    ac_note: str | None = None
    try:
        factor_lo = lo_annualisation_factor(x, periods_per_year)
        # Only caveat the case where annualisation reaches past what the sample
        # can speak to at all. A short bandwidth chosen *because* dependence
        # died out early is the adjustment working, not a limitation.
        reach = max_estimable_lag(n)
        if periods_per_year - 1 > reach:
            ac_note = (
                f"annualising to {periods_per_year} periods implies a sum over "
                f"{periods_per_year - 1} lags, but {n} observations can only speak to "
                f"about {reach}; the adjustment assumes no dependence beyond that, and "
                "cannot rule out longer-horizon persistence"
            )
    except ValueError as exc:
        factor_lo = factor_naive
        ac_note = f"fell back to sqrt(q): {exc}"

    def _scaled(scale: float, method: str) -> Estimate:
        return Estimate(
            value=sr * scale,
            ci_low=(sr - z * se) * scale,
            ci_high=(sr + z * se) * scale,
            method=method,
            n_obs=n,
            reliable=reliable,
            note=note,
        )

    return SharpeResult(
        n_obs=n,
        periods_per_year=periods_per_year,
        periodic=_scaled(1.0, "Lo/Mertens analytic SE"),
        annualised_naive=_scaled(factor_naive, f"naive sqrt({periods_per_year}) annualisation"),
        annualised_adjusted=_scaled(factor_lo, "Lo (2002) autocorrelation-adjusted annualisation"),
        factor_naive=factor_naive,
        factor_lo=factor_lo,
        skewness=g3,
        kurtosis=g4,
        tstat=sharpe_tstat(x, rf_per_period),
        autocorrelation_note=ac_note,
    )
