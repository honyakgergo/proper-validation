"""Is the alpha still there once factors are accounted for?

The first question any portfolio manager asks, and the one that most often
ends the conversation. A strategy with an attractive Sharpe that turns out to
be 0.9 beta to the market plus a momentum tilt has not found anything - it has
rediscovered two factors you can buy for a few basis points.

Standard errors are Newey-West by default. Regression residuals from a return
series are serially correlated and heteroskedastic, and plain OLS standard
errors understate the uncertainty on alpha accordingly - which is exactly the
coefficient the whole exercise turns on.

This module does no I/O. Hand it a factor frame; see :mod:`qv.data.loaders`
for fetching one from the Ken French library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import statsmodels.api as sm

from qv.stats.moments import max_estimable_lag
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Estimate, Severity

__all__ = ["FactorLoading", "AttributionResult", "factor_attribution"]


@dataclass(frozen=True)
class FactorLoading:
    """One regression coefficient with its inference."""

    name: str
    beta: float
    std_error: float
    tstat: float
    p_value: float

    @property
    def significant_at_5pct(self) -> bool:
        return self.p_value < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "beta": self.beta,
            "std_error": self.std_error,
            "tstat": self.tstat,
            "p_value": self.p_value,
            "significant_at_5pct": self.significant_at_5pct,
        }


@dataclass(frozen=True)
class AttributionResult:
    """Alpha and factor loadings from a time-series regression."""

    alpha_per_period: Estimate
    alpha_tstat: float
    alpha_p_value: float
    loadings: tuple[FactorLoading, ...]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    periods_per_year: int
    cov_type: str
    hac_lags: int | None
    note: str | None = None

    @property
    def alpha_annualised(self) -> float:
        return self.alpha_per_period.value * self.periods_per_year

    @property
    def alpha_survives(self) -> bool:
        """Is alpha significantly positive at the 5% level?"""
        return self.alpha_p_value < 0.05 and self.alpha_per_period.value > 0

    @property
    def explained_by_factors(self) -> bool:
        """Do the factors account for the strategy without residual alpha?"""
        return not self.alpha_survives and self.r_squared > 0.5

    def loading(self, name: str) -> FactorLoading:
        for load in self.loadings:
            if load.name == name:
                return load
        raise KeyError(f"{name!r} is not among {[load.name for load in self.loadings]}")

    @property
    def severity(self) -> Severity:
        """How damaging the attribution result is."""
        if self.alpha_survives:
            return Severity.INFO
        if self.r_squared > 0.75:
            return Severity.CRITICAL
        if self.r_squared > 0.4:
            return Severity.HIGH
        return Severity.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_per_period": self.alpha_per_period.to_dict(),
            "alpha_annualised": self.alpha_annualised,
            "alpha_tstat": self.alpha_tstat,
            "alpha_p_value": self.alpha_p_value,
            "alpha_survives": self.alpha_survives,
            "explained_by_factors": self.explained_by_factors,
            "loadings": [load.to_dict() for load in self.loadings],
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "n_obs": self.n_obs,
            "periods_per_year": self.periods_per_year,
            "cov_type": self.cov_type,
            "hac_lags": self.hac_lags,
            "severity": self.severity.name.lower(),
            "note": self.note,
        }


def factor_attribution(
    returns,
    factors,
    factor_names: list[str] | None = None,
    rf: float | np.ndarray | None = None,
    periods_per_year: int = 252,
    use_hac: bool = True,
    hac_lags: int | None = None,
    confidence: float = 0.95,
) -> AttributionResult:
    """Regress strategy returns on factor returns and report alpha.

    ``factors`` is a ``(T, k)`` array or DataFrame of factor returns; column
    names are taken from a DataFrame when ``factor_names`` is not given.
    ``rf`` subtracts a risk-free rate to put the strategy in excess terms -
    required for the intercept to be interpretable as alpha when the factors
    are themselves excess returns, as the Ken French ones are.

    ``hac_lags`` defaults to the same data-driven bandwidth used elsewhere in
    the package, so the autocorrelation correction here and in the Sharpe
    annualisation make the same judgement about how far dependence reaches.
    """
    # Deliberately not as_returns_array: that drops NaNs, which would silently
    # shorten the return series and misalign it against the factor rows. Here
    # the two series must stay on a common index until they are masked together.
    y = np.asarray(getattr(returns, "to_numpy", lambda: returns)(), dtype=float)
    if y.ndim != 1:
        raise ValueError(f"returns must be 1-D, got shape {y.shape}")
    x = np.asarray(getattr(factors, "to_numpy", lambda: factors)(), dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"factors must be 1-D or 2-D, got shape {x.shape}")
    if x.shape[0] != y.size:
        raise ValueError(
            f"factors have {x.shape[0]} rows but returns have {y.size}; "
            "they must describe the same time axis"
        )
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be at least 1, got {periods_per_year}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")

    if factor_names is None:
        columns = getattr(factors, "columns", None)
        factor_names = (
            [str(c) for c in columns] if columns is not None
            else [f"factor_{i}" for i in range(x.shape[1])]
        )
    if len(factor_names) != x.shape[1]:
        raise ValueError(
            f"got {len(factor_names)} factor names for {x.shape[1]} factor columns"
        )

    if rf is not None:
        rf_arr = np.asarray(rf, dtype=float)
        if rf_arr.ndim == 0:
            y = y - float(rf_arr)
        elif rf_arr.size == y.size:
            y = y - rf_arr
        else:
            raise ValueError(f"rf has {rf_arr.size} values but returns have {y.size}")

    usable = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    n_dropped = int((~usable).sum())
    y, x = y[usable], x[usable]
    n = y.size

    if n <= x.shape[1] + 1:
        raise ValueError(
            f"{n} usable observations cannot support a regression on {x.shape[1]} factors"
        )

    design = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, design)

    lags = None
    if use_hac:
        lags = hac_lags if hac_lags is not None else min(max_estimable_lag(n), n - x.shape[1] - 2)
        lags = max(int(lags), 1)
        fitted = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
        cov_type = f"Newey-West HAC, {lags} lags"
    else:
        fitted = model.fit()
        cov_type = "OLS (homoskedastic)"

    z = float(sm.stats.stattools.stats.norm.ppf(0.5 + confidence / 2.0))
    alpha, alpha_se = float(fitted.params[0]), float(fitted.bse[0])

    notes = []
    if n < MIN_OBS_FOR_ASYMPTOTICS:
        notes.append(
            f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} at which these "
            "standard errors become trustworthy"
        )
    if n_dropped:
        notes.append(f"{n_dropped} observations dropped for missing factor or return data")

    return AttributionResult(
        alpha_per_period=Estimate(
            value=alpha,
            ci_low=alpha - z * alpha_se,
            ci_high=alpha + z * alpha_se,
            method=cov_type,
            n_obs=n,
            reliable=n >= MIN_OBS_FOR_ASYMPTOTICS,
            note=notes[0] if notes else None,
        ),
        alpha_tstat=float(fitted.tvalues[0]),
        alpha_p_value=float(fitted.pvalues[0]),
        loadings=tuple(
            FactorLoading(
                name=name,
                beta=float(fitted.params[i + 1]),
                std_error=float(fitted.bse[i + 1]),
                tstat=float(fitted.tvalues[i + 1]),
                p_value=float(fitted.pvalues[i + 1]),
            )
            for i, name in enumerate(factor_names)
        ),
        r_squared=float(fitted.rsquared),
        adj_r_squared=float(fitted.rsquared_adj),
        n_obs=n,
        periods_per_year=periods_per_year,
        cov_type=cov_type,
        hac_lags=lags,
        note="; ".join(notes) if notes else None,
    )
