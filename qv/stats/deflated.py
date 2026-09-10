"""Multiple-testing corrections: PSR, Deflated Sharpe, MinBTL, BHY haircut.

The single most common defect in a published backtest is reporting the winner
of a search as though it were the only thing tried. Everything in this module
exists to price that in.

All Sharpe ratios here are **per-period**, not annualised. Mixing the two is
the classic way to get a deflated Sharpe that is wrong by a factor of
``sqrt(252)`` while still looking plausible, so the functions refuse to guess
which you meant.
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
    standardised_moments,
)
from qv.stats.sharpe import sharpe_ratio
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Severity

__all__ = [
    "EULER_MASCHERONI",
    "expected_max_sharpe",
    "sharpe_variance_term",
    "probabilistic_sharpe_ratio",
    "trial_sharpes",
    "DeflatedSharpeResult",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
    "HaircutResult",
    "bhy_haircut",
]

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent zero-edge trials.

    Bailey and Lopez de Prado (2014), the ``SR_0`` of the Deflated Sharpe:

    ``SR_0 = std(SR_trials) * [ (1 - g) * Phi_inv(1 - 1/N)
                                + g * Phi_inv(1 - 1/(N*e)) ]``

    with ``g`` the Euler-Mascheroni constant. This is the extreme-value
    approximation to the expected maximum of ``N`` standard normal draws,
    scaled by the observed dispersion of trial Sharpes.

    It grows slowly - roughly with ``sqrt(2*log(N))`` - which is why even a
    modest search produces an impressive-looking winner. At 1000 trials with
    unit trial dispersion the expected best Sharpe is already above 3.2 with no
    edge whatsoever.

    ``n_trials=1`` means no selection took place and returns 0.0 exactly.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if trial_sharpe_std < 0:
        raise ValueError(f"trial_sharpe_std must be non-negative, got {trial_sharpe_std}")
    if n_trials == 1:
        return 0.0

    n = float(n_trials)
    term = (1.0 - EULER_MASCHERONI) * _st.norm.ppf(1.0 - 1.0 / n) + EULER_MASCHERONI * _st.norm.ppf(
        1.0 - 1.0 / (n * math.e)
    )
    return float(trial_sharpe_std * term)


def sharpe_variance_term(sharpe: float, skew: float, kurt: float) -> float:
    """The ``1 - g3*SR + ((g4 - 1)/4)*SR^2`` bracket shared by PSR and DSR.

    Identical to the bracket inside the Lo/Mertens standard error - expanding
    ``1 + SR^2/2 + ((g4 - 3)/4)*SR^2`` gives ``((g4 - 1)/4)*SR^2`` - so the two
    parts of this package cannot silently disagree about how much non-normality
    matters. ``kurt`` is raw kurtosis (3 for a normal).

    Clamped at a small positive floor: extreme moment combinations can drive
    the bracket non-positive, and a negative variance has no square root.
    """
    term = 1.0 - skew * sharpe + 0.25 * (kurt - 1.0) * sharpe**2
    return max(term, 1e-12)


def probabilistic_sharpe_ratio(
    returns, benchmark_sharpe: float = 0.0, rf_per_period: float = 0.0
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    Bailey and Lopez de Prado (2012). Accounts for sample length, skewness and
    fat tails. The Deflated Sharpe is exactly this with the benchmark set to
    the expected maximum under the null rather than to zero.

    ``benchmark_sharpe`` is per-period, like everything else here.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"PSR needs at least 4 observations, got {n}")

    sr = sharpe_ratio(x, rf_per_period)
    if not math.isfinite(sr):
        return math.nan

    skew, kurt = standardised_moments(x)
    denom = math.sqrt(sharpe_variance_term(sr, skew, kurt))
    return float(_st.norm.cdf((sr - benchmark_sharpe) * math.sqrt(n - 1) / denom))


def trial_sharpes(trial_returns, rf_per_period: float = 0.0) -> np.ndarray:
    """Per-period Sharpe of every column of a ``(T, N)`` trial matrix.

    Columns with no variance yield ``nan`` and are dropped, since a flat trial
    carries no information about how much dispersion the search generated.
    """
    matrix = as_trial_matrix(trial_returns)
    if matrix.shape[0] < 2:
        raise ValueError(f"trial matrix needs at least 2 rows, got {matrix.shape[0]}")
    sharpes = columnwise_sharpe(matrix, rf_per_period)
    return sharpes[np.isfinite(sharpes)]


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Deflated Sharpe plus every input that produced it.

    ``dsr`` is a probability, not a p-value: it is the probability that the
    true Sharpe exceeds what selecting the best of ``n_trials`` would produce
    from noise alone. Low is bad. ``p_value = 1 - dsr``.
    """

    observed_sharpe: float
    expected_max_sharpe: float
    dsr: float
    n_trials: int
    trial_sharpe_std: float
    trial_sharpe_std_source: str
    n_obs: int
    skewness: float
    kurtosis: float
    reliable: bool
    note: str | None = None

    @property
    def p_value(self) -> float:
        return 1.0 - self.dsr

    @property
    def survives(self) -> bool:
        """Does the observed Sharpe clear the selection-adjusted bar at 95%?"""
        return self.dsr >= 0.95

    @property
    def severity(self) -> Severity:
        """How bad the deflation result is, for the findings table."""
        if self.dsr >= 0.95:
            return Severity.INFO
        if self.dsr >= 0.90:
            return Severity.MEDIUM
        if self.dsr >= 0.50:
            return Severity.HIGH
        return Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_sharpe": self.observed_sharpe,
            "expected_max_sharpe": self.expected_max_sharpe,
            "dsr": self.dsr,
            "p_value": self.p_value,
            "survives": self.survives,
            "n_trials": self.n_trials,
            "trial_sharpe_std": self.trial_sharpe_std,
            "trial_sharpe_std_source": self.trial_sharpe_std_source,
            "n_obs": self.n_obs,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "reliable": self.reliable,
            "note": self.note,
        }


def deflated_sharpe_ratio(
    returns,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    trial_returns=None,
    rf_per_period: float = 0.0,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio (Bailey and Lopez de Prado 2014).

    Supply the dispersion of trial Sharpes one of three ways, in order of
    preference:

    1. ``trial_returns`` - a ``(T, N)`` matrix; the dispersion is measured.
    2. ``trial_sharpe_std`` - if you know it from elsewhere.
    3. Neither, in which case a conservative default of ``1/sqrt(n)`` is used,
       the dispersion expected if every trial were pure noise on this sample
       length. This *under*-deflates whenever the real search produced more
       spread than noise alone, so the result is a lower bound on the damage
       and the report must say so.

    ``n_trials`` should be the number of configurations actually examined,
    including the ones abandoned before they were written down. It is the most
    under-reported number in backtesting; see :mod:`qv.leakage.notebook` for
    recovering a defensible lower bound from a notebook.
    """
    x = as_returns_array(returns)
    n = x.size
    if n < 4:
        raise ValueError(f"DSR needs at least 4 observations, got {n}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")

    notes: list[str] = []
    if trial_returns is not None:
        sharpes = trial_sharpes(trial_returns, rf_per_period)
        if sharpes.size < 2:
            raise ValueError("trial matrix must contain at least 2 usable columns")
        std = float(np.std(sharpes, ddof=1))
        source = f"measured from {sharpes.size} trial columns"
    elif trial_sharpe_std is not None:
        std = float(trial_sharpe_std)
        source = "supplied by caller"
    else:
        std = 1.0 / math.sqrt(n)
        source = "assumed 1/sqrt(n) - no trial matrix supplied"
        notes.append(
            "trial-Sharpe dispersion was assumed rather than measured, so this "
            "deflation is a lower bound on the true selection penalty"
        )

    if std < 0:
        raise ValueError(f"trial_sharpe_std must be non-negative, got {std}")

    sr = sharpe_ratio(x, rf_per_period)
    sr0 = expected_max_sharpe(n_trials, std)
    skew, kurt = standardised_moments(x)

    if math.isfinite(sr):
        denom = math.sqrt(sharpe_variance_term(sr, skew, kurt))
        dsr = float(_st.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))
    else:
        dsr = math.nan

    reliable = n >= MIN_OBS_FOR_ASYMPTOTICS
    if not reliable:
        notes.append(
            f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} at which the normal "
            "approximation behind DSR holds"
        )
    if n_trials == 1:
        notes.append(
            "n_trials=1 assumes the strategy was the only one ever tried, which is "
            "almost never true; DSR reduces to the probabilistic Sharpe ratio"
        )

    return DeflatedSharpeResult(
        observed_sharpe=sr,
        expected_max_sharpe=sr0,
        dsr=dsr,
        n_trials=n_trials,
        trial_sharpe_std=std,
        trial_sharpe_std_source=source,
        n_obs=n,
        skewness=skew,
        kurtosis=kurt,
        reliable=reliable,
        note="; ".join(notes) if notes else None,
    )


def minimum_backtest_length(n_trials: int, target_annual_sharpe: float = 1.0) -> float:
    """Years of data needed before an annualised Sharpe means anything.

    Bailey, Borwein, Lopez de Prado and Zhu (2014). Rearranges the expected
    maximum: with ``N`` trials on ``y`` years of data the best zero-edge
    strategy is expected to show an annualised Sharpe of roughly
    ``E[max] / sqrt(y)``, so demanding that fall below the target gives

    ``y = ( E[max_N] / SR_target )^2``

    It is a blunt, memorable number. Forty trials against a target Sharpe of 1
    already demands roughly six years; a thousand demands over ten.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if target_annual_sharpe <= 0:
        raise ValueError(f"target_annual_sharpe must be positive, got {target_annual_sharpe}")
    return (expected_max_sharpe(n_trials, 1.0) / target_annual_sharpe) ** 2


@dataclass(frozen=True)
class HaircutResult:
    """What survives a false-discovery-rate correction."""

    observed_tstat: float
    adjusted_tstat: float
    observed_pvalue: float
    adjusted_pvalue: float
    n_trials: int
    haircut: float
    observed_sharpe: float | None = None
    haircut_sharpe: float | None = None
    #: The t-statistic this result would have needed to stay significant at 5%
    #: after the adjustment. Without it the reader learns that the result
    #: failed but not by how far, and at a large trial count the two are very
    #: different stories.
    required_tstat: float | None = None
    required_sharpe: float | None = None
    #: Number of tests the adjustment was computed over, and the audited
    #: result's rank among them by p-value. Rank 1 is the most significant.
    n_tests: int | None = None
    rank: int | None = None
    #: True when the whole family of trial t-statistics was used, false when
    #: only the audited one was known and the conservative bound was applied.
    used_trial_family: bool = False

    @property
    def significant_at_5pct(self) -> bool:
        return self.adjusted_pvalue < 0.05

    @property
    def censored(self) -> bool:
        """True when the adjusted p-value hit 1 and the t-statistic floored.

        Everything below a threshold maps to exactly zero, so the adjusted
        figure stops being a measurement and becomes a floor. Anything drawing
        it has to say so, or it reads as "this strategy has no edge at all"
        rather than "this correction cannot see one".
        """
        return self.adjusted_pvalue >= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_tstat": self.observed_tstat,
            "adjusted_tstat": self.adjusted_tstat,
            "observed_pvalue": self.observed_pvalue,
            "adjusted_pvalue": self.adjusted_pvalue,
            "n_trials": self.n_trials,
            "haircut": self.haircut,
            "observed_sharpe": self.observed_sharpe,
            "haircut_sharpe": self.haircut_sharpe,
            "required_tstat": self.required_tstat,
            "required_sharpe": self.required_sharpe,
            "n_tests": self.n_tests,
            "rank": self.rank,
            "used_trial_family": self.used_trial_family,
            "significant_at_5pct": self.significant_at_5pct,
            "censored": self.censored,
            "method": (
                "Benjamini-Hochberg-Yekutieli (Harvey and Liu 2015), over the observed "
                "trial t-statistics"
                if self.used_trial_family
                else "Benjamini-Hochberg-Yekutieli (Harvey and Liu 2015), top-ranked bound"
            ),
        }


def _by_adjusted_pvalues(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Benjamini-Yekutieli adjusted p-values, ascending, with the sort order.

    Step-up over the ordered p-values: the adjusted value at rank ``i`` is
    ``min over j >= i`` of ``N c(N) p_(j) / j``, capped at 1. Taking the running
    minimum from the largest rank downwards is what makes it monotone, and it is
    the step that separates a false-discovery-rate procedure from a
    family-wise one - a test surrounded by other significant tests is judged
    less harshly than the same test standing alone.
    """
    ps = np.sort(pvalues)
    n = ps.size
    c_n = float(np.sum(1.0 / np.arange(1, n + 1)))
    terms = np.minimum(ps * n * c_n / np.arange(1, n + 1), 1.0)
    adjusted = np.minimum.accumulate(terms[::-1])[::-1]
    return ps, adjusted, c_n


def bhy_haircut(
    tstat: float,
    n_trials: int,
    observed_sharpe: float | None = None,
    trial_tstats=None,
) -> HaircutResult:
    """Benjamini-Hochberg-Yekutieli haircut on a t-statistic (Harvey and Liu 2015).

    Asks what a t-statistic is worth once the other tests that were run are
    priced in. BHY rather than Bonferroni or Holm because it is the variant
    valid under *arbitrary* dependence between tests, which is the realistic
    case when every trial runs on the same overlapping price history. Bonferroni
    and Holm are deliberately not reported: for a top-ranked test they coincide
    with each other, and printing three numbers for one question is padding.

    **Supply ``trial_tstats`` whenever the trial matrix exists.** BHY is defined
    over a family of tests, and with the family in hand the audited result is
    judged at its own rank. Without it the only defensible assumption is that
    the result is the most significant of ``n_trials``, which multiplies the
    p-value by ``N c(N)`` - Bonferroni times the harmonic number, the most
    conservative correction of the three. On a 54-trial grid that factor is
    ~248, which floors the adjusted t-statistic at zero for anything below
    t = 2.9 and reports a strategy that missed narrowly identically to one that
    never had a chance. Discarding 53 of the 54 tests you actually ran to buy
    that pessimism is not conservatism, it is throwing away evidence.

    The family is only used when it is at least as large as the declared trial
    count. A researcher who declares 200 trials and hands over 54 columns has
    146 tests nobody can see, and the bound is the honest answer there.

    The haircut is the proportional loss in t-statistic, which maps directly
    onto a proportional loss in Sharpe.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if not math.isfinite(tstat):
        raise ValueError(f"tstat must be finite, got {tstat}")

    p_obs = 2.0 * float(_st.norm.sf(abs(tstat)))

    family = None
    if trial_tstats is not None:
        observed = np.asarray(trial_tstats, dtype=float)
        observed = observed[np.isfinite(observed)]
        if observed.size >= max(2, n_trials):
            family = 2.0 * _st.norm.sf(np.abs(observed))
            # The audited configuration is normally one of the trials. When it
            # is not - a strategy reported after the grid was run - it still
            # belongs in the family it is being judged against.
            if not np.any(np.isclose(family, p_obs, rtol=1e-9, atol=1e-18)):
                family = np.append(family, p_obs)

    if family is None:
        n_tests = n_trials
        c_n = float(np.sum(1.0 / np.arange(1, n_tests + 1)))
        rank = 1
        p_adj = min(p_obs * n_tests * c_n, 1.0)
        used_family = False
    else:
        n_tests = int(family.size)
        ordered, adjusted, c_n = _by_adjusted_pvalues(family)
        rank = int(np.searchsorted(ordered, p_obs, side="left")) + 1
        rank = min(max(rank, 1), n_tests)
        p_adj = float(adjusted[rank - 1])
        used_family = True

    # Map the adjusted p-value back to the t-statistic that would have produced
    # it in a single test. A fully-adjusted-away result gives t = 0.
    t_adj = 0.0 if p_adj >= 1.0 else float(_st.norm.isf(p_adj / 2.0))
    t_adj = math.copysign(t_adj, tstat)

    haircut = 1.0 - (abs(t_adj) / abs(tstat)) if tstat != 0 else 0.0

    # The bar, in the units the reader already has: invert this test's own term
    # in the step-up at p = 0.05.
    required_tstat = float(_st.norm.isf(0.05 * rank / (2.0 * n_tests * c_n)))
    required_sharpe = (
        None
        if observed_sharpe is None or tstat == 0
        else abs(observed_sharpe) * required_tstat / abs(tstat)
    )

    return HaircutResult(
        observed_tstat=tstat,
        adjusted_tstat=t_adj,
        observed_pvalue=p_obs,
        adjusted_pvalue=p_adj,
        n_trials=n_trials,
        haircut=haircut,
        observed_sharpe=observed_sharpe,
        haircut_sharpe=None if observed_sharpe is None else observed_sharpe * (1.0 - haircut),
        required_tstat=required_tstat,
        required_sharpe=required_sharpe,
        n_tests=n_tests,
        rank=rank,
        used_trial_family=used_family,
    )
