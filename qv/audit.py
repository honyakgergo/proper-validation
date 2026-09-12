"""The orchestrator: run every test the supplied data can support.

Honesty about coverage is implemented here. Each section runs only if its
inputs exist and its suite is in scope, records why it was skipped when either
is missing, and the report leads with what could not be tested rather than
quietly omitting it. A narrow audit is not a worse audit, but the difference
has to be visible on the page.

Two suites, asking independent questions - see :class:`qv.types.Suite`. The
statistical one asks whether the measured edge is distinguishable from luck;
the engine one asks whether the backtest can be trusted as an implementation,
whatever its numbers say.

The verdict is deliberately asymmetric. There is no ``PASSED``: the best
available outcome is that the audit failed to falsify the backtest, which is
not the same as evidence that it works.
"""

from __future__ import annotations

import math
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

import numpy as np

from qv.attribution.benchmark import vol_matched_benchmark
from qv.attribution.factors import factor_attribution
from qv.costs.breakeven import break_even_cost, sharpe_vs_cost_curve
from qv.costs.models import ASSET_CLASS_LABELS, average_turnover
from qv.findings import make_finding
from qv.engine import (
    determinism_check,
    execution_delay_curve,
    signal_degeneracy,
    universe_coverage,
)
from qv.leakage.perturbation import perturbation_test
from qv.provenance import source_identity
from qv.robustness.parameters import plateau_ratio
from qv.robustness.randomization import matched_exposure_test, sign_flip_test
from qv.robustness.regimes import trailing_volatility, volatility_regime_split
from qv.robustness.subsample import rolling_origin_sensitivity, top_day_dependence
from qv.stats.bootstrap import stationary_bootstrap
from qv.stats.deflated import (
    bhy_haircut,
    deflated_sharpe_ratio,
    minimum_backtest_length,
)
from qv.stats.moments import as_returns_array
from qv.stats.null_max import empirical_max_sharpe_null, gaussian_max_sharpe_null
from qv.stats.pbo import combinatorially_symmetric_cv
from qv.stats.performance import summarise_performance
from qv.stats.risk import simulate_risk
from qv.stats.sharpe import analyse_sharpe, columnwise_sharpe_tstat, sharpe_ratio
from qv.types import MIN_OBS_FOR_ASYMPTOTICS, Finding, Severity, Suite, Tier, Verdict

__all__ = ["AuditInputs", "AuditReport", "run_audit"]


@dataclass
class AuditInputs:
    """Everything the audit can use. Almost all of it is optional by design."""

    returns: Any
    periods_per_year: int = 252
    name: str = "strategy"

    # Positions and prices: turnover, costs, the matched-exposure test
    positions: Any | None = None
    asset_returns: Any | None = None
    asset_class: str | None = None

    # Context a return series alone cannot supply
    benchmark_returns: Any | None = None
    benchmark_name: str = "benchmark"
    factors: Any | None = None
    factor_names: list[str] | None = None
    risk_free: Any | None = None
    #: Timestamps for the return series. Purely presentational - they put real
    #: years on the equity curve instead of an observation counter - so a
    #: mismatched length is ignored rather than raised.
    dates: Any | None = None

    # Data provenance. Survivorship cannot be detected from a return series,
    # only declared, so the audit asks and records the answer instead of
    # quietly assuming the universe was clean.
    universe_point_in_time: bool | None = None
    universe_note: str | None = None
    #: A parsed `qv.data.membership.Membership`, never a path - the engine
    #: stays I/O-free. Present, survivorship is measured rather than declared.
    membership: Any | None = None
    #: The window the report describes, for the membership comparison. Measuring
    #: against any other window answers a question the reader is never shown.
    sample_window: tuple[Any, Any] | None = None
    #: The traded universe's names, in declared order.
    universe_names: Any | None = None

    # Selection bias
    n_trials: int | None = None
    trial_returns: Any | None = None
    parameter_scores: Any | None = None
    #: Index of the configuration actually chosen. Defaults to the grid maximum,
    #: which is what a search would have selected - but a researcher who picked
    #: on theoretical grounds deserves to be judged on the point they picked.
    chosen_parameter_index: tuple[int, ...] | int | None = None

    # A re-runnable strategy. Unlocks the behavioural leakage test and the
    # determinism check, which is why the agent layer pushes for it.
    strategy: Callable[[np.ndarray], np.ndarray] | None = None
    strategy_data: Any | None = None

    # Engine analysis
    #: Which of the two questions to ask. They are independent: a strategy can
    #: be statistically hopeless and impeccably implemented, or the reverse -
    #: and the reverse is the dangerous case, because the numbers look fine.
    suite: Suite = Suite.FULL
    #: Per-asset returns as a ``(T, K)`` frame aligned with ``positions``, for
    #: the execution-delay curve. Distinct from ``asset_returns``, which is the
    #: single traded instrument the matched-exposure test needs.
    asset_return_frame: Any | None = None
    #: The price frame *before* any common-calendar alignment. Coverage has to
    #: be measured on the ragged frame: dropping the incomplete rows first is
    #: exactly what makes a ragged universe look clean.
    raw_prices: Any | None = None

    # Reproducibility
    seed: int = 0
    n_boot: int = 2000
    confidence: float = 0.95

    @property
    def tier(self) -> Tier:
        if self.strategy is not None and self.strategy_data is not None:
            return Tier.CALLABLE
        if self.positions is not None:
            return Tier.POSITIONS
        return Tier.RETURNS


@dataclass
class AuditReport:
    """The result of an audit: sections, findings, and what could not be tested."""

    name: str
    tier: Tier
    n_obs: int
    periods_per_year: int
    #: Which questions this report asked. A reader has to know that before any
    #: number on the page means anything: an engine-only report is silent about
    #: performance by design, not because the strategy had none.
    suite: Suite = Suite.FULL
    sections: dict[str, Any] = field(default_factory=dict)
    #: Raw arrays needed only to draw charts - null distributions, CSCV logits,
    #: the cost curve. Deliberately outside ``sections`` so they never reach
    #: report.json, which would otherwise be megabytes of numbers nobody reads.
    #: Every *summary* of these arrays does appear in the JSON, via chart specs.
    chart_data: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    not_tested: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def worst_severity(self) -> Severity:
        return max((f.severity for f in self.findings), default=Severity.INFO)

    @property
    def verdict(self) -> Verdict:
        """Asymmetric by construction: the tool falsifies, it cannot validate."""
        if self.n_obs < MIN_OBS_FOR_ASYMPTOTICS:
            return Verdict.INSUFFICIENT_DATA
        worst = self.worst_severity
        if worst >= Severity.CRITICAL:
            return Verdict.FALSIFIED
        if worst >= Severity.HIGH:
            return Verdict.WEAKENED
        return Verdict.NOT_FALSIFIED

    @property
    def headline_sharpe_stages(self) -> list[tuple[str, float, float | None, float | None]]:
        """Ordered stages for the haircut cascade chart."""
        return list(self.sections.get("_stages", []))

    def findings_by_severity(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (-int(f.severity), f.id))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "tier": int(self.tier),
            "tier_label": self.tier.label,
            "suite": self.suite.value,
            "suite_label": self.suite.label,
            "suite_question": self.suite.question,
            "verdict": self.verdict.value,
            "verdict_label": self.verdict.label,
            "worst_severity": self.worst_severity.name.lower(),
            "n_obs": self.n_obs,
            "periods_per_year": self.periods_per_year,
            "not_tested": list(self.not_tested),
            "findings": [f.to_dict() for f in self.findings_by_severity()],
            "provenance": self.provenance,
        }
        payload.update({k: v for k, v in self.sections.items() if not k.startswith("_")})
        return payload



def _run_survivorship(inputs: AuditInputs, report: AuditReport, findings: list) -> None:
    """Was the universe assembled with hindsight?

    Three bases, and they must stay distinguishable in ``report.json``. If this
    function only ever appended a finding, then a refused membership file, a file
    describing the wrong period, an exception, and a genuinely clean universe
    would all produce the same visible output: nothing at all. So
    ``survivorship_basis`` is always written, and "clean" and "could not tell"
    are never the same string.
    """
    section: dict[str, Any] = {
        "universe_point_in_time": inputs.universe_point_in_time,
        "universe_note": inputs.universe_note,
        "survivorship_basis": "not_tested",
        "survivorship_basis_reason": "not declared and not measured",
    }
    report.sections["data"] = section

    measurement = None
    if inputs.membership is not None and inputs.sample_window is not None:
        from qv.data.membership import measure_survivorship

        section["membership"] = inputs.membership.to_dict()
        try:
            measurement = measure_survivorship(
                inputs.membership,
                list(inputs.universe_names or ()),
                inputs.sample_window[0],
                inputs.sample_window[1],
            )
        except ValueError as exc:
            section["survivorship_basis_reason"] = str(exc)
            report.not_tested.append(f"Survivorship: {exc}")

    if measurement is not None:
        section["measurement"] = measurement.to_dict()
        findings.extend(measurement.to_findings())

        if measurement.looks_like_a_snapshot:
            # The file cannot support a clean verdict, so do not let one be
            # inferred from the absence of a finding.
            section["survivorship_basis"] = "not_tested"
            section["survivorship_basis_reason"] = (
                "the membership list records no departures, so it cannot distinguish a "
                "complete universe from a list of survivors"
            )
            report.not_tested.append(
                "Survivorship: the membership list supplied records no removals, so it "
                "cannot tell a complete universe from a list of today's members. See "
                "DATA-MEMBERSHIP-NOT-POINT-IN-TIME."
            )
        else:
            section["survivorship_basis"] = "measured"
            section["survivorship_basis_reason"] = (
                f"measured against {inputs.membership.source}"
            )
            if measurement.covered_fraction_of_sample < 0.999:
                report.not_tested.append(
                    "Survivorship before "
                    f"{measurement.covered_start.date()}: the membership list starts "
                    "after the sample does, so it speaks to "
                    f"{measurement.covered_fraction_of_sample:.0%} of the audited "
                    "window and says nothing about the rest. Every membership count in "
                    "this report describes the covered period only."
                )
            # A ticker with more than one spell is either a name that rejoined
            # or a symbol another company later took over. Without a permanent
            # identifier nothing in the file separates them, and a wrong join
            # would attach the wrong company's prices - a plausible number
            # rather than an error. Say so rather than let the count stand
            # unqualified.
            reused = measurement.noncontiguous_tickers
            if reused and not measurement.has_identifier:
                report.not_tested.append(
                    f"Ticker identity for {len(reused)} name(s) "
                    f"({', '.join(reused[:6])}): they hold more than one membership "
                    "spell, and with no permanent identifier in the list a re-listing "
                    "cannot be told from a symbol reused by a different company. Add "
                    "an `id` column to resolve them."
                )
            # Extent, not magnitude. True on every run that gets this far, and the
            # one sentence that stops the count being read as a haircut.
            report.not_tested.append(
                "Size of the survivorship bias: the membership list says which names "
                "were excluded, not what their returns would have been, so the count "
                "bounds how much is missing and not how much the result is overstated. "
                "That needs prices for delisted instruments."
            )

        # A declaration the data disproves is worse than an open question: the
        # reader cannot tell which of the *other* declarations still hold, and the
        # deflated Sharpe rests on one of them. Only this direction fires.
        # Declaring `false` and measuring clean is over-caution rather than a
        # defect, and a confession about how a list was built outranks a null
        # result from a file that may cover only part of the window.
        if inputs.universe_point_in_time is True and measurement.missing_exited:
            findings.append(
                make_finding(
                    "DATA-DECLARATION-CONTRADICTED",
                    detail=(
                        "The universe was declared point-in-time, but "
                        f"{len(measurement.missing_exited)} name(s) that left the index "
                        "during the window are absent from it. The declaration and the "
                        "membership list disagree, and the list is the one with evidence "
                        "behind it."
                    ),
                    evidence={
                        "declared": True,
                        "missing_exited": list(measurement.missing_exited[:20]),
                    },
                )
            )
        return

    # No usable measurement: fall back to the declaration.
    if inputs.universe_point_in_time is False:
        section["survivorship_basis"] = "declared"
        section["survivorship_basis_reason"] = "declared by the manifest, not measured"
        findings.append(
            make_finding(
                "DATA-SURVIVORSHIP",
                detail=(
                    "The universe was declared as built from present-day membership, so "
                    "every instrument that failed, delisted or merged is missing from it. "
                    "The result is an upper bound rather than an estimate, and no test "
                    "below corrects for it. Supply a point-in-time membership list as "
                    "`data.membership_frame` to replace this declaration with a count."
                    + (f" Declared: {inputs.universe_note}" if inputs.universe_note else "")
                ),
                evidence={
                    "universe_point_in_time": False,
                    "universe_note": inputs.universe_note,
                },
            )
        )
    elif inputs.universe_point_in_time is None:
        report.not_tested.append(
            "Survivorship: the universe was not declared, so whether it was assembled "
            "from present-day membership could not be checked. Declare "
            "`universe_point_in_time`, or supply a point-in-time membership list as "
            "`data.membership_frame` and it will be counted rather than declared."
        )
    else:
        section["survivorship_basis"] = "declared"
        section["survivorship_basis_reason"] = (
            "declared point-in-time by the manifest, not measured"
        )


def _run_engine_suite(
    inputs: AuditInputs,
    report: AuditReport,
    findings: list[Finding],
    excess: np.ndarray,
    rf_per_period: float,
) -> None:
    """Everything the engine suite asks, in one place.

    Kept separate from the statistical orchestration above rather than
    interleaved with it, because the two suites answer independent questions
    and a reader should be able to see the whole of one without reading the
    other.
    """
    has_callable = inputs.strategy is not None and inputs.strategy_data is not None

    # -- Does the same input give the same output? --------------------------
    if has_callable:
        try:
            determinism = determinism_check(inputs.strategy, inputs.strategy_data)
            report.sections["determinism"] = determinism.to_dict()
            finding = determinism.to_finding()
            if finding is not None:
                findings.append(finding)
        except ValueError as exc:
            report.not_tested.append(f"Determinism: {exc}")
    else:
        report.not_tested.append(
            "Determinism: needs a re-runnable strategy callable. Without it the "
            "report cannot confirm that the numbers it quotes would come back the "
            "same way twice."
        )

    # -- Is there a decision here at all? -----------------------------------
    book = inputs.positions
    if book is None and has_callable:
        try:
            book = inputs.strategy(
                np.asarray(
                    getattr(
                        inputs.strategy_data, "to_numpy", lambda: inputs.strategy_data
                    )(),
                    dtype=float,
                )
            )
        except Exception:  # pragma: no cover - a broken adapter is reported above
            book = None
    if book is not None:
        try:
            degeneracy = signal_degeneracy(book)
            report.sections["degeneracy"] = degeneracy.to_dict()
            finding = degeneracy.to_finding()
            if finding is not None:
                findings.append(finding)
        except ValueError as exc:
            report.not_tested.append(f"Signal degeneracy: {exc}")
    else:
        report.not_tested.append(
            "Signal degeneracy: no position series and no callable, so whether the "
            "book actually moves could not be checked. Every leakage result below "
            "should be read as uninformative until it is."
        )

    # -- Does the edge survive being traded late? ---------------------------
    if inputs.positions is not None and inputs.asset_return_frame is not None:
        try:
            delay = execution_delay_curve(
                inputs.positions,
                inputs.asset_return_frame,
                periods_per_year=inputs.periods_per_year,
                risk_free=rf_per_period,
            )
            report.sections["execution_delay"] = delay.to_dict()
            report.chart_data["execution_delay"] = (
                list(delay.delays),
                list(delay.sharpes),
            )
            finding = delay.to_finding()
            if finding is not None:
                findings.append(finding)
        except ValueError as exc:
            report.not_tested.append(f"Execution-delay fragility: {exc}")
    else:
        report.not_tested.append(
            "Execution-delay fragility: needs the position series and the per-asset "
            "returns it traded. This is the test that separates an edge from a "
            "timing assumption, so it is worth supplying them."
        )

    # -- Could the universe have been assembled with hindsight? -------------
    if inputs.raw_prices is not None:
        try:
            coverage = universe_coverage(inputs.raw_prices, membership=inputs.membership)
            report.sections["universe_coverage"] = coverage.to_dict()
            report.chart_data["universe_coverage"] = coverage
            finding = coverage.to_finding()
            if finding is not None:
                findings.append(finding)
        except ValueError as exc:
            report.not_tested.append(f"Universe coverage: {exc}")
    else:
        report.not_tested.append(
            "Universe coverage: the unaligned price frame was not supplied, so when "
            "each instrument's history begins could not be checked. Aligning to a "
            "common calendar first is what makes a ragged universe look clean."
        )

    # -- Does it read the future? -------------------------------------------
    if has_callable:
        try:
            leak = perturbation_test(inputs.strategy, inputs.strategy_data, seed=inputs.seed)
            report.sections["perturbation"] = leak.to_dict()
            gaps, profile = leak.horizon_profile()
            report.chart_data["lookahead_horizon"] = (
                list(gaps),
                list(profile),
                leak.detection_floor,
            )
            leak_finding = leak.to_finding()
            if leak_finding is not None:
                findings.append(leak_finding)
        except ValueError as exc:
            report.not_tested.append(f"Behavioural leakage test: {exc}")
    else:
        report.not_tested.append(
            "Behavioural leakage test: needs a re-runnable strategy callable. "
            "This is the only test that can settle whether the strategy reads the "
            "future, and it is the strongest reason to supply one."
        )


def _mean_risk_free(risk_free) -> float:
    """The sample mean of the supplied risk-free rate, or zero if none.

    A scalar mean rather than the series itself: for a Sharpe the two are
    identical - ``mean(r) - mean(rf) == mean(r - rf)``, and subtracting a
    constant leaves the standard deviation alone - and it keeps every
    downstream signature unchanged. The difference only reaches the standard
    error, where the variance of a daily bill rate is negligible beside the
    variance of anything worth auditing.
    """
    if risk_free is None:
        return 0.0
    arr = np.asarray(risk_free, dtype=float)
    if arr.ndim == 0:
        return float(arr) if math.isfinite(float(arr)) else 0.0
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else 0.0


def _mean_idle_weight(positions) -> float | None:
    """Average weight left uninvested, or ``None`` when that is not meaningful.

    Only the long-only case is answerable: a book whose weights sum to 0.7 has
    30% sitting in cash, and whether that cash earned the bill rate changes the
    return series by a real amount. A long-short book nets to something near
    zero for reasons that have nothing to do with idle balances, so the
    question is not asked of one.
    """
    if positions is None:
        return None
    arr = np.asarray(positions, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0 or np.any(arr < 0.0):
        return None
    net = np.nansum(arr, axis=1)
    return float(np.mean(np.clip(1.0 - net, 0.0, 1.0)))


def _hash_of(*arrays) -> str:
    """A stable digest of the inputs, for the reproducibility footer."""
    digest = sha256()
    for arr in arrays:
        if arr is None:
            digest.update(b"none")
            continue
        a = np.ascontiguousarray(np.asarray(arr, dtype=float))
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()[:16]


def run_audit(inputs: AuditInputs) -> AuditReport:
    """Run every test the supplied data supports and collect the findings."""
    returns = as_returns_array(inputs.returns)
    n = returns.size
    if n < 4:
        raise ValueError(f"an audit needs at least 4 return observations, got {n}")

    report = AuditReport(
        name=inputs.name,
        tier=inputs.tier,
        suite=inputs.suite,
        n_obs=n,
        periods_per_year=inputs.periods_per_year,
    )
    findings: list[Finding] = []
    stages: list[tuple[str, float, float | None, float | None]] = []

    # -- Basis --------------------------------------------------------------
    # The Sharpe ratio is defined on returns in excess of cash. Computed on
    # total returns it flatters a low-volatility strategy against a volatile
    # benchmark, because the same cash rate is a larger share of a smaller
    # denominator - and that comparison is what this report leads with. So
    # every risk-adjusted statistic below runs on `excess`, while the level and
    # path statistics - total return, drawdown, Calmar, the equity curve - stay
    # on `returns`, which is what an investor actually lived through.
    statistical = inputs.suite.runs_statistical
    engine = inputs.suite.runs_engine

    rf_per_period = _mean_risk_free(inputs.risk_free)
    excess = returns - rf_per_period
    excess_trials = (
        None
        if inputs.trial_returns is None
        else np.asarray(inputs.trial_returns, dtype=float) - rf_per_period
    )
    idle_weight = _mean_idle_weight(inputs.positions)
    report.sections["conventions"] = {
        "risk_free_supplied": inputs.risk_free is not None,
        "risk_free_per_period": rf_per_period,
        "risk_free_annualised": rf_per_period * inputs.periods_per_year,
        "sharpe_basis": (
            "excess of the risk-free rate"
            if inputs.risk_free is not None
            else "total returns (no risk-free rate supplied)"
        ),
        "level_basis": "total returns",
        "mean_idle_weight": idle_weight,
        "idle_cash_understatement": (
            None
            if idle_weight is None or rf_per_period <= 0.0
            else idle_weight * rf_per_period * inputs.periods_per_year
        ),
    }
    if inputs.risk_free is None:
        report.not_tested.append(
            "Risk-free adjustment: none was supplied, so every Sharpe on this page is "
            "computed on total returns. That overstates the risk-adjusted result by the "
            "cash rate divided by the volatility, and it overstates it most for the "
            "least volatile series in any comparison. Ken French's RF is the intended "
            "source."
        )

    regime_mask = None

    # -- Data provenance: survivorship (engine) -----------------------------
    if engine:
        _run_survivorship(inputs, report, findings)
    # Sub-tests all work in per-period Sharpe. The report speaks in annualised
    # Sharpe throughout, so each section carries the scaled figure too - a page
    # that says 0.59 in one table and 0.037 in the next is not a report, it is
    # a puzzle.
    ann = math.sqrt(inputs.periods_per_year)
    # Attribution has two detections - the factor regression and the
    # volatility-matched benchmark - of one defect. They are gathered here and
    # reported as a single finding.
    attribution_evidence: list[str] = []
    attribution_detail: dict[str, Any] = {}
    attribution_severity = Severity.INFO

    # -- Statistical validation ---------------------------------------------
    if statistical:
        sharpe = analyse_sharpe(
            returns,
            inputs.periods_per_year,
            rf_per_period=rf_per_period,
            confidence=inputs.confidence,
        )
        report.sections["sharpe"] = sharpe.to_dict()
        # "Strategy" rather than the full name: this label becomes a table column
        # header beside the benchmark ticker, and a manifest title runs to a line
        # and a half there. The report heading already says what was audited.
        report.sections["performance"] = summarise_performance(
            returns, inputs.periods_per_year, name="Strategy", rf_per_period=rf_per_period
        ).to_dict()
        if rf_per_period > 0.0:
            # Two bars rather than one, because the move from a total-return Sharpe
            # to an excess one is an adjustment like any other and belongs in the
            # cascade where the reader can see its size. It is often larger than
            # the cost adjustment underneath it.
            stages.append(("Claimed (total returns)", sharpe_ratio(returns) * ann, None, None))
            stages.append(
                ("Excess of the risk-free rate", sharpe.annualised_naive.value, None, None)
            )
        else:
            stages.append(
                ("Claimed (naive annualisation)", sharpe.annualised_naive.value, None, None)
            )

        if n < MIN_OBS_FOR_ASYMPTOTICS:
            findings.append(
                make_finding(
                    "STAT-SMALL-SAMPLE",
                    detail=(
                        f"{n} observations is below the {MIN_OBS_FOR_ASYMPTOTICS} at which the "
                        "asymptotic methods used throughout this report become trustworthy. "
                        "Every interval below is indicative rather than inferential."
                    ),
                    evidence={"n_obs": n, "threshold": MIN_OBS_FOR_ASYMPTOTICS},
                )
            )

        if sharpe.autocorrelation_haircut > 0.10:
            findings.append(
                make_finding(
                    "STAT-NAIVE-ANNUALISATION",
                    detail=(
                        f"Autocorrelation-adjusted annualisation is "
                        f"{sharpe.autocorrelation_haircut:.1%} below the naive sqrt(q) figure: "
                        f"{sharpe.annualised_naive.value:.2f} becomes "
                        f"{sharpe.annualised_adjusted.value:.2f}."
                    ),
                    evidence={
                        "naive": sharpe.annualised_naive.value,
                        "adjusted": sharpe.annualised_adjusted.value,
                        "haircut": sharpe.autocorrelation_haircut,
                    },
                )
            )

        # -- Bootstrap interval (the one the report leads with) -----------------
        boot = stationary_bootstrap(
            excess,
            sharpe_ratio,
            n_boot=inputs.n_boot,
            confidence=inputs.confidence,
            seed=inputs.seed,
        )
        scale = sharpe.factor_lo
        report.sections["bootstrap"] = {
            **boot.to_dict(),
            "annualised_observed": boot.observed * scale,
            "annualised_ci_low": boot.ci_low * scale,
            "annualised_ci_high": boot.ci_high * scale,
            # The same resampled interval scaled the naive way, so the headline row
            # reporting the naive figure can report its uncertainty too rather than
            # leaving a dash where a number belongs.
            "naive_ci_low": boot.ci_low * sharpe.factor_naive,
            "naive_ci_high": boot.ci_high * sharpe.factor_naive,
        }
        stages.append(
            (
                "Autocorrelation-adjusted",
                sharpe.annualised_adjusted.value,
                boot.ci_low * scale,
                boot.ci_high * scale,
            )
        )

        # -- Risk: how much of the realised result was the path? -----------------
        try:
            risk = simulate_risk(
                returns,
                periods_per_year=inputs.periods_per_year,
                rf_per_period=rf_per_period,
                n_boot=min(inputs.n_boot, 1500),
                n_baseline=min(inputs.n_boot, 1500),
                seed=inputs.seed,
            )
            report.sections["risk"] = risk.to_dict()
            report.chart_data["risk_draws"] = (
                risk.total_return.draws.tolist(),
                risk.sharpe.draws.tolist(),
                risk.max_drawdown.draws.tolist(),
                None if risk.max_drawdown.baseline_draws is None
                else risk.max_drawdown.baseline_draws.tolist(),
            )
            drawdown_finding = risk.to_finding()
            if drawdown_finding is not None:
                findings.append(drawdown_finding)
        except ValueError as exc:
            report.not_tested.append(f"Resampled risk distributions: {exc}")

        # -- Costs --------------------------------------------------------------
        net_returns = excess
        cost_basis = "gross"
        cost_basis_note = "Deflation runs on gross returns; no position series was supplied."
        if inputs.positions is not None:
            be = break_even_cost(
                excess,
                inputs.positions,
                asset_class=inputs.asset_class,
                periods_per_year=inputs.periods_per_year,
            )
            grid, curve = sharpe_vs_cost_curve(excess, inputs.positions)
            report.sections["costs"] = {
                **be.to_dict(),
                "average_turnover": average_turnover(inputs.positions, inputs.periods_per_year),
            }
            report.chart_data["cost_curve"] = (grid.tolist(), curve.tolist())
            # Fires on the margin, not on a hard pass/fail. A break-even of 5.8 bps
            # against a 2-5 bps cost range clears the binary test while leaving 16%
            # of headroom, which is not a strategy that survives contact with a real
            # desk - and reporting it as clean would be the kind of false comfort
            # this tool exists to remove.
            if be.realistic_bps is not None and be.severity >= Severity.MEDIUM:
                verb = (
                    "below" if be.survives_realistic_costs is False else "barely above"
                )
                findings.append(
                    make_finding(
                        "COST-BELOW-REALISTIC",
                        detail=(
                            f"Break-even cost is {be.break_even_bps:.1f} bps, {verb} the "
                            f"{be.realistic_bps[0]:.0f}-{be.realistic_bps[1]:.0f} bps that "
                            f"trading "
                            f"{ASSET_CLASS_LABELS.get(inputs.asset_class, inputs.asset_class)} "
                            f"realistically costs "
                            f"- a margin of {be.margin:.2f}x."
                        ),
                        severity=be.severity,
                        evidence=be.to_dict(),
                    )
                )
            if be.realistic_bps is not None and math.isfinite(be.break_even_bps):
                from qv.costs.models import FixedBpsCost, apply_costs

                net_returns = apply_costs(
                    excess, inputs.positions, FixedBpsCost(bps=be.realistic_bps[1])
                )
                cost_basis = "net of costs"
                cost_basis_note = (
                    f"Deflation runs on returns net of {be.realistic_bps[1]:.0f} bps, the upper "
                    f"end of realistic cost for "
                    f"{ASSET_CLASS_LABELS.get(inputs.asset_class, inputs.asset_class)}, so this "
                    "figure sits slightly below the gross Sharpe in the headline."
                )
                net = analyse_sharpe(net_returns, inputs.periods_per_year)
                stages.append(
                    (
                        f"Net of {be.realistic_bps[1]:.0f} bps costs",
                        net.annualised_adjusted.value,
                        None,
                        None,
                    )
                )
        else:
            report.not_tested.append(
                "Trading costs and break-even cost: no position series was supplied, so "
                "turnover cannot be measured. Supply the position series."
            )
            findings.append(
                make_finding(
                    "COST-ASSUMED-NOT-DERIVED",
                    detail=(
                        "No positions were supplied, so costs could not be derived from "
                        "turnover. Whatever cost the original backtest assumed is unverified."
                    ),
                    evidence={"tier": int(inputs.tier)},
                )
            )

        # -- Selection bias -----------------------------------------------------
        n_trials = inputs.n_trials
        if excess_trials is not None and n_trials is None:
            n_trials = int(excess_trials.shape[1])

        if n_trials is not None and n_trials >= 1:
            dsr = deflated_sharpe_ratio(
                net_returns, n_trials=n_trials, trial_returns=excess_trials
            )
            minbtl = minimum_backtest_length(n_trials, target_annual_sharpe=1.0)
            years = n / inputs.periods_per_year
            # The trial t-statistics, so BHY is applied over the family of tests it
            # is defined on rather than the single-test bound. Measured with the
            # same estimator as the audited result, or the audited result would not
            # find itself in its own family.
            trial_tstats = None
            if excess_trials is not None and excess_trials.shape[0] >= 4:
                trial_tstats = columnwise_sharpe_tstat(excess_trials)
            haircut = bhy_haircut(
                sharpe.tstat,
                n_trials,
                observed_sharpe=sharpe.annualised_adjusted.value,
                trial_tstats=trial_tstats,
            )
            report.sections["selection"] = {
                "deflated_sharpe": dsr.to_dict(),
                "basis": cost_basis,
                "basis_note": cost_basis_note,
                "minimum_backtest_length_years": minbtl,
                "sample_years": years,
                "sample_long_enough": years >= minbtl,
                "haircut": haircut.to_dict(),
            }
            # Applied to the running figure, not back to the gross one: a cascade
            # whose last bar silently reverts to an earlier base is not a cascade.
            # And it is labelled for the test that produced it - this is the BHY
            # false-discovery haircut, not the Deflated Sharpe, which is reported
            # separately and, on some samples, disagrees with it.
            running = stages[-1][1] if stages else sharpe.annualised_adjusted.value
            stages.append(
                (
                    f"After the BHY haircut ({n_trials} trials)",
                    running * (1.0 - haircut.haircut),
                    None,
                    None,
                    (
                        f"fully absorbed - {n_trials} trials needed "
                        f"t = {haircut.required_tstat:.2f}, this has "
                        f"{haircut.observed_tstat:.2f}"
                    )
                    if haircut.censored
                    else None,
                )
            )

            if not dsr.survives:
                findings.append(
                    make_finding(
                        "SELECT-DEFLATED-SHARPE-FAILS",
                        detail=(
                            f"Deflated Sharpe is {dsr.dsr:.3f} against {n_trials} trials. The "
                            f"observed per-period Sharpe of {dsr.observed_sharpe:.4f} sits against "
                            f"an expected best-of-{n_trials} under the null of "
                            f"{dsr.expected_max_sharpe:.4f}."
                        ),
                        severity=dsr.severity,
                        evidence=dsr.to_dict(),
                    )
                )
            if years < minbtl:
                findings.append(
                    make_finding(
                        "SELECT-INSUFFICIENT-LENGTH",
                        detail=(
                            f"{years:.1f} years of data against a minimum backtest length of "
                            f"{minbtl:.1f} years for {n_trials} trials at a target Sharpe of 1.0."
                        ),
                        evidence={"sample_years": years, "min_years": minbtl, "n_trials": n_trials},
                    )
                )

            # Empirical null: a validity check on the closed form, plus the chart.
            try:
                if excess_trials is not None:
                    null = empirical_max_sharpe_null(
                        excess_trials, n_sims=min(inputs.n_boot, 1500), seed=inputs.seed
                    )
                else:
                    null = gaussian_max_sharpe_null(
                        dsr.observed_sharpe,
                        n_trials,
                        dsr.trial_sharpe_std,
                        n_sims=20_000,
                        seed=inputs.seed,
                    )
                report.sections["null_max"] = null.to_dict()
                report.chart_data["null_distribution"] = null.null_distribution.tolist()
            except ValueError as exc:
                report.not_tested.append(f"Empirical max-Sharpe null: {exc}")
        else:
            report.not_tested.append(
                "Deflated Sharpe, minimum backtest length and the multiple-testing haircut: "
                "no trial count was declared. This is the most consequential gap in any audit "
                "- see the trial-archaeology tooling for recovering a lower bound."
            )
            findings.append(
                make_finding(
                    "SELECT-UNDECLARED-TRIALS",
                    detail=(
                        "No trial count was declared, so selection bias could not be priced at "
                        "all. A backtest reported without its trial count is not interpretable."
                    ),
                    evidence={},
                )
            )

        # -- PBO ----------------------------------------------------------------
        if excess_trials is not None:
            try:
                pbo = combinatorially_symmetric_cv(excess_trials)
                report.sections["pbo"] = pbo.to_dict()
                report.chart_data["pbo"] = (
                    pbo.logits.tolist(),
                    pbo.is_best_sharpe.tolist(),
                    pbo.oos_of_is_best.tolist(),
                )
                if pbo.overfit:
                    findings.append(
                        make_finding(
                            "SELECT-HIGH-PBO",
                            detail=(
                                f"PBO is {pbo.pbo:.1%} across {pbo.n_combinations} splits: the "
                                "in-sample winner lands in the bottom half out-of-sample more "
                                f"often than not, and loses money in {pbo.probability_of_loss:.1%} "
                                "of them."
                            ),
                            evidence=pbo.to_dict(),
                        )
                    )
            except ValueError as exc:
                report.not_tested.append(f"Probability of backtest overfitting: {exc}")
        else:
            report.not_tested.append(
                "Probability of backtest overfitting: requires the matrix of trial returns "
                "- either declare the grid in a manifest, or supply the matrix directly."
            )

        # -- Attribution --------------------------------------------------------
        if inputs.factors is not None:
            attribution = factor_attribution(
                returns,
                inputs.factors,
                inputs.factor_names,
                rf=inputs.risk_free,
                periods_per_year=inputs.periods_per_year,
            )
            report.sections["attribution"] = attribution.to_dict()
            if not attribution.alpha_survives:
                attribution_evidence.append(
                    f"alpha is {attribution.alpha_annualised:.2%} a year with a t-statistic of "
                    f"{attribution.alpha_tstat:.2f} (p={attribution.alpha_p_value:.3f}) and the "
                    f"factors explain {attribution.r_squared:.0%} of the variance"
                )
                attribution_detail["factor_regression"] = attribution.to_dict()
                attribution_severity = max(attribution_severity, attribution.severity)
        else:
            report.not_tested.append(
                "Factor attribution: no factor returns were supplied (Ken French FF5 plus "
                "momentum is the intended source)."
            )

        if inputs.benchmark_returns is not None:
            # Both sides in excess terms, or the comparison rewards the lower-
            # volatility series for cash it never earned.
            benchmark_excess = as_returns_array(inputs.benchmark_returns) - rf_per_period
            comparison = vol_matched_benchmark(
                excess, benchmark_excess, inputs.benchmark_name
            )
            report.sections["benchmark"] = {
                **comparison.to_dict(),
                "strategy_sharpe_annualised": comparison.strategy_sharpe * ann,
                "benchmark_sharpe_annualised": comparison.benchmark_sharpe * ann,
                "scaled_benchmark_sharpe_annualised": comparison.scaled_benchmark_sharpe * ann,
                "excess_sharpe_annualised": comparison.excess_sharpe * ann,
            }
            # The same metrics computed the same way for both, so the comparison is
            # like for like rather than the strategy's numbers against whatever the
            # benchmark's provider happened to publish.
            report.sections["benchmark_performance"] = summarise_performance(
                inputs.benchmark_returns,
                inputs.periods_per_year,
                name=inputs.benchmark_name,
                rf_per_period=rf_per_period,
            ).to_dict()
            report.chart_data["benchmark_returns"] = np.asarray(
                as_returns_array(inputs.benchmark_returns)
            ).tolist()
            report.chart_data["benchmark_name"] = inputs.benchmark_name
            if comparison.is_levered_beta:
                attribution_evidence.append(
                    f"correlation to {comparison.benchmark_name} is {comparison.correlation:.2f} and "
                    f"the strategy adds {comparison.excess_sharpe * ann:+.3f} annualised Sharpe over "
                    f"simply holding it at {comparison.leverage:.2f}x leverage"
                )
                attribution_detail["benchmark"] = comparison.to_dict()
                attribution_severity = max(attribution_severity, comparison.severity)
        else:
            report.not_tested.append("Volatility-matched benchmark: no benchmark returns supplied.")

        if attribution_evidence:
            findings.append(
                make_finding(
                    "ATTR-LEVERED-BETA",
                    detail=(
                        "The returns are explained by exposures you can buy directly: "
                        + "; ".join(attribution_evidence)
                        + "."
                    ),
                    severity=attribution_severity,
                    evidence=attribution_detail,
                )
            )

        # -- Robustness ---------------------------------------------------------
        try:
            top = top_day_dependence(excess)
            report.sections["top_days"] = {
                **top.to_dict(),
                "full_sharpe_annualised": top.full_sharpe * ann,
                "sharpes_annualised": [v * ann for v in top.sharpes],
                "sharpe_after_dropping_top_1pct_annualised":
                    top.sharpe_after_dropping_top_1pct * ann,
                "sharpe_after_dropping_top_5pct_annualised":
                    top.sharpe_after_dropping_top_5pct * ann,
            }
            if top.abnormally_concentrated:
                findings.append(
                    make_finding(
                        "ROBUST-TOP-DAY-DEPENDENT",
                        detail=(
                            f"The best periods supply {top.worst_concentration_ratio:.1f}x the "
                            "profit share a normal series of this Sharpe would produce."
                        ),
                        severity=top.severity,
                        evidence=top.to_dict(),
                    )
                )
        except ValueError as exc:
            report.not_tested.append(f"Top-day dependence: {exc}")

        try:
            rolling = rolling_origin_sensitivity(excess)
            report.sections["rolling_origin"] = {
                **rolling.to_dict(),
                "min_sharpe_annualised": rolling.min_sharpe * ann,
                "max_sharpe_annualised": rolling.max_sharpe * ann,
                "sharpes_annualised": [v * ann for v in rolling.sharpes],
            }
            if rolling.start_date_dependent:
                swing = (
                    "and changes sign"
                    if not rolling.sign_stable
                    else f"- the weakest is {rolling.relative_stability:.0%} of the strongest"
                )
                findings.append(
                    make_finding(
                        "ROBUST-START-DATE-SENSITIVE",
                        detail=(
                            f"Across {len(rolling.sharpes)} start dates the Sharpe ranges from "
                            f"{rolling.min_sharpe:.3f} to {rolling.max_sharpe:.3f} {swing}."
                        ),
                        severity=rolling.severity,
                        evidence=rolling.to_dict(),
                    )
                )
        except ValueError as exc:
            report.not_tested.append(f"Rolling-origin sensitivity: {exc}")

        regime_mask = None
        try:
            window = max(5, min(63, n // 8))
            regimes = volatility_regime_split(excess, window=window)
            report.sections["regimes"] = {
                **regimes.to_dict(),
                "sharpes_annualised": [v * ann for v in regimes.sharpes],
                "full_sharpe_annualised": regimes.full_sharpe * ann,
            }
            vol = trailing_volatility(excess, window=window)
            regime_mask = np.isfinite(vol) & (vol > regimes.threshold)
            if not regimes.sign_stable:
                findings.append(
                    make_finding(
                        "ROBUST-REGIME-DEPENDENT",
                        detail=(
                            f"Sharpe is {regimes.sharpes[0]:.3f} in the low-volatility regime and "
                            f"{regimes.sharpes[1]:.3f} in the high-volatility regime."
                        ),
                        severity=regimes.severity,
                        evidence=regimes.to_dict(),
                    )
                )
        except ValueError as exc:
            report.not_tested.append(f"Volatility-regime split: {exc}")

        try:
            flip = sign_flip_test(excess, n_sims=min(inputs.n_boot, 1000), seed=inputs.seed)
            report.sections["sign_flip"] = flip.to_dict()
        except ValueError as exc:
            report.not_tested.append(f"Sign-flip randomisation: {exc}")

        matched_positions = (
            None if inputs.positions is None else np.asarray(inputs.positions, dtype=float)
        )
        if matched_positions is not None and matched_positions.ndim > 1:
            report.not_tested.append(
                "Matched-exposure random-entry test: the position series is multi-asset. This "
                "test asks whether a single instrument was timed well, and reordering a "
                "cross-sectional book in time does not answer that question."
            )
        elif matched_positions is not None and float(np.ptp(matched_positions)) == 0.0:
            report.not_tested.append(
                "Matched-exposure random-entry test: exposure is constant, so there is no "
                "timing decision to randomise. A test that shuffles an unchanging series would "
                "report a p-value near 1 and look like a finding when nothing was measured."
            )
        elif inputs.positions is not None and inputs.asset_returns is not None:
            try:
                matched = matched_exposure_test(
                    inputs.asset_returns,
                    inputs.positions,
                    n_sims=min(inputs.n_boot, 1000),
                    seed=inputs.seed,
                )
                report.sections["matched_exposure"] = matched.to_dict()
                if matched.p_value > 0.20:
                    findings.append(
                        make_finding(
                            "ROBUST-NO-TIMING-SKILL",
                            detail=(
                                f"Random timing at the same average exposure matches the strategy "
                                f"(p={matched.p_value:.2f}). The edge comes from being in the "
                                "market, not from choosing when."
                            ),
                            severity=matched.severity,
                            evidence=matched.to_dict(),
                        )
                    )
            except ValueError as exc:
                report.not_tested.append(f"Matched-exposure randomisation: {exc}")
        else:
            report.not_tested.append(
                "Matched-exposure random-entry test: needs both the position series and the "
                "traded asset returns."
            )

        if inputs.parameter_scores is not None:
            try:
                plateau = plateau_ratio(
                    inputs.parameter_scores, chosen_index=inputs.chosen_parameter_index
                )
                report.sections["parameters"] = plateau.to_dict()
                if plateau.is_spike:
                    findings.append(
                        make_finding(
                            "SELECT-PARAMETER-SPIKE",
                            detail=(
                                f"Neighbouring parameters retain only {plateau.ratio:.0%} of the "
                                "chosen point, which is a spike rather than a plateau."
                            ),
                            severity=plateau.severity,
                            evidence=plateau.to_dict(),
                        )
                    )
            except ValueError as exc:
                report.not_tested.append(f"Parameter plateau: {exc}")
        else:
            report.not_tested.append(
                "Parameter-neighbourhood plateau: no parameter grid scores were supplied."
            )


    # -- Engine analysis ----------------------------------------------------
    if engine:
        _run_engine_suite(inputs, report, findings, excess, rf_per_period)

    report.sections["_stages"] = stages
    report.chart_data["regime_mask"] = regime_mask
    if inputs.dates is not None:
        dates = np.asarray(getattr(inputs.dates, "to_numpy", lambda: inputs.dates)())
        # Presentational only, so a length mismatch drops the axis labels rather
        # than failing an audit that is otherwise complete.
        report.chart_data["dates"] = dates if dates.size == n else None
    report.findings = findings
    report.provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_hash": _hash_of(returns, inputs.positions, inputs.trial_returns),
        "seed": inputs.seed,
        "n_boot": inputs.n_boot,
        "confidence": inputs.confidence,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        # Which engine produced this. The footer has always promised that the
        # same inputs and seed reproduce every number, and without these the
        # promise names no code for the reader to re-run.
        **source_identity(),
    }
    return report
