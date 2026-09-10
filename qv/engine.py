"""Engine analysis: is the backtest trustworthy as an implementation?

The statistical suite asks whether a measured edge is distinguishable from
luck. This one asks something the statistics cannot see at all: whether the
machine that produced those numbers can be believed. The two are independent,
and the dangerous combination is a sound-looking number from an unsound
implementation - because nothing in the return series betrays it.

Five questions, each answering something no other check here answers:

============================  ================================================
Question                      Test
============================  ================================================
Does it read the future?      Behavioural perturbation, plus a horizon profile
                              that says how far the dependence reaches
Does the edge survive being   Execution-delay fragility: retime the book by a
traded late?                  day, two days, a week, and watch the Sharpe
Is the same input the same    Determinism: call it repeatedly and compare
output?
Is there anything to test?    Signal degeneracy - a book that never moves
                              passes every leakage test for the wrong reason
Could the universe have been  Universe coverage: which instruments cover the
assembled with hindsight?     whole sample, and which appear part-way through
============================  ================================================

Two of these were previously *assumed*. The adapter protocol has asked for
determinism since it was written and nothing verified it, and the anti-vacuity
condition - that the book actually varies - lived only in the test suite, so a
researcher whose adapter silently returned a flat book got a clean bill of
health on every leakage test and no hint of why.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qv.findings import make_finding
from qv.leakage.perturbation import _as_signals
from qv.stats.sharpe import sharpe_ratio
from qv.types import Finding, Severity

__all__ = [
    "DEFAULT_DELAYS",
    "DeterminismResult",
    "DegeneracyResult",
    "DelayFragility",
    "UniverseCoverage",
    "determinism_check",
    "signal_degeneracy",
    "execution_delay_curve",
    "universe_coverage",
]

#: Execution delays to price, in periods. One period is the interesting one -
#: a strategy whose edge dies when the book is traded a session late was never
#: tradeable - and the longer ones show whether decay is a cliff or a slope.
DEFAULT_DELAYS = (0, 1, 2, 3, 5, 10)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterminismResult:
    """Does the same input give the same output, every time?"""

    deterministic: bool
    n_calls: int
    max_difference: float

    def to_finding(self) -> Finding | None:
        if self.deterministic:
            return None
        return make_finding(
            "ENGINE-NONDETERMINISTIC",
            detail=(
                f"Called {self.n_calls} times on identical data, the strategy "
                f"returned different positions - the largest disagreement was "
                f"{self.max_difference:.3g}. Unseeded randomness makes every other "
                "result on this page unreproducible, and it makes the behavioural "
                "leakage test meaningless: a change after corruption can no longer "
                "be distinguished from the strategy simply disagreeing with itself."
            ),
            severity=Severity.CRITICAL,
            evidence={
                "n_calls": self.n_calls,
                "max_difference": self.max_difference,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "n_calls": self.n_calls,
            "max_difference": self.max_difference,
        }


def determinism_check(
    strategy: Callable[[np.ndarray], np.ndarray], data, n_calls: int = 3
) -> DeterminismResult:
    """Call ``strategy`` repeatedly on the same data and compare the results.

    The adapter protocol has required determinism from the beginning and
    nothing checked it. It matters most for the leakage test, which reads any
    difference between a clean run and a corrupted one as evidence of
    look-ahead - so a strategy that disagrees with itself reports as leaky, and
    the researcher goes hunting for a bug that is really a missing seed.
    """
    if n_calls < 2:
        raise ValueError(f"determinism needs at least 2 calls, got {n_calls}")

    array = np.asarray(getattr(data, "to_numpy", lambda: data)(), dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    n = array.shape[0]

    baseline = _as_signals(strategy(array), n)
    worst = 0.0
    for _ in range(n_calls - 1):
        again = _as_signals(strategy(array), n)
        if again.shape != baseline.shape:
            return DeterminismResult(False, n_calls, math.inf)
        both_nan = np.isnan(baseline) & np.isnan(again)
        diff = np.abs(baseline - again)
        diff = np.where(both_nan, 0.0, np.nan_to_num(diff, nan=np.inf))
        finite = diff[np.isfinite(diff)]
        if finite.size:
            worst = max(worst, float(finite.max()))
        if not np.isfinite(diff).all():
            worst = math.inf

    return DeterminismResult(worst == 0.0, n_calls, worst)


# --------------------------------------------------------------------------
# Degeneracy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DegeneracyResult:
    """Is there actually a decision here to examine?"""

    n_obs: int
    n_instruments: int
    n_distinct_books: int
    n_changes: int
    fraction_flat: float
    constant_exposure: bool
    never_invested: bool

    @property
    def degenerate(self) -> bool:
        """Too little variation for the leakage test to mean anything."""
        return self.never_invested or self.n_changes < 2

    def to_finding(self) -> Finding | None:
        if self.never_invested:
            return make_finding(
                "ENGINE-NO-POSITIONS",
                detail=(
                    "The strategy is never invested: every position it returns is "
                    "zero. Nothing on this page describes a strategy, and every "
                    "leakage test passes for the least interesting reason available."
                ),
                severity=Severity.CRITICAL,
                evidence=self.to_dict(),
            )
        if self.n_changes < 2:
            return make_finding(
                "ENGINE-DEGENERATE-SIGNAL",
                detail=(
                    f"The book changes {self.n_changes} times in {self.n_obs} "
                    "observations, so there is almost no decision to examine. A "
                    "signal that does not move cannot be shown to depend on the "
                    "future, which means the behavioural leakage result here is "
                    "uninformative rather than reassuring."
                ),
                severity=Severity.HIGH,
                evidence=self.to_dict(),
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "n_instruments": self.n_instruments,
            "n_distinct_books": self.n_distinct_books,
            "n_changes": self.n_changes,
            "fraction_flat": self.fraction_flat,
            "constant_exposure": self.constant_exposure,
            "never_invested": self.never_invested,
            "degenerate": self.degenerate,
        }


def signal_degeneracy(signals, tolerance: float = 1e-12) -> DegeneracyResult:
    """How much the book actually varies.

    This is the anti-vacuity check. Every leakage test here reports "clean" for
    a strategy that holds nothing, or holds the same thing forever, and that
    clean result has been earned by nothing at all. Making it a finding is the
    difference between a reassuring report and an honest one.
    """
    book = np.asarray(getattr(signals, "to_numpy", lambda: signals)(), dtype=float)
    if book.ndim == 1:
        book = book[:, None]
    if book.ndim != 2:
        raise ValueError(f"signals must be 1-D or 2-D, got shape {book.shape}")
    n, k = book.shape
    if n < 2:
        raise ValueError(f"degeneracy needs at least 2 observations, got {n}")

    filled = np.nan_to_num(book, nan=0.0)
    gross = np.abs(filled).sum(axis=1)
    step = np.abs(np.diff(filled, axis=0)).max(axis=1)

    return DegeneracyResult(
        n_obs=n,
        n_instruments=k,
        n_distinct_books=int(len(np.unique(filled.round(12), axis=0))),
        n_changes=int((step > tolerance).sum()),
        fraction_flat=float((gross <= tolerance).mean()),
        constant_exposure=bool(float(np.ptp(gross)) <= tolerance),
        never_invested=bool(gross.max() <= tolerance),
    )


# --------------------------------------------------------------------------
# Execution-delay fragility
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DelayFragility:
    """What the edge is worth if the book is traded late."""

    delays: tuple[int, ...]
    sharpes: tuple[float, ...]
    periods_per_year: int
    retention: tuple[float, ...] = field(default_factory=tuple)

    @property
    def base_sharpe(self) -> float:
        return self.sharpes[0] if self.sharpes else math.nan

    @property
    def one_period_retention(self) -> float:
        """Share of the Sharpe surviving a single period of delay."""
        if len(self.sharpes) < 2 or not math.isfinite(self.base_sharpe):
            return math.nan
        if self.base_sharpe <= 0:
            return math.nan
        return self.sharpes[1] / self.base_sharpe

    @property
    def surviving_delay(self) -> int | None:
        """The largest delay at which the Sharpe is still positive."""
        survivors = [d for d, s in zip(self.delays, self.sharpes) if s > 0]
        return max(survivors) if survivors else None

    @property
    def severity(self) -> Severity:
        retention = self.one_period_retention
        if not math.isfinite(retention):
            return Severity.INFO
        if retention >= 0.8:
            return Severity.INFO
        if retention >= 0.5:
            return Severity.MEDIUM
        if retention >= 0.2:
            return Severity.HIGH
        return Severity.CRITICAL

    def to_finding(self) -> Finding | None:
        retention = self.one_period_retention
        if not math.isfinite(retention) or retention >= 0.8:
            return None
        return make_finding(
            "ENGINE-EXECUTION-FRAGILE",
            detail=(
                f"Trading the same book one period later retains "
                f"{retention:.0%} of the Sharpe ({self.base_sharpe:.2f} to "
                f"{self.sharpes[1]:.2f} annualised). An edge that depends on "
                "executing at the moment the signal is computed is an artifact of "
                "the backtest's timing assumptions rather than a property of the "
                "market, and no real book is filled that promptly."
            ),
            severity=self.severity,
            evidence={
                "base_sharpe": self.base_sharpe,
                "delayed_sharpe": self.sharpes[1],
                "one_period_retention": retention,
                "surviving_delay": self.surviving_delay,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "delays": list(self.delays),
            "sharpes": [None if not math.isfinite(s) else s for s in self.sharpes],
            "retention": [
                None if not math.isfinite(r) else r for r in self.retention
            ],
            "base_sharpe": self.base_sharpe,
            "one_period_retention": self.one_period_retention,
            "surviving_delay": self.surviving_delay,
            "severity": self.severity.name.lower(),
            "periods_per_year": self.periods_per_year,
        }


def execution_delay_curve(
    positions,
    asset_returns,
    periods_per_year: int = 252,
    delays: tuple[int, ...] = DEFAULT_DELAYS,
    risk_free: float = 0.0,
) -> DelayFragility:
    """Annualised Sharpe as the whole book is traded progressively later.

    Distinct from break-even cost, which asks what the edge can pay in fees.
    This asks what it can pay in *time*, and the two fail differently: a
    strategy trading nine liquid ETFs monthly has an enormous cost cushion and
    can still lose everything to a one-day fill delay, because the signal is
    really picking up a one-day reversal.

    The delay is applied to the positions rather than to the returns, so the
    strategy is not re-run and nothing about its logic is assumed - this is
    arithmetic on the book it produced.
    """
    book = np.asarray(getattr(positions, "to_numpy", lambda: positions)(), dtype=float)
    rets = np.asarray(
        getattr(asset_returns, "to_numpy", lambda: asset_returns)(), dtype=float
    )
    if book.ndim == 1:
        book = book[:, None]
    if rets.ndim == 1:
        rets = rets[:, None]
    if book.shape[0] != rets.shape[0]:
        raise ValueError(
            f"positions have {book.shape[0]} rows but asset returns have "
            f"{rets.shape[0]}; they must describe the same time axis"
        )
    if book.shape[1] != rets.shape[1]:
        raise ValueError(
            f"positions have {book.shape[1]} columns but asset returns have "
            f"{rets.shape[1]}; they must describe the same instruments"
        )
    if not delays or min(delays) < 0:
        raise ValueError(f"delays must be non-negative and non-empty, got {delays}")
    if sorted(delays)[0] != 0:
        raise ValueError("the delay sweep must include 0, the reported timing")

    book = np.nan_to_num(book, nan=0.0)
    rets = np.nan_to_num(rets, nan=0.0)
    n = book.shape[0]
    scale = math.sqrt(periods_per_year)

    sharpes: list[float] = []
    for delay in delays:
        if delay >= n:
            sharpes.append(math.nan)
            continue
        shifted = np.zeros_like(book)
        if delay:
            shifted[delay:] = book[:-delay]
        else:
            shifted = book
        series = (shifted * rets).sum(axis=1) - risk_free
        try:
            # Naive sqrt scaling, applied identically at every delay. The
            # autocorrelation-adjusted annualisation belongs to the statistical
            # suite; here the quantity of interest is the *ratio* between
            # delays, and a correction that changes with the delay would move
            # the thing being measured.
            sharpes.append(sharpe_ratio(series) * scale)
        except ValueError:  # pragma: no cover - guarded by the length check above
            sharpes.append(math.nan)

    base = sharpes[0]
    retention = tuple(
        s / base if math.isfinite(s) and math.isfinite(base) and base > 0 else math.nan
        for s in sharpes
    )
    return DelayFragility(
        delays=tuple(delays),
        sharpes=tuple(sharpes),
        periods_per_year=periods_per_year,
        retention=retention,
    )


# --------------------------------------------------------------------------
# Universe coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseCoverage:
    """When each instrument's data begins and ends, against the whole sample.

    The visual counterpart to the survivorship declaration. Survivorship
    itself cannot be measured from a return series - a universe of survivors
    simply looks like a universe - but *inclusion timing* can be seen, and it
    is the same family of error: an instrument that enters part-way through
    was chosen in the knowledge that it would exist.
    """

    instruments: tuple[str, ...]
    first_index: tuple[int, ...]
    last_index: tuple[int, ...]
    n_obs: int
    labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def coverage(self) -> tuple[float, ...]:
        return tuple(
            (last - first + 1) / self.n_obs if self.n_obs else 0.0
            for first, last in zip(self.first_index, self.last_index)
        )

    @property
    def late_entrants(self) -> tuple[str, ...]:
        """Instruments whose history starts after the sample does."""
        return tuple(
            name
            for name, first in zip(self.instruments, self.first_index)
            if first > 0
        )

    @property
    def early_exits(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, last in zip(self.instruments, self.last_index)
            if last < self.n_obs - 1
        )

    @property
    def complete(self) -> bool:
        return not self.late_entrants and not self.early_exits

    def to_finding(self) -> Finding | None:
        if self.complete:
            return None
        parts = []
        if self.late_entrants:
            parts.append(
                f"{len(self.late_entrants)} instrument(s) start after the sample "
                f"does ({', '.join(self.late_entrants[:6])})"
            )
        if self.early_exits:
            parts.append(
                f"{len(self.early_exits)} stop before it ends "
                f"({', '.join(self.early_exits[:6])})"
            )
        return make_finding(
            "ENGINE-INCLUSION-TIMING",
            detail=(
                f"The universe is ragged: {'; '.join(parts)}. An instrument added "
                "the day it lists, or dropped the day it stops trading, is a "
                "position taken in the knowledge that it would exist - the same "
                "family of error as survivorship, and this one is visible in the "
                "data rather than only declarable."
            ),
            severity=Severity.MEDIUM,
            evidence={
                "late_entrants": list(self.late_entrants),
                "early_exits": list(self.early_exits),
                "n_obs": self.n_obs,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruments": list(self.instruments),
            "coverage": list(self.coverage),
            "first_index": list(self.first_index),
            "last_index": list(self.last_index),
            "late_entrants": list(self.late_entrants),
            "early_exits": list(self.early_exits),
            "complete": self.complete,
            "n_obs": self.n_obs,
        }


def universe_coverage(prices, labels=None) -> UniverseCoverage:
    """Which instruments cover the whole sample, and which do not.

    Pass the frame *before* any common-calendar alignment. Dropping the ragged
    rows first is what makes a universe look clean, so a coverage check run
    after alignment can only ever report that everything is fine.
    """
    frame = getattr(prices, "to_numpy", None)
    if frame is None:
        raise ValueError("prices must be a DataFrame or an array")

    names = (
        tuple(str(c) for c in prices.columns)
        if hasattr(prices, "columns")
        else tuple(f"col{i}" for i in range(np.asarray(prices).shape[1]))
    )
    array = np.asarray(frame(), dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    n = array.shape[0]
    if n < 2:
        raise ValueError(f"coverage needs at least 2 observations, got {n}")

    first: list[int] = []
    last: list[int] = []
    for column in range(array.shape[1]):
        valid = np.flatnonzero(np.isfinite(array[:, column]))
        if valid.size == 0:
            first.append(n - 1)
            last.append(n - 1)
        else:
            first.append(int(valid[0]))
            last.append(int(valid[-1]))

    index_labels: tuple[str, ...] = ()
    if labels is not None:
        index_labels = tuple(str(x) for x in labels)
    elif hasattr(prices, "index"):
        index_labels = tuple(str(x)[:10] for x in prices.index)

    return UniverseCoverage(
        instruments=names,
        first_index=tuple(first),
        last_index=tuple(last),
        n_obs=n,
        labels=index_labels,
    )
