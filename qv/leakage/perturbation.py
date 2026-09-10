"""The behavioural leakage test.

Corrupt the data strictly *after* time ``t``, re-run the strategy, and check
that the signal at ``t`` is unchanged. If it moves, the strategy reads the
future - whatever the source code appears to say.

This is the only test here that settles the question rather than raising it.
The AST scanner in :mod:`qv.leakage.static` recognises shapes and cannot follow
data through variables or see inside a library call. This one does not read the
code at all; it observes what the code *does*. A leak buried three layers deep
in someone else's package shows up exactly as clearly as a stray ``shift(-1)``.

It needs a re-runnable ``strategy(data) -> signals`` callable, which is the
single strongest argument for asking a researcher to supply one.

Two design decisions here were arrived at by measurement rather than taste, and
both are load-bearing enough that undoing them would quietly void the test.

**A cross-sectional book is compared per asset, never reduced to one number per
row.** An equal-weight rotation holds a gross exposure of 1.0 on nearly every
row, so a leak that changed *which* names were held would be invisible to any
scalar summary. Measured on a nine-sector momentum rotation: a leak that moves
30 pre-cut books changes the largest individual weight by 0.33 and the gross
exposure by exactly zero.

**Several cut points are swept, placed just after the signal actually moves.**
A single cut is close to useless on anything that rebalances less often than
every period. ``position[t]`` depends on the most recent rebalance at or before
``t``, so a leak can only surface if the signal updates between that rebalance
and the cut - and a cut dropped at a fixed fraction of the sample lands
mid-period, where nothing has updated. Measured on the same rotation, injecting
a known nine-session look-ahead: one cut at 70% detected it never, twelve
evenly spaced cuts caught it 5 times in 12, and eleven cuts placed after signal
changes caught it 11 times out of 11.

What the test cannot see is stated rather than hidden, because a clean result
here is easy to over-read. See :attr:`PerturbationResult.detection_floor`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.findings import make_finding
from qv.types import Finding, Severity

__all__ = [
    "CORRUPTION_MODES",
    "PerturbationResult",
    "perturbation_test",
]

#: How to destroy the information content of the future window. Several,
#: because a strategy can be accidentally invariant to any one of them - a
#: signal keyed on the sign of a return survives scaling but not shuffling.
CORRUPTION_MODES = ("shuffle", "noise", "constant", "reverse")

#: How many cut points to sweep when the caller does not pin one. Each cut
#: costs about ten strategy calls, so this is the knob to turn if an expensive
#: adapter makes an audit crawl - at the cost of sensitivity to short leaks.
DEFAULT_N_CUTS = 12


def _corrupt(
    data: np.ndarray, start: int, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Return a copy with rows from ``start`` onward destroyed."""
    out = np.array(data, dtype=float, copy=True)
    tail = out[start:]
    if tail.shape[0] == 0:
        return out

    if mode == "shuffle":
        out[start:] = rng.permutation(tail, axis=0)
    elif mode == "noise":
        scale = np.nanstd(out[:start], axis=0) if start > 1 else np.nanstd(tail, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        out[start:] = rng.normal(0.0, 1.0, size=tail.shape) * scale * 10.0
    elif mode == "constant":
        out[start:] = np.nanmean(out[:start], axis=0) if start > 0 else 0.0
    elif mode == "reverse":
        out[start:] = tail[::-1]
    else:
        raise ValueError(f"unknown corruption mode {mode!r}; expected one of {CORRUPTION_MODES}")
    return out


def _as_signals(raw: Any, n: int) -> np.ndarray:
    """Coerce a strategy's return value to an ``(n, k)`` float array.

    Two shapes are legitimate, because two kinds of strategy are: ``(n,)`` for
    one signal per row, and ``(n, k)`` for a cross-sectional book of ``k``
    weights per row. The row count is not negotiable. A strategy that hands
    back fewer rows than it was given - almost always a ``dropna()`` inside an
    adapter - cannot be aligned to the data it was given, and guessing the
    alignment is how a real leak gets absorbed into an off-by-one.
    """
    out = np.asarray(raw, dtype=float)
    shape = out.shape
    if out.ndim == 1:
        out = out[:, None]
    if out.ndim != 2 or out.shape[0] != n:
        raise ValueError(
            f"strategy returned signals of shape {shape} for {n} rows of data; it must "
            "return one signal per row - shape (n,), or (n, k) for a book of k weights"
        )
    return out


def _change_rows(signals: np.ndarray, tolerance: float) -> np.ndarray:
    """Row indices where the signal differs from the row before it."""
    if signals.shape[0] < 2:
        return np.empty(0, dtype=int)
    step = np.abs(np.diff(signals, axis=0))
    step = np.nan_to_num(step, nan=np.inf)
    return np.flatnonzero(step.max(axis=1) > tolerance) + 1


def _cut_points(
    signals: np.ndarray, n: int, n_cuts: int, tolerance: float
) -> tuple[int, ...]:
    """Where to cut, preferring the rows just after the signal moves.

    A leak surfaces only if the signal updates between its last update before
    the cut and the cut itself, so a cut placed one row after an update is
    worth many placed in the flat stretch that follows. Cutting at
    ``change + 1`` keeps the row that moved on the *pre-cut* side, which is
    where it has to be for its dependence on the destroyed future to show.

    A strategy whose signal never moves - constant exposure, or a warm-up that
    swallows the whole sample - has nothing to key on, so fall back to even
    spacing rather than refusing to run.
    """
    def valid(a):
        a = np.unique(np.asarray(a, dtype=int))
        return a[(a >= 2) & (a < n)]

    # Evenly spaced cuts are kept alongside the targeted ones even though they
    # detect far less often, because they are what measures *how far* a leak
    # reaches. A targeted cut sits one row after the signal moved, so the only
    # pre-cut row it can expose is that one, and the span it reports is always
    # 1 however far the strategy is really peeking. A cut dropped mid-period
    # leaves the whole flat stretch behind it exposed, so the span it reports
    # is a real lower bound on the horizon.
    spread = valid((np.linspace(0.3, 0.9, max(1, n_cuts // 3)) * n).astype(int))
    targeted = valid(_change_rows(signals, tolerance) + 1)

    if targeted.size == 0:
        candidates = spread if spread.size else valid([n // 2])
        return tuple(int(c) for c in candidates)

    room = max(1, n_cuts - spread.size)
    if targeted.size > room:
        # Spread the chosen cuts across the whole candidate list rather than
        # taking the first few, so a strategy that churns early and settles
        # down is still probed late in its life.
        picks = np.unique(np.linspace(0, targeted.size - 1, room).round().astype(int))
        targeted = targeted[picks]

    return tuple(int(c) for c in np.unique(np.concatenate([targeted, spread])))


@dataclass(frozen=True)
class _CutOutcome:
    """What one cut point found. Aggregated across modes and repeats."""

    cut: int
    n_changed: int
    max_absolute_change: float
    first_changed_index: int | None
    modes_that_leaked: tuple[str, ...]

    @property
    def leaked(self) -> bool:
        return bool(self.modes_that_leaked)


@dataclass(frozen=True)
class PerturbationResult:
    """Whether signals before the cut point moved when the future was destroyed."""

    leaked: bool
    cut_index: int
    n_obs: int
    max_absolute_change: float
    n_changed: int
    modes_tested: tuple[str, ...]
    modes_that_leaked: tuple[str, ...]
    first_changed_index: int | None
    tolerance: float
    #: Every cut swept, and the subset that exposed a leak. A leak found at one
    #: cut and missed at eleven others is still a leak, but the ratio says how
    #: reliably a re-run would find it again.
    cuts_tested: tuple[int, ...] = ()
    cuts_that_leaked: tuple[int, ...] = ()
    #: Median gap between genuine moves in the signal, or ``None`` if it never
    #: moved. This is the resolution of the whole test - see `detection_floor`.
    signal_update_interval: int | None = None
    #: How far back each cut's contamination reached, aligned with
    #: ``cuts_tested``; 0 where that cut found nothing. The look-ahead horizon
    #: profile is built from this, so it costs no extra strategy calls.
    reaches: tuple[int, ...] = ()

    def horizon_profile(
        self, gaps: tuple[int, ...] | None = None
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Fraction of cuts showing dependence at least ``g`` periods ahead.

        A signal at index ``i`` that illegally reads ``r`` periods forward
        depends on data up to ``i + r``. Destroying everything from ``C``
        onward therefore moves some signal at or before ``C - g`` exactly when
        ``g <= r``, so sweeping ``g`` and recording how often anything moved
        traces the horizon directly: the profile is positive up to the true
        reach and zero beyond it.

        For an honest strategy every value is zero, which is why the chart
        built from this has to carry the detection floor as well - a flat line
        at zero means nothing on its own.
        """
        if gaps is None:
            # The range has to cover the detection floor, not just the observed
            # reach. A clean run has a reach of zero, and sweeping only to 8
            # while the floor sits at 21 produced a chart whose shaded
            # "cannot tell" region fell entirely outside its own axes.
            span = max([*self.reaches, self.signal_update_interval or 0, 4])
            top = max(8, min(64, span * 2))
            gaps = tuple(range(1, top + 1))
        n = len(self.reaches)
        if n == 0:
            return tuple(gaps), tuple(0.0 for _ in gaps)
        return tuple(gaps), tuple(
            sum(1 for r in self.reaches if r >= g) / n for g in gaps
        )

    @property
    def fraction_changed(self) -> float:
        return self.n_changed / self.cut_index if self.cut_index else 0.0

    @property
    def lookahead_span(self) -> int | None:
        """How many periods before the cut the contamination reaches.

        A signal at index ``cut - 3`` that changes when data from ``cut``
        onward is destroyed is reaching about 3 periods into the future.
        """
        if self.first_changed_index is None:
            return None
        return self.cut_index - self.first_changed_index

    @property
    def detection_floor(self) -> int | None:
        """The shortest look-ahead this run could have found, in periods.

        A strategy that only revises its position monthly cannot betray a leak
        shorter than the gap between revisions: the offending signal is
        overwritten by the next rebalance before anything downstream sees it.
        So a clean verdict means "no look-ahead longer than this", not "no
        look-ahead", and reporting the number is the difference between the two.

        ``None`` when the signal never moved, in which case the test has
        established nothing at all and says so.
        """
        return self.signal_update_interval

    @property
    def clean_claim(self) -> str:
        """What a clean result is actually entitled to claim."""
        if self.leaked:
            return "Look-ahead detected."
        floor = self.detection_floor
        if floor is None:
            return (
                "Nothing established: the signal never changed, so destroying the "
                "future could not have moved it."
            )
        if floor <= 1:
            return "No dependence on data after the decision point."
        return (
            f"No dependence on data more than about {floor} periods after the "
            f"decision point. The strategy revises its position every {floor} "
            "periods on average, and a leak shorter than that is overwritten "
            "before it reaches the book."
        )

    def to_finding(self) -> Finding | None:
        if not self.leaked:
            return None
        span = self.lookahead_span
        reach = "" if span is None else f" The earliest affected signal is {span} periods before the cut, suggesting a look-ahead of about that horizon."
        reliability = (
            ""
            if len(self.cuts_tested) < 2
            else (
                f" Found at {len(self.cuts_that_leaked)} of {len(self.cuts_tested)} "
                "cut points swept."
            )
        )
        return make_finding(
            "LEAK-BEHAVIOURAL",
            detail=(
                f"Destroying data from index {self.cut_index} onward changed "
                f"{self.n_changed} of {self.cut_index} signals that precede it "
                f"(largest change {self.max_absolute_change:.3g}). Modes that exposed it: "
                f"{', '.join(self.modes_that_leaked)}.{reach}{reliability}"
            ),
            severity=Severity.CRITICAL,
            evidence={
                "cut_index": self.cut_index,
                "n_changed": self.n_changed,
                "max_absolute_change": self.max_absolute_change,
                "lookahead_span": span,
                "modes_that_leaked": list(self.modes_that_leaked),
                "cuts_tested": len(self.cuts_tested),
                "cuts_that_leaked": len(self.cuts_that_leaked),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "leaked": self.leaked,
            "cut_index": self.cut_index,
            "n_obs": self.n_obs,
            "n_changed": self.n_changed,
            "fraction_changed": self.fraction_changed,
            "max_absolute_change": self.max_absolute_change,
            "first_changed_index": self.first_changed_index,
            "lookahead_span": self.lookahead_span,
            "modes_tested": list(self.modes_tested),
            "modes_that_leaked": list(self.modes_that_leaked),
            "tolerance": self.tolerance,
            "cuts_tested": list(self.cuts_tested),
            "cuts_that_leaked": list(self.cuts_that_leaked),
            "signal_update_interval": self.signal_update_interval,
            "detection_floor": self.detection_floor,
            "clean_claim": self.clean_claim,
            "reaches": list(self.reaches),
        }


def _probe_cut(
    strategy: Callable[[np.ndarray], np.ndarray],
    array: np.ndarray,
    baseline: np.ndarray,
    cut: int,
    modes: tuple[str, ...],
    tolerance: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> _CutOutcome:
    """Corrupt from ``cut`` onward every way asked for, and compare the past."""
    n, width = baseline.shape
    max_change, n_changed, first_changed = 0.0, 0, None
    leaked_modes: list[str] = []

    for mode in modes:
        # "constant" and "reverse" ignore the generator, so repeating them
        # would just re-run identical work.
        repeats = n_repeats if mode in ("shuffle", "noise") else 1
        for _ in range(repeats):
            corrupted = _as_signals(strategy(_corrupt(array, cut, mode, rng)), n)
            if corrupted.shape[1] != width:
                raise ValueError(
                    f"strategy returned {corrupted.shape[1]} signal columns under "
                    f"corruption mode {mode!r} but {width} on clean data; its output "
                    "shape must not depend on the values it is given"
                )

            before = slice(0, cut)
            # NaN in the same place in both is agreement, not a change. Every
            # operation here is elementwise, so a (cut, k) book reads exactly
            # as a (cut, 1) column does.
            both_nan = np.isnan(baseline[before]) & np.isnan(corrupted[before])
            diff = np.abs(baseline[before] - corrupted[before])
            diff = np.where(both_nan, 0.0, np.nan_to_num(diff, nan=np.inf))

            changed = diff > tolerance
            if changed.any():
                if mode not in leaked_modes:
                    leaked_modes.append(mode)
                # Rows, not cells. One contaminated weight makes the whole book
                # at that timestamp wrong, and counting rows is what keeps
                # `fraction_changed` a fraction. For a one-column signal the
                # two counts are identical, so nothing about a 1-D result moves.
                changed_rows = changed.any(axis=1)
                n_changed = max(n_changed, int(changed_rows.sum()))
                finite = diff[np.isfinite(diff)]
                if finite.size:
                    max_change = max(max_change, float(finite.max()))
                # Reduce to rows *before* taking argmax. On a 2-D array argmax
                # returns a flattened index, which would make `lookahead_span`
                # a multiple of the book width rather than a number of periods.
                idx = int(np.argmax(changed_rows))
                first_changed = idx if first_changed is None else min(first_changed, idx)

    return _CutOutcome(
        cut=cut,
        n_changed=n_changed,
        max_absolute_change=max_change,
        first_changed_index=first_changed,
        modes_that_leaked=tuple(leaked_modes),
    )


def perturbation_test(
    strategy: Callable[[np.ndarray], np.ndarray],
    data,
    cut_fraction: float | None = None,
    modes: tuple[str, ...] = CORRUPTION_MODES,
    tolerance: float = 1e-10,
    n_repeats: int = 4,
    seed: int | None = 0,
    n_cuts: int = DEFAULT_N_CUTS,
) -> PerturbationResult:
    """Does destroying the future change the past?

    ``strategy`` takes the full data array and returns one signal per row -
    either ``n`` signals, or an ``(n, k)`` book of ``k`` weights per row for a
    cross-sectional strategy. It is called once on clean data and once per
    corruption draw per cut; only the signals *strictly before* each cut are
    compared, since signals at and after it are expected to change.

    The result is the union across cuts, modes and repeats: a leak found by any
    draw is a leak. Every dimension matters, because a single draw can miss a
    real leak by luck. A signal that takes the sign of tomorrow's return has a
    coin flip's chance of surviving any one random corruption unchanged - the
    corrupted value simply happens to have the same sign. Four repeats across
    four modes reduces that to a rounding error, and the deterministic modes
    (``constant``, ``reverse``) catch what the random ones miss.

    By default the cut points are chosen from the strategy's own behaviour, one
    row after each time its signal moves; pass ``cut_fraction`` to pin a single
    cut instead. Sweeping is the default because one fixed cut is close to
    blind on any strategy that rebalances less often than every period - see
    the module docstring for the measured detection rates.
    """
    array = np.asarray(getattr(data, "to_numpy", lambda: data)(), dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"data must be 1-D or 2-D, got shape {array.shape}")

    n = array.shape[0]
    if n < 20:
        raise ValueError(f"perturbation test needs at least 20 observations, got {n}")
    if not modes:
        raise ValueError("at least one corruption mode is required")
    for mode in modes:
        if mode not in CORRUPTION_MODES:
            raise ValueError(f"unknown corruption mode {mode!r}; expected one of {CORRUPTION_MODES}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")
    if n_cuts < 1:
        raise ValueError(f"n_cuts must be at least 1, got {n_cuts}")

    baseline = _as_signals(strategy(array), n)

    if cut_fraction is not None:
        if not 0.0 < cut_fraction < 1.0:
            raise ValueError(
                f"cut_fraction must be strictly between 0 and 1, got {cut_fraction}"
            )
        cut = int(n * cut_fraction)
        if cut < 2 or cut >= n:
            raise ValueError(
                f"cut_fraction {cut_fraction} gives an unusable cut index of {cut}"
            )
        cuts: tuple[int, ...] = (cut,)
    else:
        cuts = _cut_points(baseline, n, n_cuts, tolerance)

    changes = _change_rows(baseline, tolerance)
    update_interval = (
        int(round(float(np.median(np.diff(changes))))) if changes.size >= 2 else None
    )

    rng = np.random.default_rng(seed)
    outcomes = [
        _probe_cut(strategy, array, baseline, cut, modes, tolerance, n_repeats, rng)
        for cut in cuts
    ]

    leaked = [o for o in outcomes if o.leaked]
    # Report whichever cut reached furthest back, so the headline numbers all
    # describe one coherent probe rather than a maximum taken over incompatible
    # ones. Furthest back rather than most rows changed, because the span is
    # the diagnostic that locates the offending line, and the largest span
    # observed is the best lower bound on the true horizon. With nothing found,
    # the latest cut has the most pre-cut history behind it.
    reported = (
        max(leaked, key=lambda o: (o.cut - (o.first_changed_index or 0), o.n_changed))
        if leaked
        else outcomes[-1]
    )

    modes_that_leaked = tuple(
        mode for mode in modes if any(mode in o.modes_that_leaked for o in outcomes)
    )

    return PerturbationResult(
        leaked=bool(leaked),
        cut_index=reported.cut,
        n_obs=n,
        max_absolute_change=max(o.max_absolute_change for o in outcomes),
        n_changed=reported.n_changed,
        modes_tested=tuple(modes),
        modes_that_leaked=modes_that_leaked,
        first_changed_index=reported.first_changed_index,
        tolerance=tolerance,
        cuts_tested=tuple(o.cut for o in outcomes),
        cuts_that_leaked=tuple(o.cut for o in leaked),
        signal_update_interval=update_interval,
        reaches=tuple(
            0 if o.first_changed_index is None else o.cut - o.first_changed_index
            for o in outcomes
        ),
    )
