"""Point-in-time index membership: measuring survivorship instead of declaring it.

`DATA-SURVIVORSHIP` used to fire on a declaration alone. The researcher wrote
`universe_point_in_time: false` in the manifest and the finding appeared, always
with the same sentence, always at HIGH, always saying the problem could not be
quantified. It therefore fired on almost every audit this tool has ever run - and
a finding that fires on everybody and admits it cannot be measured teaches people
to scroll past it.

Given a point-in-time membership list, the question becomes measurable. The list
says who was in the index and when; the traded universe says who the backtest
could actually choose from; the difference is the hole.

**What this measures is extent, not magnitude.** Free membership lists carry no
prices for delisted names, so the tool can count how many names were excluded and
never what their returns would have done. It cannot bound the Sharpe
overstatement, and nothing here may be phrased as though it could. That ceiling
is why `exit_miss_rate` never escalates a finding past HIGH.

Pure functions over frames. No I/O - the reader lives in `qv.data.loaders`, which
is the one module allowed to touch the outside world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from qv.findings import make_finding
from qv.types import Finding, Severity

__all__ = [
    "MEMBERSHIP_COLUMNS",
    "Membership",
    "SurvivorshipMeasurement",
    "measure_survivorship",
]

#: The documented schema. `ticker` and `start_date` are required; a blank
#: `end_date` means "still a member"; `id` is a permanent identifier that
#: survives a ticker change, and `index` lets one file carry several universes.
#:
#: This is deliberately the exact shape of the most widely used free file,
#: `sp500_ticker_start_end.csv` from fja05680/sp500, so that the common case
#: needs no conversion at all.
MEMBERSHIP_COLUMNS = ("ticker", "start_date", "end_date", "id", "index")


@dataclass(frozen=True)
class Membership:
    """Parsed membership spells, plus where they came from and what they span.

    One row per *spell*, so a name that left and rejoined appears more than
    once - `AAL` is in the S&P 500 from 1996 to 1997 and again from 2015. Any
    model keyed on one interval per ticker silently loses the gap, which is
    precisely the period a backtest must not trade it.
    """

    frame: pd.DataFrame
    source: str
    index_name: str | None = None

    @property
    def file_start(self) -> pd.Timestamp:
        return self.frame["start_date"].min()

    @property
    def file_end(self) -> pd.Timestamp:
        """Last date the file says anything about.

        Open spells (a blank ``end_date``) mean "still a member", so the file
        speaks to the present and the end of the sample is the binding limit.
        """
        if self.frame["end_date"].isna().any():
            return pd.Timestamp.max
        return self.frame["end_date"].max()

    @property
    def has_identifier(self) -> bool:
        return "id" in self.frame.columns and self.frame["id"].notna().any()

    @property
    def n_spells_with_exit(self) -> int:
        """Spells that ever end. Zero is the signature of a snapshot file."""
        return int(self.frame["end_date"].notna().sum())

    @property
    def noncontiguous_tickers(self) -> tuple[str, ...]:
        """Tickers holding more than one spell.

        Legitimate for a name that rejoined, and also what a symbol reused by a
        different company looks like. Without an `id` column the two are
        indistinguishable, which is a limit to report rather than a problem to
        solve.
        """
        counts = self.frame.groupby("ticker").size()
        return tuple(sorted(counts[counts > 1].index.astype(str)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "index_name": self.index_name,
            "n_spells": int(len(self.frame)),
            "n_tickers": int(self.frame["ticker"].nunique()),
            "n_spells_with_exit": self.n_spells_with_exit,
            "has_identifier": self.has_identifier,
            "file_start": str(self.file_start.date()),
            "noncontiguous_tickers": list(self.noncontiguous_tickers),
        }


@dataclass(frozen=True)
class SurvivorshipMeasurement:
    """The hole in a traded universe, measured against who was really a member.

    The three sets partition the names cleanly, and that is the property which
    keeps *one question, one test* intact:

    - ``missing_exited`` and ``missing_still_member`` are ``roster - universe``:
      names that were members and could not be traded. That is survivorship.
    - ``traded_and_member`` is the intersection, and belongs to the inclusion
      timing test, which asks a different question about the names you *do*
      hold.
    - ``traded_not_member`` is ``universe - roster``: traded but never a member.
      Evidence only; choosing a subset or a superset of an index is a
      legitimate decision, not a defect.
    """

    covered_start: pd.Timestamp
    covered_end: pd.Timestamp
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp
    roster: tuple[str, ...]
    universe: tuple[str, ...]
    missing_exited: tuple[str, ...]
    missing_still_member: tuple[str, ...]
    traded_not_member: tuple[str, ...]
    exits_total: int
    shortest_spell_days: int | None
    has_identifier: bool
    noncontiguous_tickers: tuple[str, ...]
    n_spells_with_exit: int

    @property
    def covered_fraction_of_sample(self) -> float:
        """How much of the audited window the membership file speaks to.

        Below 1.0 every count here is a statement about a sub-period, and the
        report has to say which. A file covering 2015 onward against a backtest
        starting 2005 would otherwise report "no names missing" for the first
        decade, which is a false clean in the dangerous direction.
        """
        span = (self.sample_end - self.sample_start).days
        if span <= 0:
            return 0.0
        covered = (self.covered_end - self.covered_start).days
        return max(0.0, min(1.0, covered / span))

    @property
    def traded_and_member(self) -> tuple[str, ...]:
        members = set(self.roster)
        return tuple(name for name in self.universe if name in members)

    @property
    def overlap_fraction(self) -> float:
        if not self.universe:
            return 0.0
        return len(self.traded_and_member) / len(self.universe)

    @property
    def exit_miss_rate(self) -> float | None:
        """Of the names that left the index, what fraction could not be traded?

        Deliberately a ratio against a denominator the file itself supplies,
        rather than a fraction of the cross-section measured against an invented
        threshold. Both endpoints mean something on their own: 0 says the
        universe contained every name that ever left, and 1 says it contained
        none of them, which is the textbook survivorship signature.

        ``None`` when nothing exited, because the ratio is then undefined - and
        an undefined ratio must never render as 0.
        """
        if self.exits_total == 0:
            return None
        return len(self.missing_exited) / self.exits_total

    @property
    def looks_like_a_snapshot(self) -> bool:
        """Is this file really a list of *current* members wearing a history?

        The "date added" column on a current-membership table reaches back
        decades and looks like point-in-time data, but it contains only the
        companies that are still in the index - it *is* the survivorship bias,
        in a history-shaped schema. Fed in naively it reports zero names
        missing.

        The decisive signature is that no spell ever ends. A real reconstruction
        has departures in it; a snapshot cannot, because everything in it is
        still a member.

        A start-date pile-up was considered as a second detector and rejected:
        every genuine reconstruction piles up at its own origin, since every
        name already in the index on day one shares that start date. It would
        have fired on the real files this feature exists to consume.
        """
        return self.n_spells_with_exit == 0

    @property
    def clean(self) -> bool:
        """No member of the index was absent from the traded universe.

        Only meaningful when the measurement is usable at all - a snapshot file
        produces `clean` for the wrong reason, which is why the caller checks
        `looks_like_a_snapshot` first.
        """
        return not self.missing_exited and not self.missing_still_member

    def to_findings(self) -> list[Finding]:
        findings: list[Finding] = []

        if self.looks_like_a_snapshot:
            findings.append(
                make_finding(
                    "DATA-MEMBERSHIP-NOT-POINT-IN-TIME",
                    detail=(
                        f"The membership list supplied has {len(self.roster)} names and "
                        "not one of them ever leaves the index. A real point-in-time "
                        "reconstruction contains departures; a list of today's members "
                        "with the date each was added cannot, because everything in it "
                        "survived. Survivorship was therefore not measured - a list "
                        "like this reports zero names missing no matter how much is "
                        "missing."
                    ),
                    evidence={
                        "n_spells_with_exit": self.n_spells_with_exit,
                        "roster_size": len(self.roster),
                    },
                )
            )
            return findings

        if not self.missing_exited:
            return findings

        rate = self.exit_miss_rate
        window = (
            ""
            if self.covered_fraction_of_sample >= 0.999
            else (
                f" Measured over {self.covered_start.date()} to "
                f"{self.covered_end.date()}, which is "
                f"{self.covered_fraction_of_sample:.0%} of the audited sample."
            )
        )
        # Name the names. A count tells a reader there is a problem; the tickers
        # let them go and look, and recognising one they know is what turns the
        # finding from a statistic into something they act on. Truncated here,
        # complete in `report.json`.
        shown = ", ".join(self.missing_exited[:8])
        if len(self.missing_exited) > 8:
            shown += f", and {len(self.missing_exited) - 8} more"
        categorical = (
            f"every name that left the index during the window is absent from it "
            f"({shown})"
            if rate is not None and rate >= 1.0
            else (
                f"{len(self.missing_exited)} of the {self.exits_total} names that "
                f"left during the window are absent from it ({shown})"
            )
        )
        findings.append(
            make_finding(
                "DATA-SURVIVORSHIP",
                detail=(
                    f"Measured against the membership list: {categorical}. The backtest "
                    "could not have chosen them, and they are the names most likely to "
                    "have done badly - that is what leaving the index usually means. "
                    "This counts how many names were excluded, not what their returns "
                    "would have been, so it bounds the extent of the bias and not its "
                    "size." + window
                ),
                evidence={
                    "missing_exited": list(self.missing_exited[:20]),
                    "n_missing_exited": len(self.missing_exited),
                    "exits_total": self.exits_total,
                    "exit_miss_rate": rate,
                    "covered_fraction_of_sample": self.covered_fraction_of_sample,
                },
            )
        )
        return findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "covered_start": str(self.covered_start.date()),
            "covered_end": str(self.covered_end.date()),
            "covered_fraction_of_sample": self.covered_fraction_of_sample,
            "roster_size": len(self.roster),
            "universe_size": len(self.universe),
            "overlap_fraction": self.overlap_fraction,
            "missing_exited": list(self.missing_exited),
            "n_missing_exited": len(self.missing_exited),
            "missing_still_member": list(self.missing_still_member),
            "n_missing_still_member": len(self.missing_still_member),
            "traded_not_member": list(self.traded_not_member),
            "exits_total": self.exits_total,
            "exit_miss_rate": self.exit_miss_rate,
            "looks_like_a_snapshot": self.looks_like_a_snapshot,
            "shortest_spell_days": self.shortest_spell_days,
            "has_identifier": self.has_identifier,
            "noncontiguous_tickers": list(self.noncontiguous_tickers),
        }


def measure_survivorship(
    membership: Membership,
    universe: list[str] | tuple[str, ...],
    sample_start: Any,
    sample_end: Any,
) -> SurvivorshipMeasurement:
    """Compare who was a member against who the backtest could trade.

    ``sample_start`` and ``sample_end`` must be the window the *report*
    describes, after any trimming, or the measurement answers a question about
    a period the reader is never shown.

    Raises ``ValueError`` rather than returning an empty result when the file
    cannot support any claim at all - a refusal and a clean bill of health must
    never look the same from outside.
    """
    sample_start = pd.Timestamp(sample_start)
    sample_end = pd.Timestamp(sample_end)
    if sample_end <= sample_start:
        raise ValueError(
            f"the audited window is empty ({sample_start.date()}..{sample_end.date()})"
        )

    frame = membership.frame
    covered_start = max(sample_start, membership.file_start)
    covered_end = min(sample_end, membership.file_end)
    if covered_end <= covered_start:
        raise ValueError(
            f"the membership list covers {membership.file_start.date()} onward, which "
            f"does not overlap the audited window "
            f"{sample_start.date()}..{sample_end.date()}; supply a list spanning the "
            "backtest, or drop the input rather than measuring nothing"
        )

    # A spell counts if it overlaps the covered window at all. An open spell
    # (blank end_date) runs to the present.
    ends = frame["end_date"].fillna(pd.Timestamp.max)
    overlaps = (frame["start_date"] <= covered_end) & (ends >= covered_start)
    live = frame[overlaps]
    if live.empty:
        raise ValueError(
            "no membership spell overlaps the audited window, so the list describes "
            "a different period than the backtest"
        )

    roster = tuple(sorted(live["ticker"].astype(str).unique()))
    universe = tuple(str(name) for name in universe)
    traded = set(universe)

    # Who was still a member when the window closed, and who had gone.
    at_end = live[(live["start_date"] <= covered_end) & (ends[live.index] >= covered_end)]
    still_member = set(at_end["ticker"].astype(str))

    exited = tuple(name for name in roster if name not in still_member)
    missing = [name for name in roster if name not in traded]
    missing_exited = tuple(name for name in missing if name not in still_member)
    missing_still_member = tuple(name for name in missing if name in still_member)
    traded_not_member = tuple(name for name in universe if name not in set(roster))

    if not set(roster) & traded:
        raise ValueError(
            f"none of the {len(universe)} traded instruments appear in the membership "
            f"list at all, so it describes a different universe; check "
            "`data.membership_index` and the ticker spellings"
        )

    spans = (ends - frame["start_date"]).dt.days
    shortest = int(spans.min()) if len(spans) else None

    return SurvivorshipMeasurement(
        covered_start=covered_start,
        covered_end=covered_end,
        sample_start=sample_start,
        sample_end=sample_end,
        roster=roster,
        universe=universe,
        missing_exited=missing_exited,
        missing_still_member=missing_still_member,
        traded_not_member=traded_not_member,
        exits_total=len(exited),
        shortest_spell_days=shortest,
        has_identifier=membership.has_identifier,
        noncontiguous_tickers=membership.noncontiguous_tickers,
        n_spells_with_exit=membership.n_spells_with_exit,
    )
