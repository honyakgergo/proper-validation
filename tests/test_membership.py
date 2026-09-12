"""Tests for point-in-time membership and the measured survivorship finding.

The danger this feature carries is not a false alarm but a **false clean**: a
membership list that is malformed, too short, or secretly a snapshot of today's
members will happily report "no names missing" and look exactly like a careful
universe. So nearly every test here is paired - one case that must report a
problem, and one that must not - and the refusals are asserted to produce *no*
finding in either direction rather than a quiet pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qv.data.loaders import read_membership_frame
from qv.data.membership import measure_survivorship
from qv.types import Suite

WINDOW = ("2012-01-02", "2020-12-31")

#: Four survivors, two names that left during the window, one that joined late.
#: `AAL` holds two disjoint spells, which is what a re-entry looks like.
SPELLS = [
    ("AAA", "2012-01-02", ""),
    ("BBB", "2012-01-02", ""),
    ("CCC", "2012-01-02", ""),
    ("DDD", "2012-01-02", ""),
    ("DEAD1", "2012-01-02", "2015-06-30"),
    ("DEAD2", "2012-01-02", "2018-02-14"),
    ("LATE", "2017-03-01", ""),
    ("AAL", "2012-01-02", "2013-01-15"),
    ("AAL", "2016-01-02", ""),
]
SURVIVORS = ["AAA", "BBB", "CCC", "DDD", "LATE", "AAL"]
COMPLETE = SURVIVORS + ["DEAD1", "DEAD2"]


def write(tmp_path, rows, name="membership.csv", columns=None):
    columns = columns or ["ticker", "start_date", "end_date"]
    path = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def measurement(tmp_path, rows=None, universe=None, window=WINDOW, **kwargs):
    path = write(tmp_path, rows if rows is not None else SPELLS)
    membership = read_membership_frame(path, kwargs.pop("index_name", None))
    return measure_survivorship(
        membership, universe if universe is not None else SURVIVORS, *window
    )


class TestTheControlPair:
    """One generator, one deletion. The test that catches the whole feature
    silently doing nothing."""

    def test_a_survivor_only_universe_is_caught(self, tmp_path):
        result = measurement(tmp_path, universe=SURVIVORS)
        assert set(result.missing_exited) == {"DEAD1", "DEAD2"}
        assert result.exit_miss_rate == 1.0
        assert [f.id for f in result.to_findings()] == ["DATA-SURVIVORSHIP"]

    def test_the_complete_universe_is_not(self, tmp_path):
        """The paired must-pass. Without it, a feature that never fires looks
        identical to a feature that works."""
        result = measurement(tmp_path, universe=COMPLETE)
        assert result.missing_exited == ()
        assert result.exit_miss_rate == 0.0
        assert result.to_findings() == []

    def test_a_name_that_rejoined_is_not_counted_as_gone(self, tmp_path):
        """`AAL` left in 2013 and came back in 2016. It is a member at the end
        of the window, so it never exited."""
        result = measurement(tmp_path, universe=COMPLETE)
        assert "AAL" not in result.missing_exited
        assert "AAL" in result.roster


class TestSetPartition:
    """Every name lands in exactly one of the three cells, so the survivorship
    finding and the inclusion-timing finding can never argue over one ticker."""

    def test_the_three_sets_partition_the_names(self, tmp_path):
        result = measurement(tmp_path, universe=SURVIVORS + ["NOTAMEMBER"])
        missing = set(result.missing_exited) | set(result.missing_still_member)
        both = set(result.traded_and_member)
        outside = set(result.traded_not_member)

        assert missing | both == set(result.roster)
        assert missing & both == set()
        assert outside == {"NOTAMEMBER"}
        assert outside & set(result.roster) == set()

    def test_traded_but_never_a_member_is_evidence_not_a_defect(self, tmp_path):
        """Trading a subset or a superset of an index is a choice, not a bug."""
        result = measurement(tmp_path, universe=COMPLETE + ["NOTAMEMBER"])
        assert result.traded_not_member == ("NOTAMEMBER",)
        assert result.to_findings() == []


class TestPartialWindow:
    """A list that starts after the sample does must never report a whole-sample
    result. NASDAQ-100 history begins in 2015 and S&P 500 in 1996, so this is
    the common case rather than an edge case."""

    LATE_FILE = [
        ("AAA", "2016-01-04", ""),
        ("BBB", "2016-01-04", ""),
        ("DEAD2", "2016-01-04", "2018-02-14"),
    ]

    def test_a_gap_is_reported_as_a_fraction_not_hidden(self, tmp_path):
        result = measurement(
            tmp_path, rows=self.LATE_FILE, universe=["AAA", "BBB"]
        )
        assert result.covered_fraction_of_sample < 0.999
        assert result.covered_start == pd.Timestamp("2016-01-04")

    def test_deletions_inside_the_covered_part_still_fire(self, tmp_path):
        """The paired must-fail: degrading must not mean going blind."""
        result = measurement(
            tmp_path, rows=self.LATE_FILE, universe=["AAA", "BBB"]
        )
        assert result.missing_exited == ("DEAD2",)
        detail = result.to_findings()[0].detail
        assert "%" in detail, "a partial measurement must state its coverage"

    def test_the_finding_never_claims_the_whole_sample(self, tmp_path):
        result = measurement(
            tmp_path, rows=self.LATE_FILE, universe=["AAA", "BBB"]
        )
        assert "2016-01-04" in result.to_findings()[0].detail

    def test_no_overlap_at_all_refuses(self, tmp_path):
        """Refuse, rather than measure nothing and call it clean."""
        rows = [("AAA", "2022-01-03", ""), ("BBB", "2022-01-03", "2023-01-03")]
        with pytest.raises(ValueError, match="does not overlap"):
            measurement(tmp_path, rows=rows, universe=["AAA"])

    def test_a_disjoint_universe_refuses(self, tmp_path):
        with pytest.raises(ValueError, match="different universe"):
            measurement(tmp_path, universe=["ZZZ", "YYY"])


class TestSnapshotDetector:
    """A table of today's members with the date each was added reaches back
    decades and *is* the survivorship bias. It must not read as a clean bill."""

    SNAPSHOT = [
        ("AAA", "1996-01-02", ""),
        ("BBB", "2004-05-10", ""),
        ("CCC", "2011-09-01", ""),
    ]

    def test_a_list_with_no_departures_is_flagged(self, tmp_path):
        result = measurement(
            tmp_path, rows=self.SNAPSHOT, universe=["AAA", "BBB", "CCC"]
        )
        assert result.looks_like_a_snapshot
        assert [f.id for f in result.to_findings()] == [
            "DATA-MEMBERSHIP-NOT-POINT-IN-TIME"
        ]

    def test_it_suppresses_the_survivorship_result_rather_than_passing_it(
        self, tmp_path
    ):
        """`clean` is true here for the wrong reason, so no survivorship finding
        may be emitted from it - in either direction."""
        result = measurement(
            tmp_path, rows=self.SNAPSHOT, universe=["AAA", "BBB", "CCC"]
        )
        assert result.clean
        assert "DATA-SURVIVORSHIP" not in {f.id for f in result.to_findings()}

    def test_one_real_departure_is_enough_to_measure(self, tmp_path):
        """The paired must-pass. A list that records even one removal is a
        reconstruction, not a snapshot."""
        rows = self.SNAPSHOT + [("DEAD1", "1996-01-02", "2015-06-30")]
        result = measurement(tmp_path, rows=rows, universe=["AAA", "BBB", "CCC"])
        assert not result.looks_like_a_snapshot
        assert [f.id for f in result.to_findings()] == ["DATA-SURVIVORSHIP"]

    def test_a_start_date_pile_up_is_not_treated_as_a_snapshot(self, tmp_path):
        """Every genuine reconstruction piles up at its own origin: each name
        already in the index on day one shares that start date. Firing on that
        would reject the real files this feature exists to consume."""
        rows = [
            ("AAA", "2012-01-02", ""),
            ("BBB", "2012-01-02", ""),
            ("CCC", "2012-01-02", ""),
            ("DEAD1", "2012-01-02", "2015-06-30"),
        ]
        result = measurement(tmp_path, rows=rows, universe=["AAA", "BBB", "CCC"])
        assert not result.looks_like_a_snapshot


class TestExitMissRate:
    def test_undefined_when_nothing_ever_left(self, tmp_path):
        """An undefined ratio must never render as 0 - that is a clean result
        the data did not earn."""
        rows = [("AAA", "2012-01-02", ""), ("BBB", "2012-01-02", "")]
        result = measurement(tmp_path, rows=rows, universe=["AAA", "BBB"])
        assert result.exit_miss_rate is None

    def test_one_when_no_departing_name_was_traded(self, tmp_path):
        assert measurement(tmp_path, universe=SURVIVORS).exit_miss_rate == 1.0

    def test_between_the_endpoints_when_some_were(self, tmp_path):
        result = measurement(tmp_path, universe=SURVIVORS + ["DEAD1"])
        assert result.exit_miss_rate == 0.5

    def test_names_still_in_the_index_do_not_drive_the_finding(self, tmp_path):
        """Trading the top 100 of a 500-name index is incompleteness, not
        survivorship. It must be recorded and must not fire."""
        result = measurement(tmp_path, universe=["AAA", "DEAD1", "DEAD2", "AAL"])
        assert set(result.missing_still_member) >= {"BBB", "CCC", "DDD"}
        assert result.missing_exited == ()
        assert result.to_findings() == []

    def test_severity_is_capped_at_high(self, tmp_path):
        """With no prices for the dead names the tool measures extent, not
        magnitude - and extent alone cannot earn a CRITICAL."""
        from qv.types import Severity

        findings = measurement(tmp_path, universe=SURVIVORS).to_findings()
        assert all(f.severity <= Severity.HIGH for f in findings)


class TestReaderRefusals:
    def test_a_missing_required_column(self, tmp_path):
        path = write(tmp_path, [("AAA", "")], columns=["ticker", "end_date"])
        with pytest.raises(ValueError, match="missing required column"):
            read_membership_frame(path)

    def test_an_unparsable_start_date(self, tmp_path):
        path = write(tmp_path, [("AAA", "not-a-date", "")])
        with pytest.raises(ValueError, match="unparsable start_date"):
            read_membership_frame(path)

    def test_a_spell_that_ends_before_it_starts(self, tmp_path):
        path = write(tmp_path, [("AAA", "2015-01-02", "2013-01-02")])
        with pytest.raises(ValueError, match="ending on or before"):
            read_membership_frame(path)

    def test_overlapping_spells_for_one_ticker(self, tmp_path):
        """A ticker cannot be two members of one index at once - the file is
        malformed, not merely ambiguous."""
        rows = [("AAA", "2012-01-02", "2016-01-02"), ("AAA", "2014-01-02", "")]
        path = write(tmp_path, rows)
        with pytest.raises(ValueError, match="overlapping membership spells"):
            read_membership_frame(path)

    def test_disjoint_repeats_are_fine(self, tmp_path):
        """The paired must-pass: names really do leave and rejoin."""
        rows = [("AAA", "2012-01-02", "2013-01-02"), ("AAA", "2016-01-02", "")]
        membership = read_membership_frame(write(tmp_path, rows))
        assert membership.noncontiguous_tickers == ("AAA",)

    def test_a_multi_index_file_refuses_to_pick_for_you(self, tmp_path):
        rows = [
            ("AAA", "2012-01-02", "", "SP500"),
            ("BBB", "2012-01-02", "", "NDX"),
        ]
        path = write(
            tmp_path, rows, columns=["ticker", "start_date", "end_date", "index"]
        )
        with pytest.raises(ValueError, match="more than one index"):
            read_membership_frame(path)

    def test_naming_the_index_selects_it(self, tmp_path):
        rows = [
            ("AAA", "2012-01-02", "", "SP500"),
            ("BBB", "2012-01-02", "", "NDX"),
        ]
        path = write(
            tmp_path, rows, columns=["ticker", "start_date", "end_date", "index"]
        )
        membership = read_membership_frame(path, "NDX")
        assert list(membership.frame["ticker"]) == ["BBB"]

    def test_an_unknown_index_name(self, tmp_path):
        rows = [("AAA", "2012-01-02", "", "SP500")]
        path = write(
            tmp_path, rows, columns=["ticker", "start_date", "end_date", "index"]
        )
        with pytest.raises(ValueError, match="no rows for index"):
            read_membership_frame(path, "NOPE")

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such membership file"):
            read_membership_frame(tmp_path / "absent.csv")


class TestInvariants:
    def test_row_order_changes_nothing(self, tmp_path):
        straight = measurement(tmp_path, rows=SPELLS, universe=SURVIVORS)
        shuffled_rows = list(SPELLS)
        np.random.default_rng(0).shuffle(shuffled_rows)
        shuffled = measurement(tmp_path, rows=shuffled_rows, universe=SURVIVORS)
        assert straight.to_dict() == shuffled.to_dict()

    def test_the_blank_end_date_means_still_a_member(self, tmp_path):
        rows = [("AAA", "2012-01-02", ""), ("BBB", "2012-01-02", "2015-01-02")]
        result = measurement(tmp_path, rows=rows, universe=["AAA", "BBB"])
        assert result.exits_total == 1

    def test_ticker_case_and_whitespace_are_normalised(self, tmp_path):
        rows = [(" aaa ", "2012-01-02", ""), ("BBB", "2012-01-02", "2015-01-02")]
        result = measurement(tmp_path, rows=rows, universe=["AAA", "BBB"])
        assert "AAA" in result.roster

    def test_the_dict_is_json_shaped(self, tmp_path):
        import json

        json.dumps(measurement(tmp_path, universe=SURVIVORS).to_dict())


class TestCoverageReclassification:
    """A membership list explains ragged edges; it must not redefine them.

    The danger is the opposite of the usual one: here a *bad* list removes a
    finding that was correctly there. So the raw raggedness is asserted to be
    unchanged in every case, and the explained set is what moves.
    """

    @staticmethod
    def _prices(tmp_path, late_start=400):
        dates = pd.bdate_range("2012-01-02", periods=800)
        frame = pd.DataFrame(
            {
                "AAA": np.linspace(100, 130, 800),
                "LATE": np.concatenate(
                    [np.full(late_start, np.nan), np.linspace(50, 70, 800 - late_start)]
                ),
            },
            index=dates,
        )
        return frame

    def test_a_late_entrant_matching_its_join_date_is_explained(self, tmp_path):
        from qv.engine import universe_coverage

        prices = self._prices(tmp_path)
        joined = prices.index[400].date()
        rows = [("AAA", "2012-01-02", ""), ("LATE", str(joined), "")]
        membership = read_membership_frame(write(tmp_path, rows))

        coverage = universe_coverage(prices, membership=membership)
        assert coverage.late_entrants == ("LATE",)
        assert coverage.explained_late == ("LATE",)
        assert coverage.unexplained_late == ()
        assert coverage.complete, "an explained edge is not a defect"

    def test_a_late_entrant_that_was_already_a_member_stays_flagged(self, tmp_path):
        """The paired must-fail. If membership says the name was a member from
        the start, a late price history is a hole in the data, not the index."""
        from qv.engine import universe_coverage

        prices = self._prices(tmp_path)
        rows = [("AAA", "2012-01-02", ""), ("LATE", "2012-01-02", "")]
        membership = read_membership_frame(write(tmp_path, rows))

        coverage = universe_coverage(prices, membership=membership)
        assert coverage.unexplained_late == ("LATE",)
        assert not coverage.complete
        assert coverage.to_finding() is not None

    def test_the_raw_raggedness_is_unchanged_either_way(self, tmp_path):
        """`ragged_total` is what lets a reader see a list muted something."""
        from qv.engine import universe_coverage

        prices = self._prices(tmp_path)
        joined = prices.index[400].date()

        explaining = read_membership_frame(
            write(tmp_path, [("AAA", "2012-01-02", ""), ("LATE", str(joined), "")])
        )
        flagging = read_membership_frame(
            write(tmp_path, [("AAA", "2012-01-02", ""), ("LATE", "2012-01-02", "")])
        )
        bare = universe_coverage(prices).to_dict()
        a = universe_coverage(prices, membership=explaining).to_dict()
        b = universe_coverage(prices, membership=flagging).to_dict()

        assert bare["ragged_total"] == a["ragged_total"] == b["ragged_total"] == 1
        assert a["ragged_explained_by_membership"] == 1
        assert b["ragged_explained_by_membership"] == 0

    def test_price_history_predating_membership_is_a_sharper_signal(self, tmp_path):
        """Data reaching back before the name was a member means the frame could
        have traded it while the index had not yet included it."""
        from qv.engine import universe_coverage

        prices = self._prices(tmp_path, late_start=0)
        rows = [("AAA", "2012-01-02", ""), ("LATE", "2016-01-04", "")]
        membership = read_membership_frame(write(tmp_path, rows))

        coverage = universe_coverage(prices, membership=membership)
        assert coverage.traded_before_membership == ("LATE",)
        assert not coverage.complete

    def test_without_membership_nothing_is_explained(self, tmp_path):
        from qv.engine import universe_coverage

        coverage = universe_coverage(self._prices(tmp_path))
        assert coverage.explained_late == ()
        assert coverage.unexplained_late == coverage.late_entrants


# --------------------------------------------------------------------------
# Through the manifest, which is the path an actual audit takes.
# --------------------------------------------------------------------------

PIPE_UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "DEAD1"]

_MANIFEST = """name: membership fixture
data:
  universe: [{universe}]
  start_date: "2015-01-02"
  end_date: "2018-06-29"
  price_frame: prices.csv
{membership}  asset_class: us_large_cap_equity
  frequency: daily
  load_factors: false
  universe_point_in_time: {declared}
strategy:
  adapter: adapter.py:positions
  fixed: {{}}
chosen_parameters:
  lookback: 60
  top_n: 2
search:
  axes:
    lookback: [40, 60]
  n_trials: null
process:
  positions_shifted: true
"""

_ADAPTER = """import pandas as pd


def positions(prices: pd.DataFrame, **params) -> pd.DataFrame:
    lookback = params.get("lookback", 60)
    top_n = params.get("top_n", 2)
    score = prices / prices.shift(lookback) - 1.0
    ranks = score.rank(axis=1, ascending=False, na_option="bottom")
    book = (ranks <= top_n).astype(float) / top_n
    book[score.isna().all(axis=1)] = 0.0
    return book.shift(1).fillna(0.0)
"""


def _project(tmp_path, universe=None, membership_rows=None, declared="null"):
    """A minimal researcher project: prices, an adapter, and a manifest."""
    from qv.pipeline import audit_from_manifest

    gen = np.random.default_rng(11)
    n = 900
    dates = pd.bdate_range("2015-01-02", periods=n)
    steps = gen.normal(0.0003, 0.011, (n, len(PIPE_UNIVERSE)))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)), index=dates, columns=PIPE_UNIVERSE
    )
    prices.index.name = "date"
    prices.to_csv(tmp_path / "prices.csv")
    (tmp_path / "adapter.py").write_text(_ADAPTER, encoding="utf-8")

    membership_line = ""
    if membership_rows is not None:
        write(tmp_path, membership_rows)
        membership_line = "  membership_frame: membership.csv\n"

    (tmp_path / "m.yaml").write_text(
        _MANIFEST.format(
            universe=", ".join(universe or PIPE_UNIVERSE),
            membership=membership_line,
            declared=declared,
        ),
        encoding="utf-8",
    )
    return audit_from_manifest(
        tmp_path / "m.yaml", offline=True, suite="engine", n_boot=150
    ).report


#: DEAD1 leaves inside the sample; everyone else is still a member.
PIPE_SPELLS = [
    ("AAA", "2015-01-02", ""),
    ("BBB", "2015-01-02", ""),
    ("CCC", "2015-01-02", ""),
    ("DDD", "2015-01-02", ""),
    ("DEAD1", "2015-01-02", "2017-03-10"),
]


class TestThroughTheManifest:
    def test_a_survivor_only_universe_is_measured_not_declared(self, tmp_path):
        report = _project(
            tmp_path,
            universe=["AAA", "BBB", "CCC", "DDD"],
            membership_rows=PIPE_SPELLS,
        )
        data = report.sections["data"]
        assert data["survivorship_basis"] == "measured"
        assert "DATA-SURVIVORSHIP" in {f.id for f in report.findings}
        assert data["measurement"]["missing_exited"] == ["DEAD1"]

    def test_the_complete_universe_produces_no_finding(self, tmp_path):
        """The paired must-pass, end to end."""
        report = _project(tmp_path, membership_rows=PIPE_SPELLS)
        assert "DATA-SURVIVORSHIP" not in {f.id for f in report.findings}
        assert report.sections["data"]["survivorship_basis"] == "measured"

    def test_extent_not_magnitude_is_always_stated(self, tmp_path):
        """Every measured run must say the count is not a haircut, or the number
        will be read as one."""
        report = _project(tmp_path, membership_rows=PIPE_SPELLS)
        assert any("not what their returns" in line for line in report.not_tested)

    def test_a_snapshot_list_does_not_earn_a_clean_result(self, tmp_path):
        snapshot = [(t, "2015-01-02", "") for t in PIPE_UNIVERSE]
        report = _project(
            tmp_path, universe=["AAA", "BBB"], membership_rows=snapshot
        )
        ids = {f.id for f in report.findings}
        assert "DATA-MEMBERSHIP-NOT-POINT-IN-TIME" in ids
        assert "DATA-SURVIVORSHIP" not in ids
        assert report.sections["data"]["survivorship_basis"] == "not_tested"

    def test_a_refused_list_is_not_a_pass(self, tmp_path):
        """A file describing another period must land in `not_tested` and emit
        no finding in either direction - the refusal must not read as clean."""
        elsewhere = [("AAA", "2022-01-03", ""), ("ZZZ", "2022-01-03", "2023-01-03")]
        report = _project(
            tmp_path, universe=["AAA", "BBB"], membership_rows=elsewhere
        )
        data = report.sections["data"]
        assert data["survivorship_basis"] == "not_tested"
        assert data["survivorship_basis_reason"]
        assert "DATA-SURVIVORSHIP" not in {f.id for f in report.findings}
        assert any("Survivorship" in line for line in report.not_tested)


class TestDeclarationMatrix:
    """Declaration x measurement. The cell that matters most is a `true`
    declaration the data disproves."""

    def test_declared_true_and_contradicted_fires_both(self, tmp_path):
        report = _project(
            tmp_path,
            universe=["AAA", "BBB", "CCC", "DDD"],
            membership_rows=PIPE_SPELLS,
            declared="true",
        )
        ids = {f.id for f in report.findings}
        assert "DATA-SURVIVORSHIP" in ids, "the universe really is biased"
        assert "DATA-DECLARATION-CONTRADICTED" in ids, "and the manifest denied it"

    def test_declared_true_and_corroborated_fires_neither(self, tmp_path):
        """The paired must-pass. A true declaration the data supports is the
        one case a researcher can actually earn."""
        report = _project(
            tmp_path, membership_rows=PIPE_SPELLS, declared="true"
        )
        ids = {f.id for f in report.findings}
        assert "DATA-SURVIVORSHIP" not in ids
        assert "DATA-DECLARATION-CONTRADICTED" not in ids

    def test_declared_false_but_measured_clean_still_fires_nothing_extra(
        self, tmp_path
    ):
        """A confession of bias outranks a null result from an optional file,
        but over-caution is not itself a defect - no contradiction finding."""
        report = _project(
            tmp_path, membership_rows=PIPE_SPELLS, declared="false"
        )
        assert "DATA-DECLARATION-CONTRADICTED" not in {f.id for f in report.findings}

    def test_declared_null_with_a_measurement_drops_the_open_question(
        self, tmp_path
    ):
        """Appending both the measurement and 'this could not be checked' is
        exactly the document-level contradiction the report tests exist for."""
        report = _project(tmp_path, membership_rows=PIPE_SPELLS)
        assert not any(
            "was not declared" in line for line in report.not_tested
        )

    def test_without_a_membership_file_nothing_changes(self, tmp_path):
        """Absent-field invariance. The five committed reports take this path,
        so the declaration behaviour must be untouched."""
        declared_false = _project(tmp_path, declared="false")
        assert "DATA-SURVIVORSHIP" in {f.id for f in declared_false.findings}
        assert declared_false.sections["data"]["survivorship_basis"] == "declared"

        not_declared = _project(tmp_path, declared="null")
        assert "DATA-SURVIVORSHIP" not in {f.id for f in not_declared.findings}
        assert not_declared.sections["data"]["survivorship_basis"] == "not_tested"
        assert any("was not declared" in line for line in not_declared.not_tested)


class TestTheCoverageChart:
    """Assert on the chart spec, not on pixels - the spec is required to hold
    exactly the values drawn, so testing it tests the picture."""

    #: The session LATE's price history begins on. Derived rather than written
    #: down: a membership date even one session off the data start reads as
    #: "traded before it was a member", which is a different finding.
    LATE_START = 250

    @classmethod
    def _coverage(cls, tmp_path, member_start=None):
        from qv.engine import universe_coverage

        dates = pd.bdate_range("2015-01-02", periods=500)
        if member_start is None:
            member_start = str(dates[cls.LATE_START].date())
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100, 130, 500),
                "LATE": np.concatenate(
                    [np.full(250, np.nan), np.linspace(50, 70, 250)]
                ),
            },
            index=dates,
        )
        rows = [("AAA", "2015-01-02", ""), ("LATE", member_start, "")]
        membership = read_membership_frame(write(tmp_path, rows))
        return universe_coverage(prices, membership=membership), dates

    def test_the_spec_carries_the_membership_spans_it_draws(self, tmp_path):
        from qv.report.charts import universe_coverage_chart

        coverage, _ = self._coverage(tmp_path)
        spec = universe_coverage_chart(coverage).spec
        assert "membership_first" in spec and "membership_last" in spec
        assert len(spec["membership_first"]) == len(spec["instruments"])
        assert spec["membership_last"][1] == coverage.n_obs - 1, "an open spell runs on"

    def test_an_explained_late_entrant_is_not_drawn_as_a_failure(self, tmp_path):
        """The chart must agree with the finding above it. Colouring an
        explained edge red would contradict the prose on the same page."""
        from qv.report.charts import universe_coverage_chart

        coverage, _ = self._coverage(tmp_path)
        spec = universe_coverage_chart(coverage).spec
        assert spec["explained_by_membership"] == ["LATE"]
        assert spec["unexplained_late"] == []
        assert spec["complete"] is True

    def test_an_unexplained_late_entrant_still_is(self, tmp_path):
        """The paired must-fail."""
        from qv.report.charts import universe_coverage_chart

        coverage, _ = self._coverage(tmp_path, "2015-01-02")
        spec = universe_coverage_chart(coverage).spec
        assert spec["unexplained_late"] == ["LATE"]
        assert spec["complete"] is False

    def test_a_name_absent_from_the_list_has_no_span(self, tmp_path):
        """-1 rather than 0: no membership span is a different thing from
        joining on day one, and drawing it as day one would invent a fact."""
        from qv.engine import universe_coverage

        dates = pd.bdate_range("2015-01-02", periods=300)
        prices = pd.DataFrame(
            {"AAA": np.linspace(100, 120, 300), "UNKNOWN": np.linspace(10, 12, 300)},
            index=dates,
        )
        membership = read_membership_frame(
            write(tmp_path, [("AAA", "2015-01-02", "")])
        )
        coverage = universe_coverage(prices, membership=membership)
        assert coverage.membership_first[1] == -1

    def test_without_membership_the_chart_is_unchanged(self, tmp_path):
        """Absent-field invariance for the picture: the five committed reports
        take this path."""
        from qv.engine import universe_coverage
        from qv.report.charts import universe_coverage_chart

        dates = pd.bdate_range("2015-01-02", periods=300)
        prices = pd.DataFrame({"AAA": np.linspace(100, 120, 300)}, index=dates)
        spec = universe_coverage_chart(universe_coverage(prices)).spec
        assert spec["membership_first"] == []
        assert spec["explained_by_membership"] == []


class TestStatedLimits:
    """A tool that measures its own resolution and keeps quiet about it is
    reporting a number more confidently than it earned."""

    def test_ticker_reuse_is_stated_when_it_cannot_be_resolved(self, tmp_path):
        from qv.audit import AuditInputs, run_audit

        rows = [
            ("AAA", "2015-01-02", ""),
            ("REUSED", "2015-01-02", "2016-01-04"),
            ("REUSED", "2017-01-03", ""),
            ("DEAD1", "2015-01-02", "2016-06-30"),
        ]
        membership = read_membership_frame(write(tmp_path, rows))
        gen = np.random.default_rng(3)
        report = run_audit(
            AuditInputs(
                returns=gen.normal(0.0004, 0.01, 600),
                membership=membership,
                sample_window=("2015-01-02", "2018-01-02"),
                universe_names=["AAA", "REUSED"],
                suite=Suite.ENGINE,
                n_boot=100,
            )
        )
        assert any("Ticker identity" in line for line in report.not_tested)
        assert any("REUSED" in line for line in report.not_tested)

    def test_an_id_column_resolves_it_and_the_limit_disappears(self, tmp_path):
        """The paired must-pass. Stating a limit that does not apply is noise,
        and noise is how a real limit gets skipped over."""
        from qv.audit import AuditInputs, run_audit

        rows = [
            ("AAA", "2015-01-02", "", "ID-A"),
            ("REUSED", "2015-01-02", "2016-01-04", "ID-R1"),
            ("REUSED", "2017-01-03", "", "ID-R2"),
            ("DEAD1", "2015-01-02", "2016-06-30", "ID-D"),
        ]
        membership = read_membership_frame(
            write(tmp_path, rows, columns=["ticker", "start_date", "end_date", "id"])
        )
        gen = np.random.default_rng(3)
        report = run_audit(
            AuditInputs(
                returns=gen.normal(0.0004, 0.01, 600),
                membership=membership,
                sample_window=("2015-01-02", "2018-01-02"),
                universe_names=["AAA", "REUSED"],
                suite=Suite.ENGINE,
                n_boot=100,
            )
        )
        assert not any("Ticker identity" in line for line in report.not_tested)

    def test_the_schema_constant_is_the_one_the_reader_enforces(self, tmp_path):
        """MEMBERSHIP_COLUMNS was declared, exported and never read. A schema
        written down twice drifts."""
        from qv.data.membership import MEMBERSHIP_COLUMNS

        path = write(tmp_path, [("AAA", "")], columns=["ticker", "end_date"])
        with pytest.raises(ValueError) as excinfo:
            read_membership_frame(path)
        assert ",".join(MEMBERSHIP_COLUMNS) in str(excinfo.value)


class TestEndpointConvention:
    """Both ends inclusive. Pinned because a name leaving on the last session is
    the kind of boundary that flips a count without anyone noticing."""

    def test_a_spell_ending_on_the_last_session_is_not_an_exit(self, tmp_path):
        """It was a member on every day of the window, so it did not leave
        during it. Being absent from the universe is then incompleteness rather
        than survivorship - which is the direction rule doing its job."""
        rows = [("AAA", "2015-01-02", ""), ("EDGE", "2015-01-02", "2018-01-02")]
        result = measurement(
            tmp_path, rows=rows, universe=["AAA"], window=("2015-01-02", "2018-01-02")
        )
        assert "EDGE" not in result.missing_exited
        assert "EDGE" in result.missing_still_member
        assert result.exits_total == 0

    def test_a_spell_ending_one_session_earlier_is(self, tmp_path):
        """The paired must-fail, one day apart."""
        rows = [("AAA", "2015-01-02", ""), ("EDGE", "2015-01-02", "2018-01-01")]
        result = measurement(
            tmp_path, rows=rows, universe=["AAA"], window=("2015-01-02", "2018-01-02")
        )
        assert result.missing_exited == ("EDGE",)
        assert result.exits_total == 1

    def test_a_single_day_spell_is_visible(self, tmp_path):
        """The shortest spell is reported so a reader knows the resolution: a
        membership shorter than the sampling interval can fall between
        observations entirely."""
        rows = [("AAA", "2015-01-02", ""), ("ONEDAY", "2016-03-01", "2016-03-02")]
        result = measurement(
            tmp_path, rows=rows, universe=["AAA"], window=("2015-01-02", "2018-01-02")
        )
        assert result.shortest_spell_days == 1
        assert result.missing_exited == ("ONEDAY",)


class TestNamedSources:
    """The tool ships a link and a parser, never a list. These tests exercise
    the parser with no network: the fetch is faked."""

    @staticmethod
    def _yaml_docs():
        """Two years of NASDAQ-100-shaped change files that agree with each other."""
        return {
            2015: (
                "year: 2015\n"
                "tickers_on_Jan_1: [AAA, BBB, CCC]\n"
                "changes:\n"
                "  2015-06-01:\n"
                "    difference: [CCC]\n"
                "    union: [DDD]\n"
            ),
            2016: (
                "year: 2016\n"
                "tickers_on_Jan_1: [AAA, BBB, DDD]\n"
                "changes:\n"
                "  2016-03-15:\n"
                "    difference: [BBB]\n"
                "    union: [EEE]\n"
            ),
        }

    def _install(self, monkeypatch, docs):
        def fake_download(url):
            for year, text in docs.items():
                if f"-{year}.yaml" in url:
                    return text.encode("utf-8")
            raise RuntimeError("no such year")

        monkeypatch.setattr("qv.data.loaders._download", fake_download)

    def test_the_reconstruction_walks_the_changes(self, monkeypatch):
        from qv.data.loaders import _nasdaq100_spells

        self._install(monkeypatch, self._yaml_docs())
        frame = _nasdaq100_spells((2015, 2018)).set_index("ticker")

        assert str(frame.loc["CCC", "end_date"].date()) == "2015-06-01"
        assert str(frame.loc["DDD", "start_date"].date()) == "2015-06-01"
        assert str(frame.loc["BBB", "end_date"].date()) == "2016-03-15"
        assert pd.isna(frame.loc["AAA", "end_date"]), "still a member"

    def test_a_disagreement_with_the_declared_list_refuses(self, monkeypatch):
        """The negative control, and the reason this reconstruction is
        trustworthy at all. Each yearly file declares its own 1 January
        membership, so the walk can be checked against it. A table that is
        quietly wrong is the worst thing this feature could produce."""
        from qv.data.loaders import _nasdaq100_spells

        docs = self._yaml_docs()
        # 2016 now claims a name the 2015 changes never added.
        docs[2016] = docs[2016].replace(
            "tickers_on_Jan_1: [AAA, BBB, DDD]", "tickers_on_Jan_1: [AAA, BBB, ZZZ]"
        )
        self._install(monkeypatch, docs)

        with pytest.raises(ValueError, match="disagrees with the declared membership"):
            _nasdaq100_spells((2015, 2018))

    def test_missing_years_simply_end_the_walk(self, monkeypatch):
        """Future years do not exist yet, which is not an error."""
        from qv.data.loaders import _nasdaq100_spells

        self._install(monkeypatch, self._yaml_docs())
        frame = _nasdaq100_spells((2015, 2030))
        assert len(frame) >= 4

    def test_an_unknown_source_names_the_ones_that_exist(self):
        from qv.data.loaders import load_index_membership

        with pytest.raises(ValueError, match="unknown membership source"):
            load_index_membership("ftse100")

    def test_offline_without_a_cache_names_the_remedy(self, monkeypatch, tmp_path):
        from qv.data.loaders import load_index_membership

        monkeypatch.setattr("qv.data.loaders.cache_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="without --offline"):
            load_index_membership("sp500", offline=True)

    def test_a_fetched_list_is_validated_like_any_other(self, monkeypatch, tmp_path):
        """A named source gets no special trust: it goes through exactly the
        same refusals a researcher's own file does."""
        import pandas as pd

        from qv.data.loaders import load_index_membership

        monkeypatch.setattr("qv.data.loaders.cache_dir", lambda: tmp_path)
        bad = pd.DataFrame({"ticker": ["AAA"], "start_date": ["2015-01-02"],
                            "end_date": ["2014-01-02"]}).to_csv(index=False)
        monkeypatch.setattr(
            "qv.data.loaders._download", lambda url: bad.encode("utf-8")
        )
        with pytest.raises(ValueError, match="ending on or before"):
            load_index_membership("sp500")

    def test_the_manifest_refuses_two_sources_at_once(self, tmp_path):
        """Picking one for the user would decide the answer for them."""
        from qv.manifest import DataSpec

        with pytest.raises(Exception, match="not both"):
            DataSpec(
                universe=["AAA", "BBB"],
                start_date="2015-01-02",
                end_date="2018-01-02",
                membership="sp500",
                membership_frame="local.csv",
            )
