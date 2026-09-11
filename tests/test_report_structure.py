"""Structural tests on the rendered HTML.

These exist because the first version of the report shipped with dashes where
numbers belonged, per-period Sharpes sitting next to annualised ones in the
same document, the decisive Attribution section buried below 900 KB of charts,
and the same finding listed twice. Every one of those was invisible to the
unit tests, because each individual number was correct - the *document* was
wrong.

So these tests read the rendered page the way a person would: parse the tables,
check every cell says something, check the sections appear in a sensible order,
and check the page never contradicts itself about units.
"""

from __future__ import annotations

import html as htmllib
import json
import re

import numpy as np
import pytest

from qv.audit import AuditInputs, run_audit
from qv.report.render import CHART_ORDER, build_charts, render_html, render_json

#: Cells that legitimately have nothing to say. Empty by default: a dash in a
#: report reads as a gap in the audit, so anything added here needs a reason.
_ALLOWED_EMPTY: set[tuple[str, str]] = set()


def _strip_svg(html: str) -> str:
    return re.sub(r"<svg.*?</svg>", "", html, flags=re.S)


def _tables(html: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every table as ``(headers, rows)`` of plain text."""
    out = []
    for block in re.findall(r"<table>(.*?)</table>", _strip_svg(html), re.S):
        rows = re.findall(r"<tr>(.*?)</tr>", block, re.S)
        if not rows:
            continue
        headers = [_text(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", rows[0], re.S)]
        body = []
        for row in rows[1:]:
            cells = [_text(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
            if cells:
                body.append(cells)
        out.append((headers, body))
    return out


def _flat(html: str) -> str:
    """Markup minus its SVGs, with runs of whitespace collapsed.

    A sentence in the template wraps across source lines; a reader sees one
    line, so a test looking for what the reader sees should look at one too.
    """
    return re.sub(r"\s+", " ", _strip_svg(html))


def _text(fragment: str) -> str:
    return htmllib.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def _headings(html: str) -> list[str]:
    return [_text(m) for m in re.findall(r"<h2[^>]*>(.*?)</h2>", _strip_svg(html), re.S)]


@pytest.fixture(scope="module")
def rendered():
    """A complete audit of both suites, with a benchmark and factors."""
    gen = np.random.default_rng(20)
    n = 900
    market = 0.0004 + 0.01 * gen.standard_normal(n)
    factors = np.column_stack(
        [market, 0.004 * gen.standard_normal(n), 0.004 * gen.standard_normal(n)]
    )
    positions = gen.integers(0, 2, n).astype(float)
    returns = positions * market

    # An honest strategy callable plus the two frames the engine charts need,
    # so the fixture exercises both suites for real. The behavioural leakage
    # and fragility sections are part of the document and the structural
    # checks below should see them.
    import pandas as pd

    raw_prices = pd.DataFrame(
        {"AAA": 100.0 * np.cumprod(1.0 + market)},
        index=pd.bdate_range("2018-01-01", periods=n),
    )

    def strategy(data):
        out = np.zeros(data.shape[0])
        out[1:] = (data[:-1, 0] > 0).astype(float)
        return out

    report = run_audit(
        AuditInputs(
            returns=returns,
            positions=positions,
            asset_returns=market,
            asset_class="us_large_cap_etf",
            benchmark_returns=market,
            benchmark_name="SPY",
            factors=factors,
            factor_names=["Mkt-RF", "SMB", "HML"],
            trial_returns=0.01 * gen.standard_normal((n, 30)),
            n_trials=30,
            parameter_scores=np.array([0.4, 0.9, 1.0, 0.85, 0.5]),
            strategy=strategy,
            strategy_data=market[:, None],
            asset_return_frame=market[:, None],
            raw_prices=raw_prices,
            n_boot=200,
            name="structure fixture",
        )
    )
    html = render_html(
        report,
        build_charts(report, returns, theme="light"),
        build_charts(report, returns, theme="dark"),
    )
    return report, html


class TestStatedConventions:
    """A page that reports a Sharpe has to say which Sharpe it means."""

    def test_an_absent_risk_free_rate_is_admitted_on_the_page(self, rendered):
        _, html = rendered
        assert "No risk-free rate was supplied" in _flat(html)

    def test_a_supplied_rate_is_stated_with_its_level(self):
        gen = np.random.default_rng(21)
        n = 900
        returns = 0.0004 + 0.01 * gen.standard_normal(n)
        report = run_audit(
            AuditInputs(
                returns=returns,
                risk_free=np.full(n, 0.00006),
                benchmark_returns=0.0003 + 0.012 * gen.standard_normal(n),
                benchmark_name="SPY",
                n_boot=100,
                name="rate fixture",
            )
        )
        body = _flat(render_html(report, build_charts(report, returns, theme="light")))
        assert "in excess of the risk-free rate" in body
        assert "1.51% a year" in body

    def test_a_small_probability_is_not_rounded_away_to_zero(self, rendered):
        """0.02% and 0.0% read the same and mean different things."""
        report, _ = rendered
        report.sections["pbo"]["probability_of_loss"] = 0.0002
        body = _flat(render_html(report, build_charts(report, None, theme="light")))
        assert "&lt;0.1%" in body


class TestNumberFormatting:
    def test_a_gap_that_rounds_to_zero_is_not_printed_as_negative(self):
        """"-0.00" in a difference column reads as a loss the table cannot
        actually resolve."""
        from qv.report.render import _performance_rows

        rows = _performance_rows(
            {"sharpe": 0.74311, "total_return": 1.0},
            {"sharpe": 0.74334, "total_return": 1.0},
            lambda v, places=2: "-" if v is None else f"{float(v):,.{places}f}",
            lambda v, places=1: "-" if v is None else f"{float(v) * 100:.{places}f}%",
        )
        sharpe = next(r for r in rows if r["label"].startswith("Sharpe"))
        assert sharpe["difference"] == "0.00"
        for row in rows:
            assert row["difference"] not in ("-0.00", "+0.00", "-0.00 pp", "+0.00 pp")


class TestNoGaps:
    def test_no_table_cell_is_empty_or_a_dash(self, rendered):
        """The regression this file was written for.

        A dash in a report reads as "the audit did not get this", not as "not
        applicable". Rows that cannot fill every column belong in prose or
        split into rows that can.
        """
        _, html = rendered
        offenders = []
        for headers, rows in _tables(html):
            for row in rows:
                for i, cell in enumerate(row):
                    column = headers[i] if i < len(headers) else f"col{i}"
                    if cell in ("", "-", "nan", "None", "N/A"):
                        if (row[0], column) not in _ALLOWED_EMPTY:
                            offenders.append(f"{row[0]!r} / {column!r} = {cell!r}")
        assert not offenders, "empty cells:\n  " + "\n  ".join(offenders)

    def test_no_nan_or_none_leaks_into_the_prose(self, rendered):
        _, html = rendered
        body = _strip_svg(html)
        for token in ("nan", "None", "inf", "Undefined"):
            assert not re.search(rf"\b{token}\b", body), f"{token!r} rendered to the page"

    def test_every_table_has_headers_for_its_widest_row(self, rendered):
        _, html = rendered
        for headers, rows in _tables(html):
            widest = max((len(r) for r in rows), default=0)
            assert len(headers) >= widest, f"{headers} has fewer headers than {widest} columns"

    def test_every_table_has_at_least_one_row(self, rendered):
        _, html = rendered
        for headers, rows in _tables(html):
            assert rows, f"table {headers} rendered with no rows"


class TestUnitsAreConsistent:
    def test_annualised_sharpes_agree_across_sections(self, rendered):
        """The bug this catches: the performance table said 0.59 while the
        benchmark table said 0.037 for the same quantity, because one was
        annualised and the other was not."""
        report, _ = rendered
        performance = report.sections["performance"]["sharpe"]
        benchmark = report.sections["benchmark"]["strategy_sharpe_annualised"]
        assert benchmark == pytest.approx(performance, rel=1e-9)

    def test_report_states_its_sharpe_convention(self, rendered):
        _, html = rendered
        assert "annualised to" in html

    #: Rows whose label mentions Sharpe but which are not a Sharpe *level*, so
    #: the annualisation check does not apply. Each is here for a reason, not
    #: to make the test pass.
    _NOT_A_SHARPE_LEVEL = (
        "deflated sharpe ratio",          # a probability in [0, 1]
        "sharpe the strategy adds",       # a difference, legitimately near zero
        "survives the selection",         # a yes/no
    )

    def test_rows_labelled_per_period_are_the_only_unannualised_ones(self, rendered):
        """Any Sharpe *level* not labelled per-period must be on the annualised
        scale, which for these fixtures means well above the per-period figure.

        This is the check that would have caught the performance table saying
        0.59 while the benchmark table said 0.037 for the same quantity.
        """
        report, html = rendered
        periodic = abs(report.sections["sharpe"]["periodic"]["value"])
        assert periodic > 0, "fixture must have a non-zero Sharpe for this to mean anything"

        checked = 0
        for _, rows in _tables(html):
            for row in rows:
                label = row[0].lower()
                if "sharpe" not in label:
                    continue
                if "per-period" in label or "per period" in label:
                    continue
                if any(skip in label for skip in self._NOT_A_SHARPE_LEVEL):
                    continue
                numbers = [abs(float(v)) for v in re.findall(r"-?\d+\.\d+", row[1])]
                if not numbers:
                    continue
                checked += 1
                assert max(numbers) > periodic, (
                    f"{row[0]!r} shows {row[1]!r}, which looks per-period in a table "
                    "that does not say so"
                )
        assert checked >= 3, "the check found almost nothing to test; has the report changed?"

    def test_annualised_fields_are_in_the_json(self, rendered):
        """Every number the page shows must be in report.json."""
        report, _ = rendered
        payload = json.loads(render_json(report))
        assert "strategy_sharpe_annualised" in payload["benchmark"]
        assert "sharpes_annualised" in payload["regimes"]
        assert "min_sharpe_annualised" in payload["rolling_origin"]
        assert "sharpe_after_dropping_top_5pct_annualised" in payload["top_days"]
        assert "naive_ci_low" in payload["bootstrap"]


class TestSectionOrder:
    def test_findings_come_before_the_evidence(self, rendered):
        """Attribution - the section that falsified the momentum strategy - was
        at the very bottom of a 900 KB page. The conclusion belongs at the top."""
        _, html = rendered
        headings = _headings(html)
        assert headings[0].startswith("Findings")
        for later in ("Headline", "Selection bias", "Robustness"):
            assert headings.index(later) > 0

    def test_findings_appear_in_the_first_quarter_of_the_page(self, rendered):
        _, html = rendered
        assert html.index("<h2>Findings") / len(html) < 0.25

    def test_gaps_are_reported_near_the_top(self, rendered):
        _, html = rendered
        headings = _headings(html)
        assert "What could not be tested" in headings
        assert headings.index("What could not be tested") <= 1

    def test_every_computed_section_is_rendered(self, rendered):
        """A section present in report.json but absent from the page is a
        result the reader silently never sees."""
        report, html = rendered
        headings = " ".join(_headings(html)).lower()
        expected = {
            "attribution": "attribution",
            "costs": "cost realism",
            "pbo": "selection procedure",
            "selection": "selection bias",
            "performance": "performance",
        }
        for key, heading in expected.items():
            if key in report.sections:
                assert heading in headings, f"{key} computed but never rendered"


class TestFindingsPresentation:
    def test_no_finding_id_appears_twice(self, rendered):
        """Factor regression and the volatility-matched benchmark are two
        detections of one defect; listing them as two identical findings with
        the same title and remediation reads as carelessness."""
        report, _ = rendered
        ids = [f.id for f in report.findings]
        assert len(ids) == len(set(ids)), f"duplicated findings: {ids}"

    def test_every_finding_carries_a_remediation(self, rendered):
        report, html = rendered
        for finding in report.findings:
            assert finding.remediation
            assert finding.id in html

    def test_finding_count_in_the_heading_matches_the_list(self, rendered):
        report, html = rendered
        heading = next(h for h in _headings(html) if h.startswith("Findings"))
        assert heading == f"Findings ({len(report.findings)})"


class TestSelfContained:
    def test_no_external_references(self, rendered):
        _, html = rendered
        assert re.findall(r'(?:src|href)="(?!#|data:)', html) == []

    def test_no_javascript(self, rendered):
        _, html = rendered
        assert "<script" not in html.lower()
        assert "onclick" not in html.lower()

    def test_both_themes_are_embedded(self, rendered):
        _, html = rendered
        assert html.count("<svg") == 2 * len(CHART_ORDER)
        assert "body:has(#theme-toggle:checked)" in html


class TestReadsCorrectly:
    """Checks on the prose, not just the numbers.

    A page can have every figure right and still not make sense. These are the
    contradictions a careful read turned up.
    """

    def test_two_per_period_sharpes_are_distinguished(self, rendered):
        """The headline showed 0.0374 and Selection bias showed 0.0369 for a
        row labelled the same way. Both were right - deflation runs on returns
        net of costs - but the page gave the reader no way to know that."""
        report, html = rendered
        headline = report.sections["sharpe"]["periodic"]["value"]
        deflated_on = report.sections["selection"]["deflated_sharpe"]["observed_sharpe"]
        if abs(headline - deflated_on) > 1e-12:
            assert "net of costs" in html
            assert "Deflation runs on" in html

    def test_the_deflation_basis_is_always_stated(self, rendered):
        report, html = rendered
        assert report.sections["selection"]["basis"] in ("gross", "net of costs")
        assert report.sections["selection"]["basis_note"] in html

    def test_asset_class_is_shown_as_prose_not_an_identifier(self, rendered):
        """"us large cap etf" is a lowercased variable name, not a market."""
        _, html = rendered
        assert "us large cap etf" not in html
        assert "US large-cap ETFs" in html

    def test_the_autocorrelation_direction_is_described_in_words(self, rendered):
        """Reporting a rise from 15.87 to 17.63 as a "-11.0% change" is the
        haircut sign convention leaking into prose, and reads as an error."""
        _, html = rendered
        assert ("cutting the annualised Sharpe" in html) or (
            "raising</em> the annualised Sharpe" in html
            or "<em>raising</em> the annualised Sharpe" in html
        )

    def test_the_post_drop_sharpe_is_given_context(self, rendered):
        """A Sharpe of -1.40 sitting beside a concentration reading of "as
        expected" looks like a contradiction without the explanation."""
        _, html = rendered
        if "dropping the best" in html:
            assert "falls steeply for almost any strategy" in html

    def test_no_row_label_is_absurdly_long(self, rendered):
        """A table row label is not a place for a sentence."""
        _, html = rendered
        for _, rows in _tables(html):
            for row in rows:
                assert len(row[0]) <= 70, f"row label runs long: {row[0]!r}"

    def test_the_verdict_paragraph_matches_the_verdict(self, rendered):
        report, html = rendered
        if report.verdict.value == "not_falsified":
            assert "Nothing below broke this backtest" in html
        else:
            assert "does not show that it does" in html or "not enough data" in html


class TestTheFooterIdentifiesTheRun:
    """The footer promises reproducibility; it has to say of what.

    "Re-running with the same inputs and seed reproduces every number above"
    was on the page long before anything named the engine or the data vintage,
    which made it a promise with no referent. These read the footer the way a
    person checking that claim would.
    """

    def test_the_engine_is_named(self, rendered):
        _, html = rendered
        footer = _flat(html).split("<footer>")[1]
        assert re.search(r"commit [0-9a-f]{7}", footer), footer
        assert re.search(r"source [0-9a-f]{12}", footer), footer

    def test_the_reproducibility_claim_refers_to_the_source(self, rendered):
        _, html = rendered
        assert "same inputs and seed, on this same source" in _flat(html)

    def test_the_identity_is_in_the_json_too(self, rendered):
        """The HTML may never make a claim `report.json` cannot back."""
        report, html = rendered
        data = json.loads(render_json(report))
        prov = data["provenance"]
        assert prov["source_digest"] in html
        assert prov["version"] and prov["source_digest"]

    def test_vintages_are_shown_as_prose(self):
        from qv.audit import AuditReport
        from qv.types import Tier

        report = AuditReport(name="v", tier=Tier.RETURNS, n_obs=100, periods_per_year=252)
        report.provenance = {
            "generated_utc": "now", "data_hash": "h", "seed": 0, "n_boot": 1,
            "python": "3", "numpy": "2",
            "data_vintages": {
                "prices": "2025-03-01",
                "factors": "2025-02-14",
                "benchmark": "2025-03-01",
                "factor_alignment": "5 sessions dropped at the join",
            },
        }
        footer = _flat(render_html(report, [])).split("<footer>")[1]
        assert "prices 2025-03-01" in footer
        assert "Fama-French factors 2025-02-14" in footer
        assert "benchmark 2025-03-01" in footer
        assert "5 sessions dropped at the join" in footer
        # `factor_alignment` is a variable name and its value is a sentence,
        # not a date. Both would be wrong in the list of dates.
        assert "factor_alignment" not in footer
        assert "factor alignment 5 sessions" not in footer.lower()
        assert "Fama-French files are periodically revised" in footer

    def test_the_revision_note_matches_the_data_actually_used(self):
        """No factors, no sentence about Dartmouth revising the factor files.

        The reason a vintage matters has to be the reason that applies to this
        report, or it reads as boilerplate and stops being read at all.
        """
        from qv.audit import AuditReport
        from qv.types import Tier

        report = AuditReport(name="v", tier=Tier.RETURNS, n_obs=100, periods_per_year=252)
        report.provenance = {
            "generated_utc": "now", "data_hash": "h", "seed": 0, "n_boot": 1,
            "python": "3", "numpy": "2",
            "data_vintages": {"prices": "2026-09-09"},
        }
        footer = _flat(render_html(report, [])).split("<footer>")[1]
        assert "prices 2026-09-09" in footer
        assert "Fama-French" not in footer
        assert "restated after the fact" in footer

    def test_a_report_with_no_vintages_says_nothing_about_them(self):
        """The paired negative control.

        A flat-file audit has no vintages - the researcher supplied the
        series. The footer must then omit the sentence entirely rather than
        print a label with nothing after it.
        """
        from qv.audit import AuditReport
        from qv.types import Tier

        report = AuditReport(name="v", tier=Tier.RETURNS, n_obs=100, periods_per_year=252)
        report.provenance = {
            "generated_utc": "now", "data_hash": "h", "seed": 0, "n_boot": 1,
            "python": "3", "numpy": "2",
        }
        footer = _flat(render_html(report, [])).split("<footer>")[1]
        assert "vintage" not in footer.lower()


class TestDegradesWithoutSections:
    """A missing section must omit a sentence, not take the page down."""

    def test_renders_with_no_sections_at_all(self):
        from qv.audit import AuditReport
        from qv.report.render import render_html
        from qv.types import Tier

        report = AuditReport(name="bare", tier=Tier.RETURNS, n_obs=100, periods_per_year=252)
        report.provenance = {
            "generated_utc": "now", "data_hash": "h", "seed": 0,
            "n_boot": 1, "python": "3", "numpy": "2",
        }
        html = render_html(report, [])
        assert "bare" in html and "</html>" in html

    def test_tier_zero_report_renders_completely(self):
        from qv.audit import AuditInputs, run_audit
        from qv.report.render import build_charts, render_html

        gen = np.random.default_rng(31)
        returns = 0.0004 + 0.01 * gen.standard_normal(400)
        report = run_audit(AuditInputs(returns=returns, n_boot=100))
        html = render_html(report, build_charts(report, returns))
        assert "</html>" in html
        for _, rows in _tables(html):
            for row in rows:
                assert all(c not in ("", "nan", "None") for c in row), row
