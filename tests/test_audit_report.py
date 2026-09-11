"""Tests for the audit orchestrator, the charts, and the report renderer."""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from qv.audit import AuditInputs, AuditReport, run_audit
from qv.report import charts as ch
from qv.report.render import (
    CHART_ORDER,
    build_charts,
    render_html,
    render_json,
    write_report,
)
from qv.types import Severity, Tier, Verdict
from tests.conftest import make_returns


@pytest.fixture
def simple_inputs():
    gen = np.random.default_rng(3)
    returns = 0.0005 + 0.01 * gen.standard_normal(800)
    return AuditInputs(returns=returns, periods_per_year=252, n_boot=200, name="test strategy")


@pytest.fixture
def rich_inputs():
    """A complete handover: returns, positions, asset, trials and a callable.

    Complete enough that every chart in both suites is drawn for real rather
    than as a placeholder, which is what makes the chart-count assertions
    below mean something.
    """
    import pandas as pd

    gen = np.random.default_rng(4)
    market = 0.0004 + 0.01 * gen.standard_normal(700)
    positions = gen.integers(0, 2, 700).astype(float)
    trials = 0.01 * gen.standard_normal((700, 40))
    prices = pd.DataFrame(
        {"AAA": 100.0 * np.cumprod(1.0 + market)},
        index=pd.bdate_range("2018-01-01", periods=700),
    )

    def strategy(data):
        # Deliberately leaky - reads the next period - so the behavioural
        # leakage test has something to find.
        x = data[:, 0]
        out = np.zeros_like(x)
        out[:-1] = np.sign(x[1:])
        return out

    return AuditInputs(
        returns=positions * market,
        positions=positions,
        asset_returns=market,
        asset_return_frame=market[:, None],
        raw_prices=prices,
        asset_class="us_large_cap_etf",
        benchmark_returns=market,
        trial_returns=trials,
        n_trials=40,
        parameter_scores=np.array([0.4, 0.9, 1.0, 0.85, 0.5]),
        strategy=strategy,
        strategy_data=market[:, None],
        n_boot=200,
        name="rich strategy",
    )


class TestRiskFreeBasis:
    """The Sharpe ratio is defined on excess returns.

    Computing it on total returns flatters the least volatile series in any
    comparison, because the same cash rate is a larger share of a smaller
    denominator - which is exactly the comparison a report leads with.
    """

    def test_a_supplied_rate_lowers_every_sharpe(self, simple_inputs):
        gross = run_audit(simple_inputs)
        simple_inputs.risk_free = np.full(800, 0.00006)
        net = run_audit(simple_inputs)
        assert net.sections["sharpe"]["periodic"]["value"] < (
            gross.sections["sharpe"]["periodic"]["value"]
        )
        assert net.sections["performance"]["sharpe"] < gross.sections["performance"]["sharpe"]
        assert net.sections["risk"]["sharpe"]["realised"] < (
            gross.sections["risk"]["sharpe"]["realised"]
        )

    def test_level_and_path_statistics_stay_on_total_returns(self, simple_inputs):
        """Drawdown is what an investor lived through, not a ratio."""
        gross = run_audit(simple_inputs)
        simple_inputs.risk_free = np.full(800, 0.00006)
        net = run_audit(simple_inputs)
        for key in ("total_return", "annualised_return", "max_drawdown", "calmar"):
            assert net.sections["performance"][key] == pytest.approx(
                gross.sections["performance"][key]
            )
        assert net.sections["risk"]["total_return"]["realised"] == pytest.approx(
            gross.sections["risk"]["total_return"]["realised"]
        )

    def test_the_comparison_narrows_when_cash_is_taken_off_both_sides(self):
        """The distortion that matters: a low-volatility strategy against a
        volatile benchmark gains a Sharpe advantage it has not earned."""
        gen = np.random.default_rng(11)
        strategy = 0.0003 + 0.004 * gen.standard_normal(1500)
        benchmark = 0.0005 + 0.012 * gen.standard_normal(1500)
        base = AuditInputs(
            returns=strategy, benchmark_returns=benchmark, n_boot=100, name="low vol"
        )
        gross = run_audit(base).sections["benchmark"]["excess_sharpe"]
        base.risk_free = np.full(1500, 0.00006)
        net = run_audit(base).sections["benchmark"]["excess_sharpe"]
        assert net < gross

    def test_an_absent_rate_is_declared_rather_than_assumed(self, simple_inputs):
        report = run_audit(simple_inputs)
        assert report.sections["conventions"]["risk_free_supplied"] is False
        assert any("Risk-free adjustment" in g for g in report.not_tested)

    def test_idle_cash_is_quantified_when_the_book_is_not_fully_invested(self):
        gen = np.random.default_rng(12)
        positions = np.full(600, 0.6)
        market = 0.0004 + 0.01 * gen.standard_normal(600)
        report = run_audit(
            AuditInputs(
                returns=positions * market,
                positions=positions,
                asset_class="us_large_cap_etf",
                risk_free=np.full(600, 0.00006),
                n_boot=100,
            )
        )
        conv = report.sections["conventions"]
        assert conv["mean_idle_weight"] == pytest.approx(0.4)
        assert conv["idle_cash_understatement"] == pytest.approx(0.4 * 0.00006 * 252)

    def test_a_scalar_rate_is_accepted_as_well_as_a_series(self):
        """A researcher who knows the average cash rate should not have to
        synthesise a series to say so."""
        gen = np.random.default_rng(14)
        returns = 0.0004 + 0.01 * gen.standard_normal(600)
        series = run_audit(
            AuditInputs(returns=returns, risk_free=np.full(600, 0.00006), n_boot=100)
        )
        scalar = run_audit(AuditInputs(returns=returns, risk_free=0.00006, n_boot=100))
        assert scalar.sections["sharpe"]["periodic"]["value"] == pytest.approx(
            series.sections["sharpe"]["periodic"]["value"]
        )

    def test_a_long_short_book_is_not_asked_about_idle_cash(self):
        gen = np.random.default_rng(13)
        positions = gen.choice([-1.0, 1.0], 600)
        market = 0.0004 + 0.01 * gen.standard_normal(600)
        report = run_audit(
            AuditInputs(returns=positions * market, positions=positions, n_boot=100)
        )
        assert report.sections["conventions"]["mean_idle_weight"] is None


class TestSurvivorship:
    """Undetectable from a return series at any tier, so it is declared."""

    def test_present_day_membership_is_a_finding(self, simple_inputs):
        simple_inputs.universe_point_in_time = False
        simple_inputs.universe_note = "S&P 500 members as of today"
        report = run_audit(simple_inputs)
        finding = next(f for f in report.findings if f.id == "DATA-SURVIVORSHIP")
        assert finding.severity is Severity.HIGH
        assert "S&P 500 members as of today" in finding.detail

    def test_an_undeclared_universe_leaves_the_question_open(self, simple_inputs):
        report = run_audit(simple_inputs)
        assert not any(f.id == "DATA-SURVIVORSHIP" for f in report.findings)
        assert any(g.startswith("Survivorship:") for g in report.not_tested)

    def test_a_declared_point_in_time_universe_raises_nothing(self, simple_inputs):
        simple_inputs.universe_point_in_time = True
        report = run_audit(simple_inputs)
        assert not any(f.id == "DATA-SURVIVORSHIP" for f in report.findings)
        assert not any(g.startswith("Survivorship:") for g in report.not_tested)
        assert report.sections["data"]["universe_point_in_time"] is True


class TestTierDetection:
    def test_returns_only_is_tier_zero(self, simple_inputs):
        assert simple_inputs.tier is Tier.RETURNS

    def test_positions_reach_tier_one(self):
        i = AuditInputs(returns=np.zeros(50) + 0.01, positions=np.ones(50))
        assert i.tier is Tier.POSITIONS

    def test_a_callable_reaches_tier_two(self, rich_inputs):
        assert rich_inputs.tier is Tier.CALLABLE


class TestRunAudit:
    def test_tier_zero_still_produces_a_report(self, simple_inputs):
        report = run_audit(simple_inputs)
        assert report.tier is Tier.RETURNS
        assert "sharpe" in report.sections
        assert "bootstrap" in report.sections

    def test_records_what_it_could_not_test(self, simple_inputs):
        """Tiered honesty: gaps are stated, not silently omitted."""
        report = run_audit(simple_inputs)
        assert report.not_tested
        joined = " ".join(report.not_tested)
        assert "position series" in joined
        assert "Behavioural leakage test" in joined

    def test_tier_two_tests_far_more(self, simple_inputs, rich_inputs):
        thin = run_audit(simple_inputs)
        rich = run_audit(rich_inputs)
        assert len(rich.sections) > len(thin.sections)
        assert len(rich.not_tested) < len(thin.not_tested)

    def test_tier_two_runs_the_behavioural_test(self, rich_inputs):
        report = run_audit(rich_inputs)
        assert "perturbation" in report.sections

    def test_a_leaky_strategy_is_falsified(self, rich_inputs):
        report = run_audit(rich_inputs)
        assert any(f.id == "LEAK-BEHAVIOURAL" for f in report.findings)
        assert report.verdict is Verdict.FALSIFIED

    def test_verdict_has_no_pass_state(self):
        """The asymmetry is the whole point: the best outcome available is
        'the tests did not break it', never 'it works'."""
        assert not hasattr(Verdict, "PASSED")
        assert Verdict.NOT_FALSIFIED.label == "Survived the tests applied"

    def test_small_sample_short_circuits_the_verdict(self):
        report = run_audit(AuditInputs(returns=make_returns(20, 0.3, seed=1), n_boot=50))
        assert report.verdict is Verdict.INSUFFICIENT_DATA
        assert any(f.id == "STAT-SMALL-SAMPLE" for f in report.findings)

    def test_undeclared_trials_is_itself_a_finding(self, simple_inputs):
        report = run_audit(simple_inputs)
        assert any(f.id == "SELECT-UNDECLARED-TRIALS" for f in report.findings)

    def test_declaring_trials_unlocks_deflation(self, simple_inputs):
        simple_inputs.n_trials = 50
        report = run_audit(simple_inputs)
        assert "selection" in report.sections
        assert not any(f.id == "SELECT-UNDECLARED-TRIALS" for f in report.findings)

    def test_stages_build_a_cascade(self, rich_inputs):
        stages = run_audit(rich_inputs).headline_sharpe_stages
        assert len(stages) >= 2
        assert stages[0][0].startswith("Claimed")

    def test_the_cascade_composes_and_names_its_last_test(self, rich_inputs):
        """Two bugs in one test.

        The final bar used to be labelled for the Deflated Sharpe while
        plotting the BHY haircut - beside a table saying the Deflated Sharpe
        passed - and it was applied to the gross figure rather than to the
        stage above it, so the cascade silently reverted its own base.
        """
        report = run_audit(rich_inputs)
        stages = report.headline_sharpe_stages
        label, value = stages[-1][0], stages[-1][1]
        assert "BHY" in label
        assert "Deflated" not in label
        haircut = report.sections["selection"]["haircut"]["haircut"]
        assert value == pytest.approx(stages[-2][1] * (1.0 - haircut))

    def test_a_censored_haircut_says_so_and_states_the_bar(self, rich_inputs):
        """A BHY haircut floors at zero for anything below its threshold, so
        the last marker is not a measurement. Left bare it reads as "no edge"
        rather than "this correction cannot resolve one"."""
        report = run_audit(rich_inputs)
        haircut = report.sections["selection"]["haircut"]
        note = report.headline_sharpe_stages[-1][4]
        if haircut["censored"]:
            assert note is not None and "fully absorbed" in note
            assert f"{haircut['required_tstat']:.2f}" in note
        else:
            assert note is None

    def test_bulk_arrays_stay_out_of_the_sections(self, rich_inputs):
        """They belong in chart_data, or report.json becomes a megabyte of
        numbers nobody reads."""
        report = run_audit(rich_inputs)
        assert "logits" not in report.sections.get("pbo", {})
        assert "distribution" not in report.sections.get("null_max", {})
        assert "pbo" in report.chart_data

    def test_findings_sort_by_severity(self, rich_inputs):
        findings = run_audit(rich_inputs).findings_by_severity()
        severities = [int(f.severity) for f in findings]
        assert severities == sorted(severities, reverse=True)

    def test_is_reproducible_from_the_seed(self, simple_inputs):
        a = run_audit(simple_inputs).to_dict()
        b = run_audit(simple_inputs).to_dict()
        a["provenance"].pop("generated_utc")
        b["provenance"].pop("generated_utc")
        assert a == b

    def test_provenance_is_recorded(self, simple_inputs):
        p = run_audit(simple_inputs).provenance
        assert set(p) >= {"data_hash", "seed", "n_boot", "python", "numpy", "generated_utc"}

    def test_provenance_names_the_engine_that_produced_it(self, simple_inputs):
        """A report that cannot be tied to its code is asking to be trusted.

        The footer's reproducibility promise has no referent without these,
        and `report.json` is where the HTML's claims have to be backed.
        """
        p = run_audit(simple_inputs).to_dict()["provenance"]
        assert set(p) >= {"version", "commit", "source_digest"}
        assert p["version"]
        assert len(p["source_digest"]) == 12

    def test_data_hash_changes_with_the_data(self, simple_inputs):
        first = run_audit(simple_inputs).provenance["data_hash"]
        simple_inputs.returns = np.asarray(simple_inputs.returns) * 1.01
        assert run_audit(simple_inputs).provenance["data_hash"] != first

    def test_to_dict_is_json_serialisable(self, rich_inputs):
        json.dumps(render_json(run_audit(rich_inputs)))

    def test_rejects_a_series_that_is_too_short(self):
        with pytest.raises(ValueError, match="at least 4"):
            run_audit(AuditInputs(returns=[0.01, 0.02]))


class TestVerdictLadder:
    @pytest.mark.parametrize(
        "severity,expected",
        [
            (Severity.CRITICAL, Verdict.FALSIFIED),
            (Severity.HIGH, Verdict.WEAKENED),
            (Severity.MEDIUM, Verdict.NOT_FALSIFIED),
            (Severity.INFO, Verdict.NOT_FALSIFIED),
        ],
    )
    def test_maps_worst_severity_to_verdict(self, severity, expected):
        from qv.findings import make_finding

        report = AuditReport(name="x", tier=Tier.RETURNS, n_obs=500, periods_per_year=252)
        report.findings = [make_finding("LEAK-BACKWARD-FILL", "d", severity=severity)]
        assert report.verdict is expected

    def test_no_findings_is_not_falsified(self):
        report = AuditReport(name="x", tier=Tier.RETURNS, n_obs=500, periods_per_year=252)
        assert report.verdict is Verdict.NOT_FALSIFIED
        assert report.worst_severity is Severity.INFO


class TestCharts:
    def test_all_five_are_built(self, rich_inputs):
        report = run_audit(rich_inputs)
        built = build_charts(report, returns=np.asarray(rich_inputs.returns))
        assert [c.name for c in built] == list(CHART_ORDER)
        assert all(c.available for c in built)

    def test_tier_zero_degrades_rather_than_failing(self, simple_inputs):
        """A Tier 0 report must still render completely."""
        report = run_audit(simple_inputs)
        built = build_charts(report, returns=np.asarray(simple_inputs.returns))
        assert len(built) == len(CHART_ORDER)
        unavailable = [c for c in built if not c.available]
        assert unavailable
        for chart in unavailable:
            assert chart.reason
            assert "chart-unavailable" in chart.svg

    def test_svg_is_inline_ready(self, rich_inputs):
        """No XML prolog or DOCTYPE, or the markup cannot sit inside a body."""
        for chart in build_charts(run_audit(rich_inputs), np.asarray(rich_inputs.returns)):
            assert "<?xml" not in chart.svg
            assert "DOCTYPE" not in chart.svg.upper()

    def test_charts_stay_within_a_sane_size_budget(self, rich_inputs):
        """A self-contained report that nobody can email is not self-contained
        in any useful sense. The PBO scatter is rasterised for this reason."""
        built = build_charts(run_audit(rich_inputs), np.asarray(rich_inputs.returns))
        total_kb = sum(len(c.svg) for c in built) / 1024
        assert total_kb < 600, f"charts total {total_kb:.0f} KB"

    def test_every_chart_carries_a_spec(self, rich_inputs):
        """The rule that keeps report.json authoritative: nothing is drawn that
        is not also stated as data."""
        for chart in build_charts(run_audit(rich_inputs), np.asarray(rich_inputs.returns)):
            if chart.available:
                assert chart.spec, f"{chart.name} drew something it did not record"

    def test_haircut_cascade_spec_matches_its_input(self):
        stages = [("A", 2.0, 1.0, 3.0), ("B", 1.0, None, None)]
        spec = ch.haircut_cascade(stages).spec
        assert [s["label"] for s in spec["stages"]] == ["A", "B"]
        assert spec["stages"][0]["ci_low"] == 1.0
        assert spec["stages"][1]["ci_low"] is None

    def test_null_chart_records_the_percentile(self):
        draws = np.linspace(0, 1, 1000)
        assert ch.max_sharpe_null(draws, observed=0.5).spec["percentile"] == pytest.approx(
            0.5, abs=0.01
        )

    def test_non_finite_values_become_null_in_specs(self):
        spec = ch.haircut_cascade([("A", 1.0, float("nan"), float("inf"))]).spec
        assert spec["stages"][0]["ci_low"] is None
        assert spec["stages"][0]["ci_high"] is None

    def test_regime_chart_records_the_split(self):
        r = np.concatenate([np.full(100, 0.01), np.full(100, -0.01)])
        mask = np.concatenate([np.zeros(100, bool), np.ones(100, bool)])
        spec = ch.regime_equity_curve(r, regime_mask=mask).spec
        assert spec["fraction_in_regime"] == pytest.approx(0.5)
        assert spec["return_in_regime_sum"] == pytest.approx(-1.0)
        assert spec["return_out_of_regime_sum"] == pytest.approx(1.0)

    def test_equity_curve_ends_where_the_performance_table_does(self):
        """Compounded, not summed. The curve sits directly above a table
        reporting total return, and two different numbers for the same quantity
        is a puzzle for the reader rather than a chart."""
        r = np.full(100, 0.01)
        spec = ch.regime_equity_curve(r).spec
        assert spec["final_cumulative_return"] == pytest.approx(1.01**100 - 1.0)

    def test_dates_put_years_on_the_axis(self):
        r = np.full(400, 0.001)
        dates = np.datetime64("2010-01-01") + np.arange(400)
        assert ch.regime_equity_curve(r, dates=dates).spec["dated"] is True
        assert ch.regime_equity_curve(r).spec["dated"] is False
        # A mismatched length is presentational, not fatal.
        assert ch.regime_equity_curve(r, dates=dates[:10]).spec["dated"] is False

    def test_a_total_loss_falls_back_to_the_cumulative_sum(self):
        """The compounded path is pinned at -100% after a wipeout and stops
        saying anything; the sum at least keeps drawing."""
        r = np.array([0.01, -1.0, 0.02, 0.03])
        spec = ch.regime_equity_curve(r).spec
        assert spec["final_cumulative_return"] == pytest.approx(r.sum())

    def test_long_series_are_decimated_for_drawing_only(self):
        """The drawn path is thinned; the recorded statistics are not."""
        long_returns = np.random.default_rng(0).standard_normal(20_000) * 0.01
        chart = ch.regime_equity_curve(long_returns)
        assert chart.spec["n_obs"] == 20_000
        assert len(chart.svg) < 400_000

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: ch.haircut_cascade([]),
            lambda: ch.max_sharpe_null([1.0], observed=0.5),
            lambda: ch.break_even_curve([], []),
            lambda: ch.pbo_panel([], [], [], 0.5),
            lambda: ch.regime_equity_curve([0.1]),
        ],
    )
    def test_degenerate_input_returns_a_placeholder(self, factory):
        chart = factory()
        assert not chart.available and chart.reason

    def test_unavailable_placeholder_is_self_describing(self):
        chart = ch.unavailable("x", "A title", "because reasons")
        assert "A title" in chart.svg and "because reasons" in chart.svg
        assert chart.to_dict()["available"] is False


class TestRenderHtml:
    def test_is_self_contained(self, rich_inputs):
        html = render_html(run_audit(rich_inputs), build_charts(run_audit(rich_inputs),
                                                                 np.asarray(rich_inputs.returns)))
        refs = re.findall(r'(?:src|href)="(?!#|data:)', html)
        assert refs == [], f"external references found: {refs}"
        assert "<script" not in html.lower()

    def test_embeds_the_charts_unescaped(self, rich_inputs):
        """Autoescaping is on for finding text; the SVG must be marked safe or
        it renders as visible angle brackets."""
        report = run_audit(rich_inputs)
        html = render_html(report, build_charts(report, np.asarray(rich_inputs.returns)))
        assert html.count("<svg") == len(CHART_ORDER)
        assert "&lt;svg" not in html

    def test_leads_with_the_falsification_framing(self, simple_inputs):
        report = run_audit(simple_inputs)
        html = render_html(report, build_charts(report, np.asarray(simple_inputs.returns)))
        assert "falsifies; it cannot validate" in html

    def test_shows_the_suite_and_verdict(self, simple_inputs):
        report = run_audit(simple_inputs)
        html = render_html(report, build_charts(report, np.asarray(simple_inputs.returns)))
        # The report leads with which questions it asked, not with a tier.
        # Input completeness still drives the "could not be tested" list, but
        # it is no longer a badge a reader has to reconcile against the suites.
        assert report.suite.label in html
        for retired in ("Tier 0", "Tier 1", "Tier 2"):
            assert retired not in html
        assert report.verdict.label in html

    def test_lists_what_could_not_be_tested(self, simple_inputs):
        report = run_audit(simple_inputs)
        html = render_html(report, build_charts(report, np.asarray(simple_inputs.returns)))
        assert "What could not be tested" in html

    def test_renders_every_finding_with_its_remediation(self, rich_inputs):
        report = run_audit(rich_inputs)
        html = render_html(report, build_charts(report, np.asarray(rich_inputs.returns)))
        for finding in report.findings:
            assert finding.id in html
        assert "What to do:" in html

    def test_escapes_untrusted_text(self):
        """Finding detail can carry a filename from a scanned notebook."""
        from qv.findings import make_finding

        report = AuditReport(name="x", tier=Tier.RETURNS, n_obs=100, periods_per_year=252)
        report.findings = [make_finding("LEAK-BACKWARD-FILL", "<img src=x onerror=1>")]
        report.provenance = {"generated_utc": "now", "data_hash": "h", "seed": 0,
                             "n_boot": 1, "python": "3", "numpy": "2"}
        html = render_html(report, [])
        assert "<img src=x" not in html
        assert "&lt;img" in html

    def test_renders_at_tier_zero_without_optional_sections(self, simple_inputs):
        report = run_audit(simple_inputs)
        html = render_html(report, build_charts(report, np.asarray(simple_inputs.returns)))
        assert "Cost realism" not in html
        assert "Headline" in html


class TestRenderJsonAndWrite:
    def test_json_is_valid_and_lean(self, rich_inputs):
        report = run_audit(rich_inputs)
        payload = render_json(report, build_charts(report, np.asarray(rich_inputs.returns)))
        data = json.loads(payload)
        assert len(payload) / 1024 < 200, "report.json should not be megabytes"
        assert set(data) >= {"verdict", "tier", "findings", "provenance", "charts"}

    def test_non_finite_numbers_become_null(self):
        report = AuditReport(name="x", tier=Tier.RETURNS, n_obs=10, periods_per_year=252)
        report.sections["thing"] = {"value": float("nan"), "other": float("inf")}
        data = json.loads(render_json(report))
        assert data["thing"]["value"] is None
        assert data["thing"]["other"] is None

    def test_write_report_produces_both_files(self, rich_inputs, tmp_path):
        report = run_audit(rich_inputs)
        html_path, json_path = write_report(
            report, tmp_path / "out", returns=np.asarray(rich_inputs.returns)
        )
        assert html_path.exists() and json_path.exists()
        assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
        json.loads(json_path.read_text(encoding="utf-8"))

    def test_report_size_is_shareable(self, rich_inputs, tmp_path):
        """A full report is nine charts in two themes, so it is over a megabyte.

        The bound moved from 900 KB when the engine suite added three charts.
        They are the cheapest on the page - 26 to 42 KB against 46 to 92 KB for
        the statistical six - so the growth is proportionate rather than a
        regression; most of a matplotlib SVG is fixed per-figure overhead.
        A single-suite report is much lighter, which is one reason to run them
        separately.
        """
        report = run_audit(rich_inputs)
        html_path, _ = write_report(report, tmp_path, returns=np.asarray(rich_inputs.returns))
        assert html_path.stat().st_size / 1024 < 1400

    def test_a_single_suite_report_is_much_lighter(self, rich_inputs, tmp_path):
        from dataclasses import replace

        from qv.types import Suite

        engine = run_audit(replace(rich_inputs, suite=Suite.ENGINE))
        full = run_audit(rich_inputs)
        engine_path, _ = write_report(
            engine, tmp_path / "engine", returns=np.asarray(rich_inputs.returns)
        )
        full_path, _ = write_report(
            full, tmp_path / "full", returns=np.asarray(rich_inputs.returns)
        )
        assert engine_path.stat().st_size < full_path.stat().st_size / 2


class TestThemes:
    """Both palettes must render, and the report must carry both."""

    def test_every_chart_renders_in_both_themes(self, rich_inputs):
        report = run_audit(rich_inputs)
        returns = np.asarray(rich_inputs.returns)
        for theme in ch.THEMES:
            built = build_charts(report, returns=returns, theme=theme)
            assert len(built) == len(CHART_ORDER)
            assert all(c.available for c in built)

    def test_dark_charts_use_the_dark_background(self, rich_inputs):
        report = run_audit(rich_inputs)
        dark = build_charts(report, np.asarray(rich_inputs.returns), theme="dark")
        light = build_charts(report, np.asarray(rich_inputs.returns), theme="light")
        assert ch.PALETTES["dark"]["background"].lstrip("#") in dark[0].svg.lower()
        assert ch.PALETTES["dark"]["background"].lstrip("#") not in light[0].svg.lower()

    def test_specs_are_identical_across_themes(self, rich_inputs):
        """Theme is presentation. If the numbers changed with the colours,
        report.json could not be authoritative for both."""
        report = run_audit(rich_inputs)
        returns = np.asarray(rich_inputs.returns)
        light = {c.name: c.spec for c in build_charts(report, returns, theme="light")}
        dark = {c.name: c.spec for c in build_charts(report, returns, theme="dark")}
        assert light == dark

    def test_html_embeds_both_sets_and_a_pure_css_toggle(self, rich_inputs):
        report = run_audit(rich_inputs)
        returns = np.asarray(rich_inputs.returns)
        html = render_html(
            report,
            build_charts(report, returns, theme="light"),
            build_charts(report, returns, theme="dark"),
        )
        assert html.count("<svg") == 2 * len(CHART_ORDER)
        assert 'id="theme-toggle"' in html
        assert "light-only" in html and "dark-only" in html
        # No JavaScript: the report stays a single inert file.
        assert "<script" not in html.lower()

    def test_falls_back_to_one_set_when_no_dark_charts_given(self, rich_inputs):
        report = run_audit(rich_inputs)
        returns = np.asarray(rich_inputs.returns)
        html = render_html(report, build_charts(report, returns))
        assert html.count("<svg") == len(CHART_ORDER)

    def test_unknown_theme_raises(self):
        with pytest.raises(KeyError):
            ch.haircut_cascade([("A", 1.0, None, None)], theme="neon")


class TestTableFormatting:
    def test_numeric_headers_are_right_aligned(self, rich_inputs):
        """Left-aligned headers over right-aligned numbers put the label at the
        far side of its own column, which is what the first draft did."""
        report = run_audit(rich_inputs)
        html = render_html(report, build_charts(report, np.asarray(rich_inputs.returns)))
        assert 'th.num, td.num { text-align:right' in html
        assert html.count('<th class="num">') >= 8

    def test_charts_span_the_text_column(self, rich_inputs):
        report = run_audit(rich_inputs)
        html = render_html(report, build_charts(report, np.asarray(rich_inputs.returns)))
        assert "figure.chart svg { width:100%; height:auto" in html


class TestPerformanceSection:
    def test_summary_is_recorded_for_the_strategy(self, simple_inputs):
        section = run_audit(simple_inputs).sections["performance"]
        assert set(section) >= {"sharpe", "sortino", "max_drawdown", "annualised_return"}

    def test_benchmark_summary_appears_when_a_benchmark_is_given(self, rich_inputs):
        report = run_audit(rich_inputs)
        assert "benchmark_performance" in report.sections
        assert report.chart_data.get("benchmark_returns")

    def test_no_benchmark_section_without_a_benchmark(self, simple_inputs):
        assert "benchmark_performance" not in run_audit(simple_inputs).sections

    def test_comparison_table_renders_both_columns(self, rich_inputs):
        report = run_audit(rich_inputs)
        html = render_html(report, build_charts(report, np.asarray(rich_inputs.returns)))
        for label in ("Annualised volatility", "Sortino ratio", "Maximum drawdown",
                      "Calmar ratio", "Hit rate"):
            assert label in html

    def test_difference_column_uses_points_for_percentages(self):
        from qv.report.render import _performance_rows

        rows = _performance_rows(
            {"total_return": 0.50, "sharpe": 1.2},
            {"total_return": 0.30, "sharpe": 0.8},
            lambda v, places=2: "-" if v is None else f"{float(v):.2f}",
            lambda v, places=1: "-" if v is None else f"{float(v) * 100:.1f}%",
        )
        by_label = {r["label"]: r for r in rows}
        # A ratio difference of +0.40 must not be rendered as a percentage.
        assert by_label["Sharpe ratio"]["difference"] == "+0.40"
        assert by_label["Total return"]["difference"] == "+20.00 pp"

    def test_missing_benchmark_leaves_dashes(self):
        from qv.report.render import _performance_rows

        rows = _performance_rows(
            {"sharpe": 1.0}, None,
            lambda v, places=2: "-" if v is None else f"{float(v):.2f}",
            lambda v, places=1: "-",
        )
        assert all(r["benchmark"] == "-" and r["difference"] == "-" for r in rows)


class TestMatchedExposureGuards:
    def test_multi_asset_positions_skip_the_test_with_a_reason(self):
        """Reordering a cross-sectional book in time does not answer the
        question this test asks."""
        gen = np.random.default_rng(11)
        report = run_audit(
            AuditInputs(
                returns=0.01 * gen.standard_normal(400),
                positions=gen.random((400, 5)),
                asset_returns=0.01 * gen.standard_normal(400),
                n_boot=100,
            )
        )
        assert "matched_exposure" not in report.sections
        assert any("multi-asset" in gap for gap in report.not_tested)

    def test_constant_exposure_skips_the_test(self):
        """A shuffled constant series would report a p-value near 1 and look
        like a finding when nothing was measured."""
        gen = np.random.default_rng(12)
        report = run_audit(
            AuditInputs(
                returns=0.01 * gen.standard_normal(400),
                positions=np.ones(400),
                asset_returns=0.01 * gen.standard_normal(400),
                n_boot=100,
            )
        )
        assert "matched_exposure" not in report.sections
        assert any("no timing decision" in gap for gap in report.not_tested)
