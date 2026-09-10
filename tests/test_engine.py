"""The engine suite: determinism, degeneracy, delay fragility, coverage.

Each of these answers a question the statistical suite cannot reach, and two
of them exist because the answer was previously *assumed*: the adapter
protocol has demanded determinism since it was written and nothing verified
it, and the condition that makes a leakage result meaningful - that the book
actually moves - lived only in the test suite, so a researcher whose adapter
silently returned a flat book got a clean bill of health and no hint why.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qv.engine import (
    DEFAULT_DELAYS,
    determinism_check,
    execution_delay_curve,
    signal_degeneracy,
    universe_coverage,
)
from qv.types import Severity


def _honest(data: np.ndarray) -> np.ndarray:
    out = np.zeros(data.shape[0])
    out[1:] = (data[:-1, 0] > 0).astype(float)
    return out


class TestDeterminism:
    @pytest.fixture
    def data(self):
        return np.random.default_rng(0).standard_normal((200, 2))

    def test_a_pure_function_is_deterministic(self, data):
        result = determinism_check(_honest, data)
        assert result.deterministic
        assert result.max_difference == 0.0
        assert result.to_finding() is None

    def test_unseeded_randomness_is_caught(self, data):
        """The failure this exists for. It also breaks the leakage test, which
        reads any difference as evidence of look-ahead - so a strategy that
        disagrees with itself reports as leaky and sends the researcher after
        a bug that is really a missing seed."""

        def noisy(d):
            return np.random.default_rng().standard_normal(d.shape[0])

        result = determinism_check(noisy, data)
        assert not result.leaked if hasattr(result, "leaked") else True
        assert not result.deterministic
        assert result.max_difference > 0
        finding = result.to_finding()
        assert finding.id == "ENGINE-NONDETERMINISTIC"
        assert finding.severity is Severity.CRITICAL

    def test_a_seeded_generator_inside_the_function_is_fine(self, data):
        def seeded(d):
            return np.random.default_rng(7).standard_normal(d.shape[0])

        assert determinism_check(seeded, data).deterministic

    def test_a_shape_that_changes_between_calls_is_caught(self, data):
        calls = []

        def unstable(d):
            calls.append(1)
            width = 1 if len(calls) == 1 else 2
            return np.zeros((d.shape[0], width))

        result = determinism_check(unstable, data)
        assert not result.deterministic
        assert not math.isfinite(result.max_difference)

    def test_matching_nans_are_not_a_difference(self, data):
        def with_nans(d):
            out = np.full(d.shape[0], np.nan)
            out[5:] = 1.0
            return out

        assert determinism_check(with_nans, data).deterministic

    def test_rejects_a_single_call(self, data):
        with pytest.raises(ValueError, match="at least 2 calls"):
            determinism_check(_honest, data, n_calls=1)

    def test_to_dict_is_json_shaped(self, data):
        import json

        json.dumps(determinism_check(_honest, data).to_dict())


class TestDegeneracy:
    def test_a_moving_book_is_not_degenerate(self):
        gen = np.random.default_rng(1)
        book = (gen.random((300, 3)) > 0.5).astype(float) / 3.0
        result = signal_degeneracy(book)
        assert not result.degenerate
        assert result.n_changes > 10
        assert result.to_finding() is None

    def test_an_uninvested_book_is_critical(self):
        """Almost always an adapter fault, and invisible in the wrong
        direction: every leakage test passes and the report reads clean."""
        result = signal_degeneracy(np.zeros((100, 4)))
        assert result.never_invested
        assert result.degenerate
        finding = result.to_finding()
        assert finding.id == "ENGINE-NO-POSITIONS"
        assert finding.severity is Severity.CRITICAL

    def test_a_book_that_never_moves_is_flagged(self):
        book = np.ones((100, 2)) / 2.0
        result = signal_degeneracy(book)
        assert result.n_changes == 0
        assert result.constant_exposure
        assert result.to_finding().id == "ENGINE-DEGENERATE-SIGNAL"

    def test_constant_exposure_alone_is_not_a_finding(self):
        """A fully-invested rotation has constant gross exposure and is a
        perfectly ordinary strategy. Only the absence of *movement* matters."""
        book = np.zeros((200, 3))
        for t in range(200):
            book[t, t % 3] = 1.0
        result = signal_degeneracy(book)
        assert result.constant_exposure
        assert not result.degenerate
        assert result.to_finding() is None

    def test_counts_distinct_books(self):
        book = np.tile(np.eye(3), (40, 1))
        assert signal_degeneracy(book).n_distinct_books == 3

    def test_accepts_a_one_dimensional_signal(self):
        signal = np.tile([0.0, 1.0], 100)
        assert signal_degeneracy(signal).n_instruments == 1

    @pytest.mark.parametrize(
        "signals,match",
        [
            (np.zeros((1, 2)), "at least 2 observations"),
            (np.zeros((4, 2, 2)), "1-D or 2-D"),
        ],
    )
    def test_rejects_bad_input(self, signals, match):
        with pytest.raises(ValueError, match=match):
            signal_degeneracy(signals)


class TestExecutionDelay:
    @staticmethod
    def _returns(n=800, seed=2, drift=0.0006):
        """A drift large enough that buy-and-hold has a clearly positive
        realised Sharpe on this sample - retention is a proportion, and a
        proportion of a non-positive base does not mean anything."""
        return np.random.default_rng(seed).normal(drift, 0.008, (n, 1))

    def test_a_fully_invested_book_survives_delay(self):
        """Delaying a book that never changes cannot change anything much,
        which is what a slow rebalance should look like."""
        rets = self._returns()
        book = np.ones_like(rets)
        result = execution_delay_curve(book, rets)
        assert result.base_sharpe > 0
        assert result.one_period_retention > 0.95
        assert result.to_finding() is None
        assert result.severity is Severity.INFO

    def test_a_one_period_edge_is_destroyed_by_one_period_of_delay(self):
        """The failure this test exists for: an edge that only exists at zero
        delay is a property of the backtest's timing, not of the market.

        The book here holds exactly when the period it is about to earn is
        up - a perfect same-period signal. Note the alignment: with no delay
        `book[t]` earns `rets[t]`, so a perfect signal is `book[t] =
        sign(rets[t])`. Setting it from `rets[t+1]` instead would be a
        *look-ahead*, which is a different defect and the leakage test's job.
        """
        rets = self._returns()
        book = (rets > 0).astype(float)
        result = execution_delay_curve(book, rets)
        assert result.base_sharpe > 5.0
        assert result.one_period_retention < 0.2
        finding = result.to_finding()
        assert finding.id == "ENGINE-EXECUTION-FRAGILE"
        assert finding.severity is Severity.CRITICAL

    def test_the_curve_starts_at_the_reported_timing(self):
        rets = self._returns()
        book = np.ones_like(rets)
        result = execution_delay_curve(book, rets)
        assert result.delays[0] == 0
        assert result.delays == DEFAULT_DELAYS
        assert len(result.sharpes) == len(DEFAULT_DELAYS)

    def test_retention_is_relative_to_zero_delay(self):
        rets = self._returns()
        book = np.ones_like(rets)
        result = execution_delay_curve(book, rets)
        assert result.retention[0] == pytest.approx(1.0)

    def test_a_negative_base_sharpe_gives_no_verdict(self):
        """A proportion of a non-positive number does not mean anything, so
        the finding is withheld rather than invented."""
        rets = self._returns()
        book = -np.ones_like(rets)
        result = execution_delay_curve(book, rets)
        if result.base_sharpe <= 0:
            assert not math.isfinite(result.one_period_retention)
            assert result.to_finding() is None

    def test_works_on_a_multi_asset_book(self):
        gen = np.random.default_rng(5)
        rets = gen.normal(0.0003, 0.01, (600, 4))
        book = np.full((600, 4), 0.25)
        result = execution_delay_curve(book, rets)
        assert math.isfinite(result.base_sharpe)

    @pytest.mark.parametrize(
        "book,rets,match",
        [
            (np.ones((10, 1)), np.ones((9, 1)), "same time axis"),
            (np.ones((10, 2)), np.ones((10, 3)), "same instruments"),
        ],
    )
    def test_rejects_mismatched_shapes(self, book, rets, match):
        with pytest.raises(ValueError, match=match):
            execution_delay_curve(book, rets)

    def test_requires_the_reported_timing_in_the_sweep(self):
        rets = self._returns()
        with pytest.raises(ValueError, match="must include 0"):
            execution_delay_curve(np.ones_like(rets), rets, delays=(1, 2, 3))

    def test_to_dict_is_json_shaped(self):
        import json

        rets = self._returns()
        json.dumps(execution_delay_curve(np.ones_like(rets), rets).to_dict())


class TestUniverseCoverage:
    @staticmethod
    def _frame(n=400):
        return pd.DataFrame(
            np.ones((n, 4)),
            index=pd.bdate_range("2010-01-04", periods=n),
            columns=["AAA", "BBB", "CCC", "DDD"],
        )

    def test_a_complete_universe_is_clean(self):
        result = universe_coverage(self._frame())
        assert result.complete
        assert result.late_entrants == ()
        assert result.early_exits == ()
        assert result.to_finding() is None
        assert all(c == pytest.approx(1.0) for c in result.coverage)

    def test_a_late_entrant_is_found(self):
        """The same family of error as survivorship, and unlike survivorship
        it is visible in the data rather than only declarable."""
        frame = self._frame()
        frame.loc[frame.index[:100], "DDD"] = np.nan
        result = universe_coverage(frame)
        assert result.late_entrants == ("DDD",)
        assert not result.complete
        finding = result.to_finding()
        assert finding.id == "ENGINE-INCLUSION-TIMING"
        assert "DDD" in finding.detail

    def test_an_early_exit_is_found(self):
        frame = self._frame()
        frame.loc[frame.index[-50:], "BBB"] = np.nan
        result = universe_coverage(frame)
        assert result.early_exits == ("BBB",)
        assert result.to_finding() is not None

    def test_coverage_is_a_fraction_of_the_sample(self):
        frame = self._frame(n=400)
        frame.loc[frame.index[:200], "AAA"] = np.nan
        result = universe_coverage(frame)
        assert result.coverage[0] == pytest.approx(0.5, abs=0.01)

    def test_measuring_after_alignment_hides_everything(self):
        """Why the *unaligned* frame is what has to be passed. Dropping the
        ragged rows first is exactly what makes a patchy universe look clean,
        so a check run afterwards can only ever report that all is well."""
        frame = self._frame()
        frame.loc[frame.index[:100], "DDD"] = np.nan
        assert not universe_coverage(frame).complete
        assert universe_coverage(frame.dropna()).complete

    def test_carries_index_labels_for_the_chart(self):
        result = universe_coverage(self._frame())
        assert len(result.labels) == result.n_obs
        assert result.labels[0].startswith("2010")

    def test_an_entirely_missing_instrument_does_not_crash(self):
        frame = self._frame()
        frame["DDD"] = np.nan
        result = universe_coverage(frame)
        assert "DDD" in result.late_entrants or "DDD" in result.early_exits

    def test_rejects_a_short_frame(self):
        with pytest.raises(ValueError, match="at least 2 observations"):
            universe_coverage(self._frame(n=1))

    def test_to_dict_is_json_shaped(self):
        import json

        json.dumps(universe_coverage(self._frame()).to_dict())


class TestSuiteSeparation:
    """The two suites must answer independently, which is the whole point."""

    @staticmethod
    def _inputs(**overrides):
        from qv.audit import AuditInputs

        gen = np.random.default_rng(3)
        n = 600
        market = 0.0004 + 0.01 * gen.standard_normal(n)
        positions = gen.integers(0, 2, n).astype(float)
        prices = pd.DataFrame(
            {"AAA": 100.0 * np.cumprod(1.0 + market)},
            index=pd.bdate_range("2018-01-01", periods=n),
        )
        base = {
            "returns": positions * market,
            "positions": positions,
            "asset_returns": market,
            "asset_return_frame": market[:, None],
            "raw_prices": prices,
            "asset_class": "us_large_cap_etf",
            "strategy": _honest,
            "strategy_data": market[:, None],
            "n_trials": 20,
            "trial_returns": 0.01 * gen.standard_normal((n, 20)),
            "n_boot": 200,
        }
        base.update(overrides)
        return AuditInputs(**base)

    def test_each_suite_reports_only_its_own_sections(self):
        from qv.audit import run_audit
        from qv.types import Suite

        statistical = run_audit(self._inputs(suite=Suite.STATISTICAL)).sections
        engine = run_audit(self._inputs(suite=Suite.ENGINE)).sections

        engine_only = {
            "determinism",
            "degeneracy",
            "execution_delay",
            "universe_coverage",
            "perturbation",
        }
        statistical_only = {"sharpe", "selection", "pbo", "costs", "risk", "regimes"}

        assert engine_only.isdisjoint(statistical)
        assert statistical_only.isdisjoint(engine)
        assert engine_only <= set(engine)
        assert statistical_only <= set(statistical)

    def test_the_full_suite_is_the_union(self):
        from qv.audit import run_audit
        from qv.types import Suite

        statistical = set(run_audit(self._inputs(suite=Suite.STATISTICAL)).sections)
        engine = set(run_audit(self._inputs(suite=Suite.ENGINE)).sections)
        full = set(run_audit(self._inputs(suite=Suite.FULL)).sections)
        assert statistical | engine == full

    def test_a_statistical_failure_does_not_move_the_engine_verdict(self):
        """The separation earns its keep here. A strategy can be statistically
        hopeless and impeccably implemented, and the reverse is the more
        dangerous case because the numbers look fine."""
        from qv.audit import run_audit
        from qv.types import Suite, Verdict

        # Pure noise with a large declared search: the statistical suite should
        # object, the engine suite has nothing to object to.
        gen = np.random.default_rng(9)
        n = 600
        market = 0.01 * gen.standard_normal(n)
        positions = np.ones(n)
        inputs = self._inputs(
            returns=positions * market,
            positions=positions,
            asset_returns=market,
            asset_return_frame=market[:, None],
            n_trials=5000,
            trial_returns=0.01 * gen.standard_normal((n, 60)),
        )
        from dataclasses import replace

        engine = run_audit(replace(inputs, suite=Suite.ENGINE))
        assert engine.verdict is not Verdict.FALSIFIED
        assert not any(f.id.startswith("SELECT") for f in engine.findings)

    def test_an_engine_report_says_it_is_not_about_performance(self):
        from qv.audit import run_audit
        from qv.report.render import render_html, build_charts
        from qv.types import Suite

        report = run_audit(self._inputs(suite=Suite.ENGINE))
        html = render_html(report, build_charts(report))
        assert "Engine analysis" in html
        assert "says nothing about whether the" in html
