"""Tests for the descriptive performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.performance import (
    PerformanceSummary,
    annualised_return,
    annualised_volatility,
    downside_deviation,
    max_drawdown,
    sortino_ratio,
    summarise_performance,
)
from tests.conftest import make_returns


class TestAnnualisedReturn:
    def test_known_answer(self):
        """1% a period over 12 periods with 12 per year compounds to 12.68%."""
        assert annualised_return([0.01] * 12, 12) == pytest.approx(1.01**12 - 1)

    def test_is_geometric_not_arithmetic(self):
        """+50% then -50% is a 25% loss, not break-even. The arithmetic mean
        would say zero, and that difference is the whole reason to compound."""
        assert annualised_return([0.5, -0.5], 2) == pytest.approx(-0.25)

    def test_scales_with_frequency(self):
        r = make_returns(504, 0.05, seed=1)
        assert annualised_return(r, 252) > annualised_return(r, 12)

    def test_total_loss_is_nan_not_a_number(self):
        assert math.isnan(annualised_return([-1.0, 0.5], 2))

    def test_empty_is_nan(self):
        assert math.isnan(annualised_return([], 252))


class TestVolatility:
    def test_scales_by_sqrt_q(self):
        r = make_returns(1000, 0.05, seed=2)
        assert annualised_volatility(r, 252) == pytest.approx(
            float(np.std(r, ddof=1)) * math.sqrt(252)
        )

    def test_needs_two_observations(self):
        assert math.isnan(annualised_volatility([0.01], 252))


class TestDownsideAndSortino:
    def test_downside_deviation_ignores_upside(self):
        """Two series with identical downside and different upside must have
        the same downside deviation."""
        mild = [-0.01, 0.01, -0.01, 0.01]
        wild = [-0.01, 0.50, -0.01, 0.50]
        assert downside_deviation(mild, 0.0, 1) == pytest.approx(
            downside_deviation(wild, 0.0, 1)
        )

    def test_sortino_exceeds_sharpe_for_positive_skew(self):
        """The reason both are reported: a right-skewed strategy looks worse on
        Sharpe than its downside risk warrants."""
        from qv.stats.sharpe import sharpe_ratio

        gen = np.random.default_rng(3)
        r = np.concatenate([0.001 + 0.005 * gen.standard_normal(900), np.full(100, 0.08)])
        sharpe = sharpe_ratio(r) * math.sqrt(252)
        assert sortino_ratio(r, 0.0, 252) > sharpe

    def test_all_positive_returns_give_nan_sortino(self):
        """No downside means no downside deviation to divide by."""
        assert math.isnan(sortino_ratio([0.01] * 50, 0.0, 252))

    def test_target_shifts_the_result(self):
        r = make_returns(500, 0.05, seed=4)
        assert sortino_ratio(r, 0.0, 252) > sortino_ratio(r, 0.01, 252)

    def test_needs_two_observations(self):
        assert math.isnan(sortino_ratio([0.01], 0.0, 252))
        assert math.isnan(downside_deviation([0.01], 0.0, 252))


class TestMaxDrawdown:
    def test_known_answer(self):
        """+100% then -50% then +10%: peak at index 0 after the gain, trough
        after the halving, so the drawdown is exactly -50%."""
        depth, peak, trough = max_drawdown([1.0, -0.5, 0.1])
        assert depth == pytest.approx(-0.5)
        assert peak == 0 and trough == 1

    def test_is_compounded_not_cumulative_sum(self):
        """A 50% loss then a 50% gain is not break-even, and an auditor should
        see the real hole rather than a cumulative-sum artefact."""
        depth, _, _ = max_drawdown([-0.5, 0.5])
        assert depth == pytest.approx(-0.5)

    def test_monotonic_gains_have_no_drawdown(self):
        assert max_drawdown([0.01] * 50)[0] == pytest.approx(0.0)

    def test_reports_peak_and_trough_indices(self):
        depth, peak, trough = max_drawdown([0.1, 0.1, -0.3, -0.3, 0.2])
        assert peak == 1 and trough == 3
        assert depth < -0.4

    def test_empty_is_nan(self):
        assert math.isnan(max_drawdown([])[0])


class TestSummary:
    def test_collects_everything(self):
        s = summarise_performance(make_returns(1000, 0.05, seed=5), 252, "test")
        assert s.name == "test" and s.n_obs == 1000
        for field in (
            "total_return", "annualised_return", "annualised_volatility",
            "sharpe", "sortino", "max_drawdown", "hit_rate",
        ):
            assert math.isfinite(getattr(s, field)), field

    def test_sharpe_matches_the_stats_module(self):
        from qv.stats.sharpe import sharpe_ratio

        r = make_returns(800, 0.05, seed=6)
        assert summarise_performance(r, 252).sharpe == pytest.approx(
            sharpe_ratio(r) * math.sqrt(252)
        )

    def test_calmar_is_return_over_drawdown_depth(self):
        s = summarise_performance(make_returns(1000, 0.05, seed=7), 252)
        assert s.calmar == pytest.approx(s.annualised_return / abs(s.max_drawdown))

    def test_calmar_is_nan_without_a_drawdown(self):
        s = summarise_performance([0.01] * 50, 252)
        assert math.isnan(s.calmar)

    def test_hit_rate_counts_positive_periods(self):
        assert summarise_performance([0.01, -0.01, 0.01, 0.01], 4).hit_rate == 0.75

    def test_best_and_worst(self):
        s = summarise_performance([0.05, -0.03, 0.01], 4)
        assert s.best_period == pytest.approx(0.05)
        assert s.worst_period == pytest.approx(-0.03)

    def test_to_dict_is_json_shaped(self):
        import json

        d = summarise_performance(make_returns(500, 0.05, seed=8), 252).to_dict()
        assert set(d) >= {"sharpe", "sortino", "max_drawdown", "calmar", "annualised_return"}
        json.dumps(d)

    def test_rejects_a_single_observation(self):
        with pytest.raises(ValueError, match="at least 2"):
            summarise_performance([0.01], 252)

    def test_comparable_across_two_series(self):
        """The whole point of the summary: strategy and benchmark measured the
        same way over the same periods."""
        a = summarise_performance(make_returns(1000, 0.08, seed=9), 252, "strategy")
        b = summarise_performance(make_returns(1000, 0.03, seed=10), 252, "benchmark")
        assert a.sharpe > b.sharpe
        assert a.periods_per_year == b.periods_per_year == 252

    def test_an_immediate_loss_is_a_drawdown(self):
        """The path starts at 1.0, before the first return. Without that, a
        strategy that drops 30% straight away and recovers reports zero."""
        depth, _, _ = max_drawdown([-0.3, 0.6])
        assert depth == pytest.approx(-0.3)

    def test_first_period_loss_indices(self):
        depth, peak, trough = max_drawdown([-0.5, 0.5])
        assert depth == pytest.approx(-0.5)
        assert peak == 0 and trough == 0
