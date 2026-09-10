"""Tests for turnover, cost models and break-even cost."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.costs.breakeven import break_even_cost, sharpe_vs_cost_curve
from qv.costs.models import (
    TYPICAL_COST_BPS,
    FixedBpsCost,
    SpreadProportionalCost,
    apply_costs,
    as_position_matrix,
    average_turnover,
    turnover_series,
)
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity


class TestPositionsAndTurnover:
    def test_one_dimensional_positions_become_one_asset(self):
        assert as_position_matrix([0.0, 1.0, 1.0]).shape == (3, 1)

    def test_charges_the_cost_of_building_the_book(self):
        """Going from flat to fully invested is a trade and must be charged."""
        assert turnover_series([1.0, 1.0, 1.0])[0] == pytest.approx(1.0)

    def test_holding_costs_nothing(self):
        assert turnover_series([1.0, 1.0, 1.0])[1:].tolist() == [0.0, 0.0]

    def test_a_full_rotation_counts_both_legs(self):
        """Selling A to buy B trades 200% of the book, and pays on both."""
        positions = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert turnover_series(positions)[1] == pytest.approx(2.0)

    def test_flip_from_long_to_short_is_two_units(self):
        assert turnover_series([1.0, -1.0])[1] == pytest.approx(2.0)

    def test_initial_position_can_be_supplied(self):
        assert turnover_series([1.0, 1.0], initial_position=1.0)[0] == 0.0

    def test_average_turnover_annualises(self):
        positions = np.array([1.0, 0.0, 1.0, 0.0])
        daily = average_turnover(positions)
        assert average_turnover(positions, periods_per_year=252) == pytest.approx(daily * 252)

    @pytest.mark.parametrize(
        "positions,match",
        [
            (np.zeros((2, 2, 2)), "1-D or 2-D"),
            ([1.0], "at least 2"),
            ([1.0, np.nan], "non-finite"),
        ],
    )
    def test_rejects_bad_positions(self, positions, match):
        with pytest.raises(ValueError, match=match):
            turnover_series(positions)


class TestCostModels:
    def test_fixed_bps_charges_per_unit_traded(self):
        assert FixedBpsCost(bps=10.0).cost_series(np.array([1.0, 2.0])).tolist() == [
            pytest.approx(0.001),
            pytest.approx(0.002),
        ]

    def test_zero_bps_is_free(self):
        assert FixedBpsCost(bps=0.0).cost_series(np.array([5.0]))[0] == 0.0

    def test_spread_model_charges_half_the_spread_plus_commission(self):
        model = SpreadProportionalCost(spread_bps=4.0, commission_bps=0.5)
        assert model.effective_bps == pytest.approx(2.5)
        assert model.cost_series(np.array([1.0]))[0] == pytest.approx(2.5 / 10_000)

    def test_models_report_their_parameters(self):
        d = SpreadProportionalCost(spread_bps=6.0).to_dict()
        assert d["spread_bps"] == 6.0 and "effective_bps" in d

    @pytest.mark.parametrize(
        "factory,match",
        [
            (lambda: FixedBpsCost(bps=-1.0), "non-negative"),
            (lambda: SpreadProportionalCost(spread_bps=-1.0), "non-negative"),
            (lambda: SpreadProportionalCost(commission_bps=-1.0), "non-negative"),
        ],
    )
    def test_reject_negative_costs(self, factory, match):
        with pytest.raises(ValueError, match=match):
            factory()

    def test_apply_costs_subtracts_the_charge(self):
        gross = np.array([0.01, 0.01, 0.01])
        positions = np.array([1.0, 0.0, 1.0])
        net = apply_costs(gross, positions, FixedBpsCost(bps=100.0))
        # turnover is 1 every period here, and 100 bps is 1%.
        assert net.tolist() == [pytest.approx(0.0)] * 3

    def test_never_trading_costs_nothing(self):
        gross = np.array([0.01] * 5)
        net = apply_costs(gross, np.zeros(5), FixedBpsCost(bps=500.0))
        assert net.tolist() == gross.tolist()

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same time axis"):
            apply_costs(np.zeros(10), np.zeros(8), FixedBpsCost())

    def test_rejects_two_dimensional_returns(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            apply_costs(np.zeros((10, 2)), np.zeros(10), FixedBpsCost())


class TestBreakEvenCost:
    def test_closed_form_matches_a_direct_search(self):
        """The zero-Sharpe case is exact, so it must agree with brute force."""
        gen = np.random.default_rng(1)
        gross = 0.0006 + 0.01 * gen.standard_normal(1000)
        positions = gen.integers(0, 2, 1000).astype(float)
        be = break_even_cost(gross, positions).break_even_bps

        at_breakeven = sharpe_ratio(apply_costs(gross, positions, FixedBpsCost(bps=be)))
        assert at_breakeven == pytest.approx(0.0, abs=1e-9)

    def test_net_sharpe_is_positive_below_and_negative_above(self):
        gen = np.random.default_rng(2)
        gross = 0.0006 + 0.01 * gen.standard_normal(800)
        positions = gen.integers(0, 2, 800).astype(float)
        be = break_even_cost(gross, positions).break_even_bps
        below = sharpe_ratio(apply_costs(gross, positions, FixedBpsCost(bps=be * 0.5)))
        above = sharpe_ratio(apply_costs(gross, positions, FixedBpsCost(bps=be * 1.5)))
        assert below > 0 > above

    def test_high_turnover_lowers_the_break_even(self):
        """The core intuition: trading more often leaves less room for cost."""
        gen = np.random.default_rng(3)
        gross = 0.0006 + 0.01 * gen.standard_normal(1000)
        patient = np.repeat([1.0, 0.0], 500)
        frantic = np.tile([1.0, 0.0], 500).astype(float)
        assert (
            break_even_cost(gross, frantic).break_even_bps
            < break_even_cost(gross, patient).break_even_bps
        )

    def test_a_strategy_that_never_trades_has_infinite_break_even(self):
        r = break_even_cost(np.full(50, 0.001), np.zeros(50))
        assert r.break_even_bps == math.inf
        assert "never trades" in r.note

    def test_a_losing_strategy_breaks_even_at_zero(self):
        gen = np.random.default_rng(4)
        gross = -0.0005 + 0.01 * gen.standard_normal(500)
        r = break_even_cost(gross, gen.integers(0, 2, 500).astype(float))
        assert r.break_even_bps == 0.0
        assert "already fails the target" in r.note

    def test_flat_returns_give_nan(self):
        r = break_even_cost(np.full(50, 0.001), np.tile([1.0, 0.0], 25).astype(float))
        assert math.isnan(r.break_even_bps)
        assert "no variance" in r.note

    def test_non_zero_target_needs_more_headroom(self):
        gen = np.random.default_rng(5)
        gross = 0.0008 + 0.01 * gen.standard_normal(1000)
        positions = gen.integers(0, 2, 1000).astype(float)
        assert (
            break_even_cost(gross, positions, target_sharpe=0.03).break_even_bps
            < break_even_cost(gross, positions, target_sharpe=0.0).break_even_bps
        )

    def test_bisection_hits_the_requested_target(self):
        gen = np.random.default_rng(6)
        gross = 0.0008 + 0.01 * gen.standard_normal(1000)
        positions = gen.integers(0, 2, 1000).astype(float)
        target = 0.02
        be = break_even_cost(gross, positions, target_sharpe=target).break_even_bps
        got = sharpe_ratio(apply_costs(gross, positions, FixedBpsCost(bps=be)))
        assert got == pytest.approx(target, abs=1e-6)

    def test_cost_insensitive_strategy_is_reported_as_such(self):
        """Real returns but negligible trading: the bisection must give up at
        the ceiling rather than search forever."""
        gen = np.random.default_rng(15)
        gross = 0.001 + 0.01 * gen.standard_normal(200)
        barely_trades = np.tile([1e-8, 0.0], 100)
        r = break_even_cost(gross, barely_trades, target_sharpe=0.001, max_bps=100.0)
        assert r.break_even_bps == math.inf
        assert "cost-insensitive" in r.note

    def test_asset_class_supplies_the_comparison(self):
        gen = np.random.default_rng(7)
        gross = 0.0006 + 0.01 * gen.standard_normal(1000)
        positions = gen.integers(0, 2, 1000).astype(float)
        r = break_even_cost(gross, positions, asset_class="us_large_cap_etf")
        assert r.realistic_bps == TYPICAL_COST_BPS["us_large_cap_etf"]
        assert r.survives_realistic_costs is not None
        assert r.margin is not None

    def test_no_asset_class_means_no_verdict(self):
        """The tool will not invent a cost estimate for a market it was not told about."""
        gen = np.random.default_rng(8)
        r = break_even_cost(
            0.0006 + 0.01 * gen.standard_normal(500), gen.integers(0, 2, 500).astype(float)
        )
        assert r.realistic_bps is None
        assert r.survives_realistic_costs is None
        assert r.margin is None
        assert r.severity is Severity.INFO

    @pytest.mark.parametrize(
        "margin,expected",
        [(5.0, Severity.INFO), (2.0, Severity.MEDIUM), (1.2, Severity.HIGH), (0.5, Severity.CRITICAL)],
    )
    def test_severity_ladder(self, margin, expected):
        from qv.costs.breakeven import BreakEvenResult

        r = BreakEvenResult(
            break_even_bps=margin * 3.0,
            gross_sharpe=0.05,
            mean_turnover=0.5,
            annual_turnover=None,
            target_sharpe=0.0,
            asset_class="us_large_cap_etf",
            realistic_bps=(1.0, 3.0),
        )
        assert r.severity is expected

    def test_annual_turnover_is_reported_when_asked(self):
        gen = np.random.default_rng(9)
        r = break_even_cost(
            0.0006 + 0.01 * gen.standard_normal(500),
            gen.integers(0, 2, 500).astype(float),
            periods_per_year=252,
        )
        assert r.annual_turnover == pytest.approx(r.mean_turnover * 252)

    def test_to_dict_is_json_shaped(self):
        gen = np.random.default_rng(10)
        d = break_even_cost(
            0.0006 + 0.01 * gen.standard_normal(500),
            gen.integers(0, 2, 500).astype(float),
            asset_class="us_large_cap_equity",
        ).to_dict()
        assert set(d) >= {"break_even_bps", "margin", "severity", "realistic_bps"}

    def test_rejects_unknown_asset_class(self):
        with pytest.raises(ValueError, match="unknown asset_class"):
            break_even_cost(np.zeros(50) + 0.001, np.ones(50), asset_class="crypto_perps")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same time axis"):
            break_even_cost(np.zeros(10) + 0.001, np.ones(8))


class TestSharpeVsCostCurve:
    def test_curve_starts_at_the_gross_sharpe(self):
        gen = np.random.default_rng(11)
        gross = 0.0006 + 0.01 * gen.standard_normal(600)
        positions = gen.integers(0, 2, 600).astype(float)
        bps, sharpes = sharpe_vs_cost_curve(gross, positions)
        assert bps[0] == 0.0
        assert sharpes[0] == pytest.approx(sharpe_ratio(gross))

    def test_curve_is_monotonically_decreasing(self):
        gen = np.random.default_rng(12)
        gross = 0.0006 + 0.01 * gen.standard_normal(600)
        positions = gen.integers(0, 2, 600).astype(float)
        _, sharpes = sharpe_vs_cost_curve(gross, positions)
        assert np.all(np.diff(sharpes) <= 1e-12)

    def test_curve_spans_the_break_even_crossing(self):
        gen = np.random.default_rng(13)
        gross = 0.0006 + 0.01 * gen.standard_normal(600)
        positions = gen.integers(0, 2, 600).astype(float)
        _, sharpes = sharpe_vs_cost_curve(gross, positions)
        assert sharpes[0] > 0 > sharpes[-1]

    def test_respects_the_requested_range_and_resolution(self):
        gen = np.random.default_rng(14)
        bps, sharpes = sharpe_vs_cost_curve(
            0.0006 + 0.01 * gen.standard_normal(300),
            gen.integers(0, 2, 300).astype(float),
            max_bps=25.0,
            n_points=11,
        )
        assert bps.size == sharpes.size == 11
        assert bps[-1] == pytest.approx(25.0)

    def test_falls_back_to_a_default_range_when_break_even_is_infinite(self):
        bps, _ = sharpe_vs_cost_curve(np.full(50, 0.001), np.zeros(50))
        assert bps[-1] == pytest.approx(50.0)

    @pytest.mark.parametrize("kwargs,match", [({"n_points": 1}, "n_points"), ({"max_bps": 0.0}, "max_bps")])
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            sharpe_vs_cost_curve(np.zeros(50) + 0.001, np.ones(50), **kwargs)
