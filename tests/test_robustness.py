"""Tests for the robustness suite: subsample, regimes, randomisation, parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.robustness.parameters import plateau_ratio
from qv.robustness.randomization import matched_exposure_test, sign_flip_test
from qv.robustness.regimes import trailing_volatility, volatility_regime_split
from qv.robustness.subsample import rolling_origin_sensitivity, top_day_dependence
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity
from tests.conftest import make_returns


class TestTopDayDependence:
    @pytest.mark.parametrize("seed", range(5))
    def test_ordinary_normal_returns_are_not_flagged(self, seed):
        """The baseline that makes this test usable at all.

        A normal series concentrates its profit in its best days purely as a
        function of its Sharpe - the best 1% of days routinely account for more
        than 100% of the profit of a low-Sharpe strategy. Flagging that would
        fire on nearly every real backtest, so only concentration *beyond* the
        normal baseline counts.
        """
        r = top_day_dependence(make_returns(1000, 0.08, seed=seed))
        assert r.worst_concentration_ratio == pytest.approx(1.0, abs=0.4)
        assert not r.abnormally_concentrated
        assert r.severity is Severity.INFO

    def test_a_lottery_ticket_is_caught(self):
        """All the profit in a handful of days, noise the rest of the time."""
        gen = np.random.default_rng(2)
        x = 0.001 * gen.standard_normal(1000)
        x[gen.choice(1000, 8, replace=False)] += 0.25
        r = top_day_dependence(x)
        assert r.full_sharpe > 0
        assert r.abnormally_concentrated
        assert r.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_reports_the_share_of_profit_in_the_best_days(self):
        gen = np.random.default_rng(3)
        x = 0.0001 * gen.standard_normal(1000)
        x[:5] += 1.0
        r = top_day_dependence(x)
        assert r.total_return_shares[0] > 0.9

    def test_expected_share_falls_as_the_sharpe_rises(self):
        from qv.robustness.subsample import expected_profit_share

        assert expected_profit_share(0.30, 0.01) < expected_profit_share(0.03, 0.01)

    def test_expected_share_is_nan_for_a_non_positive_sharpe(self):
        from qv.robustness.subsample import expected_profit_share

        assert math.isnan(expected_profit_share(0.0, 0.01))
        assert math.isnan(expected_profit_share(-0.5, 0.01))

    def test_dropping_more_days_hurts_more(self):
        r = top_day_dependence(make_returns(1000, 0.05, seed=4))
        assert r.sharpe_after_dropping_top_5pct < r.sharpe_after_dropping_top_1pct
        assert r.sharpe_after_dropping_top_1pct < r.full_sharpe

    def test_custom_drop_fractions(self):
        r = top_day_dependence(make_returns(500, 0.05, seed=5), drop_fractions=(0.02, 0.10))
        assert r.drop_fractions == (0.02, 0.10)
        with pytest.raises(KeyError):
            r.sharpe_after_dropping_top_1pct

    def test_negative_strategy_is_not_flagged(self):
        """A strategy that already loses money has no top-day problem to find."""
        assert top_day_dependence(make_returns(500, -0.05, seed=6)).severity is Severity.INFO

    def test_to_dict_is_json_shaped(self):
        d = top_day_dependence(make_returns(500, 0.05, seed=7)).to_dict()
        assert set(d) >= {"full_sharpe", "sharpe_after_dropping_top_5pct", "severity"}

    @pytest.mark.parametrize(
        "args,kwargs,match",
        [
            ((np.zeros(10),), {}, "at least 20"),
            ((make_returns(100, 0.05, seed=8),), {"drop_fractions": (1.5,)}, "between 0 and 1"),
            ((make_returns(100, 0.05, seed=8),), {"drop_fractions": (0.0,)}, "between 0 and 1"),
            ((make_returns(20, 0.05, seed=8),), {"drop_fractions": (0.9,)}, "too few"),
        ],
    )
    def test_rejects_bad_input(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            top_day_dependence(*args, **kwargs)


class TestRollingOrigin:
    def test_a_stable_edge_is_sign_stable(self):
        r = rolling_origin_sensitivity(make_returns(2000, 0.10, seed=1))
        assert r.sign_stable
        assert r.fraction_positive == 1.0
        assert r.severity is Severity.INFO

    def test_an_edge_confined_to_the_start_is_caught(self):
        """Profits only in the first quarter: start later and it disappears."""
        x = make_returns(2000, 0.0, seed=2)
        x[:500] += 0.02
        r = rolling_origin_sensitivity(x)
        assert not r.sign_stable
        assert r.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)

    def test_start_dates_advance_but_stop_before_the_end(self):
        r = rolling_origin_sensitivity(make_returns(1000, 0.05, seed=3), max_start_fraction=0.5)
        assert r.start_indices[0] == 0
        assert max(r.start_indices) <= 500
        assert list(r.start_indices) == sorted(r.start_indices)

    def test_first_window_is_the_full_sample(self):
        x = make_returns(1000, 0.05, seed=4)
        r = rolling_origin_sensitivity(x)
        assert r.sharpes[0] == pytest.approx(sharpe_ratio(x))
        assert r.full_sharpe == pytest.approx(sharpe_ratio(x))

    def test_spread_summarises_the_range(self):
        r = rolling_origin_sensitivity(make_returns(1000, 0.05, seed=5))
        assert r.spread == pytest.approx(r.max_sharpe - r.min_sharpe)

    def test_every_window_keeps_enough_data(self):
        r = rolling_origin_sensitivity(make_returns(200, 0.05, seed=6), min_obs=50)
        assert max(r.start_indices) <= 150

    def test_to_dict_is_json_shaped(self):
        d = rolling_origin_sensitivity(make_returns(500, 0.05, seed=7)).to_dict()
        assert set(d) >= {"sharpes", "sign_stable", "spread", "severity"}

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_starts": 1}, "n_starts"),
            ({"max_start_fraction": 1.0}, "max_start_fraction"),
            ({"max_start_fraction": -0.1}, "max_start_fraction"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            rolling_origin_sensitivity(make_returns(500, 0.05, seed=8), **kwargs)

    def test_rejects_too_short_a_sample(self):
        with pytest.raises(ValueError, match="needs more than"):
            rolling_origin_sensitivity(make_returns(20, 0.05, seed=9))


class TestTrailingVolatility:
    def test_uses_only_past_data(self):
        """The defining property. A spike at t must not affect the volatility
        label at t, or the regime split becomes look-ahead."""
        x = np.zeros(100)
        x[50] = 100.0
        vol = trailing_volatility(x, window=10, min_periods=2)
        assert vol[50] == pytest.approx(0.0)
        assert vol[51] > 0

    def test_leading_values_are_nan(self):
        vol = trailing_volatility(np.arange(50.0), window=10)
        assert np.all(np.isnan(vol[:10]))
        assert np.all(np.isfinite(vol[10:]))

    def test_tracks_a_volatility_change(self):
        gen = np.random.default_rng(1)
        x = np.concatenate([0.001 * gen.standard_normal(300), 0.05 * gen.standard_normal(300)])
        vol = trailing_volatility(x, window=50)
        assert np.nanmean(vol[100:250]) < np.nanmean(vol[400:])

    @pytest.mark.parametrize("kwargs,match", [({"window": 1}, "window"), ({"min_periods": 1}, "min_periods")])
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            trailing_volatility(np.arange(50.0), **kwargs)


class TestVolatilityRegimeSplit:
    def test_a_stable_edge_is_sign_stable_across_regimes(self):
        r = volatility_regime_split(make_returns(2000, 0.08, seed=1), window=50)
        assert r.sign_stable
        assert r.severity is Severity.INFO

    def test_an_edge_confined_to_calm_markets_is_caught(self):
        """Constructed so the edge exists only when trailing volatility is low."""
        gen = np.random.default_rng(2)
        calm = 0.002 + 0.003 * gen.standard_normal(700)
        wild = -0.002 + 0.03 * gen.standard_normal(700)
        r = volatility_regime_split(np.concatenate([calm, wild]), window=50)
        assert not r.sign_stable
        assert r.concentrated_in_one_regime
        assert r.severity is Severity.HIGH

    def test_regimes_are_balanced_by_construction(self):
        r = volatility_regime_split(make_returns(1000, 0.05, seed=3), window=50)
        assert abs(r.counts[0] - r.counts[1]) <= 1

    def test_quantile_shifts_the_balance(self):
        r = volatility_regime_split(make_returns(1000, 0.05, seed=4), window=50, quantile=0.75)
        assert r.counts[0] > r.counts[1]

    def test_accepts_an_external_volatility_series(self):
        """Classifying by market volatility rather than the strategy's own."""
        x = make_returns(600, 0.05, seed=5)
        external = np.linspace(0.01, 0.05, 600)
        r = volatility_regime_split(x, volatility_series=external)
        assert r.n_classified == 600
        assert r.note is None

    def test_unclassifiable_leading_observations_are_reported(self):
        r = volatility_regime_split(make_returns(1000, 0.05, seed=6), window=50)
        assert r.n_classified < r.n_obs
        assert "too little history" in r.note

    def test_sharpe_in_looks_up_by_label(self):
        r = volatility_regime_split(make_returns(1000, 0.05, seed=7), window=50)
        assert r.sharpe_in("low volatility") == r.sharpes[0]
        with pytest.raises(KeyError):
            r.sharpe_in("medium volatility")

    def test_to_dict_is_json_shaped(self):
        d = volatility_regime_split(make_returns(1000, 0.05, seed=8), window=50).to_dict()
        assert set(d) >= {"labels", "sharpes", "sign_stable", "severity", "threshold"}

    def test_rejects_a_mismatched_volatility_series(self):
        with pytest.raises(ValueError, match="same time axis"):
            volatility_regime_split(make_returns(500, 0.05, seed=9), volatility_series=np.ones(400))

    def test_rejects_bad_quantile(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            volatility_regime_split(make_returns(500, 0.05, seed=10), quantile=0.0)

    def test_refuses_when_too_little_can_be_classified(self):
        with pytest.raises(ValueError, match="too few"):
            volatility_regime_split(make_returns(60, 0.05, seed=11), window=50)


class TestSignFlipTest:
    def test_a_real_edge_beats_the_flipped_null(self):
        r = sign_flip_test(make_returns(1500, 0.12, seed=1), n_sims=500)
        assert r.p_value < 0.05
        assert r.significant_at_5pct
        assert r.severity in (Severity.INFO, Severity.LOW)

    def test_pure_noise_does_not(self):
        r = sign_flip_test(make_returns(1500, 0.0, seed=2), n_sims=500)
        assert r.p_value > 0.05

    def test_the_null_is_centred_near_zero(self):
        r = sign_flip_test(make_returns(1000, 0.05, seed=3), n_sims=500)
        assert r.null_mean == pytest.approx(0.0, abs=0.02)

    def test_observed_is_the_unflipped_sharpe(self):
        x = make_returns(500, 0.05, seed=4)
        assert sign_flip_test(x, n_sims=50).observed == pytest.approx(sharpe_ratio(x))

    def test_is_reproducible(self):
        x = make_returns(500, 0.05, seed=5)
        a = sign_flip_test(x, n_sims=100, seed=3)
        b = sign_flip_test(x, n_sims=100, seed=3)
        assert np.array_equal(a.null_distribution, b.null_distribution)

    def test_p_value_is_never_zero(self):
        r = sign_flip_test(make_returns(500, 2.0, seed=6), n_sims=100)
        assert r.p_value >= 1 / 101

    def test_to_dict_is_json_shaped(self):
        d = sign_flip_test(make_returns(500, 0.05, seed=7), n_sims=50).to_dict()
        assert set(d) >= {"test", "observed", "p_value", "severity"}
        assert "null_distribution" not in d

    @pytest.mark.parametrize(
        "args,kwargs,match",
        [
            ((np.zeros(10),), {}, "at least 20"),
            ((make_returns(100, 0.05, seed=8),), {"n_sims": 0}, "n_sims"),
            ((make_returns(100, 0.05, seed=8),), {"block_length": 0}, "block_length"),
        ],
    )
    def test_rejects_bad_input(self, args, kwargs, match):
        with pytest.raises(ValueError, match=match):
            sign_flip_test(*args, **kwargs)


class TestMatchedExposureTest:
    def test_levered_beta_is_exposed(self):
        """A strategy that is simply long a rising market must not beat random
        timing with the same average exposure. This is the test that catches it."""
        gen = np.random.default_rng(1)
        market = 0.0005 + 0.01 * gen.standard_normal(1500)
        always_long = np.ones(1500)
        r = matched_exposure_test(market, always_long, n_sims=300)
        assert r.p_value > 0.20
        assert r.severity is Severity.HIGH

    def test_genuine_timing_skill_is_rewarded(self):
        """A position series that actually predicts next-period returns should
        beat every reshuffling of itself."""
        gen = np.random.default_rng(2)
        market = 0.01 * gen.standard_normal(1500)
        clairvoyant = np.where(market > 0, 1.0, 0.0)
        r = matched_exposure_test(market, clairvoyant, n_sims=300)
        assert r.p_value < 0.01
        assert r.significant_at_5pct

    def test_average_exposure_is_preserved(self):
        gen = np.random.default_rng(3)
        pos = gen.integers(0, 2, 600).astype(float)
        r = matched_exposure_test(0.01 * gen.standard_normal(600), pos, n_sims=50)
        assert f"{float(np.mean(pos)):.3f}" in r.description

    def test_is_reproducible(self):
        gen = np.random.default_rng(4)
        market, pos = 0.01 * gen.standard_normal(400), gen.integers(0, 2, 400).astype(float)
        a = matched_exposure_test(market, pos, n_sims=80, seed=1)
        b = matched_exposure_test(market, pos, n_sims=80, seed=1)
        assert np.array_equal(a.null_distribution, b.null_distribution)

    def test_to_dict_names_the_test(self):
        gen = np.random.default_rng(5)
        d = matched_exposure_test(
            0.01 * gen.standard_normal(400), gen.integers(0, 2, 400).astype(float), n_sims=50
        ).to_dict()
        assert d["test"] == "matched-exposure random entry"

    @pytest.mark.parametrize(
        "market,pos,kwargs,match",
        [
            (np.zeros(100), np.zeros((100, 2)), {}, "must be 1-D"),
            (np.zeros(100), np.zeros(80), {}, "same time axis"),
            (np.zeros(10), np.zeros(10), {}, "at least 20"),
            (np.zeros(100), np.zeros(100), {"n_sims": 0}, "n_sims"),
        ],
    )
    def test_rejects_bad_input(self, market, pos, kwargs, match):
        with pytest.raises(ValueError, match=match):
            matched_exposure_test(market, pos, **kwargs)


class TestPlateauRatio:
    def test_a_broad_plateau_scores_near_one(self):
        grid = np.array([0.5, 0.9, 1.0, 0.95, 0.6])
        r = plateau_ratio(grid)
        assert r.ratio > 0.9
        assert not r.is_spike
        assert r.severity is Severity.INFO

    def test_a_spike_on_a_cliff_is_caught(self):
        grid = np.array([0.1, 0.05, 2.0, 0.05, 0.1])
        r = plateau_ratio(grid)
        assert r.ratio < 0.1
        assert r.is_spike
        assert r.severity is Severity.CRITICAL

    def test_defaults_to_the_grid_maximum(self):
        grid = np.array([0.1, 0.4, 0.2])
        assert plateau_ratio(grid).chosen_index == (1,)
        assert plateau_ratio(grid).is_grid_maximum

    def test_honours_an_explicit_choice(self):
        grid = np.array([0.1, 0.4, 0.2])
        r = plateau_ratio(grid, chosen_index=0)
        assert r.chosen_index == (0,)
        assert not r.is_grid_maximum

    def test_works_on_a_two_dimensional_grid(self):
        grid = np.array([[0.1, 0.2, 0.1], [0.2, 1.0, 0.2], [0.1, 0.2, 0.1]])
        r = plateau_ratio(grid)
        assert r.chosen_index == (1, 1)
        assert r.n_neighbours == 8
        assert r.is_spike

    def test_larger_radius_pulls_in_more_neighbours(self):
        grid = np.arange(25.0).reshape(5, 5)
        assert plateau_ratio(grid, chosen_index=(2, 2), radius=2).n_neighbours == 24

    def test_edge_points_have_fewer_neighbours(self):
        grid = np.array([[1.0, 0.5], [0.5, 0.25]])
        assert plateau_ratio(grid, chosen_index=(0, 0)).n_neighbours == 3

    def test_ignores_non_finite_neighbours(self):
        grid = np.array([np.nan, 0.9, 1.0, 0.95, np.nan])
        r = plateau_ratio(grid, chosen_index=2)
        assert r.n_neighbours == 2

    def test_ratio_is_nan_for_a_non_positive_peak(self):
        r = plateau_ratio(np.array([-1.0, -0.5, -2.0]), chosen_index=1)
        assert math.isnan(r.ratio)
        assert not r.is_spike
        assert r.severity is Severity.INFO

    def test_to_dict_is_json_shaped(self):
        d = plateau_ratio(np.array([0.5, 1.0, 0.5])).to_dict()
        assert set(d) >= {"ratio", "is_spike", "chosen_index", "severity"}

    @pytest.mark.parametrize(
        "grid,kwargs,match",
        [
            (np.array(1.0), {}, "at least one dimension"),
            (np.array([1.0, 2.0]), {}, "at least 3 points"),
            (np.arange(5.0), {"radius": 0}, "radius"),
            (np.full(5, np.nan), {}, "no finite values"),
            (np.arange(5.0), {"chosen_index": (1, 1)}, "1 entries|2 entries"),
            (np.arange(5.0), {"chosen_index": 9}, "outside axis"),
        ],
    )
    def test_rejects_bad_input(self, grid, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plateau_ratio(grid, **kwargs)
