"""Tests for the Sharpe ratio, its standard error, and annualisation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.moments import standardised_moments
from qv.stats.sharpe import (
    analyse_sharpe,
    sharpe_ratio,
    sharpe_standard_error,
    sharpe_tstat,
    SharpeResult,
)
from qv.types import MIN_OBS_FOR_ASYMPTOTICS
from tests.conftest import make_ar1, make_returns


class TestSharpeRatio:
    def test_known_answer_by_hand(self):
        """mean 3, sample sd (ddof=1) of [1..5] is sqrt(2.5)."""
        assert sharpe_ratio([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(3.0 / math.sqrt(2.5))

    def test_risk_free_rate_shifts_the_numerator(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sharpe_ratio(x, rf_per_period=1.0) == pytest.approx(2.0 / math.sqrt(2.5))

    def test_recovers_planted_sharpe(self):
        x = make_returns(200_000, sharpe_per_period=0.05, seed=1)
        assert sharpe_ratio(x) == pytest.approx(0.05, abs=0.005)

    def test_zero_variance_is_nan_not_an_exception(self):
        assert math.isnan(sharpe_ratio([0.01] * 50))

    def test_negative_mean_gives_negative_sharpe(self):
        assert sharpe_ratio(make_returns(5000, -0.05, seed=2)) < 0

    def test_rejects_single_observation(self):
        with pytest.raises(ValueError, match="at least 2"):
            sharpe_ratio([0.01])


class TestSharpeStandardError:
    def test_reduces_to_one_over_sqrt_n_for_normal_data(self, rng):
        """With SR ~ 0, skew ~ 0 and kurtosis ~ 3 the formula collapses to
        1/sqrt(n) - the textbook result, and a useful sanity anchor."""
        x = rng.standard_normal(50_000)
        assert sharpe_standard_error(x) == pytest.approx(1 / math.sqrt(50_000), rel=0.05)

    def test_matches_formula_recomputed_independently(self, rng):
        x = 0.001 + 0.02 * rng.standard_normal(1000)
        sr = sharpe_ratio(x)
        g3, g4 = standardised_moments(x)
        expected = math.sqrt(
            (1 + 0.5 * sr**2 - g3 * sr + 0.25 * (g4 - 3) * sr**2) / len(x)
        )
        assert sharpe_standard_error(x) == pytest.approx(expected, rel=1e-12)

    def test_fat_tails_widen_the_interval(self, rng):
        """The whole point of the higher-moment terms. Both series are scaled
        to the same Sharpe so only the tails differ."""
        n = 20_000
        normal = 0.05 + rng.standard_normal(n)
        t5 = rng.standard_t(df=5, size=n)
        t5 = 0.05 * np.std(t5, ddof=1) + t5
        assert sharpe_standard_error(t5) > sharpe_standard_error(normal)

    def test_falls_back_when_variance_term_goes_negative(self):
        """Extreme skew against a large Sharpe can make the bracket negative;
        the Gaussian term is used rather than returning nan."""
        x = np.concatenate([np.full(60, 1.0), np.array([-40.0])])
        se = sharpe_standard_error(x)
        assert math.isfinite(se) and se > 0

    def test_zero_variance_is_nan(self):
        assert math.isnan(sharpe_standard_error([0.01] * 50))

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            sharpe_standard_error([0.1, 0.2, 0.3])


class TestSharpeTstat:
    def test_is_ratio_of_sharpe_to_se(self, rng):
        x = 0.001 + 0.01 * rng.standard_normal(500)
        assert sharpe_tstat(x) == pytest.approx(sharpe_ratio(x) / sharpe_standard_error(x))

    def test_true_null_gives_small_tstat_on_average(self):
        """Under a zero-edge null the t-stat should sit near zero."""
        stats = [sharpe_tstat(make_returns(500, 0.0, seed=s)) for s in range(60)]
        assert abs(float(np.mean(stats))) < 0.5

    def test_zero_variance_is_nan(self):
        assert math.isnan(sharpe_tstat([0.01] * 50))


class TestAnalyseSharpe:
    def test_naive_annualisation_is_sqrt_q_times_periodic(self, iid_returns):
        r = analyse_sharpe(iid_returns, 252)
        assert r.annualised_naive.value == pytest.approx(r.periodic.value * math.sqrt(252))
        assert r.factor_naive == pytest.approx(math.sqrt(252))

    def test_persistent_returns_get_a_real_haircut(self):
        """The headline correction, end to end."""
        r = analyse_sharpe(make_ar1(5000, 0.5, seed=4, mean=0.3), 12)
        assert r.autocorrelation_haircut > 0.2
        assert r.annualised_adjusted.value < r.annualised_naive.value

    def test_iid_returns_get_almost_no_haircut(self, iid_returns):
        assert abs(analyse_sharpe(iid_returns, 252).autocorrelation_haircut) < 0.15

    def test_confidence_interval_widens_with_confidence(self, iid_returns):
        narrow = analyse_sharpe(iid_returns, 252, confidence=0.80).annualised_adjusted
        wide = analyse_sharpe(iid_returns, 252, confidence=0.99).annualised_adjusted
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_interval_is_centred_on_the_point_estimate(self, iid_returns):
        e = analyse_sharpe(iid_returns, 252).annualised_adjusted
        assert (e.ci_low + e.ci_high) / 2 == pytest.approx(e.value)

    def test_small_sample_is_flagged_unreliable(self):
        r = analyse_sharpe(make_returns(20, 0.3, seed=5), 4)
        assert r.periodic.reliable is False
        assert "below the 30" in r.periodic.note

    def test_sample_at_threshold_is_reliable(self):
        r = analyse_sharpe(make_returns(MIN_OBS_FOR_ASYMPTOTICS, 0.3, seed=6), 4)
        assert r.periodic.reliable is True
        assert r.periodic.note is None

    def test_falls_back_when_lo_adjustment_is_undefined(self, monkeypatch, iid_returns):
        """If the adjustment cannot be computed the result must degrade to
        sqrt(q) with a note, not blow up or emit a nan."""

        def boom(*_args, **_kwargs):
            raise ValueError("non-positive variance")

        monkeypatch.setattr("qv.stats.sharpe.lo_annualisation_factor", boom)
        r = analyse_sharpe(iid_returns, 12)
        assert r.factor_lo == r.factor_naive
        assert r.autocorrelation_haircut == 0.0
        assert "fell back to sqrt(q)" in r.autocorrelation_note

    def test_notes_truncation_when_q_exceeds_estimable_lags(self, iid_returns):
        """Annualising 1000 daily observations to 252 periods needs 251
        autocorrelations it cannot estimate. Say so rather than pretending."""
        r = analyse_sharpe(iid_returns, 252)
        assert "can only speak to" in r.autocorrelation_note

    def test_no_truncation_note_when_the_sample_supports_q(self):
        r = analyse_sharpe(make_returns(50_000, 0.05, seed=12), 12)
        assert r.autocorrelation_note is None

    def test_moments_are_carried_through(self, iid_returns):
        r = analyse_sharpe(iid_returns, 252)
        g3, g4 = standardised_moments(iid_returns)
        assert (r.skewness, r.kurtosis) == (g3, g4)

    def test_to_dict_is_json_shaped(self, iid_returns):
        d = analyse_sharpe(iid_returns, 252).to_dict()
        assert set(d) >= {"periodic", "annualised_naive", "annualised_adjusted", "tstat"}
        assert set(d["periodic"]) >= {"value", "ci_low", "ci_high", "reliable"}

    def test_haircut_is_nan_without_a_naive_factor(self, iid_returns):
        r = analyse_sharpe(iid_returns, 252)
        broken = SharpeResult(**{**r.__dict__, "factor_naive": 0.0})
        assert math.isnan(broken.autocorrelation_haircut)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"periods_per_year": 0}, "at least 1"),
            ({"periods_per_year": 252, "confidence": 0.0}, "between 0 and 1"),
            ({"periods_per_year": 252, "confidence": 1.0}, "between 0 and 1"),
        ],
    )
    def test_rejects_bad_arguments(self, iid_returns, kwargs, match):
        with pytest.raises(ValueError, match=match):
            analyse_sharpe(iid_returns, **kwargs)
