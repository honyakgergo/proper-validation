"""Tests for sample moments and Lo autocorrelation-adjusted annualisation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as ss

from qv.stats.moments import (
    as_returns_array,
    autocorrelations,
    kurtosis,
    lo_annualisation_factor,
    naive_annualisation_factor,
    skewness,
    standardised_moments,
)
from tests.conftest import make_ar1


class TestAsReturnsArray:
    def test_accepts_list(self):
        assert as_returns_array([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]

    def test_drops_nan_and_inf(self):
        out = as_returns_array([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0])
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_accepts_pandas_series(self):
        pd = pytest.importorskip("pandas")
        out = as_returns_array(pd.Series([1.0, 2.0, np.nan]))
        assert out.tolist() == [1.0, 2.0]

    def test_scalar_becomes_length_one(self):
        assert as_returns_array(3.0).shape == (1,)

    def test_rejects_2d(self):
        # A silently flattened trial matrix yields a plausible, wrong Sharpe.
        with pytest.raises(ValueError, match="1-D"):
            as_returns_array(np.zeros((10, 3)))


class TestSkewnessKurtosis:
    def test_matches_scipy_conventions(self, rng):
        x = rng.standard_normal(5000)
        assert skewness(x) == pytest.approx(float(ss.skew(x)), rel=1e-12)
        assert kurtosis(x) == pytest.approx(float(ss.kurtosis(x, fisher=False)), rel=1e-12)

    def test_kurtosis_is_raw_not_excess(self, rng):
        """Normal data must give ~3.0. Returning excess kurtosis here would
        silently shrink every Sharpe standard error in the package."""
        assert kurtosis(rng.standard_normal(200_000)) == pytest.approx(3.0, abs=0.1)

    def test_symmetric_data_has_zero_skew(self):
        assert skewness([-2.0, -1.0, 0.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-15)

    def test_right_skewed_data_is_positive(self, rng):
        assert skewness(rng.exponential(size=20_000)) > 1.5

    def test_fat_tails_exceed_three(self, rng):
        assert kurtosis(rng.standard_t(df=5, size=50_000)) > 4.0

    def test_constant_series_degenerates_gracefully(self):
        assert skewness([2.0] * 10) == 0.0
        assert kurtosis([2.0] * 10) == 3.0

    def test_standardised_moments_agrees_with_parts(self, rng):
        x = rng.standard_normal(500)
        assert standardised_moments(x) == (skewness(x), kurtosis(x))

    @pytest.mark.parametrize("fn,need", [(skewness, 3), (kurtosis, 4)])
    def test_rejects_too_few_observations(self, fn, need):
        with pytest.raises(ValueError, match="at least"):
            fn([1.0] * (need - 1))


class TestAutocorrelations:
    def test_known_answer_by_hand(self):
        """x = [1,2,3,4,5]: centred [-2,-1,0,1,2], denominator 10.

        rho_1 = (2 + 0 + 0 + 2)/10 = 0.4;  rho_2 = (0 - 1 + 0)/10 = -0.1
        """
        rho = autocorrelations([1.0, 2.0, 3.0, 4.0, 5.0], 2)
        assert rho[0] == pytest.approx(0.4)
        assert rho[1] == pytest.approx(-0.1)

    def test_recovers_ar1_coefficient(self):
        rho = autocorrelations(make_ar1(50_000, 0.6, seed=3), 2)
        assert rho[0] == pytest.approx(0.6, abs=0.02)
        assert rho[1] == pytest.approx(0.36, abs=0.03)

    def test_iid_is_near_zero(self, rng):
        assert np.all(np.abs(autocorrelations(rng.standard_normal(20_000), 5)) < 0.03)

    def test_zero_lag_returns_empty(self):
        assert autocorrelations([1.0, 2.0, 3.0], 0).size == 0

    def test_constant_series_gives_zeros(self):
        assert autocorrelations([5.0] * 10, 3).tolist() == [0.0, 0.0, 0.0]

    def test_rejects_negative_lag(self):
        with pytest.raises(ValueError, match="non-negative"):
            autocorrelations([1.0, 2.0, 3.0], -1)

    def test_rejects_lag_beyond_sample(self):
        with pytest.raises(ValueError, match="more than"):
            autocorrelations([1.0, 2.0, 3.0], 3)


class TestLoAnnualisation:
    def test_known_answer_by_hand(self):
        """Continuing the [1..5] example with q=3.

        denom = 3 + 2*[2*0.4 + 1*(-0.1)] = 4.4, so eta = 3/sqrt(4.4).
        """
        expected = 3.0 / math.sqrt(4.4)
        assert lo_annualisation_factor([1.0, 2.0, 3.0, 4.0, 5.0], 3) == pytest.approx(expected)

    def test_collapses_to_sqrt_q_without_autocorrelation(self, rng):
        x = rng.standard_normal(100_000)
        assert lo_annualisation_factor(x, 12) == pytest.approx(math.sqrt(12), rel=0.02)

    def test_matches_ar1_theory(self):
        """For AR(1), rho_k = phi**k, so eta(q) has a closed form."""
        phi, q = 0.5, 12
        rho = np.array([phi**k for k in range(1, q)])
        weights = np.array([q - k for k in range(1, q)], dtype=float)
        expected = q / math.sqrt(q + 2 * float(weights @ rho))
        got = lo_annualisation_factor(make_ar1(100_000, phi, seed=5), q)
        assert got == pytest.approx(expected, rel=0.03)

    def test_positive_autocorrelation_shrinks_the_factor(self):
        """The headline correction: persistence makes sqrt(q) too generous."""
        x = make_ar1(20_000, 0.5, seed=7)
        assert lo_annualisation_factor(x, 12) < 0.8 * naive_annualisation_factor(12)

    def test_negative_autocorrelation_raises_the_factor(self):
        x = make_ar1(20_000, -0.4, seed=8)
        assert lo_annualisation_factor(x, 12) > naive_annualisation_factor(12)

    def test_q_of_one_is_identity(self, rng):
        assert lo_annualisation_factor(rng.standard_normal(100), 1) == 1.0

    def test_caps_lags_on_short_samples(self):
        """q=252 with 40 observations must not demand 251 autocorrelations."""
        x = make_ar1(40, 0.2, seed=9)
        assert math.isfinite(lo_annualisation_factor(x, 252))

    def test_max_lag_argument_is_respected(self, rng):
        x = rng.standard_normal(500)
        capped = lo_annualisation_factor(x, 12, max_lag=1)
        uncapped = lo_annualisation_factor(x, 12)
        assert capped != uncapped

    def test_raises_on_non_positive_variance(self):
        """Truncating the lag sum can drive it negative, and there is no
        meaningful square root of a negative variance.

        An untruncated autocovariance sequence is positive semi-definite, so
        the full sum stays non-negative. Truncation breaks that guarantee -
        which is exactly what happens in production when q exceeds the number
        of estimable lags. Here: alternating data has rho_1 near -1, so with
        q=3 but only one lag available the sum is 3 + 2*2*(-0.99) < 0.
        """
        x = np.array([1.0, -1.0] * 50)
        with pytest.raises(ValueError, match="non-positive"):
            lo_annualisation_factor(x, 3, max_lag=1)

    def test_alternating_series_is_fine_when_untruncated(self):
        """The same data with the full lag sum is well defined - the guard
        must not fire on legitimate strong negative autocorrelation."""
        x = np.array([1.0, -1.0] * 50)
        assert lo_annualisation_factor(x, 4) > 0

    def test_rejects_q_below_one(self, rng):
        with pytest.raises(ValueError, match="at least 1"):
            lo_annualisation_factor(rng.standard_normal(50), 0)


def test_naive_factor_is_sqrt_q():
    assert naive_annualisation_factor(252) == pytest.approx(math.sqrt(252))
    assert naive_annualisation_factor(1) == 1.0
    with pytest.raises(ValueError, match="at least 1"):
        naive_annualisation_factor(0)
