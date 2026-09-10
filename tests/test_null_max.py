"""Tests for the empirical max-Sharpe null."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.deflated import expected_max_sharpe
from qv.stats.null_max import empirical_max_sharpe_null, gaussian_max_sharpe_null


@pytest.fixture
def independent_trials() -> np.ndarray:
    return 0.01 * np.random.default_rng(99).standard_normal((750, 100))


def _factor_trials(n_factors: int, seed: int = 5) -> np.ndarray:
    """A parameter grid: 100 trials spanned by ``n_factors`` common drivers."""
    gen = np.random.default_rng(seed)
    base = gen.standard_normal((750, n_factors))
    loadings = gen.standard_normal((n_factors, 100))
    return 0.01 * (base @ loadings + 0.25 * gen.standard_normal((750, 100)))


@pytest.fixture
def correlated_trials() -> np.ndarray:
    return _factor_trials(25)


class TestGaussianNull:
    def test_reproduces_the_closed_form(self):
        """The simulated mean maximum must match the analytic expectation, or
        one of the two implementations is wrong."""
        r = gaussian_max_sharpe_null(0.1, n_trials=200, trial_sharpe_std=0.035, n_sims=20_000)
        assert r.empirical_expected_max == pytest.approx(r.analytic_expected_max, rel=0.03)
        assert not r.analytic_disagrees

    @pytest.mark.parametrize("n_trials", [5, 50, 500])
    def test_agreement_holds_across_trial_counts(self, n_trials):
        r = gaussian_max_sharpe_null(0.1, n_trials, 1.0, n_sims=20_000)
        assert r.empirical_expected_max == pytest.approx(expected_max_sharpe(n_trials), rel=0.05)

    def test_says_it_cannot_validate_its_own_assumptions(self):
        r = gaussian_max_sharpe_null(0.1, 50, 0.05)
        assert "cannot validate them" in r.note

    def test_is_reproducible(self):
        a = gaussian_max_sharpe_null(0.1, 50, 0.05, n_sims=500, seed=3)
        b = gaussian_max_sharpe_null(0.1, 50, 0.05, n_sims=500, seed=3)
        assert np.array_equal(a.null_distribution, b.null_distribution)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_trials": 0}, "at least 1"),
            ({"n_trials": 5, "trial_sharpe_std": -1.0}, "non-negative"),
            ({"n_trials": 5, "trial_sharpe_std": 1.0, "n_sims": 0}, "n_sims"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        kwargs.setdefault("trial_sharpe_std", 1.0)
        with pytest.raises(ValueError, match=match):
            gaussian_max_sharpe_null(0.1, **kwargs)


class TestEmpiricalNull:
    def test_agrees_with_the_closed_form_on_independent_trials(self, independent_trials):
        """When the analytic assumptions hold, the two routes must agree - that
        is what makes disagreement elsewhere informative."""
        r = empirical_max_sharpe_null(independent_trials, n_sims=600)
        assert not r.analytic_disagrees
        assert r.disagreement_direction is None

    def test_detects_correlated_trials(self, correlated_trials):
        """The failure the closed form cannot see: 100 trials that are not 100
        independent bets."""
        r = empirical_max_sharpe_null(correlated_trials, n_sims=600)
        assert r.analytic_disagrees
        assert r.agreement_ratio < 0.9
        assert "over-penalises" in r.disagreement_direction

    @pytest.mark.parametrize("n_factors,expect_disagreement", [(3, False), (25, True)])
    def test_mild_correlation_does_not_trip_the_flag(self, n_factors, expect_disagreement):
        """The closed form is partly self-correcting, and the flag respects
        that rather than crying wolf.

        ``SR_0`` scales by the *measured* dispersion of trial Sharpes, which
        already shrinks when trials cluster. So a grid spanned by only three
        factors produces analytic and empirical maxima that agree to about 1%.
        The residual gap that this flag catches is the independence assumption
        inside the extreme-value term, which only bites once there are enough
        distinct drivers for ``n_trials`` to badly overstate the effective
        number of bets.
        """
        r = empirical_max_sharpe_null(_factor_trials(n_factors), n_sims=600)
        assert r.analytic_disagrees is expect_disagreement

    def test_defaults_to_the_best_trial_as_observed(self, independent_trials):
        from qv.stats.moments import columnwise_sharpe

        r = empirical_max_sharpe_null(independent_trials, n_sims=100)
        assert r.observed_sharpe == pytest.approx(
            float(np.nanmax(columnwise_sharpe(independent_trials)))
        )

    def test_mined_noise_winner_is_unremarkable_against_the_null(self, independent_trials):
        """The whole point: the best of 100 noise trials is exactly what a
        search over 100 noise trials is expected to produce."""
        r = empirical_max_sharpe_null(independent_trials, n_sims=800)
        assert 0.1 < r.percentile < 0.9
        assert r.p_value > 0.10

    def test_a_genuine_edge_beats_the_null(self, independent_trials):
        m = independent_trials.copy()
        m[:, 0] += 0.004
        r = empirical_max_sharpe_null(m, n_sims=800)
        assert r.percentile > 0.99
        assert r.p_value < 0.01

    def test_p_value_is_never_exactly_zero(self, independent_trials):
        """No finite simulation can rule out a larger draw."""
        r = empirical_max_sharpe_null(independent_trials, observed_sharpe=99.0, n_sims=200)
        assert 0 < r.p_value <= 1 / 201

    def test_percentile_and_p_value_are_complementary(self, independent_trials):
        r = empirical_max_sharpe_null(independent_trials, n_sims=400)
        assert r.percentile + r.p_value == pytest.approx(1.0, abs=0.01)

    def test_imposes_the_null_by_demeaning(self):
        """A matrix where every trial has a large positive drift must still
        produce a null distribution centred near zero."""
        gen = np.random.default_rng(7)
        m = 0.01 * gen.standard_normal((500, 30)) + 0.02
        r = empirical_max_sharpe_null(m, n_sims=400)
        assert r.empirical_expected_max < 0.5 * r.observed_sharpe

    def test_reports_the_block_length(self, independent_trials):
        r = empirical_max_sharpe_null(independent_trials, n_sims=100)
        assert r.block_length >= 1.0

    def test_explicit_block_length_is_honoured(self, independent_trials):
        r = empirical_max_sharpe_null(independent_trials, n_sims=100, block_length=12.0)
        assert r.block_length == 12.0

    def test_is_reproducible(self, independent_trials):
        a = empirical_max_sharpe_null(independent_trials, n_sims=200, seed=11)
        b = empirical_max_sharpe_null(independent_trials, n_sims=200, seed=11)
        assert np.array_equal(a.null_distribution, b.null_distribution)

    def test_agreement_ratio_is_nan_without_an_analytic_baseline(self):
        from qv.stats.null_max import MaxSharpeNullResult

        r = MaxSharpeNullResult(0.1, np.array([0.05, 0.06]), 0.0, 10, 2, "x")
        assert math.isnan(r.agreement_ratio)
        assert not r.analytic_disagrees

    def test_to_dict_is_json_shaped(self, independent_trials):
        d = empirical_max_sharpe_null(independent_trials, n_sims=100).to_dict()
        assert set(d) >= {"p_value", "percentile", "agreement_ratio", "analytic_disagrees"}
        assert "null_distribution" not in d

    @pytest.mark.parametrize(
        "matrix,match",
        [
            (np.zeros(50), "2-D"),
            (np.zeros((3, 5)), "at least 4 rows"),
            (np.zeros((50, 1)), "at least 2 columns"),
            (np.full((50, 5), np.nan), "non-finite"),
        ],
    )
    def test_rejects_bad_matrices(self, matrix, match):
        with pytest.raises(ValueError, match=match):
            empirical_max_sharpe_null(matrix)

    def test_rejects_zero_simulations(self, independent_trials):
        with pytest.raises(ValueError, match="n_sims"):
            empirical_max_sharpe_null(independent_trials, n_sims=0)
