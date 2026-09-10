"""Tests for PSR, Deflated Sharpe, MinBTL and the BHY haircut."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.deflated import (
    EULER_MASCHERONI,
    _by_adjusted_pvalues,
    bhy_haircut,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    sharpe_variance_term,
    trial_sharpes,
)
from qv.stats.sharpe import sharpe_ratio, sharpe_standard_error
from qv.types import Severity
from tests.conftest import make_returns


class TestExpectedMaxSharpe:
    def test_single_trial_means_no_selection(self):
        assert expected_max_sharpe(1) == 0.0

    def test_grows_with_trial_count(self):
        values = [expected_max_sharpe(n) for n in (2, 10, 100, 1000, 10_000)]
        assert values == sorted(values)

    def test_tracks_the_extreme_value_asymptote(self):
        """E[max of N standard normals] ~ sqrt(2*log(N)) for large N."""
        for n in (1000, 10_000, 100_000):
            assert expected_max_sharpe(n) == pytest.approx(math.sqrt(2 * math.log(n)), rel=0.15)

    def test_matches_a_monte_carlo_maximum(self):
        """The formula approximates the expected maximum of N normal draws."""
        gen = np.random.default_rng(0)
        simulated = gen.standard_normal((40_000, 50)).max(axis=1).mean()
        assert expected_max_sharpe(50) == pytest.approx(float(simulated), rel=0.05)

    def test_scales_linearly_with_dispersion(self):
        assert expected_max_sharpe(100, 0.05) == pytest.approx(0.05 * expected_max_sharpe(100))

    def test_zero_dispersion_gives_zero(self):
        assert expected_max_sharpe(500, 0.0) == 0.0

    def test_uses_the_euler_mascheroni_constant(self):
        assert EULER_MASCHERONI == pytest.approx(0.57721566, abs=1e-8)

    @pytest.mark.parametrize(
        "args,match",
        [((0,), "at least 1"), ((10, -1.0), "non-negative")],
    )
    def test_rejects_bad_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            expected_max_sharpe(*args)


class TestSharpeVarianceTerm:
    def test_normal_case_reduces_to_the_gaussian_form(self):
        assert sharpe_variance_term(0.4, 0.0, 3.0) == pytest.approx(1 + 0.5 * 0.4**2)

    def test_agrees_with_the_lo_standard_error(self, iid_returns):
        """The DSR bracket and the Lo/Mertens SE must be the same expression,
        or the two halves of the package disagree about non-normality."""
        from qv.stats.moments import standardised_moments

        sr = sharpe_ratio(iid_returns)
        g3, g4 = standardised_moments(iid_returns)
        expected = math.sqrt(sharpe_variance_term(sr, g3, g4) / len(iid_returns))
        assert sharpe_standard_error(iid_returns) == pytest.approx(expected, rel=1e-12)

    def test_negative_skew_widens_the_term(self):
        assert sharpe_variance_term(0.5, -1.0, 3.0) > sharpe_variance_term(0.5, 0.0, 3.0)

    def test_fat_tails_widen_the_term(self):
        assert sharpe_variance_term(0.5, 0.0, 9.0) > sharpe_variance_term(0.5, 0.0, 3.0)

    def test_clamped_above_zero(self):
        assert sharpe_variance_term(0.5, 40.0, 3.0) > 0


class TestProbabilisticSharpe:
    def test_strong_track_record_gives_high_probability(self):
        assert probabilistic_sharpe_ratio(make_returns(2000, 0.10, seed=1)) > 0.99

    def test_zero_edge_averages_a_half(self):
        """Averaged over seeds, not for one draw. With n=2000 the sampling
        error on the Sharpe is about 0.022, which the sqrt(n-1) scaling turns
        into a PSR anywhere in (0, 1) for any single series - so asserting on
        one seed tests the seed, not the estimator."""
        values = [probabilistic_sharpe_ratio(make_returns(2000, 0.0, seed=s)) for s in range(80)]
        assert float(np.mean(values)) == pytest.approx(0.5, abs=0.08)

    def test_falls_as_the_benchmark_rises(self):
        x = make_returns(2000, 0.08, seed=3)
        assert probabilistic_sharpe_ratio(x, 0.0) > probabilistic_sharpe_ratio(x, 0.15)

    def test_zero_variance_is_nan(self):
        assert math.isnan(probabilistic_sharpe_ratio([0.01] * 50))

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            probabilistic_sharpe_ratio([0.1, 0.2, 0.3])


class TestTrialSharpes:
    def test_one_sharpe_per_column(self, trial_matrix):
        assert trial_sharpes(trial_matrix).shape == (trial_matrix.shape[1],)

    def test_values_match_column_by_column(self, trial_matrix):
        got = trial_sharpes(trial_matrix)
        assert got[5] == pytest.approx(sharpe_ratio(trial_matrix[:, 5]))

    def test_drops_flat_columns(self, trial_matrix):
        """A never-trading column must not become the winner.

        The sample standard deviation of identical floats is about 1e-16, not
        zero, so a naive ``std > 0`` check lets a constant column through with
        a Sharpe near 1e14 - which then wins every selection it enters.
        """
        m = trial_matrix.copy()
        m[:, 0] = 0.01
        got = trial_sharpes(m)
        assert got.size == trial_matrix.shape[1] - 1
        assert got.max() < 1.0

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="2-D"):
            trial_sharpes(np.zeros(50))

    def test_rejects_single_row(self):
        with pytest.raises(ValueError, match="at least 2 rows"):
            trial_sharpes(np.zeros((1, 5)))


class TestDeflatedSharpe:
    def test_mined_noise_is_deflated_away(self, trial_matrix):
        """The headline demonstration: the best of 200 pure-noise trials looks
        publishable until the trial count is admitted."""
        sharpes = trial_sharpes(trial_matrix)
        best = trial_matrix[:, int(np.argmax(sharpes))]

        claiming_one = deflated_sharpe_ratio(best, n_trials=1)
        admitting_all = deflated_sharpe_ratio(best, n_trials=200, trial_returns=trial_matrix)

        assert claiming_one.dsr > 0.95 and claiming_one.survives
        assert admitting_all.dsr < 0.60 and not admitting_all.survives
        assert admitting_all.severity in (Severity.HIGH, Severity.CRITICAL)

    def test_winner_sits_near_the_expected_maximum(self, trial_matrix):
        """The mined-noise result in one line: the best of N noise trials lands
        right at the null expectation for the best of N noise trials."""
        sharpes = trial_sharpes(trial_matrix)
        best = trial_matrix[:, int(np.argmax(sharpes))]
        r = deflated_sharpe_ratio(best, n_trials=200, trial_returns=trial_matrix)
        assert r.observed_sharpe == pytest.approx(r.expected_max_sharpe, rel=0.25)

    def test_genuine_edge_survives_deflation(self):
        """The tool must not be a pessimism generator."""
        r = deflated_sharpe_ratio(make_returns(3000, 0.12, seed=4), n_trials=50)
        assert r.dsr > 0.95 and r.survives
        assert r.severity is Severity.INFO

    def test_more_trials_lower_the_dsr(self):
        x = make_returns(1500, 0.06, seed=5)
        assert deflated_sharpe_ratio(x, 1000).dsr < deflated_sharpe_ratio(x, 10).dsr

    def test_single_trial_matches_psr_at_zero(self):
        x = make_returns(1000, 0.05, seed=6)
        assert deflated_sharpe_ratio(x, 1).dsr == pytest.approx(probabilistic_sharpe_ratio(x))

    def test_p_value_is_the_complement(self):
        r = deflated_sharpe_ratio(make_returns(1000, 0.05, seed=7), 20)
        assert r.p_value == pytest.approx(1 - r.dsr)

    def test_measures_dispersion_from_a_trial_matrix(self, trial_matrix):
        r = deflated_sharpe_ratio(trial_matrix[:, 0], 200, trial_returns=trial_matrix)
        assert "measured from" in r.trial_sharpe_std_source
        assert r.trial_sharpe_std == pytest.approx(float(np.std(trial_sharpes(trial_matrix), ddof=1)))

    def test_accepts_a_supplied_dispersion(self):
        r = deflated_sharpe_ratio(make_returns(500, 0.05, seed=8), 100, trial_sharpe_std=0.04)
        assert r.trial_sharpe_std == 0.04
        assert r.trial_sharpe_std_source == "supplied by caller"

    def test_falls_back_to_one_over_sqrt_n_and_warns(self):
        n = 500
        r = deflated_sharpe_ratio(make_returns(n, 0.05, seed=9), 100)
        assert r.trial_sharpe_std == pytest.approx(1 / math.sqrt(n))
        assert "lower bound" in r.note

    def test_trial_matrix_takes_precedence_over_supplied_std(self, trial_matrix):
        r = deflated_sharpe_ratio(
            trial_matrix[:, 0], 200, trial_sharpe_std=99.0, trial_returns=trial_matrix
        )
        assert r.trial_sharpe_std != 99.0

    def test_single_trial_claim_is_flagged(self):
        r = deflated_sharpe_ratio(make_returns(500, 0.05, seed=10), 1)
        assert "almost never true" in r.note

    def test_small_sample_is_flagged(self):
        r = deflated_sharpe_ratio(make_returns(20, 0.3, seed=11), 10)
        assert r.reliable is False
        assert "normal approximation" in r.note

    def test_severity_ladder(self):
        from qv.stats.deflated import DeflatedSharpeResult

        def sev(dsr):
            return DeflatedSharpeResult(
                0.1, 0.05, dsr, 10, 0.01, "x", 500, 0.0, 3.0, True
            ).severity

        assert sev(0.99) is Severity.INFO
        assert sev(0.92) is Severity.MEDIUM
        assert sev(0.70) is Severity.HIGH
        assert sev(0.20) is Severity.CRITICAL

    def test_to_dict_is_json_shaped(self, trial_matrix):
        d = deflated_sharpe_ratio(trial_matrix[:, 0], 200, trial_returns=trial_matrix).to_dict()
        assert set(d) >= {"dsr", "p_value", "expected_max_sharpe", "n_trials", "survives"}

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_trials": 0}, "at least 1"),
            ({"n_trials": 10, "trial_sharpe_std": -0.5}, "non-negative"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            deflated_sharpe_ratio(make_returns(200, 0.05, seed=12), **kwargs)

    def test_rejects_a_degenerate_trial_matrix(self):
        with pytest.raises(ValueError, match="at least 2 usable columns"):
            deflated_sharpe_ratio(
                make_returns(200, 0.05, seed=13), 10, trial_returns=np.ones((200, 3))
            )


class TestMinimumBacktestLength:
    def test_reproduces_the_published_magnitude(self):
        """Bailey et al. report roughly five years for about 45 trials against
        a target annual Sharpe of 1."""
        assert minimum_backtest_length(45, 1.0) == pytest.approx(5.0, abs=0.5)

    def test_grows_with_trial_count(self):
        years = [minimum_backtest_length(n) for n in (10, 100, 1000)]
        assert years == sorted(years)

    def test_shrinks_as_the_target_rises(self):
        assert minimum_backtest_length(100, 2.0) < minimum_backtest_length(100, 1.0)

    def test_scales_inversely_with_the_square_of_the_target(self):
        assert minimum_backtest_length(100, 2.0) == pytest.approx(
            minimum_backtest_length(100, 1.0) / 4
        )

    def test_single_trial_needs_no_extra_length(self):
        assert minimum_backtest_length(1) == 0.0

    @pytest.mark.parametrize(
        "args,match", [((0,), "at least 1"), ((10, 0.0), "must be positive")]
    )
    def test_rejects_bad_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            minimum_backtest_length(*args)


class TestBhyOverTheTrialFamily:
    """BHY is defined over a family of tests, not over one test and a count.

    Judging the audited result as though it were the most significant of N
    applies Bonferroni times the harmonic number - ~248x at 54 trials - which
    floors the adjusted t-statistic at zero for anything below t = 2.9 and
    renders a near miss identically to a hopeless one. The family is normally
    right there in the trial matrix.
    """

    @staticmethod
    def _tstats(sharpes, n_obs=2000):
        """t-statistics for a set of per-period Sharpes at a given length."""
        return np.asarray(sharpes) * np.sqrt(n_obs)

    def test_a_mid_ranked_result_is_judged_at_its_own_rank(self):
        family = self._tstats(np.linspace(0.02, 0.06, 40))
        mine = float(family[20])
        bound = bhy_haircut(mine, 40)
        full = bhy_haircut(mine, 40, trial_tstats=family)
        assert full.used_trial_family and not bound.used_trial_family
        assert full.rank > 1
        assert full.haircut < bound.haircut
        assert full.adjusted_tstat > bound.adjusted_tstat

    def test_the_best_of_a_field_of_noise_is_still_destroyed(self):
        """The bound and the family agree where the bound is the right answer:
        a winner mined out of nothing."""
        gen = np.random.default_rng(5)
        family = gen.standard_normal(500) * 0.9
        best = float(np.max(family))
        h = bhy_haircut(best, 500, trial_tstats=family)
        assert h.rank == 1
        assert not h.significant_at_5pct
        assert h.haircut > 0.5

    def test_an_incomplete_family_falls_back_to_the_bound(self):
        """Declaring 200 trials and handing over 40 columns leaves 160 tests
        nobody can see, and the conservative bound is the honest answer."""
        family = self._tstats(np.linspace(0.02, 0.06, 40))
        h = bhy_haircut(float(family[20]), 200, trial_tstats=family)
        assert not h.used_trial_family
        assert h.n_tests == 200

    def test_a_result_outside_its_family_is_added_to_it(self):
        family = self._tstats(np.linspace(0.02, 0.06, 40))
        h = bhy_haircut(9.0, 40, trial_tstats=family)
        assert h.n_tests == 41
        assert h.rank == 1

    def test_adjusted_pvalues_are_monotone_and_bounded(self):
        gen = np.random.default_rng(6)
        ps = gen.uniform(0.0001, 0.9, 200)
        ordered, adjusted, c_n = _by_adjusted_pvalues(ps)
        assert np.all(np.diff(ordered) >= 0)
        assert np.all(np.diff(adjusted) >= -1e-12)
        assert np.all(adjusted <= 1.0)
        assert c_n == pytest.approx(float(np.sum(1.0 / np.arange(1, 201))))

    def test_the_family_never_makes_the_bar_harder_than_the_bound(self):
        """A step-up procedure can only be more permissive than the single-test
        bound it starts from, so the family must never punish more."""
        gen = np.random.default_rng(7)
        family = np.abs(gen.standard_normal(60)) * 1.5
        for t in (1.5, 2.4, 3.6):
            full = bhy_haircut(t, 60, trial_tstats=family)
            bound = bhy_haircut(t, full.n_tests)
            assert full.haircut <= bound.haircut + 1e-12

    def test_a_constant_column_cannot_take_over_the_family(self):
        """A column of identical returns has a standard deviation near 1e-16,
        a Sharpe near 1e14 and a p-value of zero. Left in, it becomes rank 1
        and drags every adjusted p-value down with it."""
        from qv.stats.sharpe import columnwise_sharpe_tstat

        gen = np.random.default_rng(8)
        matrix = 0.01 * gen.standard_normal((500, 5))
        matrix[:, 2] = 0.001
        tstats = columnwise_sharpe_tstat(matrix)
        assert math.isnan(tstats[2])
        assert np.isfinite(np.delete(tstats, 2)).all()


class TestBhyHaircut:
    def test_single_test_is_not_haircut(self):
        h = bhy_haircut(3.0, 1)
        assert h.haircut == pytest.approx(0.0, abs=1e-9)
        assert h.adjusted_tstat == pytest.approx(3.0)

    def test_haircut_grows_with_trial_count(self):
        cuts = [bhy_haircut(4.0, n).haircut for n in (1, 5, 20, 100)]
        assert cuts == sorted(cuts)

    def test_uses_the_harmonic_penalty(self):
        """BHY multiplies the p-value by N times the Nth harmonic number, which
        is what makes it valid under arbitrary dependence between tests."""
        n = 10
        c_n = sum(1 / i for i in range(1, n + 1))
        h = bhy_haircut(3.0, n)
        assert h.adjusted_pvalue == pytest.approx(h.observed_pvalue * n * c_n)

    def test_is_more_conservative_than_bonferroni_for_the_top_test(self):
        n = 50
        h = bhy_haircut(3.5, n)
        assert h.adjusted_pvalue > h.observed_pvalue * n

    def test_strong_result_survives_a_modest_search(self):
        assert bhy_haircut(6.0, 20).significant_at_5pct

    def test_marginal_result_dies_in_a_large_search(self):
        assert not bhy_haircut(2.1, 500).significant_at_5pct

    def test_clamps_the_adjusted_pvalue_at_one(self):
        h = bhy_haircut(1.0, 10_000)
        assert h.adjusted_pvalue == 1.0
        assert h.adjusted_tstat == 0.0
        assert h.haircut == pytest.approx(1.0)

    def test_maps_the_haircut_onto_the_sharpe(self):
        h = bhy_haircut(3.0, 10, observed_sharpe=1.5)
        assert h.haircut_sharpe == pytest.approx(1.5 * (1 - h.haircut))

    def test_sharpe_is_optional(self):
        assert bhy_haircut(3.0, 10).haircut_sharpe is None

    def test_preserves_the_sign_of_a_negative_tstat(self):
        assert bhy_haircut(-3.0, 5).adjusted_tstat < 0

    def test_to_dict_names_the_method(self):
        assert "Yekutieli" in bhy_haircut(3.0, 10).to_dict()["method"]

    @pytest.mark.parametrize(
        "args,match", [((3.0, 0), "at least 1"), ((math.nan, 10), "must be finite")]
    )
    def test_rejects_bad_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            bhy_haircut(*args)
