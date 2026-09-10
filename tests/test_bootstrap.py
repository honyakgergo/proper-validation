"""Tests for the stationary bootstrap and automatic block-length selection."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.bootstrap import (
    BootstrapResult,
    flat_top_lag_window,
    politis_white_block_length,
    stationary_bootstrap,
    stationary_bootstrap_indices,
)
from qv.stats.sharpe import sharpe_ratio
from tests.conftest import make_ar1, make_returns


class TestFlatTopLagWindow:
    def test_known_values(self):
        got = flat_top_lag_window(np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0]))
        assert got.tolist() == [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]

    def test_is_symmetric(self):
        assert flat_top_lag_window(-0.75) == flat_top_lag_window(0.75)

    def test_flat_region_then_linear_taper(self):
        assert flat_top_lag_window(0.6) == pytest.approx(0.8)


class TestPolitisWhiteBlockLength:
    def test_iid_data_needs_almost_no_blocking(self, rng):
        """With no serial dependence the rule should ask for very short blocks.

        Not exactly 1.0: the numerator is a kernel sum over sample
        autocovariances that are noise around zero, so the rule returns a small
        noisy value rather than collapsing exactly. That is the estimator
        behaving correctly - short blocks on independent data cost nothing, as
        the coverage simulation confirms.
        """
        assert 1.0 <= politis_white_block_length(rng.standard_normal(3000)) < 5.0

    def test_persistent_data_needs_long_blocks(self):
        assert politis_white_block_length(make_ar1(3000, 0.7, seed=1)) > 10

    def test_block_length_grows_with_persistence(self):
        weak = politis_white_block_length(make_ar1(4000, 0.2, seed=2))
        strong = politis_white_block_length(make_ar1(4000, 0.8, seed=2))
        assert strong > weak

    def test_respects_the_upper_bound(self):
        """b must never exceed min(3*sqrt(n), n/3), or resamples stop being
        meaningfully random."""
        x = make_ar1(200, 0.95, seed=3)
        b_max = math.ceil(min(3 * math.sqrt(200), 200 / 3))
        assert 1.0 <= politis_white_block_length(x) <= b_max

    def test_constant_series_returns_one(self):
        assert politis_white_block_length([0.5] * 100) == 1.0

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            politis_white_block_length([1.0, 2.0, 3.0])


class TestStationaryBootstrapIndices:
    def test_shape_and_range(self):
        idx = stationary_bootstrap_indices(50, 5.0, 20, np.random.default_rng(0))
        assert idx.shape == (20, 50)
        assert idx.min() >= 0 and idx.max() < 50

    def test_block_continuation_rate_matches_one_minus_one_over_b(self):
        """The defining property: each step continues the block with
        probability 1 - 1/b."""
        n, b = 400, 8.0
        idx = stationary_bootstrap_indices(n, b, 500, np.random.default_rng(1))
        continued = ((idx[:, 1:] - idx[:, :-1]) % n == 1).mean()
        assert float(continued) == pytest.approx(1 - 1 / b, abs=0.01)

    def test_block_length_one_is_iid_resampling(self):
        """b=1 means p=1, so every step jumps - the IID bootstrap."""
        n = 200
        idx = stationary_bootstrap_indices(n, 1.0, 200, np.random.default_rng(2))
        continued = ((idx[:, 1:] - idx[:, :-1]) % n == 1).mean()
        assert float(continued) == pytest.approx(1 / n, abs=0.01)

    def test_wraps_circularly(self):
        """Indices must wrap rather than truncate, or the tail of the series
        would be systematically under-sampled."""
        n = 10
        idx = stationary_bootstrap_indices(n, 1000.0, 200, np.random.default_rng(3))
        wraps = ((idx[:, :-1] == n - 1) & (idx[:, 1:] == 0)).sum()
        assert wraps > 0

    def test_is_deterministic_given_a_seed(self):
        a = stationary_bootstrap_indices(30, 4.0, 5, np.random.default_rng(7))
        b = stationary_bootstrap_indices(30, 4.0, 5, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_single_observation_degenerates(self):
        assert stationary_bootstrap_indices(1, 2.0, 3, np.random.default_rng(0)).tolist() == [
            [0],
            [0],
            [0],
        ]

    @pytest.mark.parametrize(
        "args,match",
        [
            ((0, 2.0, 5), "n must be at least 1"),
            ((10, 2.0, 0), "n_boot must be at least 1"),
            ((10, 0.5, 5), "block_length must be at least 1"),
        ],
    )
    def test_rejects_bad_arguments(self, args, match):
        with pytest.raises(ValueError, match=match):
            stationary_bootstrap_indices(*args, np.random.default_rng(0))


class TestStationaryBootstrap:
    def test_observed_matches_the_statistic_on_the_original_data(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200)
        assert res.observed == pytest.approx(sharpe_ratio(iid_returns))

    def test_interval_brackets_the_point_estimate(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=500)
        assert res.ci_low < res.observed < res.ci_high

    def test_is_reproducible_from_the_seed(self, iid_returns):
        a = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200, seed=42)
        b = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200, seed=42)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_different_seeds_give_different_intervals(self, iid_returns):
        a = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200, seed=1)
        b = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200, seed=2)
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_wider_confidence_gives_a_wider_interval(self, iid_returns):
        narrow = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=500, confidence=0.80)
        wide = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=500, confidence=0.99)
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)

    def test_explicit_block_length_overrides_the_automatic_choice(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=100, block_length=25.0)
        assert res.block_length == 25.0

    def test_serial_dependence_widens_the_interval(self):
        """The reason this bootstrap exists. An IID bootstrap (b=1) on
        persistent data produces a falsely narrow interval; blocking fixes it."""
        x = make_ar1(1500, 0.8, seed=4, mean=0.05)
        iid_like = stationary_bootstrap(x, sharpe_ratio, n_boot=600, block_length=1.0)
        blocked = stationary_bootstrap(x, sharpe_ratio, n_boot=600)
        assert blocked.block_length > 5
        assert (blocked.ci_high - blocked.ci_low) > 1.2 * (iid_like.ci_high - iid_like.ci_low)

    def test_standard_error_is_the_spread_of_the_distribution(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=300)
        assert res.standard_error == pytest.approx(float(np.std(res.distribution, ddof=1)))

    def test_percentile_of_reports_position_in_the_distribution(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=400)
        assert res.percentile_of(float(res.distribution.min()) - 1) == 0.0
        assert res.percentile_of(float(res.distribution.max()) + 1) == 1.0

    def test_drops_non_finite_replications_and_says_so(self):
        """A statistic that fails on some resamples must not poison the
        percentiles silently."""
        calls = {"n": 0}

        def flaky(sample: np.ndarray) -> float:
            calls["n"] += 1
            return math.nan if calls["n"] % 4 == 0 else float(np.mean(sample))

        res = stationary_bootstrap([0.1, 0.2, 0.3, 0.4, 0.5] * 10, flaky, n_boot=100)
        assert res.distribution.size < 100
        assert "non-finite" in res.note

    def test_raises_when_every_replication_fails(self, iid_returns):
        with pytest.raises(ValueError, match="non-finite"):
            stationary_bootstrap(iid_returns, lambda _: math.nan, n_boot=20)

    def test_small_sample_is_flagged_unreliable(self):
        res = stationary_bootstrap(make_returns(15, 0.3, seed=5), sharpe_ratio, n_boot=100)
        assert res.reliable is False
        assert "cannot manufacture information" in res.note

    def test_to_estimate_carries_the_interval(self, iid_returns):
        res = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=200)
        est = res.to_estimate("stationary bootstrap")
        assert (est.value, est.ci_low, est.ci_high) == (res.observed, res.ci_low, res.ci_high)
        assert est.method == "stationary bootstrap"

    def test_to_dict_is_json_shaped(self, iid_returns):
        d = stationary_bootstrap(iid_returns, sharpe_ratio, n_boot=100).to_dict()
        assert set(d) >= {"observed", "ci_low", "ci_high", "block_length", "n_boot"}
        assert "distribution" not in d  # 2000 floats do not belong in a report

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_boot": 0}, "n_boot must be at least 1"),
            ({"confidence": 0.0}, "between 0 and 1"),
            ({"confidence": 1.0}, "between 0 and 1"),
        ],
    )
    def test_rejects_bad_arguments(self, iid_returns, kwargs, match):
        with pytest.raises(ValueError, match=match):
            stationary_bootstrap(iid_returns, sharpe_ratio, **kwargs)

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            stationary_bootstrap([0.1, 0.2, 0.3], sharpe_ratio)


@pytest.mark.slow
class TestCoverage:
    """The real correctness test of the inference, not a smoke test.

    If the 95% interval does not cover the truth about 95% of the time, every
    confidence interval this package prints is a lie - and a validator that
    misreports its own uncertainty has no standing to criticise anyone.
    """

    def test_iid_coverage_is_near_nominal(self):
        true_sr, n_sims, n_obs = 0.05, 300, 750
        covered = 0
        for s in range(n_sims):
            x = make_returns(n_obs, true_sr, seed=1000 + s)
            res = stationary_bootstrap(x, sharpe_ratio, n_boot=300, seed=s)
            covered += res.ci_low <= true_sr <= res.ci_high
        rate = covered / n_sims
        # Monte Carlo SE at 95% over 300 sims is ~1.3%, so allow a wide band
        # but still catch a genuinely broken interval.
        assert 0.90 <= rate <= 0.99, f"coverage {rate:.3f} is not near the nominal 0.95"

    def test_coverage_holds_under_serial_dependence(self):
        """The case the IID bootstrap gets wrong. True Sharpe of an AR(1) with
        drift mu and innovation sd 1 is mu / (sd of the process)."""
        phi, n_sims, n_obs = 0.5, 250, 750
        process_sd = 1.0 / math.sqrt(1 - phi**2)
        drift = 0.05 * process_sd
        true_sr = drift / process_sd
        covered = 0
        for s in range(n_sims):
            x = make_ar1(n_obs, phi, seed=2000 + s) + drift
            res = stationary_bootstrap(x, sharpe_ratio, n_boot=300, seed=s)
            covered += res.ci_low <= true_sr <= res.ci_high
        rate = covered / n_sims
        assert 0.88 <= rate <= 0.99, f"coverage {rate:.3f} is not near the nominal 0.95"
