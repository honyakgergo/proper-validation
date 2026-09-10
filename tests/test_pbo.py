"""Tests for PBO via combinatorially symmetric cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from qv.stats.pbo import combinatorially_symmetric_cv


T, N = 1000, 150


@pytest.fixture
def noise_trials() -> np.ndarray:
    return 0.01 * np.random.default_rng(4).standard_normal((T, N))


def _noise(seed: int) -> np.ndarray:
    return 0.01 * np.random.default_rng(seed).standard_normal((T, N))


def _regime_reversing(seed: int, amplitude: float = 0.006) -> np.ndarray:
    """A search that is actively harmful, not merely useless.

    Every trial loads on a pattern that is positive in the first half of the
    sample and negative in the second, with a trial-specific loading. Whichever
    trial looked best on the in-sample blocks is, by construction, among the
    worst on the complementary ones - so PBO should be well above the 0.5 that
    a coin-flip search produces.
    """
    gen = np.random.default_rng(seed)
    base = 0.01 * gen.standard_normal((T, N))
    sign = np.where(np.arange(T) < T // 2, 1.0, -1.0)
    loading = gen.standard_normal(N)
    return base + amplitude * sign[:, None] * loading[None, :]


class TestCSCV:
    def test_pure_noise_is_a_coin_flip_on_average(self):
        """With no edge anywhere the in-sample winner lands at random
        out-of-sample, so PBO averages 0.5.

        Averaged over seeds deliberately. With 150 competing noise trials the
        identity of the winner is unstable, and a single dataset can land
        anywhere from about 0.12 to 0.73 - which is a real property of the
        statistic, not of this implementation, and a reason to read PBO
        alongside the other tests rather than alone.
        """
        values = [combinatorially_symmetric_cv(_noise(s)).pbo for s in range(12)]
        assert float(np.mean(values)) == pytest.approx(0.5, abs=0.12)

    @pytest.mark.parametrize("seed", range(4))
    def test_a_genuine_edge_gives_zero_pbo(self, seed):
        """The tool must not condemn a strategy that actually works. Unlike the
        noise case this is stable across every seed tried."""
        m = _noise(seed)
        m[:, 7] += 0.004
        r = combinatorially_symmetric_cv(m)
        assert r.pbo < 0.05
        assert not r.overfit
        assert r.median_oos_rank > 0.9
        assert r.probability_of_loss < 0.05

    @pytest.mark.parametrize("seed", range(4))
    def test_an_actively_harmful_search_is_caught(self, seed):
        r = combinatorially_symmetric_cv(_regime_reversing(seed))
        assert r.pbo > 0.5
        assert r.overfit
        assert r.probability_of_loss > 0.5

    def test_harmful_search_scores_far_worse_than_noise(self):
        """The discrimination that matters: a harmful search must separate
        clearly from a merely useless one."""
        harmful = float(np.mean([combinatorially_symmetric_cv(_regime_reversing(s)).pbo
                                 for s in range(6)]))
        useless = float(np.mean([combinatorially_symmetric_cv(_noise(s)).pbo for s in range(6)]))
        assert harmful > useless + 0.25

    def test_evaluates_every_combination(self, noise_trials):
        """C(16, 8) = 12870. Exhaustive, not sampled."""
        r = combinatorially_symmetric_cv(noise_trials, n_splits=16)
        assert r.n_combinations == 12_870

    @pytest.mark.parametrize("n_splits,expected", [(4, 6), (6, 20), (8, 70), (10, 252)])
    def test_combination_count_matches_the_binomial(self, noise_trials, n_splits, expected):
        r = combinatorially_symmetric_cv(noise_trials, n_splits=n_splits)
        assert r.n_combinations == expected
        assert r.n_splits == n_splits

    def test_shapes_line_up(self, noise_trials):
        r = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        assert r.logits.shape == r.is_best_sharpe.shape == r.oos_of_is_best.shape
        assert r.logits.size == r.n_combinations

    def test_pbo_is_the_fraction_of_non_positive_logits(self, noise_trials):
        r = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        assert r.pbo == pytest.approx(float(np.mean(r.logits <= 0)))

    def test_probability_of_loss_matches_the_oos_series(self, noise_trials):
        r = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        assert r.probability_of_loss == pytest.approx(float(np.mean(r.oos_of_is_best < 0)))

    def test_degradation_slope_is_documented_as_non_diagnostic(self, noise_trials):
        """Guards the interpretation, not the number. CSCV partitions one fixed
        sample, so the slope is negative even for a genuine edge."""
        r = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        assert r.degradation_slope_is_diagnostic is False

    def test_a_constant_column_cannot_win(self):
        """Cancellation in the block-moment form leaves a constant column with
        a variance near 1e-33 rather than zero; without a relative floor its
        Sharpe near 1e14 would win every single combination."""
        m = 0.01 * np.random.default_rng(4).standard_normal((400, 20))
        m[:, 0] = 0.005
        r = combinatorially_symmetric_cv(m, n_splits=8)
        assert np.all(np.abs(r.is_best_sharpe) < 10.0)

    def test_results_are_deterministic(self, noise_trials):
        """CSCV is exhaustive, so there is nothing random to seed."""
        a = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        b = combinatorially_symmetric_cv(noise_trials, n_splits=8)
        assert a.pbo == b.pbo and np.array_equal(a.logits, b.logits)

    def test_chunking_does_not_change_the_answer(self, monkeypatch, noise_trials):
        whole = combinatorially_symmetric_cv(noise_trials, n_splits=10)
        monkeypatch.setattr("qv.stats.pbo._CHUNK_CELLS", 300)
        chunked = combinatorially_symmetric_cv(noise_trials, n_splits=10)
        assert np.allclose(whole.logits, chunked.logits)
        assert whole.pbo == chunked.pbo

    def test_to_dict_is_json_shaped(self, noise_trials):
        d = combinatorially_symmetric_cv(noise_trials, n_splits=8).to_dict()
        assert set(d) >= {"pbo", "overfit", "probability_of_loss", "n_combinations"}
        # Thousands of logits do not belong in a report payload.
        assert "logits" not in d

    @pytest.mark.parametrize(
        "matrix,kwargs,match",
        [
            (np.zeros(100), {}, "2-D"),
            (np.zeros((100, 1)), {}, "at least 2 trials"),
            (np.zeros((100, 5)), {"n_splits": 7}, "even number"),
            (np.zeros((100, 5)), {"n_splits": 0}, "even number"),
            (np.zeros((20, 5)), {"n_splits": 16}, "at least 32 observations"),
            (np.full((100, 5), np.nan), {}, "non-finite"),
        ],
    )
    def test_rejects_bad_input(self, matrix, kwargs, match):
        with pytest.raises(ValueError, match=match):
            combinatorially_symmetric_cv(matrix, **kwargs)
