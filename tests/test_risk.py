"""Tests for the resampled risk distributions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.stats.performance import max_drawdown
from qv.stats.risk import (
    RiskDistribution,
    matched_random_walk_drawdowns,
    simulate_risk,
)
from qv.types import Severity
from tests.conftest import make_ar1, make_returns


class TestMatchedRandomWalk:
    def test_drawdowns_are_negative_or_zero(self):
        dd = matched_random_walk_drawdowns(0.0005, 0.01, 500, n_sims=200)
        assert dd.size == 200
        assert np.all(dd <= 0)

    def test_more_volatility_means_deeper_drawdowns(self):
        calm = matched_random_walk_drawdowns(0.0005, 0.005, 1000, n_sims=400)
        wild = matched_random_walk_drawdowns(0.0005, 0.02, 1000, n_sims=400)
        assert np.median(wild) < np.median(calm)

    def test_more_drift_means_shallower_drawdowns(self):
        weak = matched_random_walk_drawdowns(0.0001, 0.01, 1000, n_sims=400)
        strong = matched_random_walk_drawdowns(0.0015, 0.01, 1000, n_sims=400)
        assert np.median(strong) > np.median(weak)

    def test_longer_horizons_mean_deeper_drawdowns(self):
        """The expected maximum of a random walk grows with the horizon, which
        is why a drawdown figure is meaningless without the sample length."""
        short = matched_random_walk_drawdowns(0.0005, 0.01, 250, n_sims=400)
        long = matched_random_walk_drawdowns(0.0005, 0.01, 4000, n_sims=400)
        assert np.median(long) < np.median(short)

    def test_is_reproducible(self):
        a = matched_random_walk_drawdowns(0.0005, 0.01, 300, n_sims=100, seed=3)
        b = matched_random_walk_drawdowns(0.0005, 0.01, 300, n_sims=100, seed=3)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n_obs": 1}, "at least 2 periods"),
            ({"n_obs": 100, "n_sims": 0}, "n_sims"),
            ({"n_obs": 100, "sd": -1.0}, "non-negative"),
        ],
    )
    def test_rejects_bad_arguments(self, kwargs, match):
        kwargs.setdefault("sd", 0.01)
        with pytest.raises(ValueError, match=match):
            matched_random_walk_drawdowns(0.0, kwargs.pop("sd"), **kwargs)


class TestSimulateRisk:
    @pytest.fixture(scope="module")
    def sim(self):
        return simulate_risk(make_returns(1500, 0.05, seed=1), 252, n_boot=400, n_baseline=400)

    def test_realised_values_match_the_direct_computation(self, sim):
        x = make_returns(1500, 0.05, seed=1)
        assert sim.max_drawdown.realised == pytest.approx(max_drawdown(x)[0])
        assert sim.total_return.realised == pytest.approx(float(np.prod(1 + x) - 1))

    def test_all_three_statistics_are_resampled(self, sim):
        for dist in (sim.total_return, sim.sharpe, sim.max_drawdown):
            assert dist.draws.size > 300
            assert np.all(np.isfinite(dist.draws))

    def test_intervals_bracket_the_median(self, sim):
        for dist in (sim.total_return, sim.sharpe, sim.max_drawdown):
            assert dist.ci_low <= dist.median <= dist.ci_high

    def test_drawdowns_are_never_positive(self, sim):
        assert np.all(sim.max_drawdown.draws <= 0)
        assert sim.max_drawdown.realised <= 0

    def test_only_drawdown_carries_a_baseline(self, sim):
        """The random-walk comparison is the one that makes a drawdown
        interpretable; return and Sharpe already have intervals elsewhere."""
        assert sim.max_drawdown.baseline_median is not None
        assert sim.total_return.baseline_median is None
        assert sim.sharpe.baseline_median is None

    def test_a_typical_path_is_not_flagged(self, sim):
        """An ordinary series must not be accused of a path-dependent
        drawdown, or the finding means nothing."""
        assert not sim.drawdown_path_dependent
        assert sim.to_finding() is None
        assert sim.drawdown_severity is Severity.INFO

    @staticmethod
    def _with_slide(drag: float, length: int, seed: int = 4) -> np.ndarray:
        gen = np.random.default_rng(seed)
        x = 0.0008 + 0.006 * gen.standard_normal(1500)
        x[600 : 600 + length] -= drag
        return x

    def test_a_long_slow_decline_is_flagged(self):
        """A mild drag sustained over 300 periods is exactly what block
        resampling cannot reassemble: no individual return is unusual, so the
        drawdown lives entirely in the ordering."""
        sim = simulate_risk(self._with_slide(0.002, 300), 252, n_boot=500, n_baseline=300)
        assert sim.drawdown_path_dependent
        finding = sim.to_finding()
        assert finding.id == "RISK-DRAWDOWN-PATH-DEPENDENT"
        assert "order" in finding.detail
        assert sim.drawdown_severity >= Severity.MEDIUM

    def test_a_short_violent_crash_is_not_flagged(self):
        """The complement, and the honest limit of this test.

        A severe crash makes its own days extreme, so resampled paths that
        happen to cluster a few of them reproduce a comparable drawdown - the
        loss is in the *distribution*, which the bootstrap can recreate. The
        flag is specifically about ordering, and a magnitude-driven drawdown
        correctly stays quiet rather than being reported as path dependence.
        """
        sim = simulate_risk(self._with_slide(0.010, 200), 252, n_boot=500, n_baseline=300)
        assert sim.max_drawdown.realised < -0.5, "fixture should produce a severe drawdown"
        assert not sim.drawdown_path_dependent

    def test_serial_dependence_widens_the_drawdown_range(self):
        """Persistent returns produce deeper resampled drawdowns than IID ones.

        Volatility is matched deliberately. Without that the comparison just
        measures which series is noisier, which is not the question - the
        first version of this test failed for exactly that reason.
        """
        iid = make_returns(1200, 0.03, seed=5)
        trending = make_ar1(1200, 0.6, seed=5)
        trending = trending / trending.std() * iid.std()
        assert trending.std() == pytest.approx(iid.std())

        a = simulate_risk(iid, 252, n_boot=300, n_baseline=100)
        b = simulate_risk(trending, 252, n_boot=300, n_baseline=100)
        assert b.block_length > a.block_length
        assert b.max_drawdown.ci_low < a.max_drawdown.ci_low

    def test_states_the_block_resampling_caveat(self, sim):
        """The distribution understates severity and the report must say so."""
        assert "cannot reassemble" in sim.note

    def test_is_reproducible(self):
        x = make_returns(600, 0.05, seed=6)
        a = simulate_risk(x, 252, n_boot=200, n_baseline=200, seed=9)
        b = simulate_risk(x, 252, n_boot=200, n_baseline=200, seed=9)
        assert np.array_equal(a.max_drawdown.draws, b.max_drawdown.draws)

    def test_small_sample_is_flagged(self):
        sim = simulate_risk(make_returns(25, 0.05, seed=7), 4, n_boot=100, n_baseline=100)
        assert not sim.reliable
        assert "below the 30" in sim.note

    def test_explicit_block_length_is_honoured(self):
        sim = simulate_risk(
            make_returns(400, 0.05, seed=8), 252, n_boot=100, n_baseline=100, block_length=30.0
        )
        assert sim.block_length == 30.0

    def test_to_dict_is_json_shaped(self, sim):
        import json

        d = sim.to_dict()
        assert set(d) >= {"total_return", "sharpe", "max_drawdown", "drawdown_path_dependent"}
        # The raw draws are chart input, not report payload.
        assert "draws" not in d["max_drawdown"]
        json.dumps(d)

    @pytest.mark.parametrize(
        "args,match",
        [((np.zeros(10),), "at least 20"), ((make_returns(100, 0.05, seed=9),), "n_boot")],
    )
    def test_rejects_bad_input(self, args, match):
        kwargs = {"n_boot": 0} if "n_boot" in match else {}
        with pytest.raises(ValueError, match=match):
            simulate_risk(*args, **kwargs)


class TestRiskDistribution:
    def test_percentile_locates_the_realised_value(self):
        dist = RiskDistribution("x", realised=0.5, draws=np.linspace(0, 1, 1001))
        assert dist.percentile == pytest.approx(0.5, abs=0.01)

    def test_outside_resamples_respects_direction(self):
        """For drawdown, worse means lower; for a return, worse means higher
        is not a concern. The flag has to know which way is bad."""
        draws = np.linspace(-0.4, -0.1, 1000)
        deep = RiskDistribution("dd", realised=-0.9, draws=draws, lower_is_worse=True)
        shallow = RiskDistribution("dd", realised=-0.2, draws=draws, lower_is_worse=True)
        assert deep.outside_resamples
        assert not shallow.outside_resamples

    def test_baseline_is_optional(self):
        dist = RiskDistribution("x", realised=0.1, draws=np.linspace(0, 1, 100))
        assert dist.baseline_median is None
        assert dist.baseline_percentile is None

    def test_to_dict_is_json_shaped(self):
        d = RiskDistribution("x", 0.1, np.linspace(0, 1, 100)).to_dict()
        assert set(d) >= {"realised", "median", "ci_low", "ci_high", "percentile"}
        assert math.isfinite(d["median"])
