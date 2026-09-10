"""Tests for factor attribution and the volatility-matched benchmark."""

from __future__ import annotations

import math

import numpy as np
import pytest

from qv.attribution.benchmark import vol_matched_benchmark
from qv.attribution.factors import factor_attribution
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity


@pytest.fixture
def factor_panel():
    """Three plausible factor return series."""
    gen = np.random.default_rng(1)
    n = 1500
    mkt = 0.0004 + 0.011 * gen.standard_normal(n)
    smb = 0.0001 + 0.006 * gen.standard_normal(n)
    hml = 0.0000 + 0.006 * gen.standard_normal(n)
    return np.column_stack([mkt, smb, hml]), ["Mkt-RF", "SMB", "HML"], mkt


class TestFactorAttribution:
    def test_recovers_planted_loadings(self, factor_panel):
        x, names, mkt = factor_panel
        gen = np.random.default_rng(2)
        y = 0.8 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2] + 0.002 * gen.standard_normal(len(x))
        r = factor_attribution(y, x, names)
        assert r.loading("Mkt-RF").beta == pytest.approx(0.8, abs=0.05)
        assert r.loading("SMB").beta == pytest.approx(0.3, abs=0.05)
        assert r.loading("HML").beta == pytest.approx(-0.2, abs=0.05)

    def test_pure_factor_exposure_has_no_alpha(self, factor_panel):
        """A strategy that is only factor exposure must not show alpha."""
        x, names, _ = factor_panel
        gen = np.random.default_rng(3)
        y = 0.9 * x[:, 0] + 0.001 * gen.standard_normal(len(x))
        r = factor_attribution(y, x, names)
        assert not r.alpha_survives
        assert r.explained_by_factors
        assert r.severity is Severity.CRITICAL

    def test_genuine_alpha_survives(self, factor_panel):
        x, names, _ = factor_panel
        gen = np.random.default_rng(4)
        y = 0.0008 + 0.3 * x[:, 0] + 0.004 * gen.standard_normal(len(x))
        r = factor_attribution(y, x, names)
        assert r.alpha_survives
        assert r.alpha_p_value < 0.05
        assert r.severity is Severity.INFO

    def test_annualises_alpha(self, factor_panel):
        x, names, _ = factor_panel
        gen = np.random.default_rng(5)
        y = 0.0005 + 0.004 * gen.standard_normal(len(x))
        r = factor_attribution(y, x, names, periods_per_year=252)
        assert r.alpha_annualised == pytest.approx(r.alpha_per_period.value * 252)

    def test_hac_standard_errors_are_wider_under_autocorrelation(self):
        """The reason HAC is the default: with autocorrelated residuals, plain
        OLS understates the uncertainty on alpha - the one coefficient that
        matters."""
        gen = np.random.default_rng(6)
        n = 2000
        factor = 0.01 * gen.standard_normal(n)
        resid = np.zeros(n)
        e = 0.004 * gen.standard_normal(n)
        for t in range(1, n):
            resid[t] = 0.85 * resid[t - 1] + e[t]
        y = 0.0003 + 0.5 * factor + resid

        hac = factor_attribution(y, factor, ["f"], use_hac=True)
        ols = factor_attribution(y, factor, ["f"], use_hac=False)
        hac_se = hac.alpha_per_period.ci_high - hac.alpha_per_period.value
        ols_se = ols.alpha_per_period.ci_high - ols.alpha_per_period.value
        assert hac_se > 1.5 * ols_se

    def test_hac_lag_count_can_be_set(self, factor_panel):
        x, names, _ = factor_panel
        r = factor_attribution(x[:, 0] * 0.5, x, names, hac_lags=7)
        assert r.hac_lags == 7
        assert "7 lags" in r.cov_type

    def test_ols_mode_is_labelled(self, factor_panel):
        x, names, _ = factor_panel
        r = factor_attribution(x[:, 0] * 0.5, x, names, use_hac=False)
        assert r.hac_lags is None
        assert "homoskedastic" in r.cov_type

    def test_risk_free_scalar_is_subtracted(self, factor_panel):
        x, names, _ = factor_panel
        gen = np.random.default_rng(7)
        y = 0.001 + 0.004 * gen.standard_normal(len(x))
        with_rf = factor_attribution(y, x, names, rf=0.0002)
        without = factor_attribution(y, x, names)
        assert with_rf.alpha_per_period.value == pytest.approx(
            without.alpha_per_period.value - 0.0002, abs=1e-9
        )

    def test_risk_free_series_is_subtracted(self, factor_panel):
        x, names, _ = factor_panel
        gen = np.random.default_rng(8)
        y = 0.001 + 0.004 * gen.standard_normal(len(x))
        rf = np.full(len(x), 0.0002)
        assert factor_attribution(y, x, names, rf=rf).alpha_per_period.value == pytest.approx(
            factor_attribution(y, x, names, rf=0.0002).alpha_per_period.value
        )

    def test_single_factor_is_accepted_as_1d(self, factor_panel):
        x, _, _ = factor_panel
        r = factor_attribution(0.5 * x[:, 0], x[:, 0], ["Mkt-RF"])
        assert len(r.loadings) == 1

    def test_names_come_from_a_dataframe(self, factor_panel):
        pd = pytest.importorskip("pandas")
        x, names, _ = factor_panel
        frame = pd.DataFrame(x, columns=names)
        r = factor_attribution(0.5 * x[:, 0], frame)
        assert [load.name for load in r.loadings] == names

    def test_default_names_when_none_available(self, factor_panel):
        x, _, _ = factor_panel
        r = factor_attribution(0.5 * x[:, 0], x)
        assert [load.name for load in r.loadings] == ["factor_0", "factor_1", "factor_2"]

    def test_r_squared_is_high_for_pure_exposure(self, factor_panel):
        x, names, _ = factor_panel
        r = factor_attribution(0.9 * x[:, 0], x, names)
        assert r.r_squared > 0.99

    def test_drops_rows_with_missing_data_and_says_so(self, factor_panel):
        x, names, _ = factor_panel
        y = 0.5 * x[:, 0].copy()
        y[:10] = np.nan
        r = factor_attribution(y, x, names)
        assert r.n_obs == len(x) - 10
        assert "dropped" in r.note

    def test_small_sample_is_flagged(self, factor_panel):
        x, names, _ = factor_panel
        r = factor_attribution(0.5 * x[:20, 0], x[:20], names)
        assert not r.alpha_per_period.reliable
        assert "below the 30" in r.note

    def test_loading_lookup_raises_on_unknown_name(self, factor_panel):
        x, names, _ = factor_panel
        with pytest.raises(KeyError):
            factor_attribution(0.5 * x[:, 0], x, names).loading("MOM")

    def test_to_dict_is_json_shaped(self, factor_panel):
        x, names, _ = factor_panel
        d = factor_attribution(0.5 * x[:, 0], x, names).to_dict()
        assert set(d) >= {"alpha_per_period", "loadings", "r_squared", "severity"}
        assert len(d["loadings"]) == 3

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"factor_names": ["a"]}, "3 factor columns"),
            ({"periods_per_year": 0}, "periods_per_year"),
            ({"confidence": 1.0}, "confidence"),
        ],
    )
    def test_rejects_bad_arguments(self, factor_panel, kwargs, match):
        x, _, _ = factor_panel
        with pytest.raises(ValueError, match=match):
            factor_attribution(0.5 * x[:, 0], x, **kwargs)

    def test_rejects_mismatched_lengths(self, factor_panel):
        x, names, _ = factor_panel
        with pytest.raises(ValueError, match="same time axis"):
            factor_attribution(np.zeros(50), x, names)

    def test_rejects_mismatched_risk_free_series(self, factor_panel):
        x, names, _ = factor_panel
        with pytest.raises(ValueError, match="rf has"):
            factor_attribution(0.5 * x[:, 0], x, names, rf=np.zeros(7))

    def test_refuses_an_underdetermined_regression(self):
        with pytest.raises(ValueError, match="cannot support a regression"):
            factor_attribution(np.zeros(3), np.zeros((3, 3)))


class TestVolMatchedBenchmark:
    def test_leverage_matches_the_volatilities(self):
        gen = np.random.default_rng(1)
        bench = 0.01 * gen.standard_normal(1000)
        strat = 2.5 * bench
        r = vol_matched_benchmark(strat, bench)
        assert r.leverage == pytest.approx(2.5, rel=1e-9)

    def test_scaling_leaves_the_sharpe_unchanged(self):
        """Which is why the leverage, not the scaled Sharpe, is the output
        worth reading."""
        gen = np.random.default_rng(2)
        bench = 0.0004 + 0.01 * gen.standard_normal(1000)
        r = vol_matched_benchmark(1.8 * bench, bench)
        assert r.scaled_benchmark_sharpe == pytest.approx(r.benchmark_sharpe)

    def test_levered_beta_is_caught(self):
        """A strategy that is literally the index times a constant."""
        gen = np.random.default_rng(3)
        bench = 0.0004 + 0.01 * gen.standard_normal(1500)
        r = vol_matched_benchmark(2.0 * bench, bench, "SPY")
        assert r.correlation == pytest.approx(1.0)
        assert r.is_levered_beta
        assert r.severity is Severity.CRITICAL
        assert r.excess_sharpe == pytest.approx(0.0, abs=1e-9)

    def test_an_uncorrelated_winner_is_not_flagged(self):
        gen = np.random.default_rng(4)
        bench = 0.0004 + 0.01 * gen.standard_normal(1500)
        strat = 0.0008 + 0.01 * gen.standard_normal(1500)
        r = vol_matched_benchmark(strat, bench)
        assert abs(r.correlation) < 0.2
        assert r.beats_benchmark
        assert not r.is_levered_beta
        assert r.severity is Severity.INFO

    def test_correlated_but_better_is_a_milder_finding(self):
        gen = np.random.default_rng(5)
        bench = 0.0002 + 0.01 * gen.standard_normal(1500)
        strat = bench + 0.0006 + 0.004 * gen.standard_normal(1500)
        r = vol_matched_benchmark(strat, bench)
        assert r.correlation > 0.7
        assert r.beats_benchmark
        assert r.severity is Severity.MEDIUM

    def test_beta_is_the_regression_slope(self):
        gen = np.random.default_rng(6)
        bench = 0.01 * gen.standard_normal(1000)
        strat = 1.4 * bench + 0.002 * gen.standard_normal(1000)
        assert vol_matched_benchmark(strat, bench).beta == pytest.approx(1.4, abs=0.02)

    def test_benchmark_name_is_carried_through(self):
        gen = np.random.default_rng(7)
        b = 0.01 * gen.standard_normal(100)
        assert vol_matched_benchmark(b * 1.1, b, "QQQ").benchmark_name == "QQQ"

    def test_flat_strategy_gives_nan_correlation(self):
        gen = np.random.default_rng(8)
        r = vol_matched_benchmark(np.zeros(100), 0.01 * gen.standard_normal(100))
        assert math.isnan(r.correlation)
        assert r.leverage == 0.0

    def test_to_dict_is_json_shaped(self):
        gen = np.random.default_rng(9)
        b = 0.01 * gen.standard_normal(200)
        d = vol_matched_benchmark(1.5 * b, b).to_dict()
        assert set(d) >= {"leverage", "correlation", "is_levered_beta", "severity"}

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same time axis"):
            vol_matched_benchmark(np.zeros(100), np.zeros(80))

    def test_rejects_a_flat_benchmark(self):
        with pytest.raises(ValueError, match="no variance"):
            vol_matched_benchmark(np.arange(100.0), np.ones(100))

    def test_rejects_tiny_samples(self):
        with pytest.raises(ValueError, match="at least 4"):
            vol_matched_benchmark(np.arange(3.0), np.arange(3.0))
