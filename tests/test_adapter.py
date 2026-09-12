"""The adapter, and the behavioural leakage test on real strategies.

The tests that matter most here are the negative controls. The dangerous
outcome for a behavioural test is not a false alarm - it is a clean bill of
health earned by the corruption never reaching the strategy at all, which
looks identical to a genuinely honest strategy from the outside. So every
"honest strategy comes back clean" assertion below is paired with a leaky
variant that must be caught, and with an assertion that the book the test
judged actually varies.

The leaky variants are installed by monkeypatching the function the strategy
calls through its module global, so each one is exactly the minimal one-line
edit to `strategy.py` without touching the file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qv.adapter import PositionsStrategy, frame_adapter
from qv.leakage.perturbation import perturbation_test
from case_studies.dual_momentum import strategy as dual
from tests.fixtures import sector_strategy as sector

#: Long enough that a 252-session lookback plus a 21-session skip still leaves
#: several years of usable history, so a leak has somewhere to show up.
N_SESSIONS = 1500


def _prices(columns, seed: int = 0, n: int = N_SESSIONS) -> pd.DataFrame:
    """A seeded geometric random walk. No network, no fixtures on disk."""
    gen = np.random.default_rng(seed)
    steps = gen.normal(0.0003, 0.011, (n, len(columns)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=pd.bdate_range("2013-01-02", periods=n),
        columns=list(columns),
    )


@pytest.fixture
def sector_prices() -> pd.DataFrame:
    return _prices(sector.SECTOR_ETFS)


@pytest.fixture
def dual_prices() -> pd.DataFrame:
    return _prices(dual.UNIVERSE, seed=3)


class TestFrameAdapter:
    def test_rebuilds_a_labelled_frame(self, sector_prices):
        strategy = frame_adapter(
            sector.momentum_positions, sector_prices.index, sector_prices.columns
        )
        expected = sector.momentum_positions(sector_prices).to_numpy()
        assert np.array_equal(strategy(sector_prices.to_numpy()), expected)

    def test_uses_only_the_array_it_is_given(self, sector_prices):
        """The whole test rests on this: the adapter must read the array, not
        the frame it was built from. If it closed over the frame, corrupting
        the array would change nothing and every strategy would pass."""
        strategy = frame_adapter(
            sector.momentum_positions, sector_prices.index, sector_prices.columns
        )
        other = sector_prices.iloc[::-1].reset_index(drop=True)
        other.index = sector_prices.index
        assert not np.array_equal(
            strategy(other.to_numpy()), strategy(sector_prices.to_numpy())
        )

    def test_passes_parameters_through(self, sector_prices):
        one = frame_adapter(
            sector.momentum_positions,
            sector_prices.index,
            sector_prices.columns,
            top_n=1,
        )(sector_prices.to_numpy())
        three = frame_adapter(
            sector.momentum_positions,
            sector_prices.index,
            sector_prices.columns,
            top_n=3,
        )(sector_prices.to_numpy())
        assert not np.array_equal(one, three)
        assert np.isclose(np.abs(one).sum(axis=1).max(), 1.0)

    def test_rejects_data_the_labels_do_not_describe(self, sector_prices):
        strategy = frame_adapter(
            sector.momentum_positions, sector_prices.index, sector_prices.columns
        )
        with pytest.raises(ValueError, match="no longer describe the data"):
            strategy(np.zeros((N_SESSIONS, len(sector.SECTOR_ETFS) + 1)))

    def test_rejects_duplicate_column_labels(self, sector_prices):
        with pytest.raises(ValueError, match="must be unique"):
            frame_adapter(
                sector.momentum_positions, sector_prices.index, ["A", "A", "B"]
            )

    @pytest.mark.parametrize(
        "index,columns,match",
        [([], ["A"], "non-empty index"), ([1, 2], [], "at least one column")],
    )
    def test_rejects_empty_labels(self, index, columns, match):
        with pytest.raises(ValueError, match=match):
            frame_adapter(sector.momentum_positions, index, columns)

    def test_both_real_strategies_satisfy_the_protocol(self):
        assert isinstance(sector.momentum_positions, PositionsStrategy)
        assert isinstance(dual.dual_momentum_positions, PositionsStrategy)


def _sector_strategy(prices, **params):
    return frame_adapter(
        sector.momentum_positions, prices.index, prices.columns, **params
    )


def _dual_strategy(prices, **params):
    return frame_adapter(
        dual.dual_momentum_positions, prices.index, prices.columns, **params
    )


class TestSectorMomentumTierTwo:
    def test_the_honest_strategy_does_not_read_the_future(self, sector_prices):
        result = perturbation_test(
            _sector_strategy(sector_prices), sector_prices.to_numpy()
        )
        assert not result.leaked
        assert result.n_changed == 0
        assert result.to_finding() is None

    def test_the_book_it_judged_actually_varies(self, sector_prices):
        """Anti-vacuity. A strategy flat for the whole sample would come back
        clean for the least interesting reason available."""
        book = _sector_strategy(sector_prices)(sector_prices.to_numpy())
        assert book.shape == (N_SESSIONS, len(sector.SECTOR_ETFS))
        assert len(np.unique(book, axis=0)) > 10
        assert np.abs(book).sum() > 0

    def test_a_book_is_judged_per_asset_not_by_gross_exposure(self, sector_prices):
        """The reason a book must not be collapsed to one number per row.

        This rotation is equal-weight top-3, so its gross exposure is 1.0 on
        essentially every row. Any scalar summary of a row is therefore blind
        to a leak that changes *which* names are held, which is the only kind
        of leak a rotation can have.
        """
        book = _sector_strategy(sector_prices)(sector_prices.to_numpy())
        active = book[np.abs(book).sum(axis=1) > 0]
        assert np.allclose(active.sum(axis=1), 1.0)

    def test_a_full_sample_statistic_in_the_score_is_caught(
        self, sector_prices, monkeypatch
    ):
        honest = sector.momentum_score

        def leaky(prices, lookback=252, skip=21):
            scores = honest(prices, lookback, skip)
            # Standardised over the whole sample, so every row depends on
            # every other row - the classic full-sample-scaler leak.
            return (scores - scores.mean()) / scores.std()

        monkeypatch.setattr(sector, "momentum_score", leaky)
        result = perturbation_test(
            _sector_strategy(sector_prices), sector_prices.to_numpy()
        )
        assert result.leaked
        assert result.lookahead_span > 100
        assert result.to_finding().id == "LEAK-BEHAVIOURAL"

    def test_a_score_read_from_the_future_is_caught(self, sector_prices, monkeypatch):
        honest = sector.momentum_score

        def leaky(prices, lookback=252, skip=21):
            # Reaches `63 - skip` sessions past the decision point.
            return honest(prices, lookback, skip).shift(-63)

        monkeypatch.setattr(sector, "momentum_score", leaky)
        result = perturbation_test(
            _sector_strategy(sector_prices), sector_prices.to_numpy()
        )
        assert result.leaked
        assert result.lookahead_span >= 2
        assert len(result.cuts_that_leaked) > len(result.cuts_tested) // 2

    def test_reports_a_detection_floor_near_the_rebalance_interval(self, sector_prices):
        """A clean result is only worth what the test could have detected.

        This strategy revises monthly, so it cannot betray a leak shorter than
        a month: the offending signal is overwritten at the next rebalance
        before anything downstream sees it. The report has to say so, or a
        reader takes "clean" for more than it is.
        """
        result = perturbation_test(
            _sector_strategy(sector_prices), sector_prices.to_numpy()
        )
        assert 15 <= result.detection_floor <= 30
        assert "more than about" in result.clean_claim

    def test_a_same_day_decision_is_outside_this_tests_reach(
        self, sector_prices, monkeypatch
    ):
        """A documented blind spot, pinned so it cannot be mistaken for a pass.

        Dropping the trade shift means the position held into `t+1` is set from
        data at `t`. That is exactly what the adapter protocol *defines* as
        correct, so no pre-cut signal moves and the test reports clean. Whether
        a strategy may act on the close it just observed is a Tier 1
        accounting question, not a look-ahead question, and this test does not
        police it.
        """

        honest = sector.momentum_positions

        def same_day(prices, lookback=252, skip=21, top_n=3):
            return honest(prices, lookback, skip, top_n).shift(-1)

        strategy = frame_adapter(same_day, sector_prices.index, sector_prices.columns)
        assert not perturbation_test(strategy, sector_prices.to_numpy()).leaked


class TestDualMomentumTierTwo:
    def test_the_honest_strategy_does_not_read_the_future(self, dual_prices):
        result = perturbation_test(_dual_strategy(dual_prices), dual_prices.to_numpy())
        assert not result.leaked
        assert result.n_changed == 0

    def test_the_book_it_judged_actually_varies(self, dual_prices):
        book = _dual_strategy(dual_prices)(dual_prices.to_numpy())
        assert book.shape == (N_SESSIONS, len(dual.UNIVERSE))
        assert len(np.unique(book.round(6), axis=0)) > 10

    @pytest.mark.parametrize("mode", ["shuffle", "noise", "constant", "reverse"])
    def test_no_single_corruption_mode_produces_a_false_leak(self, dual_prices, mode):
        """`noise` drives prices negative, which flows into `pct_change` and
        the volatility overlay. Nothing may raise, and nothing may report a
        leak that is not there."""
        result = perturbation_test(
            _dual_strategy(dual_prices), dual_prices.to_numpy(), modes=(mode,)
        )
        assert not result.leaked

    def test_the_honest_book_stays_finite_under_corruption(self, dual_prices):
        book = _dual_strategy(dual_prices)(dual_prices.to_numpy())
        assert np.isfinite(book).all()

    def test_a_centred_volatility_window_is_caught(self, dual_prices, monkeypatch):
        def centred(returns, window):
            return returns.rolling(window, center=True).std() * np.sqrt(252)

        monkeypatch.setattr(dual, "realised_volatility", centred)
        result = perturbation_test(_dual_strategy(dual_prices), dual_prices.to_numpy())
        assert result.leaked
        assert result.lookahead_span >= 2

    def test_a_full_sample_statistic_in_the_score_is_caught(
        self, dual_prices, monkeypatch
    ):
        honest = dual.momentum_score

        def leaky(prices, lookback, skip):
            scores = honest(prices, lookback, skip)
            return (scores - scores.mean()) / scores.std()

        monkeypatch.setattr(dual, "momentum_score", leaky)
        result = perturbation_test(_dual_strategy(dual_prices), dual_prices.to_numpy())
        assert result.leaked
        assert result.lookahead_span > 100


class TestClosingOverTheFrameHidesALeak:
    """The trap `adapter_protocol.md` calls the dangerous one, made executable.

    The same genuinely leaky strategy is tested twice. Through the adapter it
    is caught. Closed over the original frame it comes back clean, because the
    corruption never reaches it. Without this test the difference between the
    two is a paragraph of documentation that nothing enforces.
    """

    @staticmethod
    def _leaky_book(prices: pd.DataFrame) -> pd.DataFrame:
        scores = sector.momentum_score(prices, 252, 21)
        standardised = (scores - scores.mean()) / scores.std()
        ranks = standardised.rank(axis=1, ascending=False, na_option="bottom")
        return (ranks <= 3).astype(float).div(3.0).shift(1).fillna(0.0)

    def test_the_adapter_exposes_it(self, sector_prices):
        strategy = frame_adapter(
            self._leaky_book, sector_prices.index, sector_prices.columns
        )
        assert perturbation_test(strategy, sector_prices.to_numpy()).leaked

    def test_closing_over_the_frame_hides_it(self, sector_prices):
        captured = sector_prices.copy()
        result = perturbation_test(
            lambda _ignored: self._leaky_book(captured).to_numpy(),
            sector_prices.to_numpy(),
        )
        assert not result.leaked


class TestAuditWiring:
    def test_a_cross_sectional_strategy_reaches_tier_two(self, sector_prices):
        from qv.audit import AuditInputs
        from qv.types import Tier

        positions = sector.momentum_positions(sector_prices)
        returns, _ = sector.strategy_returns(sector_prices, positions)
        inputs = AuditInputs(
            returns=returns.to_numpy(),
            positions=positions.to_numpy(),
            strategy=_sector_strategy(sector_prices),
            strategy_data=sector_prices.to_numpy(),
        )
        assert inputs.tier is Tier.CALLABLE

    def test_the_behavioural_test_runs_and_finds_nothing(self, sector_prices):
        from qv.audit import AuditInputs, run_audit

        positions = sector.momentum_positions(sector_prices)
        returns, _ = sector.strategy_returns(sector_prices, positions)
        report = run_audit(
            AuditInputs(
                returns=returns.to_numpy(),
                positions=positions.to_numpy(),
                strategy=_sector_strategy(sector_prices),
                strategy_data=sector_prices.to_numpy(),
                n_boot=200,
            )
        )
        assert report.sections["perturbation"]["leaked"] is False
        assert not any("Behavioural leakage" in gap for gap in report.not_tested)
        assert not any(f.id == "LEAK-BEHAVIOURAL" for f in report.findings)

    def test_a_broken_adapter_is_reported_not_silently_skipped(self, sector_prices):
        """`run_audit` catches the ValueError, so the only thing standing
        between a broken adapter and an unearned clean report is that the
        reason lands in `not_tested`."""
        from qv.audit import AuditInputs, run_audit

        positions = sector.momentum_positions(sector_prices)
        returns, _ = sector.strategy_returns(sector_prices, positions)
        report = run_audit(
            AuditInputs(
                returns=returns.to_numpy(),
                positions=positions.to_numpy(),
                strategy=lambda d: np.zeros(3),
                strategy_data=sector_prices.to_numpy(),
                n_boot=200,
            )
        )
        assert "perturbation" not in report.sections
        assert any("Behavioural leakage test" in gap for gap in report.not_tested)
