"""The manifest schema, the grid runner, and the path from one to a report.

Nothing here touches the network: every test either uses a synthetic price
frame or a manifest pointed at one. The two tests that matter most are
`test_reproduces_the_hand_built_matrix`, which is the evidence that moving the
grid running into the package changed no number, and
`test_the_pipeline_reaches_tier_two`, which is the whole point of the module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from qv.manifest import (
    PERIODS_PER_YEAR,
    ResearchManifest,
    SearchSpec,
    load_manifest,
)
from qv.pipeline import audit_from_manifest, load_adapter
from qv.trials import parameter_grid, run_parameter_grid
from qv.types import Tier

UNIVERSE = ["AAA", "BBB", "CCC", "DDD"]


def _prices(n: int = 900, seed: int = 5) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    steps = gen.normal(0.0003, 0.011, (n, len(UNIVERSE)))
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(steps, axis=0)),
        index=pd.bdate_range("2015-01-02", periods=n),
        columns=UNIVERSE,
    )


def _positions(prices: pd.DataFrame, lookback: int = 120, top_n: int = 2) -> pd.DataFrame:
    """A plain trailing-momentum rotation. Honest by construction."""
    score = prices / prices.shift(lookback) - 1.0
    ranks = score.rank(axis=1, ascending=False, na_option="bottom")
    weights = (ranks <= top_n).astype(float) / top_n
    weights[score.isna().all(axis=1)] = 0.0
    return weights.shift(1).fillna(0.0)


AXES = {"lookback": [60, 120, 180], "top_n": [1, 2, 3]}
CHOSEN = {"lookback": 120, "top_n": 2}


class TestParameterGrid:
    def test_crosses_every_axis_in_a_fixed_order(self):
        grid = parameter_grid({"a": [1, 2], "b": ["x", "y", "z"]})
        assert len(grid) == 6
        assert grid[0] == {"a": 1, "b": "x"}
        assert grid[-1] == {"a": 2, "b": "z"}

    @pytest.mark.parametrize(
        "axes,match",
        [({}, "at least one axis"), ({"a": []}, "no values")],
    )
    def test_rejects_empty(self, axes, match):
        with pytest.raises(ValueError, match=match):
            parameter_grid(axes)


@pytest.fixture(scope="module")
def grid_result():
    return run_parameter_grid(_positions, _prices(), AXES, CHOSEN)


class TestRunParameterGrid:
    def test_shapes_line_up_with_the_axes(self, grid_result):
        assert grid_result.trial_returns.shape[1] == 9
        assert grid_result.parameter_scores.shape == (3, 3)
        assert grid_result.n_trials == 9

    def test_locates_the_declared_configuration(self, grid_result):
        assert grid_result.chosen_index == (1, 1)
        assert grid_result.grid[grid_result.chosen_column] == CHOSEN
        assert grid_result.chosen_score == pytest.approx(
            grid_result.parameter_scores[1, 1], abs=0, rel=0
        )

    def test_the_surface_matches_the_matrix_columns(self, grid_result):
        """The reshape is what ties a grid column to a point on the surface. If
        the two ever transposed, the plateau test would judge the wrong point
        and nothing would look wrong."""
        flat = grid_result.parameter_scores.reshape(-1)
        for column, params in enumerate(grid_result.grid):
            index = tuple(AXES[name].index(params[name]) for name in AXES)
            assert grid_result.parameter_scores[index] == pytest.approx(
                flat[column], nan_ok=True
            )

    def test_reproduces_the_hand_built_matrix(self):
        """The equivalence check. The old drivers built this matrix inline; the
        package must produce exactly the same numbers, not merely similar ones."""
        prices = _prices()
        expected = []
        for params in parameter_grid(AXES):
            positions = _positions(prices, **params)
            asset_returns = prices.pct_change().fillna(0.0)
            aligned = positions.reindex_like(asset_returns).fillna(0.0)
            expected.append((aligned * asset_returns).sum(axis=1).to_numpy())
        result = run_parameter_grid(_positions, prices, AXES, CHOSEN)
        assert np.array_equal(np.column_stack(expected), result.trial_returns)

    def test_a_flat_configuration_cannot_win(self):
        """A column of identical values has a standard deviation near 1e-16,
        not 0, so an unscreened Sharpe of 1e14 would take the grid maximum."""

        def sometimes_flat(prices, top_n=2, **_):
            if top_n == 1:
                return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            return _positions(prices, top_n=top_n)

        result = run_parameter_grid(
            sometimes_flat, _prices(), {"top_n": [1, 2, 3]}, {"top_n": 2}
        )
        assert np.isnan(result.parameter_scores[0])
        assert result.best_score < 100

    def test_rejects_a_chosen_point_outside_the_grid(self):
        with pytest.raises(ValueError, match="is not in the grid"):
            run_parameter_grid(
                _positions, _prices(), AXES, {"lookback": 999, "top_n": 2}
            )

    def test_rejects_a_chosen_point_that_omits_an_axis(self):
        with pytest.raises(ValueError, match="does not declare"):
            run_parameter_grid(_positions, _prices(), AXES, {"lookback": 120})

    def test_summary_names_the_best_configuration(self, grid_result):
        assert "chosen" in grid_result.summary()
        assert "best" in grid_result.summary()


class TestManifestSchema:
    def _write(self, tmp_path, **overrides):
        payload = {
            "name": "test strategy",
            "data": {
                "universe": UNIVERSE,
                "start_date": "2015-01-01",
                "end_date": "2018-12-31",
                "asset_class": "us_large_cap_etf",
                "load_factors": False,
            },
            "chosen_parameters": dict(CHOSEN),
            "search": {"axes": AXES},
        }
        payload.update(overrides)
        path = tmp_path / "research_manifest.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_round_trips(self, tmp_path):
        manifest = load_manifest(self._write(tmp_path))
        assert manifest.name == "test strategy"
        assert manifest.data.universe == UNIVERSE
        assert manifest.periods_per_year == 252
        assert manifest.source_path is not None

    def test_an_open_end_date_is_refused(self, tmp_path):
        """A reproducible report has to reproduce the same numbers next month."""
        with pytest.raises(ValueError, match="end_date must be pinned"):
            load_manifest(
                self._write(
                    tmp_path,
                    data={
                        "universe": UNIVERSE,
                        "start_date": "2015-01-01",
                        "end_date": "today",
                    },
                )
            )

    def test_a_chosen_point_must_cover_every_axis(self, tmp_path):
        with pytest.raises(ValueError, match="does not declare"):
            load_manifest(
                self._write(tmp_path, chosen_parameters={"lookback": 120})
            )

    def test_rejects_an_unknown_frequency(self, tmp_path):
        with pytest.raises(ValueError, match="unknown frequency"):
            load_manifest(
                self._write(
                    tmp_path,
                    data={
                        "universe": UNIVERSE,
                        "start_date": "2015-01-01",
                        "end_date": "2018-12-31",
                        "frequency": "fortnightly",
                    },
                )
            )

    def test_rejects_a_repeated_instrument(self, tmp_path):
        with pytest.raises(ValueError, match="repeats"):
            load_manifest(
                self._write(
                    tmp_path,
                    data={
                        "universe": ["AAA", "AAA"],
                        "start_date": "2015-01-01",
                        "end_date": "2018-12-31",
                    },
                )
            )

    def test_rejects_a_manifest_with_no_universe(self, tmp_path):
        with pytest.raises(ValueError, match="name the instruments"):
            load_manifest(
                self._write(
                    tmp_path,
                    data={"start_date": "2015-01-01", "end_date": "2018-12-31"},
                )
            )

    @pytest.mark.parametrize(
        "data,expected",
        [
            ({"ticker": "SPY"}, ["SPY"]),
            ({"tickers": ["A", "B"]}, ["A", "B"]),
            (
                {"risk_assets": ["A", "B"], "defensive_assets": ["C"]},
                ["A", "B", "C"],
            ),
        ],
    )
    def test_accepts_the_universe_under_its_older_names(self, tmp_path, data, expected):
        """The manifests already committed use all three spellings."""
        manifest = load_manifest(
            self._write(
                tmp_path,
                data={**data, "start_date": "2015-01-01", "end_date": "2018-12-31"},
                search={},
                chosen_parameters={},
            )
        )
        assert manifest.data.universe == expected

    def test_every_declared_frequency_annualises(self):
        assert PERIODS_PER_YEAR["monthly"] == 12
        assert PERIODS_PER_YEAR["quarterly"] == 4

    def test_missing_manifest_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such manifest"):
            load_manifest(tmp_path / "absent.yaml")

    def test_a_manifest_that_is_not_a_mapping_is_refused(self, tmp_path):
        path = tmp_path / "research_manifest.yaml"
        path.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_manifest(path)


class TestDeclaredTrials:
    """A grid is a lower bound on a search, so the larger number wins."""

    def test_the_grid_is_used_when_nothing_is_declared(self):
        assert SearchSpec().declared_trials(24) == 24

    def test_a_larger_declaration_wins(self):
        """The researcher tried things they did not keep. Overstating is the
        honest direction, and the correction is logarithmic in the count."""
        assert SearchSpec(n_trials=400).declared_trials(24) == 400

    def test_an_understated_declaration_does_not_shrink_the_grid(self):
        assert SearchSpec(n_trials=2).declared_trials(24) == 24

    def test_no_grid_and_no_declaration_is_none(self):
        assert SearchSpec().declared_trials(0) is None

    def test_rejects_a_non_positive_count(self):
        with pytest.raises(ValueError, match="at least 1"):
            SearchSpec(n_trials=0)


class TestLoadAdapter:
    def test_imports_a_function_from_a_file(self, tmp_path):
        path = tmp_path / "qv_adapter.py"
        path.write_text("def positions(prices, **p):\n    return prices\n", encoding="utf-8")
        assert callable(load_adapter(f"{path}:positions"))

    def test_resolves_relative_to_a_base(self, tmp_path):
        path = tmp_path / "qv_adapter.py"
        path.write_text("def positions(prices, **p):\n    return prices\n", encoding="utf-8")
        assert callable(load_adapter("qv_adapter.py:positions", base=tmp_path))

    def test_imports_from_an_installed_module(self):
        assert load_adapter("qv.adapter:frame_adapter") is not None

    @pytest.mark.parametrize(
        "reference,exception,match",
        [
            ("no_colon_here", ValueError, "must be"),
            ("qv.adapter:not_a_thing", AttributeError, "has no attribute"),
            ("qv.adapter:__doc__", TypeError, "not callable"),
        ],
    )
    def test_rejects_bad_references(self, reference, exception, match):
        with pytest.raises(exception, match=match):
            load_adapter(reference)

    def test_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no such adapter file"):
            load_adapter(f"{tmp_path / 'absent.py'}:positions")


def _manifest() -> ResearchManifest:
    return ResearchManifest.model_validate(
        {
            "name": "synthetic rotation",
            "data": {
                "universe": UNIVERSE,
                "start_date": "2015-01-01",
                "end_date": "2018-12-31",
                "asset_class": "us_large_cap_etf",
                # Factors would mean a network call; the audit is exercised
                # here for its wiring, not its attribution.
                "load_factors": False,
                "universe_point_in_time": True,
                "universe_note": "Four synthetic instruments, fixed in advance.",
            },
            "chosen_parameters": dict(CHOSEN),
            "search": {"axes": AXES, "n_trials": 40},
        }
    )


@pytest.fixture(scope="module")
def run():
    """One full pass, with the prices supplied so nothing touches the network."""
    return audit_from_manifest(
        _manifest(), adapter=_positions, prices=_prices(), n_boot=200
    )


class TestAuditFromManifest:
    @staticmethod
    def _manifest():
        return _manifest()

    def test_the_pipeline_reaches_tier_two(self, run):
        """The reason the module exists: a manifest plus an adapter is enough
        for the behavioural leakage test, with no glue code from the caller."""
        assert run.report.tier is Tier.CALLABLE
        assert run.report.sections["perturbation"]["leaked"] is False
        assert not any(
            "Behavioural leakage" in gap for gap in run.report.not_tested
        )

    def test_it_rebuilds_the_trial_matrix_and_the_surface(self, run):
        assert run.grid is not None
        assert "pbo" in run.report.sections
        assert "parameters" in run.report.sections

    def test_it_audits_the_declared_trial_count_not_the_grid_size(self, run):
        """40 was declared against a grid of 9, and the declaration is the
        floor that gets audited."""
        assert run.report.sections["selection"]["deflated_sharpe"]["n_trials"] == 40

    def test_it_records_the_universe_declaration(self, run):
        assert not any("Survivorship" in gap for gap in run.report.not_tested)

    def test_a_manifest_without_an_adapter_is_refused(self):
        with pytest.raises(ValueError, match="no strategy adapter"):
            audit_from_manifest(self._manifest(), adapter=None)

    def test_rejects_a_price_frame_missing_a_declared_instrument(self):
        with pytest.raises(ValueError, match="is missing"):
            audit_from_manifest(
                self._manifest(),
                adapter=_positions,
                prices=_prices().drop(columns=["DDD"]),
                n_boot=200,
            )


class TestPipelineLoaders:
    """The benchmark and factor branches, with the loaders faked.

    These assemble half of `AuditInputs` - the risk-free basis every
    risk-adjusted statistic runs on, and the factors the attribution regresses
    against - so leaving them to the one network test would mean the most
    consequential wiring in the module was never checked.
    """

    @staticmethod
    def _fake_loaders(monkeypatch, prices):
        from qv.data.loaders import CachedFrame

        gen = np.random.default_rng(11)
        index = prices.index

        def fake_prices(ticker, start=None, end=None, **kwargs):
            series = pd.Series(
                100.0 * np.exp(np.cumsum(gen.normal(0.0004, 0.01, len(index)))),
                index=index,
                name="close",
            )
            return CachedFrame(
                series.to_frame("close"), "fake", "2020-01-01", True, None
            )

        def fake_factors(start=None, end=None, **kwargs):
            frame = pd.DataFrame(
                {
                    "Mkt-RF": gen.normal(0.0003, 0.01, len(index)),
                    "SMB": gen.normal(0.0, 0.004, len(index)),
                    "HML": gen.normal(0.0, 0.004, len(index)),
                    "Mom": gen.normal(0.0, 0.004, len(index)),
                    "RF": np.full(len(index), 0.00006),
                },
                index=index,
            )
            # A ragged tail, as the real file has: Ken French publishes on a
            # lag, so the last few sessions are NaN and must be masked out
            # rather than propagated into the regression.
            frame.iloc[-5:, :] = np.nan
            return CachedFrame(frame, "fake French", "2024-12-31", True, None)

        monkeypatch.setattr("qv.pipeline.load_prices", fake_prices)
        monkeypatch.setattr("qv.pipeline.load_fama_french", fake_factors)

    @staticmethod
    def _manifest_with(**data_overrides):
        payload = {
            "name": "loader wiring",
            "data": {
                "universe": UNIVERSE,
                "start_date": "2015-01-01",
                "end_date": "2018-12-31",
                "asset_class": "us_large_cap_etf",
                "universe_point_in_time": True,
                **data_overrides,
            },
            "chosen_parameters": dict(CHOSEN),
            "search": {"axes": AXES},
        }
        return ResearchManifest.model_validate(payload)

    def test_loads_the_benchmark_and_the_factors(self, monkeypatch):
        prices = _prices()
        self._fake_loaders(monkeypatch, prices)
        run = audit_from_manifest(
            self._manifest_with(benchmark="SPY", load_factors=True),
            adapter=_positions,
            prices=prices,
            n_boot=200,
        )
        assert run.benchmark_returns is not None
        assert "attribution" in run.report.sections
        assert "benchmark" in run.vintages
        assert "factors" in run.vintages

    def test_the_risk_free_basis_reaches_the_report(self, monkeypatch):
        """Every risk-adjusted statistic runs in excess of cash. If the RF
        column stopped arriving, every Sharpe here would quietly rise."""
        prices = _prices()
        self._fake_loaders(monkeypatch, prices)
        run = audit_from_manifest(
            self._manifest_with(benchmark="SPY", load_factors=True),
            adapter=_positions,
            prices=prices,
            n_boot=200,
        )
        assert run.risk_free_mean > 0
        assert run.report.sections["conventions"]["risk_free_supplied"] is True
        assert not any(
            "Risk-free adjustment" in gap for gap in run.report.not_tested
        )

    def test_a_ragged_factor_tail_is_masked_not_propagated(self, monkeypatch):
        """Ken French publishes on a lag, so the last sessions are NaN."""
        prices = _prices()
        self._fake_loaders(monkeypatch, prices)
        run = audit_from_manifest(
            self._manifest_with(load_factors=True),
            adapter=_positions,
            prices=prices,
            n_boot=200,
        )
        assert run.vintages["factor_alignment"].startswith("5 sessions dropped")
        # The sample was narrowed before anything was computed, so the returns
        # and the factors that explain them describe the same axis.
        assert len(run.returns) == run.report.n_obs == len(prices) - 5
        assert np.isfinite(run.report.sections["attribution"]["alpha_tstat"])

    def test_no_factors_means_a_stated_total_return_basis(self, monkeypatch):
        prices = _prices()
        self._fake_loaders(monkeypatch, prices)
        run = audit_from_manifest(
            self._manifest_with(load_factors=False),
            adapter=_positions,
            prices=prices,
            n_boot=200,
        )
        assert run.risk_free_mean == 0.0
        assert any("Risk-free adjustment" in gap for gap in run.report.not_tested)

    def test_it_loads_the_universe_when_no_frame_is_supplied(self, monkeypatch):
        prices = _prices()
        self._fake_loaders(monkeypatch, prices)
        monkeypatch.setattr(
            "qv.pipeline.load_universe",
            lambda *a, **k: __import__("qv.data.loaders", fromlist=["CachedFrame"]).CachedFrame(
                prices, "fake", "2020-01-01", True, None
            ),
        )
        run = audit_from_manifest(
            self._manifest_with(load_factors=False), adapter=_positions, n_boot=200
        )
        assert run.vintages["prices"] == "2020-01-01"
        assert len(run.prices) == len(prices)


class TestPriceFrameFromDisk:
    """`data.price_frame`, so no vendor is privileged."""

    def _write(self, tmp_path, frame, name="prices.csv"):
        path = tmp_path / name
        if name.endswith(".parquet"):
            frame.to_parquet(path)
        else:
            frame.to_csv(path)
        return path

    def test_reads_a_wide_csv(self, tmp_path):
        from qv.data.loaders import read_price_frame

        prices = _prices(n=100)
        loaded = read_price_frame(self._write(tmp_path, prices))
        assert list(loaded.columns) == UNIVERSE
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert np.allclose(loaded.to_numpy(), prices.to_numpy())

    def test_reads_a_wide_parquet(self, tmp_path):
        from qv.data.loaders import read_price_frame

        prices = _prices(n=100)
        loaded = read_price_frame(self._write(tmp_path, prices, "prices.parquet"))
        assert np.allclose(loaded.to_numpy(), prices.to_numpy())

    def test_pivots_long_format(self, tmp_path):
        from qv.data.loaders import read_price_frame

        prices = _prices(n=60)
        long = prices.stack().rename("close").reset_index()
        long.columns = ["date", "ticker", "close"]
        long = long.set_index("date")
        loaded = read_price_frame(self._write(tmp_path, long, "long.csv"))
        assert sorted(loaded.columns) == sorted(UNIVERSE)

    def test_a_long_file_missing_the_price_column_says_so(self, tmp_path):
        from qv.data.loaders import read_price_frame

        frame = pd.DataFrame(
            {"ticker": ["AAA", "BBB"], "mid": [1.0, 2.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        with pytest.raises(ValueError, match="no 'close' column"):
            read_price_frame(self._write(tmp_path, frame, "bad.csv"))

    def test_missing_file_says_so(self, tmp_path):
        from qv.data.loaders import read_price_frame

        with pytest.raises(FileNotFoundError, match="no such price file"):
            read_price_frame(tmp_path / "absent.csv")

    def test_the_manifest_route_uses_it(self, tmp_path, monkeypatch):
        prices = _prices(n=700)
        self._write(tmp_path, prices)
        manifest = ResearchManifest.model_validate(
            {
                "name": "from disk",
                "data": {
                    "universe": UNIVERSE,
                    "start_date": "2015-01-01",
                    "end_date": "2019-12-31",
                    "price_frame": "prices.csv",
                    "load_factors": False,
                    "universe_point_in_time": True,
                },
                "chosen_parameters": dict(CHOSEN),
                "search": {"axes": AXES},
            }
        )
        manifest.source_path = tmp_path / "research_manifest.yaml"
        run = audit_from_manifest(manifest, adapter=_positions, n_boot=200)
        assert run.vintages["prices"].startswith("local file")
        assert run.report.tier is Tier.CALLABLE
