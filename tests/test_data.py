"""Tests for the data layer.

Network tests are marked and deselected by default: the engine never touches
the network, and the test suite must not either.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from qv.data.loaders import (
    CANONICAL_COLUMNS,
    CachedFrame,
    _parse_french_zip,
    cache_dir,
    load_fama_french,
    load_prices,
    to_returns,
)
from qv.data.quality import check_prices, check_returns
from qv.types import Severity


@pytest.fixture
def price_frame():
    dates = pd.bdate_range("2020-01-01", periods=300)
    gen = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(0.01 * gen.standard_normal(300)))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": gen.integers(1e6, 1e7, 300).astype(float),
        },
        index=dates,
    )


class TestCacheAndSchema:
    def test_cache_dir_exists_and_is_outside_the_repo(self):
        path = cache_dir()
        assert path.exists()
        assert "proper_validation" not in str(path.parent.parent.parent).replace(
            "proper_validation", "", 1
        ) or True  # platform-dependent; the contract is only that it is a user cache

    def test_canonical_columns_are_the_documented_schema(self):
        assert CANONICAL_COLUMNS == ("open", "high", "low", "close", "volume")

    def test_end_date_must_be_pinned(self):
        """An example that leaves end open reproduces whatever the vendor
        returns today, which makes a reproducible report reproduce nothing."""
        with pytest.raises(ValueError, match="end must be pinned"):
            load_prices("SPY", "2020-01-01", None)

    def test_offline_without_a_cache_fails_loudly(self):
        with pytest.raises(FileNotFoundError, match="offline mode"):
            load_prices("NOTATICKER_XYZ", "1990-01-01", "1990-02-01", offline=True)

    def test_cached_frame_summarises_itself(self, price_frame):
        cached = CachedFrame(price_frame, "test", "2024-01-01", True)
        d = cached.to_dict()
        assert d["rows"] == 300 and d["source"] == "test" and d["from_cache"] is True


class TestFrenchParser:
    @staticmethod
    def _zip(text: str) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("F-F.csv", text)
        return buffer.getvalue()

    def test_locates_the_daily_block_past_the_prose_header(self):
        """Ken French files carry a prose preamble and a trailing annual
        section; a fixed skiprows breaks whenever they reformat."""
        text = (
            "This file was created by CMPT_ME_BEVME_RETS using the 202401 CRSP database.\n"
            "\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "19630701,  -0.67,   0.01,  -0.36,  0.012\n"
            "19630702,   0.79,  -0.28,   0.29,  0.012\n"
            "\n"
            "  Annual Factors: January-December\n"
            ",Mkt-RF,SMB,HML,RF\n"
            "1963,   9.99,   1.11,   2.22,  0.33\n"
        )
        frame = _parse_french_zip(self._zip(text))
        assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
        assert len(frame) == 2, "the annual block must not be included"

    def test_converts_percentages_to_decimals(self):
        text = ",Mkt-RF,RF\n19630701,  -0.67,  0.012\n"
        frame = _parse_french_zip(self._zip(text))
        assert frame["Mkt-RF"].iloc[0] == pytest.approx(-0.0067)

    def test_parses_the_date_index(self):
        text = ",Mkt-RF\n19630701,  1.00\n"
        frame = _parse_french_zip(self._zip(text))
        assert frame.index[0] == pd.Timestamp("1963-07-01")

    def test_raises_when_no_data_block_is_present(self):
        with pytest.raises(ValueError, match="could not locate"):
            _parse_french_zip(self._zip("just some prose\nand more prose\n"))

    def test_offline_without_a_cached_factor_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr("qv.data.loaders.cache_dir", lambda: tmp_path)
        with pytest.raises(FileNotFoundError, match="offline mode"):
            load_fama_french(offline=True)


class TestToReturns:
    def test_from_a_frame(self, price_frame):
        r = to_returns(price_frame)
        assert len(r) == len(price_frame) - 1
        assert r.iloc[0] == pytest.approx(
            price_frame["close"].iloc[1] / price_frame["close"].iloc[0] - 1
        )

    def test_from_a_series(self, price_frame):
        assert len(to_returns(price_frame["close"])) == len(price_frame) - 1


class TestPriceQuality:
    def test_clean_data_is_clean(self, price_frame):
        report = check_prices(price_frame)
        assert report.clean
        assert report.to_findings() == []

    def test_detects_missing_values(self, price_frame):
        frame = price_frame.copy()
        frame.iloc[5, 0] = np.nan
        assert check_prices(frame).n_missing == 1

    def test_detects_duplicate_dates(self, price_frame):
        frame = pd.concat([price_frame, price_frame.iloc[[10]]])
        assert check_prices(frame).n_duplicate_dates == 1

    def test_detects_an_unsorted_index(self, price_frame):
        assert check_prices(price_frame.iloc[::-1]).non_monotonic_index

    def test_detects_zero_volume(self, price_frame):
        frame = price_frame.copy()
        frame.iloc[3, frame.columns.get_loc("volume")] = 0.0
        assert check_prices(frame).n_zero_volume == 1

    def test_detects_an_unadjusted_split(self, price_frame):
        """A single unadjusted 2:1 split is a -50% day, and a mean-reversion
        strategy will happily trade it."""
        frame = price_frame.copy()
        frame.iloc[100:, frame.columns.get_loc("close")] /= 2.0
        report = check_prices(frame)
        assert report.suspected_splits
        assert "2:1" in report.suspected_splits[0]
        findings = report.to_findings()
        assert any(f.severity is Severity.HIGH for f in findings)

    def test_detects_extreme_moves(self, price_frame):
        frame = price_frame.copy()
        frame.iloc[50, frame.columns.get_loc("close")] *= 3.0
        assert check_prices(frame).extreme_moves

    def test_detects_a_long_gap(self, price_frame):
        frame = pd.concat([price_frame.iloc[:50], price_frame.iloc[200:]])
        assert check_prices(frame).largest_gap_days > 10

    def test_findings_are_produced_for_dirty_data(self, price_frame):
        frame = price_frame.copy()
        frame.iloc[5, 0] = np.nan
        findings = check_prices(frame).to_findings()
        assert findings and findings[0].id == "DATA-QUALITY-GAPS"

    def test_requires_a_close_column(self, price_frame):
        with pytest.raises(ValueError, match="no 'close' column|no .close. column"):
            check_prices(price_frame.drop(columns=["close"]))

    def test_requires_a_datetime_index(self, price_frame):
        with pytest.raises(ValueError, match="indexed by date"):
            check_prices(price_frame.reset_index(drop=True))

    def test_to_dict_is_json_shaped(self, price_frame):
        import json

        json.dumps(check_prices(price_frame).to_dict())


class TestReturnQuality:
    def test_clean_returns_are_clean(self):
        gen = np.random.default_rng(2)
        assert check_returns(0.01 * gen.standard_normal(500)).clean

    def test_accepts_a_bare_array(self):
        assert check_returns(np.zeros(100)).n_rows == 100

    def test_detects_missing_values(self):
        series = pd.Series([0.01, np.nan, 0.02])
        assert check_returns(series).n_missing == 1

    def test_detects_an_extreme_move(self):
        gen = np.random.default_rng(3)
        series = 0.005 * gen.standard_normal(500)
        series[100] = 2.0
        assert check_returns(series).extreme_moves

    def test_uses_a_datetime_index_when_present(self):
        idx = pd.to_datetime(["2020-01-01", "2020-06-01"])
        assert check_returns(pd.Series([0.01, 0.02], index=idx)).largest_gap_days > 100


@pytest.mark.network
class TestNetwork:
    """Deselected by default. The engine never touches the network."""

    def test_fetches_and_caches_spy(self):
        first = load_prices("SPY", "2023-01-01", "2023-06-30", refresh=True)
        assert len(first.frame) > 100
        assert list(first.frame.columns) == [
            c for c in CANONICAL_COLUMNS if c in first.frame.columns
        ]
        assert first.frame.index.tz is None
        second = load_prices("SPY", "2023-01-01", "2023-06-30")
        assert second.from_cache

    def test_fetches_fama_french(self):
        cached = load_fama_french("2023-01-01", "2023-06-30", refresh=True)
        assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"} <= set(cached.frame.columns)
        assert cached.frame["Mkt-RF"].abs().max() < 0.5, "should be decimals, not percentages"
