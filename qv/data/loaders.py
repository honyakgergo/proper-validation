"""The single network boundary in this package.

**The engine never touches the network.** Every statistical function elsewhere
takes DataFrames or Series in and returns results out. All fetching lives here,
which is what makes bring-your-own-data work for free, lets the whole test
suite run offline, and keeps a clone small.

Downloads land in a platform cache directory as Parquet and are never
committed. Ken French factor files are periodically revised, so the vintage
date of a cached copy is recorded and surfaced in the report footer - a number
reproduced from a different vintage is not the same number.

Pass ``offline=True`` to refuse the network entirely and fail loudly if the
cache cannot serve the request, rather than silently refetching and quietly
changing the numbers under a published report.
"""

from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from platformdirs import user_cache_dir

__all__ = [
    "CANONICAL_COLUMNS",
    "cache_dir",
    "CachedFrame",
    "load_prices",
    "load_universe",
    "read_price_frame",
    "load_fama_french",
    "to_returns",
]

#: The documented wide schema. Anyone with a CSV in this shape gets the full
#: engine without going near this module.
CANONICAL_COLUMNS = ("open", "high", "low", "close", "volume")

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


#: What to do about a cold cache, appended to both offline failures.
#:
#: This is the wall a newcomer hits first. Every regeneration command in
#: `CLAUDE.md`, in both `run.py` files and in `docs/make_images.py` passes
#: `--offline`, which is exactly the flag that refuses to fetch - so on a
#: fresh clone the documented command is the one guaranteed to fail, and the
#: fix is a single run without the flag. Saying so here costs one sentence;
#: leaving it out costs somebody an afternoon.
_COLD_CACHE_REMEDY = (
    "Run the same command once without --offline to fetch and cache it; every "
    "run after that can refuse the network."
)

#: Seconds to wait before each retry of a price download.
#:
#: Yahoo throttles a burst of requests and reports the refusal as an empty
#: frame, which yfinance surfaces as "possibly delisted" - indistinguishable
#: from a ticker that really has gone. Populating a cold cache is exactly a
#: burst: the worked examples walk universes of nine to forty-six names in a
#: loop, and measured here, an unthrottled walk died on the eighth ticker. So
#: the documented first run - the one a fresh clone has no choice but to make -
#: was the one that failed. Backing off and retrying carried the same walk to
#: completion.
#:
#: Bounded deliberately. Five waits totalling about four minutes is long enough
#: to outlast throttling and short enough that a genuinely bad ticker still
#: fails rather than hanging a run indefinitely.
_FETCH_BACKOFF_SECONDS = (5, 15, 30, 60, 120)


def cache_dir() -> Path:
    """Where downloads live. Gitignored by construction - it is outside the repo."""
    path = Path(user_cache_dir("qv", "proper_validation")) / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class CachedFrame:
    """Data plus where it came from and when, for the reproducibility footer."""

    frame: pd.DataFrame
    source: str
    vintage: str
    from_cache: bool
    path: Path | None = None
    #: For a universe, the wide frame *before* the common-calendar alignment.
    #: Coverage has to be measured on the ragged frame: dropping the incomplete
    #: rows is exactly what makes a patchy universe look clean, so a check run
    #: after alignment can only ever report that everything is fine.
    unaligned: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "vintage": self.vintage,
            "from_cache": self.from_cache,
            "rows": int(len(self.frame)),
            "start": str(self.frame.index.min()) if len(self.frame) else None,
            "end": str(self.frame.index.max()) if len(self.frame) else None,
        }


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # pragma: no cover - corrupt cache should not be fatal
        return None


def _vintage_of(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.date().isoformat()


def load_prices(
    ticker: str,
    start: str | date = "2005-01-01",
    end: str | date | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> CachedFrame:
    """Daily OHLCV for one ticker, in the canonical wide schema.

    ``end`` should be pinned by the caller. An example that leaves it open
    reproduces whatever the vendor returns today rather than the numbers that
    were published, which makes a "reproducible" report reproduce nothing.
    """
    if end is None:
        raise ValueError(
            "end must be pinned so results stay reproducible; pass an explicit date"
        )

    key = f"prices_{ticker.upper()}_{start}_{end}.parquet"
    path = cache_dir() / key

    cached = None if refresh else _read_cache(path)
    if cached is not None:
        return CachedFrame(cached, f"cache:{ticker}", _vintage_of(path), True, path)
    if offline:
        raise FileNotFoundError(
            f"offline mode and no cached copy of {ticker} for {start}..{end} at "
            f"{path}. {_COLD_CACHE_REMEDY}"
        )

    try:
        import yfinance
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "fetching prices needs the optional data extra: pip install 'proper-validation[data]'"
        ) from exc

    raw = None
    for attempt, wait in enumerate((0,) + _FETCH_BACKOFF_SECONDS):
        if wait:
            print(
                f"no data for {ticker} yet; likely throttling, retrying in {wait}s "
                f"({attempt}/{len(_FETCH_BACKOFF_SECONDS)})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
        raw = yfinance.download(
            ticker, start=str(start), end=str(end), progress=False, auto_adjust=True
        )
        if raw is not None and not raw.empty:
            break
    if raw is None or raw.empty:
        raise ValueError(
            f"no data returned for {ticker} between {start} and {end}, after "
            f"{len(_FETCH_BACKOFF_SECONDS)} retries. Either the ticker is wrong "
            f"or delisted, or the vendor is still refusing the request - it "
            f"reports both the same way. Any cached tickers were kept, so "
            f"re-running resumes rather than starting over."
        )

    # yfinance returns a MultiIndex column frame for a single ticker in recent
    # versions; flatten to the documented schema either way.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    frame = raw.rename(columns=str.lower)
    frame = frame[[c for c in CANONICAL_COLUMNS if c in frame.columns]].copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"

    frame.to_parquet(path)
    return CachedFrame(frame, f"yfinance:{ticker}", date.today().isoformat(), False, path)


def _download(url: str) -> bytes:  # pragma: no cover - network
    request = Request(url, headers={"User-Agent": "proper-validation/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _parse_french_zip(payload: bytes) -> pd.DataFrame:
    """Extract the daily table from a Ken French CSV zip.

    Their files carry a prose header and a trailing annual section, so the data
    block is located by finding the run of lines that begin with an 8-digit
    date rather than by a fixed skiprows count that breaks whenever they
    reformat.
    """
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("latin-1")

    lines = text.splitlines()
    rows, header = [], None
    for line in lines:
        if header is None and re.match(r"^\s*,", line):
            header = [c.strip() for c in line.split(",")]
            continue
        if re.match(r"^\s*\d{8}\s*,", line):
            rows.append([c.strip() for c in line.split(",")])
        elif rows:
            break  # the daily block has ended; annual data follows

    if not rows or header is None:
        raise ValueError("could not locate the daily data block in the Ken French file")

    frame = pd.DataFrame(rows, columns=header[: len(rows[0])])
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").astype(float)
    # Ken French publishes percentages; the rest of this package uses decimals.
    return frame / 100.0


def load_fama_french(
    start: str | date = "2005-01-01",
    end: str | date | None = None,
    offline: bool = False,
    refresh: bool = False,
) -> CachedFrame:
    """Fama-French 5 factors plus momentum, daily, as decimal returns.

    Columns: ``Mkt-RF``, ``SMB``, ``HML``, ``RMW``, ``CMA``, ``Mom``, ``RF``.

    The vintage matters and is recorded. Dartmouth revises these files, so a
    report footer that does not name the vintage cannot be reproduced with
    confidence.
    """
    path = cache_dir() / "fama_french_daily.parquet"
    cached = None if refresh else _read_cache(path)

    if cached is None:
        if offline:
            raise FileNotFoundError(
                f"offline mode and no cached factor file at {path}. "
                f"{_COLD_CACHE_REMEDY}"
            )
        five = _parse_french_zip(_download(_FF5_URL))
        momentum = _parse_french_zip(_download(_MOM_URL))
        momentum.columns = ["Mom" if c.lower().startswith("mom") else c for c in momentum.columns]
        cached = five.join(momentum, how="inner")
        cached.to_parquet(path)
        vintage, from_cache = date.today().isoformat(), False
    else:
        vintage, from_cache = _vintage_of(path), True

    window = cached.loc[str(start) : str(end)] if end else cached.loc[str(start) :]
    return CachedFrame(window, "Ken French data library", vintage, from_cache, path)


def load_universe(
    tickers: Sequence[str],
    start: str | date = "2005-01-01",
    end: str | date | None = None,
    column: str = "close",
    offline: bool = False,
    refresh: bool = False,
) -> CachedFrame:
    """One wide frame of ``column`` for every ticker, on their common calendar.

    Rows where any instrument is missing are dropped, so every column covers
    exactly the same sessions. That matters more than it sounds: a
    cross-sectional strategy ranking nine sectors against each other must rank
    them on the same days, and a ragged frame would quietly let an instrument
    that listed late look like it was merely flat beforehand.

    The reported vintage is the *oldest* across the tickers, because a
    universe is only as fresh as its stalest member.
    """
    names = list(dict.fromkeys(tickers))
    if not names:
        raise ValueError("a universe needs at least one ticker")

    frames: dict[str, pd.Series] = {}
    vintages: list[str] = []
    from_cache = True
    for ticker in names:
        cached = load_prices(ticker, start, end, offline=offline, refresh=refresh)
        if column not in cached.frame.columns:
            raise ValueError(
                f"{ticker} has no column {column!r}; found "
                f"{list(cached.frame.columns)}"
            )
        frames[ticker] = cached.frame[column]
        vintages.append(cached.vintage)
        from_cache = from_cache and cached.from_cache

    ragged = pd.DataFrame(frames)[names]
    wide = ragged.dropna()
    if wide.empty:
        raise ValueError(
            f"no sessions are common to all {len(names)} instruments over "
            f"{start}..{end}; check the tickers and the date range"
        )
    return CachedFrame(
        wide, "yfinance", min(vintages), from_cache, None, unaligned=ragged
    )


def read_price_frame(path: str | Path, column: str | None = None) -> pd.DataFrame:
    """Read a wide price frame from a local CSV or Parquet.

    The documented schema: a tz-naive ``DatetimeIndex`` in the first column,
    then one column per instrument. A long-format file with a ``ticker``
    column is pivoted. No network, no vendor - anyone whose data is better
    than yfinance uses this and gets the whole engine.

    ``column`` names the price field to keep when the file carries several per
    instrument, i.e. when it is long format with ``open``/``close`` columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such price file: {path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)

    if "ticker" in frame.columns:
        value = column or "close"
        if value not in frame.columns:
            raise ValueError(
                f"{path.name} is long format but has no {value!r} column; found "
                f"{list(frame.columns)}"
            )
        frame = frame.pivot(columns="ticker", values=value)

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)

    frame = frame.sort_index()
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] == 0:
        raise ValueError(f"{path.name} has no numeric price columns")
    return numeric


def to_returns(prices: pd.DataFrame | pd.Series, column: str = "close") -> pd.Series:
    """Simple returns from a price frame or series, with the first row dropped."""
    series = prices if isinstance(prices, pd.Series) else prices[column]
    return series.pct_change().dropna()
