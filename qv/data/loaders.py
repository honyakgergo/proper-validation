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
    "read_membership_frame",
    "load_index_membership",
    "MEMBERSHIP_SOURCES",
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


def read_membership_frame(
    path: str | Path, index_name: str | None = None
) -> "Membership":
    """Read a point-in-time index membership list from a local CSV or Parquet.

    Documented schema, one row per membership *spell*: ``ticker`` and
    ``start_date`` required, ``end_date`` blank when the name is still a member,
    optional ``id`` (a permanent identifier that survives a ticker change) and
    ``index`` (so one file can carry several universes).

    This is exactly the shape of the most widely used free file,
    `sp500_ticker_start_end.csv` from fja05680/sp500, so the common case needs
    no conversion. No network and no vendor: anyone with better data than a
    reconstruction from change announcements points this at it.

    Refuses rather than guesses. A membership list that cannot be trusted
    produces a confidently wrong number, and a wrong count of missing names is
    worse than no count - it is the false clean this whole feature exists to
    prevent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such membership file: {path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    return _build_membership(frame, f"local file {path.name}", index_name, path.name)


def _build_membership(
    frame: pd.DataFrame,
    source: str,
    index_name: str | None = None,
    label: str = "membership list",
    vintage: str | None = None,
) -> "Membership":
    """Validate a spell table and refuse it if it cannot support a claim.

    Shared by the local-file reader and the named sources, so a fetched list is
    held to exactly the same standard as one the researcher supplies. A
    membership list that cannot be trusted produces a confidently wrong count,
    and a wrong count of missing names is worse than no count - it is the false
    clean this whole feature exists to prevent.
    """
    from qv.data.membership import MEMBERSHIP_COLUMNS, REQUIRED_COLUMNS, Membership

    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required column(s) {missing}; the schema is "
            f"{','.join(MEMBERSHIP_COLUMNS)} (only the first two are required)"
        )
    if "end_date" not in frame.columns:
        frame["end_date"] = pd.NaT

    for field in ("start_date", "end_date"):
        parsed = pd.to_datetime(frame[field], errors="coerce")
        if getattr(parsed.dtype, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        frame[field] = parsed
    if frame["start_date"].isna().any():
        bad = int(frame["start_date"].isna().sum())
        raise ValueError(
            f"{label} has {bad} row(s) with an unparsable start_date; dates must "
            "be ISO (YYYY-MM-DD)"
        )

    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    if frame.empty:
        raise ValueError(f"{label} contains no membership rows")

    if "index" in frame.columns:
        names = sorted(str(v) for v in frame["index"].dropna().unique())
        if index_name is not None:
            frame = frame[frame["index"].astype(str) == index_name]
            if frame.empty:
                raise ValueError(
                    f"{label} has no rows for index {index_name!r}; it carries "
                    f"{names}"
                )
        elif len(names) > 1:
            raise ValueError(
                f"{label} carries more than one index ({names}); name which one "
                "with data.membership_index rather than letting the tool pick"
            )

    ends = frame["end_date"]
    backwards = ends.notna() & (ends <= frame["start_date"])
    if backwards.any():
        raise ValueError(
            f"{label} has {int(backwards.sum())} spell(s) ending on or before "
            "they start"
        )

    # Overlapping spells for one ticker are malformed: a name cannot be two
    # separate members of the same index at once. Non-overlapping repeats are
    # fine and expected - names leave and rejoin.
    ordered = frame.sort_values(["ticker", "start_date"])
    previous_end = ordered.groupby("ticker")["end_date"].shift(1)
    previous_ticker = ordered["ticker"].shift(1)
    overlapping = (
        (ordered["ticker"] == previous_ticker)
        & previous_end.notna()
        & (ordered["start_date"] < previous_end)
    )
    if overlapping.any():
        names = sorted(set(ordered.loc[overlapping, "ticker"]))[:6]
        raise ValueError(
            f"{label} has overlapping membership spells for {names}; a ticker "
            "cannot be two members of one index at the same time"
        )

    return Membership(
        frame=frame.reset_index(drop=True),
        source=source,
        index_name=index_name,
        vintage=vintage,
    )


#: Point-in-time membership lists the tool knows how to fetch, by name.
#:
#: Pinned URLs rather than committed files, and the reason is not convenience.
#: An index constituent list is somebody's compilation: in the US the facts in
#: it are free, but most indices select their members by committee judgement,
#: which is exactly what makes a compilation copyrightable - and the EU protects
#: databases outright, with no creativity test at all. Shipping code and a link
#: redistributes nothing, which is how this package already treats Yahoo's
#: prices and Ken French's factors. Nobody experiences those as friction.
#:
#: It also fixes staleness, which committing cannot. A frozen list silently
#: under-reports departures as it ages, and that is the false-clean direction.
#: A fetched one carries a vintage and the report prints it.
MEMBERSHIP_SOURCES: dict[str, dict[str, Any]] = {
    "sp500": {
        "label": "S&P 500",
        "url": (
            "https://raw.githubusercontent.com/fja05680/sp500/master/"
            "sp500_ticker_start_end.csv"
        ),
        "attribution": "fja05680/sp500 (MIT), reconstructed from index change announcements",
        "coverage": "1996 onward",
    },
    "nasdaq100": {
        "label": "NASDAQ-100",
        "url": (
            "https://raw.githubusercontent.com/jmccarrell/n100tickers/main/src/"
            "nasdaq_100_ticker_history/n100-ticker-changes-{year}.yaml"
        ),
        "attribution": "jmccarrell/n100tickers (MIT), from Nasdaq change announcements",
        "coverage": "2015 onward",
        "years": (2015, 2027),
    },
}


def _nasdaq100_spells(years: tuple[int, int]) -> pd.DataFrame:
    """Rebuild NASDAQ-100 membership spells from per-year change files.

    Each yearly file carries the full membership on 1 January *and* the dated
    additions and removals that follow. So the reconstruction has a built-in
    check: walk the changes forward from the first year and the running set must
    equal the next year's declared list at every boundary. If it ever does not,
    the walk is wrong and this refuses rather than returning a plausible table -
    a membership list that is quietly wrong is the worst thing this feature
    could produce.
    """
    import yaml

    first, stop = years
    docs: dict[int, dict] = {}
    for year in range(first, stop):
        try:
            payload = _download(MEMBERSHIP_SOURCES["nasdaq100"]["url"].format(year=year))
        except Exception:
            break  # future years simply do not exist yet
        docs[year] = yaml.safe_load(payload)
    if not docs:
        raise ValueError("no NASDAQ-100 change files could be fetched")

    def as_date(value: Any) -> pd.Timestamp:
        return pd.Timestamp(str(value))

    opened: dict[str, pd.Timestamp] = {}
    spells: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = []
    members: set[str] = set()

    for year in sorted(docs):
        declared = {str(t).strip().upper() for t in docs[year]["tickers_on_Jan_1"]}
        if not members:
            members = set(declared)
            opened = {t: pd.Timestamp(year=year, month=1, day=1) for t in members}
        elif declared != members:
            only_declared = sorted(declared - members)[:6]
            only_walked = sorted(members - declared)[:6]
            raise ValueError(
                f"NASDAQ-100 reconstruction disagrees with the declared membership on "
                f"{year}-01-01: the file lists {only_declared} which the walk does not, "
                f"and the walk holds {only_walked} which the file does not. The change "
                "history and the yearly snapshots are inconsistent, so no spell table "
                "is produced."
            )

        for when, change in sorted((docs[year].get("changes") or {}).items()):
            effective = as_date(when)
            for ticker in change.get("difference") or []:
                ticker = str(ticker).strip().upper()
                if ticker in members:
                    spells.append((ticker, opened.pop(ticker), effective))
                    members.discard(ticker)
            for ticker in change.get("union") or []:
                ticker = str(ticker).strip().upper()
                if ticker not in members:
                    members.add(ticker)
                    opened[ticker] = effective

    for ticker in sorted(members):
        spells.append((ticker, opened[ticker], None))

    return pd.DataFrame(spells, columns=["ticker", "start_date", "end_date"])


def load_index_membership(
    name: str, offline: bool = False, refresh: bool = False
) -> "Membership":
    """Fetch a named point-in-time membership list, cached like prices.

    ``qv`` ships the link and the parser, never the list. See
    ``MEMBERSHIP_SOURCES`` for why.
    """
    key = str(name).strip().lower()
    if key not in MEMBERSHIP_SOURCES:
        raise ValueError(
            f"unknown membership source {name!r}; known sources are "
            f"{sorted(MEMBERSHIP_SOURCES)}. For anything else, point "
            "`data.membership_frame` at a local file."
        )
    spec = MEMBERSHIP_SOURCES[key]
    path = cache_dir() / f"membership_{key}.parquet"

    cached = None if refresh else _read_cache(path)
    if cached is not None:
        return _build_membership(
            cached, f"{spec['label']} ({spec['attribution']})", None,
            f"{key} membership", _vintage_of(path),
        )
    if offline:
        raise FileNotFoundError(
            f"offline mode and no cached {spec['label']} membership list at {path}. "
            f"{_COLD_CACHE_REMEDY}"
        )

    if key == "nasdaq100":
        frame = _nasdaq100_spells(spec["years"])
    else:
        frame = pd.read_csv(io.BytesIO(_download(spec["url"])))

    membership = _build_membership(
        frame, f"{spec['label']} ({spec['attribution']})", None,
        f"{key} membership", date.today().isoformat(),
    )
    membership.frame.to_parquet(path)
    return membership


def to_returns(prices: pd.DataFrame | pd.Series, column: str = "close") -> pd.Series:
    """Simple returns from a price frame or series, with the first row dropped."""
    series = prices if isinstance(prices, pd.Series) else prices[column]
    return series.pct_change().dropna()
