# research_manifest.yaml

What the researcher asserts, recorded so the audit can be reproduced and so the assumptions are
visible rather than implied. Every example in this repository ships one, and it is validated
against a schema (`qv/manifest.py`) rather than read field by field — because a field the tool
silently ignores is worse than a missing one.

With an adapter beside it, this file is a whole audit:

```bash
qv adapter init                                   # writes both files to fill in
qv validate --manifest research_manifest.yaml --out report/
```

```yaml
name: Human-readable strategy name

# Which questions to ask by default: statistical, engine, or full. Overridable
# per run with --suite, so one manifest produces all three reports without
# being edited between them.
suite: full

data:
  # The instruments to load, in the order the adapter expects its columns.
  universe: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]
  benchmark: SPY
  start_date: "2005-01-01"
  end_date: "2024-12-31"    # pinned, deliberately - see below
  source: yfinance
  asset_class: us_large_cap_etf
  frequency: daily          # daily | weekly | monthly | quarterly | annual
  universe_point_in_time: true   # or false; see below
  universe_note: how the list was built and when
  # Optional. Counts survivorship instead of declaring it; see below.
  membership_frame: membership.csv
  membership_index: SP500        # only when the file carries several
  # Optional: a local CSV or Parquet in the wide schema, relative to this file.
  # Given, nothing is downloaded.
  price_frame: prices.csv

strategy:
  adapter: qv_adapter.py:positions   # path:function, relative to this file
  fixed: {}                          # parameters that were never searched

chosen_parameters:        # the configuration actually reported
  lookback: 252
  skip: 21
  top_n: 3

search:
  # Every configuration examined, keyed by the parameter each axis varies.
  axes:
    lookback: [126, 189, 252, 315]
    skip: [0, 21]
    top_n: [2, 3, 4]
  n_trials: 24            # see below

process:                  # free-form; documentation, not validated
  positions_shifted: true
  lookahead_possible: false
  notes: anything a reviewer would want to know
```

Anything else at the top level (`expectations`, `provenance`, `ground_truth`) is carried through
untouched. The tool has no business telling a researcher how to write their notes.

## Fields that matter most

**`strategy.adapter` is what the engine analysis runs.** It names a function the tool imports and
re-runs — once for the reported configuration, once per grid point, and repeatedly on deliberately
corrupted data for the behavioural leakage test, which is the only test that can *settle* whether
the strategy reads the future. See `adapter_protocol.md`. There is no sandbox: the file is imported
and executed in the same interpreter, so read an adapter before pointing the tool at one.

**`search.axes` is keyed by the parameter, not the plural of it.** `lookback: [126, 252]`, not
`lookbacks:`. The grid maps straight onto the adapter's keyword arguments, so nothing has to guess
that `lookbacks` meant `lookback`. `chosen_parameters` must name every axis, or the audit cannot
say which point in the grid it is reporting.

**`search.n_trials` is the number the whole audit turns on.** Count every configuration examined,
including the ones abandoned before they were written down. The grid above is a *lower bound* — it
cannot contain what was discarded — so the audit uses whichever is larger, the grid or this
declaration. `qv trials` recovers a floor from a notebook; this field is where the
interview-revised number goes. When unsure, overstate it: the multiple-testing correction is
logarithmic in the trial count, so honesty is cheap.

**`end_date` must be pinned.** An open end date reproduces whatever the vendor returns today rather
than the numbers that were published, which makes a "reproducible" report reproduce nothing. The
schema rejects `today` and an empty value outright.

**`asset_class`** selects the realistic cost range that break-even cost is compared against. Leave
it out and the tool declines to render a verdict on costs rather than inventing an estimate for a
market it was not told about.

**`frequency`** sets the annualisation. Guessing it from the index is how a monthly strategy ends up
with a Sharpe inflated by `sqrt(21)`.

**`universe_point_in_time` is the one question nothing can answer from the return series.**
Survivorship leaves no trace there — a universe of instruments that still exist today simply looks
like a universe. So the audit asks. Declare `false` and it raises `DATA-SURVIVORSHIP` and treats
the result as an upper bound; declare `true` and the report says so in its footer; leave it out and
the question is listed under *what could not be tested*, which is where an unanswered question
belongs.

**`membership_frame` turns that declaration into a count.** Point it at a point-in-time membership
list and the audit measures the hole instead of taking the researcher's word for it: how many names
were index members during the window, how many of those are absent from the traded universe, and
how many of the absent ones *left the index* while the backtest was running. Declaring
`universe_point_in_time: true` while the list disagrees raises `DATA-DECLARATION-CONTRADICTED`,
because a manifest with one disproved declaration is different evidence about all the others —
`search.n_trials` above all.

```csv
ticker,start_date,end_date
AAPL,1996-01-02,
AAL,1996-01-02,1997-01-15
AAL,2015-03-23,2024-09-23
AAMRQ,1996-01-02,2003-03-14
```

`ticker` and `start_date` are required; a blank `end_date` means still a member; `id` (a permanent
identifier) and `index` are optional. **One row per membership spell** — a name that left and
rejoined has several, and the gap between them is exactly the period a backtest must not trade it.

This is the schema of the most widely used free reconstruction, so the common case needs no
conversion. Free coverage reaches **1996 for the S&P 500** and **2015 for the NASDAQ-100**; earlier
than that needs CRSP or Compustat. A list covering only part of the sample is used for the part it
covers, with the fraction stated in the report and the uncovered period named under *what could not
be tested* — never silently counted as "nothing missing".

Three things this cannot do, all stated on every run that uses it. It measures **extent, not
magnitude**: free lists carry no prices for delisted names, so the audit counts what was excluded
and never what it would have returned. It cannot tell a re-listing from a reused ticker without an
`id` column. And it is refused outright if it records no removals at all — a table of *today's*
members with the date each was added looks like history but contains only survivors, so it would
report zero names missing no matter how many were.

**Factors are loaded by default,** which is what puts every Sharpe on a risk-free basis. The Sharpe
ratio is defined on returns in excess of cash, and computing it on total returns flatters the least
volatile series in any comparison — by 0.06 of a Sharpe point for a 12%-volatility strategy over
2005-2024. Set `data.load_factors: false` only when Ken French's file genuinely does not apply, and
expect the report to say on its face that its Sharpes are total-return figures.

## Data provenance

The repository ships no market data. Downloads land in a platform cache directory as Parquet and
are never committed:

- A 500-ticker daily universe over 20 years is 50-100 MB as Parquet and 200-400 MB as CSV, past the
  GitHub warning threshold and near its hard per-file limit.
- Ken French factor files are periodically revised, so the **vintage date** of a cached copy is
  recorded and printed in the report footer. A number reproduced from a different vintage is not
  the same number.

Each example ships this manifest and a fetch script, never the data. Run with `--offline` to refuse
the network and fail loudly if the cache cannot serve the request, rather than silently refetching
and quietly changing the numbers under a published report.

## Bring your own data

`qv` reads any CSV or Parquet in the documented wide schema, so no vendor is privileged:

- A tz-naive `DatetimeIndex` in the first column
- Then one column per instrument, named as the universe names them

Point `data.price_frame` at it and the network layer is never reached. A long-format file with a
`ticker` column is pivoted on `data.price_column`. For a single instrument the full OHLCV schema —
`open`, `high`, `low`, `close`, `volume` — is also accepted, and `qv/data/quality.py` validates
against it and surfaces gaps and split artifacts.

## What a manifest cannot express

A search that is not a parameter grid. `examples/01_mined_noise` draws ten thousand random rules
rather than crossing a few axes, so it keeps a driver script of its own and stands as the
documented escape hatch. A manifest general enough to express any search would just be a
programming language.
