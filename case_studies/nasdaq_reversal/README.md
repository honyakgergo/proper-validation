# NASDAQ-100 short-term reversal

The one of the three that **passes almost everything**. Its alpha is real on the
regression, it clears the multiple-testing bar, its selection procedure
generalises, and it beats its benchmark by a wide margin. One finding remains,
and it is the one no amount of better statistics can fix.

```bash
python case_studies/nasdaq_reversal/run.py            # fetches and caches
python case_studies/nasdaq_reversal/run.py --offline  # every run after that
```

## The theory

Over horizons of a few days, stocks that have just fallen tend to bounce and
stocks that have just risen tend to give some back. Lehmann (1990) and
Jegadeesh (1990) measured it independently on US equities, and the usual reading
is that it is not a forecast at all but a fee: someone was forced to sell in
size, the price moved further than the news warranted, and whoever took the
other side is paid for supplying that liquidity.

If that is what the effect is, then two things follow, and the audit checks
both. It should be **strongest in the days immediately after the move**, which
is why the position is traded from the session *after* the signal rather than at
its own close — booking the trade at the selecting price would capture the
reversal before it happened. And it should be **expensive to harvest**, because
a fee paid for liquidity is collected by trading constantly.

## The strategy

[`qv_adapter.py`](qv_adapter.py) — rank the universe on its trailing five-day
return, hold the eight worst performers equal weight, re-rank every fifth
session. Long only, no leverage.

The universe is the **thirty-seven names that have been in the NASDAQ-100
continuously since the constituent record begins in January 2015**, taken from
the index's own change history rather than assembled by hand. `GOOG` is dropped
in favour of `GOOGL` so one company does not hold two slots in a cross-sectional
rank. `EA` is dropped for a duller reason: it is still listed as a member, but
the price vendor no longer carries it.

## The report

[`report.html`](report.html) is the page; [`report.json`](report.json) is
authoritative for every number on it.

## What the audit found

**Verdict: materially weakened.** One finding.

| | Strategy | QQQ |
|---|---:|---:|
| Annualised return | **30.15%** | 18.44% |
| Annualised volatility | 25.5% | 21.8% |
| **Sharpe** (excess of cash) | **1.10** | 0.81 |
| Sortino | 1.60 | 1.14 |
| Maximum drawdown | −35.7% | −35.1% |
| Calmar | **0.84** | 0.53 |

### What survived

- **Alpha is real on the regression.** 12.63% a year, **t = 2.97**, p = 0.003,
  against Fama–French 5 plus momentum with Newey–West errors at 56 lags. The
  factors explain 74.9% of the variance and the alpha survives what is left.
- **Selection bias is not the problem.** Deflated Sharpe **0.993** against 30
  declared trials, p = 0.007.
- **The procedure generalises.** PBO **0.396**, below the coin-flip threshold —
  the in-sample winner usually stays a winner out of sample.
- **The engine is clean.** Deterministic, non-degenerate, no behavioural
  leakage.

### What it costs to run

Turnover is **78x a year**. Break-even cost is **35.9 bps** against the 2–5 bps
that trading US large-caps realistically costs, so the margin is about **7x** and
the audit does not raise a finding.

That number deserves reading slowly rather than being filed as a pass. Thirty-six
basis points is the *entire* budget for a strategy that turns its book over
seventy-eight times a year in eight-name concentrated blocks. The 2–5 bps figure
is a quoted spread on liquid mega-caps; it is not slippage on a concentrated
weekly rebalance, and it is not market impact. The audit says costs do not kill
this at the level it can measure, which is a narrower statement than "costs are
fine".

## The finding that remains

**`DATA-SURVIVORSHIP`** (high). Measured against the index's own constituent
history rather than declared:

- Over 2015–2024 the NASDAQ-100 had **199** distinct members. This backtest could
  choose from **37**.
- **98 names left the index during the window, and not one is in the universe.**
  `exit_miss_rate = 1.0`.
- 64 names were still members at the end and simply were not traded. That is
  *incompleteness* — a choice — and it raises nothing.

This is the whole point of the example. Everything the statistics could check,
the strategy passed. The one thing that survives is the thing the statistics
**cannot** see, because a universe of survivors looks exactly like a universe
from the inside. A reversal strategy that only ever picks among names which went
on to stay in the NASDAQ-100 for a decade is buying dips in companies that were,
in hindsight, always going to recover.

The report is explicit that this **bounds the extent of the bias and not its
size**: counting who was excluded says nothing about what they would have
returned. Sizing it needs prices for delisted names, which the vendor does not
carry. So the honest reading of a 1.10 Sharpe here is **upper bound**, not
estimate — and the 12.63% alpha with it.

It also states a limit it cannot resolve: **11 tickers hold more than one
membership spell**, and with no permanent identifier a name that rejoined the
index cannot be told from a symbol another company later took over.

## Why this one is in the set

The other two examples fail on the numbers. This one does not, and is weakened
anyway. That is the case worth having in front of a reader: a validator that only
ever fires on strategies which were obviously bad would be telling you nothing
you could not see yourself.
