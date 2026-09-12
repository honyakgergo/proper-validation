# S&P 500 cross-sectional momentum, thirty names

Thirty large, liquid S&P 500 names. Rank them on 12-1 month momentum, hold the top
six equal weight, rebalance monthly, trade from the session after the signal.
2010 to 2024, real prices.

There is nothing wrong with the construction. Every component is standard, the
signal skips the most recent month the way the literature says to, and the
positions are shifted. **The universe is the problem**, and this is the example
that shows what that costs.

```bash
python case_studies/sp500_momentum/run.py            # fetches and caches
python case_studies/sp500_momentum/run.py --offline  # every run after that
```

## The theory

Cross-sectional momentum is among the most replicated effects in the literature.
Jegadeesh and Titman (1993) showed that ranking US stocks on their trailing
six-to-twelve-month return and holding the winners earned a premium that survived
the risk adjustments available at the time; Asness, Moskowitz and Pedersen (2013)
found the same pattern across asset classes and countries. The standard
construction skips the most recent month, because the short-horizon reversal that
dominates the last few weeks runs against the medium-horizon effect being
harvested.

So the prior going in is not that this should fail. It is a real, published,
widely traded effect, implemented here exactly as the literature describes it.
The question the audit answers is what is left of it once the universe, the
factor exposures and the trial count are all accounted for.

## The strategy

[`qv_adapter.py`](qv_adapter.py) — rank on 12-1 month momentum, hold the top six
equal weight, rebalance every 21 sessions, trade from the session after the
signal. Long only, no leverage. [`research_manifest.yaml`](research_manifest.yaml)
declares the grid searched and where the data came from.

## The universe was built the way universes are actually built

From a screener, in 2026, by taking names that are in the S&P 500 today and have
been for the whole sample. That sentence is in the manifest as `universe_note`,
and it is the single most common way a backtest acquires an upward bias it cannot
see.

The other two examples in this directory trade hand-picked ETFs, where index
membership does not apply and survivorship can only be *declared*. This one is
audited against `membership: sp500` — point-in-time constituent history
reconstructed from index change announcements, fetched on first run and cached
outside the repository — so the question is **counted**.

## The report

[`report.html`](report.html) is the page; [`report.json`](report.json) is
authoritative for every number on it.

## What the audit found

**Verdict: falsified.**

| Finding | |
|---|---|
| **`ATTR-LEVERED-BETA`** (critical) | Alpha is **0.08% a year**, t = **0.03**, p = 0.97. Factors explain **72%** of the variance, correlation to SPY is 0.82, and the strategy *subtracts* 0.027 annualised Sharpe against simply holding SPY at 1.04x leverage. |
| **`DATA-SURVIVORSHIP`** (high) | **310 of the 310 names that left the index** during the window are absent from the universe. `exit_miss_rate` = 1.0. |

### The headline numbers, against the thing you could have bought instead

| Metric | Strategy | SPY |
|---|---:|---:|
| Annualised return | 13.63% | 13.70% |
| Annualised volatility | 17.7% | 17.1% |
| **Sharpe** (excess of cash) | **0.75** | **0.77** |
| Sortino | 1.06 | 1.08 |
| Maximum drawdown | −26.9% | −33.7% |

Fifteen years of stock picking, and the answer is a slightly worse Sharpe than the
index at slightly higher volatility. The shallower drawdown is the one genuine
improvement, and it is not what the strategy was sold on.

## What the membership check actually measured

Over 2010-2024 the index had **813** distinct members. This backtest could choose
from **30**.

The two ways of being absent are not the same thing, and the audit separates them:

- **473 names were still in the index at the end and were never traded.** That is
  *incompleteness*, not survivorship. Holding thirty names out of five hundred is
  a choice, and the report records it without raising a finding. This matters:
  before this distinction existed, every subset strategy got the same HIGH finding
  as a genuinely survivor-picked book, which is how a finding becomes wallpaper.
- **310 names left the index during the window, and not one of them is in the
  universe.** That is survivorship, and `exit_miss_rate = 1.0` is the textbook
  signature: every name that failed, merged or was relegated is missing, every
  name that endured is present. A universe assembled today cannot contain them.

The report states, on this run and every other, that this **bounds the extent of
the bias and not its size**. Counting who was excluded says nothing about what
they would have returned. Sizing it needs prices for delisted instruments, which
`yfinance` does not carry — and that limitation is the reason the count exists.

It also states a limit it cannot resolve: **52 tickers hold more than one
membership spell**, and with no permanent identifier in the list a name that
rejoined cannot be told from a symbol another company later took over.

## What came through clean

Costs are not the binding constraint: turnover is 5.4x a year against a
**243 bps** break-even, in a market that realistically costs 2–5. PBO is **0.446**,
below the coin-flip threshold. The engine side is clean — deterministic,
non-degenerate, no behavioural leakage.

None of that rescues it. The edge was never there: 0.08% of alpha a year is not a
discovery, it is a rounding error on a levered index position, and the universe it
was measured on flatters even that.
