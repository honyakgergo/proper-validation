# Real user tests

Three genuine strategies, on real data, with real benchmarks and real
Fama–French factors, and every configuration examined declared honestly.

Unlike `benchmarks/`, nothing here has a known ground truth. That is the point:
these are the runs where the answer is not decided in advance.

| Strategy | Universe | vs benchmark | Verdict | What it shows |
|---|---|---|---|---|
| [`dual_momentum/`](dual_momentum/) | 9 sector ETFs + bonds and gold | Sharpe **0.540** vs SPY 0.530, drawdown **−27%** vs −55% | Materially weakened | Impeccable code, weak statistics — and why one blended verdict hides that |
| [`sp500_momentum/`](sp500_momentum/) | 30 S&P 500 single names | Sharpe 0.75 vs SPY 0.77 | **Falsified** | There was never an edge, and the universe flattered even that |
| [`nasdaq_reversal/`](nasdaq_reversal/) | 37 NASDAQ-100 single names | Sharpe **1.10** vs QQQ 0.81 | Materially weakened | It passes almost everything, and is weakened by the one thing statistics cannot see |

Each folder holds the same four things: a **README** with the theory and what the
audit found, the **strategy** as plain Python, the **manifest** declaring what
was searched and where the data came from, and the **report** — `report.html` to
read, `report.json` authoritative for every number on the page.

## Why these three

They fail differently, and a validator that returned the same verdict for all
three would be telling you nothing.

**`dual_momentum` is the careful construction.** Sector rotation with an absolute
momentum filter, a defensive bond and gold leg, and volatility targeting — every
component from published work rather than found by searching the data. It beats
the S&P 500 on Sharpe and halves the worst drawdown. Engine analysis returns
**survived**; statistical validation returns **materially weakened**, on an alpha
of 0.47% a year at t = 0.30 and a PBO of 59%. Impeccably implemented and
statistically weak — exactly the split the two suites exist to separate, which is
why it ships all three reports.

**`sp500_momentum` is the ordinary mistake.** Standard 12-1 cross-sectional
momentum on thirty large caps, with the universe assembled from names in the
index *today*. Fifteen years of stock picking returns a slightly worse Sharpe
than the index, an alpha of 0.08% a year at t = 0.03, and a measured survivorship
count: over 2010–2024 the index had **813** members, this backtest could choose
from **30**, and **310 of the 310 names that left the index** are absent from it.

**`nasdaq_reversal` is the uncomfortable one.** Short-term reversal on the
NASDAQ-100's long-standing members, and it *passes* — alpha 12.6% a year at
t = 2.97, deflated Sharpe 0.993, PBO 0.396, a clean engine, Sharpe 1.10 against
QQQ's 0.81. It is weakened on survivorship alone: **98 of the 98 names that left
the index** are missing. Everything the statistics could check, it passed; the
finding that survives is the one they cannot reach.

## Two suites, and why dual momentum ships three reports

An audit asks two independent questions. **Statistical validation** asks whether
the measured edge is distinguishable from luck. **Engine analysis** asks whether
the backtest can be trusted as an implementation, whatever its numbers say.

| Report | Verdict | Findings |
|---|---|---|
| [`report_statistical/`](dual_momentum/report_statistical/) | **Materially weakened** | alpha t = 0.30, PBO 59.1% |
| [`report_engine/`](dual_momentum/report_engine/) | **Survived the tests applied** | none |
| [`report.html`](dual_momentum/) (full) | Materially weakened | the two statistical ones |

A single combined verdict blurs that into "materially weakened" and leaves the
researcher with no idea what to fix — when the answer is that there is nothing to
fix in the code, and the edge is not there.

## Survivorship: declared in one, counted in two

`dual_momentum` trades hand-picked ETFs, where index membership does not apply
and survivorship can only be *declared*. The two single-stock strategies are
audited against the indices' own constituent history — `membership: sp500` and
`membership: nasdaq100` — so the question is **counted** instead. Those lists are
fetched and cached like prices and factors; this repository ships the link and
the parser, never the list.

In both, the audit separates two things that are not the same: names that were
still members and simply were not traded (**incompleteness** — a choice, and no
finding), and names that **left the index** and could never have been held
(**survivorship**). It states on every such run that this bounds the *extent* of
the bias and not its size — counting who was excluded says nothing about what
they would have returned, and sizing that needs prices for delisted names, which
no free source carries.

## Conventions

Every Sharpe here and in every report is **in excess of the risk-free rate**
(Ken French's RF). Returns, drawdown and Calmar are total-return figures, because
those describe the path an investor lived through. An earlier version of these
pages reported total-return Sharpes throughout, which inflated every strategy and
inflated the *least volatile* one most — see the note in
[`dual_momentum/`](dual_momentum/), where it accounted for 84% of the claimed
edge over SPY.

All three are driven by `research_manifest.yaml`, which names the function the
backtest was built from, so the validator can re-run the strategy on data it
controls rather than take the code on trust.

```bash
qv validate --manifest case_studies/nasdaq_reversal/research_manifest.yaml
```

`run.py` is the same audit plus a performance preamble, and takes `--offline` to
refuse the network and `--suite` to choose which questions are asked:

```bash
python case_studies/dual_momentum/run.py            # fetches and caches
python case_studies/dual_momentum/run.py --offline  # every run after that
python case_studies/dual_momentum/run.py --offline --suite engine
```

The data loading and grid running are shared, in `qv.pipeline`, which is why the
three reports are comparable at all: they are produced by the same code, not by
three scripts that happened to agree.

**No market data is committed.** The first run of each fetches and caches it
outside the repository; every run after that can refuse the network.
