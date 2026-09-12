# Real user tests

The tool used the way it is meant to be used: a genuine strategy, on real data,
with a real benchmark and real Fama-French factors, and every configuration
examined declared honestly.

Unlike `benchmarks/`, nothing here has a known ground truth. That is the point.
These are the runs where the answer is not decided in advance.

| Test | Strategy | vs SPY | Verdict |
|---|---|---|---|
| [`sector_momentum/`](sector_momentum/) | Top 3 of 9 SPDR sectors by 12-1 momentum, monthly | Sharpe 0.510 vs 0.530 | **Falsified** |
| [`dual_momentum/`](dual_momentum/) | The above plus an absolute filter, a bond/gold leg and volatility targeting | Sharpe **0.540** vs 0.530, drawdown **-27%** vs -55% | **Materially weakened** |
| [`sp500_momentum/`](sp500_momentum/) | Thirty S&P 500 single names, 12-1 momentum, top 6 monthly, 2010-2024 | Sharpe 0.75 vs 0.77 | **Falsified** |

The first pair is the point. The plain rotation loses to the index and is falsified on
attribution. The more careful construction beats it - narrowly on Sharpe, hugely
on drawdown - and is still weakened, for entirely different reasons. A validator
that returned the same verdict for both would be telling you nothing.

**`sp500_momentum/` is here for a different reason.** The two ETF rotations above
trade hand-picked instruments, where index membership does not apply and
survivorship can only be *declared*. The thirty-name single-stock strategy is
audited against point-in-time constituent history, so the question is counted
instead: over 2010-2024 the index had **813** members, this backtest could choose
from **30**, and **310 of the 310 names that left the index** are absent from it.
The 473 names that were still members and simply were not traded raise nothing —
that is incompleteness, which is a choice, not survivorship.

Every Sharpe on this page and in both reports is **in excess of the risk-free
rate** (1.56% a year over 2005-2024, Ken French's RF). Returns, drawdown and
Calmar are total-return figures, because those describe the path an investor
lived through. An earlier version of these pages reported total-return Sharpes
throughout, which inflated both strategies and inflated the *low-volatility* one
most - see the note in [`dual_momentum/`](dual_momentum/), where it accounted
for 84% of the claimed edge over SPY.

Both are driven entirely by `research_manifest.yaml`, which names the
`*_positions` function the backtest was built from — so the validator can
re-run the strategy on data it controls rather than take the code on trust.

```bash
qv validate --manifest real_user_tests/sector_momentum/research_manifest.yaml
```

`run.py` is the same audit plus a performance preamble, and takes `--offline`
to refuse the network and `--suite` to choose the questions:

```bash
python real_user_tests/sector_momentum/run.py            # fetches and caches
python real_user_tests/sector_momentum/run.py --offline  # refuses the network
python real_user_tests/dual_momentum/run.py --suite engine
```

Each driver used to be ~170 lines of data loading and grid running. That now
lives in `qv.pipeline` and is shared, which is why the two reports are
comparable at all: they are produced by the same code, not by two scripts that
happened to agree.

### Two suites, and why dual momentum ships all three reports

An audit asks two independent questions. **Statistical validation** asks whether
the measured edge is distinguishable from luck. **Engine analysis** asks whether
the backtest can be trusted as an implementation, whatever its numbers say.

`dual_momentum/` ships the answer three ways, because the split *is* the
finding:

| Report | Verdict | Findings |
|---|---|---|
| [`report_statistical/`](dual_momentum/report_statistical/) | **Materially weakened** | alpha t = 0.30, PBO 59.1% |
| [`report_engine/`](dual_momentum/report_engine/) | **Survived the tests applied** | none |
| [`report.html`](dual_momentum/) (full) | Materially weakened | the two statistical ones |

The strategy is **impeccably implemented and statistically weak**. A single
combined verdict blurs that into "materially weakened" and leaves the
researcher with no idea what to fix — when the answer is that there is nothing
to fix in the code, and the problem is that the edge is not there.

The engine report is also a fifth the size (281 KB against 1.4 MB), because it
carries three charts instead of nine.

---

## sector_momentum: what the audit found

Cross-sectional momentum on sector ETFs is a standard, widely published
construction. The implementation is deliberately clean — signals from trailing
data only, positions shifted a session before they earn anything, a genuinely
monthly rebalance, no full-sample statistics anywhere. If it were leaky the
audit would only be finding my bug, and the interesting question is what an
audit says about something that is honestly built and genuinely popular.

### Performance, against SPY over the same 5,032 sessions

| Metric | Sector momentum | SPY | Difference |
|---|---:|---:|---:|
| Total return | 550.8% | 612.4% | −61.6 pp |
| Annualised return | 9.83% | 10.33% | −0.50 pp |
| Annualised volatility | 18.79% | 19.03% | −0.24 pp |
| Sharpe (excess) | 0.510 | 0.530 | −0.020 |
| Sortino (excess) | 0.711 | 0.743 | −0.033 |
| Maximum drawdown | −48.0% | −55.2% | +7.2 pp |
| Calmar | 0.205 | 0.187 | +0.018 |
| Hit rate | 51.6% | 55.1% | −3.5 pp |

Twenty years of monthly sector rotation, and it returns slightly less than
buying the index and doing nothing. It does deliver a shallower worst drawdown
— 48% against 55% — which is a real if modest benefit, and the only column
where the strategy wins.

### Fama-French 5 + momentum, Newey-West standard errors

| Term | Estimate | t | p |
|---|---:|---:|---:|
| **Alpha (annualised)** | **−1.03%** | **−0.70** | **0.484** |
| Mkt-RF | +0.944 | +44.09 | 0.000 |
| SMB | −0.047 | −1.65 | 0.099 |
| HML | +0.051 | +1.40 | 0.163 |
| RMW | +0.077 | +1.40 | 0.162 |
| CMA | +0.198 | +2.87 | 0.004 |
| Mom | +0.279 | +10.53 | 0.000 |
| R-squared | 84.9% | | |

This is the finding. Alpha is *negative* and nowhere near significant
(p = 0.48), the strategy is essentially fully invested in the market
(beta 0.94), and it carries a large, highly significant momentum loading
(+0.28, t = 10.5). The factors explain 85% of the variance.

The strategy works exactly as advertised — it really does harvest momentum —
and that is precisely why it is not a discovery. You can buy the momentum
factor. What this construction adds on top of it is, after twenty years,
slightly less than nothing.

### Against a volatility-matched SPY

| Measure | Value |
|---|---:|
| Strategy Sharpe (annualised, excess) | 0.510 |
| SPY at matched volatility | 0.530 |
| Sharpe the strategy adds | **−0.020** |
| Leverage to match volatility | 0.99x |
| Correlation / beta to SPY | 0.89 / 0.88 |

Twenty years of monthly rotation across nine sectors reproduces SPY at 0.99x
leverage, with a correlation of 0.89, and gives back two hundredths of a Sharpe
point for the trouble.

### What the strategy passed

Worth being explicit about, because a validator that only ever condemns is
useless:

| Test | Result |
|---|---|
| Deflated Sharpe (24 trials declared) | **0.969** — survives |
| BHY multiple-testing haircut | **fails**: t 2.27 → 1.37, against the 2.72 it wanted |
| Minimum backtest length | 3.9 years needed, 20.0 available |
| PBO | 0.474 — the search is a coin flip, not actively harmful |
| Probability of an out-of-sample loss | <0.1% |
| Parameter plateau | 97% retention — a plateau, not a spike |
| Break-even cost | 200 bps against 1–3 bps realistic — 67x margin |
| Volatility regimes | Sharpe 0.74 / 0.44, same sign in both |
| Start-date sensitivity | Sharpe 0.51 to 0.82, stable |
| Survivorship | universe declared point-in-time; nothing enters mid-sample |

Selection bias in the *Deflated Sharpe* sense is genuinely not the problem here:
the 24-configuration grid is small, the chosen point sits on a plateau rather
than a spike, and the sample is five times longer than the minimum the trial
count demands. The BHY haircut disagrees — judged at its rank of 12 among the
24 trial t-statistics, this result needed t = 2.72 and has 2.27, a 40% haircut
— and the report prints that disagreement rather than reporting only the
friendlier of the two.
Monthly rebalancing across nine liquid ETFs turns over 4.8x a year, which is
nowhere near enough to matter at ETF spreads.

**The strategy is falsified on attribution alone.** Everything else about it is
sound. That is a more useful result than a scattergun of complaints, and it is
what the report leads with.

### Resampled risk

| Statistic | Realised | Resampled median | 5th to 95th |
|---|---:|---:|---:|
| Total return | 550.8% | 566.1% | 81.8% to 2140.5% |
| Annualised Sharpe | 0.51 | 0.52 | 0.17 to 0.86 |
| Maximum drawdown | −48.0% | −39.1% | −58.3% to −26.5% |

The realised −48% drawdown sits at the 19th percentile of block-resampled
paths, and a matched random walk would have expected −40.8%. So it is
somewhat deep but well inside the range - not a path-dependent accident, and
something to expect again. The honest planning number is the −58% at the fifth
percentile, and even that understates, because block resampling cannot
reassemble a decline that unfolded over seventeen months.

### One number worth a second look

The empirical max-Sharpe null came in 32% *above* the analytic expectation
(ratio 1.32), which the report flags as the closed form under-penalising: trial
Sharpes here have fatter tails than the normal approximation assumes. With a
Deflated Sharpe of 0.969 it does not change the conclusion, but on a marginal
result it would, and the tool says so rather than quietly reporting the
friendlier number.

### Look-ahead: what the two leakage tests each conclude

The engine analysis runs the behavioural test on both strategies. It and the
static scanner disagree, and the disagreement is the interesting part.

`qv scan` raises one **critical** finding on each `strategy.py`:
`LEAK-NEGATIVE-SHIFT`, at `strategy.py:74` for the sector rotation and
`strategy.py:139` for dual momentum. Both point at `month.ne(month.shift(-1))`
— the monthly-rebalance mask. To know whether today is the last session of the
month you have to look at tomorrow's date. A trader does know next month's
calendar, so this is legitimate; but a static scanner cannot tell a calendar
from a price, and it is right to flag it and make someone look.

The behavioural test is what settles it. Corrupt every price from a cut point
onward, re-run the strategy, and no position before the cut moves — across
four corruption modes, four repeats each, at twelve cut points. **Neither
strategy reads future prices.**

An earlier version of the scanner raised a second critical finding,
`LEAK-FULL-SAMPLE-STATISTIC`, on `weights[risk_assets].sum(axis=1)` in
`dual_momentum/strategy.py`. That one was a bug in the scanner: an aggregate
taken across columns reads one timestamp and no other, so it has no future to
reach into. It is fixed, and a test pins the fix.

**What the clean result is worth.** The report says *"no dependence on data
more than about 22 periods after the decision point"*, not *"no leakage"*, and
the difference is real. A monthly rebalancer cannot betray a shorter
look-ahead — the contaminated signal is overwritten at the next rebalance
before it reaches the book. Measured on the sector strategy by injecting
look-aheads of known length, the test catches anything reaching about four
sessions or more past the decision point, and correctly reports nothing for a
leak that nets out to same-day, that being a Tier 1 accounting question rather
than look-ahead at all. The trading calendar is likewise outside its reach,
because the calendar is handed to the strategy rather than corrupted.

### What could not be tested

- **Matched-exposure random entry** — the position series is multi-asset. That
  test asks whether a single instrument was timed well, and reordering a
  cross-sectional book in time does not answer it. The tool skips it rather
  than running something that would produce a number meaning nothing.

That is now the only entry on either statistical report, and the engine reports
have none at all: everything the engine analysis wanted was supplied.

### What the engine analysis found on dual momentum

| Check | Result |
|---|---|
| Reads the future? | **no** — clean across 12 cut points, floor 21 periods |
| Same input, same output? | **yes** — 3 calls, zero disagreement |
| Is there a decision to examine? | **yes** — 226 changes over 5,032 sessions, 227 distinct books |
| Survives being traded a period late? | **94% of the Sharpe retained** (0.540 to 0.507) |
| Universe complete over the sample? | **yes** — all 11 instruments, none entering or leaving |

The delay curve is the one worth dwelling on. Sharpe decays 0.540 → 0.507 →
0.504 → 0.508 → 0.493 → 0.394 across delays of 0, 1, 2, 3, 5 and 10 sessions.
A monthly rebalance *should* be insensitive to a day or two of slippage, and it
is — which is exactly the result that makes the test meaningful when it fails
for something else. Trading cost and trading delay are different questions:
this strategy has a 67x cushion against realistic costs and would also have
survived losing a week of execution timing.
