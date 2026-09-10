# Dual momentum with volatility targeting

A serious strategy, audited honestly. It **does** beat SPY on a risk-adjusted
basis — by far less than the first version of this page claimed — and the audit
finds two real problems on top of that.

```bash
python real_user_tests/dual_momentum/run.py
python real_user_tests/dual_momentum/run.py --offline
```

## What it is

Three effects, each taken from published work rather than found by searching
this data:

- **Relative momentum** — rank the nine SPDR sector funds by 12-1 month return,
  hold the top three. Jegadeesh and Titman (1993).
- **Absolute momentum** — a leader only earns its weight if its own trailing
  return is positive; otherwise that weight goes to the strongest of IEF
  (Treasuries) or GLD (gold). Antonacci (2014). This is what lets the strategy
  step aside rather than rotate into the least-bad sector.
- **Volatility targeting** — scale the whole book by `10% / trailing realised
  volatility`, capped at 1.5x. Moreira and Muir (2017).

Monthly rebalance, signals from trailing windows only, positions traded from
the session after the signal. All 54 configurations examined are declared, and
the universe is declared point-in-time: the nine sector funds and IEF/GLD all
trade for the whole sample, so nothing enters it with hindsight.

## It beats SPY, by less than it looks

Sharpe and Sortino are **in excess of the risk-free rate** (1.56% a year over
this sample, Ken French's RF); returns, drawdown and Calmar are total-return
figures. Both columns are computed the same way.

| Metric | Dual momentum | SPY |
|---|---:|---:|
| Annualised return | 7.44% | **10.33%** |
| Annualised volatility | **11.67%** | 19.03% |
| **Sharpe (excess)** | **0.540** | 0.530 |
| Sortino (excess) | 0.7431 | **0.7433** |
| **Maximum drawdown** | **−27.1%** | −55.2% |
| **Calmar** | **0.274** | 0.187 |
| Correlation to SPY | 0.74 | — |
| Beta to SPY | 0.46 | 1.00 |
| Market beta (FF `Mkt-RF`) | 0.51 | — |

The drawdown is halved — the absolute filter moved into Treasuries through 2008,
which is exactly the thing it exists to do — and Calmar is 47% better. The
Sharpe advantage is **+0.010**, and Sortino is a dead heat.

### Why this is not what the earlier version of this page said

It used to claim a Sharpe of 0.674 against 0.612, an advantage of +0.062. Both
figures were computed on total returns rather than in excess of cash. Over
2005-2024 that credits every strategy with a 1.56% a year cash return it did
not earn, and it credits the *less volatile* one with more of it, because the
same numerator sits over a smaller denominator. Correcting it takes the
strategy to 0.540 and SPY to 0.530, and 84% of the claimed edge disappears.

There is an error in the other direction, and the report now quantifies it: the
book is only **65% invested on average**, and the backtest pays nothing on the
idle 35%. At the bill rate that is **0.54% a year** of return the strategy
should have earned and did not. Credit it and the excess Sharpe goes to roughly
0.59, an advantage of about +0.06 — close to the number this page used to
claim, but for a reason that has to be argued rather than assumed.

So: the strategy is better than SPY per unit of risk. Which number you use for
*how much* better depends on a funding assumption, and the report now makes you
state it instead of hiding it in a convention.

## And the audit still weakens it

**Verdict: materially weakened.** Two findings.

### Alpha is not significant

| Term | Estimate | t | p |
|---|---:|---:|---:|
| **Alpha (annualised)** | **+0.47%** | **+0.30** | **0.764** |
| Mkt-RF | +0.511 | +12.46 | 0.000 |
| SMB | −0.007 | −0.25 | 0.799 |
| HML | +0.015 | +0.63 | 0.530 |
| RMW | +0.016 | +0.44 | 0.660 |
| CMA | +0.122 | +2.56 | 0.010 |
| Mom | +0.235 | +10.09 | 0.000 |
| R-squared | 66.4% | | |

Alpha is *positive* — unlike the plain sector rotation next door, which came in
negative — but at t = 0.30 it is nowhere near distinguishable from zero.
Twenty years of daily data is not enough to prove an improvement this small.
That is a statement about statistical power, not about the strategy being bad,
and the difference matters.

Note the market beta of 0.51 against 0.94 for the plain rotation. The defensive
leg genuinely reduces market exposure. But bonds and gold are not spanned by
FF5 plus momentum, so part of what shows up as alpha here is simply exposure to
asset classes the factor model does not contain — the report says so rather
than claiming it as skill.

### The parameter selection does not generalise

**PBO = 59.1%**, median out-of-sample rank **0.436**. Across 12,870
combinatorial splits, the configuration that looks best in-sample lands in the
bottom half out-of-sample more often than not — worse than a coin flip.

A third thing worth saying, because it is the obvious objection: **54 trials is
a floor, not a count.** Dual momentum is a published effect, and the search that
produced it — Antonacci's, and everyone else's, largely over US equities in this
same era — is not in the number. This audit can price the 54 configurations
declared for this backtest and nothing else, and the report now says so on the
page. If you think the strategy family is overfitted to the post-1970 US sample,
that is a reasonable belief, and no amount of deflating 54 trials will settle
it. The rolling-origin test and an out-of-sample rerun on a different market are
where that question is answered.

Two things this finding is *not*. It is not a contradiction of the performance
table: the *structural* choices — an absolute filter, a defensive leg, a
volatility overlay — deliver the drawdown improvement and are visible in the
equity curve, while the *parameter* choices on top of them are noise. And it is
not a description of what was actually done here: the manifest picked the
conventional 12-1 specification, not the grid maximum (0.674 against 0.686 on
the old basis), so the in-sample-maximum rule PBO evaluates is a rule this
research did not follow. What PBO establishes is that the parameter surface
carries no structure that survives out of sample — which is a reason to stop
tuning it, not evidence that this particular point was tuned.

The report's second PBO panel now shows what the in-sample winner actually
*earns* out of sample: a distribution centred on an annualised Sharpe of 0.46
with 0.1% of splits below zero. The conventional in-sample-against-out-of-sample
scatter is deliberately gone — CSCV splits one fixed sample into complementary
halves, so the two Sharpes sum to a constant and that scatter is a line of
slope −1 for any strategy whatsoever. It looked like devastating evidence of
decay and was an identity.

## What it passed, and one thing it did not

| Test | Result |
|---|---|
| Deflated Sharpe (54 trials declared) | **0.965** — survives |
| BHY multiple-testing haircut | **fails**: t 2.38 → 1.16, against the 3.06 it wanted |
| Minimum backtest length | 5.3 years needed, 20.0 available |
| Probability of an out-of-sample loss | 0.1% |
| Parameter plateau | 90% retention — a plateau, not a spike |
| Break-even cost | 132 bps against 1–3 bps realistic — 44x margin |
| Resampled drawdown | realised −27.1% vs resampled median −27.1% |
| Volatility regimes | 0.57 / 0.54, same sign in both |
| Start-date sensitivity | 0.54 to 0.73 across 20 start dates |

**The two selection-bias corrections disagree, and the report now says so.**
The Deflated Sharpe asks whether the result beats the best of 54 draws from a
no-edge null, and it clears that bar at 0.965. The Harvey-Liu BHY haircut asks
whether the t-statistic survives a false-discovery correction across 54 tests,
and it does not: t 2.38 becomes **1.16** (p = 0.24), against the **3.06** this
result needed at its rank. That is a 51% haircut — the Sharpe of 0.64 net of
costs becomes **0.31**.

The adjustment runs over the whole family of 54 trial t-statistics, where this
configuration ranks 11th. An earlier version judged it as though it were the
most significant of 54, which is Bonferroni times the harmonic number — a 248x
multiplier that absorbed the t-statistic entirely and printed 0.00. That number
was a floor, not a measurement: every result below t = 2.9 rendered identically,
and it was reached by discarding 53 of the 54 tests the audit already had in
memory. Neither test is wrong now; the honest reading is that selection bias
here is *unsettled*, not passed.

## The drawdown resampling

The realised maximum drawdown of −27.1% sits at the **45th percentile** of
block-resampled paths, whose median is also −27.1% (5th to 95th: −40.1% to
−16.5%). A matched random walk with the same drift and volatility would expect
−25.5%.

So the drawdown is entirely typical — no ordering effect, and `RISK-DRAWDOWN-
PATH-DEPENDENT` correctly stays quiet. That is worth knowing: the −27% is a
property of the return distribution and would be expected to recur, not a piece
of bad luck that happened once.

It also puts a range on it. Anyone sizing this strategy on "it only drew down
27%" should be sizing it on the −40% at the fifth percentile instead — and even
that understates, because block resampling cannot reassemble a long decline.

## Engine analysis: the implementation, judged separately

The audit asks two independent questions, and this strategy answers them very
differently. The statistical suite weakens it; the **engine analysis clears it
completely** — no findings, nothing it could not test.

```bash
python real_user_tests/dual_momentum/run.py --suite engine
python real_user_tests/dual_momentum/run.py --suite statistical
python real_user_tests/dual_momentum/run.py                    # both
```

All three reports are committed, and the contrast is the point: there is
nothing to fix in the code, and the problem is that the edge is not there.

| Check | Result |
|---|---|
| Reads the future? | no — clean at 12 cut points, detection floor 21 periods |
| Same input, same output? | yes |
| Is there a decision to examine? | yes — 226 changes, 227 distinct books |
| Survives a period of execution delay? | 94% of the Sharpe retained |
| Universe complete over the sample? | yes — all 11 instruments |

**Execution-delay fragility** is the check most likely to catch a strategy like
this one out, and it does not. Sharpe runs 0.540 → 0.507 → 0.504 → 0.508 →
0.493 → 0.394 at delays of 0, 1, 2, 3, 5 and 10 sessions. A monthly rebalance
ought to be insensitive to a day or two of slippage, and it is. Note that this
is a different question from cost: the strategy has a 67x cushion against
realistic fees *and* would survive losing a week of execution timing, and
either could have failed without the other.

`run.py` hands the audit `dual_momentum_positions` itself, wrapped by
`qv.adapter.frame_adapter`, so the validator can re-run the strategy on data it
controls rather than take the code on trust.

`qv scan` flags one critical finding on `strategy.py` — `LEAK-NEGATIVE-SHIFT`
at line 139, the `month.ne(month.shift(-1))` that decides which session ends
the month. It is a false positive: an exchange calendar is known years ahead.
But a static scanner cannot distinguish a calendar from a price, so it flags it
and makes someone check, which is the correct behaviour for a test that
recognises shapes rather than proving anything.

The behavioural test does the proving. Every price from a cut point onward is
destroyed — four ways, four times each, at twelve different cut points — and
no weight anywhere before the cut moves. The volatility overlay is the part
most likely to leak, since `realised_volatility` is a rolling standard
deviation and a centred window there would be invisible to inspection; it
comes back clean, and a deliberately centred variant is caught by the test in
`tests/test_adapter.py`, which is what makes the clean result mean something.

**Stated precisely, the report concludes "no dependence on data more than
about 21 periods after the decision point"** — not "no leakage". A strategy
that rebalances monthly cannot reveal a shorter look-ahead, because the
contaminated weight is overwritten at the next rebalance before it ever earns
a return. Twenty-one sessions is where this strategy's resolution sits, and
the report prints it rather than letting "clean" stand unqualified.

## Honest caveats

- **Bonds had a twenty-year bull market.** The defensive leg was rewarded by a
  falling-rate regime that ended in 2022. A rerun starting in 2022 would look
  very different, and the rolling-origin test in the report is where to look.
- **Gold and Treasuries are not in FF5.** Some of the positive alpha is
  cross-asset exposure the factor model cannot see.
- **The result is a Sharpe improvement of 0.01 over twenty years** as
  backtested, or roughly 0.06 if uninvested cash is credited at the bill rate.
  Real, in the right direction, consistent with the literature — and
  statistically indistinguishable from zero on this sample either way.
