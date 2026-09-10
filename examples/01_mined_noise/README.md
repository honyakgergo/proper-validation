# Mined noise: best of 10,000 random rules on SPY

The flagship demonstration, on real market data.

```bash
python examples/01_mined_noise/mine.py            # fetches SPY once, then caches
python examples/01_mined_noise/mine.py --offline  # refuses the network
```

## What it does

Generates 10,000 trading rules on SPY between 2021-01-01 and 2024-12-31. Each rule is a random
conjunction of two thresholded technical indicators drawn from a bank of 30 — moving-average
ratios, momentum, mean-reversion z-scores, volatility, range position, sign streaks. Long when both
conditions agree, short otherwise.

Two deliberate choices make this a clean test of selection bias and nothing else:

- **Every signal is shifted forward a period**, so no rule can see the return it is scored on.
  There is no leakage here to confound the result.
- **Rules are long/short, not long-only.** A long-only rule on an index that rose over the period
  inherits the drift, and the demonstration would be about market beta instead. With long/short
  rules the **median** rule earns an annualised Sharpe of **−0.56**, confirming the family has no
  edge smuggled into it.

Then the winner goes through the full audit.

## What it finds

| Measure | Value |
|---|---|
| Best rule, annualised Sharpe | **1.42** |
| Median rule | −0.56 |
| Bootstrap 95% interval | 0.42 to 2.42 |
| Expected best-of-10,000 under the null (per period) | 0.125 |
| Observed (per period) | 0.081 |
| **Deflated Sharpe Ratio** | **0.081** |
| Minimum backtest length | 14.9 years needed, 4.0 available |
| BHY-adjusted t-statistic | 2.82 → 0.00, fully absorbed (10,000 trials wanted 4.44) |
| Percentile against the simulated null | 41st |
| Break-even cost | 32 bps (not the binding problem here) |

**Verdict: falsified.**

The winner is not merely unimpressive once selection is priced in — it is *below average* for a
search of this size on data with no edge. Mining 10,000 zero-edge rules on this window is expected
to produce a per-period Sharpe of 0.125; this one managed 0.081.

## Two things worth noticing in the report

**The empirical null disagrees with the closed form, and says why.** The simulated best-of-N null
comes in 24% below the analytic expectation. That is the tool detecting that the 10,000 rules were
not 10,000 independent bets — they share indicators and a single price history, so the effective
number of trials is far lower. The analytic Deflated Sharpe over-penalises here, and the report
says so rather than quietly reporting the harsher number.

**PBO is only 0.32.** Selection bias is severe, but the *selection procedure* is not actively
harmful: the in-sample winner does not systematically underperform out-of-sample, it is just a coin
flip. These are genuinely different questions, which is why both tests are here.

## The honest ceiling

Mining cannot manufacture an arbitrarily good Sharpe. The expected best of `N` zero-edge trials
grows only as `sqrt(2·log(N)/T)`, so on twenty years of daily data even ten thousand trials tops
out near 0.9. A four-year window is used here because that is where mining produces a number people
actually publish — which is itself the uncomfortable point, and an argument for treating short
backtests with more suspicion than long ones.

## Files

- `research_manifest.yaml` — what the audit assumed, with `end_date` pinned for reproducibility
- `mine.py` — fetch, mine, audit
- `report.html` — the committed self-contained report, charts inlined
- `report.json` — authoritative for every number the page shows
