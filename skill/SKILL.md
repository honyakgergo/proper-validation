---
name: proper-validation
description: Audit a quantitative backtest for selection bias, look-ahead leakage, cost fragility and factor exposure. Use when a user asks you to validate, check, review or stress-test a trading strategy, backtest or research notebook, or asks whether a Sharpe ratio is real.
---

# Auditing a backtest with proper_validation

You are auditing someone's research. Two things govern how you do it.

**The tool falsifies; it cannot validate.** Never tell a user their strategy works. The strongest
honest statement available is "the tests that this data supports did not break it". Say that
instead, and say which tests could not be run.

**Never compute the statistics yourself.** Every capability is a deterministic CLI subcommand. Your
job is to get the researcher's work into a shape the engine can consume, run it, and interpret what
comes back. An LLM estimating a Deflated Sharpe by hand is exactly the unearned confidence this
tool exists to remove.

## The protocol

### 0. Check the tool is actually there

```bash
qv --help
```

Do this first, before reading anything of the researcher's. You are almost certainly running in
*their* project, while `qv` was installed from a clone of this repository somewhere else — and if
that install went into a virtualenv inside the clone, the command does not exist here.

If it is not found, **stop and say so**. Do not work around it: not by reading `qv/` and computing
the statistics yourself, not by `pip install`-ing this package into the researcher's environment
(it pins numpy, pandas, scipy and statsmodels, and disturbing the environment their research runs
in is its own kind of damage). Give them the fix and wait:

```bash
pipx install --editable "/path/to/proper-validation[data]"    # or: uv tool install --editable ...
```

That puts `qv` on PATH for every project without touching theirs. `qv skill install --user` then
keeps the skill available everywhere too. The `[data]` extra matters: without it the manifest
`qv adapter init` writes cannot fetch prices.

### 1. Establish the tier

What the audit can conclude depends strictly on what exists. Find out before promising anything.

**Two suites, two independent questions.** Pick deliberately; do not reflexively run both.

| Suite | Asks | Needs |
|---|---|---|
| `statistical` | Is the measured edge distinguishable from luck, and does it survive honest accounting? | A return series. Positions add costs; `n_trials` adds deflation; factors add attribution |
| `engine` | Can this backtest be trusted as an implementation, whatever its numbers say? | A **re-runnable strategy**. Positions plus asset returns add the delay curve; the unaligned prices add coverage |
| `full` | Both | The union of the above |

```bash
qv validate --manifest research_manifest.yaml --suite statistical
qv validate --manifest research_manifest.yaml --suite engine
qv validate --manifest research_manifest.yaml            # full, the default
```

**Run them separately when the answers might differ, and say which is which.** A strategy can be
statistically hopeless and impeccably implemented, or the reverse — and the reverse is the
dangerous case, because the numbers look fine. One combined verdict blurs the two into a single
"materially weakened" that tells the researcher nothing about what to fix.

The trial matrix and parameter surface that PBO and the plateau test need are statistical, and do
not require the callable — a researcher who can hand over the returns of every configuration they tried gets
both from `--trial-matrix`. Ask for that whenever the manifest route is not available.

**Ask for the trial count separately from the grid.** `search.axes` records the grid, and the audit
treats it as a floor, because a grid cannot contain the configurations abandoned before anyone
wrote them down. `search.n_trials` is where the interview-revised number goes, and the audit uses
whichever is larger. Overstating is the honest direction: the correction is logarithmic.

**Push for a re-runnable strategy, and there is a command for it.** The behavioural leakage test is the only
thing that can settle whether a strategy reads the future. Getting there is two files, and
`qv adapter init` writes both:

```bash
qv adapter init --out audit/          # audit/qv_adapter.py + audit/research_manifest.yaml
# fill in the adapter and the manifest, then
qv validate --manifest audit/research_manifest.yaml --out audit/report/
```

Do this rather than writing a driver script. The command loads the data, evaluates the reported
configuration, re-runs the declared grid to rebuild the trial matrix and parameter surface, and
wires up the behavioural test — all of which used to be a hundred lines of glue that differed
between audits and made two reports of the same strategy incomparable.

**Writing the adapter is usually lifting one cell into a `def`.** It takes the price frame and the
parameters, and returns positions:

```python
def positions(prices: pd.DataFrame, **params) -> pd.DataFrame:
    ...   # the notebook cell that computes the signal, with `prices` as its only input
```

Point `strategy.adapter` at it as `qv_adapter.py:positions`. It may return one signal per row or a
book of weights, so a cross-sectional rotation needs no reshaping. If you are calling the Python
API directly instead, `qv.adapter.frame_adapter(positions, index, columns, **params)` derives the
single-argument callable `AuditInputs.strategy` wants, and `strategy_data` is
`prices.to_numpy()` — both are required, and either alone silently drops the engine suite.

**Never let the adapter read the researcher's DataFrame from an enclosing scope.** The corruption
then never reaches the strategy, the test passes, and the report claims a clean bill of health it
has not earned — indistinguishable from an honest result. `frame_adapter` takes labels and no
values, so it cannot express the mistake; if you write an adapter by hand, use only the array you
are given.

**Read the detection floor before repeating a clean result.** A clean run is reported as "no
dependence on data more than about N periods after the decision point", where N is how often the
signal actually moves. On a monthly rebalancer N is around 21, and a shorter look-ahead is
genuinely undetectable. Quote the qualified claim, not "no leakage".

### 2. Scan before you run

```bash
qv scan research.ipynb
```

Static findings are prompts, not verdicts. The scanner recognises shapes; it cannot follow data
through variables or see inside library calls. When it flags `LEAK-NEGATIVE-SHIFT`, go and read the
line: a negative shift building a supervised *label* is correct, and the same call building a
*feature* is fatal. Report which one you found.

### 3. Recover the trial count

```bash
qv trials research.ipynb --reported 1
```

This is the highest-value thing the agent layer does. `n_trials` is the key input to every
multiple-testing correction and almost nobody records it — not through dishonesty, but because
nobody counts. The notebook does: execution counts, parameter literals reassigned across cells,
explicit grid definitions, filename lineage.

The number that comes back is a **lower bound**. Configurations abandoned without being saved leave
no trace. Interview the researcher and revise upward:

- How many variants did you try and discard before this one?
- Did you look at out-of-sample results before finalising the parameters?
- Where did the universe list come from, and when was it constructed?
- Did you change the date range after seeing results?
- How many times did you re-run the whole notebook with different settings?

Each answer either raises `n_trials` or disables a test the data cannot support. When the
researcher is unsure, **overstate it**: the correction is logarithmic in the trial count, so
honesty is cheap and understatement is not.

### 4. Run the audit

```bash
qv validate returns.csv \
  --positions positions.csv \
  --asset-returns spy.csv \
  --trials 240 \
  --asset-class us_large_cap_etf \
  --periods-per-year 252 \
  --out audit/
```

`--positions` takes either shape: one column for a strategy timing a single instrument, or a book
of weights with one column per instrument for a cross-sectional strategy. Pass the whole book. It
is what turnover, cost and break-even are derived from, and a single column of a book measures one
instrument's trading and reports it as the portfolio's.

Supply `--trial-matrix trials.csv` when you can reconstruct the search — a `(T, N)` matrix with one
column per configuration examined. It unlocks PBO and lets the empirical null measure how
correlated the trials actually were, which the closed form assumes away.

Exit code is 1 when anything CRITICAL fires, so this drops into CI unchanged.

### 5. Interpret

Read `report.json`; it is authoritative for every number on the page. Then explain the findings in
the researcher's own terms:

```bash
qv explain SELECT-DEFLATED-SHARPE-FAILS
```

Lead with the finding that would change their decision, not with the longest list. Usually that is
one of:

- **Deflated Sharpe below 0.95** — the result is inside what a search of this size produces from
  noise. Nothing recovers this except fewer trials, more data, or genuinely fresh data.
- **Break-even cost below realistic cost** — the gross edge may be real; the net edge is not there.
- **Behavioural leakage** — the strategy reads the future. Everything else in the report is moot.
- **Alpha not significant after factors** — they have rediscovered exposures that cost a few basis
  points to buy.

### 6. Say what you could not test

The report has a *what could not be tested* section. Reproduce it in your summary. An audit that
quietly omits its gaps is doing the thing it criticises.

## Things to get right

- **Per-period versus annualised Sharpe.** The engine takes per-period returns and a
  `periods_per_year`. Mixing these up produces a deflated Sharpe wrong by a factor of `sqrt(252)`
  that still looks plausible.
- **Quarterly and monthly strategies.** Below ~30 observations the tool refuses to make asymptotic
  claims. Do not talk the user past that refusal; a decade of quarterly returns is 40 points and a
  Sharpe of 1.3 there carries a standard error near 0.35 before any multiple-testing adjustment.
- **A clean run is not a pass.** If no findings fire, say: "the tests this tier supports did not
  falsify it, and here is what could not be tested."
- **Do not soften a critical finding** because the researcher is invested in the strategy. Report
  what the engine found and what would have to change.
- **`qv fix` does not exist**, deliberately. Auto-rewriting someone's methodology would mean the
  tool implicitly blessing the result. Explain and let them change it.

## References

- `references/findings_catalog.md` — every defect the tool can name, with detection and
  remediation. Generated from `qv/findings.py`; regenerate with `qv explain --markdown`.
- `references/adapter_protocol.md` — the contract a re-runnable strategy has to satisfy.
- `references/manifest_schema.md` — `research_manifest.yaml`, which pins what the audit assumed.
