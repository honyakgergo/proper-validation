# Findings catalog

32 distinct defects. Generated from `qv/findings.py` - edit there, not here.

## Leakage

### `LEAK-NEGATIVE-SHIFT` - Signal reads future data via a negative shift

**Severity:** Critical

A negative shift moves future values backwards in time, so a signal built from it knows the answer before it is knowable. This is the single most common way a backtest becomes fiction, and it usually produces a Sharpe so high that it should have prompted suspicion on its own.

**Detection.** AST scan for .shift(-n) with a negative literal argument.

**Remediation.** Negative shifts are legitimate for constructing a forward-looking *target* in supervised learning, but never for a feature. Confirm the shifted series feeds only the label, and that features use shift(+n) or a rolling window closed at t.

### `LEAK-BACKWARD-FILL` - Missing values filled from the future

**Severity:** High

Backward fill propagates a later observation into an earlier gap. Every filled cell then contains information that did not exist at that timestamp. It is easy to miss because the call looks like ordinary data hygiene.

**Detection.** AST scan for bfill(), backfill(), and fillna(method='bfill').

**Remediation.** Use forward fill, or leave the gap and let the strategy abstain. If a series genuinely has no value until later, the honest handling is to exclude the period rather than invent one.

### `LEAK-CENTERED-WINDOW` - Centred rolling window spans future observations

**Severity:** High

A centred window of width w at time t averages data from t - w/2 to t + w/2, so half its input is unavailable at t. Centring is right for descriptive smoothing in exploratory plots and wrong for anything a signal touches.

**Detection.** AST scan for rolling(..., center=True).

**Remediation.** Set center=False, which is the default, and widen the window if needed.

### `LEAK-FULL-SAMPLE-SCALER` - Scaler or transformer fitted on the full sample

**Severity:** High

Fitting a scaler on all the data leaks the mean and variance of the test period into the training period. The effect is subtle - it rarely produces an absurd Sharpe, just a persistently optimistic one - which is what makes it so common and so durable.

**Detection.** AST scan for fit() or fit_transform() outside a fold loop.

**Remediation.** Fit the scaler inside each training fold and apply it to the corresponding validation fold. A scikit-learn Pipeline inside TimeSeriesSplit does this correctly by construction.

### `LEAK-SHUFFLED-SPLIT` - Time series split randomly rather than chronologically

**Severity:** Critical

A shuffled split puts observations from after the test period into the training set. With autocorrelated data, neighbouring observations are near duplicates, so the model is effectively tested on data it has seen.

**Detection.** AST scan for train_test_split without shuffle=False, and for KFold or cross_val_score without a time-aware splitter.

**Remediation.** Use TimeSeriesSplit, or split by date with an explicit embargo gap between train and test to break the autocorrelation.

### `LEAK-FULL-SAMPLE-STATISTIC` - Full-sample statistic used where a trailing one belongs

**Severity:** Medium

A threshold set from the mean, standard deviation or a quantile of the whole series encodes knowledge of the whole series. Legitimate when the value is only reported, a leak when it feeds a signal.

**Detection.** AST scan for mean/std/median/quantile/max/min taken over a full column without a preceding rolling or expanding window.

**Remediation.** Replace with an expanding or rolling equivalent so the threshold at time t uses only data up to t. If the value is purely descriptive, say so in the manifest and this finding can be dismissed.

### `LEAK-BEHAVIOURAL` - Signal changes when only future data is corrupted

**Severity:** Critical

The behavioural test: corrupt the data strictly after time t, re-run, and the signal at t must be unchanged. If it moves, the strategy reads the future somewhere - regardless of what the source code appears to say. This catches leaks no static scan can see, including ones inside third-party library calls.

**Detection.** Re-run the strategy with post-t data replaced and compare signals at t.

**Remediation.** Bisect: corrupt progressively narrower future windows to localise which horizon the strategy is reaching into, then inspect the code that touches it.

## Selection

### `SELECT-UNDECLARED-TRIALS` - More configurations were tried than were reported

**Severity:** High

The number of trials is the key input to every multiple-testing correction and the most under-reported number in backtesting. Notebook forensics recovers a lower bound; the true count is always at least this and usually much more, because abandoned ideas leave no trace.

**Detection.** Notebook archaeology: out-of-order and repeated execution counts, parameter literals across cells, explicit grid searches, filename lineage.

**Remediation.** Record every configuration examined, including abandoned ones, and pass the count as n_trials. When in doubt, overstate it - the correction is logarithmic in the trial count, so honesty is cheap.

### `SELECT-DEFLATED-SHARPE-FAILS` - Sharpe does not survive the selection-adjusted bar

**Severity:** Critical

Once the number of trials is priced in, the observed Sharpe is no better than the best a search of this size would produce from noise. The strategy may still be real, but this backtest is not evidence that it is.

**Detection.** Deflated Sharpe Ratio below 0.95.

**Remediation.** Reduce the trial count by fixing parameters on theoretical grounds, extend the sample, or test the strategy on genuinely fresh data. Nothing else recovers a failed deflation.

### `SELECT-HIGH-PBO` - The selection procedure does not generalise

**Severity:** High

Across combinatorial in-sample and out-of-sample splits, the configuration that looks best in-sample lands in the bottom half out-of-sample more often than not. Above 0.5 the search is worse than useless: it actively selects strategies that underperform.

**Detection.** PBO above 0.5 from combinatorially symmetric cross-validation.

**Remediation.** Simplify the strategy until the parameter space is small enough to defend, or select on economic reasoning rather than on backtest performance.

### `SELECT-PARAMETER-SPIKE` - Chosen parameters are a spike, not a plateau

**Severity:** High

Neighbouring parameter values perform far worse than the chosen point. A genuine effect degrades gracefully as parameters move; an isolated peak means the parameter was fitted to noise and will not survive contact with new data.

**Detection.** Mean neighbourhood score below half the chosen point.

**Remediation.** Choose the centre of the widest plateau rather than the global maximum, and report performance across the plateau rather than at its best point.

### `SELECT-INSUFFICIENT-LENGTH` - Backtest is too short to support its claim given the trial count

**Severity:** High

Minimum Backtest Length gives the years of data needed before an observed Sharpe means anything at this number of trials. Below it, the result is within the range that a search of this size produces from noise.

**Detection.** Sample length below the minimum backtest length for the trial count.

**Remediation.** Extend the sample, or reduce the number of configurations examined.

## Statistics

### `STAT-NAIVE-ANNUALISATION` - Sharpe annualised with sqrt(q) despite serial correlation

**Severity:** Medium

Multiplying by the square root of the number of periods assumes returns are serially independent. Positive autocorrelation - normal for slow-moving or overlapping signals - makes this materially too generous.

**Detection.** Lo autocorrelation-adjusted factor more than 10% below sqrt(q).

**Remediation.** Report the autocorrelation-adjusted Sharpe. If the gap is large, investigate whether the returns overlap - overlapping windows are the usual cause.

### `STAT-NO-CONFIDENCE-INTERVAL` - Point estimate reported with no interval

**Severity:** Medium

A Sharpe of 1.3 over 40 observations carries a standard error near 0.35 before any multiple-testing adjustment. Reported bare, it invites a precision the data cannot support.

**Detection.** Reported metrics without an accompanying interval.

**Remediation.** Report a stationary-bootstrap confidence interval alongside every headline figure.

### `STAT-SMALL-SAMPLE` - Too few observations for the methods applied

**Severity:** High

Below roughly 30 observations the asymptotic results behind the Sharpe standard error, the Deflated Sharpe and the bootstrap all degrade. Quarterly strategies hit this routinely: a decade of quarterly returns is 40 points.

**Detection.** Fewer than 30 usable observations.

**Remediation.** Increase the observation frequency if the strategy allows it, extend the sample, or state plainly that the result is indicative rather than inferential.

### `RISK-DRAWDOWN-PATH-DEPENDENT` - Worst drawdown came from the ordering, not the distribution

**Severity:** Medium

The realised maximum drawdown is deeper than 95% of block-resampled paths drawn from the same returns. The loss was produced by the sequence in which returns arrived rather than by their distribution, which means resampling cannot reproduce it and a shorter or reordered sample would never have revealed it. Reported as a single number, such a drawdown reads as a property of the strategy when it is a property of one path.

**Detection.** Realised maximum drawdown below the 5th percentile of a stationary-bootstrap distribution of the same statistic.

**Remediation.** Report the resampled interval alongside the realised figure, and size the strategy for the deeper end of it. Note that block resampling already understates severity by breaking long declines, so the true risk is worse than even the resampled range suggests.

### `MC-IID-RESAMPLE` - Monte Carlo that resamples returns independently

**Severity:** Medium

Shuffling trade order or resampling returns independently destroys serial dependence and volatility clustering, so the drawdown quantiles it produces are systematically optimistic. Reordering trades also tests nothing about whether the edge was real.

**Detection.** AST scan for np.random.permutation, np.random.choice or sample(frac=1) applied to a returns or trade series.

**Remediation.** Use a stationary bootstrap with an automatically selected block length, which preserves short-range dependence while still resampling.

### `MC-FORWARD-PROJECTION` - Simulated future paths presented as risk analysis

**Severity:** Medium

Projecting equity forward from a fitted return distribution makes a claim about the future on the strength of an assumed data-generating process. The fan chart looks rigorous and encodes nothing beyond the assumption.

**Detection.** AST scan for random draws used to build forward equity paths.

**Remediation.** Report realised drawdown and its bootstrap interval instead. Historical evidence about what did happen is worth more than simulated evidence about what an assumed model says might.

## Costs

### `COST-BELOW-REALISTIC` - Break-even cost is below a realistic trading cost

**Severity:** Critical

The cost at which the strategy stops making money is lower than what trading it would actually cost. The gross edge may be real; the net edge is not there.

**Detection.** Break-even cost in basis points below the asset class estimate.

**Remediation.** Reduce turnover, trade more liquid instruments, or accept that the effect is not tradable at this frequency.

### `COST-ASSUMED-NOT-DERIVED` - Costs asserted rather than computed from turnover

**Severity:** Medium

A round-number cost applied per period, rather than per unit of traded notional, breaks the link between how often the strategy trades and what it pays - which is the entire mechanism by which costs kill strategies.

**Detection.** Manifest declares a cost assumption with no position series supplied.

**Remediation.** Supply the position series so turnover can be measured, and let the cost follow from it.

## Attribution

### `ATTR-LEVERED-BETA` - Returns are explained by factor exposure

**Severity:** Critical

Alpha is not significant once factors are accounted for, and the factors explain most of the variance. The strategy is a repackaging of exposures available at a few basis points.

**Detection.** Factor regression alpha not significant with high R-squared, or a volatility-matched benchmark matching the strategy Sharpe.

**Remediation.** Hedge the factor exposures and re-test the residual, which is the only part that could be alpha.

## Robustness

### `ROBUST-REGIME-DEPENDENT` - Edge exists in only one market regime

**Severity:** High

Performance is positive in one volatility regime and negative in the other. The strategy is a bet on that regime persisting, which is a different and much less attractive proposition than the headline suggests.

**Detection.** Sharpe signs disagree across the volatility split.

**Remediation.** State the regime dependence explicitly and size the strategy for the possibility that the regime ends. Consider a regime filter, priced for the extra trials it costs.

### `ROBUST-START-DATE-SENSITIVE` - Conclusion depends on the start date

**Severity:** High

Moving the start of the sample forward flips the sign of the result. A finding that survives only from one particular starting point is a property of that window, not of the strategy.

**Detection.** Rolling-origin Sharpes disagree on sign.

**Remediation.** Report the full range of start dates rather than one, and investigate what the early period contains that the later ones do not.

### `ROBUST-TOP-DAY-DEPENDENT` - Profit concentrated beyond what the Sharpe explains

**Severity:** High

The best few periods supply far more of the profit than a normal series of this Sharpe would produce. The strategy is closer to a lottery ticket than its Sharpe suggests, and the next such period may not arrive.

**Detection.** Observed profit share more than 1.5x the normal-distribution baseline.

**Remediation.** Examine the outlier periods individually. They are often a data error, a corporate action, or a single position that should have been capped.

### `ROBUST-NO-TIMING-SKILL` - Signal timing does not beat random timing at the same exposure

**Severity:** High

Reordering the position series at random, holding average exposure and turnover fixed, produces the same performance. Whatever the strategy earns, it earns from being in the market, not from choosing when.

**Detection.** Matched-exposure randomisation p-value above 0.20.

**Remediation.** Compare against a constant-exposure benchmark. If the signal adds nothing, the simpler position is the honest one.

## Data

### `DATA-SURVIVORSHIP` - Universe built from current index membership

**Severity:** High

A universe of instruments that exist today excludes everything that failed, delisted or merged. The backtest then trades a portfolio nobody could have selected at the time, and the bias is upward by construction.

**Detection.** Manifest declares a universe from a current-membership source.

**Remediation.** Use a point-in-time constituent list. Where none is available, say so and treat the result as an upper bound.

### `DATA-QUALITY-GAPS` - Price series has gaps or adjustment artifacts

**Severity:** Medium

Missing days, zero volumes and unadjusted split jumps produce spurious returns. A single unadjusted 2:1 split is a -50% day that a mean-reversion strategy will happily trade.

**Detection.** Data quality scan for gaps, zero volume, and extreme single-period moves.

**Remediation.** Cross-check the flagged dates against a second data source before trusting any result that depends on them.

## Engine

### `ENGINE-NONDETERMINISTIC` - The strategy does not return the same thing twice

**Severity:** Critical

Called twice on identical data, the strategy produced different positions. Usually unseeded randomness - a shuffled split, a random initialisation, a hash-ordered set. It makes every number in the report unreproducible, and it specifically destroys the behavioural leakage test, which reads any difference between a clean run and a corrupted one as evidence of look-ahead. A strategy that disagrees with itself reports as leaky, and the researcher goes hunting for a bug that is really a missing seed.

**Detection.** The engine suite calls the strategy several times on the same array and compares the results element by element.

**Remediation.** Seed every random source inside the function - pass an explicit `random_state` or `np.random.default_rng(seed)` rather than relying on global state. If the non-determinism is intentional, the honest report is a distribution over seeds, not one run of it.

### `ENGINE-NO-POSITIONS` - The strategy is never invested

**Severity:** Critical

Every position the strategy returns is zero, so nothing in the report describes a strategy. Almost always an adapter fault: a column selected by the wrong name, a warm-up window longer than the sample, an index that failed to align. It matters because it is invisible in the wrong direction - every leakage test passes, every robustness test is quiet, and the report reads clean.

**Detection.** The engine suite checks the gross exposure of the returned book across the whole sample.

**Remediation.** Call the adapter directly and look at the book it returns. Check the column labels match the universe, and that the warm-up window leaves usable history.

### `ENGINE-DEGENERATE-SIGNAL` - The book barely changes, so there is little to test

**Severity:** High

A signal that does not move cannot be shown to depend on the future. That makes a clean behavioural leakage result uninformative rather than reassuring: the test had nothing to work with. It also means most of the robustness suite is describing a buy-and-hold position under another name.

**Detection.** The engine suite counts how many times the book actually changes and how much of the sample is spent flat.

**Remediation.** Confirm the rebalance is firing. If the strategy really is close to static, say so - and compare it against buy-and-hold, which is the benchmark it is actually competing with.

### `ENGINE-EXECUTION-FRAGILE` - The edge does not survive being traded late

**Severity:** High

Re-timing the same book by a single period destroys most of the Sharpe. An edge that requires filling at the instant the signal is computed is a property of the backtest's timing assumptions rather than of the market: no real book is filled that promptly, and the usual cause is that the signal is picking up a short-horizon reversal that has already reverted by the time anyone could trade it. This is independent of cost - a strategy can have an enormous break-even cost cushion and still fail here, because the two ask different questions, one about fees and one about time.

**Detection.** The engine suite shifts the position series forward by one, two, three, five and ten periods and recomputes the annualised Sharpe of each.

**Remediation.** Re-run the backtest with a realistic execution lag for the venue and report that as the headline result. If the edge only exists at zero delay, it is not an edge.

### `ENGINE-INCLUSION-TIMING` - Instruments enter or leave the universe mid-sample

**Severity:** Medium

An instrument added the day it lists, or dropped the day it stops trading, is a position taken in the knowledge that it would exist. It is the same family of error as survivorship, with one important difference: survivorship cannot be measured from a return series and has to be declared, while inclusion timing is visible in the data.

**Detection.** The engine suite records where each instrument's history begins and ends against the whole sample, before any common-calendar alignment - aligning first is what makes a ragged universe look clean.

**Remediation.** Either start the sample where the whole universe exists, or hold the instrument at zero weight until its inclusion date would have been known and say in the manifest which you did.

