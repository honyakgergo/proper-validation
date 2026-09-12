"""The findings catalog: every defect this tool knows how to name.

One entry per genuinely distinct defect. Twenty real entries beat forty where
half are rephrasings - a reviewer who reads the catalog notices padding, and
padding is the exact sin this package exists to detect.

This module is the single source of truth. The markdown reference shipped with
the Claude Code skill is generated from it (``qv explain --all --markdown``),
so the documentation cannot drift from what the scanner actually reports.

Each entry carries a **remediation**, because a validator that only says "this
is wrong" teaches nothing. The tool deliberately has no ``qv fix``: it names
the problem and says what a correct approach looks like, and the researcher
makes the change.
"""

from __future__ import annotations

from dataclasses import dataclass

from qv.types import Finding, Severity

__all__ = ["CatalogEntry", "CATALOG", "get_entry", "make_finding", "catalog_markdown"]


@dataclass(frozen=True)
class CatalogEntry:
    """A defect type: what it is, how it is found, and what to do about it."""

    id: str
    title: str
    severity: Severity
    category: str
    explanation: str
    detection: str
    remediation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.name.lower(),
            "category": self.category,
            "explanation": self.explanation,
            "detection": self.detection,
            "remediation": self.remediation,
        }


_ENTRIES: tuple[CatalogEntry, ...] = (
    # ---- Look-ahead and leakage -------------------------------------------
    CatalogEntry(
        id="LEAK-NEGATIVE-SHIFT",
        title="Signal reads future data via a negative shift",
        severity=Severity.CRITICAL,
        category="leakage",
        explanation=(
            "A negative shift moves future values backwards in time, so a signal built "
            "from it knows the answer before it is knowable. This is the single most "
            "common way a backtest becomes fiction, and it usually produces a Sharpe so "
            "high that it should have prompted suspicion on its own."
        ),
        detection="AST scan for .shift(-n) with a negative literal argument.",
        remediation=(
            "Negative shifts are legitimate for constructing a forward-looking *target* "
            "in supervised learning, but never for a feature. Confirm the shifted series "
            "feeds only the label, and that features use shift(+n) or a rolling window "
            "closed at t."
        ),
    ),
    CatalogEntry(
        id="LEAK-BACKWARD-FILL",
        title="Missing values filled from the future",
        severity=Severity.HIGH,
        category="leakage",
        explanation=(
            "Backward fill propagates a later observation into an earlier gap. Every "
            "filled cell then contains information that did not exist at that timestamp. "
            "It is easy to miss because the call looks like ordinary data hygiene."
        ),
        detection="AST scan for bfill(), backfill(), and fillna(method='bfill').",
        remediation=(
            "Use forward fill, or leave the gap and let the strategy abstain. If a series "
            "genuinely has no value until later, the honest handling is to exclude the "
            "period rather than invent one."
        ),
    ),
    CatalogEntry(
        id="LEAK-CENTERED-WINDOW",
        title="Centred rolling window spans future observations",
        severity=Severity.HIGH,
        category="leakage",
        explanation=(
            "A centred window of width w at time t averages data from t - w/2 to "
            "t + w/2, so half its input is unavailable at t. Centring is right for "
            "descriptive smoothing in exploratory plots and wrong for anything a signal "
            "touches."
        ),
        detection="AST scan for rolling(..., center=True).",
        remediation="Set center=False, which is the default, and widen the window if needed.",
    ),
    CatalogEntry(
        id="LEAK-FULL-SAMPLE-SCALER",
        title="Scaler or transformer fitted on the full sample",
        severity=Severity.HIGH,
        category="leakage",
        explanation=(
            "Fitting a scaler on all the data leaks the mean and variance of the test "
            "period into the training period. The effect is subtle - it rarely produces "
            "an absurd Sharpe, just a persistently optimistic one - which is what makes "
            "it so common and so durable."
        ),
        detection="AST scan for fit() or fit_transform() outside a fold loop.",
        remediation=(
            "Fit the scaler inside each training fold and apply it to the corresponding "
            "validation fold. A scikit-learn Pipeline inside TimeSeriesSplit does this "
            "correctly by construction."
        ),
    ),
    CatalogEntry(
        id="LEAK-SHUFFLED-SPLIT",
        title="Time series split randomly rather than chronologically",
        severity=Severity.CRITICAL,
        category="leakage",
        explanation=(
            "A shuffled split puts observations from after the test period into the "
            "training set. With autocorrelated data, neighbouring observations are near "
            "duplicates, so the model is effectively tested on data it has seen."
        ),
        detection=(
            "AST scan for train_test_split without shuffle=False, and for KFold or "
            "cross_val_score without a time-aware splitter."
        ),
        remediation=(
            "Use TimeSeriesSplit, or split by date with an explicit embargo gap between "
            "train and test to break the autocorrelation."
        ),
    ),
    CatalogEntry(
        id="LEAK-FULL-SAMPLE-STATISTIC",
        title="Full-sample statistic used where a trailing one belongs",
        severity=Severity.MEDIUM,
        category="leakage",
        explanation=(
            "A threshold set from the mean, standard deviation or a quantile of the "
            "whole series encodes knowledge of the whole series. Legitimate when the "
            "value is only reported, a leak when it feeds a signal."
        ),
        detection=(
            "AST scan for mean/std/median/quantile/max/min taken over a full column "
            "without a preceding rolling or expanding window."
        ),
        remediation=(
            "Replace with an expanding or rolling equivalent so the threshold at time t "
            "uses only data up to t. If the value is purely descriptive, say so in the "
            "manifest and this finding can be dismissed."
        ),
    ),
    CatalogEntry(
        id="LEAK-BEHAVIOURAL",
        title="Signal changes when only future data is corrupted",
        severity=Severity.CRITICAL,
        category="leakage",
        explanation=(
            "The behavioural test: corrupt the data strictly after time t, re-run, and "
            "the signal at t must be unchanged. If it moves, the strategy reads the "
            "future somewhere - regardless of what the source code appears to say. This "
            "catches leaks no static scan can see, including ones inside third-party "
            "library calls."
        ),
        detection="Re-run the strategy with post-t data replaced and compare signals at t.",
        remediation=(
            "Bisect: corrupt progressively narrower future windows to localise which "
            "horizon the strategy is reaching into, then inspect the code that touches it."
        ),
    ),
    # ---- Selection bias ---------------------------------------------------
    CatalogEntry(
        id="SELECT-UNDECLARED-TRIALS",
        title="More configurations were tried than were reported",
        severity=Severity.HIGH,
        category="selection",
        explanation=(
            "The number of trials is the key input to every multiple-testing correction "
            "and the most under-reported number in backtesting. Notebook forensics "
            "recovers a lower bound; the true count is always at least this and usually "
            "much more, because abandoned ideas leave no trace."
        ),
        detection=(
            "Notebook archaeology: out-of-order and repeated execution counts, parameter "
            "literals across cells, explicit grid searches, filename lineage."
        ),
        remediation=(
            "Record every configuration examined, including abandoned ones, and pass the "
            "count as n_trials. When in doubt, overstate it - the correction is "
            "logarithmic in the trial count, so honesty is cheap."
        ),
    ),
    CatalogEntry(
        id="SELECT-DEFLATED-SHARPE-FAILS",
        title="Sharpe does not survive the selection-adjusted bar",
        severity=Severity.CRITICAL,
        category="selection",
        explanation=(
            "Once the number of trials is priced in, the observed Sharpe is no better "
            "than the best a search of this size would produce from noise. The strategy "
            "may still be real, but this backtest is not evidence that it is."
        ),
        detection="Deflated Sharpe Ratio below 0.95.",
        remediation=(
            "Reduce the trial count by fixing parameters on theoretical grounds, extend "
            "the sample, or test the strategy on genuinely fresh data. Nothing else "
            "recovers a failed deflation."
        ),
    ),
    CatalogEntry(
        id="SELECT-HIGH-PBO",
        title="The selection procedure does not generalise",
        severity=Severity.HIGH,
        category="selection",
        explanation=(
            "Across combinatorial in-sample and out-of-sample splits, the configuration "
            "that looks best in-sample lands in the bottom half out-of-sample more often "
            "than not. Above 0.5 the search is worse than useless: it actively selects "
            "strategies that underperform."
        ),
        detection="PBO above 0.5 from combinatorially symmetric cross-validation.",
        remediation=(
            "Simplify the strategy until the parameter space is small enough to defend, "
            "or select on economic reasoning rather than on backtest performance."
        ),
    ),
    CatalogEntry(
        id="SELECT-PARAMETER-SPIKE",
        title="Chosen parameters are a spike, not a plateau",
        severity=Severity.HIGH,
        category="selection",
        explanation=(
            "Neighbouring parameter values perform far worse than the chosen point. A "
            "genuine effect degrades gracefully as parameters move; an isolated peak "
            "means the parameter was fitted to noise and will not survive contact with "
            "new data."
        ),
        detection="Mean neighbourhood score below half the chosen point.",
        remediation=(
            "Choose the centre of the widest plateau rather than the global maximum, and "
            "report performance across the plateau rather than at its best point."
        ),
    ),
    CatalogEntry(
        id="SELECT-INSUFFICIENT-LENGTH",
        title="Backtest is too short to support its claim given the trial count",
        severity=Severity.HIGH,
        category="selection",
        explanation=(
            "Minimum Backtest Length gives the years of data needed before an observed "
            "Sharpe means anything at this number of trials. Below it, the result is "
            "within the range that a search of this size produces from noise."
        ),
        detection="Sample length below the minimum backtest length for the trial count.",
        remediation="Extend the sample, or reduce the number of configurations examined.",
    ),
    # ---- Statistical treatment -------------------------------------------
    CatalogEntry(
        id="STAT-NAIVE-ANNUALISATION",
        title="Sharpe annualised with sqrt(q) despite serial correlation",
        severity=Severity.MEDIUM,
        category="statistics",
        explanation=(
            "Multiplying by the square root of the number of periods assumes returns are "
            "serially independent. Positive autocorrelation - normal for slow-moving or "
            "overlapping signals - makes this materially too generous."
        ),
        detection="Lo autocorrelation-adjusted factor more than 10% below sqrt(q).",
        remediation=(
            "Report the autocorrelation-adjusted Sharpe. If the gap is large, investigate "
            "whether the returns overlap - overlapping windows are the usual cause."
        ),
    ),
    CatalogEntry(
        id="STAT-NO-CONFIDENCE-INTERVAL",
        title="Point estimate reported with no interval",
        severity=Severity.MEDIUM,
        category="statistics",
        explanation=(
            "A Sharpe of 1.3 over 40 observations carries a standard error near 0.35 "
            "before any multiple-testing adjustment. Reported bare, it invites a "
            "precision the data cannot support."
        ),
        detection="Reported metrics without an accompanying interval.",
        remediation=(
            "Report a stationary-bootstrap confidence interval alongside every headline "
            "figure."
        ),
    ),
    CatalogEntry(
        id="STAT-SMALL-SAMPLE",
        title="Too few observations for the methods applied",
        severity=Severity.HIGH,
        category="statistics",
        explanation=(
            "Below roughly 30 observations the asymptotic results behind the Sharpe "
            "standard error, the Deflated Sharpe and the bootstrap all degrade. Quarterly "
            "strategies hit this routinely: a decade of quarterly returns is 40 points."
        ),
        detection="Fewer than 30 usable observations.",
        remediation=(
            "Increase the observation frequency if the strategy allows it, extend the "
            "sample, or state plainly that the result is indicative rather than "
            "inferential."
        ),
    ),
    CatalogEntry(
        id="RISK-DRAWDOWN-PATH-DEPENDENT",
        title="Worst drawdown came from the ordering, not the distribution",
        severity=Severity.MEDIUM,
        category="statistics",
        explanation=(
            "The realised maximum drawdown is deeper than 95% of block-resampled paths "
            "drawn from the same returns. The loss was produced by the sequence in which "
            "returns arrived rather than by their distribution, which means resampling "
            "cannot reproduce it and a shorter or reordered sample would never have "
            "revealed it. Reported as a single number, such a drawdown reads as a property "
            "of the strategy when it is a property of one path."
        ),
        detection=(
            "Realised maximum drawdown below the 5th percentile of a stationary-bootstrap "
            "distribution of the same statistic."
        ),
        remediation=(
            "Report the resampled interval alongside the realised figure, and size the "
            "strategy for the deeper end of it. Note that block resampling already "
            "understates severity by breaking long declines, so the true risk is worse "
            "than even the resampled range suggests."
        ),
    ),
    CatalogEntry(
        id="MC-IID-RESAMPLE",
        title="Monte Carlo that resamples returns independently",
        severity=Severity.MEDIUM,
        category="statistics",
        explanation=(
            "Shuffling trade order or resampling returns independently destroys serial "
            "dependence and volatility clustering, so the drawdown quantiles it produces "
            "are systematically optimistic. Reordering trades also tests nothing about "
            "whether the edge was real."
        ),
        detection=(
            "AST scan for np.random.permutation, np.random.choice or sample(frac=1) "
            "applied to a returns or trade series."
        ),
        remediation=(
            "Use a stationary bootstrap with an automatically selected block length, "
            "which preserves short-range dependence while still resampling."
        ),
    ),
    CatalogEntry(
        id="MC-FORWARD-PROJECTION",
        title="Simulated future paths presented as risk analysis",
        severity=Severity.MEDIUM,
        category="statistics",
        explanation=(
            "Projecting equity forward from a fitted return distribution makes a claim "
            "about the future on the strength of an assumed data-generating process. The "
            "fan chart looks rigorous and encodes nothing beyond the assumption."
        ),
        detection="AST scan for random draws used to build forward equity paths.",
        remediation=(
            "Report realised drawdown and its bootstrap interval instead. Historical "
            "evidence about what did happen is worth more than simulated evidence about "
            "what an assumed model says might."
        ),
    ),
    # ---- Costs and capacity ----------------------------------------------
    CatalogEntry(
        id="COST-BELOW-REALISTIC",
        title="Break-even cost is below a realistic trading cost",
        severity=Severity.CRITICAL,
        category="costs",
        explanation=(
            "The cost at which the strategy stops making money is lower than what "
            "trading it would actually cost. The gross edge may be real; the net edge is "
            "not there."
        ),
        detection="Break-even cost in basis points below the asset class estimate.",
        remediation=(
            "Reduce turnover, trade more liquid instruments, or accept that the effect "
            "is not tradable at this frequency."
        ),
    ),
    CatalogEntry(
        id="COST-ASSUMED-NOT-DERIVED",
        title="Costs asserted rather than computed from turnover",
        severity=Severity.MEDIUM,
        category="costs",
        explanation=(
            "A round-number cost applied per period, rather than per unit of traded "
            "notional, breaks the link between how often the strategy trades and what it "
            "pays - which is the entire mechanism by which costs kill strategies."
        ),
        detection="Manifest declares a cost assumption with no position series supplied.",
        remediation=(
            "Supply the position series so turnover can be measured, and let the cost "
            "follow from it."
        ),
    ),
    # ---- Attribution and construction ------------------------------------
    CatalogEntry(
        id="ATTR-LEVERED-BETA",
        title="Returns are explained by factor exposure",
        severity=Severity.CRITICAL,
        category="attribution",
        explanation=(
            "Alpha is not significant once factors are accounted for, and the factors "
            "explain most of the variance. The strategy is a repackaging of exposures "
            "available at a few basis points."
        ),
        detection=(
            "Factor regression alpha not significant with high R-squared, or a "
            "volatility-matched benchmark matching the strategy Sharpe."
        ),
        remediation=(
            "Hedge the factor exposures and re-test the residual, which is the only part "
            "that could be alpha."
        ),
    ),
    CatalogEntry(
        id="ROBUST-REGIME-DEPENDENT",
        title="Edge exists in only one market regime",
        severity=Severity.HIGH,
        category="robustness",
        explanation=(
            "Performance is positive in one volatility regime and negative in the other. "
            "The strategy is a bet on that regime persisting, which is a different and "
            "much less attractive proposition than the headline suggests."
        ),
        detection="Sharpe signs disagree across the volatility split.",
        remediation=(
            "State the regime dependence explicitly and size the strategy for the "
            "possibility that the regime ends. Consider a regime filter, priced for the "
            "extra trials it costs."
        ),
    ),
    CatalogEntry(
        id="ROBUST-START-DATE-SENSITIVE",
        title="Conclusion depends on the start date",
        severity=Severity.HIGH,
        category="robustness",
        explanation=(
            "Moving the start of the sample forward flips the sign of the result. A "
            "finding that survives only from one particular starting point is a property "
            "of that window, not of the strategy."
        ),
        detection="Rolling-origin Sharpes disagree on sign.",
        remediation=(
            "Report the full range of start dates rather than one, and investigate what "
            "the early period contains that the later ones do not."
        ),
    ),
    CatalogEntry(
        id="ROBUST-TOP-DAY-DEPENDENT",
        title="Profit concentrated beyond what the Sharpe explains",
        severity=Severity.HIGH,
        category="robustness",
        explanation=(
            "The best few periods supply far more of the profit than a normal series of "
            "this Sharpe would produce. The strategy is closer to a lottery ticket than "
            "its Sharpe suggests, and the next such period may not arrive."
        ),
        detection="Observed profit share more than 1.5x the normal-distribution baseline.",
        remediation=(
            "Examine the outlier periods individually. They are often a data error, a "
            "corporate action, or a single position that should have been capped."
        ),
    ),
    CatalogEntry(
        id="ROBUST-NO-TIMING-SKILL",
        title="Signal timing does not beat random timing at the same exposure",
        severity=Severity.HIGH,
        category="robustness",
        explanation=(
            "Reordering the position series at random, holding average exposure and "
            "turnover fixed, produces the same performance. Whatever the strategy earns, "
            "it earns from being in the market, not from choosing when."
        ),
        detection="Matched-exposure randomisation p-value above 0.20.",
        remediation=(
            "Compare against a constant-exposure benchmark. If the signal adds nothing, "
            "the simpler position is the honest one."
        ),
    ),
    # ---- Data -------------------------------------------------------------
    CatalogEntry(
        id="DATA-SURVIVORSHIP",
        title="Universe built from current index membership",
        severity=Severity.HIGH,
        category="data",
        explanation=(
            "A universe of instruments that exist today excludes everything that failed, "
            "delisted or merged. The backtest then trades a portfolio nobody could have "
            "selected at the time, and the bias is upward by construction."
        ),
        detection=(
            "Declared by the manifest, or measured against a supplied point-in-time "
            "membership list: names that were index members during the window and are "
            "absent from the traded universe."
        ),
        remediation=(
            "Use a point-in-time constituent list. Where none is available, say so and "
            "treat the result as an upper bound. Supplying one to "
            "`data.membership_frame` replaces the declaration with a count."
        ),
    ),
    CatalogEntry(
        id="DATA-MEMBERSHIP-NOT-POINT-IN-TIME",
        title="The membership list supplied is a snapshot, not a history",
        severity=Severity.HIGH,
        category="data",
        explanation=(
            "A table of today's index members with the date each was added reaches back "
            "decades and looks like point-in-time data, but it contains only the "
            "companies that are still members - it is the survivorship bias itself in a "
            "history-shaped schema. Measured against it, every universe looks complete."
        ),
        detection=(
            "No membership spell in the supplied list ever ends. A real reconstruction "
            "contains departures; a list of current members cannot."
        ),
        remediation=(
            "Use a list that records removals as well as additions. A reconstruction "
            "from index change announcements has them; the current-membership table on "
            "an encyclopedia page does not."
        ),
    ),
    CatalogEntry(
        id="DATA-DECLARATION-CONTRADICTED",
        title="A declared fact is contradicted by the data supplied with it",
        severity=Severity.HIGH,
        category="data",
        explanation=(
            "The manifest asserts something the data alongside it disproves. That is "
            "worse than an open question, because the reader has no way to tell which "
            "of the remaining declarations are still load-bearing - and the whole "
            "selection-bias correction rests on one of them, the trial count, which the "
            "researcher alone can supply."
        ),
        detection=(
            "A declared field is compared against the evidence: the universe declared "
            "point-in-time while instruments demonstrably enter part-way through."
        ),
        remediation=(
            "Correct the manifest and re-run. Then state how the other declared fields "
            "were arrived at, the trial count first."
        ),
    ),
    CatalogEntry(
        id="DATA-QUALITY-GAPS",
        title="Price series has gaps or adjustment artifacts",
        severity=Severity.MEDIUM,
        category="data",
        explanation=(
            "Missing days, zero volumes and unadjusted split jumps produce spurious "
            "returns. A single unadjusted 2:1 split is a -50% day that a mean-reversion "
            "strategy will happily trade."
        ),
        detection="Data quality scan for gaps, zero volume, and extreme single-period moves.",
        remediation=(
            "Cross-check the flagged dates against a second data source before trusting "
            "any result that depends on them."
        ),
    ),
    # ---- Engine analysis: is the implementation trustworthy? --------------
    CatalogEntry(
        id="ENGINE-NONDETERMINISTIC",
        title="The strategy does not return the same thing twice",
        severity=Severity.CRITICAL,
        category="engine",
        explanation=(
            "Called twice on identical data, the strategy produced different "
            "positions. Usually unseeded randomness - a shuffled split, a random "
            "initialisation, a hash-ordered set. It makes every number in the report "
            "unreproducible, and it specifically destroys the behavioural leakage "
            "test, which reads any difference between a clean run and a corrupted "
            "one as evidence of look-ahead. A strategy that disagrees with itself "
            "reports as leaky, and the researcher goes hunting for a bug that is "
            "really a missing seed."
        ),
        detection=(
            "The engine suite calls the strategy several times on the same array and "
            "compares the results element by element."
        ),
        remediation=(
            "Seed every random source inside the function - pass an explicit "
            "`random_state` or `np.random.default_rng(seed)` rather than relying on "
            "global state. If the non-determinism is intentional, the honest report "
            "is a distribution over seeds, not one run of it."
        ),
    ),
    CatalogEntry(
        id="ENGINE-NO-POSITIONS",
        title="The strategy is never invested",
        severity=Severity.CRITICAL,
        category="engine",
        explanation=(
            "Every position the strategy returns is zero, so nothing in the report "
            "describes a strategy. Almost always an adapter fault: a column selected "
            "by the wrong name, a warm-up window longer than the sample, an index "
            "that failed to align. It matters because it is invisible in the wrong "
            "direction - every leakage test passes, every robustness test is quiet, "
            "and the report reads clean."
        ),
        detection=(
            "The engine suite checks the gross exposure of the returned book across "
            "the whole sample."
        ),
        remediation=(
            "Call the adapter directly and look at the book it returns. Check the "
            "column labels match the universe, and that the warm-up window leaves "
            "usable history."
        ),
    ),
    CatalogEntry(
        id="ENGINE-DEGENERATE-SIGNAL",
        title="The book barely changes, so there is little to test",
        severity=Severity.HIGH,
        category="engine",
        explanation=(
            "A signal that does not move cannot be shown to depend on the future. "
            "That makes a clean behavioural leakage result uninformative rather than "
            "reassuring: the test had nothing to work with. It also means most of "
            "the robustness suite is describing a buy-and-hold position under "
            "another name."
        ),
        detection=(
            "The engine suite counts how many times the book actually changes and "
            "how much of the sample is spent flat."
        ),
        remediation=(
            "Confirm the rebalance is firing. If the strategy really is close to "
            "static, say so - and compare it against buy-and-hold, which is the "
            "benchmark it is actually competing with."
        ),
    ),
    CatalogEntry(
        id="ENGINE-EXECUTION-FRAGILE",
        title="The edge does not survive being traded late",
        severity=Severity.HIGH,
        category="engine",
        explanation=(
            "Re-timing the same book by a single period destroys most of the Sharpe. "
            "An edge that requires filling at the instant the signal is computed is a "
            "property of the backtest's timing assumptions rather than of the market: "
            "no real book is filled that promptly, and the usual cause is that the "
            "signal is picking up a short-horizon reversal that has already reverted "
            "by the time anyone could trade it. This is independent of cost - a "
            "strategy can have an enormous break-even cost cushion and still fail "
            "here, because the two ask different questions, one about fees and one "
            "about time."
        ),
        detection=(
            "The engine suite shifts the position series forward by one, two, three, "
            "five and ten periods and recomputes the annualised Sharpe of each."
        ),
        remediation=(
            "Re-run the backtest with a realistic execution lag for the venue and "
            "report that as the headline result. If the edge only exists at zero "
            "delay, it is not an edge."
        ),
    ),
    CatalogEntry(
        id="ENGINE-INCLUSION-TIMING",
        title="Instruments enter or leave the universe mid-sample",
        severity=Severity.MEDIUM,
        category="engine",
        explanation=(
            "An instrument added the day it lists, or dropped the day it stops "
            "trading, is a position taken in the knowledge that it would exist. It is "
            "the same family of error as survivorship, with one important difference: "
            "survivorship has to be declared, or counted against a point-in-time "
            "membership list, because a universe of survivors looks exactly like a "
            "universe. Inclusion timing needs neither - it is visible in the price "
            "data itself."
        ),
        detection=(
            "The engine suite records where each instrument's history begins and ends "
            "against the whole sample, before any common-calendar alignment - aligning "
            "first is what makes a ragged universe look clean."
        ),
        remediation=(
            "Either start the sample where the whole universe exists, or hold the "
            "instrument at zero weight until its inclusion date would have been known "
            "and say in the manifest which you did."
        ),
    ),
)

CATALOG: dict[str, CatalogEntry] = {entry.id: entry for entry in _ENTRIES}


def get_entry(finding_id: str) -> CatalogEntry:
    """Look up a catalog entry, with a helpful error for a near miss."""
    if finding_id in CATALOG:
        return CATALOG[finding_id]
    close = [k for k in CATALOG if finding_id.upper() in k or k.startswith(finding_id.upper())]
    hint = f"; did you mean {close[0]}?" if close else ""
    raise KeyError(f"{finding_id!r} is not in the findings catalog{hint}")


def make_finding(
    finding_id: str,
    detail: str,
    evidence: dict | None = None,
    severity: Severity | None = None,
) -> Finding:
    """Build a :class:`~qv.types.Finding` from a catalog entry.

    ``severity`` overrides the catalog default for cases where the same defect
    is worse or milder in context - a failed deflation at 0.2 is not the same
    as one at 0.94.
    """
    entry = get_entry(finding_id)
    return Finding(
        id=entry.id,
        title=entry.title,
        severity=entry.severity if severity is None else severity,
        detail=detail,
        remediation=entry.remediation,
        evidence=evidence or {},
    )


def catalog_markdown() -> str:
    """Render the whole catalog as markdown, grouped by category.

    This is what generates the reference shipped with the skill, so the
    documentation cannot drift from the code.
    """
    lines = [
        "# Findings catalog",
        "",
        f"{len(CATALOG)} distinct defects. Generated from `qv/findings.py` - edit there, not here.",
        "",
    ]
    for category in dict.fromkeys(entry.category for entry in _ENTRIES):
        lines += [f"## {category.capitalize()}", ""]
        for entry in _ENTRIES:
            if entry.category != category:
                continue
            lines += [
                f"### `{entry.id}` - {entry.title}",
                "",
                f"**Severity:** {entry.severity.label}",
                "",
                entry.explanation,
                "",
                f"**Detection.** {entry.detection}",
                "",
                f"**Remediation.** {entry.remediation}",
                "",
            ]
    return "\n".join(lines)
