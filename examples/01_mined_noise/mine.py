"""Data-mine thousands of trading rules on real SPY, then audit the winner.

The flagship demonstration. Every rule here is a random conjunction of two
thresholded technical indicators - a family with no predictive content beyond
what chance supplies, and whose *median* member loses money - and the best of
them is then put through the full audit.

The point is not that the winner is bad. The point is that the winner looks
*good*: an attractive equity curve and a Sharpe most people would publish. The
audit recovers a true edge of zero, because the true edge is zero by
construction.

    python examples/01_mined_noise/mine.py

Data is fetched once and cached; `end_date` is pinned in research_manifest.yaml
so re-running next month reproduces the published numbers rather than whatever
the vendor returns that day. Pass --offline to refuse the network and fail
loudly if the cache cannot serve the request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qv.adapter import frame_adapter  # noqa: E402
from qv.audit import AuditInputs, run_audit  # noqa: E402
from qv.data.loaders import load_prices, to_returns  # noqa: E402
from qv.data.quality import check_prices  # noqa: E402
from qv.report.render import write_report  # noqa: E402

HERE = Path(__file__).parent


def load_manifest() -> dict:
    return yaml.safe_load((HERE / "research_manifest.yaml").read_text(encoding="utf-8"))


def indicator_bank(prices: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """A pool of ordinary technical indicators, each standardised to a z-score.

    Deliberately mundane - moving-average ratios, momentum, mean reversion,
    volatility and range position. Nothing here is exotic, which is the point:
    these are the ingredients of the strategies people actually publish.

    Every indicator is computed from a trailing window and shifted by one
    period before use, so no rule can see the return it is scored on. The
    demonstration is about selection bias alone, uncontaminated by leakage.
    """
    close = prices["close"]
    high, low = prices.get("high", close), prices.get("low", close)
    returns = close.pct_change()
    bank: list[tuple[str, np.ndarray]] = []

    def add(name: str, series: pd.Series) -> None:
        rolled = (series - series.rolling(250).mean()) / series.rolling(250).std()
        bank.append((name, rolled.to_numpy()))

    for window in (5, 10, 20, 50, 100, 200):
        add(f"ma_ratio_{window}", close / close.rolling(window).mean() - 1.0)
        add(f"momentum_{window}", close.pct_change(window))
        add(f"volatility_{window}", returns.rolling(window).std())
    for window in (5, 10, 20, 50):
        add(f"zscore_{window}", -(close - close.rolling(window).mean()) / close.rolling(window).std())
        span = high.rolling(window).max() - low.rolling(window).min()
        add(f"range_pos_{window}", (close - low.rolling(window).min()) / span.replace(0, np.nan))
        add(f"streak_{window}", np.sign(returns).rolling(window).sum())
    return bank


def rule_positions(
    prices: pd.DataFrame,
    first: int,
    second: int,
    cut_a: float,
    cut_b: float,
    flip_a: float,
    flip_b: float,
) -> pd.DataFrame:
    """One mined rule, as a re-runnable function of prices alone.

    The same arithmetic as the vectorised loop in `mine`, for a single rule.
    It exists so the winner can be handed to the audit as a Tier 2 callable:
    the mined-noise study is the cleanest possible negative control for the
    behavioural leakage test, because every indicator here is trailing and
    shifted by construction, so the true answer is known to be "no leak" while
    the true edge is known to be zero. A tool that confused the two would
    report this rule as leaky, and it is worth demonstrating that it does not.
    """
    bank = indicator_bank(prices)
    values = np.column_stack([v for _, v in bank])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    cond_a = flip_a * (values[:, first] - cut_a) > 0
    cond_b = flip_b * (values[:, second] - cut_b) > 0
    signal = np.where(cond_a & cond_b, 1.0, -1.0)

    # Traded from the next period, so the signal never sees its own return.
    position = np.zeros(len(prices))
    position[1:] = signal[:-1]
    return pd.DataFrame({"position": position}, index=prices.index)


def mine(
    prices: pd.DataFrame, n_rules: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate ``n_rules`` random conjunction rules and score every one.

    Each rule picks two indicators from the bank at random, picks a threshold
    and a direction for each, and goes long when both conditions agree and
    short otherwise. Conjunctions of two conditions rather than a single
    moving-average crossover, because crossovers on one price series are
    almost the same bet - measured inter-rule correlation above 0.5 - and a
    search over ten thousand near-identical rules is really a search over a few
    dozen. Diversity is what lets selection bias actually bite.

    Returns ``(trial_returns, positions, rules)``: the ``(T, N)`` matrix the
    audit needs to price selection honestly, the position series of each rule,
    and the parameters of each rule so the winner can be rebuilt as a callable.

    Rules are long/short rather than long-only. A long-only rule on an index
    that rose over the period inherits the drift, and the demonstration would
    be about market beta rather than about mining.
    """
    close = prices["close"]
    market = np.nan_to_num(close.pct_change().to_numpy(), nan=0.0)
    n = len(close)

    bank = indicator_bank(prices)
    values = np.column_stack([v for _, v in bank])
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(bank), n_rules)
    second = rng.integers(0, len(bank), n_rules)
    cut_a = rng.normal(0.0, 0.7, n_rules)
    cut_b = rng.normal(0.0, 0.7, n_rules)
    flip_a = rng.choice([-1.0, 1.0], n_rules)
    flip_b = rng.choice([-1.0, 1.0], n_rules)

    trials = np.zeros((n, n_rules))
    positions = np.zeros((n, n_rules))
    for i in range(n_rules):
        cond_a = flip_a[i] * (values[:, first[i]] - cut_a[i]) > 0
        cond_b = flip_b[i] * (values[:, second[i]] - cut_b[i]) > 0
        signal = np.where(cond_a & cond_b, 1.0, -1.0)
        # Traded from the next period, so the signal never sees its own return.
        position = np.zeros(n)
        position[1:] = signal[:-1]
        positions[:, i] = position
        trials[:, i] = position * market

    rules = [
        {
            "first": int(first[i]),
            "second": int(second[i]),
            "cut_a": float(cut_a[i]),
            "cut_b": float(cut_b[i]),
            "flip_a": float(flip_a[i]),
            "flip_b": float(flip_b[i]),
        }
        for i in range(n_rules)
    ]
    return trials, positions, rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Refuse the network.")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    manifest = load_manifest()
    data = manifest["data"]
    search = manifest["search"]

    print(f"Fetching {data['ticker']} {data['start_date']}..{data['end_date']}")
    cached = load_prices(
        data["ticker"], data["start_date"], data["end_date"], offline=args.offline
    )
    prices = cached.frame
    print(f"  {len(prices)} rows, {'cache' if cached.from_cache else 'network'}, "
          f"vintage {cached.vintage}")

    quality = check_prices(prices)
    print(f"  data quality: {'clean' if quality.clean else 'issues found'}")
    for finding in quality.to_findings():
        print(f"    [{finding.severity.label}] {finding.detail}")

    n_rules = search["n_rules"]
    print(f"\nMining {n_rules} random conjunction rules...")
    trials, positions, rules = mine(prices, search["n_rules"], search["seed"])

    sds = trials.std(axis=0, ddof=1)
    sharpes = np.where(sds > 0, trials.mean(axis=0) / sds, -np.inf)
    best = int(np.argmax(sharpes))
    annualised = sharpes[best] * np.sqrt(252)
    print(f"  best rule #{best}: annualised Sharpe {annualised:.2f}")
    print(f"  median rule:      annualised Sharpe {np.median(sharpes) * np.sqrt(252):.2f}")

    market = to_returns(prices).to_numpy()

    # -- Tier 2: the winning rule, re-runnable on data the audit controls ---
    # Labels only, never the price frame: closing over `prices` would stop the
    # corruption ever reaching the rule, and the leakage test would pass
    # without having tested anything.
    strategy = frame_adapter(
        rule_positions, prices.index, prices.columns, **rules[best]
    )

    report = run_audit(
        AuditInputs(
            returns=trials[:, best],
            positions=positions[:, best],
            strategy=strategy,
            strategy_data=prices.to_numpy(),
            # Engine analysis. The rule trades one instrument, so its "book" is
            # a single column and its universe is trivially complete - which is
            # the point of including it here: the engine suite should come back
            # clean while the statistical suite demolishes the result, and the
            # two conclusions should not contaminate each other.
            asset_return_frame=np.concatenate([[0.0], market])[:, None],
            # Labelled with the ticker rather than "close", so the coverage
            # chart names the instrument it is describing.
            raw_prices=prices[["close"]].rename(columns={"close": data["ticker"]}),
            asset_returns=np.concatenate([[0.0], market]),
            asset_class=data["asset_class"],
            dates=prices.index,
            # One instrument, fixed in advance: there is no membership list to
            # have assembled with hindsight. Declared rather than assumed,
            # because the audit will otherwise - correctly - list the question
            # as unanswered.
            universe_point_in_time=True,
            universe_note=f"A single instrument, {data['ticker']}, fixed before the search.",
            benchmark_returns=np.concatenate([[0.0], market]),
            benchmark_name=data["ticker"],
            trial_returns=trials,
            n_trials=search["n_rules"],
            periods_per_year=252,
            name=manifest["name"],
            seed=search["seed"],
        )
    )

    # This driver builds its inputs directly rather than going through
    # `audit_from_manifest`, so the vintage has to be handed over by hand. It
    # matters more here than anywhere: the whole demonstration rests on real
    # SPY prices over a pinned window, and a reader who refetches a revised
    # history is not reproducing this number.
    report.provenance["data_vintages"] = {"prices": cached.vintage}

    html_path, json_path = write_report(report, args.out, returns=trials[:, best])
    print(f"\nVerdict: {report.verdict.label.upper()}")
    for finding in report.findings_by_severity():
        print(f"  [{finding.severity.label:8s}] {finding.id}")
    print(f"\nWrote {html_path.name} and {json_path.name} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
