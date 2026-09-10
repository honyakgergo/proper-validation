"""The bit of a real audit that is not the audit: printing it.

Both strategies here used to ship ~170 lines of driver each, and almost all of
it was the same seven steps of data loading and grid running. Those now live in
``qv.pipeline`` and are reachable as ``qv validate --manifest``, so what is left
is a console preamble: how the strategy did against its benchmark, which is
worth seeing before the verdict rather than only after it.

The preamble uses the *same basis as the report it is about to write* - Sharpe
and Sortino in excess of the risk-free rate, levels and drawdown on total
returns. A preamble that disagrees with its own report is worse than no
preamble.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qv.pipeline import audit_from_manifest
from qv.report.render import write_report
from qv.stats.performance import summarise_performance
from qv.types import Suite


def main(here: Path, description: str | None = None) -> int:
    """Audit the strategy whose manifest sits in ``here``, and report."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--offline", action="store_true", help="Refuse the network."
    )
    parser.add_argument(
        "--suite",
        default="full",
        choices=[s.value for s in Suite],
        help="Which questions to ask. 'statistical': is the edge real? "
        "'engine': is the backtest trustworthy? 'full': both.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = here / "research_manifest.yaml"
    run = audit_from_manifest(
        manifest_path, offline=args.offline, suite=args.suite
    )
    # A single-suite run writes beside the full one rather than over it, so the
    # three reports can sit next to each other and be compared.
    out = args.out or (here if args.suite == "full" else here / f"report_{args.suite}")
    manifest, report = run.manifest, run.report

    print(f"{manifest.name}\n")
    print(
        f"Loaded {len(run.prices)} sessions of {len(manifest.data.universe)} "
        f"instruments, {run.prices.index[0].date()} to {run.prices.index[-1].date()}"
    )
    for label, vintage in run.vintages.items():
        print(f"    {label} vintage {vintage}")

    print(f"\nChosen configuration: {manifest.chosen_parameters}")
    rf = run.risk_free_mean
    summaries = [summarise_performance(run.returns, manifest.periods_per_year, "strategy", rf)]
    if run.benchmark_returns is not None:
        summaries.append(
            summarise_performance(
                run.benchmark_returns,
                manifest.periods_per_year,
                manifest.data.benchmark or "benchmark",
                rf,
            )
        )
    for s in summaries:
        print(
            f"    {s.name:14s} annualised {s.annualised_return:+.2%}  "
            f"Sharpe {s.sharpe:.2f}  Sortino {s.sortino:.2f}  "
            f"vol {s.annualised_volatility:.1%}  maxDD {s.max_drawdown:.1%}"
        )

    if run.grid is not None:
        print(f"\nEvaluated all {run.grid.n_trials} configurations examined")
        print(f"    {run.grid.summary()}")

    html_path, _ = write_report(report, out, returns=run.returns)

    print(f"\n{'=' * 70}")
    print(f"{report.suite.label}")
    print(f"Verdict: {report.verdict.label.upper()}")
    print("=" * 70)
    for finding in report.findings_by_severity():
        print(f"  [{finding.severity.label:8s}] {finding.id}")
        print(f"             {finding.detail[:300]}")
    if report.not_tested:
        print(f"\nCould not test ({len(report.not_tested)}):")
        for gap in report.not_tested:
            print(f"  - {gap[:150]}")
    print(f"\nWrote {html_path.name}")
    return 0
