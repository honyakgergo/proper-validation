"""Measure the tool against its own synthetic ground truth.

Runs the full audit over ``R`` independent replications of every labelled
strategy and reports, with confidence intervals, how often each label is
correctly falsified - and how often ``genuine_weak``, the one label with a real
edge, is wrongly condemned.

Detection rates come with intervals because a bare point estimate invites the
question "out of how many?". A Wilson interval is used rather than the normal
approximation, since rates near 0 and 1 are exactly where the naive interval
misbehaves and exactly where these results live.

    python benchmarks/run_roc.py --replications 20 --out benchmarks/roc_results.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.generate import LABELS, generate  # noqa: E402
from qv.audit import AuditInputs, run_audit  # noqa: E402
from qv.types import Severity, Verdict  # noqa: E402

__all__ = ["LabelOutcome", "wilson_interval", "run_label", "run_all", "format_markdown"]


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred to the normal approximation because these rates sit near 0 and 1,
    where the naive interval runs outside [0, 1] and understates uncertainty.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class LabelOutcome:
    """How the audit performed across replications of one label."""

    label: str
    has_edge: bool
    description: str
    replications: int
    flagged: int
    verdicts: dict[str, int]
    expected_findings: tuple[str, ...]
    expected_hits: dict[str, int]
    mean_findings: float

    @property
    def flag_rate(self) -> float:
        return self.flagged / self.replications if self.replications else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.flagged, self.replications)

    @property
    def correct_rate(self) -> float:
        """Fraction of replications where the tool reached the right conclusion.

        For a no-edge label, flagging it is correct. For ``genuine_weak``, *not*
        flagging it is correct - and that asymmetry is the whole point of
        including it.
        """
        return 1.0 - self.flag_rate if self.has_edge else self.flag_rate

    def to_dict(self) -> dict:
        lo, hi = self.interval
        return {
            "label": self.label,
            "has_edge": self.has_edge,
            "description": self.description,
            "replications": self.replications,
            "flag_rate": self.flag_rate,
            "flag_rate_ci": [lo, hi],
            "correct_rate": self.correct_rate,
            "verdicts": self.verdicts,
            "expected_findings": list(self.expected_findings),
            "expected_hits": self.expected_hits,
            "mean_findings": self.mean_findings,
        }


def run_label(label: str, replications: int, n: int, n_boot: int) -> LabelOutcome:
    """Audit ``replications`` independent draws of one labelled strategy."""
    flagged = 0
    verdicts: dict[str, int] = {}
    expected_hits: dict[str, int] = {}
    total_findings = 0
    spec = generate(label, seed=0, n=n)

    for rep in range(replications):
        strat = generate(label, seed=rep, n=n)
        report = run_audit(
            AuditInputs(
                returns=strat.returns,
                periods_per_year=252,
                name=f"{label}#{rep}",
                positions=strat.positions,
                asset_returns=strat.asset_returns,
                asset_class=strat.asset_class or "us_large_cap_equity",
                benchmark_returns=(
                    strat.asset_returns if strat.label == "levered_beta" else None
                ),
                n_trials=strat.n_trials,
                trial_returns=strat.trial_returns,
                strategy=strat.strategy,
                strategy_data=strat.strategy_data,
                n_boot=n_boot,
                seed=rep,
            )
        )
        verdicts[report.verdict.value] = verdicts.get(report.verdict.value, 0) + 1
        total_findings += len(report.findings)
        if report.worst_severity >= Severity.HIGH:
            flagged += 1
        ids = {f.id for f in report.findings}
        for expected in strat.expected_findings:
            expected_hits[expected] = expected_hits.get(expected, 0) + int(expected in ids)

    return LabelOutcome(
        label=label,
        has_edge=spec.has_edge,
        description=spec.description,
        replications=replications,
        flagged=flagged,
        verdicts=verdicts,
        expected_findings=spec.expected_findings,
        expected_hits=expected_hits,
        mean_findings=total_findings / replications if replications else float("nan"),
    )


def run_all(replications: int = 20, n: int = 2000, n_boot: int = 400) -> list[LabelOutcome]:
    outcomes = []
    for label in LABELS:
        start = time.time()
        outcome = run_label(label, replications, n, n_boot)
        outcomes.append(outcome)
        lo, hi = outcome.interval
        print(
            f"  {label:16s} flagged {outcome.flag_rate:6.1%} "
            f"[{lo:.2f}, {hi:.2f}]  ({time.time() - start:.1f}s)",
            flush=True,
        )
    return outcomes


def format_markdown(outcomes: list[LabelOutcome], replications: int) -> str:
    """The committed results table."""
    lines = [
        "# Measured detection rates",
        "",
        f"{replications} independent replications per label, generated by "
        "`benchmarks/generate.py` and audited end to end. Reproduce with:",
        "",
        "```",
        f"python benchmarks/run_roc.py --replications {replications}",
        "```",
        "",
        "A label is *flagged* when the audit returns at least one finding of severity",
        "HIGH or above. For the eight no-edge labels a high flag rate is correct. For",
        "`genuine_weak` - which has a real, planted edge - a **low** flag rate is",
        "correct, and that row is the one that shows the tool is not simply a pessimism",
        "generator. Intervals are Wilson 95%.",
        "",
        "| Label | True edge | Flagged | 95% interval | Correct | Findings/run |",
        "|---|---|---:|---|---:|---:|",
    ]
    for o in outcomes:
        lo, hi = o.interval
        truth = "**real**" if o.has_edge else "none"
        lines.append(
            f"| `{o.label}` | {truth} | {o.flag_rate:.0%} | [{lo:.2f}, {hi:.2f}] | "
            f"{o.correct_rate:.0%} | {o.mean_findings:.1f} |"
        )

    no_edge = [o for o in outcomes if not o.has_edge]
    with_edge = [o for o in outcomes if o.has_edge]
    lines += [
        "",
        f"**Detection across the {len(no_edge)} no-edge labels:** "
        f"{sum(o.flagged for o in no_edge)} of {sum(o.replications for o in no_edge)} "
        f"replications flagged "
        f"({sum(o.flagged for o in no_edge) / sum(o.replications for o in no_edge):.0%}).",
        "",
    ]
    if with_edge:
        fp = sum(o.flagged for o in with_edge)
        total = sum(o.replications for o in with_edge)
        lines.append(
            f"**False positives on the real edge:** {fp} of {total} "
            f"({fp / total:.0%}). Every one of these is the tool being wrong in the "
            "direction that matters most."
        )
        lines.append("")

    lines += ["## Which specific finding fired", ""]
    lines += ["| Label | Expected finding | Hit rate |", "|---|---|---:|"]
    for o in outcomes:
        for expected in o.expected_findings:
            hits = o.expected_hits.get(expected, 0)
            lines.append(f"| `{o.label}` | `{expected}` | {hits / o.replications:.0%} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--observations", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/roc_results.md"))
    parser.add_argument("--json-out", type=Path, default=Path("benchmarks/roc_results.json"))
    args = parser.parse_args()

    print(f"Running {args.replications} replications of {len(LABELS)} labels...")
    started = time.time()
    outcomes = run_all(args.replications, args.observations, args.n_boot)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_markdown(outcomes, args.replications), encoding="utf-8")
    args.json_out.write_text(
        json.dumps(
            {
                "replications": args.replications,
                "observations": args.observations,
                "results": [o.to_dict() for o in outcomes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out} and {args.json_out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
