"""Command line interface.

Every capability the agent layer needs is a deterministic subcommand, so an
agent orchestrates this tool rather than improvising statistics of its own.

    qv validate returns.csv --positions pos.csv --trials 40 --out report/
    qv validate --manifest research_manifest.yaml --out report/
    qv adapter init --out my_audit/
    qv scan research.ipynb
    qv trials research.ipynb --reported 1
    qv explain SELECT-DEFLATED-SHARPE-FAILS
    qv demo null_mined --out demo/

There are two ways into `validate`. Flat files take a return series and
optionally positions and a trial matrix. A manifest additionally names a
strategy the tool can **re-run**, which is what unlocks the engine analysis -
the only way to settle whether the strategy reads the future rather than merely
raise the question. Start from `qv adapter init`.

`--suite` selects the questions: `statistical` asks whether the edge is real,
`engine` asks whether the backtest can be trusted as an implementation, and
`full` asks both. They are independent, and running them separately is often
clearer than running them together.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import typer

from qv.findings import CATALOG, catalog_markdown, get_entry
from qv.types import Severity, Suite

app = typer.Typer(
    add_completion=False,
    help="An adversarial validator for quantitative backtests. It falsifies; it cannot validate.",
    no_args_is_help=True,
)

_SEVERITY_MARK = {
    Severity.CRITICAL: "!!",
    Severity.HIGH: "! ",
    Severity.MEDIUM: "~ ",
    Severity.LOW: ". ",
    Severity.INFO: "  ",
}


def _version_line() -> str:
    """The same identity string the report footer carries.

    Printed here so that "was this report built from that commit?" can be
    answered by running the command in a clean checkout and comparing, rather
    than being taken on trust.
    """
    import platform

    from qv.provenance import source_identity

    ident = source_identity()
    parts = [f"proper_validation {ident['version']}"]
    if ident["commit"] is not None:
        parts.append(f"commit {ident['commit']}")
    parts.append(f"source {ident['source_digest']}")
    # ASCII only. This goes to a Windows console as readily as to a UTF-8 one,
    # and a middot that renders as a replacement character helps nobody.
    return ", ".join(parts) + (
        f"\nPython {sys.version.split()[0]}, {platform.platform()}"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", is_eager=True, help="Print the engine identity and exit."
    ),
) -> None:
    # No docstring: the help text lives on the Typer app above, and a second
    # copy here would be a second thing to keep in step.
    # `invoke_without_command` is what lets this run at all: without it click
    # rejects `qv --version` as a missing command before the callback is
    # reached. A bare `qv` is still click's business - `no_args_is_help` on the
    # app catches it earlier than this and prints the help.
    del ctx
    if version:
        typer.echo(_version_line())
        raise typer.Exit()



def _repo_root() -> Path | None:
    """The source checkout this package was installed from, if there is one.

    `benchmarks/` and `skill/` sit beside `qv/` rather than inside it, so they
    are reachable from an editable install or a clone but not from a wheel.
    Resolved from the package location rather than the working directory,
    because the console script does not put the working directory on
    `sys.path` - which is exactly why `qv demo` used to fail under `qv` and
    work under `python -m qv.cli`.
    """
    root = Path(__file__).resolve().parent.parent
    return root if (root / "skill").is_dir() or (root / "benchmarks").is_dir() else None

def qv_is_reachable_anywhere() -> tuple[bool, str | None]:
    """Will `qv` resolve from a directory other than this one?

    This is the joint the whole agent layer hangs on, and the documented setup
    used to break it. The skill installs into `~/.claude/skills`, so it loads
    in *every* project; but the README told people to make a virtualenv inside
    the clone, which puts the console script on `PATH` only while that
    environment is active. A researcher then opens Claude Code on their own
    notebook, in their own directory, the skill activates, the agent runs
    `qv validate`, and gets `command not found` - at which point the one thing
    the skill forbids, improvising the statistics, is the only route left.

    Returns whether the command is reachable and, if not, why - so
    `skill install` can say it at the moment it matters rather than leaving it
    to be discovered in someone else's session.
    """
    import shutil

    found = shutil.which("qv")
    if found is None:
        return False, (
            "`qv` is not on PATH, so it will not resolve in any other directory."
        )
    in_venv = sys.prefix != sys.base_prefix
    if in_venv and Path(found).resolve().is_relative_to(Path(sys.prefix).resolve()):
        return False, (
            f"`qv` resolves to {found}, inside the virtualenv at {sys.prefix}. "
            "That is on PATH only while this environment is active, so it will "
            "not resolve in the project you actually want to audit."
        )
    return True, None


def _reachability_remedy(root: Path | None) -> str:
    """How to get an isolated `qv` onto PATH without a package index.

    Deliberately not `pip install` into the researcher's own environment: this
    package pins numpy, pandas, scipy and statsmodels, and the whole point is
    to audit somebody's research without disturbing the environment that
    research runs in.
    """
    where = root if root is not None else Path("/path/to/proper-validation")
    return (
        "To make it reachable from any project, install it as an isolated tool:\n"
        f'    pipx install --editable "{where}[data]"\n'
        f'    uv tool install --editable "{where}[data]"    # or, if you use uv\n'
        "Both put `qv` on PATH without touching the environment your own "
        "research runs in. The [data] extra is what lets the manifest "
        "`qv adapter init` writes fetch prices."
    )


def _echo_err(message: str) -> None:
    typer.echo(message, err=True)


def load_series(
    path: Path, column: str | None = None, *, selector: str = "--column"
) -> np.ndarray:
    """Read a 1-D numeric series from CSV, Parquet or a plain text column.

    Deliberately forgiving about layout and strict about content: a file that
    parses to something non-numeric is an error, not a silently-coerced array
    of NaNs.

    ``selector`` names the option that can disambiguate an ambiguous file, and
    exists because ``--column`` only applies to the returns argument. Telling
    someone who passed a two-column ``--benchmark`` to "name one with
    --column" sends them to a flag that would rename a column of a different
    file; pass ``selector=None`` for those and the message says what is
    actually needed instead.
    """
    import pandas as pd

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    if column is not None:
        if column not in frame.columns:
            raise typer.BadParameter(
                f"{path.name} has no column {column!r}; found {list(frame.columns)}"
            )
        series = frame[column]
    else:
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] == 0:
            raise typer.BadParameter(f"{path.name} contains no numeric column")
        if numeric.shape[1] > 1:
            # Prefer an obviously-named column before giving up.
            for candidate in ("returns", "return", "ret", "pnl", "value"):
                if candidate in numeric.columns:
                    return numeric[candidate].to_numpy(dtype=float)
            remedy = (
                f"name one with {selector}"
                if selector
                else "supply a file holding a single numeric series"
            )
            raise typer.BadParameter(
                f"{path.name} has {numeric.shape[1]} numeric columns "
                f"({list(numeric.columns)}); {remedy}"
            )
        series = numeric.iloc[:, 0]

    return series.to_numpy(dtype=float)


def load_positions(path: Path) -> np.ndarray:
    """Read a position series or a whole book of weights.

    One numeric column is a single-instrument timing strategy and comes back
    1-D; two or more are a cross-sectional book and come back as ``(T, N)``.
    Both shapes are what `qv.costs.models.as_position_matrix` already accepts,
    and a book is the shape `adapter_protocol.md` calls normal - so reading
    this with the returns loader, as this used to, refused the ordinary case
    and told the user to name a single column. Complying would have measured
    one instrument's turnover and labelled it the portfolio's, which is worse
    than the error it replaced.

    An unnamed integer index column is dropped rather than counted as an
    instrument: ``DataFrame.to_csv()`` on a default RangeIndex writes one, and
    a monotonically rising "weight" would corrupt turnover without ever
    looking wrong.
    """
    import pandas as pd

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")

    frame = (
        pd.read_parquet(path)
        if path.suffix.lower() in (".parquet", ".pq")
        else pd.read_csv(path)
    )
    numeric = frame.select_dtypes("number")

    for name in list(numeric.columns):
        if not re.fullmatch(r"Unnamed: \d+", str(name)):
            continue
        values = numeric[name].to_numpy()
        if np.array_equal(values, np.arange(len(values))):
            numeric = numeric.drop(columns=[name])

    if numeric.shape[1] == 0:
        raise typer.BadParameter(f"{path.name} contains no numeric column")
    if numeric.shape[1] == 1:
        return numeric.iloc[:, 0].to_numpy(dtype=float)
    return numeric.to_numpy(dtype=float)


def load_matrix(path: Path) -> np.ndarray:
    """Read a ``(T, N)`` trial matrix: one column per configuration tried."""
    import pandas as pd

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")
    frame = (
        pd.read_parquet(path)
        if path.suffix.lower() in (".parquet", ".pq")
        else pd.read_csv(path)
    )
    numeric = frame.select_dtypes("number")
    if numeric.shape[1] < 2:
        raise typer.BadParameter(
            f"{path.name} needs at least 2 numeric columns to be a trial matrix"
        )
    return numeric.to_numpy(dtype=float)


@app.command()
def validate(
    returns: Path | None = typer.Argument(
        None, help="CSV or Parquet holding the return series. Omit when using --manifest."
    ),
    manifest: Path | None = typer.Option(
        None,
        help="A research_manifest.yaml. Loads the data, re-runs the declared "
        "parameter grid, and unlocks the engine analysis.",
    ),
    adapter: str | None = typer.Option(
        None,
        help="Override the manifest's strategy adapter, as 'file.py:function'.",
    ),
    offline: bool = typer.Option(
        False, help="Refuse the network and fail loudly if the cache cannot serve."
    ),
    suite: str = typer.Option(
        "full",
        help="Which questions to ask: 'statistical' (is the edge real?), "
        "'engine' (is the backtest trustworthy?), or 'full' for both.",
    ),
    column: str | None = typer.Option(None, help="Which column holds the returns."),
    positions: Path | None = typer.Option(None, help="Position series, or a book of weights with one column per instrument. Unlocks turnover, costs and break-even cost."),
    asset_returns: Path | None = typer.Option(
        None, help="Returns of the traded instrument, for the matched-exposure test."
    ),
    benchmark: Path | None = typer.Option(None, help="Benchmark returns to compare against."),
    trial_matrix: Path | None = typer.Option(
        None, help="(T, N) matrix of trial returns, one column per configuration."
    ),
    trials: int | None = typer.Option(
        None, "--trials", help="Number of configurations examined. The most under-reported "
        "number in backtesting; overstate it if unsure."
    ),
    asset_class: str | None = typer.Option(
        None, help="For the realistic-cost comparison, e.g. us_large_cap_etf."
    ),
    periods_per_year: int = typer.Option(252, help="252 daily, 12 monthly, 4 quarterly."),
    name: str = typer.Option("strategy", help="Name shown on the report."),
    out: Path = typer.Option(Path("qv_report"), help="Directory for report.html and report.json."),
    n_boot: int = typer.Option(2000, help="Bootstrap resamples."),
    seed: int = typer.Option(0, help="Seed. Every number in the report is reproducible from it."),
) -> None:
    """Audit a backtest and write a self-contained HTML and JSON report.

    A return series alone supports the statistical suite. With `--manifest` the
    tool imports and re-runs the strategy the manifest names, which unlocks the
    engine analysis and rebuilds the trial matrix and parameter surface from
    the declared grid rather than taking them on trust.

    `--suite statistical` asks whether the edge is real, `--suite engine`
    whether the backtest is trustworthy, `--suite full` both.
    """
    from qv.report.render import write_report

    if suite not in {s.value for s in Suite}:
        _echo_err(
            f"unknown suite {suite!r}; expected one of "
            f"{', '.join(s.value for s in Suite)}"
        )
        raise typer.Exit(code=2)

    if (returns is None) == (manifest is None):
        _echo_err(
            "give either a return series or --manifest, not both and not neither. "
            "Only a manifest can name a strategy to re-run, which is what the "
            "engine analysis needs."
        )
        raise typer.Exit(code=2)

    if manifest is not None:
        from qv.pipeline import audit_from_manifest

        try:
            run = audit_from_manifest(
                manifest,
                adapter=adapter,
                offline=offline,
                n_boot=n_boot,
                suite=suite,
            )
        except (FileNotFoundError, ValueError, AttributeError, TypeError) as exc:
            _echo_err(str(exc))
            raise typer.Exit(code=2) from exc

        report = run.report
        audited_returns = run.returns
        typer.echo(
            f"Loaded {len(run.prices)} sessions of "
            f"{len(run.manifest.data.universe)} instruments"
            + "".join(
                f"\n  {label} vintage {vintage}"
                for label, vintage in run.vintages.items()
            )
        )
        if run.grid is not None:
            typer.echo(f"Evaluated {run.grid.n_trials} configurations: {run.grid.summary()}")
    else:
        from qv.audit import AuditInputs, run_audit

        inputs = AuditInputs(
            returns=load_series(returns, column),
            periods_per_year=periods_per_year,
            name=name,
            positions=None if positions is None else load_positions(positions),
            asset_returns=(
                None
                if asset_returns is None
                else load_series(asset_returns, selector=None)
            ),
            benchmark_returns=(
                None if benchmark is None else load_series(benchmark, selector=None)
            ),
            asset_class=asset_class,
            n_trials=trials,
            trial_returns=None if trial_matrix is None else load_matrix(trial_matrix),
            n_boot=n_boot,
            seed=seed,
        )
        report = run_audit(inputs)
        audited_returns = inputs.returns

    html_path, json_path = write_report(report, out, returns=audited_returns)

    typer.echo(f"\n{report.name} - {report.suite.label}")
    typer.echo(f"Verdict: {report.verdict.label.upper()}")
    typer.echo(
        "\nThis tool falsifies; it cannot validate. A clean run means the tests applied "
        "did not break the backtest, not that the strategy works."
    )

    if report.findings:
        typer.echo(f"\n{len(report.findings)} findings:")
        for finding in report.findings_by_severity():
            typer.echo(
                f"  {_SEVERITY_MARK[finding.severity]} [{finding.severity.label:8s}] "
                f"{finding.id}"
            )
            typer.echo(f"       {finding.detail}")
    else:
        typer.echo("\nNo findings from the tests that this tier supports.")

    if report.not_tested:
        typer.echo(f"\nCould not test ({len(report.not_tested)}):")
        for gap in report.not_tested:
            typer.echo(f"  - {gap}")

    typer.echo(f"\nWrote {html_path} and {json_path}")
    raise typer.Exit(code=1 if report.worst_severity >= Severity.CRITICAL else 0)


@app.command()
def scan(
    path: Path = typer.Argument(..., help="A .py file or a .ipynb notebook."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """Scan source for look-ahead and simulation anti-patterns.

    A static scan finds recognisable shapes, not proof. It cannot tell whether
    a negative shift feeds a label or a feature. Use
    `qv validate --manifest research_manifest.yaml --suite engine` for the
    behavioural test that settles it, and `qv adapter init` to get there.
    """
    from qv.leakage.notebook import parse_notebook
    from qv.leakage.static import scan_file

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")

    if path.suffix == ".ipynb":
        result = parse_notebook(path)
        findings = result.to_findings()
        payload = result.to_dict()
    else:
        result = scan_file(path)
        findings = result.to_findings()
        payload = result.to_dict()

    if as_json:
        typer.echo(json.dumps({**payload, "findings": [f.to_dict() for f in findings]}, indent=2))
    else:
        if not findings:
            typer.echo(f"{path.name}: no known leakage patterns found.")
            typer.echo(
                "That is not a clean bill of health - a static scan cannot see inside "
                "library calls or follow data through variables."
            )
        for finding in findings:
            typer.echo(f"  {_SEVERITY_MARK[finding.severity]} [{finding.severity.label}] {finding.id}")
            typer.echo(f"       {finding.detail}")
            typer.echo(f"       -> {finding.remediation}")
    raise typer.Exit(code=1 if any(f.severity >= Severity.CRITICAL for f in findings) else 0)


@app.command()
def trials(
    notebook: Path = typer.Argument(..., help="A .ipynb to excavate."),
    directory: Path | None = typer.Option(
        None, help="Sibling directory to check for filename lineage."
    ),
    reported: int | None = typer.Option(None, help="Trial count the writeup claims."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """Recover a lower bound on how many configurations were really examined."""
    from qv.leakage.notebook import estimate_trials, parse_notebook

    scan_result = parse_notebook(notebook)
    estimate = estimate_trials(
        scan_result, directory=directory or notebook.parent, reported=reported
    )

    if as_json:
        typer.echo(json.dumps(estimate.to_dict(), indent=2))
        raise typer.Exit(code=0)

    typer.echo(f"{notebook.name}: at least {estimate.lower_bound} configurations examined.")
    typer.echo(
        "This is a lower bound. Configurations abandoned without being saved leave no "
        "trace, so the true count is higher - interview the researcher and revise upward."
    )
    if estimate.evidence:
        typer.echo("\nEvidence:")
        for item in estimate.evidence:
            typer.echo(f"  - [{item.kind}] implies >= {item.implied_trials}: {item.detail}")
    if estimate.understated:
        typer.echo(f"\nThe writeup declares {estimate.reported}. That is contradicted above.")
    raise typer.Exit(code=1 if estimate.understated else 0)


@app.command()
def explain(
    finding_id: str = typer.Argument(None, help="A catalog id, e.g. LEAK-NEGATIVE-SHIFT."),
    all_entries: bool = typer.Option(False, "--all", help="List every catalog entry."),
    markdown: bool = typer.Option(False, "--markdown", help="Render the catalog as markdown."),
) -> None:
    """Explain a finding: what it means and what to do about it."""
    if markdown:
        typer.echo(catalog_markdown())
        raise typer.Exit(code=0)

    if all_entries or finding_id is None:
        typer.echo(f"{len(CATALOG)} defects in the catalog:\n")
        for entry in CATALOG.values():
            typer.echo(f"  [{entry.severity.label:8s}] {entry.id:32s} {entry.title}")
        raise typer.Exit(code=0)

    try:
        entry = get_entry(finding_id)
    except KeyError as exc:
        _echo_err(str(exc))
        raise typer.Exit(code=2) from exc

    typer.echo(f"{entry.id} - {entry.title}")
    typer.echo(f"Severity: {entry.severity.label}    Category: {entry.category}\n")
    typer.echo(f"{entry.explanation}\n")
    typer.echo(f"Detection: {entry.detection}\n")
    typer.echo(f"Remediation: {entry.remediation}")


@app.command()
def demo(
    label: str = typer.Argument("null_mined", help="A synthetic label with known ground truth."),
    out: Path = typer.Option(Path("qv_demo"), help="Where to write the report."),
    observations: int = typer.Option(2000, help="Length of the generated series."),
    seed: int = typer.Option(0),
) -> None:
    """Audit a synthetic strategy whose true edge is known by construction.

    The fastest way to see what the tool does. `null_mined` is the flagship:
    the best of several hundred random rules, which looks publishable and has
    an edge of exactly zero.
    """
    root = _repo_root()
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from benchmarks.generate import LABELS, generate
    except ImportError as exc:  # pragma: no cover - only when installed without the repo
        _echo_err(
            "the synthetic benchmarks live beside the package rather than inside "
            "it, so `qv demo` needs a source checkout: clone the repository and "
            "`pip install -e .` from it."
        )
        raise typer.Exit(code=2) from exc

    if label not in LABELS:
        _echo_err(f"unknown label {label!r}; expected one of {', '.join(LABELS)}")
        raise typer.Exit(code=2)

    from qv.audit import AuditInputs, run_audit
    from qv.report.render import write_report

    strat = generate(label, seed=seed, n=observations)
    typer.echo(f"{label}: {strat.description}")
    typer.echo(f"True edge: {'REAL' if strat.has_edge else 'none, by construction'}\n")

    report = run_audit(
        AuditInputs(
            returns=strat.returns,
            positions=strat.positions,
            asset_returns=strat.asset_returns,
            asset_class=strat.asset_class or "us_large_cap_equity",
            benchmark_returns=strat.asset_returns if label == "levered_beta" else None,
            n_trials=strat.n_trials,
            trial_returns=strat.trial_returns,
            strategy=strat.strategy,
            strategy_data=strat.strategy_data,
            name=label,
            seed=seed,
        )
    )
    html_path, json_path = write_report(report, out, returns=strat.returns)

    typer.echo(f"Verdict: {report.verdict.label.upper()}")
    for finding in report.findings_by_severity():
        typer.echo(f"  {_SEVERITY_MARK[finding.severity]} [{finding.severity.label}] {finding.id}")
    typer.echo(f"\nWrote {html_path}")


adapter_app = typer.Typer(
    add_completion=False,
    help="Scaffold the two files a re-runnable audit needs.",
    no_args_is_help=True,
)
app.add_typer(adapter_app, name="adapter")


_ADAPTER_TEMPLATE = '''"""The adapter: this strategy, stated precisely enough to re-run.

`qv validate --manifest research_manifest.yaml` imports `positions` from here,
calls it once for the reported configuration, once for every point in the
declared grid, and repeatedly on deliberately corrupted data to check that
nothing before a cut point depends on anything after it.

Three requirements, all load-bearing:

1. Read only the `prices` frame you are handed. Do not open files, hit the
   network, or reach for a DataFrame from an enclosing scope - if the function
   does not read its argument, the leakage test cannot corrupt what it reads
   and will pass without having tested anything.
2. Be deterministic. Seed anything stochastic inside the function.
3. Return one row of positions per row of `prices`, on the same index. A book
   of weights, one column per instrument, is the normal shape.
"""

from __future__ import annotations

import pandas as pd


def positions(prices: pd.DataFrame, **params) -> pd.DataFrame:
    """Return the weights to hold, one row per row of `prices`.

    `prices` is wide: a DatetimeIndex, one column per instrument, in the order
    the manifest declares. Row t of the result is the book held *into* t+1, so
    it may use data up to and including row t and nothing after it.

    Replace the body with your own signal. The `.shift(1)` at the end is what
    stops a position earning the return that produced it - keep it, or state
    plainly in the manifest why it is not needed.
    """
    lookback = params.get("lookback", 252)
    top_n = params.get("top_n", 2)

    score = prices / prices.shift(lookback) - 1.0
    ranks = score.rank(axis=1, ascending=False, na_option="bottom")
    weights = (ranks <= top_n).astype(float) / top_n
    weights[score.isna().all(axis=1)] = 0.0
    return weights.shift(1).fillna(0.0)
'''


_MANIFEST_TEMPLATE = '''# What this research asserts, recorded so the audit can be reproduced and so
# the assumptions are visible rather than implied.
#
#   qv validate --manifest research_manifest.yaml --out report/

name: REPLACE ME - a human-readable name for the strategy

data:
  # The instruments to load, in the order the adapter expects its columns.
  # Replace with your own. More than one, if the strategy ranks a cross-section
  # against itself - `top_n` below has nothing to choose from otherwise.
  universe: [SPY, IEF, GLD, XLK, XLP]
  benchmark: SPY
  start_date: "2005-01-01"
  # Pinned deliberately. An open end date reproduces whatever the vendor
  # returns today rather than the numbers that were published.
  end_date: "2024-12-31"
  source: yfinance
  # Selects the realistic cost range that break-even cost is judged against.
  # Omit it and the tool declines to render a verdict on costs rather than
  # inventing an estimate for a market it was not told about.
  asset_class: us_large_cap_etf
  frequency: daily
  # The one question no tier can answer from the data: survivorship leaves no
  # trace in a return series. Declare it, or the report lists it under what
  # could not be tested.
  universe_point_in_time: null
  universe_note: >
    REPLACE ME - how the instrument list was built, and when.

strategy:
  adapter: qv_adapter.py:positions
  # Parameters that were never searched, passed to every call.
  fixed: {}

# The configuration actually reported. It must name every axis below.
chosen_parameters:
  lookback: 252
  top_n: 2

search:
  # Every configuration examined, including the ones abandoned. Keyed by the
  # parameter each axis varies.
  axes:
    lookback: [126, 189, 252]
    top_n: [1, 2, 3]
  # The number the whole audit turns on. Defaults to the size of the grid
  # above; raise it if more was really tried, because a grid cannot include
  # what was discarded before it was written down. Overstating is the honest
  # direction - the correction is logarithmic, so honesty is cheap.
  n_trials: null

process:
  positions_shifted: true
  lookahead_possible: false
  notes: >
    REPLACE ME - anything a reviewer would want to know.
'''


@adapter_app.command("init")
def adapter_init(
    out: Path = typer.Option(Path("."), help="Directory to write the two files into."),
    force: bool = typer.Option(False, help="Overwrite files that already exist."),
) -> None:
    """Write a `qv_adapter.py` and a `research_manifest.yaml` to fill in.

    Together they are everything an audit needs, both suites included. Edit the adapter to
    compute your signal, edit the manifest to declare what you searched, then
    run `qv validate --manifest research_manifest.yaml`.
    """
    out.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for filename, content in (
        ("qv_adapter.py", _ADAPTER_TEMPLATE),
        ("research_manifest.yaml", _MANIFEST_TEMPLATE),
    ):
        path = out / filename
        if path.exists() and not force:
            skipped.append(path)
            continue
        path.write_text(content, encoding="utf-8")
        written.append(path)

    for path in written:
        typer.echo(f"Wrote {path}")
    for path in skipped:
        typer.echo(f"Kept existing {path} (pass --force to overwrite)")

    if written:
        typer.echo(
            "\nNext: put your signal in `positions`, declare the grid you "
            "searched, then\n  qv validate --manifest "
            f"{out / 'research_manifest.yaml'} --out report/"
        )
    if skipped and not written:
        raise typer.Exit(code=1)


skill_app = typer.Typer(
    add_completion=False,
    help="Install the Claude Code skill, so an agent can drive this tool.",
    no_args_is_help=True,
)
app.add_typer(skill_app, name="skill")


@skill_app.command("install")
def skill_install(
    user: bool = typer.Option(
        False,
        "--user",
        help="Install for every project, into ~/.claude/skills, instead of just this one.",
    ),
    target: Path | None = typer.Option(
        None, help="Install into this directory instead of a .claude/skills path."
    ),
    force: bool = typer.Option(False, help="Overwrite an existing installation."),
) -> None:
    """Copy the skill where Claude Code looks for it.

    Claude Code reads skills from `.claude/skills/<name>/SKILL.md`, relative to
    the project by default or under the home directory with `--user`. The skill
    ships in this repository at `skill/`, which is the wrong place for Claude
    Code to find it, so this copies it across.

    Installing it is what turns the CLI into something an agent can drive: it
    carries the audit protocol, the adapter contract, the manifest schema and
    the findings catalog.
    """
    import shutil

    root = _repo_root()
    source = None if root is None else root / "skill"
    if source is None or not (source / "SKILL.md").exists():
        _echo_err(
            "cannot find the skill source. It ships beside the package at "
            "`skill/`, so this needs a source checkout: clone the repository "
            "and `pip install -e .` from it."
        )
        raise typer.Exit(code=2)

    if target is not None:
        destination = target
    elif user:
        destination = Path.home() / ".claude" / "skills" / "proper-validation"
    else:
        destination = Path.cwd() / ".claude" / "skills" / "proper-validation"

    if destination.exists() and not force:
        _echo_err(
            f"{destination} already exists. Pass --force to overwrite it, or "
            "delete it first."
        )
        raise typer.Exit(code=1)

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)

    files = sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file())
    typer.echo(f"Installed the skill to {destination}")
    for name in files:
        typer.echo(f"  {name}")
    typer.echo(
        "\nStart a new Claude Code session in this directory and ask it to audit "
        "a backtest.\nThe skill activates on its own; there is nothing to enable."
    )

    # The skill is useless without the command it drives, and the failure shows
    # up in a different directory from the one this ran in. Say it here.
    reachable, why = qv_is_reachable_anywhere()
    if not reachable:
        _echo_err(f"\nWarning: {why}")
        _echo_err(_reachability_remedy(root))


def main() -> int:  # pragma: no cover - console entry point
    app()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
