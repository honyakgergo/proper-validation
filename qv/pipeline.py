"""Manifest plus adapter to audited report, with no glue code in between.

This is the generalisation of the driver scripts that used to sit beside each
example. Every one of them did the same seven things - load a universe, load a
benchmark, load the factors, evaluate the reported configuration, re-run the
whole declared grid to rebuild the trial matrix and the parameter surface, wrap
the strategy for the behavioural test, and hand the lot to ``run_audit`` - and
each did them slightly differently, which is exactly how two reports of the
same strategy stop being comparable.

It also closes the gap that made the engine analysis unreachable in practice:
the skill tells the agent to get a re-runnable strategy out of the researcher,
and until this existed the only way to use one was a bespoke script. Now the
agent writes an adapter and fills in a manifest, and one command produces the
report - for either suite, or both.

What it does not cover, deliberately: a search that is not a parameter grid.
``examples/01_mined_noise`` draws ten thousand random rules rather than
crossing a few axes, so it keeps its own driver and stands as the documented
escape hatch. A manifest that can express any search would just be a
programming language.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qv.adapter import frame_adapter
from qv.audit import AuditInputs, AuditReport, run_audit
from qv.data.loaders import (
    read_membership_frame,
    load_fama_french,
    load_prices,
    load_universe,
    read_price_frame,
)
from qv.manifest import ResearchManifest, load_manifest
from qv.types import Suite
from qv.trials import GridResult, run_parameter_grid

__all__ = ["AuditRun", "load_adapter", "audit_from_manifest"]


def load_adapter(reference: str, base: Path | None = None) -> Callable[..., Any]:
    """Import ``path/to/file.py:function`` (or ``module:function``).

    The file is executed to import it. There is no sandbox and no subprocess:
    auditing a re-runnable strategy means running the researcher's code in this
    interpreter, so read an adapter before you point the tool at it.
    """
    if ":" not in reference:
        raise ValueError(
            f"adapter reference {reference!r} must be 'path/to/file.py:function' "
            "or 'package.module:function'"
        )
    target, _, attribute = reference.rpartition(":")

    path = Path(target)
    if base is not None and not path.is_absolute():
        path = base / path

    if path.suffix == ".py":
        if not path.exists():
            raise FileNotFoundError(f"no such adapter file: {path}")
        spec = importlib.util.spec_from_file_location(f"qv_adapter_{path.stem}", path)
        if spec is None or spec.loader is None:  # pragma: no cover - import machinery
            raise ImportError(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        # Registered before execution so a dataclass or pickle inside the
        # adapter can find its own module.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    if not hasattr(module, attribute):
        available = [n for n in dir(module) if not n.startswith("_")]
        raise AttributeError(
            f"{target} has no attribute {attribute!r}; found {available}"
        )
    resolved = getattr(module, attribute)
    if not callable(resolved):
        raise TypeError(f"{reference} is not callable")
    return resolved


@dataclass
class AuditRun:
    """The report, plus the pieces a caller may want to print or inspect."""

    report: AuditReport
    returns: np.ndarray
    prices: pd.DataFrame
    positions: pd.DataFrame
    grid: GridResult | None
    manifest: ResearchManifest
    vintages: dict[str, str]
    #: Mean risk-free rate per period, or 0.0 if no factors were loaded. The
    #: audit computes every risk-adjusted statistic in excess of this, so a
    #: caller printing its own performance preamble must use the same basis or
    #: contradict the report it is about to write.
    risk_free_mean: float = 0.0
    #: Benchmark returns as the audit saw them, for that same preamble.
    benchmark_returns: Any = None


def _benchmark_returns(
    ticker: str, index: pd.Index, start: str, end: str, offline: bool
) -> tuple[pd.Series, str]:
    cached = load_prices(ticker, start, end, offline=offline)
    series = (
        cached.frame["close"]
        .reindex(index)
        .ffill()
        .pct_change()
        .fillna(0.0)
    )
    return series, cached.vintage


def audit_from_manifest(
    manifest: str | Path | ResearchManifest,
    adapter: str | Callable[..., Any] | None = None,
    offline: bool = False,
    n_boot: int = 2000,
    prices: pd.DataFrame | None = None,
    suite: Suite | str | None = None,
) -> AuditRun:
    """Run the full audit a manifest describes.

    ``adapter`` overrides ``strategy.adapter`` in the manifest, and may be the
    callable itself rather than a reference, which is what the tests use.

    ``prices`` supplies the price frame directly instead of loading it, which
    is how anyone holding data in the documented wide schema uses this without
    going near the network layer - and how the test suite runs offline.

    ``suite`` selects which questions to ask, overriding the manifest. The
    data loading and the grid run are shared, so asking one suite costs no
    more than asking both minus the tests it skips.
    """
    if not isinstance(manifest, ResearchManifest):
        manifest = load_manifest(manifest)

    data = manifest.data
    start, end = data.start_date, data.end_date
    chosen_suite = Suite(suite) if suite is not None else manifest.suite

    reference = adapter if adapter is not None else manifest.adapter_reference()
    if reference is None:
        raise ValueError(
            "no strategy adapter: set `strategy.adapter` in the manifest or pass "
            "one explicitly. Without it the audit cannot re-run the strategy, and "
            "the engine analysis has nothing to examine"
        )
    positions_of = (
        reference
        if callable(reference)
        else load_adapter(
            reference,
            base=manifest.source_path.parent if manifest.source_path else None,
        )
    )

    vintages: dict[str, str] = {}
    #: The frame before the common-calendar alignment, which is the only place
    #: inclusion timing is visible - dropping the ragged rows is exactly what
    #: makes a patchy universe look clean.
    raw_prices = None
    if prices is not None:
        missing = [t for t in data.universe if t not in prices.columns]
        if missing:
            raise ValueError(
                f"the supplied price frame is missing {missing}, which the "
                "manifest declares in its universe"
            )
        # Reordered to the declared universe, because a strategy that selects
        # columns by position would otherwise silently trade the wrong ones.
        prices = prices[list(data.universe)]
        vintages["prices"] = "supplied by the caller"
    elif data.price_frame is not None:
        path = manifest.resolve(data.price_frame)
        prices = read_price_frame(path, data.price_column).loc[str(start) : str(end)]
        missing = [t for t in data.universe if t not in prices.columns]
        if missing:
            raise ValueError(
                f"{path.name} is missing {missing}, which the manifest declares "
                "in its universe"
            )
        prices = prices[list(data.universe)]
        vintages["prices"] = f"local file {path.name}"
    else:
        universe = load_universe(
            data.universe, start, end, column=data.price_column, offline=offline
        )
        prices = universe.frame
        raw_prices = universe.unaligned
        vintages["prices"] = universe.vintage

    # Parsed here so a malformed list fails before the strategy runs, but
    # *evaluated* later against the trimmed window - see the AuditInputs call.
    # Measuring against any window other than the one the report describes
    # answers a question the reader is never shown.
    membership = None
    if data.membership_frame is not None:
        membership_path = manifest.resolve(data.membership_frame)
        membership = read_membership_frame(membership_path, data.membership_index)
        vintages["membership"] = membership.source

    # Factors are loaded before the strategy runs, so the sample can be
    # narrowed to where every input exists *before* anything is computed on it.
    # Ken French publishes on a lag, so the last sessions of the factor file
    # are NaN whenever someone audits up to the present. Masking the factors
    # after the fact would leave the returns on a longer axis than the factors
    # that are supposed to explain them - and a report whose Sharpe and whose
    # alpha describe different samples is not internally consistent, however
    # right each number looks alone.
    raw_factors = None
    if data.load_factors:
        cached = load_fama_french(start, end, offline=offline)
        vintages["factors"] = cached.vintage
        raw_factors = cached.frame.reindex(prices.index)
        usable = raw_factors.notna().all(axis=1)
        dropped = int((~usable).sum())
        if dropped:
            prices = prices.loc[usable[usable].index]
            raw_factors = raw_factors.loc[prices.index]
            vintages["factor_alignment"] = (
                f"{dropped} sessions dropped: no factor data, most likely the "
                f"publication lag at the end of the sample"
            )

    fixed = dict(manifest.strategy.fixed)
    chosen = dict(manifest.chosen_parameters)

    positions = pd.DataFrame(positions_of(prices, **chosen, **fixed))
    asset_returns = prices.pct_change().fillna(0.0)
    aligned = positions.reindex_like(asset_returns).fillna(0.0)
    returns = (aligned * asset_returns).sum(axis=1)

    grid: GridResult | None = None
    # The grid exists to price selection bias, which is a statistical question.
    # An engine-only report has no use for the trial matrix and should not imply
    # a search was priced, so it does not pay to re-run one.
    if manifest.search.axes and chosen_suite.runs_statistical:
        grid = run_parameter_grid(
            positions_of,
            prices,
            manifest.search.axes,
            chosen,
            periods_per_year=manifest.periods_per_year,
            **fixed,
        )

    benchmark_returns = None
    if data.benchmark:
        series, vintage = _benchmark_returns(
            data.benchmark, prices.index, start, end, offline
        )
        benchmark_returns = series.to_numpy()
        vintages["benchmark"] = vintage

    factors = factor_names = risk_free = None
    if raw_factors is not None:
        factor_names = [c for c in raw_factors.columns if c != "RF"]
        factors = raw_factors[factor_names]
        risk_free = raw_factors["RF"].to_numpy()

    # The re-runnable form. Labels only - closing over `prices` would stop the corruption
    # ever reaching the strategy, and the behavioural test would pass without
    # having tested anything.
    strategy = frame_adapter(
        positions_of, prices.index, prices.columns, **chosen, **fixed
    )

    report = run_audit(
        AuditInputs(
            returns=returns.to_numpy(),
            periods_per_year=manifest.periods_per_year,
            name=manifest.name,
            dates=prices.index,
            positions=positions.to_numpy(),
            strategy=strategy,
            strategy_data=prices.to_numpy(),
            suite=chosen_suite,
            asset_return_frame=asset_returns.to_numpy(),
            raw_prices=raw_prices if raw_prices is not None else prices,
            asset_class=data.asset_class,
            benchmark_returns=benchmark_returns,
            benchmark_name=data.benchmark or "benchmark",
            factors=factors,
            factor_names=factor_names,
            risk_free=risk_free,
            membership=membership,
            #: The post-trim window, which is what the rest of the report
            #: describes. `prices` has already been narrowed by the factor
            #: alignment above.
            sample_window=(prices.index[0], prices.index[-1]),
            universe_names=list(prices.columns),
            universe_point_in_time=data.universe_point_in_time,
            universe_note=(
                None
                if data.universe_note is None
                else " ".join(data.universe_note.split())
            ),
            n_trials=manifest.search.declared_trials(
                0 if grid is None else grid.n_trials
            ),
            trial_returns=None if grid is None else grid.trial_returns,
            parameter_scores=None if grid is None else grid.parameter_scores,
            chosen_parameter_index=None if grid is None else grid.chosen_index,
            n_boot=n_boot,
            seed=manifest.seed,
        )
    )
    report.provenance["data_vintages"] = vintages

    return AuditRun(
        report=report,
        returns=returns.to_numpy(),
        prices=prices,
        positions=positions,
        grid=grid,
        manifest=manifest,
        vintages=vintages,
        risk_free_mean=0.0 if risk_free is None else float(np.mean(risk_free)),
        benchmark_returns=benchmark_returns,
    )
