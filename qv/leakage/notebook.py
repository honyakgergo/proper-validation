"""Trial archaeology: recovering how many configurations were really tried.

The number of trials is the key input to every multiple-testing correction and
the most under-reported number in backtesting - not usually through dishonesty
but because nobody counts. A researcher genuinely does not remember the eleven
lookback windows they tried on a Tuesday three months ago.

A notebook remembers. Execution counts record how many times cells were run and
in what order; parameter literals record which values were written down; grid
definitions record the search space explicitly; and filenames record the
lineage of abandoned attempts.

**Everything here is a lower bound**, and the report must say so. Configurations
tried and deleted leave no trace, and the researcher interview exists to revise
this number upward. A lower bound is still worth a great deal: it is almost
always far above the "1" that the writeup implies.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qv.findings import make_finding
from qv.leakage.static import ScanResult, scan_source
from qv.types import Finding

__all__ = [
    "Evidence",
    "TrialEstimate",
    "NotebookScan",
    "parse_notebook",
    "estimate_trials",
    "filename_lineage",
]

#: Filename fragments that mark a re-attempt. A directory containing
#: ``test``, ``test_v2`` and ``test_final`` is a specification search with the
#: search left visible.
_LINEAGE_MARKERS = (
    "final",
    "fixed",
    "new",
    "old",
    "copy",
    "v2",
    "v3",
    "test",
    "try",
    "attempt",
    "proper",
    "correct",
    "unbiased",
    "hope",
    "real",
    "clean",
    "backup",
    "draft",
    "rev",
)

_GRID_CALLS = frozenset({"product", "ParameterGrid", "GridSearchCV", "RandomizedSearchCV"})

_PARAM_HINTS = (
    "window",
    "lookback",
    "period",
    "threshold",
    "lag",
    "span",
    "alpha",
    "beta",
    "gamma",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "hidden",
    "layers",
    "dropout",
    "epochs",
    "seed",
    "top_n",
    "quantile",
    "z_score",
    "holding",
    "rebalance",
)


@dataclass(frozen=True)
class Evidence:
    """One reason to believe more trials were run than were reported."""

    kind: str
    detail: str
    implied_trials: int
    locations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "implied_trials": self.implied_trials,
            "locations": list(self.locations),
        }


@dataclass(frozen=True)
class TrialEstimate:
    """A defensible lower bound on the number of configurations examined."""

    lower_bound: int
    evidence: tuple[Evidence, ...]
    reported: int | None = None

    @property
    def understated(self) -> bool:
        """Does the notebook contradict the reported trial count?"""
        return self.reported is not None and self.lower_bound > self.reported

    def to_finding(self) -> Finding | None:
        """A finding, when the evidence contradicts the declared count."""
        if self.lower_bound <= 1:
            return None
        reported = "not declared" if self.reported is None else str(self.reported)
        return make_finding(
            "SELECT-UNDECLARED-TRIALS",
            detail=(
                f"Notebook forensics imply at least {self.lower_bound} configurations were "
                f"examined; the manifest declares {reported}. This is a lower bound - "
                "configurations abandoned without being saved leave no trace."
            ),
            evidence={
                "lower_bound": self.lower_bound,
                "reported": self.reported,
                "evidence": [e.to_dict() for e in self.evidence],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower_bound": self.lower_bound,
            "reported": self.reported,
            "understated": self.understated,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(frozen=True)
class NotebookScan:
    """A parsed notebook: its code, its execution history, its leakage hits."""

    path: str
    n_cells: int
    n_code_cells: int
    execution_counts: tuple[int, ...]
    source_by_cell: tuple[str, ...] = field(repr=False, default=())
    scans: tuple[ScanResult, ...] = field(repr=False, default=())

    @property
    def max_execution_count(self) -> int:
        return max(self.execution_counts, default=0)

    @property
    def executions_beyond_cell_count(self) -> int:
        """Cell runs in excess of one per code cell.

        A notebook of 20 code cells whose highest execution count is 140 was
        run through at least seven times, and re-runs of an analysis cell are
        how a parameter search leaves fingerprints.
        """
        return max(0, self.max_execution_count - self.n_code_cells)

    @property
    def out_of_order(self) -> int:
        """Cells whose execution count is lower than an earlier cell.

        Evidence that the notebook was not run top to bottom, which means the
        saved state does not correspond to any single clean run.
        """
        counts = [c for c in self.execution_counts if c > 0]
        return sum(1 for a, b in zip(counts, counts[1:]) if b < a)

    @property
    def all_hits(self) -> tuple:
        return tuple(hit for scan in self.scans for hit in scan.hits)

    def to_findings(self) -> list[Finding]:
        """Leakage findings across every cell, merged and de-duplicated."""
        merged = ScanResult(
            hits=self.all_hits,
            source_name=Path(self.path).name,
            n_lines=sum(s.n_lines for s in self.scans),
        )
        return merged.to_findings()

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n_cells": self.n_cells,
            "n_code_cells": self.n_code_cells,
            "max_execution_count": self.max_execution_count,
            "executions_beyond_cell_count": self.executions_beyond_cell_count,
            "out_of_order": self.out_of_order,
            "hits": [h.to_dict() for h in self.all_hits],
        }


def parse_notebook(path: str | Path) -> NotebookScan:
    """Read a ``.ipynb`` and scan every code cell.

    Uses the raw JSON rather than ``nbformat.read`` so that a notebook written
    by an older or unusual tool still parses. Magics and shell escapes are
    stripped before the AST scan, since ``%matplotlib inline`` is not Python.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such notebook: {p}")

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p.name} is not valid JSON: {exc}") from exc

    cells = payload.get("cells")
    if cells is None:
        raise ValueError(f"{p.name} has no 'cells' key; is it really a notebook?")

    counts, sources, scans = [], [], []
    n_code = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        n_code += 1
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(source)

        count = cell.get("execution_count")
        if isinstance(count, int):
            counts.append(count)

        scans.append(scan_source(_strip_magics(source), source_name=p.name, cell=i))

    return NotebookScan(
        path=str(p),
        n_cells=len(cells),
        n_code_cells=n_code,
        execution_counts=tuple(counts),
        source_by_cell=tuple(sources),
        scans=tuple(scans),
    )


def _strip_magics(source: str) -> str:
    """Blank out IPython magics and shell escapes so the AST parses."""
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")) or stripped.endswith("?"):
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def filename_lineage(directory: str | Path, pattern: str = "*.ipynb") -> list[str]:
    """Filenames whose names betray a specification search.

    A directory holding ``model.ipynb``, ``model_v2.ipynb`` and
    ``model_final_fixed.ipynb`` is showing you three trials that the writeup
    will describe as one.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")
    return sorted(
        f.name
        for f in d.glob(pattern)
        if any(marker in f.stem.lower() for marker in _LINEAGE_MARKERS)
    )


def _count_parameter_literals(sources: tuple[str, ...]) -> tuple[int, list[str]]:
    """Distinct values assigned to parameter-looking names across all cells.

    Assigning ``window = 20`` in one cell and ``window = 50`` in another is two
    trials, recorded in the notebook itself.
    """
    seen: dict[str, set] = {}
    for source in sources:
        try:
            tree = ast.parse(_strip_magics(source))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, (int, float)):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and any(
                    hint in target.id.lower() for hint in _PARAM_HINTS
                ):
                    seen.setdefault(target.id, set()).add(node.value.value)

    multi = {name: values for name, values in seen.items() if len(values) > 1}
    if not multi:
        return 1, []

    combinations = 1
    for values in multi.values():
        combinations *= len(values)
    detail = [f"{name} took {len(values)} values" for name, values in sorted(multi.items())]
    return combinations, detail


def _count_grid_sizes(sources: tuple[str, ...]) -> tuple[int, list[str]]:
    """Size of explicit grids: itertools.product, ParameterGrid, list literals."""
    total, details = 1, []
    for source in sources:
        try:
            tree = ast.parse(_strip_magics(source))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name not in _GRID_CALLS:
                continue
            size = 1
            found = False
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for literal in ast.walk(arg):
                    if isinstance(literal, (ast.List, ast.Tuple)) and literal.elts:
                        size *= len(literal.elts)
                        found = True
            if found:
                total *= size
                details.append(f"{name}() over {size} combinations")
    return total, details


def estimate_trials(
    notebook: NotebookScan | None = None,
    directory: str | Path | None = None,
    reported: int | None = None,
) -> TrialEstimate:
    """Combine every source of evidence into a lower bound on ``n_trials``.

    Sources are combined by taking the **maximum**, not the product. They
    overlap heavily - a grid search also inflates the execution count, and
    re-running it also leaves parameter literals - so multiplying them would
    manufacture a number the evidence does not support. Taking the maximum
    keeps the bound defensible, which is the only thing that makes it useful in
    an argument.
    """
    evidence: list[Evidence] = []

    if notebook is not None:
        extra = notebook.executions_beyond_cell_count
        if extra > 0:
            evidence.append(
                Evidence(
                    kind="execution_count",
                    detail=(
                        f"{notebook.n_code_cells} code cells but a highest execution count "
                        f"of {notebook.max_execution_count}, so cells were re-run at least "
                        f"{extra} times beyond a single clean pass"
                    ),
                    implied_trials=max(2, extra // max(notebook.n_code_cells, 1) + 1),
                )
            )
        if notebook.out_of_order > 0:
            evidence.append(
                Evidence(
                    kind="execution_order",
                    detail=(
                        f"{notebook.out_of_order} cells run out of order, so the saved "
                        "output does not correspond to any single top-to-bottom run"
                    ),
                    implied_trials=2,
                )
            )

        params, param_detail = _count_parameter_literals(notebook.source_by_cell)
        if params > 1:
            evidence.append(
                Evidence(
                    kind="parameter_literals",
                    detail="parameters reassigned across cells: " + "; ".join(param_detail),
                    implied_trials=params,
                )
            )

        grid, grid_detail = _count_grid_sizes(notebook.source_by_cell)
        if grid > 1:
            evidence.append(
                Evidence(
                    kind="explicit_grid",
                    detail="explicit search: " + "; ".join(grid_detail),
                    implied_trials=grid,
                )
            )

    if directory is not None:
        lineage = filename_lineage(directory)
        if len(lineage) > 1:
            evidence.append(
                Evidence(
                    kind="filename_lineage",
                    detail=(
                        f"{len(lineage)} notebooks whose names suggest successive attempts "
                        "at the same analysis"
                    ),
                    implied_trials=len(lineage),
                    locations=tuple(lineage),
                )
            )

    lower_bound = max((e.implied_trials for e in evidence), default=1)
    return TrialEstimate(
        lower_bound=lower_bound,
        evidence=tuple(sorted(evidence, key=lambda e: e.implied_trials, reverse=True)),
        reported=reported,
    )
