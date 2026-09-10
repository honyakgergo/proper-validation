"""Static scan for look-ahead and simulation anti-patterns.

An AST walk, not a regex sweep, so a match is a real call rather than a string
that happens to appear in a comment.

What this can and cannot do is worth stating plainly, because a scanner that
oversells itself is the same failure as a backtest that does. It finds
recognisable *shapes*: a negative shift, a backward fill, a scaler fitted
outside a fold. It cannot follow data through variables, so it cannot tell
whether a negative shift feeds a label (legitimate) or a feature (fatal), and
it cannot see inside a library call at all. Every finding here is a prompt to
look, not a verdict.

The behavioural test in :mod:`qv.leakage.perturbation` is what actually settles
the question. This module is the cheap first pass that tells you where to point
it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qv.findings import make_finding
from qv.types import Finding, Severity

__all__ = ["ScanHit", "ScanResult", "scan_source", "scan_file"]

#: Aggregations that look at every observation at once. Safe when only
#: reported, a leak when they set a threshold a signal compares against.
_FULL_SAMPLE_AGGREGATES = frozenset(
    {"mean", "std", "median", "quantile", "max", "min", "var", "sum"}
)

#: Window methods that make a following aggregate trailing rather than global.
_WINDOW_METHODS = frozenset({"rolling", "expanding", "ewm", "resample", "groupby"})

_RANDOM_RESAMPLERS = frozenset({"permutation", "choice", "shuffle"})

#: Names that suggest a variable holds returns or trades rather than anything
#: else one might legitimately permute.
_RETURNISH = ("return", "ret", "pnl", "profit", "trade", "equity", "curve")

_FORWARD_PROJECTION_HINTS = (
    "future",
    "forward",
    "project",
    "simulate_path",
    "simulated_path",
    "monte_carlo",
    "montecarlo",
    "fan_chart",
    "ruin",
)


@dataclass(frozen=True)
class ScanHit:
    """One pattern match, with enough context to check it by hand."""

    finding_id: str
    line: int
    column: int
    snippet: str
    cell: int | None = None

    @property
    def location(self) -> str:
        where = f"line {self.line}"
        return where if self.cell is None else f"cell {self.cell}, {where}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "line": self.line,
            "column": self.column,
            "cell": self.cell,
            "location": self.location,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class ScanResult:
    """Everything one static scan found."""

    hits: tuple[ScanHit, ...]
    source_name: str
    n_lines: int
    syntax_error: str | None = None

    @property
    def clean(self) -> bool:
        return not self.hits and self.syntax_error is None

    def hits_for(self, finding_id: str) -> tuple[ScanHit, ...]:
        return tuple(h for h in self.hits if h.finding_id == finding_id)

    def to_findings(self) -> list[Finding]:
        """Collapse hits into one finding per defect type, listing locations."""
        by_id: dict[str, list[ScanHit]] = {}
        for hit in self.hits:
            by_id.setdefault(hit.finding_id, []).append(hit)

        findings = []
        for finding_id, hits in by_id.items():
            locations = ", ".join(h.location for h in hits[:6])
            if len(hits) > 6:
                locations += f", and {len(hits) - 6} more"
            findings.append(
                make_finding(
                    finding_id,
                    detail=(
                        f"{len(hits)} occurrence{'s' if len(hits) > 1 else ''} in "
                        f"{self.source_name} at {locations}."
                    ),
                    evidence={
                        "count": len(hits),
                        "locations": [h.to_dict() for h in hits],
                        "source": self.source_name,
                    },
                )
            )
        return sorted(findings, key=lambda f: f.severity, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "n_lines": self.n_lines,
            "clean": self.clean,
            "syntax_error": self.syntax_error,
            "hits": [h.to_dict() for h in self.hits],
        }


def _attribute_chain(node: ast.AST) -> list[str]:
    """Dotted names in a call target, e.g. ``df.rolling(3).mean`` -> rolling, mean."""
    parts: list[str] = []
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Subscript):
            current = current.value
        elif isinstance(current, ast.Name):
            parts.append(current.id)
            break
        else:
            break
    return list(reversed(parts))


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_column_axis(node: ast.expr | None) -> bool:
    """Is this an explicit ``axis=1`` / ``axis="columns"``?

    Only an explicit declaration counts. The default is ``axis=0``, which is
    exactly the down-the-time-axis aggregate the finding is about, so silence
    must not be read as reassurance.
    """
    if not isinstance(node, ast.Constant):
        return False
    return node.value == 1 or node.value == "columns"


def _is_false(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is False


def _mentions(text: str, needles) -> bool:
    lowered = text.lower()
    return any(n in lowered for n in needles)


class _Scanner(ast.NodeVisitor):
    def __init__(self, source: str, cell: int | None) -> None:
        self.lines = source.splitlines()
        self.cell = cell
        self.hits: list[ScanHit] = []
        # Depth of enclosing for/while loops, used to tell a scaler fitted
        # inside a fold loop from one fitted once over everything.
        self._loop_depth = 0

    # -- helpers ---------------------------------------------------------
    def _record(self, finding_id: str, node: ast.AST) -> None:
        line = getattr(node, "lineno", 0)
        snippet = self.lines[line - 1].strip() if 0 < line <= len(self.lines) else ""
        self.hits.append(
            ScanHit(
                finding_id=finding_id,
                line=line,
                column=getattr(node, "col_offset", 0),
                snippet=snippet[:200],
                cell=self.cell,
            )
        )

    # -- traversal -------------------------------------------------------
    def visit_For(self, node: ast.For) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        chain = _attribute_chain(node.func)
        method = chain[-1] if chain else ""

        self._check_shift(node, method)
        self._check_backward_fill(node, method)
        self._check_centered_window(node, method)
        self._check_split(node, method)
        self._check_scaler_fit(node, method)
        self._check_full_sample_statistic(node, chain, method)
        self._check_random_resample(node, chain, method)
        self._check_forward_projection(node, chain, method)

        self.generic_visit(node)

    # -- individual detectors -------------------------------------------
    def _check_shift(self, node: ast.Call, method: str) -> None:
        if method != "shift":
            return
        args = list(node.args)
        periods = _keyword(node, "periods")
        if periods is not None:
            args.append(periods)
        for arg in args:
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                self._record("LEAK-NEGATIVE-SHIFT", node)
                return
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                self._record("LEAK-NEGATIVE-SHIFT", node)
                return

    def _check_backward_fill(self, node: ast.Call, method: str) -> None:
        if method in ("bfill", "backfill"):
            self._record("LEAK-BACKWARD-FILL", node)
            return
        if method == "fillna":
            arg = _keyword(node, "method")
            if isinstance(arg, ast.Constant) and arg.value in ("bfill", "backfill"):
                self._record("LEAK-BACKWARD-FILL", node)

    def _check_centered_window(self, node: ast.Call, method: str) -> None:
        if method in ("rolling", "ewm") and _is_true(_keyword(node, "center")):
            self._record("LEAK-CENTERED-WINDOW", node)

    def _check_split(self, node: ast.Call, method: str) -> None:
        if method == "train_test_split":
            # Shuffling is the default, so only an explicit shuffle=False is safe.
            if not _is_false(_keyword(node, "shuffle")):
                self._record("LEAK-SHUFFLED-SPLIT", node)
        elif method in ("KFold", "StratifiedKFold", "RepeatedKFold", "ShuffleSplit"):
            self._record("LEAK-SHUFFLED-SPLIT", node)

    def _check_scaler_fit(self, node: ast.Call, method: str) -> None:
        if method not in ("fit", "fit_transform"):
            return
        # Inside a loop it is plausibly per-fold, which is the correct pattern.
        if self._loop_depth == 0:
            self._record("LEAK-FULL-SAMPLE-SCALER", node)

    def _check_full_sample_statistic(
        self, node: ast.Call, chain: list[str], method: str
    ) -> None:
        if method not in _FULL_SAMPLE_AGGREGATES:
            return
        if any(part in _WINDOW_METHODS for part in chain[:-1]):
            return
        # An aggregate taken across columns cannot leak across time. Summing a
        # row of portfolio weights, or ranking a cross-section, reads one
        # timestamp and no other, so there is no future for it to reach into.
        # Without this the scanner raises a critical finding on the ordinary
        # `weights.sum(axis=1)` of any long-only book.
        if _is_column_axis(_keyword(node, "axis")):
            return
        # Require a subscript somewhere in the chain (df['col'].mean()) so bare
        # builtins like max(a, b) and np.mean on a local list do not fire.
        if not isinstance(node.func, ast.Attribute):
            return
        if not _contains_subscript(node.func):
            return
        self._record("LEAK-FULL-SAMPLE-STATISTIC", node)

    def _check_random_resample(self, node: ast.Call, chain: list[str], method: str) -> None:
        text = ".".join(chain)
        if method == "sample":
            frac = _keyword(node, "frac")
            if isinstance(frac, ast.Constant) and frac.value == 1:
                self._record("MC-IID-RESAMPLE", node)
            return
        if method in _RANDOM_RESAMPLERS and _mentions(text, ("random",)):
            target = node.args[0] if node.args else None
            target_text = ast.unparse(target) if target is not None else ""
            if _mentions(target_text, _RETURNISH):
                self._record("MC-IID-RESAMPLE", node)

    def _check_forward_projection(
        self, node: ast.Call, chain: list[str], method: str
    ) -> None:
        if method not in ("normal", "standard_normal", "choice", "multivariate_normal"):
            return
        line = self.lines[node.lineno - 1] if 0 < node.lineno <= len(self.lines) else ""
        if _mentions(line, _FORWARD_PROJECTION_HINTS):
            self._record("MC-FORWARD-PROJECTION", node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A function whose name promises a forward projection is worth flagging
        # even when the random draw inside it looks innocuous in isolation.
        if _mentions(node.name, _FORWARD_PROJECTION_HINTS) and _mentions(
            ast.unparse(node), ("random", "rvs")
        ):
            self._record("MC-FORWARD-PROJECTION", node)
        self.generic_visit(node)


def _contains_subscript(node: ast.AST) -> bool:
    current = node
    while True:
        if isinstance(current, ast.Subscript):
            return True
        if isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        else:
            return False


def scan_source(source: str, source_name: str = "<string>", cell: int | None = None) -> ScanResult:
    """Scan a block of Python source for known leakage patterns.

    A syntax error is reported rather than raised: notebooks routinely contain
    cells with shell escapes or incomplete code, and one bad cell must not
    abort the audit.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ScanResult(
            hits=(),
            source_name=source_name,
            n_lines=len(source.splitlines()),
            syntax_error=f"{exc.msg} at line {exc.lineno}",
        )

    scanner = _Scanner(source, cell)
    scanner.visit(tree)
    return ScanResult(
        hits=tuple(sorted(scanner.hits, key=lambda h: (h.cell or 0, h.line))),
        source_name=source_name,
        n_lines=len(source.splitlines()),
    )


def scan_file(path: str | Path) -> ScanResult:
    """Scan a ``.py`` file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    return scan_source(p.read_text(encoding="utf-8"), source_name=p.name)
