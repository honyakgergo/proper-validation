"""Data-quality checks.

A validator built on bad data would be self-refuting, so the tool inspects its
inputs before drawing conclusions from them. A single unadjusted 2:1 split is a
-50% day, and a mean-reversion strategy will happily trade it and report an
excellent Sharpe.

Pure functions over a frame or series. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from qv.findings import make_finding
from qv.types import Finding, Severity

__all__ = ["QualityReport", "check_prices", "check_returns"]


@dataclass(frozen=True)
class QualityReport:
    """What is wrong with the data, before anything is concluded from it."""

    n_rows: int
    n_missing: int
    n_duplicate_dates: int
    n_zero_volume: int
    largest_gap_days: int
    extreme_moves: tuple[str, ...]
    suspected_splits: tuple[str, ...]
    non_monotonic_index: bool

    @property
    def clean(self) -> bool:
        return not (
            self.n_missing
            or self.n_duplicate_dates
            or self.extreme_moves
            or self.suspected_splits
            or self.non_monotonic_index
        )

    def to_findings(self) -> list[Finding]:
        findings: list[Finding] = []
        problems = []
        if self.n_missing:
            problems.append(f"{self.n_missing} missing values")
        if self.n_duplicate_dates:
            problems.append(f"{self.n_duplicate_dates} duplicated dates")
        if self.non_monotonic_index:
            problems.append("the index is not sorted")
        if self.largest_gap_days > 10:
            problems.append(f"a {self.largest_gap_days}-day gap")
        if self.n_zero_volume:
            problems.append(f"{self.n_zero_volume} zero-volume days")
        if self.extreme_moves:
            problems.append(f"{len(self.extreme_moves)} moves beyond 10 standard deviations")

        if problems:
            findings.append(
                make_finding(
                    "DATA-QUALITY-GAPS",
                    detail="Data quality: " + "; ".join(problems) + ".",
                    severity=Severity.HIGH if self.suspected_splits else Severity.MEDIUM,
                    evidence=self.to_dict(),
                )
            )
        if self.suspected_splits:
            findings.append(
                make_finding(
                    "DATA-QUALITY-GAPS",
                    detail=(
                        f"{len(self.suspected_splits)} days move by close to a round "
                        f"split ratio ({', '.join(self.suspected_splits[:5])}), which is the "
                        "signature of an unadjusted corporate action rather than a price move."
                    ),
                    severity=Severity.HIGH,
                    evidence={"dates": list(self.suspected_splits)},
                )
            )
        return findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_missing": self.n_missing,
            "n_duplicate_dates": self.n_duplicate_dates,
            "n_zero_volume": self.n_zero_volume,
            "largest_gap_days": self.largest_gap_days,
            "extreme_moves": list(self.extreme_moves),
            "suspected_splits": list(self.suspected_splits),
            "non_monotonic_index": self.non_monotonic_index,
            "clean": self.clean,
        }


def _suspected_splits(returns: pd.Series) -> tuple[str, ...]:
    """Days whose move is close to a common split or reverse-split ratio."""
    ratios = {
        "2:1": -0.50, "3:1": -2 / 3, "3:2": -1 / 3, "4:1": -0.75,
        "1:2": 1.0, "1:3": 2.0,
    }
    out = []
    for stamp, value in returns.items():
        for label, target in ratios.items():
            if abs(value - target) < 0.015:
                out.append(f"{_fmt(stamp)} ({value:+.1%}, looks like {label})")
                break
    return tuple(out)


def _fmt(stamp) -> str:
    return stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp)


def check_prices(prices: pd.DataFrame, close_column: str = "close") -> QualityReport:
    """Inspect a price frame for gaps, duplicates and adjustment artifacts."""
    if close_column not in prices.columns:
        raise ValueError(
            f"no {close_column!r} column; found {list(prices.columns)}"
        )
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("price frame must be indexed by date")

    close = prices[close_column]
    returns = close.pct_change().dropna()

    gaps = prices.index.to_series().diff().dt.days.dropna()
    volume = prices["volume"] if "volume" in prices.columns else None

    sd = float(returns.std())
    extreme = (
        returns[np.abs(returns - returns.mean()) > 10 * sd] if sd > 0 else returns.iloc[:0]
    )

    return QualityReport(
        n_rows=int(len(prices)),
        n_missing=int(prices.isna().sum().sum()),
        n_duplicate_dates=int(prices.index.duplicated().sum()),
        n_zero_volume=int((volume == 0).sum()) if volume is not None else 0,
        largest_gap_days=int(gaps.max()) if len(gaps) else 0,
        extreme_moves=tuple(f"{_fmt(i)} ({v:+.1%})" for i, v in extreme.items()),
        suspected_splits=_suspected_splits(returns),
        non_monotonic_index=not prices.index.is_monotonic_increasing,
    )


def check_returns(returns: pd.Series | np.ndarray) -> QualityReport:
    """The same inspection for a bare return series, where less is knowable."""
    series = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns))
    clean = series.dropna()
    sd = float(clean.std())
    extreme = clean[np.abs(clean - clean.mean()) > 10 * sd] if sd > 0 else clean.iloc[:0]

    gaps = pd.Series(dtype=float)
    if isinstance(series.index, pd.DatetimeIndex):
        gaps = series.index.to_series().diff().dt.days.dropna()

    return QualityReport(
        n_rows=int(len(series)),
        n_missing=int(series.isna().sum()),
        n_duplicate_dates=int(series.index.duplicated().sum()),
        n_zero_volume=0,
        largest_gap_days=int(gaps.max()) if len(gaps) else 0,
        extreme_moves=tuple(f"{_fmt(i)} ({v:+.1%})" for i, v in extreme.items()),
        suspected_splits=_suspected_splits(clean),
        non_monotonic_index=(
            isinstance(series.index, pd.DatetimeIndex)
            and not series.index.is_monotonic_increasing
        ),
    )
