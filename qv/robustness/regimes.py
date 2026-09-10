"""Does the edge depend on market state?

One split, by realised volatility. Calendar and drawdown-state splits are
deliberately absent: rolling-origin analysis answers the calendar question
better, and drawdown state is both strongly correlated with volatility and
plainly visible on the equity curve, so a separate statistic for it would be
the same test wearing a different name.

The volatility split is computed from the strategy's own returns when no
external series is supplied, using a **trailing** window. That detail matters:
classifying a period as high-volatility using data from inside or after that
period would be exactly the look-ahead this package exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qv.stats.moments import as_returns_array
from qv.stats.sharpe import sharpe_ratio
from qv.types import Severity

__all__ = ["RegimeSplitResult", "trailing_volatility", "volatility_regime_split"]


def trailing_volatility(returns, window: int = 63, min_periods: int | None = None) -> np.ndarray:
    """Rolling standard deviation using only past data.

    Element ``t`` is the standard deviation of returns over ``[t-window, t)``,
    strictly excluding ``t`` itself. Leading elements with too little history
    are ``nan``. Excluding the current observation is what keeps the resulting
    classification usable as of time ``t``.
    """
    x = as_returns_array(returns)
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    floor = window if min_periods is None else min_periods
    if floor < 2:
        raise ValueError(f"min_periods must be at least 2, got {floor}")

    out = np.full(x.size, np.nan)
    for t in range(x.size):
        past = x[max(0, t - window) : t]
        if past.size >= floor:
            out[t] = np.std(past, ddof=1)
    return out


@dataclass(frozen=True)
class RegimeSplitResult:
    """Performance within each regime."""

    labels: tuple[str, ...]
    sharpes: tuple[float, ...]
    counts: tuple[int, ...]
    mean_returns: tuple[float, ...]
    full_sharpe: float
    n_classified: int
    n_obs: int
    threshold: float
    note: str | None = None

    def sharpe_in(self, label: str) -> float:
        for name, value in zip(self.labels, self.sharpes):
            if name == label:
                return value
        raise KeyError(f"{label!r} is not one of {self.labels}")

    @property
    def spread(self) -> float:
        return max(self.sharpes) - min(self.sharpes)

    @property
    def sign_stable(self) -> bool:
        """Do all regimes agree the edge has the same sign?"""
        signs = {s > 0 for s in self.sharpes}
        return len(signs) == 1

    @property
    def concentrated_in_one_regime(self) -> bool:
        """Is the edge positive in exactly one regime and not the others?"""
        return sum(s > 0 for s in self.sharpes) == 1 and len(self.sharpes) > 1

    @property
    def severity(self) -> Severity:
        if self.sign_stable:
            return Severity.INFO
        if self.concentrated_in_one_regime:
            return Severity.HIGH
        return Severity.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "sharpes": list(self.sharpes),
            "counts": list(self.counts),
            "mean_returns": list(self.mean_returns),
            "full_sharpe": self.full_sharpe,
            "spread": self.spread,
            "sign_stable": self.sign_stable,
            "concentrated_in_one_regime": self.concentrated_in_one_regime,
            "severity": self.severity.name.lower(),
            "threshold": self.threshold,
            "n_classified": self.n_classified,
            "n_obs": self.n_obs,
            "note": self.note,
        }


def volatility_regime_split(
    returns,
    volatility_series=None,
    window: int = 63,
    quantile: float = 0.5,
    min_regime_obs: int = 20,
) -> RegimeSplitResult:
    """Split returns into low- and high-volatility regimes and compare.

    ``volatility_series`` lets the caller supply an external measure - VIX, or
    the volatility of the underlying market rather than of the strategy. When
    omitted, trailing volatility of the strategy itself is used.

    The threshold is the ``quantile`` of the classifiable volatility values,
    so the split is balanced by construction rather than by a hard-coded level
    that would mean different things in different decades.
    """
    x = as_returns_array(returns)
    n = x.size
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be strictly between 0 and 1, got {quantile}")

    if volatility_series is None:
        vol = trailing_volatility(x, window=window)
    else:
        vol = np.asarray(
            getattr(volatility_series, "to_numpy", lambda: volatility_series)(), dtype=float
        )
        if vol.shape != x.shape:
            raise ValueError(
                f"volatility series has {vol.shape} but returns have {x.shape}; "
                "they must describe the same time axis"
            )

    usable = np.isfinite(vol)
    if usable.sum() < 2 * min_regime_obs:
        raise ValueError(
            f"only {int(usable.sum())} observations could be classified, which is too few "
            f"for two regimes of at least {min_regime_obs}; try a shorter window"
        )

    threshold = float(np.quantile(vol[usable], quantile))
    low = usable & (vol <= threshold)
    high = usable & (vol > threshold)

    notes = []
    if int(usable.sum()) < n:
        notes.append(
            f"{n - int(usable.sum())} leading observations had too little history to "
            "classify and are excluded"
        )

    labels, sharpes, counts, means = [], [], [], []
    for label, mask in (("low volatility", low), ("high volatility", high)):
        subset = x[mask]
        if subset.size < min_regime_obs:
            raise ValueError(
                f"the {label} regime holds only {subset.size} observations, "
                f"below the {min_regime_obs} required"
            )
        labels.append(label)
        sharpes.append(sharpe_ratio(subset))
        counts.append(int(subset.size))
        means.append(float(np.mean(subset)))

    return RegimeSplitResult(
        labels=tuple(labels),
        sharpes=tuple(sharpes),
        counts=tuple(counts),
        mean_returns=tuple(means),
        full_sharpe=sharpe_ratio(x),
        n_classified=int(usable.sum()),
        n_obs=n,
        threshold=threshold,
        note="; ".join(notes) if notes else None,
    )
