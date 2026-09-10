"""Shared vocabulary for the whole package.

Deliberately dependency-light: only the standard library. Every statistical
module imports from here, so pulling pydantic or pandas in at this level would
make the numeric core impossible to test in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

__all__ = [
    "Tier",
    "Suite",
    "Severity",
    "Verdict",
    "Estimate",
    "Finding",
    "MIN_OBS_FOR_ASYMPTOTICS",
]


#: Below this many observations the asymptotic results used throughout this
#: package (Lo's Sharpe standard error, the DSR normal approximation, the
#: bootstrap's own consistency) stop being trustworthy. Functions guard on it
#: and mark their output unreliable rather than printing a confident number.
MIN_OBS_FOR_ASYMPTOTICS = 30


class Tier(IntEnum):
    """How much the researcher handed over, and therefore what can be tested.

    An ordering of input completeness, not a quality judgement: an audit given
    only a return series is not a worse audit, it is a narrower one.

    This is deliberately *not* what the report leads with. It used to be, under
    the name "Tier 2", and it competed with the far more useful question of
    which suites were run - a reader had two ladders to reconcile and no
    guidance on how they related. What survives is the ordering itself, which
    is what gates individual tests and populates *what could not be tested*.
    """

    RETURNS = 0
    POSITIONS = 1
    CALLABLE = 2

    @property
    def label(self) -> str:
        return {
            Tier.RETURNS: "a return series",
            Tier.POSITIONS: "returns, positions and prices",
            Tier.CALLABLE: "returns, positions and a re-runnable strategy",
        }[self]


class Suite(str, Enum):
    """Which of the two questions an audit is asking.

    They are genuinely different questions and used to be muddled into levels
    of one ladder, which hid that. Running one does not weaken the other; a
    strategy can be statistically hopeless and impeccably implemented, or the
    reverse - and the reverse is the more dangerous case, because the numbers
    look good.
    """

    #: Is the measured edge distinguishable from luck, and does it survive
    #: honest accounting? Significance, selection bias, costs, factor
    #: exposure, and the robustness of the *edge*.
    STATISTICAL = "statistical"

    #: Can the backtest be trusted as an implementation, whatever its numbers
    #: say? Look-ahead in behaviour, survivorship in the universe, data
    #: quality, determinism, and the fragility of the *code*.
    ENGINE = "engine"

    #: Both.
    FULL = "full"

    @property
    def label(self) -> str:
        return {
            Suite.STATISTICAL: "Statistical validation",
            Suite.ENGINE: "Engine analysis",
            Suite.FULL: "Statistical validation and engine analysis",
        }[self]

    @property
    def question(self) -> str:
        """What this suite is trying to falsify, in one sentence."""
        return {
            Suite.STATISTICAL: (
                "Is the measured edge distinguishable from luck, and does it "
                "survive honest accounting?"
            ),
            Suite.ENGINE: (
                "Can this backtest be trusted as an implementation, whatever "
                "its numbers say?"
            ),
            Suite.FULL: (
                "Is the measured edge real, and is the backtest that measured "
                "it trustworthy?"
            ),
        }[self]

    @property
    def runs_statistical(self) -> bool:
        return self in (Suite.STATISTICAL, Suite.FULL)

    @property
    def runs_engine(self) -> bool:
        return self in (Suite.ENGINE, Suite.FULL)


class Severity(IntEnum):
    """Ordered so findings sort by severity descending with a plain sort."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()


class Verdict(str, Enum):
    """The headline conclusion.

    Note the asymmetry, which is the whole point of the tool: there is no
    ``PASSED`` member. The best available outcome is that the audit failed to
    falsify the backtest, which is not the same as evidence that it works.
    """

    NOT_FALSIFIED = "not_falsified"
    WEAKENED = "weakened"
    FALSIFIED = "falsified"
    INSUFFICIENT_DATA = "insufficient_data"

    @property
    def label(self) -> str:
        return {
            Verdict.NOT_FALSIFIED: "Survived the tests applied",
            Verdict.WEAKENED: "Materially weakened",
            Verdict.FALSIFIED: "Falsified",
            Verdict.INSUFFICIENT_DATA: "Insufficient data to conclude",
        }[self]


@dataclass(frozen=True)
class Estimate:
    """A number plus everything needed to distrust it.

    ``reliable=False`` means the estimate was computed but its assumptions do
    not hold here - almost always a small sample. Consumers must surface the
    ``note`` rather than printing ``value`` bare.
    """

    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    method: str = ""
    n_obs: int | None = None
    reliable: bool = True
    note: str | None = None

    @property
    def has_ci(self) -> bool:
        return self.ci_low is not None and self.ci_high is not None

    @property
    def is_finite(self) -> bool:
        return math.isfinite(self.value)

    def excludes(self, x: float) -> bool:
        """Does the confidence interval exclude ``x``? False if there is no CI."""
        if not self.has_ci:
            return False
        assert self.ci_low is not None and self.ci_high is not None
        return x < self.ci_low or x > self.ci_high

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _jsonable(self.value),
            "ci_low": _jsonable(self.ci_low),
            "ci_high": _jsonable(self.ci_high),
            "method": self.method,
            "n_obs": self.n_obs,
            "reliable": self.reliable,
            "note": self.note,
        }


@dataclass(frozen=True)
class Finding:
    """One defect, tied to a catalog entry.

    ``evidence`` holds the numbers that triggered it so a reader can check the
    call rather than take it on faith.
    """

    id: str
    title: str
    severity: Severity
    detail: str
    remediation: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.name.lower(),
            "severity_rank": int(self.severity),
            "detail": self.detail,
            "remediation": self.remediation,
            "evidence": {k: _jsonable(v) for k, v in self.evidence.items()},
        }


def _jsonable(x: Any) -> Any:
    """NaN and infinity are not valid JSON; emit null instead of a parse error."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
