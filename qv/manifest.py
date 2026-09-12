"""``research_manifest.yaml``: what the researcher asserts, in a checked schema.

The manifest is the contract that makes an audit reproducible. It records the
things the data cannot tell you - how many configurations were really tried,
whether the universe was assembled with hindsight, which asset class the costs
should be judged against - and it pins the ones that would otherwise drift,
above all ``end_date``.

Validating it is worth a schema rather than a pile of ``dict.get`` calls for
one reason in particular: a field the tool silently ignores is worse than a
missing one. Both of the momentum manifests in this repository declared
``search.n_trials`` and ``trial_matrix_supplied``, and nothing read either -
the trial count was recomputed from the grid instead. A researcher who
overstated their trial count in good faith, exactly as the documentation asks
them to, would have had the honest number thrown away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qv.types import Suite

__all__ = [
    "PERIODS_PER_YEAR",
    "DataSpec",
    "SearchSpec",
    "StrategySpec",
    "ResearchManifest",
    "load_manifest",
]

#: Observations a year, by declared frequency. The audit needs this to
#: annualise, and guessing it from the index is how a monthly strategy gets a
#: Sharpe inflated by sqrt(21).
PERIODS_PER_YEAR = {
    "daily": 252,
    "weekly": 52,
    "monthly": 12,
    "quarterly": 4,
    "annual": 1,
}


class DataSpec(BaseModel):
    """Where the prices came from, and what was assumed about them."""

    model_config = ConfigDict(extra="forbid")

    #: The instruments to load, in the order the strategy expects its columns.
    #: ``ticker`` and ``tickers`` are accepted as aliases for a one-name or
    #: many-name universe, since that is how the existing manifests read.
    universe: list[str] = Field(default_factory=list)
    ticker: str | None = None
    tickers: list[str] | None = None

    benchmark: str | None = None
    start_date: str
    #: Pinned, deliberately. An open end date reproduces whatever the vendor
    #: returns today rather than the numbers that were published, which makes
    #: a "reproducible" report reproduce nothing.
    end_date: str
    source: str = "yfinance"
    asset_class: str | None = None
    frequency: str = "daily"
    price_column: str = "close"
    load_factors: bool = True
    #: A local CSV or Parquet in the documented wide schema, relative to the
    #: manifest. Given, no download happens and no vendor is privileged.
    price_frame: str | None = None

    #: The one question no tier can answer from the data *alone*. Left unset,
    #: the report lists it under *what could not be tested*, which is where an
    #: unanswered question belongs. Supplying `membership_frame` below replaces
    #: this declaration with a measurement, and contradicting it raises
    #: DATA-DECLARATION-CONTRADICTED.
    universe_point_in_time: bool | None = None
    universe_note: str | None = None
    #: A named membership source the tool knows how to fetch: `sp500` or
    #: `nasdaq100`. Fetched once and cached outside the repository, exactly as
    #: prices and factors are - the package ships the link and the parser, never
    #: the list.
    membership: str | None = None
    #: A local point-in-time membership list, relative to this file:
    #: ticker,start_date,end_date with an optional id and index. Use this for any
    #: index the named sources do not cover, or for better data than a free
    #: reconstruction.
    membership_frame: str | None = None
    #: Which index to keep when the file carries several. Required in that
    #: case - picking one silently would decide the answer for the reader.
    membership_index: str | None = None

    # Declared but not consumed by the engine; kept so a manifest can carry
    # them without being rejected.
    risk_assets: list[str] | None = None
    defensive_assets: list[str] | None = None

    @field_validator("frequency")
    @classmethod
    def _known_frequency(cls, value: str) -> str:
        if value not in PERIODS_PER_YEAR:
            raise ValueError(
                f"unknown frequency {value!r}; expected one of "
                f"{sorted(PERIODS_PER_YEAR)}"
            )
        return value

    @field_validator("end_date")
    @classmethod
    def _pinned(cls, value: str) -> str:
        if not value or value in ("today", "now", "None"):
            raise ValueError(
                "end_date must be pinned to a real date so results stay "
                "reproducible; an open end date reproduces whatever the vendor "
                "returns today rather than the numbers that were published"
            )
        return value

    @model_validator(mode="after")
    def _one_membership_source(self) -> "DataSpec":
        """Two lists would mean the tool picking one, and the choice decides the
        answer. Refuse rather than pick."""
        if self.membership and self.membership_frame:
            raise ValueError(
                "set either data.membership (a named source) or "
                "data.membership_frame (a local file), not both"
            )
        return self

    @model_validator(mode="after")
    def _resolve_universe(self) -> DataSpec:
        if not self.universe:
            if self.tickers:
                self.universe = list(self.tickers)
            elif self.ticker:
                self.universe = [self.ticker]
            elif self.risk_assets:
                self.universe = list(self.risk_assets) + list(
                    self.defensive_assets or []
                )
        if not self.universe:
            raise ValueError(
                "data must name the instruments to load, via `universe` "
                "(or `ticker` / `tickers`)"
            )
        duplicates = {t for t in self.universe if self.universe.count(t) > 1}
        if duplicates:
            raise ValueError(f"the universe repeats {sorted(duplicates)}")
        return self

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.frequency]


class SearchSpec(BaseModel):
    """Every configuration examined, including the ones that were discarded."""

    model_config = ConfigDict(extra="allow")

    #: Parameter name to the values tried, e.g. ``{"lookback": [126, 252]}``.
    #: Named explicitly rather than by pluralising the parameter, so the
    #: mapping from a grid axis to a keyword argument is not a guess.
    axes: dict[str, list[Any]] = Field(default_factory=dict)

    #: The declared count. Used when it exceeds the grid, because a grid is a
    #: lower bound on a search - it cannot include the configurations that were
    #: abandoned before anyone wrote them down. Overstating is the honest
    #: direction: the correction is logarithmic in the count, so honesty is
    #: cheap.
    n_trials: int | None = None
    note: str | None = None

    @field_validator("n_trials")
    @classmethod
    def _positive(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError(f"n_trials must be at least 1, got {value}")
        return value

    @field_validator("axes")
    @classmethod
    def _non_empty_axes(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        for name, values in value.items():
            if not values:
                raise ValueError(f"search axis {name!r} has no values")
        return value

    def declared_trials(self, grid_size: int) -> int | None:
        """The trial count to audit against: the larger of grid and declaration."""
        if self.n_trials is None:
            return grid_size or None
        return max(self.n_trials, grid_size)


class StrategySpec(BaseModel):
    """How to re-run the strategy, which is what the engine analysis needs."""

    model_config = ConfigDict(extra="forbid")

    #: ``path/to/file.py:function``, resolved relative to the manifest.
    adapter: str | None = None
    #: Parameters that were never searched, passed to every call.
    fixed: dict[str, Any] = Field(default_factory=dict)


class ResearchManifest(BaseModel):
    """The whole file."""

    model_config = ConfigDict(extra="allow")

    name: str
    data: DataSpec
    chosen_parameters: dict[str, Any] = Field(default_factory=dict)
    search: SearchSpec = Field(default_factory=SearchSpec)
    strategy: StrategySpec = Field(default_factory=StrategySpec)
    seed: int = 0
    #: Which questions to ask by default. Overridable per run, so the same
    #: manifest can produce a statistical report, an engine report and a full
    #: one without being edited between them.
    suite: Suite = Suite.FULL

    #: Free-form documentation. Carried so the file stays a single record of
    #: the research, and deliberately unvalidated - the tool has no business
    #: telling a researcher how to write their notes.
    process: dict[str, Any] | None = None
    expectations: list[Any] | None = None
    ground_truth: dict[str, Any] | None = None
    provenance: list[Any] | None = None

    #: Where the file was read from, so relative adapter paths resolve.
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _chosen_covers_every_axis(self) -> ResearchManifest:
        missing = [a for a in self.search.axes if a not in self.chosen_parameters]
        if missing:
            raise ValueError(
                f"chosen_parameters does not declare {missing}, which search.axes "
                "varies; the audit has to know which point in the grid is the "
                "one being reported"
            )
        return self

    @property
    def periods_per_year(self) -> int:
        return self.data.periods_per_year

    def adapter_reference(self) -> str | None:
        return self.strategy.adapter

    def resolve(self, relative: str) -> Path:
        """A path in the manifest, relative to the manifest itself."""
        base = self.source_path.parent if self.source_path else Path.cwd()
        return (base / relative).resolve()


def load_manifest(path: str | Path) -> ResearchManifest:
    """Read and validate a ``research_manifest.yaml``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such manifest: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a YAML mapping")
    manifest = ResearchManifest.model_validate(payload)
    manifest.source_path = path.resolve()
    return manifest
