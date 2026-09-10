"""Tests for the findings catalog.

The catalog is the single source of truth for what the tool can name, and the
markdown reference is generated from it - so these tests are mostly about the
invariants that keep documentation from drifting away from behaviour.
"""

from __future__ import annotations

import re

import pytest

from qv.findings import CATALOG, catalog_markdown, get_entry, make_finding
from qv.types import Severity


class TestCatalogIntegrity:
    def test_is_not_padded(self):
        """Twenty real entries beat forty where half are rephrasings. A reviewer
        who reads the catalog notices padding, and padding is the exact sin
        this package exists to detect.

        The ceiling moved from 30 to 35 when the engine suite added five
        entries - non-determinism, an uninvested book, a degenerate signal,
        execution-delay fragility and inclusion timing. Each names a defect no
        other entry covers, which is the standard; the bound is a guard against
        rephrasings, not against the catalog growing when the tool learns to
        detect something new."""
        assert 15 <= len(CATALOG) <= 35

    def test_ids_are_unique_and_well_formed(self):
        for key, entry in CATALOG.items():
            assert key == entry.id
            assert re.fullmatch(r"[A-Z]+(-[A-Z0-9]+)+", entry.id), entry.id

    def test_every_entry_is_complete(self):
        for entry in CATALOG.values():
            assert entry.title and not entry.title.endswith(".")
            assert len(entry.explanation) > 80, f"{entry.id} explanation is too thin"
            assert entry.detection.endswith("."), entry.id
            assert len(entry.remediation) > 40, f"{entry.id} has no real remediation"

    def test_every_entry_has_a_category(self):
        categories = {e.category for e in CATALOG.values()}
        assert categories <= {
            "leakage", "selection", "statistics", "costs", "attribution",
            "robustness", "data", "engine",
        }

    def test_severities_are_real(self):
        for entry in CATALOG.values():
            assert isinstance(entry.severity, Severity)

    def test_no_duplicate_titles(self):
        """Two entries with the same title are the same finding twice."""
        titles = [e.title.lower() for e in CATALOG.values()]
        assert len(titles) == len(set(titles))

    def test_the_two_monte_carlo_antipatterns_are_present(self):
        """Refusing to build these and flagging them instead is a deliberate
        stance, so it must survive in the catalog."""
        assert "MC-IID-RESAMPLE" in CATALOG
        assert "MC-FORWARD-PROJECTION" in CATALOG

    def test_ascii_only(self):
        """Reports render in many contexts; a stray non-ASCII character in a
        catalog entry shows up as a mojibake box in at least one of them."""
        for entry in CATALOG.values():
            blob = entry.title + entry.explanation + entry.detection + entry.remediation
            assert blob.isascii(), f"{entry.id} contains non-ASCII text"


class TestGetEntry:
    def test_looks_up_by_id(self):
        assert get_entry("LEAK-NEGATIVE-SHIFT").severity is Severity.CRITICAL

    def test_suggests_a_near_miss(self):
        with pytest.raises(KeyError, match="did you mean"):
            get_entry("LEAK-NEGATIVE")

    def test_plain_failure_for_nonsense(self):
        with pytest.raises(KeyError, match="not in the findings catalog"):
            get_entry("NOT-A-THING-AT-ALL")


class TestMakeFinding:
    def test_inherits_catalog_fields(self):
        f = make_finding("LEAK-BACKWARD-FILL", detail="found at line 12")
        assert f.id == "LEAK-BACKWARD-FILL"
        assert f.severity is Severity.HIGH
        assert f.remediation == CATALOG["LEAK-BACKWARD-FILL"].remediation

    def test_severity_can_be_overridden(self):
        """The same defect is worse in some contexts than others - a deflation
        at 0.2 is not a deflation at 0.94."""
        f = make_finding("LEAK-BACKWARD-FILL", "x", severity=Severity.LOW)
        assert f.severity is Severity.LOW

    def test_evidence_is_carried(self):
        f = make_finding("LEAK-BACKWARD-FILL", "x", evidence={"count": 3})
        assert f.evidence["count"] == 3

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            make_finding("MADE-UP-ID", "x")

    def test_to_dict_is_json_shaped(self):
        d = make_finding("LEAK-BACKWARD-FILL", "x", evidence={"n": 1}).to_dict()
        assert set(d) >= {"id", "title", "severity", "detail", "remediation", "evidence"}
        assert d["severity"] == "high"


class TestCatalogMarkdown:
    def test_includes_every_entry(self):
        md = catalog_markdown()
        for entry in CATALOG.values():
            assert f"`{entry.id}`" in md

    def test_states_that_it_is_generated(self):
        """Otherwise someone edits the markdown and the drift begins."""
        assert "Generated from" in catalog_markdown()

    def test_groups_by_category(self):
        md = catalog_markdown()
        for category in {e.category for e in CATALOG.values()}:
            assert f"## {category.capitalize()}" in md

    def test_carries_remediation(self):
        assert catalog_markdown().count("**Remediation.**") == len(CATALOG)
