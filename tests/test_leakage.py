"""Tests for the leakage layer: static scan, notebook archaeology, perturbation."""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from qv.findings import CATALOG
from qv.leakage.notebook import (
    estimate_trials,
    filename_lineage,
    parse_notebook,
)
from qv.leakage.perturbation import perturbation_test
from qv.leakage.static import scan_file, scan_source
from qv.types import Severity


def _ids(result) -> set[str]:
    return {hit.finding_id for hit in result.hits}


class TestStaticScannerDetections:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("y = df['close'].shift(-1)", "LEAK-NEGATIVE-SHIFT"),
            ("y = df['close'].shift(periods=-3)", "LEAK-NEGATIVE-SHIFT"),
            ("df = df.bfill()", "LEAK-BACKWARD-FILL"),
            ("df = df.backfill()", "LEAK-BACKWARD-FILL"),
            ("df = df.fillna(method='bfill')", "LEAK-BACKWARD-FILL"),
            ("s = df['x'].rolling(10, center=True).mean()", "LEAK-CENTERED-WINDOW"),
            ("a, b = train_test_split(X, y)", "LEAK-SHUFFLED-SPLIT"),
            ("cv = KFold(n_splits=5)", "LEAK-SHUFFLED-SPLIT"),
            ("X = scaler.fit_transform(features)", "LEAK-FULL-SAMPLE-SCALER"),
            ("threshold = df['ret'].quantile(0.9)", "LEAK-FULL-SAMPLE-STATISTIC"),
            # axis=0 is the default and *is* the down-the-time-axis aggregate
            # the finding is about, so it must still fire when stated.
            ("mu = df['ret'].mean(axis=0)", "LEAK-FULL-SAMPLE-STATISTIC"),
            ("shuffled = returns.sample(frac=1)", "MC-IID-RESAMPLE"),
            ("paths = np.random.permutation(trade_returns)", "MC-IID-RESAMPLE"),
        ],
    )
    def test_detects(self, code, expected):
        assert expected in _ids(scan_source(code))

    @pytest.mark.parametrize(
        "code",
        [
            "y = df['close'].shift(1)",
            "y = df['close'].shift()",
            "df = df.ffill()",
            "df = df.fillna(method='ffill')",
            "df = df.fillna(0)",
            "s = df['x'].rolling(10).mean()",
            "s = df['x'].rolling(10, center=False).mean()",
            "a, b = train_test_split(X, y, shuffle=False)",
            "cv = TimeSeriesSplit(n_splits=5)",
            "s = df['x'].expanding().mean()",
            "s = df['x'].rolling(20).std()",
            "biggest = max(a, b)",
            "sub = df.sample(n=10)",
            # Aggregates across columns read one timestamp and no other, so
            # they cannot reach into the future. Every long-only book computes
            # its gross exposure this way.
            "gross = weights[names].sum(axis=1)",
            "gross = weights[names].sum(axis='columns')",
            "best = scores[names].max(axis=1)",
        ],
    )
    def test_does_not_fire_on_correct_code(self, code):
        """False positives are expensive: a scanner that cries wolf gets muted,
        and then it catches nothing."""
        assert scan_source(code).hits == ()

    def test_scaler_inside_a_fold_loop_is_allowed(self):
        """Fitting per fold is the correct pattern and must not be flagged."""
        code = textwrap.dedent(
            """
            for train_idx, test_idx in TimeSeriesSplit(5).split(X):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
            """
        )
        assert "LEAK-FULL-SAMPLE-SCALER" not in _ids(scan_source(code))

    def test_scaler_outside_a_loop_is_flagged(self):
        code = "scaler = StandardScaler()\nX_all = scaler.fit_transform(X)"
        assert "LEAK-FULL-SAMPLE-SCALER" in _ids(scan_source(code))

    def test_forward_projection_by_function_name(self):
        code = textwrap.dedent(
            """
            def simulate_future_paths(mu, sigma, n):
                return np.random.normal(mu, sigma, n).cumsum()
            """
        )
        assert "MC-FORWARD-PROJECTION" in _ids(scan_source(code))

    def test_ordinary_random_draw_is_not_a_forward_projection(self):
        assert scan_source("noise = np.random.normal(0, 1, 100)").hits == ()

    def test_every_reported_id_exists_in_the_catalog(self):
        """A scanner that emits an id with no catalog entry cannot explain
        itself, which defeats the point of reporting it."""
        code = textwrap.dedent(
            """
            y = df['c'].shift(-1)
            df = df.bfill()
            s = df['x'].rolling(3, center=True).mean()
            a, b = train_test_split(X, y)
            X = scaler.fit_transform(F)
            t = df['ret'].quantile(0.9)
            z = returns.sample(frac=1)
            """
        )
        for hit in scan_source(code).hits:
            assert hit.finding_id in CATALOG


class TestScanResult:
    def test_reports_line_numbers_and_snippets(self):
        result = scan_source("a = 1\nb = df['c'].shift(-1)\n")
        hit = result.hits[0]
        assert hit.line == 2
        assert "shift(-1)" in hit.snippet

    def test_syntax_errors_are_reported_not_raised(self):
        """One unparseable notebook cell must not abort the whole audit."""
        result = scan_source("this is not python(")
        assert result.syntax_error is not None
        assert not result.clean
        assert result.hits == ()

    def test_clean_code_is_clean(self):
        assert scan_source("x = 1 + 1").clean

    def test_hits_for_filters_by_id(self):
        result = scan_source("a = df['x'].shift(-1)\nb = df['y'].shift(-2)\nc = df.bfill()")
        assert len(result.hits_for("LEAK-NEGATIVE-SHIFT")) == 2
        assert len(result.hits_for("LEAK-BACKWARD-FILL")) == 1

    def test_findings_collapse_by_type(self):
        result = scan_source("a = df['x'].shift(-1)\nb = df['y'].shift(-1)")
        findings = result.to_findings()
        assert len(findings) == 1
        assert findings[0].evidence["count"] == 2
        assert "2 occurrences" in findings[0].detail

    def test_findings_are_sorted_by_severity(self):
        result = scan_source("t = df['r'].quantile(0.9)\ny = df['c'].shift(-1)")
        findings = result.to_findings()
        assert findings[0].severity == Severity.CRITICAL
        assert findings[0].severity >= findings[-1].severity

    def test_findings_carry_remediation(self):
        finding = scan_source("y = df['c'].shift(-1)").to_findings()[0]
        assert finding.remediation
        assert "label" in finding.remediation

    def test_many_locations_are_truncated(self):
        result = scan_source("\n".join(f"a{i} = df['x'].shift(-1)" for i in range(10)))
        assert "and 4 more" in result.to_findings()[0].detail

    def test_to_dict_is_json_shaped(self):
        d = scan_source("y = df['c'].shift(-1)").to_dict()
        assert set(d) >= {"hits", "clean", "source_name"}
        json.dumps(d)

    def test_scan_file_reads_from_disk(self, tmp_path):
        p = tmp_path / "strategy.py"
        p.write_text("signal = df['close'].shift(-1)\n", encoding="utf-8")
        result = scan_file(p)
        assert result.source_name == "strategy.py"
        assert "LEAK-NEGATIVE-SHIFT" in _ids(result)

    def test_scan_file_rejects_a_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_file(tmp_path / "nope.py")


def _notebook(cells) -> dict:
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _code_cell(source: str, execution_count: int | None = 1) -> dict:
    return {
        "cell_type": "code",
        "source": source,
        "execution_count": execution_count,
        "outputs": [],
        "metadata": {},
    }


@pytest.fixture
def notebook_path(tmp_path):
    def _write(cells, name="analysis.ipynb"):
        p = tmp_path / name
        p.write_text(json.dumps(_notebook(cells)), encoding="utf-8")
        return p

    return _write


class TestNotebookParsing:
    def test_counts_cells(self, notebook_path):
        p = notebook_path(
            [
                _code_cell("x = 1", 1),
                {"cell_type": "markdown", "source": "# notes", "metadata": {}},
                _code_cell("y = 2", 2),
            ]
        )
        scan = parse_notebook(p)
        assert scan.n_cells == 3
        assert scan.n_code_cells == 2

    def test_scans_each_code_cell_for_leakage(self, notebook_path):
        p = notebook_path([_code_cell("y = df['c'].shift(-1)", 1)])
        scan = parse_notebook(p)
        assert any(h.finding_id == "LEAK-NEGATIVE-SHIFT" for h in scan.all_hits)

    def test_hits_record_their_cell(self, notebook_path):
        p = notebook_path([_code_cell("x = 1", 1), _code_cell("y = df['c'].shift(-1)", 2)])
        hit = parse_notebook(p).all_hits[0]
        assert hit.cell == 1
        assert "cell 1" in hit.location

    def test_magics_do_not_break_parsing(self, notebook_path):
        """A cell with %matplotlib is not valid Python but must still scan."""
        p = notebook_path([_code_cell("%matplotlib inline\n!pip install x\ny = df['c'].shift(-1)")])
        scan = parse_notebook(p)
        assert any(h.finding_id == "LEAK-NEGATIVE-SHIFT" for h in scan.all_hits)

    def test_source_as_a_list_of_lines_is_handled(self, notebook_path):
        p = notebook_path([{"cell_type": "code", "source": ["a = 1\n", "b = 2\n"],
                            "execution_count": 1, "outputs": [], "metadata": {}}])
        assert parse_notebook(p).source_by_cell[0] == "a = 1\nb = 2\n"

    def test_unexecuted_cells_are_skipped_in_counts(self, notebook_path):
        p = notebook_path([_code_cell("x = 1", None), _code_cell("y = 2", 5)])
        assert parse_notebook(p).execution_counts == (5,)

    def test_detects_out_of_order_execution(self, notebook_path):
        p = notebook_path([_code_cell("a", 7), _code_cell("b", 3), _code_cell("c", 9)])
        assert parse_notebook(p).out_of_order == 1

    def test_counts_reruns_beyond_a_clean_pass(self, notebook_path):
        p = notebook_path([_code_cell("a", 1), _code_cell("b", 2), _code_cell("c", 45)])
        scan = parse_notebook(p)
        assert scan.max_execution_count == 45
        assert scan.executions_beyond_cell_count == 42

    def test_to_findings_merges_across_cells(self, notebook_path):
        p = notebook_path(
            [_code_cell("a = df['x'].shift(-1)", 1), _code_cell("b = df['y'].shift(-1)", 2)]
        )
        findings = parse_notebook(p).to_findings()
        assert len(findings) == 1
        assert findings[0].evidence["count"] == 2

    def test_to_dict_is_json_shaped(self, notebook_path):
        p = notebook_path([_code_cell("y = df['c'].shift(-1)", 1)])
        json.dumps(parse_notebook(p).to_dict())

    def test_rejects_a_missing_notebook(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_notebook(tmp_path / "nope.ipynb")

    def test_rejects_invalid_json(self, tmp_path):
        p = tmp_path / "bad.ipynb"
        p.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_notebook(p)

    def test_rejects_json_that_is_not_a_notebook(self, tmp_path):
        p = tmp_path / "other.ipynb"
        p.write_text('{"foo": 1}', encoding="utf-8")
        with pytest.raises(ValueError, match="no 'cells' key"):
            parse_notebook(p)


class TestFilenameLineage:
    def test_finds_successive_attempts(self, tmp_path):
        for name in ("model.ipynb", "model_v2.ipynb", "model_final_fixed.ipynb", "eda.ipynb"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        found = filename_lineage(tmp_path)
        assert "model_v2.ipynb" in found
        assert "model_final_fixed.ipynb" in found
        assert "eda.ipynb" not in found

    def test_rejects_a_non_directory(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            filename_lineage(p)


class TestEstimateTrials:
    def test_a_clean_notebook_implies_one_trial(self, notebook_path):
        p = notebook_path([_code_cell("x = 1", 1), _code_cell("y = 2", 2)])
        est = estimate_trials(parse_notebook(p))
        assert est.lower_bound == 1
        assert est.to_finding() is None

    def test_reassigned_parameters_count_as_trials(self, notebook_path):
        p = notebook_path(
            [
                _code_cell("window = 10", 1),
                _code_cell("window = 20", 2),
                _code_cell("window = 50", 3),
            ]
        )
        est = estimate_trials(parse_notebook(p))
        assert est.lower_bound >= 3
        assert any(e.kind == "parameter_literals" for e in est.evidence)

    def test_multiplies_across_distinct_parameters(self, notebook_path):
        p = notebook_path(
            [
                _code_cell("window = 10\nthreshold = 1.0", 1),
                _code_cell("window = 20\nthreshold = 2.0", 2),
            ]
        )
        est = estimate_trials(parse_notebook(p))
        assert est.lower_bound >= 4

    def test_explicit_grid_is_counted(self, notebook_path):
        p = notebook_path(
            [_code_cell("grid = list(itertools.product([1, 2, 3], [10, 20, 30, 40]))", 1)]
        )
        est = estimate_trials(parse_notebook(p))
        assert est.lower_bound >= 12
        assert any(e.kind == "explicit_grid" for e in est.evidence)

    def test_evidence_is_combined_by_maximum_not_product(self, notebook_path):
        """Sources overlap heavily - a grid search also inflates the execution
        count - so multiplying them would manufacture a bound the evidence does
        not support."""
        p = notebook_path(
            [
                _code_cell("grid = list(itertools.product([1, 2, 3], [10, 20]))", 90),
                _code_cell("window = 10", 91),
                _code_cell("window = 20", 92),
            ]
        )
        est = estimate_trials(parse_notebook(p))
        implied = [e.implied_trials for e in est.evidence]
        assert est.lower_bound == max(implied)
        assert est.lower_bound < np.prod(implied)

    def test_filename_lineage_contributes(self, tmp_path):
        for name in ("a_v2.ipynb", "a_final.ipynb", "a_test.ipynb"):
            (tmp_path / name).write_text(json.dumps(_notebook([])), encoding="utf-8")
        est = estimate_trials(directory=tmp_path)
        assert est.lower_bound >= 3
        assert any(e.kind == "filename_lineage" for e in est.evidence)

    def test_flags_an_understated_count(self, notebook_path):
        p = notebook_path([_code_cell("window = 10", 1), _code_cell("window = 20", 2)])
        est = estimate_trials(parse_notebook(p), reported=1)
        assert est.understated
        finding = est.to_finding()
        assert finding.id == "SELECT-UNDECLARED-TRIALS"
        assert "lower bound" in finding.detail

    def test_evidence_is_sorted_by_strength(self, notebook_path):
        p = notebook_path(
            [
                _code_cell("grid = list(itertools.product([1,2,3,4,5], [1,2,3,4,5]))", 80),
                _code_cell("window = 1", 81),
            ]
        )
        est = estimate_trials(parse_notebook(p))
        implied = [e.implied_trials for e in est.evidence]
        assert implied == sorted(implied, reverse=True)

    def test_to_dict_is_json_shaped(self, notebook_path):
        p = notebook_path([_code_cell("window = 10", 1), _code_cell("window = 20", 2)])
        json.dumps(estimate_trials(parse_notebook(p), reported=1).to_dict())


class TestPerturbation:
    @staticmethod
    def _honest(data: np.ndarray) -> np.ndarray:
        """Signal from a trailing window only - no look-ahead."""
        x = data[:, 0]
        out = np.zeros_like(x)
        out[1:] = np.sign(x[:-1])
        return out

    @staticmethod
    def _leaky(data: np.ndarray) -> np.ndarray:
        """Signal reads the next observation. The classic shift(-1)."""
        x = data[:, 0]
        out = np.zeros_like(x)
        out[:-1] = np.sign(x[1:])
        return out

    @staticmethod
    def _centred(data: np.ndarray, half: int = 10) -> np.ndarray:
        """A centred rolling mean, so it reaches `half` periods forward."""
        x = data[:, 0]
        out = np.zeros_like(x)
        for t in range(len(x)):
            out[t] = x[max(0, t - half) : min(len(x), t + half + 1)].mean()
        return out

    @pytest.fixture
    def data(self):
        return np.random.default_rng(0).standard_normal((400, 1))

    def test_honest_strategy_passes(self, data):
        r = perturbation_test(self._honest, data)
        assert not r.leaked
        assert r.n_changed == 0
        assert r.to_finding() is None

    def test_leaky_strategy_is_caught(self, data):
        r = perturbation_test(self._leaky, data)
        assert r.leaked
        assert r.modes_that_leaked
        finding = r.to_finding()
        assert finding.id == "LEAK-BEHAVIOURAL"
        assert finding.severity is Severity.CRITICAL

    def test_reports_the_lookahead_span(self, data):
        """A shift(-1) reaches exactly one period forward."""
        r = perturbation_test(self._leaky, data)
        assert r.lookahead_span == 1

    def test_centred_window_reaches_further(self, data):
        r = perturbation_test(lambda d: self._centred(d, half=10), data)
        assert r.leaked
        assert r.lookahead_span == 10

    def test_accepts_one_dimensional_data(self):
        x = np.random.default_rng(1).standard_normal(300)
        assert not perturbation_test(lambda d: self._honest(d), x).leaked

    def test_only_signals_before_the_cut_are_compared(self, data):
        """Signals at and after the cut are expected to change - comparing them
        would flag every strategy."""
        r = perturbation_test(self._honest, data, cut_fraction=0.5)
        assert r.cut_index == 200
        assert not r.leaked

    def test_matching_nans_are_not_a_change(self):
        def nan_leading(d):
            out = np.full(d.shape[0], np.nan)
            out[10:] = 1.0
            return out

        assert not perturbation_test(nan_leading, np.zeros((100, 1)) + 1.0).leaked

    def test_specific_modes_can_be_selected(self, data):
        r = perturbation_test(self._leaky, data, modes=("noise",), n_repeats=6)
        assert r.modes_tested == ("noise",)
        assert r.leaked

    def test_is_reproducible(self, data):
        a = perturbation_test(self._leaky, data, seed=5)
        b = perturbation_test(self._leaky, data, seed=5)
        assert a.max_absolute_change == b.max_absolute_change

    def test_to_dict_is_json_shaped(self, data):
        json.dumps(perturbation_test(self._leaky, data).to_dict())

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"cut_fraction": 0.0}, "cut_fraction"),
            ({"cut_fraction": 1.0}, "cut_fraction"),
            ({"modes": ()}, "at least one corruption mode"),
            ({"modes": ("teleport",)}, "unknown corruption mode"),
        ],
    )
    def test_rejects_bad_arguments(self, data, kwargs, match):
        with pytest.raises(ValueError, match=match):
            perturbation_test(self._honest, data, **kwargs)

    def test_rejects_short_data(self):
        with pytest.raises(ValueError, match="at least 20"):
            perturbation_test(self._honest, np.zeros((10, 1)))

    def test_rejects_a_strategy_returning_the_wrong_length(self, data):
        with pytest.raises(ValueError, match="one signal per row"):
            perturbation_test(lambda d: np.zeros(5), data)

    def test_rejects_three_dimensional_data(self):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            perturbation_test(self._honest, np.zeros((10, 2, 2)))


    def test_repeats_catch_a_leak_a_single_draw_can_miss(self):
        """A sign-based signal survives any one random corruption half the time
        by pure luck; repeats are what make a single-mode run trustworthy."""
        data = np.random.default_rng(11).standard_normal((400, 1))
        results = [
            perturbation_test(
                TestPerturbation._leaky, data, modes=("noise",), n_repeats=6, seed=s
            ).leaked
            for s in range(8)
        ]
        assert all(results)

    def test_rejects_zero_repeats(self):
        with pytest.raises(ValueError, match="n_repeats"):
            perturbation_test(
                TestPerturbation._honest,
                np.random.default_rng(0).standard_normal((100, 1)),
                n_repeats=0,
            )


class TestPerturbationBooks:
    """A cross-sectional book: ``(n, k)`` weights rather than one signal a row.

    Most real strategies are this shape, so until the test accepted it, Tier 2
    was unreachable for anything but a single-instrument timing rule.
    """

    @pytest.fixture
    def data(self):
        return np.random.default_rng(4).standard_normal((400, 3))

    @staticmethod
    def _honest_book(data: np.ndarray) -> np.ndarray:
        """Equal weight on whichever column rose yesterday. Trailing only."""
        out = np.zeros_like(data)
        out[1:] = (data[:-1] > 0).astype(float) / 3.0
        return out

    @staticmethod
    def _rotating_book(data: np.ndarray, forward: int = 0) -> np.ndarray:
        """Hold the strongest single column. Rows always sum to exactly 1.0.

        With ``forward > 0`` the ranking reads that many rows ahead, so the
        book rotates on future information while its gross exposure never
        moves off 1.0 - which is precisely the leak a scalar summary of each
        row cannot see.
        """
        n = data.shape[0]
        source = np.roll(data, -forward, axis=0) if forward else data
        out = np.zeros_like(data)
        winners = np.argmax(source, axis=1)
        out[np.arange(n), winners] = 1.0
        return np.vstack([np.zeros((1, data.shape[1])), out[:-1]])

    def test_accepts_a_cross_sectional_book(self, data):
        result = perturbation_test(self._honest_book, data)
        assert not result.leaked
        assert result.n_changed == 0

    def test_a_leak_in_one_column_is_caught(self, data):
        def one_leaky_column(d):
            out = TestPerturbationBooks._honest_book(d)
            out[:-1, 1] = np.sign(d[1:, 1])
            return out

        result = perturbation_test(one_leaky_column, data)
        assert result.leaked
        assert result.lookahead_span == 1

    def test_a_book_is_compared_per_asset_not_by_gross_exposure(self, data):
        """The rejected design, pinned so it cannot be reintroduced.

        The rows of this book always sum to 1.0, so every scalar reduction of a
        row is constant and blind to the leak. Only a per-asset comparison
        sees it.
        """
        honest = self._rotating_book(data)
        assert np.allclose(honest[1:].sum(axis=1), 1.0)

        # forward=3, not 1: the book already lags its own decision by a row,
        # so a one-row peek nets out to deciding on today and is legitimately
        # invisible. Two rows is the first genuine look-ahead.
        result = perturbation_test(lambda d: self._rotating_book(d, forward=3), data)
        assert result.leaked
        gross = self._rotating_book(data, forward=3).sum(axis=1)
        assert np.allclose(gross[1:], 1.0)

    def test_n_changed_counts_rows_not_cells(self, data):
        """Every cell of every pre-cut row moves, so the count must stay at or
        below the number of rows - otherwise it stops being a fraction."""

        def all_columns_leak(d):
            out = np.zeros_like(d)
            out[:-1] = np.sign(d[1:])
            return out

        result = perturbation_test(all_columns_leak, data, cut_fraction=0.5)
        assert result.n_changed <= result.cut_index
        assert 0.0 <= result.fraction_changed <= 1.0

    def test_first_changed_index_is_a_row_index(self, data):
        """A flattened argmax over an (n, k) diff would report a span that is a
        multiple of the book width rather than a number of periods."""

        def centred_book(d, half=10):
            out = np.zeros_like(d)
            for t in range(len(d)):
                out[t] = d[max(0, t - half) : min(len(d), t + half + 1)].mean(axis=0)
            return out

        result = perturbation_test(centred_book, data, cut_fraction=0.5)
        assert result.leaked
        assert result.lookahead_span == 10

    def test_a_single_column_book_matches_a_flat_signal(self, data):
        """The byte-identity requirement, made executable rather than asserted
        in prose: promoting (n,) to (n, 1) must change nothing at all."""
        flat = perturbation_test(lambda d: self._honest_book(d)[:, 0], data)
        book = perturbation_test(lambda d: self._honest_book(d)[:, :1], data)
        assert flat.to_dict() == book.to_dict()

    def test_rejects_a_book_that_dropped_a_row(self, data):
        with pytest.raises(ValueError, match="one signal per row"):
            perturbation_test(lambda d: self._honest_book(d)[1:], data)

    def test_rejects_a_width_that_depends_on_the_data(self, data):
        """A strategy whose output *shape* moves with the values it is given
        cannot be compared against its own baseline."""
        calls = []

        def unstable(d):
            calls.append(1)
            width = 3 if len(calls) == 1 else 2
            return np.zeros((d.shape[0], width))

        with pytest.raises(ValueError, match="must not depend on the values"):
            perturbation_test(unstable, data)

    def test_matching_nans_in_a_book_are_not_a_change(self):
        def nan_leading(d):
            out = np.ones_like(d)
            out[:10] = np.nan
            return out

        assert not perturbation_test(nan_leading, np.zeros((100, 3)) + 1.0).leaked


class TestCutPlacement:
    """Why the cuts are swept, and placed where they are.

    A single cut is close to blind on any strategy that revises its position
    less often than every period, because a leak can only surface if the
    signal moves between its last revision and the cut. These tests are the
    evidence for that design, and what stops a future refactor quietly
    reverting to one fixed cut.
    """

    @staticmethod
    def _periodic(data: np.ndarray, peek: int = 0, every: int = 21) -> np.ndarray:
        """Revises every ``every`` rows, optionally reading ``peek`` ahead."""
        n = data.shape[0]
        source = np.roll(data[:, 0], -peek) if peek else data[:, 0]
        decided = np.sign(source)
        out = np.zeros(n)
        held = 0.0
        for t in range(n):
            if t % every == 0:
                held = decided[t]
            out[t] = held
        return np.concatenate([[0.0], out[:-1]])

    @pytest.fixture
    def data(self):
        return np.random.default_rng(7).standard_normal((1200, 1))

    def test_a_single_cut_misses_a_leak_the_sweep_finds(self, data):
        """The measurement the default rests on. One fixed cut lands mid-period
        where nothing has revised, so it sees nothing."""

        def leaky(d):
            return self._periodic(d, peek=10)

        assert not perturbation_test(leaky, data, cut_fraction=0.7).leaked
        assert perturbation_test(leaky, data).leaked

    def test_an_honest_periodic_strategy_stays_clean(self, data):
        assert not perturbation_test(lambda d: self._periodic(d), data).leaked

    def test_the_detection_floor_tracks_the_revision_interval(self, data):
        result = perturbation_test(lambda d: self._periodic(d, every=21), data)
        assert result.detection_floor is not None
        assert 10 <= result.detection_floor <= 45
        assert "more than about" in result.clean_claim

    def test_a_signal_that_never_moves_establishes_nothing(self):
        """Reported honestly rather than as a clean bill of health: destroying
        the future cannot move a signal that never moves at all."""
        result = perturbation_test(lambda d: np.ones(d.shape[0]), np.zeros((200, 1)))
        assert not result.leaked
        assert result.detection_floor is None
        assert "Nothing established" in result.clean_claim

    def test_every_cut_is_inside_the_sample(self, data):
        result = perturbation_test(lambda d: self._periodic(d), data)
        assert result.cuts_tested
        assert all(2 <= c < len(data) for c in result.cuts_tested)
        assert len(set(result.cuts_tested)) == len(result.cuts_tested)

    def test_pinning_a_cut_disables_the_sweep(self, data):
        result = perturbation_test(lambda d: self._periodic(d), data, cut_fraction=0.5)
        assert result.cuts_tested == (600,)
        assert result.cut_index == 600

    def test_rejects_zero_cuts(self, data):
        with pytest.raises(ValueError, match="n_cuts"):
            perturbation_test(lambda d: self._periodic(d), data, n_cuts=0)

    @pytest.mark.slow
    def test_detection_rises_with_the_leak_horizon(self, data):
        """The calibration table from the design work, as a test.

        Detection is not all-or-nothing: it depends on how far the leak reaches
        relative to how often the strategy revises. A leak shorter than the
        revision interval is genuinely hard to see, and the tool should not
        pretend otherwise - but a leak comparable to it must be caught.
        """
        rates = {}
        for peek in (0, 5, 21, 63):
            result = perturbation_test(lambda d: self._periodic(d, peek=peek), data)
            rates[peek] = len(result.cuts_that_leaked)

        assert rates[0] == 0, "an honest strategy must never be flagged"
        assert rates[21] > 0, "a leak one revision deep must be caught"
        assert rates[63] > 0, "a leak three revisions deep must be caught"
        assert rates[63] >= rates[5], "detection must not fall as the leak grows"
