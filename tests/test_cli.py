"""Tests for the command line interface.

Every agent capability is backed by a deterministic subcommand, so these tests
also guard the contract the Claude Code skill depends on: stable exit codes,
stable `--json` shapes, and output that says what could not be tested.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from qv.cli import app, load_matrix, load_series

runner = CliRunner()


@pytest.fixture
def returns_csv(tmp_path):
    import pandas as pd

    gen = np.random.default_rng(2)
    path = tmp_path / "returns.csv"
    pd.DataFrame({"returns": 0.0004 + 0.01 * gen.standard_normal(600)}).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def positions_csv(tmp_path):
    import pandas as pd

    gen = np.random.default_rng(3)
    path = tmp_path / "positions.csv"
    pd.DataFrame({"position": gen.integers(0, 2, 600).astype(float)}).to_csv(
        path, index=False
    )
    return path


class TestLoaders:
    def test_single_numeric_column_needs_no_hint(self, returns_csv):
        assert load_series(returns_csv).shape == (600,)

    def test_named_column(self, returns_csv):
        assert load_series(returns_csv, "returns").shape == (600,)

    def test_prefers_an_obviously_named_column(self, tmp_path):
        import pandas as pd

        path = tmp_path / "multi.csv"
        pd.DataFrame({"junk": [1.0] * 5, "returns": [0.1] * 5}).to_csv(path, index=False)
        assert load_series(path).tolist() == [0.1] * 5

    def test_ambiguous_columns_ask_for_a_name(self, tmp_path):
        import pandas as pd

        path = tmp_path / "ambiguous.csv"
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}).to_csv(path, index=False)
        with pytest.raises(Exception, match="name one with --column"):
            load_series(path)

    def test_unknown_column_lists_what_is_available(self, returns_csv):
        with pytest.raises(Exception, match="no column"):
            load_series(returns_csv, "nope")

    def test_missing_file(self, tmp_path):
        with pytest.raises(Exception, match="no such file"):
            load_series(tmp_path / "absent.csv")

    def test_no_numeric_column(self, tmp_path):
        import pandas as pd

        path = tmp_path / "text.csv"
        pd.DataFrame({"name": ["a", "b"]}).to_csv(path, index=False)
        with pytest.raises(Exception, match="no numeric column"):
            load_series(path)

    def test_matrix_needs_two_columns(self, returns_csv):
        with pytest.raises(Exception, match="at least 2 numeric columns"):
            load_matrix(returns_csv)

    def test_matrix_loads(self, tmp_path):
        import pandas as pd

        path = tmp_path / "trials.csv"
        gen = np.random.default_rng(4)
        pd.DataFrame(gen.standard_normal((100, 5)) * 0.01).to_csv(path, index=False)
        assert load_matrix(path).shape == (100, 5)


class TestValidate:
    def test_writes_both_report_files(self, returns_csv, tmp_path):
        out = tmp_path / "report"
        result = runner.invoke(app, ["validate", str(returns_csv), "--out", str(out),
                                     "--n-boot", "100"])
        assert result.exit_code in (0, 1)
        assert (out / "report.html").exists()
        assert (out / "report.json").exists()

    def test_prints_the_falsification_framing(self, returns_csv, tmp_path):
        result = runner.invoke(app, ["validate", str(returns_csv), "--out", str(tmp_path / "r"),
                                     "--n-boot", "100"])
        assert "falsifies; it cannot validate" in result.stdout

    def test_lists_what_could_not_be_tested(self, returns_csv, tmp_path):
        result = runner.invoke(app, ["validate", str(returns_csv), "--out", str(tmp_path / "r"),
                                     "--n-boot", "100"])
        assert "Could not test" in result.stdout

    def test_positions_unlock_tier_one(self, returns_csv, positions_csv, tmp_path):
        out = tmp_path / "t1"
        runner.invoke(app, ["validate", str(returns_csv), "--positions", str(positions_csv),
                            "--asset-class", "us_large_cap_etf", "--out", str(out),
                            "--n-boot", "100"])
        data = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert data["tier"] == 1
        assert "costs" in data

    def test_trials_unlock_deflation(self, returns_csv, tmp_path):
        out = tmp_path / "defl"
        runner.invoke(app, ["validate", str(returns_csv), "--trials", "500",
                            "--out", str(out), "--n-boot", "100"])
        data = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert data["selection"]["deflated_sharpe"]["n_trials"] == 500

    def test_exit_code_one_on_a_critical_finding(self, tmp_path):
        """CI needs to be able to fail a build on this."""
        import pandas as pd

        # Deliberately marginal: a periodic Sharpe of about 0.15 that a search
        # of 100,000 configurations would be expected to produce from noise
        # alone. A genuinely strong result survives even a huge trial count,
        # which is the point of deflation rather than a blanket penalty.
        gen = np.random.default_rng(5)
        path = tmp_path / "mined.csv"
        pd.DataFrame({"returns": 0.0015 + 0.01 * gen.standard_normal(400)}).to_csv(
            path, index=False
        )
        result = runner.invoke(app, ["validate", str(path), "--trials", "100000",
                                     "--out", str(tmp_path / "o"), "--n-boot", "100"])
        assert result.exit_code == 1

    def test_is_reproducible_across_runs(self, returns_csv, tmp_path):
        outs = []
        for i in range(2):
            out = tmp_path / f"run{i}"
            runner.invoke(app, ["validate", str(returns_csv), "--out", str(out),
                                "--n-boot", "100", "--seed", "7"])
            data = json.loads((out / "report.json").read_text(encoding="utf-8"))
            data["provenance"].pop("generated_utc")
            outs.append(data)
        assert outs[0] == outs[1]


class TestScan:
    def test_flags_a_leaky_file(self, tmp_path):
        path = tmp_path / "strategy.py"
        path.write_text("signal = df['close'].shift(-1)\n", encoding="utf-8")
        result = runner.invoke(app, ["scan", str(path)])
        assert result.exit_code == 1
        assert "LEAK-NEGATIVE-SHIFT" in result.stdout

    def test_clean_file_says_so_without_overclaiming(self, tmp_path):
        path = tmp_path / "ok.py"
        path.write_text("x = 1 + 1\n", encoding="utf-8")
        result = runner.invoke(app, ["scan", str(path)])
        assert result.exit_code == 0
        assert "not a clean bill of health" in result.stdout

    def test_json_output_is_machine_readable(self, tmp_path):
        path = tmp_path / "s.py"
        path.write_text("y = df['c'].shift(-1)\n", encoding="utf-8")
        result = runner.invoke(app, ["scan", str(path), "--json"])
        data = json.loads(result.stdout)
        assert data["findings"][0]["id"] == "LEAK-NEGATIVE-SHIFT"

    def test_scans_a_notebook(self, tmp_path):
        nb = {
            "cells": [{"cell_type": "code", "source": "y = df['c'].shift(-1)",
                       "execution_count": 1, "outputs": [], "metadata": {}}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }
        path = tmp_path / "n.ipynb"
        path.write_text(json.dumps(nb), encoding="utf-8")
        result = runner.invoke(app, ["scan", str(path)])
        assert "LEAK-NEGATIVE-SHIFT" in result.stdout

    def test_missing_file_is_a_usage_error(self, tmp_path):
        assert runner.invoke(app, ["scan", str(tmp_path / "nope.py")]).exit_code != 0


class TestTrials:
    @pytest.fixture
    def searched_notebook(self, tmp_path):
        cells = [
            {"cell_type": "code", "source": "window = 10", "execution_count": 40,
             "outputs": [], "metadata": {}},
            {"cell_type": "code", "source": "window = 20", "execution_count": 41,
             "outputs": [], "metadata": {}},
            {"cell_type": "code", "source": "window = 50", "execution_count": 42,
             "outputs": [], "metadata": {}},
        ]
        path = tmp_path / "search.ipynb"
        path.write_text(
            json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
            encoding="utf-8",
        )
        return path

    def test_reports_a_lower_bound(self, searched_notebook):
        result = runner.invoke(app, ["trials", str(searched_notebook)])
        assert "at least" in result.stdout
        assert "lower bound" in result.stdout

    def test_flags_an_understated_count(self, searched_notebook):
        result = runner.invoke(app, ["trials", str(searched_notebook), "--reported", "1"])
        assert result.exit_code == 1
        assert "contradicted" in result.stdout

    def test_json_output(self, searched_notebook):
        result = runner.invoke(app, ["trials", str(searched_notebook), "--json"])
        data = json.loads(result.stdout)
        assert data["lower_bound"] >= 3


class TestExplain:
    def test_explains_a_known_finding(self):
        result = runner.invoke(app, ["explain", "LEAK-NEGATIVE-SHIFT"])
        assert result.exit_code == 0
        assert "Remediation:" in result.stdout

    def test_lists_the_whole_catalog(self):
        from qv.findings import CATALOG

        result = runner.invoke(app, ["explain", "--all"])
        assert f"{len(CATALOG)} defects" in result.stdout

    def test_renders_markdown(self):
        result = runner.invoke(app, ["explain", "--markdown"])
        assert result.stdout.startswith("# Findings catalog")

    def test_unknown_id_exits_nonzero(self):
        result = runner.invoke(app, ["explain", "NOT-A-FINDING"])
        assert result.exit_code == 2


class TestDemo:
    def test_runs_the_flagship_label(self, tmp_path):
        result = runner.invoke(app, ["demo", "null_mined", "--out", str(tmp_path / "d"),
                                     "--observations", "600"])
        assert result.exit_code == 0
        assert "none, by construction" in result.stdout
        assert (tmp_path / "d" / "report.html").exists()

    def test_the_real_edge_is_labelled_real(self, tmp_path):
        result = runner.invoke(app, ["demo", "genuine_weak", "--out", str(tmp_path / "g"),
                                     "--observations", "800"])
        assert "True edge: REAL" in result.stdout

    def test_unknown_label_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["demo", "nonsense", "--out", str(tmp_path / "x")])
        assert result.exit_code == 2


class TestAdapterInit:
    """The scaffold has to produce something that actually runs.

    A template that needs debugging before it works is worse than no template,
    because the researcher cannot tell their mistake from ours.
    """

    def test_writes_both_files(self, tmp_path):
        result = runner.invoke(app, ["adapter", "init", "--out", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "qv_adapter.py").exists()
        assert (tmp_path / "research_manifest.yaml").exists()

    def test_the_scaffolded_manifest_validates(self, tmp_path):
        from qv.manifest import load_manifest

        runner.invoke(app, ["adapter", "init", "--out", str(tmp_path)])
        manifest = load_manifest(tmp_path / "research_manifest.yaml")
        assert manifest.search.axes
        assert manifest.adapter_reference() == "qv_adapter.py:positions"
        # Every axis the grid varies must be pinned by the chosen point, or
        # the audit cannot say which configuration it is reporting.
        assert set(manifest.search.axes) <= set(manifest.chosen_parameters)

    def test_the_scaffolded_adapter_imports_and_runs(self, tmp_path):
        import numpy as np
        import pandas as pd

        from qv.pipeline import load_adapter

        runner.invoke(app, ["adapter", "init", "--out", str(tmp_path)])
        positions = load_adapter("qv_adapter.py:positions", base=tmp_path)

        gen = np.random.default_rng(0)
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(gen.normal(0.0003, 0.01, (600, 5)), axis=0)),
            index=pd.bdate_range("2018-01-01", periods=600),
            columns=["SPY", "IEF", "GLD", "XLK", "XLP"],
        )
        book = positions(prices, lookback=252, top_n=2)
        assert book.shape == prices.shape
        assert np.isfinite(book.to_numpy()).all()
        # The template must be honest out of the box, or it teaches the wrong
        # thing to everyone who starts from it.
        active = book[book.abs().sum(axis=1) > 0]
        assert np.allclose(active.sum(axis=1), 1.0)

    def test_the_scaffolded_adapter_does_not_read_the_future(self, tmp_path):
        import numpy as np
        import pandas as pd

        from qv.adapter import frame_adapter
        from qv.leakage.perturbation import perturbation_test
        from qv.pipeline import load_adapter

        runner.invoke(app, ["adapter", "init", "--out", str(tmp_path)])
        positions = load_adapter("qv_adapter.py:positions", base=tmp_path)
        gen = np.random.default_rng(1)
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(gen.normal(0.0003, 0.01, (700, 4)), axis=0)),
            index=pd.bdate_range("2018-01-01", periods=700),
            columns=list("ABCD"),
        )
        strategy = frame_adapter(
            positions, prices.index, prices.columns, lookback=252, top_n=2
        )
        assert not perturbation_test(strategy, prices.to_numpy()).leaked

    def test_existing_files_are_kept_unless_forced(self, tmp_path):
        (tmp_path / "qv_adapter.py").write_text("# mine\n", encoding="utf-8")
        (tmp_path / "research_manifest.yaml").write_text("name: mine\n", encoding="utf-8")
        result = runner.invoke(app, ["adapter", "init", "--out", str(tmp_path)])
        assert result.exit_code == 1
        assert "Kept existing" in result.output
        assert (tmp_path / "qv_adapter.py").read_text(encoding="utf-8") == "# mine\n"

    def test_force_overwrites(self, tmp_path):
        (tmp_path / "qv_adapter.py").write_text("# mine\n", encoding="utf-8")
        result = runner.invoke(
            app, ["adapter", "init", "--out", str(tmp_path), "--force"]
        )
        assert result.exit_code == 0
        assert "# mine" not in (tmp_path / "qv_adapter.py").read_text(encoding="utf-8")


class TestValidateWithAManifest:
    """The route that makes Tier 2 reachable without hand-written Python."""

    def _project(self, tmp_path):
        """An adapter and a manifest whose prices are a local CSV."""
        import numpy as np
        import pandas as pd

        (tmp_path / "qv_adapter.py").write_text(
            "import pandas as pd\n\n\n"
            "def positions(prices, lookback=120, top_n=2):\n"
            "    score = prices / prices.shift(lookback) - 1.0\n"
            "    ranks = score.rank(axis=1, ascending=False, na_option='bottom')\n"
            "    weights = (ranks <= top_n).astype(float) / top_n\n"
            "    weights[score.isna().all(axis=1)] = 0.0\n"
            "    return weights.shift(1).fillna(0.0)\n",
            encoding="utf-8",
        )
        (tmp_path / "research_manifest.yaml").write_text(
            "name: csv rotation\n"
            "data:\n"
            "  universe: [AAA, BBB, CCC]\n"
            "  start_date: '2016-01-01'\n"
            "  end_date: '2019-12-31'\n"
            "  source: csv\n"
            "  price_frame: prices.csv\n"
            "  load_factors: false\n"
            "  universe_point_in_time: true\n"
            "strategy:\n"
            "  adapter: qv_adapter.py:positions\n"
            "chosen_parameters:\n"
            "  lookback: 120\n"
            "  top_n: 2\n"
            "search:\n"
            "  axes:\n"
            "    lookback: [60, 120]\n"
            "    top_n: [1, 2]\n",
            encoding="utf-8",
        )
        gen = np.random.default_rng(3)
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(gen.normal(0.0004, 0.011, (800, 3)), axis=0)),
            index=pd.bdate_range("2016-01-04", periods=800),
            columns=["AAA", "BBB", "CCC"],
        )
        prices.to_csv(tmp_path / "prices.csv")
        return tmp_path / "research_manifest.yaml"

    def test_reaches_tier_two_and_writes_a_report(self, tmp_path):
        import json

        manifest = self._project(tmp_path)
        out = tmp_path / "report"
        result = runner.invoke(
            app,
            ["validate", "--manifest", str(manifest), "--out", str(out), "--n-boot", "200"],
        )
        assert result.exit_code in (0, 1), result.output
        payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        assert payload["tier"] == 2
        assert payload["perturbation"]["leaked"] is False
        assert not any(
            "Behavioural leakage" in gap for gap in payload["not_tested"]
        )

    def test_it_reruns_the_declared_grid(self, tmp_path):
        manifest = self._project(tmp_path)
        result = runner.invoke(
            app,
            [
                "validate",
                "--manifest",
                str(manifest),
                "--out",
                str(tmp_path / "r"),
                "--n-boot",
                "200",
            ],
        )
        assert "Evaluated 4 configurations" in result.output

    def test_a_bad_adapter_reference_is_a_usage_error_not_a_traceback(self, tmp_path):
        manifest = self._project(tmp_path)
        result = runner.invoke(
            app,
            [
                "validate",
                "--manifest",
                str(manifest),
                "--adapter",
                "qv_adapter.py:not_there",
                "--out",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code == 2
        assert "has no attribute" in result.output

    def test_giving_both_a_series_and_a_manifest_is_refused(self, returns_csv, tmp_path):
        """They describe different audits, and silently preferring one would
        make the reported tier depend on argument order."""
        result = runner.invoke(
            app,
            [
                "validate",
                str(returns_csv),
                "--manifest",
                str(tmp_path / "whatever.yaml"),
                "--out",
                str(tmp_path / "r"),
            ],
        )
        assert result.exit_code == 2
        assert "not both and not neither" in result.output

    def test_giving_neither_is_refused(self, tmp_path):
        result = runner.invoke(app, ["validate", "--out", str(tmp_path / "r")])
        assert result.exit_code == 2
        assert "not both and not neither" in result.output


class TestSkillInstall:
    """Installing the skill is the step that turns the CLI into something an
    agent can drive. It had no route at all before: the skill shipped in
    `skill/` with correct frontmatter and nothing put it where Claude Code
    looks for it."""

    def test_installs_the_whole_skill(self, tmp_path):
        target = tmp_path / "proper-validation"
        result = runner.invoke(app, ["skill", "install", "--target", str(target)])
        assert result.exit_code == 0
        assert (target / "SKILL.md").exists()
        # The references are what stop the agent improvising statistics.
        for name in (
            "adapter_protocol.md",
            "manifest_schema.md",
            "findings_catalog.md",
            "interview.md",
            "trial_archaeology.md",
        ):
            assert (target / "references" / name).exists(), name

    def test_the_installed_skill_keeps_its_frontmatter(self, tmp_path):
        """Claude Code discovers a skill by its name and description. Copying
        the body without the frontmatter would install something inert."""
        target = tmp_path / "s"
        runner.invoke(app, ["skill", "install", "--target", str(target)])
        head = (target / "SKILL.md").read_text(encoding="utf-8")[:400]
        assert head.startswith("---")
        assert "name: proper-validation" in head
        assert "description:" in head

    def test_refuses_to_overwrite_without_force(self, tmp_path):
        target = tmp_path / "s"
        target.mkdir()
        (target / "mine.txt").write_text("keep me", encoding="utf-8")
        result = runner.invoke(app, ["skill", "install", "--target", str(target)])
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert (target / "mine.txt").read_text(encoding="utf-8") == "keep me"

    def test_force_replaces_it(self, tmp_path):
        target = tmp_path / "s"
        target.mkdir()
        (target / "stale.txt").write_text("old", encoding="utf-8")
        result = runner.invoke(app, ["skill", "install", "--target", str(target), "--force"])
        assert result.exit_code == 0
        assert not (target / "stale.txt").exists()
        assert (target / "SKILL.md").exists()

    def test_says_where_it_went_and_what_to_do_next(self, tmp_path):
        target = tmp_path / "s"
        result = runner.invoke(app, ["skill", "install", "--target", str(target)])
        assert str(target) in result.output
        assert "Claude Code" in result.output


class TestDemoRunsFromTheConsoleScript:
    """`qv demo` imports `benchmarks`, which sits beside the package rather
    than inside it. That works under `python -m`, which puts the working
    directory on sys.path, and used to fail under the installed `qv` console
    script, which does not - so the first command the README offered a new
    reader was broken."""

    def test_the_repo_root_is_discoverable_from_the_package(self):
        from qv.cli import _repo_root

        root = _repo_root()
        assert root is not None
        assert (root / "skill" / "SKILL.md").exists()
        assert (root / "benchmarks" / "generate.py").exists()

    def test_demo_writes_a_report_without_help_from_the_working_directory(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            ["demo", "null_mined", "--out", str(tmp_path / "r"), "--observations", "400"],
        )
        assert result.exit_code in (0, 1), result.output
        assert (tmp_path / "r" / "report.html").exists()
        assert (tmp_path / "r" / "report.json").exists()
