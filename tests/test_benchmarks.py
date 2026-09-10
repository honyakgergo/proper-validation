"""Tests for the synthetic ground truth and the ROC measurement.

These are the tests that guard the credibility claim. If the generators do not
have the properties they advertise, the detection-rate table measures nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.generate import GENERATORS, LABELS, generate, generate_all
from benchmarks.run_roc import format_markdown, run_label, wilson_interval
from qv.audit import AuditInputs, run_audit
from qv.stats.sharpe import analyse_sharpe
from qv.types import Severity


class TestGenerators:
    def test_every_label_generates(self):
        assert len(generate_all(seed=0, n=400)) == len(LABELS)

    @pytest.mark.parametrize("label", LABELS)
    def test_output_is_well_formed(self, label):
        s = generate(label, seed=0, n=400)
        assert s.label == label
        assert s.returns.shape == (400,)
        assert np.all(np.isfinite(s.returns))
        assert s.description
        assert isinstance(s.has_edge, bool)

    @pytest.mark.parametrize("label", LABELS)
    def test_is_deterministic_given_a_seed(self, label):
        a = generate(label, seed=7, n=300)
        b = generate(label, seed=7, n=300)
        assert np.array_equal(a.returns, b.returns)

    @pytest.mark.parametrize("label", LABELS)
    def test_different_seeds_give_different_data(self, label):
        a = generate(label, seed=1, n=300)
        b = generate(label, seed=2, n=300)
        assert not np.array_equal(a.returns, b.returns)

    def test_exactly_one_label_has_a_real_edge(self):
        """The tool must be measurable in both directions. Without a label that
        should pass, a 100% detection rate proves only that it condemns
        everything."""
        with_edge = [s.label for s in generate_all(seed=0, n=400) if s.has_edge]
        assert with_edge == ["genuine_weak"]

    def test_unknown_label_raises(self):
        with pytest.raises(KeyError, match="unknown label"):
            generate("not_a_label")

    def test_generators_dict_matches_labels(self):
        assert tuple(GENERATORS) == LABELS


class TestGeneratorProperties:
    """Each label must actually have the property its name claims."""

    def test_null_mined_winner_looks_publishable(self):
        s = generate("null_mined", seed=0, n=2000)
        annual = analyse_sharpe(s.returns, 252).annualised_adjusted.value
        assert annual > 0.7, "a mined winner that looks bad demonstrates nothing"
        assert s.trial_returns.shape[1] == s.n_trials

    def test_leaky_shift_is_wildly_profitable(self):
        """Knowing tomorrow should look absurd. If it does not, the leak is not
        wired up and the detector is being tested against nothing."""
        s = generate("leaky_shift", seed=0, n=1000)
        assert analyse_sharpe(s.returns, 252).annualised_adjusted.value > 5

    def test_leaky_scaler_exposes_a_behavioural_leak(self):
        s = generate("leaky_scaler", seed=0, n=800)
        from qv.leakage.perturbation import perturbation_test

        assert perturbation_test(s.strategy, s.strategy_data).leaked

    def test_regime_fluke_is_concentrated_in_its_window(self):
        s = generate("regime_fluke", seed=0, n=2000)
        inside = s.returns[500:900].mean()
        outside = np.concatenate([s.returns[:500], s.returns[900:]]).mean()
        assert inside > 5 * abs(outside)

    def test_cost_fragile_breaks_even_inside_realistic_costs(self):
        from qv.costs.breakeven import break_even_cost

        s = generate("cost_fragile", seed=0, n=2000)
        be = break_even_cost(s.returns, s.positions, asset_class=s.asset_class)
        assert be.severity >= Severity.MEDIUM
        assert be.break_even_bps < 12

    def test_levered_beta_is_perfectly_correlated_with_the_market(self):
        s = generate("levered_beta", seed=0, n=1000)
        assert np.corrcoef(s.returns, s.asset_returns)[0, 1] == pytest.approx(1.0)

    def test_lottery_ticket_profit_is_concentrated(self):
        from qv.robustness.subsample import top_day_dependence

        s = generate("lottery_ticket", seed=0, n=2000)
        assert top_day_dependence(s.returns).worst_concentration_ratio > 2.0

    def test_genuine_weak_has_real_timing_skill(self):
        """It must beat random timing at the same exposure, or the
        matched-exposure test is right to flag it and the label is a lie."""
        from qv.robustness.randomization import matched_exposure_test

        s = generate("genuine_weak", seed=0, n=2000)
        assert matched_exposure_test(
            s.asset_returns, s.positions, n_sims=300
        ).p_value < 0.05

    def test_genuine_weak_is_modest_not_absurd(self):
        """A validator that only clears implausibly good strategies is no more
        useful than one that clears nothing."""
        annual = analyse_sharpe(generate("genuine_weak", seed=0, n=2000).returns, 252)
        assert 0.5 < annual.annualised_adjusted.value < 3.0


@pytest.mark.slow
class TestGroundTruthEndToEnd:
    """The claim the whole repository rests on."""

    @staticmethod
    def _audit(s):
        return run_audit(
            AuditInputs(
                returns=s.returns,
                positions=s.positions,
                asset_returns=s.asset_returns,
                asset_class=s.asset_class or "us_large_cap_equity",
                benchmark_returns=s.asset_returns if s.label == "levered_beta" else None,
                n_trials=s.n_trials,
                trial_returns=s.trial_returns,
                strategy=s.strategy,
                strategy_data=s.strategy_data,
                n_boot=200,
                name=s.label,
            )
        )

    #: regime_fluke is the hardest label and is measured at roughly 80% rather
    #: than 100% (see benchmarks/roc_results.md). Asserting it always fires
    #: would be asserting something untrue, so it is checked as a rate below.
    _RELIABLE = tuple(
        lbl for lbl in LABELS if lbl not in ("genuine_weak", "regime_fluke")
    )

    @pytest.mark.parametrize("label", _RELIABLE)
    def test_no_edge_labels_are_flagged(self, label):
        report = self._audit(generate(label, seed=0, n=2000))
        assert report.worst_severity >= Severity.HIGH, (
            f"{label} has no edge by construction but was not flagged"
        )

    def test_regime_fluke_is_flagged_most_of_the_time(self):
        """The weakest detection in the suite, and reported as such.

        An edge confined to one window need not flip the sign of any
        rolling-origin start date, so it can slip past. The published table
        measures this rather than hiding it.
        """
        flagged = sum(
            self._audit(generate("regime_fluke", seed=s, n=2000)).worst_severity
            >= Severity.HIGH
            for s in range(6)
        )
        assert flagged >= 4, f"only {flagged}/6 regime_fluke replications were flagged"

    @pytest.mark.parametrize("seed", range(4))
    def test_the_real_edge_survives(self, seed):
        """The most important test here. A false positive on this label is the
        tool being wrong in the direction that matters most."""
        report = self._audit(generate("genuine_weak", seed=seed, n=2000))
        assert report.worst_severity < Severity.HIGH, (
            f"genuine_weak was flagged at seed {seed}: "
            f"{[f.id for f in report.findings if f.severity >= Severity.HIGH]}"
        )

    @pytest.mark.parametrize("label", LABELS)
    def test_expected_findings_actually_fire(self, label):
        s = generate(label, seed=0, n=2000)
        ids = {f.id for f in self._audit(s).findings}
        for expected in s.expected_findings:
            assert expected in ids, f"{label} did not produce {expected}"


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self):
        lo, hi = wilson_interval(15, 20)
        assert lo < 0.75 < hi

    def test_stays_inside_zero_and_one_at_the_extremes(self):
        """The reason for Wilson over the normal approximation: these rates sit
        at the boundary, where the naive interval runs outside [0, 1]."""
        for successes in (0, 20):
            lo, hi = wilson_interval(successes, 20)
            assert 0.0 <= lo <= hi <= 1.0
        assert wilson_interval(20, 20)[0] > 0.8
        assert wilson_interval(0, 20)[1] < 0.2

    def test_narrows_with_more_trials(self):
        narrow = wilson_interval(500, 1000)
        wide = wilson_interval(5, 10)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])

    def test_zero_trials_is_maximally_uncertain(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestRunRoc:
    def test_run_label_reports_a_rate(self):
        outcome = run_label("leaky_shift", replications=2, n=600, n_boot=100)
        assert outcome.replications == 2
        assert 0.0 <= outcome.flag_rate <= 1.0
        assert outcome.interval[0] <= outcome.flag_rate <= outcome.interval[1]

    def test_correct_rate_inverts_for_the_real_edge(self):
        """Flagging a no-edge label is correct; flagging genuine_weak is not."""
        no_edge = run_label("leaky_shift", replications=2, n=600, n_boot=100)
        real = run_label("genuine_weak", replications=2, n=800, n_boot=100)
        assert no_edge.correct_rate == no_edge.flag_rate
        assert real.correct_rate == 1.0 - real.flag_rate

    def test_tracks_which_expected_finding_fired(self):
        outcome = run_label("leaky_shift", replications=2, n=600, n_boot=100)
        assert "LEAK-BEHAVIOURAL" in outcome.expected_hits

    def test_markdown_names_every_label_and_the_reproduction_command(self):
        outcomes = [run_label("null_pure", replications=1, n=400, n_boot=50)]
        md = format_markdown(outcomes, 1)
        assert "`null_pure`" in md
        assert "python benchmarks/run_roc.py" in md
        assert "Wilson" in md

    def test_outcome_serialises(self):
        import json

        json.dumps(run_label("null_pure", replications=1, n=400, n_boot=50).to_dict())
