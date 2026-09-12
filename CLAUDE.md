# Working in this repository

Guidance for Claude Code when editing `proper_validation` itself. If you are
here to **audit someone's backtest** rather than to change this tool, that is
the skill's job, not this file's — run `qv skill install` and start a fresh
session.

## What this project is, in one paragraph

An adversarial validator for quantitative backtests. It attacks an existing
backtest and reports how much of its claimed performance survives. **It
falsifies; it cannot validate.** There is deliberately no `PASSED` verdict —
the best available outcome is `NOT_FALSIFIED`, "the tests applied did not break
it". That asymmetry lives in `qv/types.py::Verdict` and runs through every
piece of report copy. It is the single most important thing not to soften by
accident.

## Rules that everything else follows from

1. **One question, one test.** No two components may answer the same question.
   Several things have been cut for violating this — Bonferroni and Holm
   haircuts, calendar and drawdown regime splits, the timing-shuffle null. If a
   change would make two tests answer the same question, cut one.
2. **Every number in the report must be in `report.json`.** The JSON is
   authoritative; the HTML may never make a claim the JSON cannot back. Each
   chart function returns a `chart_spec` holding exactly the values plotted,
   and the tests assert on the spec rather than on pixels.
3. **The engine never touches the network.** `qv/data/loaders.py` is the single
   network boundary. Everything else takes DataFrames in and returns results
   out, which is what lets the whole suite run offline and deterministically.
   Do not import `yfinance` anywhere else.
4. **State what could not be tested.** Every skipped test appends a reason to
   `AuditReport.not_tested`. A quietly omitted test is worse than a stated gap.
5. **Never report a number more confidently than it deserves.** Below
   `MIN_OBS_FOR_ASYMPTOTICS` observations, refuse asymptotic claims. Where a
   result has a resolution limit, report the limit beside it — see the
   behavioural leakage test's detection floor.

## The two suites

An audit asks two independent questions, and `qv/types.py::Suite` selects
which:

- **Statistical validation** — is the measured edge distinguishable from luck?
- **Engine analysis** — can the backtest be trusted as an implementation,
  whatever its numbers say?

They must stay independent. No test appears in both;
`tests/test_engine.py::TestSuiteSeparation` asserts the section sets are
disjoint and that their union is the full report. `run_audit` gates on
`inputs.suite`; `build_charts` and the template gate on `report.suite`.

There is no longer any user-facing notion of a "tier". `Tier` survives
internally as an ordering of input completeness, because that is what gates
individual tests, but no string a reader sees mentions it.

## Layout

| Path | What it is |
|---|---|
| `qv/audit.py` | The orchestrator. Runs every test the inputs and suite support. Most changes land here. |
| `qv/types.py` | `Suite`, `Tier`, `Severity`, `Verdict`, `Estimate`, `Finding`. Standard library only, so the numeric core is testable in isolation. |
| `qv/findings.py` | The catalog: every defect the tool can name, with detection and remediation. **Single source of truth** — `skill/references/findings_catalog.md` is generated from it. |
| `qv/engine.py` | The engine suite: determinism, degeneracy, execution-delay fragility, universe coverage. |
| `qv/leakage/` | AST scanner, notebook trial archaeology, behavioural perturbation test. |
| `qv/stats/` | Sharpe and its standard error, Lo annualisation, stationary bootstrap, Deflated Sharpe, PBO, resampled risk. |
| `qv/pipeline.py` | `audit_from_manifest`: a manifest plus an adapter to a finished report. What `qv validate --manifest` runs. |
| `qv/report/` | Charts (inline SVG) and the Jinja template. |
| `skill/` | The Claude Code skill. Installed by `qv skill install`. |
| `benchmarks/` | Synthetic strategies with known ground truth, and the measured detection rates. |

## Version control

**Never run `git commit` or `git push` in this repository.** The maintainer
commits and pushes everything themselves — leave your work in the working tree
and say what changed. This holds even if a task looks finished, if the change
is small, or if a previous step seemed to invite a commit. Read-only git
commands (`status`, `diff`, `log`) are fine.

## Before you hand work back

```bash
pytest -q                      # must stay green; 1062 tests, no network
pytest -q -m "not slow"        # faster, skips the coverage simulations
pytest --cov=qv                # coverage should not fall
```

If you changed `qv/findings.py`, regenerate the catalog the skill ships:

```bash
qv explain --markdown > skill/references/findings_catalog.md
```

If you changed **anything under `qv/`**, regenerate the committed examples —
they are artifacts people read without running anything, so a stale report is a
wrong report:

```bash
python real_user_tests/dual_momentum/run.py --offline
python real_user_tests/dual_momentum/run.py --offline --suite statistical
python real_user_tests/dual_momentum/run.py --offline --suite engine
python real_user_tests/sector_momentum/run.py --offline
python examples/01_mined_noise/mine.py --offline
```

The trigger is any source change, not only one that alters a rendered number:
the footer carries a digest over `qv/**/*.py`, so an edit to a docstring moves
it. That is the cost of an identifier precise enough to be worth printing —
a report whose digest matches no checkout that exists identifies nothing.

`--offline` refuses the network and fails loudly rather than silently
refetching, which would quietly change the numbers under a published report.

**On a cold cache, run each of those once without `--offline` first.** No market
data is committed — `qv/data/loaders.py` fetches it and caches it outside the
repo, and every manifest pins `start_date` and `end_date`, so a fetch returns
the same window rather than a moving one. The first run populates the cache;
every run after it can refuse the network. Skipping that step means the
documented command is the one guaranteed to fail, which is why both offline
failures now name the remedy in the error itself.

That first run is a burst of requests, and Yahoo throttles a burst by returning
an empty frame — which `yfinance` reports as "possibly delisted", identical to a
ticker that really has gone. Measured on a cold cache, an unthrottled walk of
the dual-momentum universe died on the eighth ticker. `load_prices` therefore
backs off and retries on a bounded schedule, and its final error names both
causes rather than asserting either. A partial cache is kept, so a re-run
resumes.

A refetch will not reproduce the published figures to the last digit — Yahoo
restates and Dartmouth revises, which is why the footer prints data vintages.
What should be stable across vintages is the **verdict and the finding IDs**.
That was measured on 2026-09-12 against a genuinely cold cache: all three
fetching examples came back with identical verdicts and identical finding IDs
on fresh data. If one of those moves, it is a real regression and not drift.

## Testing conventions

- **Every generator is seeded.** A validator whose own test suite is flaky has
  no business telling anyone their results are not reproducible.
- **`tests/test_report_structure.py` reads the rendered HTML the way a person
  would** — no empty table cells, no `nan` in prose, no raw identifiers, every
  Sharpe annualised unless labelled. *Every check in it exists because a first
  draft failed it.* Add to this file whenever a document-level bug appears;
  unit tests cannot see them, because each individual number is right.
- **Slow tests (`-m slow`) are the correctness evidence** — bootstrap coverage
  simulations, the ground-truth sweep, the leakage calibration table. Do not
  delete them to speed up CI.
- **Negative controls are not optional.** Any test asserting that something
  passes must be paired with a variant that must fail. The dangerous outcome
  for a behavioural test is not a false alarm but a clean bill of health earned
  by the corruption never reaching the strategy, which looks identical to an
  honest result from outside.

## Chart style

One colourblind-safe palette, applied consistently: a single accent for the
observed strategy, grey for nulls and benchmarks, and **red reserved
exclusively for failure thresholds**, so red always means the same thing
wherever it appears. Two palettes rather than one CSS filter, because inverting
a chart turns a blue accent orange and a red threshold cyan.

No 3D, no gradients, no dual axes, no chartjunk. A chart that does not change
the reader's belief is worse than no chart in a document whose whole argument
is about unearned confidence.

Two traps worth knowing, both found by looking at a rendered page rather than
by a failing test:

- **An annotation placed outside the axes wrecks the layout silently.**
  `bbox_inches="tight"` grows the figure to include it, and the plot ends up
  squeezed into a corner of a mostly empty image.
- **A discrete sweep is plotted on evenly spaced positions, not its values.**
  Otherwise the interesting step gets crushed into a twentieth of the width.

## Prose style

The reports and docstrings are part of the deliverable. Explain *why*, not
just *what* — a docstring that restates the signature is noise. Where a
decision was arrived at by measurement, record the measurement, because that is
what stops someone undoing it later. Where something was deliberately cut, say
so and say why.

Avoid hedging that softens a real finding, and avoid confidence a number has
not earned. The tool's credibility rests entirely on the reader believing it
would tell them bad news.
