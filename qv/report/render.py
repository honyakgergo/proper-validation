"""Render an audit to a self-contained HTML file and a JSON payload.

No CDN, no JavaScript, no external fonts. The report has to open from a
downloads folder in five years and look the same, and the committed example
reports are the artifact most readers will judge without running anything.

The JSON is authoritative. Every number drawn in a chart also appears there,
so a reader can check any claim the page makes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, StrictUndefined
from markupsafe import Markup

from qv.audit import AuditReport
from qv.report import charts as ch
from qv.types import Severity, Suite, Verdict

__all__ = [
    "CHART_ORDER",
    "STATISTICAL_CHARTS",
    "ENGINE_CHARTS",
    "charts_for",
    "build_charts",
    "render_html",
    "render_json",
    "write_report",
]

#: The charts each suite carries, in the order they are built. Tests derive
#: their expected counts from these rather than hardcoding a number, so adding
#: a chart does not break seven unrelated assertions.
STATISTICAL_CHARTS = (
    "haircut_cascade",
    "max_sharpe_null",
    "break_even_curve",
    "pbo_panel",
    "risk_distributions",
    "regime_equity_curve",
)

ENGINE_CHARTS = (
    "lookahead_horizon",
    "execution_delay_fragility",
    "universe_coverage",
)

#: Every chart, in report order. A full report carries all of them.
CHART_ORDER = STATISTICAL_CHARTS + ENGINE_CHARTS


def charts_for(suite: Suite) -> tuple[str, ...]:
    """Which charts a suite is entitled to draw."""
    names: tuple[str, ...] = ()
    if suite.runs_statistical:
        names += STATISTICAL_CHARTS
    if suite.runs_engine:
        names += ENGINE_CHARTS
    return names

_VERDICT_CLASS = {
    Verdict.FALSIFIED: "verdict-bad",
    Verdict.WEAKENED: "verdict-warn",
    Verdict.NOT_FALSIFIED: "verdict-ok",
    Verdict.INSUFFICIENT_DATA: "verdict-unknown",
}

_SEVERITY_CLASS = {
    Severity.CRITICAL: "sev-critical",
    Severity.HIGH: "sev-high",
    Severity.MEDIUM: "sev-medium",
    Severity.LOW: "sev-low",
    Severity.INFO: "sev-info",
}



def _engine_charts(report: AuditReport, theme: str) -> list[ch.Chart]:
    """The engine suite's three charts.

    Kept apart from the statistical ones rather than interleaved, because the
    two suites are independent and a reader of an engine-only report should not
    have to step over placeholders for tests that were never in scope.
    """
    out: list[ch.Chart] = []

    horizon = report.chart_data.get("lookahead_horizon")
    if horizon:
        gaps, detected, floor = horizon
        out.append(ch.lookahead_horizon(gaps, detected, detection_floor=floor, theme=theme))
    else:
        out.append(
            ch.unavailable(
                "lookahead_horizon",
                "How far the strategy reaches into the future",
                "no strategy callable was supplied, so its behaviour could not be probed",
            )
        )

    delay = report.chart_data.get("execution_delay")
    if delay:
        out.append(
            ch.execution_delay_fragility(
                delay[0], delay[1], periods_per_year=report.periods_per_year, theme=theme
            )
        )
    else:
        out.append(
            ch.unavailable(
                "execution_delay_fragility",
                "What the edge is worth if the book is traded late",
                "positions and per-asset returns were not both supplied",
            )
        )

    coverage = report.chart_data.get("universe_coverage")
    if coverage is not None:
        out.append(ch.universe_coverage_chart(coverage, theme=theme))
    else:
        out.append(
            ch.unavailable(
                "universe_coverage",
                "When each instrument enters and leaves the universe",
                "the price frame before common-calendar alignment was not supplied",
            )
        )

    return out


def build_charts(
    report: AuditReport, returns=None, theme: str = "light"
) -> list[ch.Chart]:
    """Build this suite's charts, degrading to a placeholder where unsupported."""
    out: list[ch.Chart] = []

    # An engine-only report is nothing but the engine charts. Returning early
    # rather than filtering at the end keeps a reader of an engine report from
    # stepping over placeholders for statistical tests that were never in scope.
    if not report.suite.runs_statistical:
        return _engine_charts(report, theme)

    stages = report.headline_sharpe_stages
    out.append(
        ch.haircut_cascade(stages, theme=theme)
        if stages
        else ch.unavailable("haircut_cascade", "Sharpe after each adjustment",
                            "no Sharpe stages were computed")
    )

    null = report.sections.get("null_max")
    distribution = report.chart_data.get("null_distribution")
    if null and distribution:
        out.append(
            ch.max_sharpe_null(
                distribution,
                observed=null["observed_sharpe"],
                analytic_expected_max=null.get("analytic_expected_max"),
                n_trials=null.get("n_trials"),
                theme=theme,
            )
        )
    else:
        out.append(
            ch.unavailable(
                "max_sharpe_null",
                "Where the reported result sits against a search with no edge",
                "no trial count was declared, so the best-of-N null cannot be simulated",
            )
        )

    costs = report.sections.get("costs")
    cost_curve = report.chart_data.get("cost_curve")
    if costs and cost_curve:
        realistic = costs.get("realistic_bps")
        out.append(
            ch.break_even_curve(
                cost_curve[0],
                cost_curve[1],
                break_even_bps=costs.get("break_even_bps"),
                realistic_bps=tuple(realistic) if realistic else None,
                periods_per_year=report.periods_per_year,
                theme=theme,
            )
        )
    else:
        out.append(
            ch.unavailable(
                "break_even_curve",
                "How much trading cost the edge can absorb",
                "no position series was supplied, so turnover cannot be measured",
            )
        )

    pbo = report.sections.get("pbo")
    pbo_arrays = report.chart_data.get("pbo")
    if pbo and pbo_arrays:
        out.append(
            ch.pbo_panel(
                *pbo_arrays,
                pbo["pbo"],
                periods_per_year=report.periods_per_year,
                theme=theme,
            )
        )
    else:
        out.append(
            ch.unavailable(
                "pbo_panel",
                "Does the selection procedure generalise?",
                "requires the matrix of trial returns, and none was supplied",
            )
        )

    risk = report.sections.get("risk")
    risk_draws = report.chart_data.get("risk_draws")
    if risk and risk_draws:
        out.append(
            ch.risk_distributions(
                risk["total_return"], risk["sharpe"], risk["max_drawdown"],
                risk_draws[0], risk_draws[1], risk_draws[2], risk_draws[3],
                theme=theme,
            )
        )
    else:
        out.append(
            ch.unavailable(
                "risk_distributions",
                "How much of this was the path it took?",
                "too few observations to resample",
            )
        )

    if returns is not None:
        out.append(
            ch.regime_equity_curve(
                returns,
                regime_mask=report.chart_data.get("regime_mask"),
                benchmark_returns=report.chart_data.get("benchmark_returns"),
                benchmark_label=report.chart_data.get("benchmark_name", "benchmark"),
                dates=report.chart_data.get("dates"),
                theme=theme,
            )
        )
    else:
        out.append(
            ch.unavailable("regime_equity_curve", "Equity curve, with the volatile regime shaded",
                           "return series unavailable")
        )

    if report.suite.runs_engine:
        out.extend(_engine_charts(report, theme))
    return out


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy types and non-finite floats for json.dump."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def render_json(report: AuditReport, chart_list: list[ch.Chart] | None = None) -> str:
    """The machine-readable payload. Authoritative for every number."""
    payload = _jsonable(report.to_dict())
    if chart_list is not None:
        payload["charts"] = [_jsonable(c.to_dict()) for c in chart_list]
    return json.dumps(payload, indent=2)


_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ r.name }} - backtest audit</title>
<style>
  /* Light is the default. The toggle below is a checkbox and a :has() rule,
     so the theme switches with no JavaScript at all - the report stays a
     single inert file that will open from a downloads folder in five years. */
  :root {
    --ink:#1a1a1a; --muted:#666; --rule:#e3e3e3; --page:#ffffff; --panel:#fafafa;
    --accent:#1f4e79; --bad:#c0392b; --warn:#b9770e; --ok:#1e7a46;
    --bad-bg:#fdf3f2; --warn-bg:#fdf8ef; --ok-bg:#f2f9f5;
    --code-bg:#f4f4f4; --remedy-bg:#f7f9fb;
  }
  body:has(#theme-toggle:checked) {
    --ink:#e8e8e8; --muted:#a0a0a0; --rule:#333940; --page:#1b1f24; --panel:#22272e;
    --accent:#6cb2eb; --bad:#f0705e; --warn:#e0a458; --ok:#5cc98a;
    --bad-bg:#2c2020; --warn-bg:#2c2618; --ok-bg:#1c2a22;
    --code-bg:#2a2f36; --remedy-bg:#232a32;
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:0 1.25rem 4rem; font:15px/1.6 -apple-system,BlinkMacSystemFont,
         "Segoe UI",Roboto,Helvetica,Arial,sans-serif; color:var(--ink); background:var(--page); }
  main { max-width: 900px; margin: 0 auto; }
  h1 { font-size:1.7rem; margin:2rem 0 .25rem; letter-spacing:-.01em; }
  h2 { font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.35rem;
       border-bottom:2px solid var(--rule); }
  p, li { margin:.5rem 0; }
  .sub { color:var(--muted); margin:0 0 1.5rem; }

  /* Theme switch */
  #theme-toggle { position:absolute; opacity:0; pointer-events:none; }
  .theme-switch { position:absolute; top:1rem; right:1rem; cursor:pointer;
                  font-size:.75rem; letter-spacing:.05em; text-transform:uppercase;
                  border:1px solid var(--rule); border-radius:999px; padding:.3rem .8rem;
                  color:var(--muted); user-select:none; background:var(--panel); }
  .theme-switch:hover { color:var(--ink); border-color:var(--muted); }
  .theme-switch .to-dark { display:inline; }
  .theme-switch .to-light { display:none; }
  body:has(#theme-toggle:checked) .theme-switch .to-dark { display:none; }
  body:has(#theme-toggle:checked) .theme-switch .to-light { display:inline; }
  #theme-toggle:focus-visible + .theme-switch { outline:2px solid var(--accent); }

  .banner { padding:1rem 1.1rem; border-radius:6px; margin:1.25rem 0;
            border-left:5px solid var(--muted); background:var(--panel); }
  .verdict-bad { border-left-color:var(--bad); background:var(--bad-bg); }
  .verdict-warn { border-left-color:var(--warn); background:var(--warn-bg); }
  .verdict-ok { border-left-color:var(--ok); background:var(--ok-bg); }
  .verdict-unknown { border-left-color:var(--muted); }
  .verdict-title { font-size:1.2rem; font-weight:700; }
  .tier { display:inline-block; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
          background:var(--accent); color:var(--page); padding:.2rem .55rem; border-radius:3px; }

  table { border-collapse:collapse; width:100%; margin:.75rem 0; font-size:.9rem; }
  th, td { padding:.45rem .6rem; border-bottom:1px solid var(--rule); vertical-align:top;
           text-align:left; }
  th { font-weight:600; color:var(--muted); font-size:.8rem; text-transform:uppercase;
       letter-spacing:.04em; }
  /* Numeric headers must sit over their own column, not at the far left of it. */
  th.num, td.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  tbody tr:last-child td { border-bottom:none; }

  /* Charts span the full text column rather than sitting at their intrinsic
     matplotlib width, which was narrower than the prose beside them. */
  figure.chart { margin:1.5rem 0 2.25rem; padding:0; width:100%; }
  figure.chart svg { width:100%; height:auto; display:block; margin:0 auto; }
  /* Two charts that answer adjacent questions read better beside each other
     than stacked, and each is authored at about half the page width so its
     text stays legible when scaled down. Collapses to one column on a phone,
     where side-by-side would make both illegible. */
  .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:1.25rem;
               align-items:end; margin:1.5rem 0 2.25rem; }
  .chart-row figure.chart { margin:0; }
  @media (max-width:820px) { .chart-row { grid-template-columns:1fr; } }
  .light-only, .dark-only { display:block; }
  .dark-only { display:none; }
  body:has(#theme-toggle:checked) .light-only { display:none; }
  body:has(#theme-toggle:checked) .dark-only { display:block; }
  .chart-unavailable { padding:1.75rem; border:1px dashed var(--rule); border-radius:5px;
                       color:var(--muted); text-align:center; font-size:.9rem; }

  .sev { display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.05em;
         text-transform:uppercase; padding:.15rem .45rem; border-radius:3px; color:#fff;
         white-space:nowrap; }
  .sev-critical { background:#c0392b; } .sev-high { background:#d35400; }
  .sev-medium { background:#b9770e; } .sev-low { background:#7f8c8d; }
  .sev-info { background:#95a5a6; }
  .finding { border:1px solid var(--rule); border-radius:5px; padding:.8rem 1rem; margin:.7rem 0; }
  .finding-head { display:flex; gap:.6rem; align-items:baseline; flex-wrap:wrap; }
  .finding-id { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem;
                color:var(--muted); }
  .finding-title { font-weight:650; }
  .remedy { font-size:.88rem; background:var(--remedy-bg); border-left:3px solid var(--accent);
            padding:.5rem .75rem; margin-top:.6rem; }
  .note { font-size:.85rem; color:var(--muted); font-style:italic; }
  ul.gaps li { font-size:.92rem; }
  footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--rule);
           font-size:.8rem; color:var(--muted); }
  code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em;
         background:var(--code-bg); padding:.05rem .3rem; border-radius:3px; }
  @media print { .theme-switch { display:none; } }
</style></head><body>

<input type="checkbox" id="theme-toggle" aria-label="Switch to dark theme">
<label class="theme-switch" for="theme-toggle"><span class="to-dark">Dark</span><span class="to-light">Light</span></label>

<main>

<h1>{{ r.name }}</h1>
<p class="sub"><span class="tier">{{ r.suite.label }}</span>
&nbsp; {{ r.n_obs }} observations &middot; {{ r.periods_per_year }} periods per year</p>
<p class="sub" style="margin-top:-.35rem">{{ r.suite.question }}
{%- if r.suite.value == "statistical" %} This report says nothing about how the
backtest was built; the engine analysis is what asks that.
{%- elif r.suite.value == "engine" %} This report says nothing about whether the
strategy makes money; the statistical validation is what asks that.
{%- endif %}</p>

<div class="banner {{ verdict_class }}">
  <div class="verdict-title">{{ r.verdict.label }}</div>
  <p style="margin:.4rem 0 0">
    This tool falsifies; it cannot validate. It finds reasons to disbelieve a backtest,
    never reasons to believe one.
    {% if r.verdict.value == 'not_falsified' %}Nothing below broke this backtest. That is a
    much weaker statement than "the strategy works", and the gaps listed under
    <em>What could not be tested</em> are part of the result.{% elif r.verdict.value ==
    'insufficient_data' %}There is not enough data here to draw a conclusion either way.
    {% else %}The findings below are what the tests actually established. Nothing here says
    the strategy cannot work - only that this backtest does not show that it does.{% endif %}
  </p>
</div>

<h2>Findings ({{ findings|length }})</h2>
{% if not findings %}<p>No findings. Note that this means the tests applied did not falsify the
backtest, which is a much weaker statement than the backtest being sound.</p>{% endif %}
{% for fd in findings %}
<div class="finding">
  <div class="finding-head">
    <span class="sev {{ sev_class(fd) }}">{{ fd.severity.label }}</span>
    <span class="finding-title">{{ fd.title }}</span>
    <span class="finding-id">{{ fd.id }}</span>
  </div>
  <p>{{ fd.detail }}</p>
  {% if fd.remediation %}<div class="remedy"><strong>What to do:</strong>
    {{ fd.remediation }}</div>{% endif %}
</div>
{% endfor %}

{% if gaps %}
<h2>What could not be tested</h2>
<ul class="gaps">{% for g in gaps %}<li>{{ g }}</li>{% endfor %}</ul>
{% endif %}

{% if r.suite.runs_statistical %}
<h2>Headline</h2>
<p class="note">Every Sharpe ratio on this page is annualised to {{ r.periods_per_year }} periods
unless the row says otherwise.
{% if conv.risk_free_supplied %}Every one of them is also computed on returns <strong>in excess of
the risk-free rate</strong> ({{ pct(conv.risk_free_annualised, 2) }} a year over this sample), and
the benchmark is treated identically. Total return, drawdown and Calmar are left on total returns,
because those describe the path an investor lived through.{% else %}<strong>No risk-free rate was
supplied</strong>, so these are total-return Sharpes. They overstate the risk-adjusted result by
roughly the cash rate divided by the volatility - which flatters the least volatile series in any
comparison on this page.{% endif %}</p>
{% if conv.idle_cash_understatement > 0 %}
<p class="note">The book is on average {{ pct(1 - conv.mean_idle_weight, 0) }} invested. If the
backtest did not credit the idle balance at the bill rate, the return series understates by about
{{ pct(conv.idle_cash_understatement, 2) }} a year - in the opposite direction to the adjustment
above, and worth checking before reading the excess figures as final.</p>
{% endif %}
{{ chart_html('haircut_cascade') }}
<table>
  <thead><tr><th>Measure</th><th class="num">Value</th><th class="num">95% interval</th></tr></thead>
  <tbody>
  <tr><td>Annualised Sharpe (naive annualisation)</td>
      <td class="num">{{ f(s.annualised_naive.value) }}</td>
      <td class="num">{{ f(b.naive_ci_low) }} to {{ f(b.naive_ci_high) }}</td></tr>
  <tr><td>Autocorrelation-adjusted (Lo 2002)</td>
      <td class="num">{{ f(s.annualised_adjusted.value) }}</td>
      <td class="num">{{ f(b.annualised_ci_low) }} to {{ f(b.annualised_ci_high) }}</td></tr>
  <tr><td>Per-period Sharpe</td><td class="num">{{ f(s.periodic.value, 4) }}</td>
      <td class="num">{{ f(b.ci_low, 4) }} to {{ f(b.ci_high, 4) }}</td></tr>
  </tbody>
</table>
<p class="note">The interval is a stationary bootstrap with a Politis-White block length of
{{ f(b.block_length, 1) }} over {{ b.n_boot }} resamples - not the analytic standard error,
which assumes away the serial dependence these returns have.</p>
<p class="note">Lo's adjustment moves the annualisation factor from {{ f(s.factor_naive, 2) }}
to {{ f(s.factor_lo, 2) }}, {% if s.autocorrelation_haircut > 0 %}cutting the annualised Sharpe
by {{ pct(s.autocorrelation_haircut, 1) }}: these returns are positively autocorrelated, so naive
sqrt(q) annualisation was too generous.{% else %}<em>raising</em> the annualised Sharpe by
{{ pct(-s.autocorrelation_haircut, 1) }}: these returns are negatively autocorrelated at short
lags, so naive sqrt(q) annualisation was too harsh.{% endif %}</p>
{% if s.autocorrelation_note %}<p class="note">Caveat: {{ s.autocorrelation_note }}.</p>{% endif %}

{% if perf %}
<h2>Performance{% if benchperf %} against {{ benchperf.name }}{% endif %}</h2>
{{ chart_html('regime_equity_curve') }}
<table>
  <thead><tr><th>Metric</th><th class="num">{{ perf.name }}</th>
    {% if benchperf %}<th class="num">{{ benchperf.name }}</th>
    <th class="num">Difference</th>{% endif %}</tr></thead>
  <tbody>
  {% for row in performance_rows %}
  <tr><td>{{ row.label }}</td><td class="num">{{ row.strategy }}</td>
      {% if benchperf %}<td class="num">{{ row.benchmark }}</td>
      <td class="num">{{ row.difference }}</td>{% endif %}</tr>
  {% endfor %}
  </tbody>
</table>
<p class="note">Both columns are computed the same way over the same periods. Returns are
compounded rather than arithmetic, and maximum drawdown is measured on the compounded path from a
starting value of 1, so it is the loss an investor would have lived through.</p>
{% endif %}

{% if attr or bench %}
<h2>Attribution: is the edge just factor exposure?</h2>
{% if attr %}
<p>Regression of excess strategy returns on the Fama-French factors. Alpha is the intercept - what
is left once the factors are accounted for. A p-value above 0.05 means the strategy has not been
shown to add anything the factors do not already explain.</p>
<table>
  <thead><tr><th>Term</th><th class="num">Estimate</th><th class="num">t</th>
    <th class="num">p</th><th class="num">Significant</th></tr></thead>
  <tbody>
  <tr><td><strong>Alpha (annualised)</strong></td>
      <td class="num"><strong>{{ pct(attr.alpha_annualised, 2) }}</strong></td>
      <td class="num">{{ f(attr.alpha_tstat) }}</td>
      <td class="num">{{ f(attr.alpha_p_value, 3) }}</td>
      <td class="num">{{ "yes" if attr.alpha_p_value is not none and attr.alpha_p_value < 0.05
                         else "no" }}</td></tr>
  {% for l in attr.loadings %}
  <tr><td>{{ l.name }}</td><td class="num">{{ f(l.beta, 3) }}</td>
      <td class="num">{{ f(l.tstat) }}</td><td class="num">{{ f(l.p_value, 3) }}</td>
      <td class="num">{{ "yes" if l.significant_at_5pct else "no" }}</td></tr>
  {% endfor %}
  </tbody>
</table>
<p class="note">R-squared {{ pct(attr.r_squared, 1) }} over {{ attr.n_obs }} observations.
Standard errors: {{ attr.cov_type }}. Regression residuals from a return series are serially
correlated, and plain OLS standard errors would understate the uncertainty on alpha - the one
coefficient the whole exercise turns on.</p>
{% endif %}
{% if bench %}
<h3>Against a volatility-matched {{ bench.benchmark_name }}</h3>
<table>
  <thead><tr><th>Measure</th><th class="num">Value</th></tr></thead>
  <tbody>
  <tr><td>Strategy Sharpe (annualised)</td>
      <td class="num">{{ f(bench.strategy_sharpe_annualised, 3) }}</td></tr>
  <tr><td>{{ bench.benchmark_name }} at matched volatility (annualised)</td>
      <td class="num">{{ f(bench.scaled_benchmark_sharpe_annualised, 3) }}</td></tr>
  <tr><td>Sharpe the strategy adds</td>
      <td class="num">{{ f(bench.excess_sharpe_annualised, 3) }}</td></tr>
  <tr><td>Leverage required to match volatility</td>
      <td class="num">{{ f(bench.leverage, 2) }}x</td></tr>
  <tr><td>Correlation to {{ bench.benchmark_name }}</td>
      <td class="num">{{ f(bench.correlation, 2) }}</td></tr>
  <tr><td>Beta to {{ bench.benchmark_name }}</td><td class="num">{{ f(bench.beta, 2) }}</td></tr>
  </tbody>
</table>
{% endif %}
{% endif %}

{% if risk %}
<h2>How much of this was the path it took?</h2>
<p>Maximum drawdown is an extreme-value statistic, and one realisation of it says much less than
it appears to. These are the same returns resampled in blocks, so the distributions show how
different the headline figures could plausibly have been on another draw of the same process.</p>
{{ chart_html('risk_distributions') }}
<table>
  <thead><tr><th>Statistic</th><th class="num">Realised</th>
    <th class="num">Resampled median</th><th class="num">5th to 95th percentile</th>
    <th class="num">Realised percentile</th></tr></thead>
  <tbody>
  <tr><td>Total return</td>
      <td class="num">{{ pct(risk.total_return.realised, 2) }}</td>
      <td class="num">{{ pct(risk.total_return.median, 2) }}</td>
      <td class="num">{{ pct(risk.total_return.ci_low, 2) }} to
          {{ pct(risk.total_return.ci_high, 2) }}</td>
      <td class="num">{{ pct(risk.total_return.percentile, 0) }}</td></tr>
  <tr><td>Annualised Sharpe</td>
      <td class="num">{{ f(risk.sharpe.realised, 2) }}</td>
      <td class="num">{{ f(risk.sharpe.median, 2) }}</td>
      <td class="num">{{ f(risk.sharpe.ci_low, 2) }} to {{ f(risk.sharpe.ci_high, 2) }}</td>
      <td class="num">{{ pct(risk.sharpe.percentile, 0) }}</td></tr>
  <tr><td><strong>Maximum drawdown</strong></td>
      <td class="num"><strong>{{ pct(risk.max_drawdown.realised, 2) }}</strong></td>
      <td class="num">{{ pct(risk.max_drawdown.median, 2) }}</td>
      <td class="num">{{ pct(risk.max_drawdown.ci_low, 2) }} to
          {{ pct(risk.max_drawdown.ci_high, 2) }}</td>
      <td class="num">{{ pct(risk.max_drawdown.percentile, 0) }}</td></tr>
  <tr><td>Drawdown a matched random walk would expect</td>
      <td class="num">{{ pct(risk.max_drawdown.baseline_median, 2) }}</td>
      <td class="num">same drift and volatility</td>
      <td class="num">IID normal, no fat tails</td>
      <td class="num">{{ pct(risk.max_drawdown.baseline_percentile, 0) }}</td></tr>
  </tbody>
</table>
<p class="note">Both comparisons understate, and in opposite ways.
{{ risk.note }}. The matched random walk assumes IID normal returns, so it has neither volatility
clustering nor fat tails and will also come out shallower than reality. A realised drawdown far
outside <em>both</em> means the loss came from the order in which returns arrived, and neither
baseline can recreate it.</p>
{% if risk.drawdown_path_dependent %}
<p class="note"><strong>That is the case here:</strong> the realised drawdown is deeper than 95%
of resampled paths, so it is a property of this particular sequence rather than of the return
distribution. Size the strategy for the deeper end of the range, not for the realised figure.</p>
{% endif %}
{% endif %}

{% if sel %}
<h2>Selection bias</h2>
{{ chart_html('max_sharpe_null') }}
<table>
  <thead><tr><th>Measure</th><th class="num">Value</th></tr></thead>
  <tbody>
  <tr><td>Trials declared</td><td class="num">{{ sel.deflated_sharpe.n_trials }}</td></tr>
  <tr><td>Expected best-of-N per-period Sharpe under the null</td>
      <td class="num">{{ f(sel.deflated_sharpe.expected_max_sharpe, 4) }}</td></tr>
  <tr><td>Observed per-period Sharpe ({{ sel.basis }})</td>
      <td class="num">{{ f(sel.deflated_sharpe.observed_sharpe, 4) }}</td></tr>
  <tr><td><strong>Deflated Sharpe Ratio</strong></td>
      <td class="num"><strong>{{ f(sel.deflated_sharpe.dsr, 3) }}</strong></td></tr>
  <tr><td>Survives the selection-adjusted bar</td>
      <td class="num">{{ "yes" if sel.deflated_sharpe.survives else "no" }}</td></tr>
  <tr><td>Minimum backtest length (target Sharpe 1.0)</td>
      <td class="num">{{ f(sel.minimum_backtest_length_years, 1) }} years vs
          {{ f(sel.sample_years, 1) }} available</td></tr>
  <tr><td>BHY-adjusted t-statistic</td>
      <td class="num">{{ f(sel.haircut.observed_tstat) }} &rarr;
          {{ f(sel.haircut.adjusted_tstat) }}{% if sel.haircut.censored %}
          (fully absorbed){% endif %}</td></tr>
  <tr><td>t-statistic BHY needs at {{ sel.deflated_sharpe.n_trials }} trials</td>
      <td class="num">{{ f(sel.haircut.required_tstat) }}{% if sel.haircut.required_sharpe %}
          &middot; annualised Sharpe {{ f(sel.haircut.required_sharpe) }}{% endif %}</td></tr>
  <tr><td>Rank of this result among the trials</td>
      <td class="num">{% if sel.haircut.used_trial_family %}{{ sel.haircut.rank }} of
          {{ sel.haircut.n_tests }} by t-statistic{% else %}assumed best of
          {{ sel.haircut.n_tests }} - no trial matrix{% endif %}</td></tr>
  </tbody>
</table>
<p class="note">{{ sel.basis_note }} Trial-Sharpe dispersion
{{ sel.deflated_sharpe.trial_sharpe_std_source }}.
Sharpe ratios in this table are per-period because that is the scale the Deflated Sharpe is
defined on; scaling both sides would not change the comparison.</p>
<p class="note">The two selection tests here answer different questions and can disagree. The
Deflated Sharpe asks whether this result beats the <em>best of {{ sel.deflated_sharpe.n_trials }}
draws</em> from a no-edge null; the Harvey-Liu BHY haircut asks whether the t-statistic survives a
false-discovery correction across {{ sel.deflated_sharpe.n_trials }} tests.
{% if sel.deflated_sharpe.survives and not sel.haircut.significant_at_5pct %}<strong>They disagree
on this sample</strong>: the Deflated Sharpe clears its bar at
{{ f(sel.deflated_sharpe.dsr, 3) }} while BHY takes the t-statistic to
{{ f(sel.haircut.adjusted_tstat) }} (p = {{ f(sel.haircut.adjusted_pvalue, 3) }}), which is not
significant. Read the selection-bias conclusion as unsettled rather than passed: one of the two
standard corrections does not clear it.
{% if sel.haircut.censored %}BHY absorbs this t-statistic entirely, which is a floor rather than a
measurement - anything below its threshold lands in the same place. The distance is the number to
read: at {{ sel.deflated_sharpe.n_trials }} trials it wanted
t = {{ f(sel.haircut.required_tstat) }}{% if sel.haircut.required_sharpe %}, an annualised Sharpe
of about {{ f(sel.haircut.required_sharpe) }} on a sample this long{% endif %}, against
{{ f(sel.haircut.observed_tstat) }} here.{% endif %}
{% elif not sel.deflated_sharpe.survives and sel.haircut.significant_at_5pct %}<strong>They
disagree on this sample</strong>: BHY leaves the t-statistic significant at
{{ f(sel.haircut.adjusted_tstat) }} while the Deflated Sharpe does not clear its bar. Read the
selection-bias conclusion as unsettled rather than failed.
{% elif sel.haircut.significant_at_5pct %}Both clear here: BHY leaves the t-statistic at
{{ f(sel.haircut.adjusted_tstat) }}, still significant, and the Deflated Sharpe clears its bar.
{% else %}Both fail here: BHY takes the t-statistic to {{ f(sel.haircut.adjusted_tstat) }} and the
Deflated Sharpe does not clear its bar either.{% endif %}</p>
<p class="note">The trial count prices the configurations declared for <em>this</em> backtest.
An effect taken from published work arrives with a search history of its own - the papers that
found it, and the ones that were never written - and nothing in this report can see that. Read
{{ sel.deflated_sharpe.n_trials }} as a floor on the number of things that were tried, not as the
number of things that were tried.</p>
{% if nm %}
<p class="note">Simulated null: the observed result sits at the
{{ pct(nm.percentile, 1) }} percentile of {{ nm.n_sims }} replications
(p = {{ f(nm.p_value, 3) }}).
{% if nm.analytic_disagrees %}<strong>Validity check:</strong> the simulation and the closed form
disagree by {{ f((nm.agreement_ratio - 1) * 100, 1) }}% - {{ nm.disagreement_direction }}.
{% else %}The simulation and the closed form agree to within
{{ f((nm.agreement_ratio - 1) * 100, 1) }}%, so the normality assumption behind the Deflated
Sharpe holds here.{% endif %}</p>
{% endif %}
{% endif %}

{% if costs %}
<h2>Cost realism</h2>
{{ chart_html('break_even_curve') }}
<table>
  <thead><tr><th>Measure</th><th class="num">Value</th></tr></thead>
  <tbody>
  <tr><td><strong>Break-even cost</strong></td>
      <td class="num"><strong>{{ f(costs.break_even_bps, 1) }} bps</strong></td></tr>
  {% if costs.realistic_bps %}
  <tr><td>Realistic cost for {{ costs.asset_class_label }}</td>
      <td class="num">{{ f(costs.realistic_bps[0], 1) }} to
          {{ f(costs.realistic_bps[1], 1) }} bps</td></tr>
  <tr><td>Margin over the realistic ceiling</td>
      <td class="num">{{ f(costs.margin, 2) }}x</td></tr>
  <tr><td>Survives realistic costs</td>
      <td class="num">{{ "yes" if costs.survives_realistic_costs else "no" }}</td></tr>
  {% endif %}
  <tr><td>Turnover per period</td><td class="num">{{ f(costs.mean_turnover, 3) }}</td></tr>
  <tr><td>Turnover annualised</td><td class="num">{{ f(costs.annual_turnover, 1) }}x</td></tr>
  <tr><td>Gross Sharpe used for the crossing</td>
      <td class="num">{{ f(costs.gross_sharpe, 4) }} per period</td></tr>
  </tbody>
</table>
{% if costs.note %}<p class="note">{{ costs.note }}</p>{% endif %}
{% endif %}

{% if pbo %}
<h2>Does the selection procedure generalise?</h2>
{{ chart_html('pbo_panel') }}
<table>
  <thead><tr><th>Measure</th><th class="num">Value</th></tr></thead>
  <tbody>
  <tr><td><strong>Probability of backtest overfitting</strong></td>
      <td class="num"><strong>{{ pct(pbo.pbo) }}</strong></td></tr>
  <tr><td>Worse than a coin flip</td>
      <td class="num">{{ "yes" if pbo.overfit else "no" }}</td></tr>
  <tr><td>Probability of an out-of-sample loss</td>
      <td class="num">{{ small_pct(pbo.probability_of_loss) }}</td></tr>
  <tr><td>Median out-of-sample rank of the in-sample winner</td>
      <td class="num">{{ f(pbo.median_oos_rank, 3) }}</td></tr>
  <tr><td>Splits evaluated</td><td class="num">{{ pbo.n_combinations }}</td></tr>
  </tbody>
</table>
<p class="note">The degradation slope is deliberately not reported as evidence, and neither is
the in-sample-against-out-of-sample scatter that usually accompanies this chart. CSCV splits one
fixed sample into complementary halves, so the two Sharpes sum to a constant: the scatter falls on
a line of slope near -1 whatever the strategy does, and the apparent decay is an identity rather
than a finding. The right-hand panel shows what the winner actually earns out of sample instead.</p>
{% endif %}

<h2>Robustness</h2>
<table>
  <thead><tr><th>Test</th><th class="num">Result</th><th class="num">Reading</th></tr></thead>
  <tbody>
  {% if reg %}
  <tr><td>Annualised Sharpe, low / high volatility</td>
      <td class="num">{{ f(reg.sharpes_annualised[0], 2) }} /
          {{ f(reg.sharpes_annualised[1], 2) }}</td>
      <td class="num">{{ "same sign" if reg.sign_stable else "sign flips" }}</td></tr>
  {% endif %}
  {% if roll %}
  <tr><td>Annualised Sharpe across {{ roll.sharpes|length }} start dates</td>
      <td class="num">{{ f(roll.min_sharpe_annualised, 2) }} to
          {{ f(roll.max_sharpe_annualised, 2) }}</td>
      <td class="num">{{ "stable" if not roll.start_date_dependent else "start-date dependent" }}</td></tr>
  {% endif %}
  {% if top %}
  <tr><td>Profit concentration vs a normal baseline</td>
      <td class="num">{{ f(top.worst_concentration_ratio, 2) }}x</td>
      <td class="num">{{ "abnormal" if top.abnormally_concentrated else "as expected" }}</td></tr>
  <tr><td>Annualised Sharpe dropping the best 1% / 5% of periods</td>
      <td class="num">{{ f(top.sharpe_after_dropping_top_1pct_annualised, 2) }} /
          {{ f(top.sharpe_after_dropping_top_5pct_annualised, 2) }}</td>
      <td class="num">from {{ f(top.full_sharpe_annualised, 2) }}</td></tr>
  {% endif %}
  {% if flip %}
  <tr><td>Sign-flip randomisation p-value</td>
      <td class="num">{{ f(flip.p_value, 3) }}</td>
      <td class="num">{{ "signal informative" if flip.significant_at_5pct
                         else "not distinguishable" }}</td></tr>
  {% endif %}
  {% if matched %}
  <tr><td>Matched-exposure random entry p-value</td>
      <td class="num">{{ f(matched.p_value, 3) }}</td>
      <td class="num">{{ "timing adds value" if matched.significant_at_5pct
                         else "no timing skill shown" }}</td></tr>
  {% endif %}
  {% if params %}
  <tr><td>Parameter neighbourhood retention</td>
      <td class="num">{{ pct(params.ratio) }}</td>
      <td class="num">{{ "spike" if params.is_spike else "plateau" }}</td></tr>
  {% endif %}
  </tbody>
</table>
{% if top %}
<p class="note">The post-drop Sharpe falls steeply for almost any strategy - removing the best 5%
of periods from a Sharpe-{{ f(top.full_sharpe_annualised, 2) }} series is expected to push it
below zero, which is why the concentration ratio, not the post-drop Sharpe, is what this test
actually reads. A ratio near 1.0 means the profit is no more concentrated than the Sharpe alone
implies.</p>
{% endif %}


{% endif %}

{% if r.suite.runs_engine %}
<h2>Engine analysis: can this backtest be trusted as an implementation?</h2>

<p class="note">Independent of whether the strategy makes money. A sound-looking
number from an unsound implementation is the more dangerous of the two failures,
because nothing in the return series betrays it.</p>

<table>
  <thead><tr><th>Check</th><th class="num">Result</th><th class="num">Detail</th></tr></thead>
  <tbody>
  {% if pert %}
  <tr><td>Reads the future?</td>
      <td class="num">{{ "LEAKED" if pert.leaked else "no" }}</td>
      <td class="num">{% if pert.leaked %}reaches {{ pert.lookahead_span }} periods forward
          {%- elif pert.detection_floor %}none beyond {{ pert.detection_floor }} periods
          {%- else %}signal never moved, so nothing was established{% endif %}</td></tr>
  {% endif %}
  {% if det %}
  <tr><td>Same input, same output?</td>
      <td class="num">{{ "yes" if det.deterministic else "NO" }}</td>
      <td class="num">{{ det.n_calls }} calls compared</td></tr>
  {% endif %}
  {% if deg %}
  <tr><td>Is there a decision to examine?</td>
      <td class="num">{{ "no" if deg.degenerate else "yes" }}</td>
      <td class="num">{{ deg.n_changes }} changes over {{ deg.n_obs }} periods,
          {{ pct(deg.fraction_flat) }} of it flat</td></tr>
  {% endif %}
  {% if delay %}
  <tr><td>Survives being traded a period late?</td>
      <td class="num">{{ pct(delay.one_period_retention) }} retained</td>
      <td class="num">Sharpe {{ f(delay.base_sharpe, 2) }} to
          {{ f(delay.sharpes[1], 2) }} annualised</td></tr>
  {% endif %}
  {% if cover %}
  <tr><td>Universe complete over the sample?</td>
      <td class="num">{{ "yes" if cover.complete else "NO" }}</td>
      <td class="num">{% if cover.complete %}all {{ cover.instruments|length }} instruments
          {%- else %}{{ cover.late_entrants|length }} start late,
          {{ cover.early_exits|length }} stop early{% endif %}</td></tr>
  {% endif %}
  </tbody>
</table>

{% if pert %}
<p class="note">{{ pert.clean_claim }}</p>
{% endif %}

{{ chart_html('lookahead_horizon') }}

{% if delay %}
<p class="note">Trading cost and trading delay are different questions and fail
independently. A monthly rotation across liquid funds can absorb hundreds of
basis points in fees and still lose its entire edge to a single session of
delay, because the signal was picking up a reversal that had already reverted
by the time anyone could act on it.</p>
{% endif %}

<p class="note">A universe of survivors looks exactly like a universe, so
survivorship cannot be read out of the returns - it has to be declared, or
counted against a point-in-time membership list. Inclusion timing is the part
that is visible in the price data on its own: an instrument added the day it
listed was chosen in the knowledge that it would exist.</p>

<div class="chart-row">
{{ chart_html('execution_delay_fragility') }}
{{ chart_html('universe_coverage') }}
</div>
{% endif %}

<footer>
  <p>Universe:
  {% if data.survivorship_basis == 'measured' %}measured against a point-in-time membership list.
     The traded universe is missing {{ data.measurement.n_missing_exited }} of the
     {{ data.measurement.exits_total }} name(s) that left the index during the covered window
     {%- if data.measurement.covered_fraction_of_sample < 0.999 %}, over the
     {{ '%.0f'|format(data.measurement.covered_fraction_of_sample * 100) }}% of the sample the list
     covers{% endif %}. This counts what was excluded, not what it would have returned.
  {% elif data.survivorship_basis == 'declared' and data.universe_point_in_time is true %}declared
     point-in-time - it was not assembled from present-day membership, so nothing entered it with
     hindsight. Declared, not measured.
  {% elif data.survivorship_basis == 'declared' %}declared as present-day membership, which is
     survivorship-biased by construction - see the finding above.
  {% else %}not established. {{ data.survivorship_basis_reason or 'Survivorship leaves no trace in
     a return series, so this question stays open.' }}{% endif %}
  {% if data.universe_note %} {{ data.universe_note }}{% endif %}</p>
  <p>Generated {{ p.generated_utc }} by <code>proper_validation</code>
     {%- if engine %} {{ engine }}{% endif %}.
     Data hash <code>{{ p.data_hash }}</code> &middot; seed {{ p.seed }} &middot;
     {{ p.n_boot }} bootstrap resamples &middot; Python {{ p.python }} &middot;
     numpy {{ p.numpy }}.</p>
  {% for sentence in vintages %}
  <p>{{ sentence }}</p>
  {% endfor %}
  <p>Re-running with the same inputs and seed, on this same source, reproduces every number
     above. <code>report.json</code> beside this file carries every figure shown here.</p>
</footer>
</main></body></html>
"""


#: (label, key, kind). One place where the comparison table decides how each
#: metric is rendered, so the strategy and benchmark columns cannot drift apart.
#: Every row here produces a value in every column - a table with dashes in it
#: reads as a gap in the audit rather than as "not applicable".
_PERFORMANCE_ROWS = (
    ("Total return", "total_return", "pct"),
    ("Annualised return", "annualised_return", "pct"),
    ("Annualised volatility", "annualised_volatility", "pct"),
    ("Sharpe ratio", "sharpe", "num"),
    ("Sortino ratio", "sortino", "num"),
    ("Maximum drawdown", "max_drawdown", "pct"),
    ("Calmar ratio", "calmar", "num"),
    ("Hit rate", "hit_rate", "pct"),
    ("Best period", "best_period", "pct"),
    ("Worst period", "worst_period", "pct"),
    ("Skewness", "skewness", "num"),
    ("Kurtosis", "kurtosis", "num"),
)


def _performance_rows(strategy: dict, benchmark: dict | None, fmt, pct) -> list[dict]:
    """Rows for the side-by-side performance table.

    The difference column is a percentage-point gap for percentage metrics and
    a plain difference for ratios, because "Sharpe 1.2 vs 0.8, +50%" invites
    exactly the misreading it looks like.
    """
    rows = []
    for label, key, kind in _PERFORMANCE_ROWS:
        # Two decimals on percentages: "9.8%" hides the difference between 9.75
        # and 9.84, which is the resolution this column exists to show.
        show = (lambda v: pct(v, 2)) if kind == "pct" else fmt
        left = strategy.get(key) if strategy else None
        right = benchmark.get(key) if benchmark else None
        if left is None or right is None:
            difference = "-"
        else:
            gap = float(left) - float(right)
            scaled = gap * 100 if kind == "pct" else gap
            # A gap that rounds to zero carries no sign. Printed with one it
            # reads as "-0.00", a loss at a resolution the table does not have,
            # or as "+0.00", which is the same lie in the other direction.
            sign = "" if abs(round(scaled, 2)) == 0.0 else "+"
            unit = " pp" if kind == "pct" else ""
            difference = f"{0.0 if not sign else scaled:{sign}.2f}{unit}"
        rows.append({
            "label": label,
            "strategy": show(left),
            "benchmark": show(right) if benchmark else "-",
            "difference": difference,
        })
    return rows


#: How the provenance keys `qv.pipeline` records are named to a reader. A
#: footer is prose, and `factor_alignment` is a variable name, not English.
_VINTAGE_LABELS = {
    "prices": "prices",
    "factors": "Fama-French factors",
    "benchmark": "benchmark",
}


def _engine_identity(provenance: dict) -> str:
    """"0.1.0, commit ab68694, source cf2fa6f9eba5", or "" if unrecorded.

    Each part is optional on its own: a wheel install has no commit to read,
    and a checkout that was never installed has no version. An older report
    rendered before any of this existed still has to render.
    """
    parts = []
    if provenance.get("version"):
        parts.append(str(provenance["version"]))
    if provenance.get("commit"):
        parts.append(f"commit {provenance['commit']}")
    if provenance.get("source_digest"):
        parts.append(f"source {provenance['source_digest']}")
    return ", ".join(parts)


def _vintage_sentences(provenance: dict) -> list[str]:
    """The data vintages, as prose, or an empty list when there are none.

    `qv/data/loaders.py` has always said the vintage "is recorded and surfaced
    in the report footer", and until this existed it reached `report.json` and
    stopped there. It is the fact a reader needs in order to judge whether
    their own re-run should match: Dartmouth revises the factor files, and a
    number reproduced from a different vintage is not the same number.

    A flat-file audit has no vintages at all - the researcher supplied the
    series - and then the footer says nothing rather than printing a label
    with nothing after it.
    """
    vintages = provenance.get("data_vintages") or {}
    if not vintages:
        return []

    dated = [
        f"{_VINTAGE_LABELS[key]} {vintages[key]}"
        for key in _VINTAGE_LABELS
        if vintages.get(key)
    ]
    # Anything the label map does not know about is still someone's data, so
    # it is shown rather than dropped - just without a hand-written name.
    dated += [
        f"{key.replace('_', ' ')} {value}"
        for key, value in vintages.items()
        if key not in _VINTAGE_LABELS and key != "factor_alignment" and value
    ]

    sentences = []
    if dated:
        why = "A number reproduced from a different vintage is not the same number"
        # Name the revision the reader is most likely to hit, but only when it
        # applies: quoting Dartmouth's revision policy under a report with no
        # factors in it is a non-sequitur.
        why += (
            " - the Fama-French files are periodically revised."
            if vintages.get("factors")
            else ", and a price history can be restated after the fact."
        )
        sentences.append("Data vintages: " + ", ".join(dated) + ". " + why)
    # Not a date. Its value is already a sentence about how many sessions the
    # factor join dropped, so it gets its own clause instead of joining a list
    # of dates where it would read as one.
    alignment = vintages.get("factor_alignment")
    if alignment:
        sentences.append(f"Factor alignment: {alignment}.")
    return sentences


def render_html(
    report: AuditReport,
    chart_list: list[ch.Chart],
    dark_charts: list[ch.Chart] | None = None,
) -> str:
    """Render the self-contained HTML report.

    ``dark_charts`` holds the same charts drawn with the dark palette. Both
    sets are embedded and CSS shows one; a single set recoloured by a CSS
    filter would turn the blue strategy accent orange and the red failure
    threshold cyan, destroying the colour convention the report depends on.
    """
    by_name = {c.name: c for c in chart_list}
    dark_by_name = {c.name: c for c in (dark_charts or [])}

    def chart_html(name: str) -> Markup:
        # Markup, not a plain string: autoescaping is on for everything else in
        # this template (finding text is untrusted enough to warrant it), and
        # without this the SVG markup renders as visible angle brackets.
        chart = by_name.get(name)
        if chart is None:
            return Markup("")
        dark = dark_by_name.get(name)
        if dark is None or not chart.available:
            return Markup(f'<figure class="chart">{chart.svg}</figure>')
        return Markup(
            f'<figure class="chart">'
            f'<span class="light-only">{chart.svg}</span>'
            f'<span class="dark-only">{dark.svg}</span>'
            f"</figure>"
        )

    def fmt(value, places: int = 2) -> str:
        if value is None or (isinstance(value, dict) and not value):
            return "-"
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not math.isfinite(f):
            return "infinite" if f > 0 else "-"
        return f"{f:,.{places}f}"

    def pct(value, places: int = 1) -> str:
        if value is None or (isinstance(value, dict) and not value):
            return "-"
        f = float(value)
        return "-" if not math.isfinite(f) else f"{f * 100:.{places}f}%"

    def small_pct(value, places: int = 1) -> str:
        """Like ``pct``, but never rounds a nonzero probability down to zero.

        "0.0%" and "0.0%" read the same and mean different things; a reader
        planning around a tail deserves to know the difference between a
        probability that is small and one that is absent.
        """
        if value is None or (isinstance(value, dict) and not value):
            return "-"
        f = float(value)
        if not math.isfinite(f):
            return "-"
        floor = 10.0 ** (-places) / 100.0
        if 0.0 < f < floor:
            return f"<{floor * 100:.{places}f}%"
        return f"{f * 100:.{places}f}%"

    env = Environment(autoescape=True, undefined=StrictUndefined, trim_blocks=True,
                      lstrip_blocks=True)
    template = env.from_string(_TEMPLATE)

    sections = _jsonable({k: v for k, v in report.sections.items() if not k.startswith("_")})
    return template.render(
        r=report,
        p=report.provenance,
        engine=_engine_identity(report.provenance),
        vintages=_vintage_sentences(report.provenance),
        s=_Dot(sections.get("sharpe", {})),
        b=_Dot(sections.get("bootstrap", {})),
        sel=_Dot(sections["selection"]) if "selection" in sections else None,
        nm=_Dot(sections["null_max"]) if "null_max" in sections else None,
        costs=_Dot(sections["costs"]) if "costs" in sections else None,
        pbo=_Dot(sections["pbo"]) if "pbo" in sections else None,
        attr=_Dot(sections["attribution"]) if "attribution" in sections else None,
        bench=_Dot(sections["benchmark"]) if "benchmark" in sections else None,
        perf=_Dot(sections["performance"]) if "performance" in sections else None,
        risk=_Dot(sections["risk"]) if "risk" in sections else None,
        benchperf=(
            _Dot(sections["benchmark_performance"])
            if "benchmark_performance" in sections
            else None
        ),
        reg=_Dot(sections["regimes"]) if "regimes" in sections else None,
        roll=_Dot(sections["rolling_origin"]) if "rolling_origin" in sections else None,
        top=_Dot(sections["top_days"]) if "top_days" in sections else None,
        flip=_Dot(sections["sign_flip"]) if "sign_flip" in sections else None,
        matched=_Dot(sections["matched_exposure"]) if "matched_exposure" in sections else None,
        params=_Dot(sections["parameters"]) if "parameters" in sections else None,
        conv=_Dot(sections.get("conventions", {})),
        data=_Dot(sections.get("data", {})),
        pert=_Dot(sections["perturbation"]) if "perturbation" in sections else None,
        det=_Dot(sections["determinism"]) if "determinism" in sections else None,
        deg=_Dot(sections["degeneracy"]) if "degeneracy" in sections else None,
        delay=_Dot(sections["execution_delay"]) if "execution_delay" in sections else None,
        cover=(
            _Dot(sections["universe_coverage"])
            if "universe_coverage" in sections
            else None
        ),
        performance_rows=_performance_rows(
            sections.get("performance", {}),
            sections.get("benchmark_performance"),
            fmt,
            pct,
        ),
        gaps=report.not_tested,
        findings=report.findings_by_severity(),
        verdict_class=_VERDICT_CLASS[report.verdict],
        sev_class=lambda f: _SEVERITY_CLASS[f.severity],
        chart_html=chart_html,
        f=fmt,
        pct=pct,
        small_pct=small_pct,
    )


class _Dot(dict):
    """Attribute access over a plain dict, so templates read cleanly.

    A missing key yields an empty ``_Dot`` rather than raising or returning
    ``None``. That keeps chained access such as ``s.annualised_naive.value``
    safe when a whole section is absent - which happens legitimately, since
    which sections exist depends on the tier. An empty ``_Dot`` is falsy, so
    ``{% if section %}`` still behaves, and the formatter renders it as a dash.
    """

    def __getattr__(self, item):
        # Dunders must fail normally. Answering them would make hasattr(obj,
        # "__html__") true, and Jinja would then try to call the empty _Dot it
        # got back instead of escaping the value.
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        value = self.get(item)
        return _Dot(value) if isinstance(value, dict) else _Dot() if value is None else value

    # A missing value compares false against anything, so a template can write
    # `{% if x.y > 0 %}` without first checking that the whole section exists.
    # Without these, an absent section raises a TypeError mid-render and takes
    # the entire report down rather than omitting one sentence.
    def __gt__(self, other):
        return False

    def __lt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __le__(self, other):
        return False

    def __neg__(self):
        return _Dot()

    def __float__(self):
        raise TypeError("missing value has no float representation")


def write_report(
    report: AuditReport, directory: str | Path, returns=None, basename: str = "report"
) -> tuple[Path, Path]:
    """Write ``report.html`` and ``report.json``, returning both paths."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    chart_list = build_charts(report, returns=returns, theme="light")
    dark_charts = build_charts(report, returns=returns, theme="dark")
    html_path = out / f"{basename}.html"
    json_path = out / f"{basename}.json"

    html_path.write_text(render_html(report, chart_list, dark_charts), encoding="utf-8")
    json_path.write_text(render_json(report, chart_list), encoding="utf-8")
    return html_path, json_path
