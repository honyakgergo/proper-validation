"""Inline-SVG charts for the report.

Five charts, each carrying a claim. Anything merely decorative is cut, because
in a document whose whole argument is about unearned confidence, a chart that
does not change the reader's belief is worse than no chart.

**Every number drawn here must also appear in report.json.** The JSON is
authoritative; a chart may never make a claim the machine-readable output
cannot back. Each function therefore returns a ``chart_spec`` alongside the
SVG, holding exactly the values plotted - which is also what the tests assert
on, since snapshotting pixels is flaky and tells you nothing about correctness.

Matplotlib to inline SVG rather than a JavaScript charting library: the report
must be a single self-contained file with no CDN, it has to work offline five
years from now, and a few hundred kilobytes of vector markup beats a bundle.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "PALETTES",
    "THEMES",
    "PALETTE",
    "Chart",
    "unavailable",
    "haircut_cascade",
    "max_sharpe_null",
    "break_even_curve",
    "pbo_panel",
    "regime_equity_curve",
    "risk_distributions",
    "lookahead_horizon",
    "execution_delay_fragility",
    "universe_coverage_chart",
]

#: One palette per theme, applied consistently. A single accent for the
#: observed strategy, grey for nulls and benchmarks, and red reserved
#: *exclusively* for failure thresholds - so red always means the same thing
#: wherever it appears.
#:
#: Two palettes rather than one CSS filter over a single set of images: an
#: inverted chart turns a blue accent orange and a red threshold cyan, which
#: destroys exactly the colour convention the report relies on.
PALETTES = {
    "light": {
        "strategy": "#1f4e79",
        "strategy_light": "#9dc3e6",
        "null": "#9a9a9a",
        "null_fill": "#d9d9d9",
        "benchmark": "#6a6a6a",
        "fail": "#c0392b",
        "grid": "#e6e6e6",
        "text": "#222222",
        "muted": "#666666",
        "axis": "#cccccc",
        "background": "white",
    },
    "dark": {
        "strategy": "#6cb2eb",
        "strategy_light": "#2c5578",
        "null": "#8a8a8a",
        "null_fill": "#3a3f45",
        "benchmark": "#a8a8a8",
        "fail": "#f0705e",
        "grid": "#333940",
        "text": "#e8e8e8",
        "muted": "#a0a0a0",
        "axis": "#4a5058",
        "background": "#1b1f24",
    },
}

THEMES = tuple(PALETTES)

#: Kept for callers that want the default palette without naming a theme.
PALETTE = PALETTES["light"]


def _style(theme: str) -> dict:
    palette = PALETTES[theme]
    return {
    "figure.facecolor": palette["background"],
    "axes.facecolor": palette["background"],
    "axes.edgecolor": palette["axis"],
    "axes.labelcolor": palette["text"],
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": palette["grid"],
    "grid.linewidth": 0.7,
    "xtick.color": palette["muted"],
    "ytick.color": palette["muted"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.labelcolor": palette["text"],
    "text.color": palette["text"],
    "font.family": "sans-serif",
    "svg.fonttype": "none",
    }


@dataclass(frozen=True)
class Chart:
    """A rendered chart and the exact numbers behind it."""

    name: str
    title: str
    svg: str
    spec: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "available": self.available,
            "reason": self.reason,
            "spec": self.spec,
        }


def unavailable(name: str, title: str, reason: str) -> Chart:
    """A placeholder for a chart this tier cannot support.

    Rendering "not available, and here is why" is part of the tiered-honesty
    contract: the report says what it could not test rather than quietly
    omitting it.
    """
    svg = (
        '<div class="chart-unavailable">'
        f"<strong>{title}</strong><br><span>{reason}</span></div>"
    )
    return Chart(name=name, title=title, svg=svg, spec={}, available=False, reason=reason)


def _finish(fig, name: str, title: str, spec: dict[str, Any]) -> Chart:
    """Serialise a figure to inline-ready SVG."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=False, dpi=100)
    plt.close(fig)
    svg = buffer.getvalue()
    # Strip the XML prolog and DOCTYPE so the markup can sit inside <body>.
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.IGNORECASE)
    return Chart(name=name, title=title, svg=svg.strip(), spec=spec)


#: Vertices above which a line is thinned for drawing. A decade of daily data
#: is 2500 points, and an SVG path with that many vertices costs a couple of
#: hundred kilobytes to say something a fifth as many would say identically.
_MAX_LINE_VERTICES = 900


def _decimate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Thin a long line for drawing, always keeping the final point.

    Only ever applied to what is drawn; the ``chart_spec`` and report.json
    always carry the statistics of the full series.
    """
    if y.size <= _MAX_LINE_VERTICES:
        return x, y
    step = int(np.ceil(y.size / _MAX_LINE_VERTICES))
    keep = np.unique(np.append(np.arange(0, y.size, step), y.size - 1))
    return x[keep], y[keep]


def _compounded(r: np.ndarray) -> np.ndarray:
    """Cumulative compounded return, matching the performance table.

    Falls back to the cumulative sum for a series containing a total loss,
    where the compounded path is pinned at -100% and stops being informative.
    """
    if np.any(r <= -1.0):
        return np.cumsum(r)
    return np.cumprod(1.0 + r) - 1.0


def _clean(value) -> Any:
    """JSON-safe scalar: NaN and infinity become null."""
    if isinstance(value, (int, float, np.floating, np.integer)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def haircut_cascade(
    stages: list[tuple[str, float, float | None, float | None]], theme: str = "light"
) -> Chart:
    """The hero chart: the whole audit in one image.

    ``stages`` is an ordered list of ``(label, value, ci_low, ci_high)``, with an
    optional fifth element carrying a short note to print under the marker. It
    runs from the claimed Sharpe through each adjustment in turn.

    Not every stage cuts. Lo annualisation *raises* the Sharpe of a negatively
    autocorrelated series, and a chart that promised only haircuts would have to
    hide that - so the title says adjustment rather than correction, and each
    stage is labelled with the test that produced it.

    The markers are joined, because five dots in a column read as five separate
    measurements rather than one figure being adjusted five times.

    The note exists for censored stages. A BHY haircut that absorbs the whole
    t-statistic lands at exactly zero for any result below its threshold, so the
    marker is a floor and not a measurement; printing "0.00" with nothing beside
    it says the strategy has no edge, which is not what the test established.
    """
    if not stages:
        return unavailable("haircut_cascade", "Sharpe after each adjustment",
                           "no Sharpe stages were computed")

    labels = [s[0] for s in stages]
    values = [float(s[1]) for s in stages]
    lows = [s[2] for s in stages]
    highs = [s[3] for s in stages]
    notes = [s[4] if len(s) > 4 else None for s in stages]

    pal = PALETTES[theme]
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(stages) + 1.4))
        y = np.arange(len(stages))[::-1]

        ax.plot(values, y, color=pal["strategy"], linewidth=1.0, alpha=0.35, zorder=2)

        for yi, value, lo, hi, note in zip(y, values, lows, highs, notes):
            if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
                ax.plot([lo, hi], [yi, yi], color=pal["strategy_light"], linewidth=6,
                        solid_capstyle="round", zorder=2)
            ax.plot([value], [yi], "o", color=pal["strategy"], markersize=8, zorder=3)
            ax.annotate(
                f"{value:.2f}",
                (value, yi),
                textcoords="offset points",
                xytext=(0, 11),
                ha="center",
                fontsize=8,
                color=pal["text"],
                weight="bold",
            )
            if note:
                # Left-aligned near the left edge, where a centred note would
                # run off the axis and sit under the tick labels.
                near_left = value <= min(values) + 0.2 * (max(values) - min(values) or 1.0)
                ax.annotate(
                    note,
                    (value, yi),
                    textcoords="offset points",
                    xytext=(2 if near_left else 0, -17),
                    ha="left" if near_left else "center",
                    fontsize=7.5,
                    color=pal["fail"],
                    style="italic",
                )

        ax.axvline(0.0, color=pal["fail"], linewidth=1.2, linestyle="--", zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Annualised Sharpe ratio")
        ax.set_title("Sharpe after each adjustment")
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.18)

    return _finish(
        fig,
        "haircut_cascade",
        "Sharpe after each adjustment",
        {
            "stages": [
                {
                    "label": lab,
                    "value": _clean(v),
                    "ci_low": _clean(lo),
                    "ci_high": _clean(hi),
                    "note": note,
                }
                for lab, v, lo, hi, note in zip(labels, values, lows, highs, notes)
            ],
            "zero_line": 0.0,
        },
    )


def max_sharpe_null(
    null_distribution, observed: float, analytic_expected_max: float | None = None,
    n_trials: int | None = None, theme: str = "light",
) -> Chart:
    """The null distribution of the maximum Sharpe, with the winner marked.

    The chart that makes selection bias legible: when the reported strategy
    sits inside the distribution of "best of N tries on data with no edge",
    the argument is over.
    """
    draws = np.asarray(null_distribution, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return unavailable("max_sharpe_null", "Best-of-N Sharpe under the null",
                           "no usable null distribution was simulated")

    percentile = float(np.mean(draws <= observed))

    pal = PALETTES[theme]
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        counts, edges, _ = ax.hist(
            draws, bins=min(60, max(12, draws.size // 30)),
            color=pal["null_fill"], edgecolor=pal["null"], linewidth=0.5,
        )
        top = counts.max() if len(counts) else 1.0

        ax.axvline(observed, color=pal["strategy"], linewidth=2.2, zorder=5)
        ax.annotate(
            f"reported\n{observed:.3f}",
            (observed, top * 0.97),
            textcoords="offset points",
            xytext=(6, 0),
            ha="left",
            va="top",
            fontsize=8,
            color=pal["strategy"],
            weight="bold",
        )
        if analytic_expected_max is not None and math.isfinite(analytic_expected_max):
            ax.axvline(analytic_expected_max, color=pal["benchmark"], linewidth=1.2,
                       linestyle=":", zorder=4)
            ax.annotate(
                "analytic E[max]",
                (analytic_expected_max, top * 0.55),
                textcoords="offset points",
                xytext=(-6, 0),
                ha="right",
                fontsize=7.5,
                color=pal["benchmark"],
            )

        trials = "" if n_trials is None else f" across {n_trials} trials"
        ax.set_xlabel(f"Best Sharpe{trials} when there is no edge")
        ax.set_ylabel("Simulations")
        ax.set_title("Where the reported result sits against a search with no edge")

    return _finish(
        fig,
        "max_sharpe_null",
        "Where the reported result sits against a search with no edge",
        {
            "observed": _clean(observed),
            "percentile": _clean(percentile),
            "n_sims": int(draws.size),
            "n_trials": n_trials,
            "null_mean": _clean(float(np.mean(draws))),
            "null_p05": _clean(float(np.quantile(draws, 0.05))),
            "null_p50": _clean(float(np.quantile(draws, 0.50))),
            "null_p95": _clean(float(np.quantile(draws, 0.95))),
            "analytic_expected_max": _clean(analytic_expected_max),
        },
    )


def break_even_curve(
    cost_bps, net_sharpe, break_even_bps: float | None = None,
    realistic_bps: tuple[float, float] | None = None,
    periods_per_year: int = 1, theme: str = "light",
) -> Chart:
    """Sharpe against cost, with the realistic cost band shaded.

    The most clarifying single number in the report, made visual: if the
    shaded band sits to the right of the crossing, the strategy does not
    survive being traded.

    Both labels are placed off the curve and boxed. The break-even label used to
    sit up and to the left of the crossing, which is precisely where a
    downward-sloping line passes, so the curve struck straight through the text.
    Below and to the left is the one quadrant next to the crossing that the
    curve cannot enter.
    """
    x = np.asarray(cost_bps, dtype=float)
    root = math.sqrt(periods_per_year)
    y = np.asarray(net_sharpe, dtype=float) * root
    if x.size < 2 or x.size != y.size:
        return unavailable("break_even_curve", "Sharpe against trading cost",
                           "positions were not supplied, so turnover cannot be measured")

    pal = PALETTES[theme]
    boxed = dict(facecolor=pal["background"], edgecolor="none", alpha=0.85, pad=1.5)
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.plot(x, y, color=pal["strategy"], linewidth=2.0, zorder=3)
        ax.axhline(0.0, color=pal["fail"], linewidth=1.2, linestyle="--", zorder=2)

        if realistic_bps is not None:
            lo, hi = realistic_bps
            ax.axvspan(lo, hi, color=pal["fail"], alpha=0.12, zorder=1)
            # Anchored to the top of the axes rather than to whatever the limits
            # happened to be mid-draw, and left-aligned when the band sits
            # against the left spine, where a centred label runs off the figure.
            span_mid = (lo + hi) / 2
            at_left = span_mid < x[0] + 0.1 * (x[-1] - x[0])
            ax.annotate(
                "realistic cost",
                (span_mid, 1.0),
                xycoords=("data", "axes fraction"),
                textcoords="offset points",
                xytext=(4 if at_left else 0, -11),
                ha="left" if at_left else "center",
                va="top",
                fontsize=7.5,
                color=pal["fail"],
                bbox=boxed,
                zorder=5,
            )

        if (
            break_even_bps is not None
            and math.isfinite(break_even_bps)
            and x[0] <= break_even_bps <= x[-1]
        ):
            ax.plot([break_even_bps], [0.0], "o", color=pal["strategy"], markersize=8, zorder=4)
            ax.annotate(
                f"break-even {break_even_bps:.1f} bps",
                (break_even_bps, 0.0),
                textcoords="offset points",
                xytext=(-10, -14),
                ha="right",
                va="top",
                fontsize=8,
                weight="bold",
                color=pal["strategy"],
                bbox=boxed,
                zorder=5,
            )

        ax.set_xlabel("Cost per unit of traded notional (basis points)")
        ax.set_ylabel(
            "Net annualised Sharpe ratio" if periods_per_year > 1 else "Net Sharpe ratio"
        )
        ax.set_title("How much trading cost the edge can absorb")

    return _finish(
        fig,
        "break_even_curve",
        "How much trading cost the edge can absorb",
        {
            "cost_bps": [_clean(v) for v in x.tolist()],
            "net_sharpe": [_clean(v) for v in y.tolist()],
            "break_even_bps": _clean(break_even_bps),
            "realistic_bps": list(realistic_bps) if realistic_bps else None,
            "periods_per_year": int(periods_per_year),
        },
    )


def pbo_panel(
    logits, is_best_sharpe, oos_of_is_best, pbo: float,
    periods_per_year: int = 1, theme: str = "light",
) -> Chart:
    """Two questions CSCV can actually answer, one panel each.

    Left: how the in-sample winner ranks out of sample, which is what PBO
    counts. Right: what it *earns* out of sample, which is the question a
    reader asks next and a rank cannot answer.

    The conventional second panel - in-sample Sharpe against out-of-sample
    Sharpe - is deliberately not drawn. CSCV splits one fixed sample into
    complementary halves, so the two means sum to a constant and the scatter
    collapses onto a line of slope near -1 whatever the strategy does. It looks
    like devastating evidence of decay and is an identity. The module already
    refuses to report that slope as evidence; drawing it would smuggle it back
    in through the picture.
    """
    lg = np.asarray(logits, dtype=float)
    ys = np.asarray(oos_of_is_best, dtype=float)
    if lg.size < 2:
        return unavailable("pbo_panel", "Probability of backtest overfitting",
                           "a trial matrix is required, and none was supplied")

    root = math.sqrt(periods_per_year)
    ys_ann = ys[np.isfinite(ys)] * root
    median_oos = float(np.median(ys)) if ys.size else math.nan

    pal = PALETTES[theme]
    with plt.rc_context(_style(theme)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

        ax1.hist(lg, bins=40, color=pal["null_fill"], edgecolor=pal["null"],
                 linewidth=0.5)
        ax1.axvline(0.0, color=pal["fail"], linewidth=1.4, linestyle="--")
        ax1.set_xlabel("Logit of out-of-sample rank")
        ax1.set_ylabel("Combinations")
        ax1.set_title(f"PBO = {pbo:.1%}", fontsize=10)

        if ys_ann.size:
            ax2.hist(ys_ann, bins=40, color=pal["null_fill"], edgecolor=pal["null"],
                     linewidth=0.5)
            ax2.axvline(0.0, color=pal["fail"], linewidth=1.4, linestyle="--")
            ax2.axvline(float(np.median(ys_ann)), color=pal["strategy"], linewidth=1.6)
            ax2.annotate(
                f"median {np.median(ys_ann):.2f}",
                xy=(float(np.median(ys_ann)), 0.94),
                xycoords=("data", "axes fraction"),
                xytext=(5, 0), textcoords="offset points",
                fontsize=8, color=pal["strategy"], weight="bold",
            )
        ax2.set_xlabel(
            "Out-of-sample Sharpe"
            + (" (annualised)" if periods_per_year > 1 else " (per period)")
        )
        ax2.set_ylabel("Combinations")
        ax2.set_title("What the winner earns out of sample", fontsize=10)

        fig.tight_layout()

    return _finish(
        fig,
        "pbo_panel",
        "Does the selection procedure generalise?",
        {
            "pbo": _clean(pbo),
            "n_combinations": int(lg.size),
            "fraction_negative_logit": _clean(float(np.mean(lg <= 0))),
            "median_oos_sharpe": _clean(median_oos),
            "median_oos_sharpe_annualised": _clean(median_oos * root),
            "probability_of_loss": _clean(float(np.mean(ys < 0))),
        },
    )


def regime_equity_curve(
    returns,
    regime_mask=None,
    regime_label: str = "high volatility",
    benchmark_returns=None,
    benchmark_label: str = "benchmark",
    dates=None,
    theme: str = "light",
) -> Chart:
    """Cumulative return with the high-volatility regime shaded.

    Needs only a return series, so it appears in every statistical report. It
    turns "the edge only existed in
    one regime" from a row in a table into something the reader cannot unsee -
    which is why it is worth the space even though the regime split is also
    reported numerically.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return unavailable("regime_equity_curve", "Equity curve by regime",
                           "too few return observations to plot")

    equity = _compounded(r)
    x = np.arange(equity.size)
    dated = False
    if dates is not None:
        d = np.asarray(dates)
        if d.size == r.size:
            x = d
            dated = True

    pal = PALETTES[theme]
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 3.4))

        if regime_mask is not None:
            mask = np.asarray(regime_mask, dtype=bool)
            if mask.size == r.size:
                ax.fill_between(
                    x, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color=pal["null_fill"], alpha=0.65, linewidth=0, zorder=0,
                    label=regime_label,
                )

        if benchmark_returns is not None:
            b = np.asarray(benchmark_returns, dtype=float)
            if b.size == r.size:
                bx, by = _decimate(x, _compounded(b))
                ax.plot(bx, by, color=pal["benchmark"], linewidth=1.3,
                        linestyle="--", zorder=2, label=benchmark_label)

        ex, ey = _decimate(x, equity)
        ax.plot(ex, ey, color=pal["strategy"], linewidth=1.8, zorder=3, label="strategy")
        ax.axhline(0.0, color=pal["muted"], linewidth=0.8, zorder=1)
        ax.set_xlabel("" if dated else "Period")
        if dated:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel("Cumulative return (compounded)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("Equity curve, with the volatile regime shaded")
        ax.legend(loc="upper left")

    spec: dict[str, Any] = {
        "n_obs": int(r.size),
        # Compounded, so the end of the curve is the total return the
        # performance table reports rather than a second, smaller number the
        # reader has to reconcile with it.
        "final_cumulative_return": _clean(float(equity[-1])),
        "max_cumulative_return": _clean(float(equity.max())),
        "min_cumulative_return": _clean(float(equity.min())),
        "regime_label": regime_label,
        "dated": dated,
    }
    if regime_mask is not None:
        mask = np.asarray(regime_mask, dtype=bool)
        if mask.size == r.size:
            # Sums, not compounded: these two are meant to add up to the whole,
            # and compounded contributions do not. The key names say so.
            spec["fraction_in_regime"] = _clean(float(mask.mean()))
            spec["return_in_regime_sum"] = _clean(float(r[mask].sum()))
            spec["return_out_of_regime_sum"] = _clean(float(r[~mask].sum()))

    return _finish(fig, "regime_equity_curve", "Equity curve, with the volatile regime shaded", spec)


def risk_distributions(
    total_return: dict,
    sharpe: dict,
    drawdown: dict,
    return_draws,
    sharpe_draws,
    drawdown_draws,
    baseline_draws=None,
    theme: str = "light",
) -> Chart:
    """Three resampled distributions with the realised value marked on each.

    Drawdown carries a second distribution - a matched random walk - because
    that is the panel where a single realisation is least representative and
    where the reader most needs a baseline to judge it against.
    """
    panels = [
        ("Total return", np.asarray(return_draws, dtype=float), total_return, True, None),
        ("Annualised Sharpe", np.asarray(sharpe_draws, dtype=float), sharpe, False, None),
        (
            "Maximum drawdown",
            np.asarray(drawdown_draws, dtype=float),
            drawdown,
            True,
            None if baseline_draws is None else np.asarray(baseline_draws, dtype=float),
        ),
    ]
    if any(p[1].size < 2 for p in panels):
        return unavailable(
            "risk_distributions",
            "How much of this was the path it took?",
            "too few resamples to form a distribution",
        )

    pal = PALETTES[theme]
    with plt.rc_context(_style(theme)):
        fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
        for ax, (title, draws, stats, as_pct, baseline) in zip(axes, panels):
            finite = draws[np.isfinite(draws)]
            ax.hist(
                finite,
                bins=45,
                color=pal["null_fill"],
                edgecolor=pal["null"],
                linewidth=0.4,
                label="resampled",
            )
            if baseline is not None and baseline.size > 1:
                ax.hist(
                    baseline[np.isfinite(baseline)],
                    bins=45,
                    histtype="step",
                    color=pal["benchmark"],
                    linewidth=1.2,
                    linestyle="--",
                    label="random walk",
                )
            realised = stats.get("realised")
            if realised is not None and math.isfinite(float(realised)):
                ax.axvline(
                    float(realised), color=pal["strategy"], linewidth=2.0, zorder=5,
                    label="realised",
                )
            ax.set_title(title, fontsize=9.5)
            ax.set_yticks([])
            if as_pct:
                ax.xaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
                )
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("Resamples")
        axes[2].legend(loc="upper left", fontsize=6.5)
        fig.tight_layout()

    return _finish(
        fig,
        "risk_distributions",
        "How much of this was the path it took?",
        {
            "total_return": {k: _clean(v) for k, v in total_return.items()},
            "sharpe": {k: _clean(v) for k, v in sharpe.items()},
            "max_drawdown": {k: _clean(v) for k, v in drawdown.items()},
        },
    )


# --------------------------------------------------------------------------
# Engine analysis
# --------------------------------------------------------------------------


def lookahead_horizon(
    gaps, detected, detection_floor: int | None = None, theme: str = "light"
) -> Chart:
    """How far into the future the strategy's decisions reach.

    A signal that illegally reads ``r`` periods ahead is caught by destroying
    data ``g`` periods past it for every ``g <= r`` and by none beyond, so the
    curve falls off a cliff at the true horizon. That makes the shape
    diagnostic rather than decorative: a step down at 1 is a ``shift(-1)``, a
    step at 20 is a centred 40-period window, a curve still positive at 200 is
    a full-sample statistic.

    For an honest strategy every value is zero, and a flat line at zero says
    nothing on its own - which is why the **detection floor** is drawn. It
    marks the shortest look-ahead this sample could have exposed, so the empty
    region left of it is the part of the claim that has not been tested rather
    than the part that passed.
    """
    x = np.asarray(gaps, dtype=float)
    y = np.asarray(detected, dtype=float)
    if x.size < 2 or x.size != y.size:
        return unavailable(
            "lookahead_horizon",
            "How far the strategy reaches into the future",
            "no strategy callable was supplied, so behaviour could not be probed",
        )

    pal = PALETTES[theme]
    leaked = bool(np.nanmax(y) > 0)
    # The floor is clamped into the axes. It used to be drawn at its true
    # value against a range derived only from the observed reach, so a clean
    # run swept 1..8 while the floor sat at 21 - the shaded region and its
    # label landed outside the plot, which wrecked the layout and said nothing.
    floor = None
    if detection_floor and detection_floor > 1:
        floor = float(min(detection_floor, x.max()))

    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.2, 3.0))

        if floor is not None:
            ax.axvspan(x.min(), floor, color=pal["muted"], alpha=0.12, linewidth=0)
            ax.axvline(floor, color=pal["muted"], linewidth=1.1, linestyle="--")

        colour = pal["fail"] if leaked else pal["strategy"]
        ax.fill_between(x, 0.0, y, color=colour, alpha=0.18, step="mid")
        ax.step(x, y, where="mid", color=colour, linewidth=1.9, zorder=3)

        ax.set_xlabel("periods past the decision point that were destroyed")
        ax.set_ylabel("probes whose\npast moved")
        ax.set_ylim(-0.04, 1.08)
        ax.set_xlim(x.min(), x.max())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

        # Label the two regions in the axes rather than leaving a bare zero
        # line to speak for itself. A flat line means nothing without knowing
        # which part of the range was actually testable.
        # The untestable band is only labelled when it is wide enough to
        # hold the text: a floor of 2 or 3 leaves a sliver a few percent of
        # the axis wide, and a centred label there spills over the y-axis.
        # The dashed line marks the boundary well enough on its own.
        span = float(x.max() - x.min()) or 1.0
        if floor is not None and (floor - x.min()) / span > 0.16:
            ax.annotate(
                f"under {int(floor)} periods:\nnot testable here",
                xy=(0.5 * (x.min() + floor), 0.5), xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=8, color=pal["muted"],
            )
        if not leaked:
            tested_from = int(floor) if floor is not None else int(x.min())
            ax.annotate(
                f"{tested_from} to {int(x.max())} periods: nothing found",
                xy=(0.5 * ((floor if floor is not None else x.min()) + x.max()), 0.5),
                xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=8.5, color=pal["strategy"],
            )
            ax.set_title("No dependence on the future, above the testable floor")
        else:
            horizon_x = float(x[y > 0].max())
            ax.annotate(
                f"reaches {int(horizon_x)} periods forward",
                xy=(horizon_x, float(y[x == horizon_x][0])),
                xytext=(10, 16), textcoords="offset points",
                fontsize=8.5, color=pal["fail"],
                arrowprops={"arrowstyle": "-", "color": pal["fail"], "linewidth": 0.9},
            )
            ax.set_title("Look-ahead detected")

    horizon = float(x[y > 0].max()) if leaked else None
    return _finish(
        fig,
        "lookahead_horizon",
        "How far the strategy reaches into the future",
        {
            "gaps": [_clean(v) for v in x],
            "detected": [_clean(v) for v in y],
            "detection_floor": detection_floor,
            "horizon": horizon,
            "leaked": leaked,
        },
    )


def execution_delay_fragility(
    delays, sharpes, periods_per_year: int = 252, theme: str = "light"
) -> Chart:
    """Annualised Sharpe as the same book is traded progressively later.

    The question is what the edge can pay in *time* rather than in fees, and
    the two fail independently: a monthly rotation across liquid ETFs can
    absorb hundreds of basis points and still lose everything to a one-session
    delay, because the signal is really picking up a reversal that has already
    reverted by the time anyone could trade it.

    Zero is marked in red, because a curve crossing it means the strategy
    inverts when traded late - which is a stronger statement than merely
    decaying.
    """
    delay_values = np.asarray(delays, dtype=float)
    y = np.asarray(sharpes, dtype=float)
    if delay_values.size < 2 or delay_values.size != y.size or not np.any(np.isfinite(y)):
        return unavailable(
            "execution_delay_fragility",
            "What the edge is worth if the book is traded late",
            "positions and per-asset returns were not both supplied",
        )
    # Evenly spaced positions, labelled with the real delays. This is a sweep
    # over a handful of discrete values, not a continuous function: plotting
    # 0,1,2,3,5,10 on a linear axis crushes the 0-to-1 step - which is where
    # the entire finding lives - into a twentieth of the width, while giving
    # the widest gap to the two points that matter least.
    x = np.arange(delay_values.size, dtype=float)

    pal = PALETTES[theme]
    base = y[0]
    # Narrower than the statistical charts: this one and the coverage chart sit
    # side by side in the report, so each is drawn at about half the page width
    # and needs its text proportionally larger to stay legible.
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(4.9, 3.4))
        ax.axhline(0.0, color=pal["fail"], linewidth=1.1, linestyle="--")
        ax.plot(x, y, color=pal["strategy"], linewidth=1.8, marker="o",
                markersize=4.5, zorder=3)
        ax.fill_between(x, 0.0, y, color=pal["strategy"], alpha=0.13)

        ax.scatter([x[0]], [base], s=58, facecolor=pal["background"],
                   edgecolor=pal["strategy"], linewidth=1.8, zorder=4)

        # Both labels below the curve. It slopes down from its own maximum at
        # zero delay, so the space above the first point is where the curve is
        # about to be - putting the label there struck it through.
        # The curve descends from its own maximum at zero delay, so the space
        # above the first point is the one region it cannot enter, and the
        # space well below the second point is the other. No leader lines:
        # a leader from the second point to anywhere useful crosses the curve.
        ax.annotate(
            f"as reported: {base:.2f}",
            xy=(x[0], base), xytext=(2, 9), textcoords="offset points",
            ha="left", va="bottom", fontsize=8, color=pal["text"],
            bbox={"facecolor": pal["background"], "edgecolor": "none", "pad": 1.8,
                  "alpha": 0.9},
        )

        if len(y) > 1 and math.isfinite(y[1]) and math.isfinite(base) and base > 0:
            ax.annotate(
                f"{y[1] / base:.0%} of it after one period",
                xy=(x[1], 0.42), xycoords=("data", "axes fraction"),
                ha="left", va="center", fontsize=8, color=pal["muted"],
                bbox={"facecolor": pal["background"], "edgecolor": "none",
                      "pad": 2.0, "alpha": 0.9},
            )

        ax.set_xlabel("execution delay, periods")
        ax.set_ylabel("annualised Sharpe")
        ax.set_title("If the book is traded late")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(d)}" for d in delay_values])
        ax.set_xlim(x.min() - 0.35, x.max() + 0.35)
        top = float(np.nanmax(y))
        # Headroom for the label that sits above the first point.
        ax.set_ylim(min(0.0, float(np.nanmin(y))) - 0.06 * max(top, 0.1), top * 1.24)

    return _finish(
        fig,
        "execution_delay_fragility",
        "What the edge is worth if the book is traded late",
        {
            "delays": [int(v) for v in delay_values],
            "annualised_sharpes": [_clean(v) for v in y],
            "base_sharpe": _clean(base),
            "one_period_retention": _clean(
                y[1] / base if len(y) > 1 and base not in (0.0,) else math.nan
            ),
            "periods_per_year": periods_per_year,
        },
    )


def universe_coverage_chart(coverage, theme: str = "light") -> Chart:
    """When each instrument's history begins and ends, against the sample.

    The visual counterpart to the survivorship declaration. Survivorship
    itself cannot be measured from a return series - a universe of survivors
    looks exactly like a universe - but *inclusion timing* can be seen, and it
    is the same family of error: an instrument added the day it lists was
    chosen in the knowledge that it would exist.

    A clean universe draws as a solid block of equal bars, which is the point:
    "nothing entered mid-sample" becomes something a reader can check at a
    glance instead of a claim they have to take.

    Where a point-in-time membership list was supplied, each row also carries a
    thin grey reference bar for the period the instrument was actually an index
    member. That turns the chart into a comparison rather than a shape: a short
    data bar under an equally short grey bar entered late because the *index*
    added it late, which is correct; a short data bar under a full-length grey
    bar is a hole in the data during a period the strategy should have been able
    to trade it. Grey because it is context, in the same role as a benchmark -
    red stays reserved for failure thresholds.
    """
    instruments = tuple(getattr(coverage, "instruments", ()) or ())
    if not instruments:
        return unavailable(
            "universe_coverage",
            "When each instrument enters and leaves the universe",
            "the unaligned price frame was not supplied",
        )

    first = np.asarray(coverage.first_index, dtype=float)
    last = np.asarray(coverage.last_index, dtype=float)
    n = int(coverage.n_obs)
    labels = tuple(getattr(coverage, "labels", ()) or ())
    # Only the *unexplained* edges draw as failures. A late entrant the
    # membership list accounts for is not a defect, and colouring it like one
    # would contradict the finding printed above the chart.
    ragged = set(coverage.unexplained_late) | set(coverage.early_exits)
    ragged |= set(getattr(coverage, "traded_before_membership", ()) or ())
    member_first = tuple(getattr(coverage, "membership_first", ()) or ())
    member_last = tuple(getattr(coverage, "membership_last", ()) or ())
    has_membership = len(member_first) == len(instruments)

    pal = PALETTES[theme]
    # Half-width, to sit beside the delay chart. The height still grows with
    # the universe, but it is capped: a 40-name universe drawn at 0.26in a row
    # would be taller than the page.
    height = min(4.4, max(2.6, 0.2 * len(instruments) + 1.3))
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(4.9, height))
        order = np.argsort(first, kind="stable")
        for row, i in enumerate(order):
            name = instruments[i]
            is_ragged = name in ragged
            if has_membership and member_first[i] >= 0:
                # Drawn first and taller, so the data bar reads as sitting
                # inside the period the name was eligible.
                ax.barh(
                    row,
                    member_last[i] - member_first[i] + 1,
                    left=member_first[i],
                    height=0.82,
                    color=pal["null"],
                    alpha=0.30,
                )
            ax.barh(
                row,
                last[i] - first[i] + 1,
                left=first[i],
                height=0.62,
                color=pal["fail"] if is_ragged else pal["strategy"],
                alpha=0.85 if is_ragged else 0.55,
            )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([instruments[i] for i in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, n)
        ax.grid(axis="y", visible=False)

        if labels and len(labels) == n:
            ticks = np.linspace(0, n - 1, min(6, n)).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([labels[t] for t in ticks], fontsize=7.5)
            ax.set_xlabel("")
        else:
            ax.set_xlabel("observation")

        complete = bool(coverage.complete)
        if complete and has_membership:
            title = f"All {len(instruments)} cover the period they were members"
        elif complete:
            title = f"All {len(instruments)} instruments span the sample"
        elif has_membership:
            title = (
                f"{len(ragged)} of {len(instruments)} do not cover their membership"
            )
        else:
            title = f"{len(ragged)} of {len(instruments)} do not span the sample"
        ax.set_title(title)
        if has_membership:
            ax.plot(
                [], [], color=pal["null"], alpha=0.30, linewidth=6,
                label="index member",
            )
            ax.plot([], [], color=pal["strategy"], alpha=0.55, linewidth=6, label="data")
            ax.legend(loc="lower right", fontsize=7, frameon=False)

    return _finish(
        fig,
        "universe_coverage",
        "When each instrument enters and leaves the universe",
        {
            "instruments": list(instruments),
            "coverage": [_clean(v) for v in coverage.coverage],
            "late_entrants": list(coverage.late_entrants),
            "unexplained_late": list(coverage.unexplained_late),
            "explained_by_membership": list(coverage.explained_late),
            "membership_first": list(member_first),
            "membership_last": list(member_last),
            "early_exits": list(coverage.early_exits),
            "complete": bool(coverage.complete),
            "n_obs": n,
        },
    )
