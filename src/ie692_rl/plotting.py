"""Report figures written by ``tune.py``: one PNG per problem and figure type.

    learning_curves_<problem>.png   exact optimality gap of the current policy vs environment steps
                                    (mean over the test seeds, band = +-1 standard error)
    tuning_<problem>.png            the grid search: every combination's selection metric per method

Titles carry only the problem's display name and legends the methods' display names (``label`` in
``config.cfg``); descriptions belong in the figure captions. Math is rendered with matplotlib's
mathtext in STIX fonts (LaTeX syntax such as ``$V^{\\pi}(s_0)$``, no TeX installation needed).

Colors follow the method, never its rank: each method gets a fixed categorical slot from its position
in ``[tuning] methods`` (validated palette, adjacent pairs CVD-safe), so a method keeps its color in
every figure even when other methods are skipped.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.typing import RcKeyType

from ie692_rl.experiment import Row, best_per_method, gap_curve

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_SECONDARY, AXIS, GRID = "#0b0b0b", "#3d3c39", "#a3a29b", "#e4e3dd"
CURVE_POINTS = np.linspace(0, 1, 201)
STYLE: dict[RcKeyType, Any] = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": True,
    "savefig.dpi": 200,
}
GAP = r"$V^{\pi_t}(s_0) - V^{\star}(s_0)$"
TUNING_METRICS = {  # selection -> (value column, standard-error column, y-axis label)
    "gap_lower": (
        "mean_gap_lower",
        "gap_lower_standard_error",
        r"Lower Bound on $V^{\pi}(s_0) - V^{\star}(s_0)$",
    ),
    "exact_gap": ("mean_final_gap", "standard_error", r"Optimality Gap $V^{\pi}(s_0) - V^{\star}(s_0)$"),
}


def method_colors(methods: Sequence[str]) -> dict[str, str]:
    if len(methods) > len(SERIES):
        raise ValueError("At most eight methods can be colored; facet or fold the rest")
    return dict(zip(methods, SERIES, strict=False))


def figure_path(output_dir: Path, kind: str, problem: str) -> Path:
    return output_dir / f"{kind}_{problem.lower()}.png"


def step_unit(budget: int) -> tuple[float, str]:
    """Scale and axis-label unit so step counts read as small numbers (no '1e7' offset text)."""
    if budget >= 2_000_000:
        return 1e6, "Millions"
    if budget >= 2_000:
        return 1e3, "Thousands"
    return 1.0, ""


def _style_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(color=AXIS, labelcolor=INK_SECONDARY)
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)


def _decimal_axis(axis: Any) -> None:
    """At most ~6 ticks at round steps (1, 2, 2.5, 5 x 10^k), plain decimals with no scientific offset."""
    axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def plot_learning_curves(
    history: Sequence[Row],
    budgets: dict[str, int],
    methods: Sequence[str],
    labels: dict[str, str],
    problem_labels: dict[str, str],
    output_dir: Path,
) -> list[Path]:
    """One figure per problem: mean exact gap (+-1 SE band) of the policy in use vs environment steps.

    The y axis spans 0 to just above the highest curve (the shared starting gap of the uniform policy),
    so the initial drop and the late differences between methods are both visible.
    """
    colors, written = method_colors(methods), []
    for problem in dict.fromkeys(r["problem"] for r in history):
        budget = budgets[problem]
        scale, unit = step_unit(budget)
        x = CURVE_POINTS * budget / scale
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
            top = 0.0
            for method in methods:
                runs: dict[int, list[Row]] = {}
                for r in history:
                    if r["problem"] == problem and r["method"] == method:
                        runs.setdefault(r["seed"], []).append(r)
                if not runs:
                    continue
                curves = np.array([gap_curve(h, budget, CURVE_POINTS) for h in runs.values()])
                mean = curves.mean(axis=0)
                se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else 0 * mean
                ax.fill_between(x, mean - se, mean + se, color=colors[method], alpha=0.15, linewidth=0)
                ax.plot(x, mean, color=colors[method], linewidth=2, label=labels.get(method, method))
                top = max(top, float((mean + se).max()))
            ax.set_xlim(0, x[-1])
            ax.set_ylim(0, 1.05 * top if top > 0 else 1.0)
            _decimal_axis(ax.xaxis)
            _decimal_axis(ax.yaxis)
            xlabel = f"Environment Steps ({unit})" if unit else "Environment Steps"
            _style_axes(ax, problem_labels.get(problem, problem), xlabel, f"Optimality Gap {GAP}")
            fig.legend(loc="outside lower center", ncol=min(3, len(ax.lines)), frameon=False)
            path = figure_path(output_dir, "learning_curves", problem)
            fig.savefig(path)
            plt.close(fig)
        written.append(path)
    return written


def plot_tuning(
    rows: Sequence[Row],
    methods: Sequence[str],
    labels: dict[str, str],
    problem_labels: dict[str, str],
    output_dir: Path,
    selection: str = "gap_lower",
) -> list[Path]:
    """One figure per problem: every grid-search combination's selection metric (+-1 SE), grouped by
    method; the chosen combination is ringed and combinations without an update are hollow."""
    value_key, se_key, ylabel = TUNING_METRICS[selection]
    colors, chosen, written = method_colors(methods), best_per_method(rows, selection), []
    for problem in dict.fromkeys(r["problem"] for r in rows):
        present = [m for m in methods if any(r["problem"] == problem and r["method"] == m for r in rows)]
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(1.25 * len(present) + 2.4, 4.4), layout="constrained")
            for x, method in enumerate(present):
                points = [r for r in rows if r["problem"] == problem and r["method"] == method]
                offsets = np.linspace(-0.28, 0.28, len(points)) if len(points) > 1 else np.zeros(1)
                for dx, r in zip(offsets, points, strict=True):
                    se = r.get(se_key, np.nan)
                    hollow = r["min_updates"] == 0
                    ax.errorbar(
                        x + dx,
                        r[value_key],
                        yerr=se if np.isfinite(se) else None,
                        fmt="o",
                        markersize=6,
                        color=colors[method],
                        mfc="white" if hollow else colors[method],
                        elinewidth=1,
                        capsize=0,
                        zorder=3,
                    )
                    if chosen.get((problem, method)) is r:
                        ax.plot(
                            x + dx,
                            r[value_key],
                            "o",
                            markersize=15,
                            fillstyle="none",
                            color=INK,
                            markeredgewidth=1.4,
                            zorder=4,
                        )
            rotate = len(present) > 3  # long method names only need rotating when there are many
            ax.set_xticks(
                range(len(present)),
                [labels.get(m, m) for m in present],
                rotation=25 if rotate else 0,
                ha="right" if rotate else "center",
            )
            ax.set_xlim(-0.6, len(present) - 0.4)
            _decimal_axis(ax.yaxis)
            top = ax.get_ylim()[1]
            ax.set_ylim(-0.04 * top, top)  # a small margin so combinations at exactly 0 stay visible
            _style_axes(ax, problem_labels.get(problem, problem), "", ylabel)
            ax.grid(axis="x", visible=False)
            markers = [
                Line2D([], [], marker="o", markersize=12, fillstyle="none", color=INK, ls="", label="Chosen"),
                Line2D([], [], marker="o", markersize=6, mfc="white", color=AXIS, ls="", label="No Update"),
            ]
            ax.legend(handles=markers, loc="upper left", frameon=False)
            path = figure_path(output_dir, "tuning", problem)
            fig.savefig(path)
            plt.close(fig)
        written.append(path)
    return written
