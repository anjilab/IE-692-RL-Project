"""Report figures written by ``tune.py`` (matplotlib, static PNG).

Colors follow the method, never its rank: each method gets a fixed categorical slot from its
position in ``[tuning] methods`` (validated palette, adjacent pairs CVD-safe), so a method keeps its
color in every figure even when other methods are skipped.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ie692_rl.experiment import Row, best_per_method, gap_curve

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, INK, INK_SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
CURVE_POINTS = np.linspace(0, 1, 101)


def method_colors(methods: Sequence[str]) -> dict[str, str]:
    if len(methods) > len(SERIES):
        raise ValueError("At most eight methods can be colored; facet or fold the rest")
    return dict(zip(methods, SERIES, strict=False))


def _style(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)


def plot_tuning(rows: Sequence[Row], methods: Sequence[str], path: Path) -> None:
    """One panel per problem: every combination's mean tuning gap (+-1 SE), one column per method.

    The chosen combination is ringed; hollow markers made no update in the tuning budget.
    """
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    colors, chosen = method_colors(methods), best_per_method(rows)
    fig, axes = plt.subplots(
        1,
        len(problems),
        figsize=(4.8 * len(problems), 3.8),
        squeeze=False,
        layout="constrained",
        facecolor=SURFACE,
    )
    for ax, problem in zip(axes[0], problems, strict=True):
        present = [m for m in methods if any(r["problem"] == problem and r["method"] == m for r in rows)]
        for x, method in enumerate(present):
            points = [r for r in rows if r["problem"] == problem and r["method"] == method]
            offsets = np.linspace(-0.25, 0.25, len(points)) if len(points) > 1 else np.zeros(1)
            for dx, r in zip(offsets, points, strict=True):
                hollow = r["min_updates"] == 0
                se = r.get("standard_error", np.nan)
                ax.errorbar(
                    x + dx,
                    r["mean_final_gap"],
                    yerr=se if np.isfinite(se) else None,
                    fmt="o",
                    markersize=6,
                    color=colors[method],
                    mfc=SURFACE if hollow else colors[method],
                    elinewidth=1,
                    capsize=0,
                    zorder=3,
                )
                if chosen.get((problem, method)) is r:
                    ax.plot(
                        x + dx,
                        r["mean_final_gap"],
                        "o",
                        markersize=15,
                        fillstyle="none",
                        color=INK,
                        markeredgewidth=1.4,
                        zorder=4,
                    )
        ax.set_xticks(range(len(present)), present, rotation=20, ha="right")
        budget = next(r["tuning_budget"] for r in rows if r["problem"] == problem)
        _style(ax, f"{problem}: tuning budget {budget:,} steps", "", "mean final gap to optimal")
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "Grid search: each dot is one combination (ring = chosen, hollow = no update)", fontsize=11, color=INK
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_learning_curves(
    history: Sequence[Row], budgets: dict[str, int], methods: Sequence[str], path: Path
) -> None:
    """Full runs, per problem: exact gap vs observations and estimated V(s0) vs iteration (mean +- SE).

    These are the two progressions the guideline asks for (value and observations); the exact gap is
    the noise-free version of the value curve.
    """
    problems = list(dict.fromkeys(r["problem"] for r in history))
    colors = method_colors(methods)
    fig, axes = plt.subplots(
        2,
        len(problems),
        figsize=(5.2 * len(problems), 7.0),
        squeeze=False,
        layout="constrained",
        facecolor=SURFACE,
    )
    for j, problem in enumerate(problems):
        budget = budgets[problem]
        for method in methods:
            runs: dict[int, list[Row]] = {}
            for r in history:
                if r["problem"] == problem and r["method"] == method:
                    runs.setdefault(r["seed"], []).append(r)
            if not runs:
                continue
            curves = np.array([gap_curve(h, budget, CURVE_POINTS) for h in runs.values()])
            mean = curves.mean(axis=0)
            se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
            x = CURVE_POINTS * budget
            axes[0, j].fill_between(x, mean - se, mean + se, color=colors[method], alpha=0.15, linewidth=0)
            axes[0, j].plot(x, mean, color=colors[method], linewidth=2, label=method)

            # Latest estimate of each seed at each budget fraction (NaN before a seed's first update).
            estimates = np.array(
                [gap_curve(h, budget, CURVE_POINTS, key="estimated_value") for h in runs.values()]
            )
            counts = np.isfinite(estimates).sum(axis=0)
            totals = np.where(np.isfinite(estimates), estimates, 0.0).sum(axis=0)
            mean_estimate = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
            axes[1, j].plot(x, mean_estimate, color=colors[method], linewidth=2, label=method)
        _style(
            axes[0, j],
            f"{problem}: exact gap of the current policy",
            "environment steps used",
            "gap to optimal (lower is better)",
        )
        _style(
            axes[1, j],
            f"{problem}: latest estimated V(s0) (mean over seeds)",
            "environment steps used",
            "estimated discounted cost",
        )
        axes[0, j].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=max(1, len(labels)), frameon=False, fontsize=9
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
