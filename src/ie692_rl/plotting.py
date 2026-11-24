"""Report figures written by ``tune.py``: one PNG per problem, figure type and figure group.

    learning_curves_<problem>_<group>.png   exact optimality gap of the current policy vs environment
                                            steps (mean over the test seeds, band = +-1 standard error)
    tuning_<problem>_<group>.png            the grid search: every combination's selection metric per
                                            method
    tuning_grid_<problem>_<group>.png       the same grid search as heatmaps: one panel per method, rows
                                            = step size, columns = the other tuned parameters, chosen
                                            cell outlined

A figure group is a ``[figures]`` entry in ``config.cfg`` (at most eight methods); without that
section all methods share one figure and the file names have no ``_<group>`` suffix.

Titles carry only the problem's display name and legends the methods' display names (``label`` in
``config.cfg``); descriptions belong in the figure captions. Math is rendered with matplotlib's
mathtext in STIX fonts (LaTeX syntax such as ``$V^{\\pi}(s_0)$``, no TeX installation needed).

Colors follow the method, never its rank: each method gets a fixed categorical slot from its position
in its figure group (validated palette, adjacent pairs CVD-safe), so a method keeps its color in
every figure of that group even when other methods are skipped.
"""

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
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


# Sequential blue ramp (light = small gap, dark = large gap) and the fill for combinations without an update.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
).with_extremes(bad="#f0efec")
PARAM_LABELS = {  # config key -> (axis label, symbol for group headers)
    "spmd.step_size": (r"Step Size $\eta$", r"$\eta$"),
    "spmd.tau": (r"Entropy Weight $\tau$", r"$\tau$"),
    "spmd.tau_power": (r"Annealing Power $p$", r"$p$"),
    "estimator.bias": (r"TOMC Bias $\epsilon$", r"$\epsilon$"),
    "estimator.eta": (r"TD Step Size $\alpha$", r"$\alpha$"),
    "estimator.lam": (r"Trace Decay $\lambda$", r"$\lambda$"),
    "estimator.n_steps": ("TD Steps per Iteration", r"$n$"),
    "divergence.index": (r"Tsallis Index $q$", r"$q$"),
}
GRID_UNITS = 10  # panel widths per figure row, in heatmap columns
TITLE_CHARS_PER_UNIT = 8  # panel-title characters that fit in one heatmap column (0.62 in, 11 pt)
LEGEND_CHARS_PER_COLUMN = 26  # longest label that still fits three legend columns under a curve plot


def method_colors(methods: Sequence[str]) -> dict[str, str]:
    if len(methods) > len(SERIES):
        raise ValueError("At most eight methods can be colored; facet or fold the rest")
    return dict(zip(methods, SERIES, strict=False))


def figure_path(output_dir: Path, kind: str, problem: str, group: str = "") -> Path:
    suffix = f"_{group}" if group else ""
    return output_dir / f"{kind}_{problem.lower()}{suffix}.png"


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
    group: str = "",
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
            longest = max(_plain_length(line.get_label()) for line in ax.lines)
            ncol = 3 if longest <= LEGEND_CHARS_PER_COLUMN else 2
            fig.legend(loc="outside lower center", ncol=min(ncol, len(ax.lines)), frameon=False)
            path = figure_path(output_dir, "learning_curves", problem, group)
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
    group: str = "",
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
            path = figure_path(output_dir, "tuning", problem, group)
            fig.savefig(path)
            plt.close(fig)
        written.append(path)
    return written


class _Grid:
    """One method's grid-search results as a matrix: rows = step size (or the first tuned parameter),
    columns = every combination of the other tuned parameters (outer x inner)."""

    def __init__(self, points: Sequence[Row], value_key: str) -> None:
        keys = sorted(k for k in points[0] if "." in k and len({r.get(k) for r in points}) > 1)
        self.row_key = "spmd.step_size" if "spmd.step_size" in keys else (keys[0] if keys else None)
        self.col_keys = [k for k in keys if k != self.row_key]
        self.row_values = sorted({r[self.row_key] for r in points}) if self.row_key else [None]
        self.col_values = sorted({tuple(r[k] for k in self.col_keys) for r in points})
        self.values = np.full((len(self.row_values), len(self.col_values)), np.nan)
        self.cells: dict[int, tuple[int, int]] = {}
        for r in points:
            i = self.row_values.index(r[self.row_key]) if self.row_key else 0
            j = self.col_values.index(tuple(r[k] for k in self.col_keys))
            self.cells[id(r)] = (i, j)
            if r["min_updates"] > 0:
                self.values[i, j] = r[value_key]

    @property
    def width(self) -> int:
        return max(2, len(self.col_values))


def _plain_length(text: str) -> int:
    """Rendered length in characters; a mathtext symbol renders about one character wide."""
    return len(re.sub(r"\$[^$]*\$", "x", text))


def _slot_width(grid: _Grid, title: str) -> int:
    """Heatmap columns a panel needs: its own width, or more if its title is longer."""
    return max(grid.width, math.ceil(_plain_length(title) / TITLE_CHARS_PER_UNIT))


def _cell_text(value: float) -> str:
    if value == 0:
        return "0"
    return "<0.001" if value < 0.0005 else f"{value:.3f}"


def _on_dark(rgba: tuple[float, ...]) -> bool:
    """True if white text reads better than ink on this fill (relative luminance below 0.2, where the
    two contrasts cross)."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.2


def _sqrt_ticks(vmax: float) -> list[float]:
    """Round colorbar ticks spread evenly on the square-root scale (no crowding near the top)."""
    candidates = [m * 10.0**e for e in range(-4, 3) for m in (1, 2, 5)]
    ticks = [0.0]
    for value in candidates:
        if value <= vmax and np.sqrt(value) - np.sqrt(ticks[-1]) >= 0.14 * np.sqrt(vmax):
            ticks.append(value)
    return ticks


def _draw_grid(ax: Any, grid: _Grid, norm: PowerNorm, chosen: tuple[int, int] | None, title: str) -> Any:
    n_rows, n_cols = grid.values.shape
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(grid.values), cmap=SEQUENTIAL, norm=norm, edgecolors="white", linewidth=2
    )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)  # first row (smallest step size) on top
    for (i, j), value in np.ndenumerate(grid.values):
        if np.isfinite(value):
            dark = _on_dark(SEQUENTIAL(norm(value)))
            ax.text(
                j + 0.5,
                i + 0.5,
                _cell_text(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if dark else INK,
            )
    if chosen is not None:
        i, j = chosen
        ax.add_patch(Rectangle((j + 0.06, i + 0.06), 0.88, 0.88, fill=False, edgecolor=INK, linewidth=2))
    ax.set_title(title, color=INK, fontsize=11)
    ax.tick_params(length=0, labelcolor=INK_SECONDARY)
    ax.spines[:].set_visible(False)
    if grid.row_key:
        ax.set_yticks(np.arange(n_rows) + 0.5, [f"{v:g}" for v in grid.row_values])
        ax.set_ylabel(PARAM_LABELS.get(grid.row_key, (grid.row_key,))[0], color=INK_SECONDARY)
    else:
        ax.set_yticks([])
    if not grid.col_keys:
        ax.set_xticks([])
        return mesh
    inner = grid.col_keys[-1]
    ax.set_xticks(np.arange(n_cols) + 0.5, [f"{c[-1]:g}" for c in grid.col_values])
    ax.set_xlabel(PARAM_LABELS.get(inner, (inner,))[0], color=INK_SECONDARY)
    if len(grid.col_keys) == 2:  # outer parameter: group headers on top, separated by wider gaps
        outer = grid.col_keys[0]
        groups = list(dict.fromkeys(c[0] for c in grid.col_values))
        size = n_cols // len(groups)
        symbol = PARAM_LABELS.get(outer, (outer, outer))[1]
        top = ax.secondary_xaxis("top")
        top.set_xticks([size * (g + 0.5) for g in range(len(groups))], [f"{symbol} = {v:g}" for v in groups])
        top.tick_params(length=0, labelcolor=INK_SECONDARY)
        top.spines[:].set_visible(False)
        for g in range(1, len(groups)):
            ax.axvline(g * size, color="white", linewidth=6)
    return mesh


def plot_tuning_grid(
    rows: Sequence[Row],
    methods: Sequence[str],
    labels: dict[str, str],
    problem_labels: dict[str, str],
    output_dir: Path,
    selection: str = "gap_lower",
    group: str = "",
) -> list[Path]:
    """One figure per problem: each method's grid search as a heatmap of the selection metric (shared
    square-root color scale, value in every cell), with the chosen combination outlined."""
    value_key, _, metric_label = TUNING_METRICS[selection]
    chosen, written = best_per_method(rows, selection), []
    for problem in dict.fromkeys(r["problem"] for r in rows):
        grids, picks = [], []
        for method in methods:
            points = [r for r in rows if r["problem"] == problem and r["method"] == method]
            if points:
                grid = _Grid(points, value_key)
                best = chosen.get((problem, method))
                grids.append((method, grid))
                picks.append(grid.cells.get(id(best)) if best is not None else None)
        # Each panel gets a slot wide enough for its heatmap and its title (centered in the slot).
        slots = [_slot_width(grid, labels.get(method, method)) for method, grid in grids]
        layout: list[list[int]] = [[]]  # panel indices per figure row, at most GRID_UNITS wide
        for index, slot in enumerate(slots):
            if layout[-1] and sum(slots[k] for k in layout[-1]) + slot > GRID_UNITS:
                layout.append([])
            layout[-1].append(index)
        units = max(sum(slots[k] for k in line) for line in layout)
        finite = [v for _, g in grids for v in g.values[np.isfinite(g.values)]]
        norm = PowerNorm(0.5, vmin=0.0, vmax=max(finite) if finite and max(finite) > 0 else 1.0)
        with plt.rc_context(STYLE):
            fig = plt.figure(figsize=(0.62 * units + 2.2, 2.25 * len(layout) + 0.9), layout="constrained")
            spec = fig.add_gridspec(len(layout), units)
            axes, mesh = [], None
            for line_number, line in enumerate(layout):
                start = 0
                for k in line:
                    method, grid = grids[k]
                    left = start + (slots[k] - grid.width) // 2
                    ax = fig.add_subplot(spec[line_number, left : left + grid.width])
                    mesh = _draw_grid(ax, grid, norm, picks[k], labels.get(method, method))
                    axes.append(ax)
                    start += slots[k]
            colorbar = fig.colorbar(mesh, ax=axes, shrink=0.9, aspect=30, ticks=_sqrt_ticks(norm.vmax))
            colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            colorbar.set_label(metric_label, color=INK_SECONDARY)
            colorbar.outline.set_visible(False)
            colorbar.ax.tick_params(color=AXIS, labelcolor=INK_SECONDARY)
            fig.suptitle(problem_labels.get(problem, problem), color=INK, fontsize=13)
            legend = [Patch(facecolor="none", edgecolor=INK, linewidth=2, label="Chosen")]
            if any(np.isnan(g.values).any() for _, g in grids):
                legend.append(Patch(facecolor=SEQUENTIAL.get_bad(), edgecolor=AXIS, label="No Update"))
            fig.legend(handles=legend, loc="outside lower center", ncol=len(legend), frameon=False)
            path = figure_path(output_dir, "tuning_grid", problem, group)
            fig.savefig(path)
            plt.close(fig)
        written.append(path)
    return written
