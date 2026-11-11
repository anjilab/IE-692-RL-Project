"""Learning curves: baseline SPMD (Euclidean) vs Entropy-Regularized SPMD for each tau.

Reruns the 5 test seeds of compare_exploration.py with the step size each method was
tuned to (read from the saved CSVs), scores the policy exactly after every iteration,
and plots the mean gap to the optimal cost against environment steps used, with a
band of +/- 1 standard error over seeds.

    python "Entropy-Regularized SPMD/plot_trajectories.py"                # all settings (~10 min)
    python "Entropy-Regularized SPMD/plot_trajectories.py" --plot-only    # replot from cache
    python "Entropy-Regularized SPMD/plot_trajectories.py" --online-only  # GridWorld + blackjack online (~1 min)
Outputs (in this folder): trajectories.npz (cache), trajectories_original.png,
trajectories_larger_gridworld.png, trajectories_online.png. With --online-only the cache
is trajectories_online.npz and only trajectories_online.png is drawn.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_exploration as ce
import kl_spmd
import utils
from exact_eval import ExactScorer
from pmd import run_pmd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "trajectories.npz"
ONLINE_CACHE = HERE / "trajectories_online.npz"
ONLINE_PROBLEMS = ["GridWorld", "Blackjack"]
GRID = np.linspace(0, 1, 101)  # fractions of the budget at which curves are read off
FIGURES = {
    "trajectories_original.png": (HERE / "exploration_results.csv",
                                  ["GridWorld", "Blackjack", "DeceptiveGridWorld", "FullBlackjack"]),
    "trajectories_larger_gridworld.png": (HERE / "larger_gridworld_results.csv",
                                          ["GridWorld10x10", "GridWorld20x20", "GridWorld30x30"]),
}
SAMPLINGS = ["generative", "online"]

# Baseline in orange; tau values as an ordered light-to-dark blue ramp (tau is a magnitude).
BASELINE_COLOR = "#eb6834"
TAU_COLORS = {0.0: "#86b6ef", 0.001: "#3987e5", 0.01: "#256abf", 0.03: "#184f95", 0.1: "#0d366b"}
TAU_STYLES = {0.0: (0, (1, 1.5)), 0.001: "-", 0.01: (0, (5, 2)), 0.03: (0, (2, 1, 1, 1)), 0.1: (0, (8, 2, 1, 2))}
INK, MUTED, GRIDLINE, AXIS = "#0b0b0b", "#6b6a64", "#e1e0d9", "#c3c2b7"


def tuned_methods(csv_path, problem, sampling):
    """(label, tau, step size, update) for each method, using the saved tuned step sizes."""
    with csv_path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["problem"] == problem and r["sampling"] == sampling]
    for r in rows:
        tau, step = float(r["tau"]), float(r["step_size"])
        if r["method"] == "Euclidean":
            yield "baseline SPMD (Euclidean)", 0.0, step, utils.mirror_step
        else:
            yield f"KL, τ = {tau:g}", tau, step, (lambda p, q, s, tau=tau: run_pmd(p, -q, s, tau))


def trajectory(env, estimator, estimator_args, budget, update, step, tau, seed, scorer):
    """Gap to optimal after every iteration, read off on GRID (fractions of budget)."""
    rng = np.random.default_rng(seed)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    used, xs, gaps = 0, [0.0], [scorer.score(policy) - scorer.optimal]
    while used < budget:
        sampled_env = kl_spmd.EntropyCostEnv(env, policy, tau) if tau > 0 else env
        q, count = estimator(sampled_env, policy, *estimator_args, rng)
        used += count
        policy = update(policy, q, step)
        xs.append(used / budget)
        gaps.append(scorer.score(policy) - scorer.optimal)
    # As in compare_exploration.py, fraction f is scored with the first policy whose
    # cumulative samples reach f (so the final point matches the saved CSVs).
    return np.array(gaps)[np.searchsorted(xs, GRID, side="left")]


def run_settings(settings, cache):
    """settings: (CSV with tuned step sizes, problem, sampling). Saves curves to cache."""
    curves = {}
    for csv_path, problem, sampling in settings:
        env, samplings = ce.PROBLEMS[problem]
        scorer = ExactScorer(env)
        estimator, args, budget = samplings[sampling]
        print(f"{problem}, {sampling}", flush=True)
        for label, tau, step, update in tuned_methods(csv_path, problem, sampling):
            runs = [trajectory(env, estimator, args, budget, update, step, tau, seed, scorer)
                    for seed in ce.TEST_SEEDS]
            curves[f"{problem}|{sampling}|{label}|{tau}|{budget}"] = np.array(runs)
            print(f"  {label:28s} step={step:<5g} final gap {np.mean(runs, axis=0)[-1]:.4f}", flush=True)
    np.savez(cache, **curves)
    return curves


def all_settings():
    return [(csv_path, problem, sampling) for csv_path, problems in FIGURES.values()
            for problem in problems for sampling in SAMPLINGS]


def draw_panel(ax, curves, problem, sampling, xlabel=True, ylabel=True):
    """Mean gap (+/- 1 SE band) for every method in one setting."""
    keys = [k for k in curves if k.startswith(f"{problem}|{sampling}|")]
    budget = float(keys[0].split("|")[4])
    scale, unit = (1e6, "millions") if budget >= 1e6 else (1e3, "thousands")
    for key in sorted(keys, key=lambda k: (not k.split("|")[2].startswith("baseline"), float(k.split("|")[3]))):
        _, _, label, tau, _ = key.split("|")
        tau, runs = float(tau), curves[key]
        x = GRID * budget / scale
        mean, se = runs.mean(0), runs.std(0, ddof=1) / np.sqrt(len(runs))
        is_base = label.startswith("baseline")
        color = BASELINE_COLOR if is_base else TAU_COLORS.get(tau, MUTED)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2.6 if is_base else 2,
                linestyle="-" if is_base else TAU_STYLES.get(tau, "-"), label=label,
                zorder=3 if is_base else 2)
    ax.set_title(f"{problem} - {sampling} sampling", fontsize=10, color=INK, loc="left")
    # Every run starts at the uniform policy's gap, far above where the methods
    # separate; cut the axis to the curves after the first 10% of the budget.
    late = max((curves[k].mean(0) + curves[k].std(0, ddof=1) / np.sqrt(len(curves[k])))[GRID >= 0.1].max()
               for k in keys)
    start = curves[keys[0]].mean(0)[0]
    top = min(start, 1.15 * late)
    ax.set_ylim(0, top)
    if top < start:
        ax.text(0.99, 0.97, f"axis cut: all methods start at {start:.2f} (uniform policy)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, budget / scale)
    ax.grid(True, color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if xlabel:
        ax.set_xlabel(f"environment steps used ({unit})", fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel("gap to optimal cost", fontsize=9, color=MUTED)


def finish(fig, axes, filename, title):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=11, color=INK)
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=9,
               handlelength=3)
    fig.savefig(HERE / filename, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {HERE / filename}")


TITLE = "Baseline SPMD vs Entropy-Regularized SPMD: mean gap over 5 test seeds (band = ±1 SE; lower is better)"


def plot_online(curves):
    """Online sampling only, GridWorld and blackjack side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for j, problem in enumerate(ONLINE_PROBLEMS):
        draw_panel(axes[j], curves, problem, "online", ylabel=j == 0)
    finish(fig, axes, "trajectories_online.png",
           "Online sampling: baseline SPMD vs Entropy-Regularized SPMD (mean of 5 test seeds, band = ±1 SE)")


def plot(curves):
    for filename, (_, problems) in FIGURES.items():
        fig, axes = plt.subplots(len(problems), 2, figsize=(11, 2.9 * len(problems) + 0.8), squeeze=False,
                                 layout="constrained")
        for i, problem in enumerate(problems):
            for j, sampling in enumerate(SAMPLINGS):
                draw_panel(axes[i, j], curves, problem, sampling,
                           xlabel=i == len(problems) - 1, ylabel=j == 0)
        finish(fig, axes, filename, TITLE)
    plot_online(curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot-only", action="store_true", help="replot from the saved cache")
    parser.add_argument("--online-only", action="store_true",
                        help="only GridWorld and blackjack with online sampling")
    parser.add_argument("--tuned", type=Path, default=HERE / "exploration_results.csv",
                        help="CSV from compare_exploration.py with the tuned step sizes (--online-only)")
    args = parser.parse_args()
    if args.online_only:
        settings = [(args.tuned, problem, "online") for problem in ONLINE_PROBLEMS]
        curves = dict(np.load(ONLINE_CACHE)) if args.plot_only else run_settings(settings, ONLINE_CACHE)
        plot_online(curves)
    else:
        curves = dict(np.load(CACHE)) if args.plot_only else run_settings(all_settings(), CACHE)
        plot(curves)


if __name__ == "__main__":
    main()
