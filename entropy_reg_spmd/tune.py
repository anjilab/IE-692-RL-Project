"""Hyperparameter tuning on a fraction (10%) of the budget, then full runs with the chosen settings.

Follows the project guideline: grid search over up to three parameters with three values each,
each combination run on a limited observation budget, and the best combination run on the full
budget. All settings are in config.cfg ([tuning] and the problem sections; a problem's
tune_full_budget, if set, replaces its budget here).

    method                 parameters tuned
    baseline SPMD          step size x TOMC tail tolerance (bias)        (the team's KL step, tau = 0)
    Entropy-Reg SPMD       tau x step size x TOMC tail tolerance (bias)

A combination is scored by the exact gap to the optimal cost of its final policy, averaged over the
tuning seeds. Combinations that complete no update within the tuning budget are not eligible.

Run from the repository root:
    .venv/bin/python entropy_reg_spmd/tune.py                         # every problem in [tuning]
    .venv/bin/python entropy_reg_spmd/tune.py --problems Blackjack --tune-only
    .venv/bin/python entropy_reg_spmd/tune.py --plot-only            # redraw tuning.png from tuning.csv
    .venv/bin/python entropy_reg_spmd/tune.py --reuse-tuning         # full runs with the saved choices

Writes to entropy_reg_spmd/results/tuned/: tuning.csv (every combination), tuning.png (the search,
one panel per problem and TOMC bias, chosen combinations ringed), history.csv (every
iteration of every full run) and summary.csv (final gaps vs the baseline).
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from experiment import HERE, ExactScorer, floats, ints, load_config, make_env, run

BASELINE = "baseline SPMD"


@lru_cache(maxsize=None)
def problem(name):
    """(environment, scorer); cached once per worker process."""
    env = make_env(load_config()[name])
    return env, ExactScorer(env)


def one_run(job):
    """Worker: job = (problem name, method, tau, eta, bias, seed, budget) -> (job, history)."""
    name, _, tau, eta, bias, seed, budget = job
    env, scorer = problem(name)
    return job, run(env, scorer, tau, eta, bias, budget, seed)


def full_budget(config, name):
    """The problem's tune_full_budget if set, otherwise its budget."""
    section = config[name]
    return int(section.get("tune_full_budget", section["budget"]))


def grid(config, name):
    """[(method, tau, eta, bias)] for one problem."""
    tuning, section = config["tuning"], config[name]
    steps = floats(section.get("tune_step_sizes", tuning["step_sizes"]))
    biases = floats(tuning["biases"])
    combos = [(BASELINE, 0.0, eta, bias) for eta, bias in product(steps, biases)]
    combos += [("Entropy-Reg SPMD", tau, eta, bias) for tau, eta, bias in product(floats(tuning["taus"]), steps, biases)]
    return combos


def run_all(pool, jobs):
    return dict(pool.map(one_run, jobs, chunksize=1))


def tune(pool, config, name):
    """Run every combination on the tuning budget; return tuning rows and the best combination per method."""
    tuning = config["tuning"]
    budget = int(full_budget(config, name) * float(tuning["tune_fraction"]))
    seeds = ints(tuning["tuning_seeds"])
    combos = grid(config, name)
    results = run_all(pool, [(name, *combo, seed, budget) for combo in combos for seed in seeds])
    rows = []
    for method, tau, eta, bias in combos:
        histories = [results[name, method, tau, eta, bias, seed, budget] for seed in seeds]
        updates = min(len(h) - 1 for h in histories)
        finals = np.array([h[-1]["gap"] for h in histories])
        rows.append(dict(problem=name, method=method, tau=tau, step_size=eta, bias=bias,
                         tuning_budget=budget, min_updates=updates, mean_final_gap=finals.mean(),
                         standard_error=finals.std(ddof=1) / np.sqrt(len(finals))))
    return rows, {method: row for (_, method), row in best_per_method(rows).items()}


def full_runs(pool, config, name, best):
    """Run each method's chosen combination on the full budget with the test seeds."""
    budget, seeds = full_budget(config, name), ints(config["tuning"]["test_seeds"])
    jobs = [(name, m, row["tau"], row["step_size"], row["bias"], seed, budget)
            for m, row in best.items() for seed in seeds]
    rows = []
    for (_, method, tau, eta, bias, seed, _), history in run_all(pool, jobs).items():
        rows += [dict(problem=name, method=method, tau=tau, step_size=eta, bias=bias, seed=seed, **record)
                 for record in history]
    return rows


def summarize(history_rows, z_threshold=2.0, min_worse_diff=0.001):
    """Final gap per (problem, method) with a z-test against the baseline."""
    finals = {}
    for row in history_rows:  # the last record of each run is its final policy
        finals.setdefault((row["problem"], row["method"]), {})[row["seed"]] = row["gap"]
    summary = []
    for (name, method), by_seed in finals.items():
        gaps = np.array(list(by_seed.values()))
        base = np.array(list(finals.get((name, BASELINE), {}).values()))
        se = np.sqrt(gaps.var(ddof=1) / len(gaps) + base.var(ddof=1) / len(base)) if len(base) > 1 else 0.0
        diff = gaps.mean() - base.mean() if len(base) else np.nan
        z = diff / se if se > 0 else 0.0
        verdict = "" if method == BASELINE else (
            "better" if z <= -z_threshold else "worse" if z >= z_threshold and diff >= min_worse_diff else "tie")
        summary.append(dict(problem=name, method=method, seeds=len(gaps), mean_final_gap=gaps.mean(),
                            standard_error=gaps.std(ddof=1) / np.sqrt(len(gaps)), diff_vs_baseline=diff,
                            z=z, verdict=verdict))
    return summary


def read_tuning(path):
    """Rows of a tuning.csv written by this script, with numbers converted back."""
    rows = list(csv.DictReader(open(path)))
    for row in rows:
        for key in ("tau", "step_size", "bias", "mean_final_gap", "standard_error"):
            if row.get(key):
                row[key] = float(row[key])
        row["min_updates"], row["tuning_budget"] = int(row["min_updates"]), int(row["tuning_budget"])
    return rows


def best_per_method(rows):
    """The chosen combination per (problem, method): lowest mean gap among those with an update."""
    best = {}
    for row in rows:
        key = row["problem"], row["method"]
        if row["min_updates"] > 0 and (key not in best or row["mean_final_gap"] < best[key]["mean_final_gap"]):
            best[key] = row
    return best


def plot_tuning(rows, path):
    """One panel per (problem, TOMC bias): mean tuning gap vs step size, one line per method / tau.

    Rings mark each method's chosen combination; bars are +-1 standard error over the tuning seeds.
    Combinations that made no update in the tuning budget are drawn as hollow markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_color, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    # Baseline in orange; Entropy-Reg SPMD in one blue hue, lighter to darker as tau grows.
    blues = ["#86b6ef", "#2a78d6", "#184f95"]
    markers, dashes = ["s", "o", "^", "D"], ["-", (0, (5, 2)), (0, (2, 1.5)), (0, (6, 2, 1, 2))]
    problems = list(dict.fromkeys(r["problem"] for r in rows))
    biases = sorted({r["bias"] for r in rows})
    taus = sorted({r["tau"] for r in rows if r["method"] != BASELINE})
    series = [(BASELINE, 0.0, "baseline SPMD (KL, τ = 0)", "#eb6834")]
    series += [("Entropy-Reg SPMD", tau, f"Entropy-Reg SPMD, τ = {tau:g}", blues[min(i, 2)]) for i, tau in enumerate(taus)]
    chosen = best_per_method(rows)

    fig, axes = plt.subplots(len(problems), len(biases), figsize=(4.2 * len(biases), 3.4 * len(problems)),
                             squeeze=False, sharex=True, sharey="row", layout="constrained", facecolor=surface)
    for i, name in enumerate(problems):
        budget = next(r["tuning_budget"] for r in rows if r["problem"] == name)
        for j, bias in enumerate(biases):
            ax = axes[i, j]
            ax.set_facecolor(surface)
            for k, (method, tau, label, color) in enumerate(series):
                pts = sorted((r for r in rows if r["problem"] == name and r["method"] == method
                              and r["tau"] == tau and r["bias"] == bias), key=lambda r: r["step_size"])
                x = [r["step_size"] for r in pts]
                y = np.array([r["mean_final_gap"] for r in pts])
                se = np.array([r.get("standard_error") or 0.0 for r in pts])
                ax.errorbar(x, y, yerr=se, color=color, linestyle=dashes[k], linewidth=2, capsize=0,
                            elinewidth=1, marker=markers[k], markersize=7, markeredgecolor=surface,
                            markeredgewidth=1.5, label=label, zorder=3)
                for r in pts:
                    if r["min_updates"] == 0:  # not eligible: no update within the tuning budget
                        ax.plot(r["step_size"], r["mean_final_gap"], marker=markers[k], markersize=7,
                                color=surface, markeredgecolor=color, zorder=4)
                    if chosen.get((name, method)) is r:
                        ax.plot(r["step_size"], r["mean_final_gap"], marker="o", markersize=17,
                                fillstyle="none", color=ink, markeredgewidth=1.5, zorder=5)
                        # Baseline's label above the ring, Entropy-Reg's below; kept inside the panel.
                        steps = sorted({q["step_size"] for q in rows})
                        ha = "left" if r["step_size"] == steps[0] else "right" if r["step_size"] == steps[-1] else "center"
                        dx = {"left": -6, "right": 6, "center": 0}[ha]
                        above = method == BASELINE
                        short = "baseline" if above else f"entropy, τ = {tau:g}"
                        row_max = max(q["mean_final_gap"] for q in rows if q["problem"] == name)
                        near_axis = r["mean_final_gap"] < 0.15 * row_max  # no room below: stack above
                        dy = 13 if above else 27 if near_axis else -20
                        ax.annotate(f"chosen: {short}", (r["step_size"], r["mean_final_gap"]), textcoords="offset points",
                                    xytext=(dx, dy), ha=ha, fontsize=8, color=ink, zorder=6)
            ax.set_xscale("log")
            ax.set_xticks(sorted({r["step_size"] for r in rows}))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.grid(color=grid_color, linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#c3c2b7")
            ax.tick_params(colors=muted, labelcolor="#52514e", labelsize=8)
            ax.set_title(f"{name} · TOMC bias {bias:g}", loc="left", fontsize=10, color=ink)
            if j == 0:
                ax.set_ylabel(f"gap after {budget:,} steps", fontsize=9, color="#52514e")
            if i == len(problems) - 1:
                ax.set_xlabel("step size η", fontsize=9, color="#52514e")
        axes[i, 0].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(series), frameon=False, fontsize=9)
    fig.suptitle("Hyperparameter search on 10% of the budget: mean exact gap to optimal over the tuning seeds "
                 "(lower is better; ring = chosen, hollow = no update)", fontsize=11, color=ink)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", nargs="+", default=[p.strip() for p in config["tuning"]["problems"].split(",")])
    parser.add_argument("--tune-only", action="store_true", help="skip the full runs")
    parser.add_argument("--plot-only", action="store_true", help="only redraw tuning.png from an existing tuning.csv")
    parser.add_argument("--reuse-tuning", action="store_true",
                        help="skip the search: take the chosen combinations from an existing tuning.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=HERE / "results" / "tuned")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = [r for r in read_tuning(args.output / "tuning.csv") if r["problem"] in args.problems]
        plot_tuning(rows, args.output / "tuning.png")
        print(f"wrote {args.output / 'tuning.png'}")
        return
    tuning_rows, history_rows = [], []
    saved = read_tuning(args.output / "tuning.csv") if args.reuse_tuning else []
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.problems:
            if args.reuse_tuning:
                rows = [r for r in saved if r["problem"] == name]
                best = {method: row for (_, method), row in best_per_method(rows).items()}
            else:
                rows, best = tune(pool, config, name)
                tuning_rows += rows
                write_csv(args.output / "tuning.csv", tuning_rows)
                plot_tuning(tuning_rows, args.output / "tuning.png")
            print(f"\n{name}: tuning budget {rows[0]['tuning_budget']:,} steps per run")
            for method in dict.fromkeys(r["method"] for r in rows):
                if method in best:
                    b = best[method]
                    print(f"  {method:17s} tau {b['tau']}, step size {b['step_size']:g}, bias {b['bias']:g}: "
                          f"gap {b['mean_final_gap']:.4f} after >= {b['min_updates']} updates")
                else:
                    print(f"  {method:17s} no combination completed an update in the tuning budget")
            if not args.tune_only:
                history_rows += full_runs(pool, config, name, best)
                write_csv(args.output / "history.csv", history_rows)
    if history_rows:
        settings = config["experiment"]
        summary = summarize(history_rows, float(settings["z_threshold"]), float(settings["min_worse_diff"]))
        write_csv(args.output / "summary.csv", summary)
        print(f"\n{'problem':15s} {'method':17s} {'final gap':>17s} {'vs baseline':>12s} {'z':>6s}  verdict")
        for row in summary:
            print(f"{row['problem']:15s} {row['method']:17s} {row['mean_final_gap']:8.4f} ± {row['standard_error']:.4f} "
                  f"{row['diff_vs_baseline']:+12.4f} {row['z']:+6.1f}  {row['verdict']}")
    print(f"\nresults in {args.output}")


if __name__ == "__main__":
    main()
