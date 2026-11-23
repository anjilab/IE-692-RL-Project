"""Hyperparameter tuning on a fraction (10%) of the budget, then full runs with the chosen settings.

Follows the project guideline: grid search over up to three parameters with three values each,
each combination run on a limited observation budget, and the best combination run on the full
budget. All settings are in config.cfg ([tuning] and the problem sections; a problem's
tune_full_budget, if set, replaces its budget here).

    method                 parameters tuned
    baseline SPMD          step size x coverage rounds
    PMD, tau=0 (plain KL)  step size x coverage rounds
    Entropy-Reg SPMD       tau x step size x coverage rounds

A combination is scored by the exact gap to the optimal cost of its final policy, averaged over the
tuning seeds. Combinations that complete no update within the tuning budget are not eligible.

Run from the repository root:
    .venv/bin/python entropy_reg_spmd/tune.py                         # every problem in [tuning]
    .venv/bin/python entropy_reg_spmd/tune.py --problems Blackjack --tune-only

Writes to entropy_reg_spmd/results/tuned/: tuning.csv (every combination), history.csv (every
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
    """(environment, scorer, horizon); cached once per worker process."""
    section = load_config()[name]
    env = make_env(section)
    return env, ExactScorer(env), int(section["horizon"])


def one_run(job):
    """Worker: job = (problem name, method, tau, eta, rounds, seed, budget) -> (job, history)."""
    name, _, tau, eta, rounds, seed, budget = job
    env, scorer, horizon = problem(name)
    return job, run(env, scorer, tau, eta, rounds, horizon, budget, seed)


def full_budget(config, name):
    """The problem's tune_full_budget if set, otherwise its budget."""
    section = config[name]
    return int(section.get("tune_full_budget", section["budget"]))


def grid(config, name):
    """[(method, tau, eta, rounds)] for one problem."""
    tuning, section = config["tuning"], config[name]
    base_steps = floats(section.get("tune_baseline_step_sizes", tuning["baseline_step_sizes"]))
    pmd_steps = floats(section.get("tune_pmd_step_sizes", tuning["pmd_step_sizes"]))
    rounds = ints(tuning["coverage_rounds"])
    combos = [(BASELINE, None, eta, r) for eta, r in product(base_steps, rounds)]
    combos += [("PMD, tau=0", 0.0, eta, r) for eta, r in product(pmd_steps, rounds)]
    combos += [("Entropy-Reg SPMD", tau, eta, r) for tau, eta, r in product(floats(tuning["taus"]), pmd_steps, rounds)]
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
    for method, tau, eta, rounds in combos:
        histories = [results[name, method, tau, eta, rounds, seed, budget] for seed in seeds]
        updates = min(len(h) - 1 for h in histories)
        rows.append(dict(problem=name, method=method, tau=tau, step_size=eta, coverage_rounds=rounds,
                         tuning_budget=budget, min_updates=updates,
                         mean_final_gap=np.mean([h[-1]["gap"] for h in histories])))
    best = {}
    for row in rows:
        if row["min_updates"] > 0 and (row["method"] not in best or row["mean_final_gap"] < best[row["method"]]["mean_final_gap"]):
            best[row["method"]] = row
    return rows, best


def full_runs(pool, config, name, best):
    """Run each method's chosen combination on the full budget with the test seeds."""
    budget, seeds = full_budget(config, name), ints(config["tuning"]["test_seeds"])
    jobs = [(name, m, row["tau"], row["step_size"], row["coverage_rounds"], seed, budget)
            for m, row in best.items() for seed in seeds]
    rows = []
    for (_, method, tau, eta, rounds, seed, _), history in run_all(pool, jobs).items():
        rows += [dict(problem=name, method=method, tau=tau, step_size=eta, coverage_rounds=rounds, seed=seed, **record)
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
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--output", type=Path, default=HERE / "results" / "tuned")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tuning_rows, history_rows = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for name in args.problems:
            rows, best = tune(pool, config, name)
            tuning_rows += rows
            write_csv(args.output / "tuning.csv", tuning_rows)
            print(f"\n{name}: tuning budget {rows[0]['tuning_budget']:,} steps per run")
            for method in dict.fromkeys(r["method"] for r in rows):
                if method in best:
                    b = best[method]
                    print(f"  {method:17s} tau {b['tau']}, step size {b['step_size']:g}, rounds {b['coverage_rounds']}: "
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
