"""Tune every method on a fraction of the budget, then run the chosen settings on the full budget.

Follows the project guideline: for each method, a grid search over at most three parameters with
three values each; every combination runs on ``tune_fraction`` (10%) of the full observation budget,
averaged over the tuning seeds, and the best combination (among those that made an update with every
seed) runs on the full budget with the test seeds. "Best" is model-free by default: the smallest mean
certified lower bound on the optimality gap (advantage gap / policy improvement, from validation
rollouts; ties broken by the upper bound). ``selection = exact_gap`` uses the exact gap instead.

    method (config.cfg)    estimator         divergence   tuned parameters
    baseline               TOMC              KL           step size x TOMC bias
    entropy_reg            TOMC              KL, tau > 0  tau x step size x TOMC bias
    entropy_reg_annealed   TOMC              KL, tau_k    tau_0 x step size x TOMC bias
    true_online_td         True Online TD    KL           step size x TD step size x lambda
    bregman_euclidean      TOMC              Euclidean    step size x TOMC bias
    bregman_tsallis        TOMC              Tsallis      step size x Tsallis index

Examples:
    python tune.py                                   # every problem and method in [tuning]
    python tune.py --problems Blackjack --tune-only
    python tune.py --methods baseline entropy_reg --workers 4
    python tune.py --reuse-tuning                    # full runs with the choices saved in tuning.csv
    python tune.py --plot-only                       # redraw the figures from the saved CSVs

Writes to --output (default results/tuned/): tuning.csv, tuning.png, history.csv (every iteration of
every full run), summary.csv (final gaps vs the baseline) and learning_curves.png.
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are only saved to files; set before pyplot is imported

from ie692_rl.experiment import (
    SELECTION_KEYS,
    Job,
    MethodSpec,
    Row,
    best_per_method,
    default_workers,
    int_list,
    is_available,
    load_config,
    map_jobs,
    method_spec,
    names,
    params_of,
    problem_budget,
    read_csv,
    selection_of,
    summarize,
    tuning_jobs,
    tuning_rows,
    write_csv,
)
from ie692_rl.plotting import plot_learning_curves, plot_tuning

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.cfg")
    parser.add_argument("--problems", nargs="+", help="default: [tuning] problems")
    parser.add_argument("--methods", nargs="+", help="default: [tuning] methods")
    parser.add_argument("--tune-only", action="store_true", help="skip the full runs")
    parser.add_argument(
        "--reuse-tuning", action="store_true", help="take the choices from an existing tuning.csv"
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="only redraw the figures from the saved CSVs"
    )
    parser.add_argument(
        "--selection",
        choices=sorted(SELECTION_KEYS),
        help="how to pick each method's combination (default: [tuning] selection)",
    )
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tuned")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Row]:
    """Run the pipeline; returns the summary rows (empty with --tune-only or --plot-only)."""
    args = parse_args(argv)
    config, config_path = load_config(args.config), str(args.config.resolve())
    tuning = config["tuning"]
    problems = args.problems or names(tuning["problems"])
    all_methods = names(tuning["methods"])
    methods = args.methods or all_methods
    budgets = {p: problem_budget(config, p) for p in problems}
    selection = args.selection or selection_of(config)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_tuning(
            list(read_csv(args.output / "tuning.csv")), all_methods, args.output / "tuning.png", selection
        )
        if (args.output / "history.csv").exists():
            plot_learning_curves(
                list(read_csv(args.output / "history.csv")),
                budgets,
                all_methods,
                args.output / "learning_curves.png",
            )
        print(f"figures redrawn in {args.output}")
        return []

    specs = {(p, m): method_spec(config, m, p) for p in problems for m in methods}
    for (problem, method), spec in list(specs.items()):
        if not is_available(spec):
            print(
                f"skipping {method} on {problem}: its divergence {spec.divergence!r} is not implemented yet"
            )
            del specs[problem, method]

    tuned = tune(
        config_path, specs, budgets, float(tuning["tune_fraction"]), int_list(tuning["tuning_seeds"]), args
    )
    write_csv(args.output / "tuning.csv", tuned)
    plot_tuning(tuned, all_methods, args.output / "tuning.png", selection)
    report_choices(specs, tuned, selection)
    if args.tune_only:
        return []

    started = time.perf_counter()
    jobs = [
        Job(config_path, problem, method, params_of(spec, row), seed, budgets[problem])
        for (problem, method), row in best_per_method(tuned, selection).items()
        for spec in [specs[problem, method]]
        for seed in int_list(tuning["test_seeds"])
    ]
    history = [
        row
        for _, rows in sorted(map_jobs(jobs, args.workers).items(), key=lambda item: _order(item[0]))
        for row in rows
    ]
    print(f"full runs: {len(jobs)} jobs in {time.perf_counter() - started:.1f}s")
    write_csv(args.output / "history.csv", history)
    plot_learning_curves(history, budgets, all_methods, args.output / "learning_curves.png")
    summary = summarize(
        history,
        tuning.get("baseline", "baseline"),
        float(tuning["z_threshold"]),
        float(tuning["min_worse_diff"]),
    )
    write_csv(args.output / "summary.csv", summary)
    print_summary(summary)
    print(f"\nresults in {args.output}")
    return summary


def _order(job: Job) -> tuple[str, str, int]:
    return job.problem, job.method, job.seed


def tune(
    config_path: str,
    specs: dict[tuple[str, str], MethodSpec],
    budgets: dict[str, int],
    fraction: float,
    seeds: list[int],
    args: argparse.Namespace,
) -> list[Row]:
    """Grid search (or reload the saved one); one row per (problem, method, combination)."""
    if args.reuse_tuning:
        keep = set(specs)
        return [r for r in read_csv(args.output / "tuning.csv") if (r["problem"], r["method"]) in keep]
    started = time.perf_counter()
    jobs, plan = [], []
    for (problem, _), spec in specs.items():
        budget = int(budgets[problem] * fraction)
        plan.append((spec, problem, budget))
        jobs += tuning_jobs(config_path, problem, spec, budget, seeds)
    results = map_jobs(jobs, args.workers)
    print(f"tuning: {len(jobs)} jobs in {time.perf_counter() - started:.1f}s")
    return [row for spec, problem, budget in plan for row in tuning_rows(spec, problem, budget, results)]


def report_choices(specs: dict[tuple[str, str], MethodSpec], rows: list[Row], selection: str) -> None:
    print(f"selection: {selection}")
    best = best_per_method(rows, selection)
    for problem, method in specs:
        if (problem, method) in best:
            b = best[problem, method]
            chosen = ", ".join(f"{k} = {b[k]}" for k in b if "." in k)
            print(
                f"{problem:12s} {method:22s} {chosen}: "
                f"gap lower bound {b['mean_gap_lower']:.4f}, exact gap {b['mean_final_gap']:.4f} "
                f"(>= {b['min_updates']} updates)"
            )
        else:
            print(f"{problem:12s} {method:22s} no combination completed an update in the tuning budget")


def print_summary(summary: list[Row]) -> None:
    print(
        f"\n{'problem':12s} {'method':22s} {'exact final gap':>17s} {'lower bd':>9s} {'vs baseline':>12s} "
        f"{'z':>6s}  verdict"
    )
    for r in summary:
        versus = "" if r["verdict"] == "" else f"{r['diff_vs_baseline']:+12.4f} {r['z']:+6.1f}"
        print(
            f"{r['problem']:12s} {r['method']:22s} {r['mean_final_gap']:8.4f} ± {r['standard_error']:.4f} "
            f"{r['mean_final_gap_lower']:9.4f} {versus:>19s}  {r['verdict']}"
        )


if __name__ == "__main__":
    main()
