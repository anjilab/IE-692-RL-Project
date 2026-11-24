"""Tune every method on a fraction of the budget, then run the chosen settings on the full budget.

Follows the project guideline: for each method, a grid search over at most three parameters with
three values each; every combination runs on ``tune_fraction`` (10%) of the full observation budget,
averaged over the tuning seeds, and the best combination (among those that made an update with every
seed) runs on the full budget with the test seeds. "Best" is model-free by default: the smallest mean
certified lower bound on the optimality gap (advantage gap / policy improvement, from validation
rollouts; ties broken by the upper bound). ``selection = exact_gap`` uses the exact gap instead.

    method (config.cfg)       estimator               divergence   tuned parameters
    baseline                  TOMC                    KL           step size
    bregman_euclidean         TOMC                    Euclidean    step size
    bregman_tsallis           TOMC                    Tsallis      step size x Tsallis index
    entropy_reg               TOMC                    KL, tau > 0  tau x step size
    true_online_td            True Online TD, tabular KL           step size x TD step size x lambda
    true_online_td_linear     True Online TD, linear  KL           step size x TD step size x lambda
    td_zero_tabular / _linear TD(0)                   KL           step size x TD step size
    td_lambda_tabular/_linear TD(lambda)              KL           step size x TD step size x lambda
    td_euclidean              True Online TD          Euclidean    step size
    td_tsallis                True Online TD          Tsallis      step size x Tsallis index
    combined_entropy          True Online TD          KL, tau > 0  tau x step size
    combined_annealed         True Online TD          KL, tau_k    tau_0 x step size

The methods form four grids (``[figures]`` groups), each tuned on every problem: bregman,
evaluation, entropy, and combined. Methods with ``estimator_from`` (the combined grid) are tuned in a
second stage with the estimator settings their source method chose on the same problem.

Every TD estimator restarts from Q = 0 at each SPMD iteration (``estimator.q_init``, default 0).

Examples:
    python tune.py                                   # every problem and method in [tuning]
    python tune.py --problems Blackjack --tune-only
    python tune.py --methods baseline entropy_reg --workers 4
    python tune.py --reuse-tuning                    # full runs with the choices saved in tuning.csv
    python tune.py --plot-only                       # redraw the figures from the saved CSVs

Writes to --output (default results/tuned/): tuning.csv, history.csv (every iteration of every full
run), summary.csv (final gaps vs the baseline), and per problem and ``[figures]`` group
tuning_<problem>_<group>.png (dot plot), tuning_grid_<problem>_<group>.png (heatmap of every
combination) and learning_curves_<problem>_<group>.png (display names from the ``label`` keys in
config.cfg).
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
    figure_groups,
    inherit_estimator,
    int_list,
    load_config,
    map_jobs,
    method_label,
    method_spec,
    names,
    params_of,
    problem_budget,
    problem_label,
    read_csv,
    selection_of,
    summarize,
    tuning_jobs,
    tuning_rows,
    write_csv,
)
from ie692_rl.plotting import plot_learning_curves, plot_tuning, plot_tuning_grid

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
    all_problems = [s.removeprefix("problem ") for s in config.sections() if s.startswith("problem ")]
    budgets = {p: problem_budget(config, p) for p in all_problems}
    labels = {m: method_label(config, m) for m in all_methods}
    problem_labels = {p: problem_label(config, p) for p in all_problems}
    selection = args.selection or selection_of(config)
    args.output.mkdir(parents=True, exist_ok=True)

    groups = figure_groups(config, all_methods)

    def plotted(rows: list[Row]) -> dict[str, list[str]]:
        """Figure groups with data (all members are passed on, so each keeps its color slot)."""
        present = {r["method"] for r in rows}
        return {group: members for group, members in groups.items() if present.intersection(members)}

    def plot_tuning_figures(rows: list[Row]) -> None:
        for group, members in plotted(rows).items():
            plot_tuning(rows, members, labels, problem_labels, args.output, selection, group)
            plot_tuning_grid(rows, members, labels, problem_labels, args.output, selection, group)

    def plot_curve_figures(history: list[Row]) -> None:
        for group, members in plotted(history).items():
            plot_learning_curves(history, budgets, members, labels, problem_labels, args.output, group)

    if args.plot_only:
        plot_tuning_figures(list(read_csv(args.output / "tuning.csv")))
        if (args.output / "history.csv").exists():
            plot_curve_figures(list(read_csv(args.output / "history.csv")))
        print(f"figures redrawn in {args.output}")
        return []

    specs = {(p, m): method_spec(config, m, p) for p in problems for m in methods}

    tuned = tune(
        config_path,
        specs,
        budgets,
        float(tuning["tune_fraction"]),
        int_list(tuning["tuning_seeds"]),
        selection,
        args,
    )
    write_csv(args.output / "tuning.csv", tuned)
    plot_tuning_figures(tuned)
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
    plot_curve_figures(history)
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
    selection: str,
    args: argparse.Namespace,
) -> list[Row]:
    """Grid search (or reload the saved one); one row per (problem, method, combination).

    Two stages: methods with ``estimator_from`` are tuned after the method they take their estimator
    settings from, using its choice on the same problem; their entries in ``specs`` are replaced by
    the resolved specs, which the full runs then use.
    """
    saved = list(read_csv(args.output / "tuning.csv")) if args.reuse_tuning else None
    rows: list[Row] = []
    for second_stage in (False, True):
        stage = {
            key: spec for key, spec in specs.items() if (spec.estimator_from is not None) == second_stage
        }
        if second_stage:
            chosen = best_per_method(rows, selection)
            for (problem, method), spec in stage.items():
                parent = (problem, spec.estimator_from)
                if parent not in chosen:
                    raise ValueError(
                        f"{method} takes its estimator settings from {spec.estimator_from}, which has no "
                        f"tuned choice on {problem}; tune both together"
                    )
                stage[problem, method] = inherit_estimator(spec, specs[parent], chosen[parent])
            specs.update(stage)
        if saved is not None:
            rows += [r for r in saved if (r["problem"], r["method"]) in stage]
            continue
        started = time.perf_counter()
        jobs, plan = [], []
        for (problem, _), spec in stage.items():
            budget = int(budgets[problem] * fraction)
            plan.append((spec, problem, budget))
            jobs += tuning_jobs(config_path, problem, spec, budget, seeds)
        results = map_jobs(jobs, args.workers)
        print(f"tuning: {len(jobs)} jobs in {time.perf_counter() - started:.1f}s")
        rows += [row for spec, problem, budget in plan for row in tuning_rows(spec, problem, budget, results)]
    return rows


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
