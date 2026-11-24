"""Estimated V(s0) of the full runs with the entropy term removed, so methods are comparable.

With tau > 0 the estimator samples the regularized cost, so its V(s0) includes the discounted sum of
tau * sum_a pi log pi. TOMC's sampling path does not depend on the costs, so every estimate is replayed
from the same random state without the entropy cost: same samples, same observations, plain cost. The
replayed runs must match results/full_runs/history.csv exactly (checked).

    .venv/bin/python benchmarks/estimated_value.py

Reads tuning.csv and history.csv from results/full_runs and writes estimated_value.csv and
estimated_value_<problem>.png there.
"""

import copy
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ie692_rl.core.estimator import Estimate, QEstimator
from ie692_rl.core.exact import ExactScorer
from ie692_rl.core.spmd import run_spmd
from ie692_rl.envs.base import Environment
from ie692_rl.experiment import (
    Job,
    Row,
    best_per_method,
    build_method,
    default_workers,
    int_list,
    load_config,
    make_env,
    method_spec,
    names,
    params_of,
    problem_budget,
    read_csv,
    write_csv,
)
from ie692_rl.plotting import (
    AXIS,
    STYLE,
    _decimal_axis,
    _style_axes,
    figure_path,
    method_colors,
    step_unit,
)
from ie692_rl.types import FloatArray

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.cfg"
RESULTS = ROOT / "results" / "full_runs"
BINS = 40


@dataclass(frozen=True)
class PlainValueRecorder(QEstimator):
    """Passes estimates through and records sum_a pi(a|s0) Q(s0, a) without the entropy cost."""

    inner: QEstimator
    values: list[float] = field(default_factory=list)

    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        replay = copy.deepcopy(rng)
        result = self.inner.estimate(env, policy, rng, max_observations, extra_cost)
        plain = result
        if extra_cost is not None:
            plain = self.inner.estimate(env, policy, replay, max_observations, None)
            if plain.observations != result.observations:
                raise RuntimeError("The replay used a different number of observations")
        self.values.append(float(policy[env.start] @ plain.q[env.start]))
        return result


def replay(job: Job) -> list[Row]:
    config = load_config(job.config_path)
    env = make_env(config, job.problem)
    method = build_method(method_spec(config, job.method, job.problem), job.params)
    recorder = PlainValueRecorder(method.estimator)
    scorer = ExactScorer(env)
    result = run_spmd(
        env,
        recorder,
        method.divergence,
        method.step_size,
        job.budget,
        job.seed,
        tau=method.tau,
        scorer=scorer,
    )
    updated = [r for r in result.history if r.status == "updated"]
    return [
        {
            "problem": job.problem,
            "method": job.method,
            "seed": job.seed,
            "iteration": record.iteration,
            "observations": record.observations,
            "cumulative_observations": record.cumulative_observations,
            "estimated_value": record.estimated_value,
            "estimated_value_plain": value,
            "optimal_value": scorer.optimal,
        }
        for record, value in zip(updated, recorder.values, strict=True)
    ]


def check_against_history(rows: list[Row]) -> None:
    saved = {
        (r["problem"], r["method"], r["seed"], r["iteration"]): (
            r["cumulative_observations"],
            r["estimated_value"],
        )
        for r in read_csv(RESULTS / "history.csv")
        if r["status"] == "updated"
    }
    new = {
        (r["problem"], r["method"], r["seed"], r["iteration"]): (
            r["cumulative_observations"],
            r["estimated_value"],
        )
        for r in rows
    }
    if saved.keys() != new.keys():
        raise RuntimeError("The replay made different updates from history.csv")
    for key, (observations, value) in new.items():
        if saved[key][0] != observations or not math.isclose(
            saved[key][1], value, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError(f"The replay differs from history.csv at {key}")
    print(f"replay matches history.csv on all {len(new):,} updates")


def binned(rows: list[Row], budget: int) -> FloatArray:
    """One seed: mean estimate in each of BINS equal parts of the budget (the last one is carried
    forward through parts without an update; NaN before the first estimate)."""
    used = np.array([r["cumulative_observations"] for r in rows])
    values = np.array([r["estimated_value_plain"] for r in rows])
    part = np.minimum((used * BINS / budget).astype(int), BINS - 1)
    curve = np.full(BINS, np.nan)
    for b in range(BINS):
        if np.any(part == b):
            curve[b] = values[part == b].mean()
        elif b > 0:
            curve[b] = curve[b - 1]
    return curve


def plot(rows: list[Row], budgets: dict[str, int], methods: list[str]) -> None:
    config = load_config(CONFIG)
    colors = method_colors(methods)
    for problem, budget in budgets.items():
        scale, unit = step_unit(budget)
        x = (np.arange(BINS) + 1) / BINS * budget / scale
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(6.4, 4.2), layout="constrained")
            for method in methods:
                by_seed: dict[int, list[Row]] = defaultdict(list)
                for r in rows:
                    if r["problem"] == problem and r["method"] == method:
                        by_seed[r["seed"]].append(r)
                if not by_seed:
                    continue
                curves = np.array([binned(h, budget) for h in by_seed.values()])
                ready = np.all(np.isfinite(curves), axis=0)  # every seed has an estimate
                mean, se = curves.mean(axis=0), curves.std(axis=0, ddof=1) / np.sqrt(len(curves))
                low, high = (mean - se)[ready], (mean + se)[ready]
                ax.fill_between(x[ready], low, high, color=colors[method], alpha=0.15, linewidth=0)
                label = config[f"method {method}"].get("label", method)
                ax.plot(x[ready], mean[ready], color=colors[method], linewidth=2, label=label)
            optimal = next(r["optimal_value"] for r in rows if r["problem"] == problem)
            ax.axhline(optimal, color=AXIS, linewidth=1.4, linestyle="--", label=r"$V^{\star}(s_0)$")
            ax.set_xlim(0, budget / scale)
            _decimal_axis(ax.xaxis)
            _decimal_axis(ax.yaxis)
            title = config[f"problem {problem}"].get("label", problem)
            xlabel = f"Environment Steps ({unit})" if unit else "Environment Steps"
            _style_axes(ax, title, xlabel, r"Estimated $\hat{V}^{\pi_t}(s_0)$")
            fig.legend(loc="outside lower center", ncol=3, frameon=False)
            fig.savefig(figure_path(RESULTS, "estimated_value", problem))
            plt.close(fig)


def main() -> None:
    config = load_config(CONFIG)
    tuning = config["tuning"]
    best = best_per_method(read_csv(RESULTS / "tuning.csv"), "exact_gap")
    budgets = {p: problem_budget(config, p) for p in dict.fromkeys(p for p, _ in best)}
    jobs = [
        Job(str(CONFIG), p, m, params_of(method_spec(config, m, p), row), seed, budgets[p])
        for (p, m), row in best.items()
        for seed in int_list(tuning["test_seeds"])
    ]
    with ProcessPoolExecutor(default_workers()) as pool:
        rows = [row for part in pool.map(replay, jobs) for row in part]
    check_against_history(rows)
    write_csv(RESULTS / "estimated_value.csv", rows)
    plot(rows, budgets, names(tuning["methods"]))
    for (p, m), _ in best.items():
        last: dict[int, Row] = {r["seed"]: r for r in rows if (r["problem"], r["method"]) == (p, m)}
        raw = np.mean([r["estimated_value"] for r in last.values()])
        tail = [
            r["estimated_value_plain"]
            for r in rows
            if (r["problem"], r["method"]) == (p, m) and r["cumulative_observations"] > budgets[p] / 2
        ]
        plain = np.mean([r["estimated_value_plain"] for r in last.values()])
        print(
            f"{p:10s} {m:12s} last estimate: with entropy {raw:.3f}, plain {plain:.3f}; "
            f"second half plain mean {np.mean(tail):.3f}"
        )


if __name__ == "__main__":
    main()
