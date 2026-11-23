"""Performance and memory benchmark for the project pipeline (10-minute limit).

Measures, on this machine:
  1. kernel throughput (environment steps per second) for TOMC and True Online TD on every problem,
  2. a projection of the full ``tune.py`` pipeline from the config (total steps / throughput / workers),
  3. (with --run) the actual end-to-end ``tune.py`` wall time and peak memory, checked against the limit,
  4. memory of the data structures that grow with the problem size: the old stored-trajectory TOMC
     versus the ring buffer, dense versus sparse models and features,
  5. GridWorld 30x30 feasibility: how many TOMC pairs the uniform policy covers within a step budget.

    python benchmarks/bench.py                 # 1, 2, 4, 5 (about a minute)
    python benchmarks/bench.py --run           # also time the real pipeline

Writes a Markdown report to --output (default results/benchmark.md) and exits non-zero if the
projected (or, with --run, measured) time exceeds --limit seconds.
"""

import argparse
import math
import resource
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ie692_rl.core import TOMC, BudgetExhaustedError, ExactScorer, uniform_policy
from ie692_rl.enhancements.true_online_td import TrueOnlineTD
from ie692_rl.envs import Blackjack, Environment, GridWorld, RetailInventory
from ie692_rl.experiment import is_available, load_config, make_certifier, method_spec, names, problem_budget

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))


def peak_rss_mb() -> float:
    """Peak resident memory of this process and its finished children, in MB."""
    scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, kB on Linux
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return max(own, children) / 1e6


def throughput(run: Callable[[np.random.Generator], int], seconds: float = 1.0) -> float:
    """Steps per second of ``run(rng) -> steps`` after one warm-up call (compilation excluded)."""
    run(np.random.default_rng(0))
    steps, started, seed = 0, time.perf_counter(), 1
    while time.perf_counter() - started < seconds:
        steps += run(np.random.default_rng(seed))
        seed += 1
    return steps / (time.perf_counter() - started)


def tomc_run(env: Environment, steps: int) -> Callable[[np.random.Generator], int]:
    policy = uniform_policy(env.n_states, env.n_actions)

    def run(rng: np.random.Generator) -> int:
        try:
            return TOMC().estimate(env, policy, rng, steps).observations
        except BudgetExhaustedError:
            return steps

    return run


def td_run(env: Environment, steps: int) -> Callable[[np.random.Generator], int]:
    policy, estimator = uniform_policy(env.n_states, env.n_actions), TrueOnlineTD(n_steps=steps)
    return lambda rng: estimator.estimate(env, policy, rng, steps).observations


class Rates(NamedTuple):
    tomc: float  # steps per second
    td: float
    scoring: float  # seconds per exact score (done after every update)
    tomc_estimate: float  # steps in one TOMC estimate under the uniform policy (inf if it never finishes)
    certify: float  # seconds per certificate of a run's final policy (validation rollouts)


def tomc_estimate_length(env: Environment, cap: int = 50_000_000) -> float:
    policy = uniform_policy(env.n_states, env.n_actions)
    lengths = []
    for seed in range(3):
        try:
            lengths.append(TOMC().estimate(env, policy, np.random.default_rng(seed), cap).observations)
        except BudgetExhaustedError:
            return math.inf
    return float(np.mean(lengths))


def section_throughput(envs: dict[str, Environment]) -> tuple[list[str], dict[str, Rates]]:
    lines = [
        "## Kernel throughput (one core, compilation excluded)",
        "",
        "| problem | states | TOMC steps/s | True Online TD steps/s | exact scoring of one policy "
        "| certificate of one policy |",
        "|---|---|---|---|---|---|",
    ]
    rates = {}
    for name, env in envs.items():
        tomc = throughput(tomc_run(env, 2_000_000))
        td = throughput(td_run(env, 200_000))
        scorer = ExactScorer(env)
        policy = uniform_policy(env.n_states, env.n_actions)
        scorer.score(policy)  # warm-up (loads the cached kernel)
        started = time.perf_counter()
        scorer.score(policy)
        scoring = time.perf_counter() - started
        certifier = make_certifier(load_config(ROOT / "config.cfg"))
        certifier.certify(env, policy, np.random.default_rng(0))  # warm-up (compiles the rollout kernels)
        started = time.perf_counter()
        certifier.certify(env, policy, np.random.default_rng(1))
        certify = time.perf_counter() - started
        rates[name] = Rates(tomc, td, scoring, tomc_estimate_length(env), certify)
        lines.append(
            f"| {name} | {env.n_states} | {tomc / 1e6:.1f} M | {td / 1e6:.1f} M | {scoring * 1e3:.1f} ms "
            f"| {certify:.2f} s |"
        )
    return lines, rates


def section_projection(
    config_path: Path, problems: list[str], methods: list[str], rates: dict[str, Rates], workers: int
) -> tuple[list[str], float]:
    """Every run spends its budget: time = steps / rate + updates * exact-scoring time + one
    certificate of the final policy, over the workers.

    Updates per run are budget / steps-per-estimate (TOMC: one uniform-policy estimate; TD: n_steps).
    """
    config = load_config(config_path)
    tuning = config["tuning"]
    fraction = float(tuning["tune_fraction"])
    n_tune, n_test = len(names(tuning["tuning_seeds"])), len(names(tuning["test_seeds"]))
    lines = [
        "## Projected pipeline time",
        "",
        f"{workers} worker processes; tuning on {fraction:.0%} of each budget with {n_tune} seeds, "
        f"then {n_test} test seeds on the full budget.",
        "",
        "| problem | method | combinations | environment steps | projected time |",
        "|---|---|---|---|---|",
    ]
    total_seconds = 0.0
    for problem in problems:
        budget = problem_budget(config, problem)
        for method in methods:
            spec = method_spec(config, method, problem)
            combos = len(spec.combinations())
            steps = combos * n_tune * budget * fraction + n_test * budget
            rate = rates[problem]
            if spec.estimator == "true_online_td":
                per_estimate = max(float(v) for v in spec.grid.get("estimator.n_steps", [10_000]))
                sampling = steps / rate.td
            else:
                per_estimate, sampling = rate.tomc_estimate, steps / rate.tomc
            runs = combos * n_tune + n_test
            seconds = (
                (sampling + steps / per_estimate * rate.scoring + runs * rate.certify) / workers
                if math.isfinite(per_estimate)
                else math.inf
            )  # TOMC never finishes an estimate: no update is ever made
            total_seconds += seconds
            lines.append(f"| {problem} | {method} | {combos} | {steps:.3g} | {seconds:.1f} s |")
    lines += [
        "",
        f"**Projected time: {total_seconds:.0f} s** (plus about 1-3 s of compilation per worker). "
        "TOMC rows assume every estimate is as long as one under the uniform policy; an infinite "
        "projection means a single estimate does not finish. CPUs that mix performance and "
        "efficiency cores give less than a linear speed-up, so use --run for the measured time.",
    ]
    return lines, total_seconds


def section_memory() -> list[str]:
    from reference import reference_tomc

    env = GridWorld()
    policy = uniform_policy(env.n_states, env.n_actions)
    tracemalloc.start()
    _, n = reference_tomc(env, policy, np.random.default_rng(0), 0.01, 10**8)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    per_step = peak / n
    big = GridWorld(size=30)
    sparse = big.model()
    S, A = big.n_states, big.n_actions
    sparse_bytes = sparse.next_states.nbytes + sparse.probs.nbytes + sparse.costs.nbytes
    return [
        "## Memory",
        "",
        "| structure | before | now |",
        "|---|---|---|",
        f"| TOMC stream (stored trajectory vs ring buffer) | {per_step:.0f} B per step: "
        f"{per_step * 1e8 / 1e9:.1f} GB for 10^8 steps | O(S A + 350) values: "
        f"{(3 * S * A * 8 + 350 * 9) / 1e3:.0f} kB on 30x30, any length |",
        f"| exact model, 30x30 (dense P vs sparse) | {S * A * S * 8 / 1e6:.0f} MB "
        f"| {sparse_bytes / 1e6:.2f} MB |",
        f"| TD tabular features, 30x30 (dense Phi vs index table) | {(S * A) ** 2 * 8 / 1e6:.0f} MB "
        f"| {S * A * 8 / 1e3:.0f} kB |",
        f"| TD history (Q snapshot per record vs V(s0)) | {S * A * 8 / 1e3:.0f} kB per record "
        "| 16 B per record |",
    ]


def section_gridworld30(max_steps: int) -> list[str]:
    """Coverage of TOMC's required pairs on 30x30 under the uniform policy, by step budget."""
    env = GridWorld(size=30)
    policy = uniform_policy(env.n_states, env.n_actions)
    lines = [
        "## GridWorld 30x30 feasibility (TOMC, uniform policy)",
        "",
        "| step budget | required pairs covered | time |",
        "|---|---|---|",
    ]
    budget = 1_000_000
    while budget <= max_steps:
        started = time.perf_counter()
        try:
            TOMC().estimate(env, policy, np.random.default_rng(0), budget)
            covered = "all 3600"
        except BudgetExhaustedError as error:
            covered = str(error).split(": ")[1].split(" after")[0]
        lines.append(f"| {budget:.0e} | {covered} | {time.perf_counter() - started:.1f} s |")
        budget *= 10
    lines += [
        "",
        "Hazards cover 10% of the 30x30 grid and send the agent back to the start with probability "
        "0.8, so a uniformly random walk rarely reaches the far side: coverage grows by only a few "
        "hundred pairs per tenfold increase in steps. Extrapolating that trend, a single TOMC estimate "
        "would not finish within the ~1.8e10 steps one core samples in 10 minutes. This is a property "
        "of the problem, not of the step rate.",
    ]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.cfg")
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--limit", type=float, default=600.0, help="seconds")
    parser.add_argument("--run", action="store_true", help="also run tune.py end to end and time it")
    parser.add_argument(
        "--gridworld30-steps", type=float, default=1e8, help="largest TOMC budget to try on 30x30"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark.md")
    args = parser.parse_args()

    config = load_config(args.config)
    problems = args.problems or names(config["tuning"]["problems"])
    methods = args.methods or [
        m
        for m in names(config["tuning"]["methods"])
        if all(is_available(method_spec(config, m, p)) for p in problems)
    ]
    envs: dict[str, Environment] = {
        "GridWorld": GridWorld(),
        "GridWorld30": GridWorld(size=30),
        "Blackjack": Blackjack(),
        "Inventory": RetailInventory.from_m5(),
    }
    report = [f"# Benchmark ({time.strftime('%Y-%m-%d %H:%M')}, limit {args.limit:.0f} s)", ""]
    lines, rates = section_throughput(envs)
    report += [*lines, ""]
    lines, projected = section_projection(args.config, problems, methods, rates, args.workers)
    report += [*lines, ""]
    ok = projected <= args.limit
    if args.run:
        import tune

        argv = [
            "--config",
            str(args.config),
            "--workers",
            str(args.workers),
            "--problems",
            *problems,
            "--methods",
            *methods,
            "--output",
            str(args.output.parent / "benchmark_run"),
        ]
        started = time.perf_counter()
        tune.main(argv)
        elapsed = time.perf_counter() - started
        ok = elapsed <= args.limit
        report += [
            "## Measured pipeline",
            "",
            f"`tune.py` end to end: **{elapsed:.1f} s** wall time, peak "
            f"memory {peak_rss_mb():.0f} MB per process ({'within' if ok else 'OVER'} the limit).",
            "",
        ]
    report += [*section_memory(), ""]
    report += [*section_gridworld30(int(args.gridworld30_steps)), ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nreport written to {args.output}")
    if not math.isfinite(projected) or not ok:
        print(f"FAIL: over the {args.limit:.0f} s limit", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
