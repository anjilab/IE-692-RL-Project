"""Config-driven experiments: build problems and methods, tune on a fraction of the budget, run in full.

Follows the project guideline: grid search over at most three algorithmic parameters (three values
each); every combination runs on ``tune_fraction`` of the full observation budget, averaged over the
tuning seeds; the best combination per method then runs on the full budget with fresh test seeds.

``config.cfg`` sections (see the file for the full list):

    [tuning]                        problems, methods, seeds, tune_fraction, baseline, z test settings
    [problem <name>]                environment = gridworld | blackjack | inventory, budget, env settings
    [method <name>]                 estimator = tomc | true_online_td, divergence = kl | euclidean | tsallis,
                                    then parameters: spmd.step_size, spmd.tau, spmd.tau_power,
                                    estimator.<field>, divergence.<field>. A comma-separated value is a
                                    grid axis; a single value is fixed. estimator_from = <method> takes
                                    the estimator.* settings this method leaves unset from that
                                    method's tuned choice on the same problem (two-stage tuning).
    [method <name> @ <problem>]     overrides for one problem (for example a TD budget per iteration)

Each run is a ``Job``; workers rebuild (and cache) the environment from the config path, so jobs are
small and picklable for a process pool.
"""

import configparser
import csv
import math
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from ie692_rl.core.divergence import BregmanDivergence
from ie692_rl.core.estimator import QEstimator
from ie692_rl.core.exact import ExactScorer
from ie692_rl.core.optimality import RolloutCertifier
from ie692_rl.core.spmd import Schedule, run_spmd
from ie692_rl.core.tomc import TOMC
from ie692_rl.enhancements.bregman_spmd import KL, Euclidean, Tsallis
from ie692_rl.enhancements.entropy_reg_spmd import AnnealedTau
from ie692_rl.enhancements.true_online_td import SarsaLambda, TDLambda, TDZero, TrueOnlineTD
from ie692_rl.envs import Blackjack, Environment, GridWorld, RetailInventory
from ie692_rl.types import FloatArray

Value = int | float | bool | str
Params = tuple[tuple[str, Value], ...]  # sorted (key, value) pairs: hashable and picklable
Row = dict[str, Any]

ENVIRONMENTS: dict[str, Any] = {
    "gridworld": GridWorld,
    "blackjack": Blackjack,
    "inventory": RetailInventory.from_m5,
}
ESTIMATORS: dict[str, type[QEstimator]] = {
    "tomc": TOMC,
    "true_online_td": TrueOnlineTD,
    "td_zero": TDZero,
    "td_lambda": TDLambda,
    "sarsa_lambda": SarsaLambda,
}
DIVERGENCES: dict[str, type[BregmanDivergence]] = {"kl": KL, "euclidean": Euclidean, "tsallis": Tsallis}
SPMD_KEYS = ("step_size", "tau", "tau_power")
PROBLEM_KEYS = ("environment", "budget", "label")
MAX_TUNED_PARAMETERS = 3


# ----------------------------------------------------------------------------- config


def load_config(path: Path | str) -> configparser.ConfigParser:
    config = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not config.read(path):
        raise FileNotFoundError(path)
    return config


def parse_value(text: str) -> Value:
    """int, float, bool (true/false) or the stripped string."""
    text = text.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for convert in (int, float):
        try:
            return convert(text)
        except ValueError:
            continue
    return text


def parse_list(text: str) -> list[Value]:
    return [parse_value(item) for item in text.split(",") if item.strip()]


def int_list(text: str) -> list[int]:
    values = parse_list(text)
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise ValueError(f"Expected integers: {text!r}")
    return [int(v) for v in values]


def names(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass(frozen=True)
class MethodSpec:
    """A method's estimator/divergence names and its parameter grid on one problem."""

    name: str
    estimator: str
    divergence: str
    grid: dict[str, list[Value]]
    estimator_from: str | None = None  # method whose tuned estimator settings fill the unset ones

    def combinations(self) -> list[Params]:
        keys = sorted(self.grid)
        return [tuple(zip(keys, values, strict=True)) for values in product(*(self.grid[k] for k in keys))]

    @property
    def tuned(self) -> list[str]:
        return [key for key, values in self.grid.items() if len(values) > 1]


def method_spec(config: configparser.ConfigParser, method: str, problem: str) -> MethodSpec:
    section = f"method {method}"
    if not config.has_section(section):
        raise KeyError(f"No [{section}] section in the config")
    settings = dict(config[section])
    problem_section = config[f"problem {problem}"]
    settings.update({key: value for key, value in problem_section.items() if key.startswith("spmd.")})
    if config.has_section(f"{section} @ {problem}"):
        settings.update(config[f"{section} @ {problem}"])
    settings.pop("label", None)  # display name for figures, see method_label
    estimator_from = settings.pop("estimator_from", None)
    estimator, divergence = settings.pop("estimator"), settings.pop("divergence")
    if estimator not in ESTIMATORS or divergence not in DIVERGENCES:
        raise ValueError(f"[{section}]: unknown estimator {estimator!r} or divergence {divergence!r}")
    grid = {key: parse_list(value) for key, value in settings.items()}
    for key in grid:
        prefix, _, field = key.partition(".")
        if prefix not in ("spmd", "estimator", "divergence") or not field:
            raise ValueError(
                f"[{section}]: parameter {key!r} must start with spmd., estimator. or divergence."
            )
        if prefix == "spmd" and field not in SPMD_KEYS:
            raise ValueError(f"[{section}]: unknown SPMD parameter {key!r}")
    if "spmd.step_size" not in grid:
        raise ValueError(f"[{section}]: spmd.step_size is required")
    spec = MethodSpec(method, estimator, divergence, grid, estimator_from)
    if len(spec.tuned) > MAX_TUNED_PARAMETERS:
        raise ValueError(f"[{section}]: the guideline allows at most 3 tuned parameters, got {spec.tuned}")
    return spec


def inherit_estimator(spec: MethodSpec, parent: MethodSpec, chosen: Row) -> MethodSpec:
    """``spec`` with every estimator.* setting it leaves unset fixed at ``parent``'s tuned choice."""
    if parent.estimator != spec.estimator:
        raise ValueError(
            f"{spec.name} ({spec.estimator}) cannot take estimator settings from {parent.name} "
            f"({parent.estimator})"
        )
    inherited = {k: [chosen[k]] for k in parent.grid if k.startswith("estimator.") and k not in spec.grid}
    return MethodSpec(spec.name, spec.estimator, spec.divergence, spec.grid | inherited, spec.estimator_from)


def make_env(config: configparser.ConfigParser, problem: str) -> Environment:
    section = config[f"problem {problem}"]
    settings = {
        key: parse_value(value)
        for key, value in section.items()
        if key not in PROBLEM_KEYS and not key.startswith("spmd.")
    }
    env: Environment = ENVIRONMENTS[section["environment"]](**settings)
    return env


def method_label(config: configparser.ConfigParser, method: str) -> str:
    """Display name of a method in figures (``label`` in its section; mathtext allowed)."""
    return config[f"method {method}"].get("label", method)


def problem_label(config: configparser.ConfigParser, problem: str) -> str:
    """Display name of a problem in figures (``label`` in its section; mathtext allowed)."""
    return config[f"problem {problem}"].get("label", problem)


def problem_budget(config: configparser.ConfigParser, problem: str) -> int:
    return int(config[f"problem {problem}"]["budget"])


def figure_groups(config: configparser.ConfigParser, methods: Sequence[str]) -> dict[str, list[str]]:
    """Methods shown together in one set of figures: ``[figures]`` maps a group name (the figure file
    suffix) to its methods. Without that section, all ``methods`` share one figure and no suffix."""
    if not config.has_section("figures"):
        return {"": list(methods)}
    groups = {group: names(value) for group, value in config["figures"].items()}
    for group, members in groups.items():
        unknown = [m for m in members if m not in methods]
        if unknown:
            raise ValueError(f"[figures] {group}: {unknown} not in [tuning] methods")
    return groups


# ----------------------------------------------------------------------------- selection


SELECTION_KEYS = {
    # model-free (default): smallest lower confidence bound on the gap, ties broken by the upper bound
    "gap_lower": ("mean_gap_lower", "mean_gap_upper"),
    # uses the exact model: smallest exact gap to the optimum
    "exact_gap": ("mean_final_gap",),
}


def selection_of(config: configparser.ConfigParser) -> str:
    selection = config["tuning"].get("selection", "gap_lower")
    if selection not in SELECTION_KEYS:
        raise ValueError(f"[tuning] selection must be one of {sorted(SELECTION_KEYS)}, got {selection!r}")
    return selection


def make_certifier(config: configparser.ConfigParser) -> RolloutCertifier:
    """The optimality certifier applied to every run's final policy ([tuning] certifier_* settings)."""
    tuning = config["tuning"]
    return RolloutCertifier(
        n_rollouts=int(tuning.get("certifier_rollouts", "20")),
        n_start_rollouts=int(tuning.get("certifier_start_rollouts", "2000")),
        z=float(tuning.get("certifier_z", "3.0")),
    )


# ----------------------------------------------------------------------------- building a method


@dataclass(frozen=True)
class Method:
    estimator: QEstimator
    divergence: BregmanDivergence
    step_size: float
    tau: float | Schedule


def build_method(spec: MethodSpec, params: Params) -> Method:
    """Dependency injection: instantiate the estimator and divergence named in the spec."""
    groups: dict[str, dict[str, Value]] = {"spmd": {}, "estimator": {}, "divergence": {}}
    for key, value in params:
        prefix, _, field = key.partition(".")
        groups[prefix][field] = value
    spmd = groups["spmd"]
    tau0 = float(spmd.get("tau", 0.0))
    tau: float | Schedule = AnnealedTau(tau0, float(spmd["tau_power"])) if "tau_power" in spmd else tau0
    return Method(
        ESTIMATORS[spec.estimator](**groups["estimator"]),
        DIVERGENCES[spec.divergence](**groups["divergence"]),
        float(spmd["step_size"]),
        tau,
    )


# ----------------------------------------------------------------------------- running jobs


@dataclass(frozen=True)
class Job:
    config_path: str
    problem: str
    method: str
    params: Params
    seed: int
    budget: int


@cache
def _problem(config_path: str, problem: str) -> tuple[Environment, ExactScorer]:
    """(env, exact scorer), built once per worker process."""
    env = make_env(load_config(config_path), problem)
    return env, ExactScorer(env)


def run_job(job: Job) -> tuple[Job, list[Row]]:
    """Worker: one SPMD run; returns one row per iteration."""
    config = load_config(job.config_path)
    env, scorer = _problem(job.config_path, job.problem)
    method = build_method(method_spec(config, job.method, job.problem), job.params)
    result = run_spmd(
        env,
        method.estimator,
        method.divergence,
        method.step_size,
        job.budget,
        job.seed,
        tau=method.tau,
        scorer=scorer,
        certifier=make_certifier(config),
    )
    base = {"problem": job.problem, "method": job.method, **dict(job.params), "seed": job.seed}
    return job, [{**base, **record.as_dict()} for record in result.history]


def map_jobs(jobs: Sequence[Job], workers: int) -> dict[Job, list[Row]]:
    """Run jobs in parallel (largest budgets first for load balance), or in-process if workers == 1."""
    ordered = sorted(jobs, key=lambda job: -job.budget)
    if workers <= 1:
        return dict(map(run_job, ordered))
    with ProcessPoolExecutor(min(workers, len(ordered)) or 1) as pool:
        return dict(pool.map(run_job, ordered, chunksize=1))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


# ----------------------------------------------------------------------------- tuning and summaries


def mean_and_se(values: Sequence[float] | FloatArray) -> tuple[float, float]:
    """Mean and standard error; the error is NaN for fewer than two values (no ddof warning)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan, math.nan
    se = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else math.nan
    return float(array.mean()), se


def tuning_jobs(
    config_path: str, problem: str, spec: MethodSpec, budget: int, seeds: Sequence[int]
) -> list[Job]:
    return [
        Job(config_path, problem, spec.name, params, seed, budget)
        for params in spec.combinations()
        for seed in seeds
    ]


def tuning_rows(spec: MethodSpec, problem: str, budget: int, results: dict[Job, list[Row]]) -> list[Row]:
    """One row per combination: mean final exact gap and certified gap bounds over the tuning seeds,
    and the fewest updates made."""
    by_params: dict[Params, list[list[Row]]] = {}
    for job, history in results.items():
        if job.problem == problem and job.method == spec.name:
            by_params.setdefault(job.params, []).append(history)
    rows = []
    for params in spec.combinations():
        histories = by_params.get(params, [])
        mean, se = mean_and_se([h[-1]["gap"] for h in histories])
        lower, lower_se = mean_and_se([h[-1]["gap_lower"] for h in histories])
        upper, _ = mean_and_se([h[-1]["gap_upper"] for h in histories])
        updates = min((sum(r["status"] == "updated" for r in h) for h in histories), default=0)
        rows.append(
            {
                "problem": problem,
                "method": spec.name,
                **dict(params),
                "tuning_budget": budget,
                "seeds": len(histories),
                "min_updates": updates,
                "mean_final_gap": mean,
                "standard_error": se,
                "mean_gap_lower": lower,
                "gap_lower_standard_error": lower_se,
                "mean_gap_upper": upper,
            }
        )
    return rows


def best_per_method(rows: Iterable[Row], selection: str = "gap_lower") -> dict[tuple[str, str], Row]:
    """Chosen combination per (problem, method), among those where every seed made an update.

    ``gap_lower``: smallest mean certified lower bound on the gap (model-free), ties broken by the mean
    upper bound; ``exact_gap``: smallest mean exact gap (needs the model).
    """
    if selection not in SELECTION_KEYS:
        raise ValueError(f"selection must be one of {sorted(SELECTION_KEYS)}")
    keys = SELECTION_KEYS[selection]
    best: dict[tuple[str, str], Row] = {}
    for row in rows:
        key = (row["problem"], row["method"])
        score = tuple(float(row.get(k, math.nan)) for k in keys)
        eligible = row["min_updates"] > 0 and math.isfinite(score[0])
        if eligible and (key not in best or score < tuple(float(best[key].get(k, math.nan)) for k in keys)):
            best[key] = row
    return best


def params_of(spec: MethodSpec, row: Row) -> Params:
    return tuple((key, row[key]) for key in sorted(spec.grid))


def z_verdict(diff: float, se: float, z_threshold: float, min_worse_diff: float) -> tuple[float, str]:
    """Two-sample z statistic and verdict; se = 0 means a deterministic difference (z = +-inf)."""
    if not math.isfinite(se):
        return math.nan, "n/a"
    deterministic = 0.0 if diff == 0 else math.copysign(math.inf, diff)
    z = deterministic if se == 0 else diff / se
    if z <= -z_threshold:
        return z, "better"
    if z >= z_threshold and diff >= min_worse_diff:
        return z, "worse"
    return z, "tie"


def summarize(
    history_rows: Iterable[Row], baseline: str, z_threshold: float = 2.0, min_worse_diff: float = 0.001
) -> list[Row]:
    """Final exact gap per (problem, method) with a z-test against the baseline method."""
    finals: dict[tuple[str, str], dict[int, float]] = {}
    lowers: dict[tuple[str, str], dict[int, float]] = {}
    for row in history_rows:  # rows are in iteration order, so the last row per seed is the final policy
        finals.setdefault((row["problem"], row["method"]), {})[row["seed"]] = row["gap"]
        lowers.setdefault((row["problem"], row["method"]), {})[row["seed"]] = row.get("gap_lower", math.nan)
    summary = []
    for (problem, method), by_seed in finals.items():
        gaps = np.array(list(by_seed.values()))
        base = np.array(list(finals.get((problem, baseline), {}).values()))
        mean, se = mean_and_se(gaps)
        diff, z, verdict = math.nan, math.nan, ""
        if method != baseline and base.size:
            diff = mean - float(base.mean())
            se_diff = (
                math.sqrt(gaps.var(ddof=1) / gaps.size + base.var(ddof=1) / base.size)
                if gaps.size > 1 and base.size > 1
                else math.nan
            )
            z, verdict = z_verdict(diff, se_diff, z_threshold, min_worse_diff)
        summary.append(
            {
                "problem": problem,
                "method": method,
                "seeds": int(gaps.size),
                "mean_final_gap": mean,
                "standard_error": se,
                "mean_final_gap_lower": float(np.mean(list(lowers[problem, method].values()))),
                "diff_vs_baseline": diff,
                "z": z,
                "verdict": verdict,
            }
        )
    return summary


def gap_curve(history: Sequence[Row], budget: int, points: FloatArray, key: str = "gap") -> FloatArray:
    """``key`` (default the exact gap) at each fraction of the budget: the value for the policy in use
    once that many samples have been spent (the policy only changes when an iteration's samples are
    complete). Closing rows without an update ("budget", "diverged") do not change the policy."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    rows = [h for h in history if h.get("status", "updated") in ("start", "updated")]
    used = np.array([h["cumulative_observations"] for h in rows], dtype=np.float64) / budget
    values = np.array([h[key] for h in rows], dtype=np.float64)
    curve: FloatArray = values[np.searchsorted(used, points, side="right") - 1]
    return curve


# ----------------------------------------------------------------------------- CSV


def write_csv(path: Path, rows: Sequence[Row]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> Iterator[Row]:
    """Rows of a CSV written by ``write_csv``, numbers converted back (empty cells dropped)."""
    with open(path) as handle:
        for raw in csv.DictReader(handle):
            yield {
                key: parse_value(value) if key not in ("problem", "method") else value
                for key, value in raw.items()
                if value != ""
            }
