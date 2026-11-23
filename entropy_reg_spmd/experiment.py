"""Shared pieces for the online experiments (optimization.ipynb and tune.py).

- problems built from config.cfg or from a dotted class name,
- the entropy-cost wrapper that makes sampled Q the entropy-regularized Q,
- one online SPMD run with the team's TOMC estimator (baseline.tomc), recording what the project
  guideline asks for at every iteration: the estimated value at the start state and the number of
  observations, plus the exact value as a noise-free check.

The baseline is the team's KL-SPMD step (baseline.spmd), which is entropy_reg_spmd with tau = 0.
"""
import configparser
import importlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the team's code lives in the repository root
import baseline  # noqa: E402

from entropy_reg_spmd.entropy_reg_spmd import entropy_reg_spmd  # noqa: E402
from entropy_reg_spmd.exact_eval import ExactScorer  # noqa: E402

CONFIG = HERE / "config.cfg"
CURVE_POINTS = np.linspace(0, 1, 101)  # fractions of the budget at which learning curves are recorded
# Settings in a problem section passed on to the environment constructor, with their types.
ENV_SETTINGS = {"capacity": int, "unit_cost_fraction": float, "holding": float, "fixed_order": float,
                "start_stock": int, "start_weekday": int, "weekly": lambda text: text.lower() == "true"}


def load_config(path=CONFIG):
    config = configparser.ConfigParser()
    if not config.read(path):
        raise FileNotFoundError(path)
    return config


def floats(text):
    return [float(x) for x in text.split(",")]


def ints(text):
    return [int(x) for x in text.split(",")]


def resolve(dotted):
    """Object for a dotted name such as grid_world_env.GridWorld or a.module.Class.method."""
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
            break
        except ModuleNotFoundError:
            continue
    else:
        raise ImportError(dotted)
    for name in parts[split:]:
        obj = getattr(obj, name)
    return obj


def make_env(section):
    """Environment described by a problem section of config.cfg."""
    settings = {key: convert(section[key]) for key, convert in ENV_SETTINGS.items() if key in section}
    if "data" in section:
        settings["path"] = HERE / section["data"]
    return resolve(section["environment"])(**settings)


@lru_cache(maxsize=None)
def problem(environment):
    """(env, exact scorer) for a dotted class name; built once per process."""
    env = resolve(environment)()
    return env, ExactScorer(env)


class EntropyCost:
    """The same environment, with tau * sum_a pi log pi added to every step's cost."""

    def __init__(self, env, policy, tau):
        self.env = env
        self.extra = tau * np.sum(policy * np.log(np.maximum(policy, 1e-300)), axis=1)

    def __getattr__(self, name):  # gamma, n_states, start, blackjack's deal, ... come from the environment
        return getattr(self.env, name)

    def step_one(self, state, action, rng):
        next_state, cost, done = self.env.step_one(state, action, rng)
        return next_state, cost + self.extra[state], done


def run(env, scorer, tau, eta, bias, budget, seed):
    """Online SPMD from the uniform policy; one record per iteration (iteration 0 = uniform policy).

    Each iteration estimates Q with baseline.tomc (tail tolerance `bias`) and takes the KL step with
    entropy weight tau (tau = 0: the team's KL-SPMD). estimated_value is sum_a pi(a|s0) Q_hat(s0, a)
    from that iteration's samples (it includes the entropy cost when tau > 0); exact_value and gap
    score the updated policy exactly. If the budget runs out before an estimate is complete, the
    remaining steps are spent without an update.
    """
    rng = np.random.default_rng(seed)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    used, exact = 0, scorer.score(policy)
    history = [dict(iteration=0, observations=0, cumulative_observations=0, estimated_value=np.nan,
                    exact_value=exact, gap=exact - scorer.optimal)]
    while used < budget:
        sampled = EntropyCost(env, policy, tau) if tau > 0 else env
        try:
            q, n = baseline.tomc(sampled, policy, rng, bias=bias, max_observations=budget - used)
        except RuntimeError:  # coverage or return tail not finished within the budget
            break
        used += n
        estimated = float(scorer.start @ (policy * q).sum(axis=1))
        policy = entropy_reg_spmd(policy, q, eta, tau)
        exact = scorer.score(policy)
        history.append(dict(iteration=len(history), observations=n, cumulative_observations=used,
                            estimated_value=estimated, exact_value=exact, gap=exact - scorer.optimal))
    return history


def gap_curve(history, budget, points=CURVE_POINTS):
    """Exact gap at each fraction of the budget: the gap of the policy in use once that many
    samples have been spent (the policy only changes when an iteration's samples are complete)."""
    used = np.array([h["cumulative_observations"] for h in history]) / budget
    gaps = np.array([h["gap"] for h in history])
    return gaps[np.searchsorted(used, points, side="right") - 1]


def curve_job(job):
    """Worker for a process pool: (environment name, tau, eta, bias, budget, seed) -> gap curve."""
    environment, tau, eta, bias, budget, seed = job
    env, scorer = problem(environment)
    return gap_curve(run(env, scorer, tau, eta, bias, budget, seed), budget)
