"""Shared fixtures: the project's problems as set up in the handout and proposal.

GridWorld 10x10 (handout: 10 hazards, 3 goals, 1 start), GridWorld 30x30 (90 hazards), simplified
Blackjack (cards 2-5, dealer stands at 9, bust above 11) and the M5 inventory problem; gamma = 0.9.
Environments and exact scorers are built once per test session.
"""

import numpy as np
import pytest

from ie692_rl.core.exact import ExactScorer
from ie692_rl.envs import Blackjack, Environment, GridWorld, RetailInventory
from ie692_rl.types import FloatArray

PROBLEMS = ("gridworld10", "gridworld30", "blackjack", "inventory")
SMALL_PROBLEMS = ("gridworld10", "blackjack", "inventory")


def build(name: str) -> Environment:
    if name == "gridworld10":
        return GridWorld()
    if name == "gridworld30":
        return GridWorld(size=30)
    if name == "blackjack":
        return Blackjack()
    if name == "inventory":
        return RetailInventory.from_m5()
    raise ValueError(name)


_ENVS: dict[str, Environment] = {}
_SCORERS: dict[str, ExactScorer] = {}


def get_env(name: str) -> Environment:
    if name not in _ENVS:
        _ENVS[name] = build(name)
    return _ENVS[name]


def get_scorer(name: str) -> ExactScorer:
    if name not in _SCORERS:
        _SCORERS[name] = ExactScorer(get_env(name))
    return _SCORERS[name]


@pytest.fixture(params=PROBLEMS)
def problem(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


@pytest.fixture(params=SMALL_PROBLEMS)
def small_problem(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    return name


def random_policy(env: Environment, seed: int, floor: float = 0.0) -> FloatArray:
    """Dirichlet rows mixed with ``floor`` of the uniform policy."""
    rows = np.random.default_rng(seed).dirichlet(np.ones(env.n_actions), size=env.n_states)
    policy: FloatArray = (1 - floor) * rows + floor / env.n_actions
    return policy
