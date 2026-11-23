"""Interface shared by the policy-evaluation methods plugged into SPMD.

An estimator turns on-policy samples into an action-value estimate ``Q[s, a]`` of the current
policy (costs: lower is better). Concrete estimators are the baseline TOMC (``core.tomc.TOMC``)
and True Online TD(lambda) (``enhancements.true_online_td.TrueOnlineTD``).
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from ie692_rl.envs.base import Environment
from ie692_rl.types import FloatArray


class BudgetExhaustedError(RuntimeError):
    """The estimator could not finish within the remaining observation budget."""


class Estimate(NamedTuple):
    q: FloatArray  # Q[s, a] of the evaluated policy
    observations: int  # environment transitions used
    states_visited: int  # distinct states seen (GridWorld coverage diagnostic)


class QEstimator(ABC):
    """Policy evaluation from online samples."""

    @abstractmethod
    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        """Estimate Q of ``policy`` with at most ``max_observations`` transitions.

        ``extra_cost[s]`` is added to every sampled cost at state ``s`` (entropy-regularized SPMD
        uses it to sample the regularized Q). Raises ``BudgetExhaustedError`` if the budget is too small.
        """


def resolve_extra_cost(env: Environment, extra_cost: FloatArray | None) -> FloatArray:
    """Zero vector by default; validates shape and finiteness otherwise."""
    if extra_cost is None:
        return np.zeros(env.n_states)
    extra = np.asarray(extra_cost, dtype=np.float64)
    if extra.shape != (env.n_states,) or not np.all(np.isfinite(extra)):
        raise ValueError("extra_cost must be a finite vector with one entry per state")
    return extra
