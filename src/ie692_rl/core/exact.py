"""Exact (noise-free) policy scoring from an environment's known model.

Used only to *report* results (the gap to the optimal cost) and in tests; learners never see the
model. Policy values solve (I - gamma P_pi) V = c_pi directly (the problems have at most a few
hundred states, so the S x S system is small and the solve takes well under a millisecond). Optimal
values come from value iteration on the sparse ``TabularModel``, stopped once successive iterates
differ by at most tol * (1 - gamma) / gamma, which bounds the error by ``tol``.
"""

from functools import cached_property

import numpy as np

from ie692_rl.core.policy import check_policy
from ie692_rl.envs.base import Environment, TabularModel
from ie692_rl.types import FloatArray, IntArray

TOLERANCE = 1e-10
MAX_SWEEPS = 100_000


class ExactScorer:
    """Caches an environment's model so many policies can be scored quickly."""

    def __init__(self, env: Environment, tol: float = TOLERANCE) -> None:
        if not tol > 0:
            raise ValueError("tol must be positive")
        self.n_states, self.n_actions = env.n_states, env.n_actions
        self.gamma = env.gamma
        self.model: TabularModel = env.model()
        self.start = env.start_distribution()
        self.tol = tol

    def values(self, policy: FloatArray) -> FloatArray:
        """V^pi for every state."""
        check_policy(policy, self.n_states, self.n_actions)
        m, S = self.model, self.n_states
        rows = np.arange(S)[:, None, None] * S + m.next_states  # flat index of (s, s') in P_pi
        P = np.bincount(rows.ravel(), (policy[:, :, None] * m.probs).ravel(), minlength=S * S)
        cost = (policy * m.costs).sum(axis=1)
        return np.linalg.solve(np.eye(S) - self.gamma * P.reshape(S, S), cost)

    def q_values(self, policy: FloatArray) -> FloatArray:
        """Q^pi(s, a) = c(s, a) + gamma * E[V^pi(s')]."""
        m = self.model
        return backup(m.next_states, m.probs, m.costs, self.values(policy), self.gamma)

    @cached_property
    def optimal_values(self) -> FloatArray:
        m, v = self.model, np.zeros(self.n_states)
        for _ in range(MAX_SWEEPS):
            new = backup(m.next_states, m.probs, m.costs, v, self.gamma).min(axis=1)
            if np.abs(new - v).max() <= self._stop:
                return new
            v = new
        return v

    @cached_property
    def optimal(self) -> float:
        """Optimal expected discounted cost from the start distribution."""
        return float(self.start @ self.optimal_values)

    def score(self, policy: FloatArray) -> float:
        """Expected discounted cost from the start distribution (lower is better)."""
        return float(self.start @ self.values(policy))

    def gap(self, policy: FloatArray) -> float:
        """score(policy) - optimal (non-negative up to the tolerance)."""
        return self.score(policy) - self.optimal

    @property
    def _stop(self) -> float:
        return self.tol * (1 - self.gamma) / self.gamma


def backup(
    next_states: IntArray, probs: FloatArray, costs: FloatArray, v: FloatArray, gamma: float
) -> FloatArray:
    """Q(s, a) = c(s, a) + gamma * sum_k probs[s, a, k] * v[next_states[s, a, k]]."""
    return costs + gamma * (probs * v[next_states]).sum(axis=2)
