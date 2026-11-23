"""Exact (noise-free) policy scoring from an environment's known model.

Used only to *report* results (the gap to the optimal cost) and in tests; learners never see the
model. Policy evaluation and value iteration run on the sparse ``TabularModel`` in numba, so a
30x30 GridWorld is scored in milliseconds with O(|S||A|K) memory instead of a dense S x A x S
array and an S x S linear solve.

Both iterations are gamma-contractions in the sup norm; they stop once successive iterates differ
by at most tol * (1 - gamma) / gamma, which bounds the error of the result by ``tol``.
"""

from functools import cached_property

import numpy as np

from ie692_rl._jit import njit
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
        m = self.model
        return _policy_values(m.next_states, m.probs, m.costs, policy, self.gamma, self._stop, MAX_SWEEPS)

    def q_values(self, policy: FloatArray) -> FloatArray:
        """Q^pi(s, a) = c(s, a) + gamma * E[V^pi(s')]."""
        return backup(
            self.model.next_states, self.model.probs, self.model.costs, self.values(policy), self.gamma
        )

    @cached_property
    def optimal_values(self) -> FloatArray:
        m = self.model
        return _optimal_values(m.next_states, m.probs, m.costs, self.gamma, self._stop, MAX_SWEEPS)

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


@njit
def backup(
    next_states: IntArray, probs: FloatArray, costs: FloatArray, v: FloatArray, gamma: float
) -> FloatArray:
    S, A, K = probs.shape
    q = np.empty((S, A))
    for s in range(S):
        for a in range(A):
            expected = 0.0
            for k in range(K):
                expected += probs[s, a, k] * v[next_states[s, a, k]]
            q[s, a] = costs[s, a] + gamma * expected
    return q


@njit
def _policy_values(
    next_states: IntArray,
    probs: FloatArray,
    costs: FloatArray,
    policy: FloatArray,
    gamma: float,
    stop: float,
    max_sweeps: int,
) -> FloatArray:
    S, A = costs.shape
    v = np.zeros(S)
    for _ in range(max_sweeps):
        q = backup(next_states, probs, costs, v, gamma)
        change = 0.0
        for s in range(S):
            new = 0.0
            for a in range(A):
                new += policy[s, a] * q[s, a]
            change = max(change, abs(new - v[s]))
            v[s] = new
        if change <= stop:
            break
    return v


@njit
def _optimal_values(
    next_states: IntArray, probs: FloatArray, costs: FloatArray, gamma: float, stop: float, max_sweeps: int
) -> FloatArray:
    S, A = costs.shape
    v = np.zeros(S)
    for _ in range(max_sweeps):
        q = backup(next_states, probs, costs, v, gamma)
        change = 0.0
        for s in range(S):
            new = q[s, 0]
            for a in range(1, A):
                new = min(new, q[s, a])
            change = max(change, abs(new - v[s]))
            v[s] = new
        if change <= stop:
            break
    return v
