"""Tsallis divergence with entropic index q (``index``), q > 0 and q != 1.

Distance-generating function  w(x) = (sum_a x_a^q - 1) / (q (q - 1)),  grad w(x)_a = x_a^(q-1) / (q - 1),
so q -> 1 recovers KL and q = 2 is the squared Euclidean distance. For q < 1 the gradient blows up at
0 (a barrier, like KL: zero-probability actions stay at 0); for q > 1 actions can reach exactly 0.
pg-termination (Ju & Lan) uses q = 1 / (1 - log(1 - gamma) / log 2), about 0.231 for gamma = 0.9
(``index_from_gamma``).

SPMD step, per state: argmin_x eta <Q, x> + D(x, pi). The KKT conditions give, with multiplier lam
for sum_a x_a = 1 and c_a = pi_a^(q-1) / (q-1) - eta Q_a,

    x_a(lam) = ((q - 1) (c_a - lam))^(1 / (q - 1))     (0 where the base is <= 0)

and lam is the root of the decreasing function f(lam) = sum_a x_a(lam) - 1.

Root finding (the reference implementation does an exponential search for a bracket, then bisection
to 1e-6 with every candidate clamped at 1e-6; about 40+ evaluations per state, in Python):
  * Exact bracket, no search: f(lo) >= 0 and f(hi) <= 0 for lo = max_a c_a - 1/(q-1) (the largest
    term alone is 1) and hi = max_a c_a - n^(1-q)/(q-1) (every term is at most 1/n).
  * Safeguarded Newton: f'(lam) = -sum_{x_a > 0} x_a^(2-q) in closed form, so Newton steps converge
    quadratically; a step that leaves the bracket is replaced by bisection.
  * Warm start from the first-order solution lam ~ -sum pi^(2-q) eta Q / sum pi^(2-q) (exact as
    eta -> 0) when it lies in the bracket, else from lo. Machine precision in about 1-10 iterations
    per state (tests/test_divergence.py checks the average).
  * Compiled with numba and looped over states, without clamping, then renormalized.
"""

import math

import numpy as np

from ie692_rl._jit import njit
from ie692_rl.core.divergence import BregmanDivergence
from ie692_rl.types import FloatArray

MAX_ITERATIONS = 100


def index_from_gamma(gamma: float) -> float:
    """The Tsallis index used by pg-termination: 1 / (1 - log(1 - gamma) / log 2)."""
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    return 1.0 / (1.0 - math.log(1.0 - gamma) / math.log(2.0))


class Tsallis(BregmanDivergence):
    name = "tsallis"

    def __init__(self, index: float = 0.5) -> None:
        if not (math.isfinite(index) and index > 0) or index == 1:
            raise ValueError("Tsallis index must be positive, finite and != 1 (index 1 is KL)")
        self.index = float(index)

    def divergence(self, p: FloatArray, center: FloatArray) -> FloatArray:
        """w(p) - w(c) - <grad w(c), p - c>; +inf if q < 1 and p uses an action that c excludes."""
        q = self.index
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            grad = np.where(center > 0, center ** (q - 1), 0.0 if q > 1 else np.inf) / (q - 1)
            cross = np.where(p == center, 0.0, grad * (p - center))
            terms = (p**q - center**q) / (q * (q - 1)) - cross
        result: FloatArray = terms.sum(axis=1)
        return result

    def step(self, policy: FloatArray, q: FloatArray, eta: float, tau: float = 0.0) -> FloatArray:
        self.check_step_inputs(policy, q, eta, tau)
        if tau > 0:
            raise NotImplementedError("Entropy regularization (tau > 0) has a closed form only for KL")
        new, _ = tsallis_step(policy, self.scaled_costs(q, eta), self.index)
        return new

    def __repr__(self) -> str:
        return f"Tsallis(index={self.index:g})"


@njit
def tsallis_step(policy: FloatArray, scaled_costs: FloatArray, index: float) -> tuple[FloatArray, int]:
    """Row-wise Tsallis SPMD step given eta * (Q - row min); also returns the total Newton iterations."""
    n_states, n_actions = policy.shape
    e = index - 1.0
    new = np.zeros((n_states, n_actions))
    offsets = np.empty(n_actions)  # c_a; -inf for actions that stay at 0
    total_iterations = 0
    for s in range(n_states):
        top, n_free = -np.inf, 0
        weight_sum, weighted_cost = 0.0, 0.0
        for a in range(n_actions):
            if index < 1.0 and policy[s, a] <= 0.0:
                offsets[a] = -np.inf  # barrier: an excluded action stays excluded
            else:
                offsets[a] = policy[s, a] ** e / e - scaled_costs[s, a]
                n_free += 1
                if policy[s, a] > 0.0 and np.isfinite(scaled_costs[s, a]):
                    weight = policy[s, a] ** (1.0 - e)
                    weight_sum += weight
                    weighted_cost += weight * scaled_costs[s, a]
            top = max(top, offsets[a])
        lo, hi = top - 1.0 / e, top - float(n_free) ** (-e) / e  # f(lo) >= 0 >= f(hi)
        # Warm start: linearizing x_a around pi_a gives lam ~ -sum pi^(2-q) eta Q / sum pi^(2-q), nearly
        # exact for small steps; otherwise start at lo, where the largest term alone equals 1.
        lam = -weighted_cost / weight_sum if weight_sum > 0.0 else lo
        if not lo < lam < hi:
            lam = lo
        for _ in range(MAX_ITERATIONS):
            total_iterations += 1
            total, slope = 0.0, 0.0
            for a in range(n_actions):
                base = e * (offsets[a] - lam)
                if base > 0.0:
                    x = base ** (1.0 / e)
                    total += x
                    if x > 0.0:
                        slope += x ** (1.0 - e)  # x^(2 - q) = -d x / d lam
            if total > 1.0:
                lo = lam
            else:
                hi = lam
            if abs(total - 1.0) <= 1e-15 or hi - lo <= 1e-15 * max(1.0, abs(lam)):
                break
            newton = lam + (total - 1.0) / slope if slope > 0.0 else lo
            lam = newton if lo < newton < hi else 0.5 * (lo + hi)
        row_sum = 0.0
        for a in range(n_actions):
            base = e * (offsets[a] - lam)
            new[s, a] = base ** (1.0 / e) if base > 0.0 else 0.0
            row_sum += new[s, a]
        for a in range(n_actions):
            new[s, a] /= row_sum
    return new, total_iterations
