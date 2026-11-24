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
  * Vectorized over states (each row stops updating once it has converged), without clamping,
    then renormalized.
"""

import math

import numpy as np

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


def tsallis_step(policy: FloatArray, scaled_costs: FloatArray, index: float) -> tuple[FloatArray, int]:
    """Row-wise Tsallis SPMD step given eta * (Q - row min); also returns the total Newton iterations
    (summed over states)."""
    e = index - 1.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore", under="ignore"):
        free = policy > 0.0 if index < 1.0 else np.ones(policy.shape, dtype=bool)  # q < 1: 0 stays 0
        offsets = np.where(free, policy**e / e - scaled_costs, -np.inf)  # c_a
        top = offsets.max(axis=1)
        lo, hi = top - 1.0 / e, top - free.sum(axis=1) ** (-e) / e  # f(lo) >= 0 >= f(hi)
        # Warm start: linearizing x_a around pi_a gives lam ~ -sum pi^(2-q) eta Q / sum pi^(2-q), nearly
        # exact for small steps; otherwise start at lo, where the largest term alone equals 1.
        weights = np.where((policy > 0.0) & np.isfinite(scaled_costs), policy ** (1.0 - e), 0.0)
        weight_sum = weights.sum(axis=1)
        weighted_cost = (weights * np.where(weights > 0.0, scaled_costs, 0.0)).sum(axis=1)
        lam = np.where(weight_sum > 0.0, -weighted_cost / weight_sum, lo)
        lam = np.where((lo < lam) & (lam < hi), lam, lo)
        active, total_iterations = np.ones(len(policy), dtype=bool), 0
        for _ in range(MAX_ITERATIONS):
            total_iterations += int(active.sum())
            x = _solution(offsets, lam, e)
            total = x.sum(axis=1)
            slope = np.where(x > 0.0, x ** (1.0 - e), 0.0).sum(axis=1)  # x^(2 - q) = -d x / d lam
            lo = np.where(active & (total > 1.0), lam, lo)
            hi = np.where(active & (total <= 1.0), lam, hi)
            active &= (np.abs(total - 1.0) > 1e-15) & (hi - lo > 1e-15 * np.maximum(1.0, np.abs(lam)))
            if not active.any():
                break
            newton = np.where(slope > 0.0, lam + (total - 1.0) / slope, lo)
            lam = np.where(active, np.where((lo < newton) & (newton < hi), newton, 0.5 * (lo + hi)), lam)
        new = _solution(offsets, lam, e)
    return new / new.sum(axis=1, keepdims=True), total_iterations


def _solution(offsets: FloatArray, lam: FloatArray, e: float) -> FloatArray:
    """x_a(lam) = ((q - 1) (c_a - lam))^(1 / (q - 1)), 0 where the base is <= 0."""
    base = e * (offsets - lam[:, None])
    return np.where(base > 0.0, np.where(base > 0.0, base, 1.0) ** (1.0 / e), 0.0)
