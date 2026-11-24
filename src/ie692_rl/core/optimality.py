"""Model-free optimality certificate: the advantage gap.

For costs and any policy pi, let V(s) = sum_a pi(a|s) Q(s, a) with Q = Q^pi, and define the advantage gap

    g(s) = V(s) - min_a Q(s, a)  >= 0.

Two facts bound the true gap to the optimum without knowing V*:

  * lower bound: Q^pi >= Q* gives min_a Q^pi(s, a) >= V*(s), so g(s) <= V^pi(s) - V*(s);
  * upper bound: the performance-difference lemma gives
        V^pi(s) - V*(s) = E_{s' ~ d_s^{pi*}} [ V^pi(s') - <Q^pi(s', .), pi*(.|s')> ] / (1 - gamma)
                        <= max_s' g(s') / (1 - gamma).

So  rho . g  <=  rho . (V^pi - V*)  <=  max_s g(s) / (1 - gamma)  for any start distribution rho, and g = 0
everywhere exactly when pi is optimal. The upper bound is loose (the 1 / (1 - gamma) factor and the max
over states).

A tighter lower bound follows from the same gap. For any policy pi', V*(s) <= V^pi'(s), so
V^pi(s) - V^pi'(s) <= V^pi(s) - V*(s). With pi' greedy for Q^pi the performance-difference lemma gives
V^pi(s) - V^pi'(s) = E_{s' ~ d_s^{pi'}} [g(s')] / (1 - gamma) >= g(s): the advantage gap summed along
the improved policy's path. It stays valid when pi' is only approximately greedy (noisy Q-hat), and is
much tighter in long-horizon problems (random GridWorld policy: 0.31 against a true gap of 0.36, where
g(s0) is 0.02). The certificate's lower bound is the larger of the two.

Plugging in the training estimator's Q-hat does NOT give a valid certificate: TOMC fills actions with
pi(a|s) < pi_min with the pessimistic value cost_bound / (1 - gamma), so an action the policy abandoned is
invisible to min_a Q-hat and g looks like 0 (the Blackjack lock-in). ``RolloutCertifier`` therefore
estimates Q^pi for every (state, action) with fresh rollouts: take action a in state s, then follow pi.
This needs only the simulator (starting it in any state, i.e. a generative model), never the model P.
Rollouts run until gamma^t <= 1e-16 or the episode ends, so truncation is negligible; sampling noise is
handled with ``z``-standard-error intervals on the differences D_b (see ``gap_bounds``), using many
more rollouts at the start states, where the lower bound is read.
"""

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numba import njit

from ie692_rl.core.policy import check_policy, policy_cdf, sample_action
from ie692_rl.core.tomc import return_window
from ie692_rl.envs.base import Environment, StepKernel
from ie692_rl.types import FloatArray, IntArray


class GapBounds(NamedTuple):
    lower: float  # rho . g  <=  true gap from the start distribution
    upper: float  # max_s g / (1 - gamma)  >=  true gap from any state
    per_state: FloatArray  # g(s)


def advantage_gap(q: FloatArray, policy: FloatArray) -> FloatArray:
    """g(s) = sum_a pi(a|s) Q(s, a) - min_a Q(s, a), clipped at 0 against rounding."""
    if q.shape != policy.shape:
        raise ValueError("Q and policy must have the same shape")
    if not np.all(np.isfinite(q)):
        raise ValueError("Q must be finite")
    gap: FloatArray = np.maximum((policy * q).sum(axis=1) - q.min(axis=1), 0.0)
    return gap


def gap_bounds(
    q: FloatArray,
    policy: FloatArray,
    gamma: float,
    start: FloatArray,
    q_se: FloatArray | None = None,
    z: float = 0.0,
) -> GapBounds:
    """Lower and upper bounds on the optimality gap from Q^pi (exact or estimated).

    g(s) = max_b D_b(s) with D_b(s) = sum_a pi(a|s) Q(s, a) - Q(s, b). With independent estimates of
    standard errors ``q_se``, sd(D_b)^2 = sum_a (pi(a|s) - [a = b])^2 se(s, a)^2, and each D_b is
    replaced by D_b - z sd (for the lower bound) or D_b + z sd (for the upper bound). With exact Q
    (no ``q_se``) both reduce to g.
    """
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    if not (math.isfinite(z) and z >= 0):
        raise ValueError("z must be non-negative and finite")
    if start.shape != (q.shape[0],) or np.any(start < 0) or not math.isclose(float(start.sum()), 1.0):
        raise ValueError("start must be a probability vector over states")
    se = np.zeros_like(q) if q_se is None else np.asarray(q_se, dtype=np.float64)
    if se.shape != q.shape or not np.all(np.isfinite(se)) or np.any(se < 0):
        raise ValueError("q_se must be finite, non-negative and shaped like Q")
    plain = advantage_gap(q, policy)
    coefficients = policy[:, None, :] - np.eye(q.shape[1])[None, :, :]  # [s, b, a]
    differences = np.einsum("sba,sa->sb", coefficients, q)
    spread = z * np.sqrt(np.einsum("sba,sa->sb", coefficients**2, se**2))
    smallest = np.maximum((differences - spread).max(axis=1), 0.0)
    largest = np.maximum((differences + spread).max(axis=1), 0.0)
    return GapBounds(float(start @ smallest), float(largest.max() / (1 - gamma)), plain)


class Certificate(NamedTuple):
    lower: float  # with probability ~ the z-level: lower <= rho . (V^pi - V*)
    upper: float  #                                upper >= V^pi(s) - V*(s) for every s
    observations: int  # validation transitions used (separate from the training budget)
    gap_lower: float  # the rho . g part of ``lower``
    improvement_lower: float  # the rho . (V^pi - V^greedy) part of ``lower``


@dataclass(frozen=True)
class RolloutCertifier:
    """Model-free gap bounds from rollouts of every (state, action), then pi.

    ``n_rollouts`` per pair everywhere (for the upper bound, a max over all states) and
    ``n_start_rollouts`` per pair at the start states (for the lower bound, which only needs Q there
    and is useless unless the noise there is small).
    """

    n_rollouts: int = 20
    n_start_rollouts: int = 2000
    z: float = 3.0

    def __post_init__(self) -> None:
        if self.n_rollouts < 2 or self.n_start_rollouts < 2:
            raise ValueError("rollout counts must be at least 2 (standard errors need two samples)")
        if not (math.isfinite(self.z) and self.z >= 0):
            raise ValueError("z must be non-negative and finite")

    def rollout_q(
        self, env: Environment, policy: FloatArray, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray, int]:
        """(Q-hat, standard error of Q-hat, transitions used) for every (state, action)."""
        check_policy(policy, env.n_states, env.n_actions)
        counts = np.where(env.start_distribution() > 0, self.n_start_rollouts, self.n_rollouts).astype(
            np.int64
        )
        kernel = env.kernel
        mean, variance, steps = _rollout_kernel(
            kernel.step,
            kernel.params,
            policy,
            policy_cdf(policy),
            env.gamma,
            return_window(env.gamma),
            counts,
            rng,
        )
        return mean, np.sqrt(variance / counts[:, None]), int(steps)

    def improvement_lower(
        self, env: Environment, policy: FloatArray, other: FloatArray, rng: np.random.Generator
    ) -> tuple[float, int]:
        """z-level lower confidence bound on rho . (V^policy - V^other) (<= the optimality gap of
        ``policy`` for any ``other``), from ``n_start_rollouts`` rollouts of each policy per start state."""
        check_policy(other, env.n_states, env.n_actions)
        rho, horizon = env.start_distribution(), return_window(env.gamma)
        difference, variance, steps = 0.0, 0.0, 0
        kernel = env.kernel
        for s in np.flatnonzero(rho):
            mean_pi, var_pi, used_pi = _value_kernel(
                kernel.step,
                kernel.params,
                policy,
                policy_cdf(policy),
                int(s),
                env.gamma,
                horizon,
                self.n_start_rollouts,
                rng,
            )
            mean_o, var_o, used_o = _value_kernel(
                kernel.step,
                kernel.params,
                other,
                policy_cdf(other),
                int(s),
                env.gamma,
                horizon,
                self.n_start_rollouts,
                rng,
            )
            difference += rho[s] * (mean_pi - mean_o)
            variance += rho[s] ** 2 * (var_pi + var_o) / self.n_start_rollouts
            steps += used_pi + used_o
        return max(0.0, difference - self.z * math.sqrt(variance)), steps

    def certify(self, env: Environment, policy: FloatArray, rng: np.random.Generator) -> Certificate:
        q, se, steps = self.rollout_q(env, policy, rng)
        bounds = gap_bounds(q, policy, env.gamma, env.start_distribution(), se, self.z)
        greedy = np.eye(env.n_actions)[q.argmin(axis=1)]
        improvement, improvement_steps = self.improvement_lower(env, policy, greedy, rng)
        return Certificate(
            max(bounds.lower, improvement), bounds.upper, steps + improvement_steps, bounds.lower, improvement
        )


@njit
def _rollout_kernel(
    step: StepKernel,
    params: tuple[Any, ...],
    policy: FloatArray,
    cdf: FloatArray,
    gamma: float,
    horizon: int,
    counts: IntArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, int]:
    """Mean and sample variance of the discounted return of (s, a) then pi, per pair (Welford)."""
    n_states, n_actions = policy.shape
    mean = np.zeros((n_states, n_actions))
    squares = np.zeros((n_states, n_actions))
    steps = 0
    variance = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            for r in range(counts[s]):
                state, action, total, discount = s, a, 0.0, 1.0
                for _ in range(horizon):
                    next_state, cost, done = step(params, state, action, rng)
                    steps += 1
                    total += discount * cost
                    if done:
                        break
                    discount *= gamma
                    state = next_state
                    action = sample_action(cdf[state], policy[state], rng)
                delta = total - mean[s, a]
                mean[s, a] += delta / (r + 1)
                squares[s, a] += delta * (total - mean[s, a])
            variance[s, a] = squares[s, a] / (counts[s] - 1)
    return mean, variance, steps


@njit
def _value_kernel(
    step: StepKernel,
    params: tuple[Any, ...],
    policy: FloatArray,
    cdf: FloatArray,
    start: int,
    gamma: float,
    horizon: int,
    n_rollouts: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Mean and sample variance of the discounted return of ``policy`` from ``start``."""
    mean, squares, steps = 0.0, 0.0, 0
    for r in range(n_rollouts):
        state, total, discount = start, 0.0, 1.0
        for _ in range(horizon):
            action = sample_action(cdf[state], policy[state], rng)
            next_state, cost, done = step(params, state, action, rng)
            steps += 1
            total += discount * cost
            if done:
                break
            discount *= gamma
            state = next_state
        delta = total - mean
        mean += delta / (r + 1)
        squares += delta * (total - mean)
    return mean, squares / (n_rollouts - 1), steps
