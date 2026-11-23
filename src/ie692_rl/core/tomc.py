"""Baseline policy evaluation: Truncated On-Policy Monte Carlo (TOMC) with dynamic exploration time.

One estimate follows the current policy in a single on-policy stream (a new Blackjack hand is dealt
after each finished one) until

  1. every *required* pair, pi(a|s) >= pi_min (default (1 - gamma) / |A|), has been visited, and
  2. the discount factor of the tail after the most recent first visit, gamma^(n - t_last), is at
     most ``bias * (1 - gamma)``, so truncating the stream changes no first-visit return by more
     than ``bias`` times the cost bound.

Q(s, a) is the discounted return after the first visit to (s, a) (it stops at the end of a
Blackjack hand). Pairs that are not required get the pessimistic value ``cost_bound / (1 - gamma)``.

Memory: the reference implementation stored the whole stream, which for long streams (30x30
GridWorld needs tens of millions of steps) costs gigabytes. Here each first-visit return is
computed from a ring buffer of the last ``W`` costs, where ``gamma^W <= 1e-16``, so later costs
cannot change the return beyond double precision. Memory is O(|S||A| + W) regardless of stream
length, and the results equal the full backward recursion up to that 1e-16 relative error.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ie692_rl._jit import njit, njit_generic
from ie692_rl.core.estimator import BudgetExhaustedError, Estimate, QEstimator, resolve_extra_cost
from ie692_rl.core.policy import check_policy, policy_cdf, sample_action
from ie692_rl.envs.base import Environment, ResetKernel, StepKernel
from ie692_rl.types import BoolArray, FloatArray

RETURN_PRECISION = 1e-16


def tail_steps(gamma: float, bias: float) -> int:
    """Smallest m with gamma**m <= bias * (1 - gamma): steps needed after the last first visit."""
    target = bias * (1 - gamma)
    if not (0 < gamma < 1 and target > 0):
        raise ValueError("Need 0 < gamma < 1 and bias * (1 - gamma) > 0")
    m = max(0, math.floor(math.log(target) / math.log(gamma)) - 1)
    while gamma**m > target:
        m += 1
    return m


def return_window(gamma: float) -> int:
    """Ring-buffer length W with gamma**W <= RETURN_PRECISION."""
    return tail_steps(gamma, RETURN_PRECISION / (1 - gamma)) + 1


@dataclass(frozen=True)
class TOMC(QEstimator):
    """TOMC with dynamic exploration time (the project baseline)."""

    bias: float = 0.01
    pi_min: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.bias) and self.bias > 0):
            raise ValueError("bias must be positive and finite")
        if self.pi_min is not None and not 0 < self.pi_min <= 1:
            raise ValueError("pi_min must be in (0, 1]")

    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        check_policy(policy, env.n_states, env.n_actions)
        if max_observations < 1:
            raise BudgetExhaustedError("No observations left in the budget")
        pi_min = (1 - env.gamma) / env.n_actions if self.pi_min is None else self.pi_min
        required: BoolArray = policy >= pi_min
        if not required.any():
            raise ValueError("No state-action pairs meet pi_min")
        extra = resolve_extra_cost(env, extra_cost)
        fill = (env.cost_bound + max(0.0, float(extra.max()))) / (1 - env.gamma)
        kernel = env.kernel
        q, n, covered, finished, visited = _tomc_kernel(
            kernel.step,
            kernel.reset,
            kernel.params,
            policy,
            policy_cdf(policy),
            required,
            extra,
            env.gamma,
            tail_steps(env.gamma, self.bias),
            return_window(env.gamma),
            int(max_observations),
            fill,
            rng,
        )
        if not finished:
            raise BudgetExhaustedError(
                f"TOMC incomplete: {covered}/{int(required.sum())} pairs after {n} observations"
            )
        return Estimate(q, int(n), int(visited))


@njit
def _window_return(
    costs: FloatArray, dones: BoolArray, first: int, last: int, window: int, gamma: float
) -> float:
    """Discounted return from step ``first`` to ``last`` (inclusive), stopping at an episode end."""
    total = 0.0
    for t in range(last, first - 1, -1):
        slot = t % window
        total = costs[slot] + (0.0 if dones[slot] else gamma * total)
    return total


@njit_generic
def _tomc_kernel(
    step: StepKernel,
    reset: ResetKernel,
    params: tuple[Any, ...],
    policy: FloatArray,
    cdf: FloatArray,
    required: BoolArray,
    extra_cost: FloatArray,
    gamma: float,
    tail: int,
    window: int,
    max_observations: int,
    fill: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, int, int, bool, int]:
    n_states, n_actions = policy.shape
    q = np.full((n_states, n_actions), fill)
    seen = np.zeros((n_states, n_actions), dtype=np.bool_)
    visited = np.zeros(n_states, dtype=np.bool_)
    n_required = 0
    for s in range(n_states):
        for a in range(n_actions):
            n_required += required[s, a]
    # First visits waiting for their return; times increase, so this is a FIFO queue.
    pending_state = np.empty(n_required, dtype=np.int64)
    pending_action = np.empty(n_required, dtype=np.int64)
    pending_time = np.empty(n_required, dtype=np.int64)
    head = tail_index = 0
    costs = np.zeros(window)
    dones = np.zeros(window, dtype=np.bool_)

    state = reset(params, rng)
    last_cover, n, finished = -1, 0, False
    for t in range(max_observations):
        action = sample_action(cdf[state], policy[state], rng)
        next_state, cost, done = step(params, state, action, rng)
        visited[state] = True
        costs[t % window] = cost + extra_cost[state]
        dones[t % window] = done
        if required[state, action] and not seen[state, action]:
            seen[state, action] = True
            pending_state[tail_index], pending_action[tail_index], pending_time[tail_index] = state, action, t
            tail_index += 1
            last_cover = t
        # A return is final once its window is full: older costs would leave the ring buffer.
        while head < tail_index and pending_time[head] + window - 1 <= t:
            q[pending_state[head], pending_action[head]] = _window_return(
                costs, dones, pending_time[head], t, window, gamma
            )
            head += 1
        n = t + 1
        if tail_index == n_required and n - last_cover >= tail:
            finished = True
            break
        state = reset(params, rng) if done else next_state

    while head < tail_index:  # returns still in the buffer are truncated at the end of the stream
        q[pending_state[head], pending_action[head]] = _window_return(
            costs, dones, pending_time[head], n - 1, window, gamma
        )
        head += 1
    n_visited = 0
    for s in range(n_states):
        n_visited += visited[s]
    return q, n, tail_index, finished, n_visited
