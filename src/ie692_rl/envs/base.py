"""Common interface for the project's finite MDPs.

Each environment is an online simulator: algorithms only ever see sampled
``(next_state, cost, done)`` transitions. To make sampling fast, every environment exposes its rules
as a numba ``Kernel``:

    step(params, state, action, rng) -> (next_state, cost, done)
    reset(params, rng) -> start state of a new stream / episode

``params`` is a tuple of scalars and arrays describing one environment instance. Estimators are
compiled once per (step, reset) pair and receive ``params`` at run time. The Python methods
``step_one`` and ``step`` call the same kernel, so there is a single source of truth for the rules.

For exact (noise-free) scoring each environment also gives its transition model in sparse form
(``TabularModel``): at most ``K`` successors per (state, action) instead of a dense S x A x S
array, so a 30x30 GridWorld needs about 0.3 MB rather than 26 MB.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numba import njit

from ie692_rl.types import BoolArray, FloatArray, IntArray

StepKernel = Callable[[tuple[Any, ...], int, int, np.random.Generator], tuple[int, float, bool]]
ResetKernel = Callable[[tuple[Any, ...], np.random.Generator], int]


class Kernel(NamedTuple):
    """numba-compiled environment rules plus the instance parameters they read."""

    step: StepKernel
    reset: ResetKernel
    params: tuple[Any, ...]


class TabularModel(NamedTuple):
    """Sparse exact model.

    ``probs[s, a, k]`` is the probability of moving to ``next_states[s, a, k]``; unused slots have
    probability 0. Rows may sum to less than 1 where an episode can end (Blackjack). ``costs[s, a]``
    is the expected one-step cost.
    """

    next_states: IntArray
    probs: FloatArray
    costs: FloatArray

    def dense_transitions(self) -> FloatArray:
        """Dense P[s, a, s'] (only for small problems and tests)."""
        S, A, _ = self.probs.shape
        P = np.zeros((S, A, S))
        s_idx, a_idx = np.meshgrid(np.arange(S), np.arange(A), indexing="ij")
        for k in range(self.probs.shape[2]):
            np.add.at(P, (s_idx, a_idx, self.next_states[:, :, k]), self.probs[:, :, k])
        return P


class Environment(ABC):
    """Finite, discounted, cost-minimising MDP with an online simulator."""

    n_states: int
    n_actions: int
    gamma: float
    start: int  # evaluation state s0 used for V(s0) and exact scoring
    cost_bound: float  # largest possible one-step cost; TOMC's pessimistic fill is cost_bound / (1 - gamma)
    action_names: tuple[str, ...] | None = None

    @property
    @abstractmethod
    def kernel(self) -> Kernel:
        """numba step/reset functions and their parameters."""

    @abstractmethod
    def model(self) -> TabularModel:
        """Exact sparse transition model (for scoring and tests only; never used by learners)."""

    @abstractmethod
    def state_label(self, state: int) -> dict[str, int]:
        """Human-readable description of a state index."""

    # ------------------------------------------------------------------ shared helpers

    def check_state_action(self, states: IntArray, actions: IntArray) -> None:
        if np.any((states < 0) | (states >= self.n_states)) or np.any(
            (actions < 0) | (actions >= self.n_actions)
        ):
            raise ValueError("Invalid state or action")

    def step_one(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, float, bool]:
        """One online transition: (next_state, cost, done)."""
        self.check_state_action(np.array([state], dtype=np.int64), np.array([action], dtype=np.int64))
        kernel = self.kernel
        next_state, cost, done = kernel.step(kernel.params, int(state), int(action), rng)
        return int(next_state), float(cost), bool(done)

    def step(
        self, states: IntArray, actions: IntArray, rng: np.random.Generator
    ) -> tuple[IntArray, FloatArray, BoolArray]:
        """Batched transitions (applied in order, so the result equals repeated ``step_one`` calls)."""
        s, a = np.broadcast_arrays(np.asarray(states, dtype=np.int64), np.asarray(actions, dtype=np.int64))
        self.check_state_action(s, a)
        kernel = self.kernel
        next_states, costs, done = step_many(
            kernel.step, kernel.params, s.ravel().copy(), a.ravel().copy(), rng
        )
        return next_states.reshape(s.shape), costs.reshape(s.shape), done.reshape(s.shape)

    def reset(self, rng: np.random.Generator) -> int:
        """Start state of a fresh stream (GridWorld: the origin; Blackjack: a new deal)."""
        kernel = self.kernel
        return int(kernel.reset(kernel.params, rng))

    def initial_states(self, count: int, rng: np.random.Generator) -> IntArray:
        """``count`` copies of the evaluation state (kept for the notebooks)."""
        return np.full(count, self.start, dtype=np.int64)

    def start_distribution(self) -> FloatArray:
        """Point mass on the evaluation state."""
        weights = np.zeros(self.n_states)
        weights[self.start] = 1.0
        return weights


@njit
def step_many(
    step: StepKernel,
    params: tuple[Any, ...],
    states: IntArray,
    actions: IntArray,
    rng: np.random.Generator,
) -> tuple[IntArray, FloatArray, BoolArray]:
    n = states.shape[0]
    next_states = np.empty(n, dtype=np.int64)
    costs = np.empty(n, dtype=np.float64)
    done = np.empty(n, dtype=np.bool_)
    for i in range(n):
        next_states[i], costs[i], done[i] = step(params, states[i], actions[i], rng)
    return next_states, costs, done
