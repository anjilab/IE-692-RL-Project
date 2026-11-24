"""Tabular stochastic policies stored as pi[state, action]."""

import numpy as np
from numba import njit

from ie692_rl.types import FloatArray


def uniform_policy(n_states: int, n_actions: int) -> FloatArray:
    return np.full((n_states, n_actions), 1.0 / n_actions)


def check_policy(policy: FloatArray, n_states: int, n_actions: int) -> None:
    """Raise ValueError unless every row is a probability vector of the right shape."""
    if policy.shape != (n_states, n_actions):
        raise ValueError(f"Policy shape must be ({n_states}, {n_actions}), got {policy.shape}")
    if not np.all(np.isfinite(policy)) or np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1.0):
        raise ValueError("Each policy row must contain action probabilities summing to 1")


def policy_cdf(policy: FloatArray) -> FloatArray:
    """Row-wise cumulative probabilities used by ``sample_action``."""
    cdf: FloatArray = np.cumsum(policy, axis=1)
    return cdf


@njit(cache=True)
def sample_action(cdf_row: FloatArray, prob_row: FloatArray, rng: np.random.Generator) -> int:
    """Draw an action from one policy row with a single uniform number.

    Actions with probability 0 are never returned. If rounding leaves the row's total slightly
    below the uniform draw, the last action with positive probability is used.
    """
    u = rng.random()
    last = 0
    for a in range(cdf_row.shape[0]):
        if prob_row[a] > 0.0:
            if u < cdf_row[a]:
                return a
            last = a
    return last
