"""Enhancement 3: entropy-regularized SPMD.

The SPMD subproblem gains tau * h(p), h(p) = sum_a p_a log p_a (proposal Section 2.3):

    pi_{k+1}(.|s) = argmin_p  eta * ( <Q(s, .), p> + tau * h(p) ) + KL(p || pi_k(.|s))
                  proportional to pi_k^(1/(1+eta tau)) * exp(-eta Q / (1 + eta tau)).

Where the pieces live:
    closed-form step     ``bregman_spmd.KL.step(policy, q, eta, tau)`` (tau = 0 is the baseline)
    regularized Q        ``run_spmd(..., tau=...)`` adds tau * h(pi(.|s)) to every sampled cost at s
    annealed tau         ``AnnealedTau`` below, tau_k = tau_0 / (k + 1)^power
    regularized optimum  ``soft_optimal_policy`` below, the policy the method converges to
"""

import math
from dataclasses import dataclass

import numpy as np

from ie692_rl.core.exact import MAX_SWEEPS, ExactScorer, backup
from ie692_rl.types import FloatArray


@dataclass(frozen=True)
class AnnealedTau:
    """tau_k = tau_0 / (k + 1)^power: early exploration without the long-run bias of a fixed tau."""

    tau0: float
    power: float = 1.0

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.tau0) and self.tau0 >= 0 and math.isfinite(self.power) and self.power >= 0
        ):
            raise ValueError("tau0 and power must be non-negative and finite")

    def __call__(self, k: int) -> float:
        if k < 0:
            raise ValueError("iteration index must be non-negative")
        return float(self.tau0 / (k + 1) ** self.power)


def soft_optimal_policy(scorer: ExactScorer, tau: float) -> FloatArray:
    """Optimal policy of the entropy-regularized problem (soft value iteration on the exact model).

    V(s) = min_p <Q(s, .), p> + tau h(p) = -tau log sum_a exp(-Q(s, a) / tau), and the optimal policy
    is softmax(-Q / tau). Computed with the row minimum subtracted so it never overflows.
    """
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError("tau must be positive and finite")
    m, gamma = scorer.model, scorer.gamma
    v = np.zeros(scorer.n_states)
    for _ in range(MAX_SWEEPS):
        q = backup(m.next_states, m.probs, m.costs, v, gamma)
        low = q.min(axis=1)
        new = low - tau * np.log(np.exp(-(q - low[:, None]) / tau).sum(axis=1))
        change = float(np.abs(new - v).max())
        v = new
        if change <= scorer.tol * (1 - gamma) / gamma:
            break
    q = backup(m.next_states, m.probs, m.costs, v, gamma)
    weights = np.exp(-(q - q.min(axis=1, keepdims=True)) / tau)
    policy: FloatArray = weights / weights.sum(axis=1, keepdims=True)
    return policy
