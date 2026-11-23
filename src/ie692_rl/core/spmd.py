"""Stochastic Policy Mirror Descent (SPMD) with injected policy evaluation and Bregman divergence.

    pi_0 = uniform
    for k = 0, 1, ...:   Q_k   = estimator.estimate(env, pi_k)                  (online samples)
                         pi_{k+1} = divergence.step(pi_k, Q_k, eta_k, tau_k)

The baseline is ``run_spmd(env, TOMC(bias), KL(), step_size, ...)`` (KL from enhancements.bregman_spmd).
The enhancements swap one part:
    Bregman divergences      divergence = Euclidean() / Tsallis(index)   (enhancements.bregman_spmd)
    True Online TD(lambda)   estimator = TrueOnlineTD(...)                  (enhancements.true_online_td)
    Entropy-regularized      tau > 0 (fixed or annealed); every sampled cost at state s then includes
                             tau * sum_a pi(a|s) log pi(a|s), so the estimator returns the regularized Q

The run stops when the observation budget is spent. Every iteration is recorded with the two metrics
the project guideline asks for (the estimated value at the evaluation state and the number of
observations) plus diagnostics: the exact value and gap to optimal, the smallest action probability
and the number of distinct states visited.
"""

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

from ie692_rl.core.divergence import BregmanDivergence, negative_entropy
from ie692_rl.core.estimator import BudgetExhaustedError, QEstimator
from ie692_rl.core.exact import ExactScorer
from ie692_rl.core.optimality import RolloutCertifier
from ie692_rl.core.policy import uniform_policy
from ie692_rl.envs.base import Environment
from ie692_rl.types import FloatArray

Schedule = Callable[[int], float]
Status = Literal["start", "updated", "budget", "diverged"]


def as_schedule(value: float | Schedule) -> Schedule:
    """A constant becomes a constant schedule; callables are returned unchanged."""
    if callable(value):
        return value
    constant = float(value)
    return lambda _k: constant


@dataclass(frozen=True)
class IterationRecord:
    """One SPMD iteration.

    Row 0 is the uniform starting policy. Row k >= 1 holds the estimate made for pi_{k-1}
    (``estimated_value`` = sum_a pi_{k-1}(a|s0) Q_hat(s0, a), with the entropy cost when tau > 0)
    and the exact score of the updated policy pi_k (``exact_value``, ``gap``). ``gap_lower`` and
    ``gap_upper`` are the model-free advantage-gap bounds for pi_k (NaN where it was not certified),
    computed from ``validation_observations`` extra transitions outside the training budget.
    """

    iteration: int
    observations: int
    cumulative_observations: int
    estimated_value: float
    exact_value: float
    gap: float
    min_prob: float
    states_visited: int
    step_size: float
    tau: float
    status: Status
    gap_lower: float = math.nan
    gap_upper: float = math.nan
    validation_observations: int = 0

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class SPMDResult:
    policy: FloatArray
    history: list[IterationRecord]


def run_spmd(
    env: Environment,
    estimator: QEstimator,
    divergence: BregmanDivergence,
    step_size: float | Schedule,
    budget: int,
    seed: int,
    tau: float | Schedule = 0.0,
    scorer: ExactScorer | None = None,
    max_iterations: int | None = None,
    certifier: RolloutCertifier | None = None,
    certify_every: int = 0,
) -> SPMDResult:
    """Online SPMD from the uniform policy until ``budget`` observations are used.

    If the remaining budget is too small for a complete estimate, those steps give no update and
    the run ends (status ``budget``). A non-finite Q estimate (for example a TD step size that is
    too large) ends the run with status ``diverged`` and keeps the last valid policy.

    With a ``certifier``, the final policy (and, if ``certify_every`` > 0, every certify_every-th
    updated policy) gets model-free optimality-gap bounds. Validation uses its own random stream and
    is not charged to ``budget``, so the training run is identical with or without it.
    """
    if budget < 0 or certify_every < 0:
        raise ValueError("budget and certify_every must be non-negative")
    eta_of, tau_of = as_schedule(step_size), as_schedule(tau)
    rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng((seed, 0x5EED))
    policy = uniform_policy(env.n_states, env.n_actions)
    exact = scorer.score(policy) if scorer else math.nan
    optimal = scorer.optimal if scorer else math.nan
    history = [
        IterationRecord(
            0, 0, 0, math.nan, exact, exact - optimal, float(policy.min()), 0, math.nan, math.nan, "start"
        )
    ]
    used, k = 0, 0
    while used < budget and (max_iterations is None or k < max_iterations):
        eta_k, tau_k = float(eta_of(k)), float(tau_of(k))
        if not (math.isfinite(eta_k) and eta_k > 0):
            raise ValueError(f"Step size must be positive and finite, got {eta_k} at iteration {k}")
        extra = tau_k * negative_entropy(policy) if tau_k > 0 else None
        try:
            q, n, visited = estimator.estimate(env, policy, rng, budget - used, extra)
        except BudgetExhaustedError:
            history.append(_final_record(history[-1], k + 1, budget, "budget"))
            break
        used += n
        if not np.all(np.isfinite(q)):
            history.append(_final_record(history[-1], k + 1, used, "diverged"))
            break
        estimated = float(policy[env.start] @ q[env.start])
        policy = divergence.step(policy, q, eta_k, tau_k)
        exact = scorer.score(policy) if scorer else math.nan
        k += 1
        history.append(
            IterationRecord(
                k,
                n,
                used,
                estimated,
                exact,
                exact - optimal,
                float(policy.min()),
                visited,
                eta_k,
                tau_k,
                "updated",
            )
        )
        if certifier and certify_every and k % certify_every == 0:
            history[-1] = _certified(history[-1], certifier, env, policy, validation_rng)
    if certifier and math.isnan(history[-1].gap_lower):
        history[-1] = _certified(history[-1], certifier, env, policy, validation_rng)
    return SPMDResult(policy, history)


def _certified(
    record: IterationRecord,
    certifier: RolloutCertifier,
    env: Environment,
    policy: FloatArray,
    rng: np.random.Generator,
) -> IterationRecord:
    certificate = certifier.certify(env, policy, rng)
    return replace(
        record,
        gap_lower=certificate.lower,
        gap_upper=certificate.upper,
        validation_observations=certificate.observations,
    )


def _final_record(last: IterationRecord, iteration: int, used: int, status: Status) -> IterationRecord:
    """Closing row when no update was made: observations are spent, the policy is unchanged."""
    return IterationRecord(
        iteration,
        used - last.cumulative_observations,
        used,
        math.nan,
        last.exact_value,
        last.gap,
        last.min_prob,
        0,
        math.nan,
        math.nan,
        status,
    )
