"""Enhancement 2: online, fixed-policy, cost-based action-value TD (lecture notation).

    Q_theta(s, a) = phi(s, a)^T theta,    delta_t = c_t + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)

    TD(0)                   theta += eta * delta * phi
    TD(lambda)              z = lambda z - delta phi;                       theta -= eta * z
    SARSA(lambda)           e = gamma lambda e + phi;                       theta += eta * delta * e
    True Online TD(lambda)  e = gamma lambda e + (1 - eta gamma lambda e^T phi) phi           (Dutch trace)
                            theta += eta (delta + q - q_old) e - eta (q - q_old) phi
                            (van Seijen & Sutton, ICML 2014)

The next action is sampled from the policy (SARSA-style target); the trace and q_old reset when a
Blackjack hand ends. GridWorld is continuing.

Speed and memory: the features are one-hot (``features.py``), and an accumulating or Dutch trace
only has non-zero entries for features visited in the last ~log(cutoff)/log(gamma lambda) steps.
The kernel keeps those entries in an active list and drops entries below ``trace_cutoff``, so a
step costs O(active features) instead of O(d). With cutoff 0 it is exactly the dense algorithm.
History stores the estimated V(s0) every ``record_every`` steps. Optional ``value_history`` and
``q_history`` buffers retain per-update state values and checkpoint Q snapshots for experiments.
The accumulated-operator method keeps all past error contributions without trace pruning.
"""

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, NamedTuple

import numpy as np
from numba import njit

from ie692_rl.core.estimator import BudgetExhaustedError, Estimate, QEstimator, resolve_extra_cost
from ie692_rl.core.policy import check_policy, policy_cdf, sample_action
from ie692_rl.enhancements.true_online_td.features import FEATURES, FeatureMap
from ie692_rl.envs.base import Environment, ResetKernel, StepKernel
from ie692_rl.types import FloatArray, IntArray

ACCUMULATING, TRUE_ONLINE, ACCUMULATED_OPERATOR = 0, 1, 2
TRACE_CUTOFF = 1e-12


def estimated_state_value(Q: FloatArray, pi: FloatArray, state: int) -> float:
    """Return sum_a pi(a|state) Q(state,a); Q and pi have shape (S, A)."""
    return float(pi[state] @ Q[state])


class TDResult(NamedTuple):
    q: FloatArray
    observations: int
    history_steps: IntArray  # observation counts at which V(s0) was recorded
    history_values: FloatArray  # sum_a pi(a|s0) Q_t(s0, a)
    states_visited: int


def td_zero(
    env: Environment,
    pi: FloatArray,
    features: FeatureMap,
    n_steps: int,
    eta: float,
    rng: np.random.Generator,
    record_every: int = 100,
    *,
    evaluation_state: int | None = None,
    value_history: FloatArray | None = None,
    q_history: list[tuple[int, FloatArray]] | None = None,
) -> TDResult:
    """Stochastic TD(0) (TD(lambda) with lambda = 0)."""
    return run_td(
        env,
        pi,
        features,
        n_steps,
        eta,
        0.0,
        rng,
        ACCUMULATING,
        record_every,
        evaluation_state=evaluation_state,
        value_history=value_history,
        q_history=q_history,
    )


def td_lambda(
    env: Environment,
    pi: FloatArray,
    features: FeatureMap,
    n_steps: int,
    eta: float,
    lam: float,
    rng: np.random.Generator,
    record_every: int = 100,
    *,
    evaluation_state: int | None = None,
    value_history: FloatArray | None = None,
    q_history: list[tuple[int, FloatArray]] | None = None,
) -> TDResult:
    """Accumulate past TD errors times their features, with lambda decay.

    Lecture 19: tilde_F = -delta * phi.
    Lecture 20: z = lam * z + tilde_F; theta = theta - eta * z.
    The operator sum resets at an episode boundary.
    """
    return run_td(
        env,
        pi,
        features,
        n_steps,
        eta,
        lam,
        rng,
        ACCUMULATED_OPERATOR,
        record_every,
        evaluation_state=evaluation_state,
        value_history=value_history,
        q_history=q_history,
    )


def td_lambda_sarsa(
    env: Environment,
    pi: FloatArray,
    features: FeatureMap,
    n_steps: int,
    eta: float,
    lam: float,
    rng: np.random.Generator,
    record_every: int = 100,
    *,
    evaluation_state: int | None = None,
    value_history: FloatArray | None = None,
    q_history: list[tuple[int, FloatArray]] | None = None,
) -> TDResult:
    """SARSA(lambda): apply the current TD error to accumulated features."""
    return run_td(
        env,
        pi,
        features,
        n_steps,
        eta,
        lam,
        rng,
        ACCUMULATING,
        record_every,
        evaluation_state=evaluation_state,
        value_history=value_history,
        q_history=q_history,
    )


def true_online_td_lambda(
    env: Environment,
    pi: FloatArray,
    features: FeatureMap,
    n_steps: int,
    eta: float,
    lam: float,
    rng: np.random.Generator,
    record_every: int = 100,
    *,
    evaluation_state: int | None = None,
    value_history: FloatArray | None = None,
    q_history: list[tuple[int, FloatArray]] | None = None,
) -> TDResult:
    """True Online TD(lambda): Dutch trace plus the true-online correction."""
    return run_td(
        env,
        pi,
        features,
        n_steps,
        eta,
        lam,
        rng,
        TRUE_ONLINE,
        record_every,
        evaluation_state=evaluation_state,
        value_history=value_history,
        q_history=q_history,
    )


def run_td(
    env: Environment,
    pi: FloatArray,
    features: FeatureMap,
    n_steps: int,
    eta: float,
    lam: float,
    rng: np.random.Generator,
    mode: int,
    record_every: int = 100,
    extra_cost: FloatArray | None = None,
    q_init: float = 0.0,
    trace_cutoff: float = TRACE_CUTOFF,
    *,
    evaluation_state: int | None = None,
    value_history: FloatArray | None = None,
    q_history: list[tuple[int, FloatArray]] | None = None,
) -> TDResult:
    """Validate the inputs and run the TD kernel."""
    check_policy(pi, env.n_states, env.n_actions)
    if features.feature_of.shape != (env.n_states, env.n_actions):
        raise ValueError("Feature map does not match the environment")
    if n_steps < 0 or record_every < 0:
        raise ValueError("n_steps and record_every must be non-negative")
    if not (math.isfinite(eta) and eta > 0 and 0 <= lam <= 1 and math.isfinite(q_init) and trace_cutoff >= 0):
        raise ValueError("Need eta > 0, 0 <= lam <= 1, a finite q_init and trace_cutoff >= 0")
    if mode not in (ACCUMULATING, TRUE_ONLINE, ACCUMULATED_OPERATOR):
        raise ValueError("Unknown TD mode")
    if evaluation_state is not None and not 0 <= evaluation_state < env.n_states:
        raise ValueError("Provide a valid evaluation_state")
    if value_history is not None:
        if evaluation_state is None:
            raise ValueError("Provide a valid evaluation_state for value_history")
        if value_history.shape != (n_steps + 1,) or not np.issubdtype(value_history.dtype, np.floating):
            raise ValueError("value_history must be a floating-point array of length n_steps + 1")
        value_history[0] = q_init
    eval_state = env.start if evaluation_state is None else evaluation_state
    per_update_values = np.empty(0) if value_history is None else value_history
    n_snapshots = n_steps // record_every if q_history is not None and record_every > 0 else 0
    parameter_history = np.empty((n_snapshots, features.n_features))
    extra = resolve_extra_cost(env, extra_cost)
    kernel = env.kernel
    theta, steps, values, visited = _td_kernel(
        kernel.step,
        kernel.reset,
        kernel.params,
        pi,
        policy_cdf(pi),
        features.feature_of,
        features.n_features,
        int(n_steps),
        float(eta),
        float(lam),
        mode,
        extra,
        float(q_init),
        env.gamma,
        int(eval_state),
        int(record_every),
        float(trace_cutoff),
        rng,
        per_update_values,
        parameter_history,
    )
    if q_history is not None:
        q_history.extend(
            (int(step), parameters[features.feature_of])
            for step, parameters in zip(steps, parameter_history, strict=True)
        )
    return TDResult(theta[features.feature_of], int(n_steps), steps, values, int(visited))


@dataclass(frozen=True)
class _TDEstimator(QEstimator):
    """Shared adapter: fresh TD parameters and ``n_steps`` on-policy transitions per SPMD call."""

    _mode: ClassVar[int]

    eta: float = 0.05
    lam: float = 0.8
    n_steps: int = 10_000
    features: Literal["tabular", "linear"] = "tabular"
    q_init: float = 0.0
    trace_cutoff: float = TRACE_CUTOFF

    def __post_init__(self) -> None:
        if self.features not in FEATURES:
            raise ValueError(f"features must be one of {sorted(FEATURES)}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")

    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        if max_observations < self.n_steps:
            raise BudgetExhaustedError(f"Need {self.n_steps} observations, {max_observations} left")
        result = run_td(
            env,
            policy,
            FEATURES[self.features](env),
            self.n_steps,
            self.eta,
            self.lam,
            rng,
            self._mode,
            record_every=0,
            extra_cost=extra_cost,
            q_init=self.q_init,
            trace_cutoff=self.trace_cutoff,
        )
        return Estimate(result.q, result.observations, result.states_visited)


@dataclass(frozen=True)
class TrueOnlineTD(_TDEstimator):
    """True Online TD(lambda) with Dutch traces as SPMD's Q estimator."""

    _mode: ClassVar[int] = TRUE_ONLINE


@dataclass(frozen=True)
class TDZero(_TDEstimator):
    """TD(0) as SPMD's Q estimator; lambda is fixed at zero."""

    lam: float = 0.0
    _mode: ClassVar[int] = ACCUMULATING

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lam != 0.0:
            raise ValueError("TDZero requires lam = 0")


@dataclass(frozen=True)
class TDLambda(_TDEstimator):
    """Accumulated TD operators (``td_lambda``), with lambda decay, as SPMD's Q estimator."""

    _mode: ClassVar[int] = ACCUMULATED_OPERATOR


@dataclass(frozen=True)
class SarsaLambda(_TDEstimator):
    """SARSA(lambda) with an accumulating trace (``td_lambda_sarsa``) as SPMD's Q estimator."""

    _mode: ClassVar[int] = ACCUMULATING


@njit
def _td_kernel(
    step: StepKernel,
    reset: ResetKernel,
    params: tuple[Any, ...],
    policy: FloatArray,
    cdf: FloatArray,
    index: IntArray,
    n_features: int,
    n_steps: int,
    eta: float,
    lam: float,
    mode: int,
    extra_cost: FloatArray,
    q_init: float,
    gamma: float,
    eval_state: int,
    record_every: int,
    trace_cutoff: float,
    rng: np.random.Generator,
    per_update_values: FloatArray,
    parameter_history: FloatArray,
) -> tuple[FloatArray, IntArray, FloatArray, int]:
    n_states, n_actions = policy.shape
    theta = np.full(n_features, q_init)
    trace = np.zeros(n_features)
    active = np.empty(n_features, dtype=np.int64)  # features with a non-zero trace
    is_active = np.zeros(n_features, dtype=np.bool_)
    n_active = 0
    visited = np.zeros(n_states, dtype=np.bool_)
    n_records = n_steps // record_every if record_every > 0 else 0
    history_steps = np.empty(n_records, dtype=np.int64)
    history_values = np.empty(n_records)
    decay = lam if mode == ACCUMULATED_OPERATOR else gamma * lam
    q_old = 0.0

    state = reset(params, rng)
    action = sample_action(cdf[state], policy[state], rng)
    for t in range(n_steps):
        next_state, cost, done = step(params, state, action, rng)
        visited[state] = True
        cost += extra_cost[state]
        feature = index[state, action]
        next_action = -1
        q_next = 0.0
        if not done:
            next_action = sample_action(cdf[next_state], policy[next_state], rng)
            q_next = theta[index[next_state, next_action]]
        q = theta[feature]
        delta = cost + gamma * q_next - q

        # Trace: decay every active entry, drop negligible ones, then add the current feature.
        trace_at_feature = trace[feature]
        i = 0
        while i < n_active:
            f = active[i]
            trace[f] *= decay
            if mode != ACCUMULATED_OPERATOR and abs(trace[f]) <= trace_cutoff and f != feature:
                trace[f] = 0.0
                is_active[f] = False
                n_active -= 1
                active[i] = active[n_active]
            else:
                i += 1
        if not is_active[feature]:
            is_active[feature] = True
            active[n_active] = feature
            n_active += 1
        if mode == TRUE_ONLINE:
            trace[feature] += 1.0 - eta * decay * trace_at_feature
            scale, correction = eta * (delta + q - q_old), eta * (q - q_old)
            q_old = q_next
        elif mode == ACCUMULATED_OPERATOR:
            # z = lam * z + tilde_F, where tilde_F = -delta * phi.
            trace[feature] -= delta
            scale, correction = -eta, 0.0
        else:
            trace[feature] += 1.0
            scale, correction = eta * delta, 0.0
        for i in range(n_active):
            f = active[i]
            update = scale * trace[f]
            if f == feature:
                update -= correction
            theta[f] += update

        checkpoint = record_every > 0 and (t + 1) % record_every == 0
        if per_update_values.size > 0 or checkpoint:
            value = 0.0
            for a in range(n_actions):
                value += policy[eval_state, a] * theta[index[eval_state, a]]
            if per_update_values.size > 0:
                per_update_values[t + 1] = value
            if checkpoint:
                snapshot_index = (t + 1) // record_every - 1
                history_steps[snapshot_index] = t + 1
                history_values[snapshot_index] = value
                if parameter_history.shape[0] > 0:
                    parameter_history[snapshot_index] = theta
        if done:  # new Blackjack hand = new episode
            for i in range(n_active):
                trace[active[i]] = 0.0
                is_active[active[i]] = False
            n_active = 0
            q_old = 0.0
            state = reset(params, rng)
            action = sample_action(cdf[state], policy[state], rng)
        else:
            state, action = next_state, next_action

    n_visited = 0
    for s in range(n_states):
        n_visited += visited[s]
    return theta, history_steps, history_values, n_visited
