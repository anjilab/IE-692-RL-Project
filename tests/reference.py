"""Straightforward pure-Python versions of the team's original algorithms, used as test oracles.

They follow the original code line by line (full trajectory for TOMC, dense Phi and dense traces for
TD) but draw actions with the package's ``sample_action`` and step with ``env.step_one``, which use
the random stream in the same order as the numba kernels. The fast kernels must therefore reproduce
these results to rounding error.
"""

from collections.abc import Callable

import numpy as np

from ie692_rl.core.policy import policy_cdf, sample_action
from ie692_rl.envs.base import Environment
from ie692_rl.types import FloatArray


def reference_tomc(
    env: Environment,
    pi: FloatArray,
    rng: np.random.Generator,
    bias: float,
    max_observations: int,
    extra_cost: FloatArray | None = None,
) -> tuple[FloatArray, int]:
    """The original ``baseline.tomc``: store the stream, then compute returns backwards."""
    gamma = env.gamma
    extra = np.zeros(env.n_states) if extra_cost is None else extra_cost
    pi_min = (1 - gamma) / env.n_actions
    required = pi >= pi_min
    cdf = policy_cdf(pi)
    first = np.full(pi.shape, -1, dtype=int)
    Q = np.full(pi.shape, (env.cost_bound + max(0.0, float(extra.max()))) / (1 - gamma))
    trajectory = []
    state = env.reset(rng)
    n = 0
    for t in range(max_observations):
        action = int(sample_action(cdf[state], pi[state], rng))
        next_state, cost, done = env.step_one(state, action, rng)
        trajectory.append((state, action, cost + extra[state], done))
        if required[state, action] and first[state, action] == -1:
            first[state, action] = t
        n = t + 1
        if np.all(first[required] >= 0) and gamma ** (n - first[required].max()) <= bias * (1 - gamma):
            break
        state = env.reset(rng) if done else next_state
    else:
        raise RuntimeError("incomplete")
    G = 0.0
    for t in range(n - 1, -1, -1):
        s, a, cost, done = trajectory[t]
        G = cost + (0.0 if done else gamma * G)
        if required[s, a] and first[s, a] == t:
            Q[s, a] = G
    return Q, n


def reference_td(
    env: Environment,
    pi: FloatArray,
    Phi: FloatArray,
    n_steps: int,
    eta: float,
    lam: float,
    rng: np.random.Generator,
    mode: str,
) -> FloatArray:
    """The original td.py loops (mode: 'td0', 'td_lambda' or 'true_online') with dense features."""
    S, A = env.n_states, env.n_actions
    cdf = policy_cdf(pi)
    theta = np.zeros(Phi.shape[1])
    e = np.zeros_like(theta)
    q_old = 0.0

    def start() -> tuple[int, int]:
        s = env.reset(rng)
        return s, int(sample_action(cdf[s], pi[s], rng))

    s, a = start()
    for _ in range(n_steps):
        next_s, cost, done = env.step_one(s, a, rng)
        phi = Phi[s * A + a]
        if done:
            phi_next, next_a = np.zeros_like(phi), -1
        else:
            next_a = int(sample_action(cdf[next_s], pi[next_s], rng))
            phi_next = Phi[next_s * A + next_a]
        if mode == "td0":
            delta = cost + env.gamma * (phi_next @ theta) - (phi @ theta)
            theta += eta * delta * phi
        elif mode == "td_lambda":
            delta = cost + env.gamma * (phi_next @ theta) - (phi @ theta)
            e = env.gamma * lam * e + phi
            theta += eta * delta * e
        else:
            q, q_next = phi @ theta, phi_next @ theta
            delta = cost + env.gamma * q_next - q
            e = env.gamma * lam * e + (1 - eta * env.gamma * lam * (e @ phi)) * phi
            theta += eta * (delta + q - q_old) * e - eta * (q - q_old) * phi
            q_old = q_next
        if done:
            e.fill(0)
            q_old = 0.0
            s, a = start()
        else:
            s, a = next_s, next_a
    result: FloatArray = (Phi @ theta).reshape(S, A)
    return result


def regularized_q(P: FloatArray, c: FloatArray, gamma: float, policy: FloatArray, tau: float) -> FloatArray:
    """Exact entropy-regularized Q of ``policy`` from a dense model (small problems only)."""
    S = c.shape[0]
    log_pi = np.log(np.where(policy > 0, policy, 1.0))
    c_pi = (policy * c).sum(axis=1) + tau * (policy * log_pi).sum(axis=1)
    P_pi = np.einsum("sa,sat->st", policy, P)
    v = np.linalg.solve(np.eye(S) - gamma * P_pi, c_pi)
    q: FloatArray = c + gamma * P @ v
    return q


def reference_tsallis_step(policy: FloatArray, q: FloatArray, eta: float, index: float) -> FloatArray:
    """Port of pg-termination's ``tsallis_policy_update`` (jucaleb4/pg-termination, utils.py#L45).

    Changes: the Tsallis index is passed directly (the original derives it from gamma as
    1 / (1 - log(1 - gamma) / log 2)) and arrays are [state, action]. Two bugs of the original are
    fixed, otherwise it is not a usable oracle:
      * the early exit in ``tsallis_update_lambda`` returns the policy row instead of the multiplier;
        here it returns 0;
      * the bisection for a negative multiplier swaps ``lo`` and ``hi``: when sum(x) < 1 the root lies
        to the left, but the original moves ``lo`` right, so it converges to the far end of the
        bracket, outside the domain, where the 1e-6 clamp yields huge entries (for example the step
        [0.39, 0.59, 0.01, 0.001] -> [0.5, 0.5, 0, 0], with a worse objective than not moving).
    """
    tol, p = 1e-6, index
    new = np.empty_like(policy)
    for s in range(policy.shape[0]):
        g = q[s] + np.power(policy[s], p - 1) / ((1.0 - p) * eta)

        def get_x_star(y: float, g: FloatArray = g) -> FloatArray:
            result: FloatArray = np.power(np.maximum(tol, (1.0 - p) * eta * (g + y)), 1.0 / (p - 1))
            return result

        lam = _reference_tsallis_lambda(get_x_star, tol)
        new[s] = get_x_star(lam)
        new[s] /= new[s].sum()
    return new


def _reference_tsallis_lambda(get_x_star: Callable[[float], FloatArray], tol: float) -> float:
    u = get_x_star(0)
    if abs(np.sum(u) - 1) <= tol:
        return 0.0
    direction = 1 if np.sum(u) > 1 else -1
    lam, no_exp_search = float(direction), True
    u = get_x_star(lam)
    while (direction == 1 and np.sum(u) > 1) or (direction == -1 and np.sum(u) < 1):  # exponential search
        no_exp_search = False
        lam *= 2
        u = get_x_star(lam)
    if direction == 1:  # binary search
        lo, hi = (0.0 if no_exp_search else lam / 2.0), lam
        while hi - lo > tol:
            lam = (hi + lo) / 2
            u = get_x_star(lam)
            if abs(np.sum(u) - 1) <= tol:
                break
            if np.sum(u) < 1:
                hi = lam
            else:
                lo = lam
        return hi
    lo, hi = lam, (0.0 if no_exp_search else lam / 2.0)
    while hi - lo > tol:
        lam = (hi + lo) / 2
        u = get_x_star(lam)
        if abs(np.sum(u) - 1) <= tol:
            break
        if np.sum(u) < 1:  # the original sets lo here (and hi below): the root is to the left, see docstring
            hi = lam
        else:
            lo = lam
    return lo
