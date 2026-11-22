""" Baseline components integration:

1. Implement a Truncated On-Policy Monte Carlo (TOMC) with dynamic exploration time for
Policy Evaluation. More details about tuning given in the Details section;
2. Implement a tuned Stochastic Policy Mirror Descent (SPMD) for Policy Optimization;

Policies and Q estimates are [state, action]; costs are minimized.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def new_start(env, rng):
    """GridWorld starts at zero; Blackjack begins with a fresh deal."""
    return int(env.deal(1, rng)[0]) if hasattr(env, "deal") else env.start


def _policy_check(env, pi):
    if pi.shape != (env.n_states, env.n_actions):
        raise ValueError("Policy shape must be (n_states, n_actions)")
    if not np.all(np.isfinite(pi)) or np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1):
        raise ValueError("Each policy row must contain action probabilities summing to 1")


def tomc(env, pi, rng, bias=0.01, pi_min=None,
         max_observations=200_000, log_transition=None):
    """First-visit TOMC with dynamic exploration time (continuous on-policy data).

    Each required (state, action) is visited once, then sampling continues until
    the discount tail after the latest first visit is small enough. Blackjack
    returns stop at the end of each hand; the next hand is a fresh deal.

    Returns (Q, number_of_observations). Costs must be at most 1, as in both
    project environments; unrequired actions receive the upper bound 1/(1-gamma).
    """
    _policy_check(env, pi)
    gamma = env.gamma
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    if pi_min is None:
        pi_min = (1 - gamma) / env.n_actions
    if not 0 < pi_min <= 1 or bias <= 0 or max_observations < 1:
        raise ValueError("Invalid threshold, bias, or observation limit")

    required = pi >= pi_min
    if not required.any():
        raise ValueError("No state-action pairs meet pi_min")

    first = np.full(pi.shape, -1, dtype=int)
    Q = np.full(pi.shape, 1 / (1 - gamma), dtype=float)
    trajectory = []
    state = new_start(env, rng)

    for t in range(max_observations):
        action = int(rng.choice(env.n_actions, p=pi[state]))
        next_state, cost, done = env.step_one(state, action, rng)
        trajectory.append((state, action, float(cost), bool(done)))
        if log_transition is not None:
            log_transition(state, action, cost, next_state, done)

        if required[state, action] and first[state, action] == -1:
            first[state, action] = t

        n = t + 1
        if np.all(first[required] >= 0):
            longest_tail_error = gamma ** (n - first[required].max())
            if longest_tail_error <= bias * (1 - gamma):
                break

        state = new_start(env, rng) if done else next_state
    else:
        covered = int((first[required] >= 0).sum())
        raise RuntimeError(
            f"TOMC incomplete: {covered}/{int(required.sum())} pairs "
            f"after {max_observations} observations"
        )

    # Backward discounted returns, restarting at episode boundaries.
    G = 0.0
    for t in range(n - 1, -1, -1):
        s, a, cost, done = trajectory[t]
        G = cost + (0.0 if done else gamma * G)
        if required[s, a] and first[s, a] == t:
            Q[s, a] = G

    return Q, n


def spmd(env, estimate_Q, eta_func, s_0, n_iters=100, seed=0):
    """KL-SPMD from the lecture, with policy stored as pi_t[state, action].

    estimate_Q(env, pi_t, rng) returns (Q_t, number of observations).
    V_s_0_arr records estimated costs before each policy update.
    """
    if not 0 <= s_0 < env.n_states:
        raise ValueError("Invalid evaluation state")

    rng = np.random.default_rng(seed)
    pi_t = np.ones((env.n_states, env.n_actions), dtype=float) / env.n_actions
    V_s_0_arr = np.zeros(n_iters, dtype=float)
    sample_arr = np.zeros(n_iters + 1, dtype=int)

    for t in range(n_iters):
        Q_t, n_samples = estimate_Q(env, pi_t, rng)
        Q_t = np.asarray(Q_t, dtype=float)
        if Q_t.shape != pi_t.shape or not np.all(np.isfinite(Q_t)):
            raise ValueError("Invalid Q estimate")

        V_t_s_0 = np.dot(Q_t[s_0, :], pi_t[s_0, :])
        V_s_0_arr[t] = V_t_s_0
        sample_arr[t + 1] = sample_arr[t] + n_samples

        eta_t = float(eta_func(t))
        if not np.isfinite(eta_t) or eta_t <= 0:
            raise ValueError("Step size must be positive and finite")

        # KL(new_pi || current_pi) update, written in log form for stability.
        log_pi_t = np.where(pi_t > 0, np.log(np.maximum(pi_t, 1e-300)), -np.inf)
        log_pi_t = log_pi_t - eta_t * Q_t
        log_pi_t -= log_pi_t.max(axis=1, keepdims=True)
        pi_t = np.exp(log_pi_t)
        pi_t /= pi_t.sum(axis=1, keepdims=True)

    return pi_t, V_s_0_arr, sample_arr[1:]


def run_environment(name, args):
    """Run one environment and save raw transitions, values, and final policy."""
    if name == "gridworld":
        from grid_world_env import GridWorld
        env = GridWorld()
        s_0 = env.start
    else:
        from blackjack_env import Blackjack
        env = Blackjack()
        s_0 = env.encode(8, 3)

    output = Path(args.output) / name
    output.mkdir(parents=True, exist_ok=True)
    iteration = 0
    transition_count = 0

    with (output / "transitions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "observation", "state", "action", "cost", "next_state", "done"])

        def log_transition(state, action, cost, next_state, done):
            nonlocal transition_count
            transition_count += 1
            writer.writerow([iteration, transition_count, state, action, cost, next_state, done])

        def estimate_Q(environment, policy, rng):
            nonlocal iteration
            iteration += 1
            return tomc(environment, policy, rng, T=args.horizon,
                        max_observations=args.max_observations,
                        coverage=args.coverage, log_transition=log_transition)

        try:
            policy, values, samples = spmd(
                env, estimate_Q, lambda t: args.eta, s_0,
                n_iters=args.iters, seed=args.seed)
        except RuntimeError as error:
            (output / "status.txt").write_text(f"Training incomplete: {error}\n"
                                                f"Transitions collected: {transition_count}\n")
            print(f"{name}: {error}. Raw transitions saved; no final policy claimed.")
            return

    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["iteration", "estimated_V_before_update", "cumulative_observations"])
        for i, (value, count) in enumerate(zip(values, samples), start=1):
            writer.writerow([i, value, count])

    with (output / "policy.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state"] + [f"action_{a}" for a in range(env.n_actions)])
        for s, row in enumerate(policy):
            writer.writerow([s, *row])

    (output / "status.txt").write_text(
        f"Completed {args.iters} iterations; transitions collected: {transition_count}\n"
        f"Coverage mode: {args.coverage} (auto = exploring starts for GridWorld)\n"
        "Values are truncated, pre-update estimates; final policy not independently evaluated.\n"
    )
    print(f"{name}: completed {args.iters} updates, {transition_count} transitions. Saved in {output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TOMC + KL-SPMD and save observations")
    parser.add_argument("--env", choices=["gridworld", "blackjack", "both"], default="both")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--max-observations", type=int, default=100_000,
                        help="Maximum samples per TOMC iteration")
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", choices=["auto", "continuous", "exploring_starts"],
                        default="auto")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    if args.iters < 1 or args.horizon < 1 or args.max_observations < 1 or not np.isfinite(args.eta) or args.eta <= 0:
        parser.error("iters, horizon, max-observations and eta must be positive")

    names = ["gridworld", "blackjack"] if args.env == "both" else [args.env]
    for name in names:
        run_environment(name, args)
