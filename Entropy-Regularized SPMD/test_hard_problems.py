"""Checks for the hard problems, exact scoring and the online estimator.

Run from this folder with:  python test_hard_problems.py
Each sampled check compares against exact values computed from the rules, with a
tolerance of several standard errors.
"""
import sys
import traceback

import numpy as np

import kl_spmd
from baseline import GridWorld
from baseline_blackjack import Blackjack
from estimators import generative_q, online_q
from exact_eval import ExactScorer
from hard_problems import DeceptiveGridWorld, FullBlackjack

ENVIRONMENTS = [GridWorld(), Blackjack(), DeceptiveGridWorld(), FullBlackjack()]


def random_policy(env, seed):
    return np.random.default_rng(seed).dirichlet(np.ones(env.n_actions), size=env.n_states)


def test_exact_models_are_valid():
    for env in ENVIRONMENTS:
        P, c = ExactScorer(env).P, ExactScorer(env).c
        assert np.all(P >= 0), type(env).__name__
        assert np.all(P.sum(axis=2) <= 1 + 1e-12), type(env).__name__
        assert np.isfinite(c).all(), type(env).__name__


def test_known_optimal_values():
    # GridWorld and blackjack optima were computed independently earlier; the hard
    # problem values are the ones documented in hard_problems.py.
    np.testing.assert_allclose(ExactScorer(GridWorld()).optimal, 1.9533, atol=1e-4)
    blackjack = ExactScorer(Blackjack())
    start = int(Blackjack().encode(8, 3))
    np.testing.assert_allclose(blackjack.optimal_values()[start], 0.2750, atol=1e-4)
    np.testing.assert_allclose(ExactScorer(DeceptiveGridWorld()).optimal, 1.198, atol=1e-3)


def test_generative_sampling_matches_exact_q():
    # Cost noise is switched off here so a small tolerance can catch modelling errors.
    rng = np.random.default_rng(0)
    for env, rollouts, horizon in [(GridWorld(), 2000, 120), (Blackjack(), 20000, 10),
                                   (DeceptiveGridWorld(cost_noise=0.0), 2000, 150), (FullBlackjack(), 20000, 12)]:
        policy = random_policy(env, 1)
        q, _ = generative_q(env, policy, rollouts, horizon, rng)
        exact = ExactScorer(env).q_values(policy)
        assert np.abs(q - exact).max() < 0.1, f"{type(env).__name__}: {np.abs(q - exact).max():.3f}"


def test_cost_noise_averages_out():
    # With noise sd 1, single Q estimates are noisy, but on average they must be unbiased.
    env = DeceptiveGridWorld()
    policy = random_policy(env, 1)
    q, _ = generative_q(env, policy, 2000, 150, np.random.default_rng(0))
    error = q - ExactScorer(env).q_values(policy)
    assert abs(error.mean()) < 0.01, f"average error {error.mean():.4f}"
    assert error.std() < 0.07, f"spread {error.std():.3f}"  # about 2.3 / sqrt(2000) = 0.05 expected


def test_online_sampling_matches_exact_q_where_visited():
    rng = np.random.default_rng(2)
    for env, episodes, horizon in [(DeceptiveGridWorld(cost_noise=0.0), 20000, 160), (FullBlackjack(), 200000, 14)]:
        policy = random_policy(env, 3)
        q, _ = online_q(env, policy, episodes, horizon, rng)
        exact = ExactScorer(env).q_values(policy)
        # Compare only pairs with plenty of visits; rarely visited pairs are noisy.
        _, counts = online_counts(env, policy, episodes, horizon, np.random.default_rng(2))
        well_visited = counts >= 1000
        assert well_visited.sum() >= 20, f"{type(env).__name__}: only {well_visited.sum()} well-visited pairs"
        error = np.abs(q - exact)[well_visited].max()
        assert error < 0.1, f"{type(env).__name__}: {error:.3f}"


def online_counts(env, policy, episodes, horizon, rng):
    """Visit counts per (state, action) for the same sampling scheme as online_q."""
    record = horizon // 2
    states = np.array(env.initial_states(episodes, rng), dtype=int)
    alive = np.ones(episodes, dtype=bool)
    counts = np.zeros((env.n_states, env.n_actions))
    for t in range(record):
        active = np.flatnonzero(alive)
        if not len(active):
            break
        actions = kl_spmd.utils.draw_actions(policy, states[active], rng)
        np.add.at(counts, (states[active], actions), 1)
        next_states, _, done = env.step(states[active], actions, rng)
        states[active] = next_states
        alive[active[done]] = False
    return None, counts


def test_online_q_leaves_unvisited_pairs_alone():
    env = DeceptiveGridWorld(cost_noise=0.0)
    policy = np.zeros((env.n_states, 4))
    policy[:, 3] = 1.0  # always "right": from the start, walks straight into the decoy
    q, _ = online_q(env, policy, 50, 20, np.random.default_rng(4))
    far_state = 11 * 12 + 11
    assert np.all(q[far_state] == q[far_state][0]), "unvisited state should get a constant row"
    start = q[env.start]
    assert np.all(start == start[3]), "untried actions should get the state's average return"


def test_entropy_wrapper_matches_exact_regularized_q():
    env, tau = FullBlackjack(), 0.5
    policy = random_policy(env, 5)
    wrapped = kl_spmd.EntropyCostEnv(env, policy, tau)
    q, _ = generative_q(wrapped, policy, 20000, 12, np.random.default_rng(6))
    scorer = ExactScorer(env)
    entropy_cost = tau * np.sum(policy * np.log(policy), axis=1)
    P_pi = np.einsum("sa,sat->st", policy, scorer.P)
    c_pi = (policy * scorer.c).sum(axis=1) + entropy_cost
    v = np.linalg.solve(np.eye(env.n_states) - env.gamma * P_pi, c_pi)
    exact = scorer.c + entropy_cost[:, None] + env.gamma * scorer.P @ v
    assert np.abs(q - exact).max() < 0.1, f"{np.abs(q - exact).max():.3f}"


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failed += 1
            location = traceback.extract_tb(error.__traceback__)[-1]
            summary = " ".join(str(error).split())[:160]
            print(f"FAIL  {name}\n      {type(error).__name__} at line {location.lineno}: {summary}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
