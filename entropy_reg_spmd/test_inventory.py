"""Checks for the M5 retail inventory problem.

Run from the repository root:  .venv/bin/python entropy_reg_spmd/test_inventory.py
Sampled checks (the simulator and the team's TOMC estimator, baseline.tomc) compare against the exact model with a tolerance of several standard errors.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import baseline  # noqa: E402  (the team's TOMC estimator, in the repository root)
from entropy_reg_spmd import ExactScorer, entropy_reg_spmd  # noqa: E402
from entropy_reg_spmd.inventory import RetailInventory, load_series  # noqa: E402

ENV = RetailInventory.from_m5()
SCORER = ExactScorer(ENV)


def random_policy(seed):
    return np.random.default_rng(seed).dirichlet(np.ones(ENV.n_actions), size=ENV.n_states)


def test_size_and_data():
    units, weekdays, price = load_series()
    assert len(units) == 730 and units.min() >= 0
    # 13 x 7 = 91 (stock, weekday) cells, minus a full shelf on Thursday (no zero-sale Wednesdays)
    assert (ENV.capacity, ENV.n_states, ENV.n_actions) == (12, 90, 13)
    assert ENV.encode(12, 3) == -1 and ENV.demand_pmf[2, 0] == 0
    np.testing.assert_allclose(price, 1.65, atol=0.01)
    for day in range(7):  # each weekday's demand distribution is that weekday's sales histogram
        counts = np.bincount(units[weekdays == day], minlength=ENV.demand_pmf.shape[1])
        np.testing.assert_allclose(ENV.demand_pmf[day], counts / counts.sum())


def test_hand_computed_steps():
    rng = np.random.default_rng(0)
    # Empty shelf, order nothing: nothing to sell, no cost, next day is Tuesday with 0 units.
    s = int(ENV.encode(0, 0))
    next_states, costs, done = ENV.step(np.full(50, s), np.zeros(50, dtype=int), rng)
    assert np.all(costs == 0) and np.all(next_states == ENV.encode(0, 1)) and not done.any()
    # 5 units, order 12: only 7 fit and are paid for; with demand d the cost is
    # 1 + 7 * unit_cost + 0.05 * (12 - min(12, d)) - price * min(12, d).
    s = int(ENV.encode(5, 2))
    P, c = SCORER.P, SCORER.c
    d = np.arange(ENV.demand_pmf.shape[1])
    sold = np.minimum(12, d)
    expected = 1 + 7 * ENV.unit_cost + 0.05 * (12 - sold) - ENV.price * sold
    np.testing.assert_allclose(c[s, 12], ENV.demand_pmf[2] @ expected)
    np.testing.assert_allclose(c[s, 7], c[s, 12])  # ordering more than fits changes nothing
    np.testing.assert_allclose(P[s, 7], P[s, 12])


def test_exact_model_is_valid():
    assert np.allclose(SCORER.P.sum(axis=2), 1) and (SCORER.P >= 0).all() and np.isfinite(SCORER.c).all()


def test_simulator_matches_exact_model():
    rng = np.random.default_rng(1)
    n = 200_000
    for s, a in [(ENV.encode(0, 0), 8), (ENV.encode(3, 5), 6), (ENV.encode(10, 6), 0), (ENV.encode(2, 4), 12)]:
        s = int(s)
        next_states, costs, _ = ENV.step(np.full(n, s), np.full(n, a), rng)
        assert abs(costs.mean() - SCORER.c[s, a]) < 5 * costs.std() / np.sqrt(n), (s, a)
        freq = np.bincount(next_states, minlength=ENV.n_states) / n
        assert np.abs(freq - SCORER.P[s, a]).max() < 0.005, (s, a)


def test_every_state_is_reachable_and_step_one_matches_the_model():
    # Online coverage needs every state to be reachable from the start.
    reached, frontier = {ENV.start}, [ENV.start]
    while frontier:
        for nxt in set(np.flatnonzero(SCORER.P[frontier.pop()].sum(0)).tolist()) - reached:
            reached.add(nxt)
            frontier.append(nxt)
    assert len(reached) == ENV.n_states
    rng, s, a, n = np.random.default_rng(2), int(ENV.encode(4, 5)), 3, 50_000
    samples = [ENV.step_one(s, a, rng) for _ in range(n)]
    costs = np.array([c for _, c, _ in samples])
    assert abs(costs.mean() - SCORER.c[s, a]) < 5 * costs.std() / np.sqrt(n)
    freq = np.bincount([t for t, _, _ in samples], minlength=ENV.n_states) / n
    assert np.abs(freq - SCORER.P[s, a]).max() < 0.01


def test_value_iteration_matches_policy_iteration():
    P, c, gamma = SCORER.P, SCORER.c, ENV.gamma
    policy = np.zeros(ENV.n_states, dtype=int)
    for _ in range(100):
        P_pi, c_pi = P[np.arange(ENV.n_states), policy], c[np.arange(ENV.n_states), policy]
        v = np.linalg.solve(np.eye(ENV.n_states) - gamma * P_pi, c_pi)
        new = (c + gamma * P @ v).argmin(axis=1)
        if np.array_equal(new, policy):
            break
        policy = new
    np.testing.assert_allclose(v, SCORER.optimal_values(), atol=1e-8)


def test_team_tomc_estimator_matches_exact_q():
    # The team's TOMC (baseline.tomc) visits every pair under the uniform policy and records one
    # return per pair, cut off once the discounted tail is below bias * (1 - gamma) = 0.001, so the
    # cut-off changes Q by at most 0.001 * max |V| < 0.03. Average 30 independent estimates; allow
    # 5 standard errors per pair, and 4 standard errors (measured across the 30 runs) for the
    # average error over all pairs, each plus 0.03.
    policy = np.full((ENV.n_states, ENV.n_actions), 1 / ENV.n_actions)
    exact = SCORER.q_values(policy)
    errors = np.array([baseline.tomc(ENV, policy, np.random.default_rng(seed), bias=0.01,
                                     max_observations=1_000_000)[0] - exact for seed in range(30)])
    se = errors.std(0, ddof=1) / np.sqrt(len(errors))
    z = np.abs(errors.mean(0)) / (5 * se + 0.03)
    assert z.max() < 1, f"worst pair off by {z.max():.2f} x tolerance"
    run_means = errors.mean(axis=(1, 2))
    assert abs(run_means.mean()) < 4 * run_means.std(ddof=1) / np.sqrt(len(run_means)) + 0.03, \
        "average error should be about zero"


def test_entropy_reg_spmd_converges_to_soft_optimum():
    tau, eta, P, c, gamma = 0.05, 5.0, SCORER.P, SCORER.c, ENV.gamma
    v = np.zeros(ENV.n_states)
    for _ in range(3000):  # soft value iteration (costs): V = -tau log sum exp(-Q / tau)
        q = c + gamma * P @ v
        m = q.min(1, keepdims=True)
        v = (m - tau * np.log(np.exp(-(q - m) / tau).sum(1, keepdims=True)))[:, 0]
    q = c + gamma * P @ v
    soft = np.exp(-(q - q.min(1, keepdims=True)) / tau)
    soft /= soft.sum(1, keepdims=True)
    policy = np.full((ENV.n_states, ENV.n_actions), 1 / ENV.n_actions)
    for _ in range(300):
        c_pi = (policy * c).sum(1) + tau * np.sum(policy * np.log(policy), axis=1)
        P_pi = np.einsum("sa,sat->st", policy, P)
        v_pi = np.linalg.solve(np.eye(ENV.n_states) - gamma * P_pi, c_pi)
        policy = entropy_reg_spmd(policy, c + gamma * P @ v_pi, eta, tau)
    assert np.abs(policy - soft).max() < 1e-6, np.abs(policy - soft).max()


def main():
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failed += 1
            line = traceback.extract_tb(error.__traceback__)[-1].lineno
            print(f"FAIL  {name}\n      {type(error).__name__} at line {line}: {str(error)[:200]}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
