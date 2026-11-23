"""M5 retail inventory problem (real-data bonus).

Sampled checks compare the simulator and the team's TOMC estimator against the exact model with a
tolerance of several standard errors.
"""

from pathlib import Path

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import TOMC, uniform_policy
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.envs.inventory import SERIES_FILE, RetailInventory, load_series
from reference import regularized_q

_ENV = get_env("inventory")
assert isinstance(_ENV, RetailInventory)
ENV: RetailInventory = _ENV
SCORER = get_scorer("inventory")
P = SCORER.model.dense_transitions()
C = SCORER.model.costs


def test_size_and_data() -> None:
    units, weekdays, price = load_series()
    assert len(units) == 730
    assert units.min() >= 0
    # 13 x 7 = 91 (stock, weekday) cells, minus a full shelf on Thursday (no zero-sale Wednesdays)
    assert (ENV.capacity, ENV.n_states, ENV.n_actions) == (12, 90, 13)
    assert ENV.encode(12, 3) == -1
    assert ENV.demand_pmf[2, 0] == 0
    np.testing.assert_allclose(price, 1.65, atol=0.01)
    for day in range(7):  # each weekday's demand distribution is that weekday's sales histogram
        counts = np.bincount(units[weekdays == day], minlength=ENV.demand_pmf.shape[1])
        np.testing.assert_allclose(ENV.demand_pmf[day], counts / counts.sum())


def test_data_file_ships_with_the_package() -> None:
    assert SERIES_FILE.exists()
    assert SERIES_FILE.parent.name == "data"


def test_hand_computed_steps() -> None:
    rng = np.random.default_rng(0)
    # Empty shelf, order nothing: nothing to sell, no cost, next day is Tuesday with 0 units.
    s = int(ENV.encode(0, 0))
    next_states, costs, done = ENV.step(np.full(50, s), np.zeros(50, dtype=np.int64), rng)
    assert np.all(costs == 0)
    assert np.all(next_states == ENV.encode(0, 1))
    assert not done.any()
    # 5 units, order 12: only 7 fit and are paid for; with demand d the cost is
    # 1 + 7 * unit_cost + 0.05 * (12 - min(12, d)) - price * min(12, d).
    s = int(ENV.encode(5, 2))
    d = np.arange(ENV.demand_pmf.shape[1])
    sold = np.minimum(12, d)
    expected = 1 + 7 * ENV.unit_cost + 0.05 * (12 - sold) - ENV.price * sold
    np.testing.assert_allclose(C[s, 12], ENV.demand_pmf[2] @ expected)
    np.testing.assert_allclose(C[s, 7], C[s, 12])  # ordering more than fits changes nothing
    np.testing.assert_allclose(P[s, 7], P[s, 12])


def test_exact_model_is_valid() -> None:
    np.testing.assert_allclose(P.sum(axis=2), 1.0)
    assert (P >= 0).all()
    assert np.isfinite(C).all()
    assert C.max() <= ENV.cost_bound  # the TOMC fill value cost_bound / (1 - gamma) is pessimistic


def test_simulator_matches_exact_model() -> None:
    rng, n = np.random.default_rng(1), 200_000
    for cell, a in [
        (ENV.encode(0, 0), 8),
        (ENV.encode(3, 5), 6),
        (ENV.encode(10, 6), 0),
        (ENV.encode(2, 4), 12),
    ]:
        s = int(cell)
        next_states, costs, _ = ENV.step(np.full(n, s), np.full(n, a), rng)
        assert abs(costs.mean() - C[s, a]) < 5 * costs.std() / np.sqrt(n), (s, a)
        freq = np.bincount(next_states, minlength=ENV.n_states) / n
        assert np.abs(freq - P[s, a]).max() < 0.005, (s, a)


def test_every_state_is_reachable_and_step_one_matches_the_model() -> None:
    # Online coverage needs every state to be reachable from the start.
    reached, frontier = {ENV.start}, [ENV.start]
    while frontier:
        for nxt in set(np.flatnonzero(P[frontier.pop()].sum(0)).tolist()) - reached:
            reached.add(nxt)
            frontier.append(nxt)
    assert len(reached) == ENV.n_states
    rng, s, a, n = np.random.default_rng(2), int(ENV.encode(4, 5)), 3, 50_000
    samples = [ENV.step_one(s, a, rng) for _ in range(n)]
    costs = np.array([c for _, c, _ in samples])
    assert abs(costs.mean() - C[s, a]) < 5 * costs.std() / np.sqrt(n)
    freq = np.bincount([t for t, _, _ in samples], minlength=ENV.n_states) / n
    assert np.abs(freq - P[s, a]).max() < 0.01


def test_value_iteration_matches_policy_iteration() -> None:
    gamma = ENV.gamma
    policy = np.zeros(ENV.n_states, dtype=int)
    v = np.zeros(ENV.n_states)
    for _ in range(100):
        P_pi, c_pi = P[np.arange(ENV.n_states), policy], C[np.arange(ENV.n_states), policy]
        v = np.linalg.solve(np.eye(ENV.n_states) - gamma * P_pi, c_pi)
        new = (C + gamma * P @ v).argmin(axis=1)
        if np.array_equal(new, policy):
            break
        policy = new
    np.testing.assert_allclose(v, SCORER.optimal_values, atol=1e-8)


def test_team_tomc_estimator_matches_exact_q() -> None:
    # TOMC visits every pair under the uniform policy and records one return per pair, cut off once the
    # discounted tail is below bias * (1 - gamma) = 0.001, so the cut-off changes Q by at most
    # 0.001 * max |V| < 0.03. Average 30 independent estimates; allow 5 standard errors per pair, and
    # 4 standard errors (across the 30 runs) for the average error over all pairs, each plus 0.03.
    policy = uniform_policy(ENV.n_states, ENV.n_actions)
    exact = SCORER.q_values(policy)
    errors = np.array(
        [
            TOMC(bias=0.01).estimate(ENV, policy, np.random.default_rng(seed), 1_000_000).q - exact
            for seed in range(30)
        ]
    )
    se = errors.std(0, ddof=1) / np.sqrt(len(errors))
    z = np.abs(errors.mean(0)) / (5 * se + 0.03)
    assert z.max() < 1, f"worst pair off by {z.max():.2f} x tolerance"
    run_means = errors.mean(axis=(1, 2))
    assert abs(run_means.mean()) < 4 * run_means.std(ddof=1) / np.sqrt(len(run_means)) + 0.03


def test_entropy_reg_spmd_converges_to_soft_optimum() -> None:
    from ie692_rl.enhancements.entropy_reg_spmd import soft_optimal_policy

    tau, eta = 0.05, 5.0
    soft = soft_optimal_policy(SCORER, tau)
    policy = uniform_policy(ENV.n_states, ENV.n_actions)
    for _ in range(300):
        policy = KL().step(policy, regularized_q(P, C, ENV.gamma, policy, tau), eta, tau)
    assert np.abs(policy - soft).max() < 1e-6


@pytest.mark.parametrize(
    "kwargs", [{"capacity": 12, "weekly": False}, {"capacity": 6}, {"start_stock": 3, "start_weekday": 4}]
)
def test_other_settings_build_valid_models(kwargs: dict[str, int | bool]) -> None:
    env = RetailInventory.from_m5(**kwargs)  # type: ignore[arg-type]
    P_env = env.model().dense_transitions()
    np.testing.assert_allclose(P_env.sum(axis=2), 1.0)
    assert env.start >= 0
    q = TOMC().estimate(env, random_policy(env, 0, floor=0.5), np.random.default_rng(0), 10_000_000).q
    assert np.all(np.isfinite(q))


@pytest.mark.parametrize(
    ("units", "weekdays", "price", "kwargs"),
    [
        ([], [], 1.0, {}),
        ([1, -1], [0, 1], 1.0, {}),
        ([1, 2], [0], 1.0, {}),
        ([1] * 7, list(range(7)), 0.0, {}),
        ([1] * 7, list(range(7)), 1.0, {"gamma": 1.0}),
        ([1] * 6, list(range(6)), 1.0, {}),  # no Sunday data
        ([1] * 7, list(range(7)), 1.0, {"capacity": 2, "start_stock": 5}),
    ],
)
def test_invalid_inputs(
    units: list[int], weekdays: list[int], price: float, kwargs: dict[str, float]
) -> None:
    with pytest.raises((ValueError, IndexError)):
        RetailInventory(np.array(units, dtype=np.int64), np.array(weekdays, dtype=np.int64), price, **kwargs)  # type: ignore[arg-type]


def test_load_series_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "series.csv"
    path.write_text("date,weekday,units_sold,sell_price\n2020-01-06,Monday,3,2.0\n2020-01-07,Tuesday,0,\n")
    units, weekdays, price = load_series(path)
    np.testing.assert_array_equal(units, [3, 0])
    np.testing.assert_array_equal(weekdays, [0, 1])
    assert price == 2.0
