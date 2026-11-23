"""True Online TD(lambda) and the TD(0)/TD(lambda) baselines: features, exactness vs the original dense
implementation, convergence to the exact Q, history, and the SPMD estimator wrapper."""

import math

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import BudgetExhaustedError, run_spmd, uniform_policy
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.enhancements.true_online_td import (
    TrueOnlineTD,
    linear_features,
    tabular_features,
    td_lambda,
    td_zero,
    true_online_td_lambda,
)
from ie692_rl.enhancements.true_online_td.td import ACCUMULATING, TRUE_ONLINE, run_td
from ie692_rl.envs import Blackjack, GridWorld
from reference import reference_td

# ----------------------------------------------------------------------------- features


def test_tabular_features_are_one_per_pair(small_problem: str) -> None:
    env = get_env(small_problem)
    features = tabular_features(env)
    assert features.n_features == env.n_states * env.n_actions
    assert sorted(features.feature_of.ravel().tolist()) == list(range(features.n_features))
    np.testing.assert_array_equal(features.dense(), np.eye(features.n_features))


@pytest.mark.parametrize(("size", "groups"), [(10, 25), (5, 9), (30, 225)])
def test_gridworld_aggregation_uses_2x2_blocks(size: int, groups: int) -> None:
    env = GridWorld(size=size)
    features = linear_features(env)
    assert features.n_features == groups * 4
    fid = features.feature_of
    assert fid[0, 0] == fid[1, 0] == fid[size, 0] == fid[size + 1, 0]  # one block
    assert fid[0, 0] != fid[2, 0]
    assert fid[0, 0] != fid[0, 1]  # separate parameters per action
    dense = features.dense()
    np.testing.assert_array_equal(dense.sum(axis=1), 1.0)


def test_blackjack_aggregation() -> None:
    env = Blackjack()
    features = linear_features(env)
    assert features.n_features == 16
    fid = features.feature_of
    for dealer in range(2, 6):
        low = {int(fid[env.encode(total, dealer), 0]) for total in range(4, 8)}
        high = {int(fid[env.encode(total, dealer), 0]) for total in range(8, 12)}
        assert len(low) == len(high) == 1
        assert low != high


def test_no_aggregation_for_inventory() -> None:
    with pytest.raises(ValueError, match="No aggregated features"):
        linear_features(get_env("inventory"))


# ----------------------------------------------------------------------------- exactness


MODES = [("td0", ACCUMULATING, 0.0), ("td_lambda", ACCUMULATING, 0.8), ("true_online", TRUE_ONLINE, 0.8)]


@pytest.mark.parametrize(("mode_name", "mode", "lam"), MODES)
@pytest.mark.parametrize("feature_name", ["tabular", "linear"])
@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
def test_matches_original_dense_implementation(
    name: str, feature_name: str, mode_name: str, mode: int, lam: float
) -> None:
    env = get_env(name)
    features = (tabular_features if feature_name == "tabular" else linear_features)(env)
    policy = random_policy(env, 1, floor=0.3)
    expected = reference_td(
        env, policy, features.dense(), 3000, 0.05, lam, np.random.default_rng(9), mode_name
    )
    exact = run_td(
        env, policy, features, 3000, 0.05, lam, np.random.default_rng(9), mode, 0, trace_cutoff=0.0
    )
    np.testing.assert_allclose(exact.q, expected, rtol=0, atol=1e-10)
    pruned = run_td(env, policy, features, 3000, 0.05, lam, np.random.default_rng(9), mode, 0)
    np.testing.assert_allclose(pruned.q, expected, rtol=0, atol=1e-8)


def test_lambda_zero_reduces_to_td0() -> None:
    env = get_env("gridworld10")
    features, policy = tabular_features(env), uniform_policy(100, 4)
    td0 = td_zero(env, policy, features, 5000, 0.1, np.random.default_rng(0))
    accumulating = td_lambda(env, policy, features, 5000, 0.1, 0.0, np.random.default_rng(0))
    true_online = true_online_td_lambda(env, policy, features, 5000, 0.1, 0.0, np.random.default_rng(0))
    np.testing.assert_allclose(accumulating.q, td0.q, atol=1e-12)
    np.testing.assert_allclose(true_online.q, td0.q, atol=1e-12)


@pytest.mark.parametrize("method", [td_zero, td_lambda, true_online_td_lambda])
def test_tabular_td_converges_to_exact_q_on_blackjack(method: object) -> None:
    env, scorer = get_env("blackjack"), get_scorer("blackjack")
    policy = uniform_policy(32, 2)
    exact = scorer.q_values(policy)
    rng = np.random.default_rng(1)
    features = tabular_features(env)
    if method is td_zero:
        result = td_zero(env, policy, features, 2_000_000, 0.002, rng)
    elif method is td_lambda:
        result = td_lambda(env, policy, features, 2_000_000, 0.002, 0.8, rng)
    else:
        result = true_online_td_lambda(env, policy, features, 2_000_000, 0.002, 0.8, rng)
    assert np.abs(result.q - exact).mean() < 0.02
    assert np.abs(result.q - exact).max() < 0.1


def test_history_records_value_at_the_evaluation_state() -> None:
    env = get_env("gridworld10")
    policy = uniform_policy(100, 4)
    result = true_online_td_lambda(
        env, policy, tabular_features(env), 1050, 0.1, 0.8, np.random.default_rng(2), 100
    )
    np.testing.assert_array_equal(result.history_steps, np.arange(100, 1001, 100))
    assert len(result.history_values) == 10
    assert result.observations == 1050
    no_history = true_online_td_lambda(
        env, policy, tabular_features(env), 1050, 0.1, 0.8, np.random.default_rng(2), 0
    )
    assert len(no_history.history_values) == 0
    assert policy[env.start] @ no_history.q[env.start] == pytest.approx(
        policy[env.start] @ result.q[env.start]
    )


@pytest.mark.parametrize(
    ("n_steps", "eta", "lam", "record_every", "q_init", "cutoff"),
    [
        (-1, 0.1, 0.5, 1, 0.0, 0.0),
        (10, 0.0, 0.5, 1, 0.0, 0.0),
        (10, math.inf, 0.5, 1, 0.0, 0.0),
        (10, 0.1, 1.5, 1, 0.0, 0.0),
        (10, 0.1, -0.1, 1, 0.0, 0.0),
        (10, 0.1, 0.5, -1, 0.0, 0.0),
        (10, 0.1, 0.5, 1, math.nan, 0.0),
        (10, 0.1, 0.5, 1, 0.0, -1.0),
    ],
)
def test_run_td_validation(
    n_steps: int, eta: float, lam: float, record_every: int, q_init: float, cutoff: float
) -> None:
    env = get_env("blackjack")
    with pytest.raises(ValueError):  # noqa: PT011
        run_td(
            env,
            uniform_policy(32, 2),
            tabular_features(env),
            n_steps,
            eta,
            lam,
            np.random.default_rng(0),
            TRUE_ONLINE,
            record_every,
            q_init=q_init,
            trace_cutoff=cutoff,
        )


def test_feature_map_must_match_environment() -> None:
    with pytest.raises(ValueError, match="Feature map"):
        run_td(
            get_env("blackjack"),
            uniform_policy(32, 2),
            tabular_features(get_env("gridworld10")),
            10,
            0.1,
            0.5,
            np.random.default_rng(0),
            TRUE_ONLINE,
        )


def test_q_init_fills_unvisited_pairs() -> None:
    env = get_env("gridworld30")
    out = run_td(
        env,
        uniform_policy(900, 4),
        tabular_features(env),
        50,
        0.1,
        0.5,
        np.random.default_rng(0),
        TRUE_ONLINE,
        0,
        q_init=7.0,
    )
    assert out.states_visited < 900
    assert np.sum(out.q == 7.0) > 3000  # most of the 30x30 grid is untouched in 50 steps


# ----------------------------------------------------------------------------- SPMD estimator


def test_estimator_budget_and_observations() -> None:
    env = get_env("blackjack")
    estimator = TrueOnlineTD(eta=0.05, lam=0.8, n_steps=500)
    estimate = estimator.estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 600)
    assert estimate.observations == 500
    assert estimate.q.shape == (32, 2)
    with pytest.raises(BudgetExhaustedError):
        estimator.estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 499)


@pytest.mark.parametrize("kwargs", [{"features": "cubic"}, {"n_steps": 0}])
def test_estimator_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        TrueOnlineTD(**kwargs)  # type: ignore[arg-type]


def test_estimator_passes_extra_cost() -> None:
    env = get_env("blackjack")
    plain = TrueOnlineTD(n_steps=2000).estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 10**6)
    shifted = TrueOnlineTD(n_steps=2000).estimate(
        env, uniform_policy(32, 2), np.random.default_rng(0), 10**6, np.full(32, -0.5)
    )
    assert shifted.q.mean() < plain.q.mean()


@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
def test_divergent_step_size_stops_spmd(name: str) -> None:
    """eta = 5 with lambda = 0.95 overflows; SPMD must stop cleanly ('diverged') instead of producing NaN."""
    env = get_env(name)
    result = run_spmd(env, TrueOnlineTD(eta=5.0, lam=0.95, n_steps=20_000), KL(), 1.0, budget=10**6, seed=0)
    assert result.history[-1].status == "diverged"
    assert np.all(np.isfinite(result.policy))


def test_spmd_with_true_online_td_improves_gridworld() -> None:
    env, scorer = get_env("gridworld10"), get_scorer("gridworld10")
    result = run_spmd(
        env,
        TrueOnlineTD(eta=0.5, lam=0.8, n_steps=50_000),
        KL(),
        1.0,
        budget=2_000_000,
        seed=0,
        scorer=scorer,
    )
    assert result.history[-1].gap < 0.5 * result.history[0].gap
