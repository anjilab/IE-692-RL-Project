"""True Online TD(lambda) and the TD(0)/TD(lambda) baselines: features, exactness vs the original dense
implementation, convergence to the exact Q, history, and the SPMD estimator wrapper."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import BudgetExhaustedError, run_spmd, uniform_policy
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.enhancements.true_online_td import (
    FeatureMap,
    SarsaLambda,
    TDLambda,
    TDResult,
    TDZero,
    TrueOnlineTD,
    estimated_state_value,
    linear_features,
    tabular_features,
    td_lambda,
    td_lambda_sarsa,
    td_zero,
    true_online_td_lambda,
)
from ie692_rl.enhancements.true_online_td.td import (
    ACCUMULATED_OPERATOR,
    ACCUMULATING,
    TRUE_ONLINE,
    run_td,
)
from ie692_rl.envs import Blackjack, Environment, GridWorld
from ie692_rl.types import FloatArray
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


def test_inventory_aggregates_pairs_of_stock_levels_per_weekday() -> None:
    env = get_env("inventory")
    features = linear_features(env)
    stock, day = env.decode(np.arange(env.n_states))
    for s in range(env.n_states):
        same_group = (day == day[s]) & (stock // 2 == stock[s] // 2)
        assert (features.feature_of[same_group] == features.feature_of[s]).all()  # shared parameters
        assert not np.isin(features.feature_of[~same_group], features.feature_of[s]).any()
    assert len(np.unique(features.feature_of)) == len(np.unique(features.feature_of[:, 0])) * env.n_actions


# ----------------------------------------------------------------------------- exactness


MODES = [
    ("td0", ACCUMULATING, 0.0),
    ("td_lambda", ACCUMULATING, 0.8),
    ("accumulated_operator", ACCUMULATED_OPERATOR, 0.8),
    ("true_online", TRUE_ONLINE, 0.8),
]


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
    sarsa = td_lambda_sarsa(env, policy, features, 5000, 0.1, 0.0, np.random.default_rng(0))
    true_online = true_online_td_lambda(env, policy, features, 5000, 0.1, 0.0, np.random.default_rng(0))
    np.testing.assert_allclose(accumulating.q, td0.q, atol=1e-12)
    np.testing.assert_allclose(sarsa.q, td0.q, atol=1e-12)
    np.testing.assert_allclose(true_online.q, td0.q, atol=1e-12)


@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
@pytest.mark.parametrize("feature_fn", [tabular_features, linear_features])
@pytest.mark.parametrize(
    ("method", "mode"),
    [
        (td_zero, "td0"),
        (td_lambda, "accumulated_operator"),
        (td_lambda_sarsa, "td_lambda"),
        (true_online_td_lambda, "true_online"),
    ],
)
def test_local_methods_preserve_every_value_and_q_checkpoint(
    name: str,
    feature_fn: Callable[[Environment], FeatureMap],
    method: Callable[..., TDResult],
    mode: str,
) -> None:
    env = get_env(name)
    features, policy = feature_fn(env), random_policy(env, 17, floor=0.3)
    # Exercise a state other than the default, and a final step between checkpoints.
    state, n_steps, interval = env.n_states - 1, 137, 17
    expected_values, values = np.empty(n_steps + 1), np.empty(n_steps + 1)
    expected_history: list[tuple[int, FloatArray]] = []
    history: list[tuple[int, FloatArray]] = []
    expected = reference_td(
        env, policy, features.dense(), n_steps, 0.05, 0.8, np.random.default_rng(9), mode,
        evaluation_state=state, value_history=expected_values,
        q_history=expected_history, record_every=interval,
    )
    kwargs = {} if method is td_zero else {"lam": 0.8}
    result = method(
        env, policy, features, n_steps, 0.05, rng=np.random.default_rng(9),
        record_every=interval, evaluation_state=state, value_history=values,
        q_history=history, **kwargs,
    )
    np.testing.assert_allclose(result.q, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(values, expected_values, rtol=0, atol=1e-10)
    assert [step for step, _ in history] == [step for step, _ in expected_history]
    for (_, q), (_, expected_q) in zip(history, expected_history, strict=True):
        np.testing.assert_allclose(q, expected_q, rtol=0, atol=1e-10)
    assert values[-1] == pytest.approx(estimated_state_value(result.q, policy, state))
    np.testing.assert_allclose(result.history_values, values[result.history_steps], atol=1e-12)


def test_accumulated_operator_is_distinct_from_sarsa() -> None:
    env = get_env("gridworld10")
    features, policy = tabular_features(env), uniform_policy(100, 4)
    operator = td_lambda(env, policy, features, 1000, 0.05, 0.8, np.random.default_rng(9))
    sarsa = td_lambda_sarsa(env, policy, features, 1000, 0.05, 0.8, np.random.default_rng(9))
    assert not np.allclose(operator.q, sarsa.q)


@pytest.mark.parametrize("n_steps", [0, 23])
def test_per_update_values_without_q_checkpoints(n_steps: int) -> None:
    env = get_env("blackjack")
    policy = uniform_policy(env.n_states, env.n_actions)
    values = np.full(n_steps + 1, np.nan)
    history: list[tuple[int, FloatArray]] = []
    result = td_lambda(
        env, policy, tabular_features(env), n_steps, 0.05, 0.8, np.random.default_rng(9),
        record_every=0, evaluation_state=env.start, value_history=values, q_history=history,
    )
    assert np.isfinite(values).all()
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(estimated_state_value(result.q, policy, env.start))
    assert history == []
    assert result.history_values.size == 0


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


@pytest.mark.parametrize(("estimator_cls", "mode", "lam"), [
    (TDZero, ACCUMULATING, 0.0),
    (TDLambda, ACCUMULATED_OPERATOR, 0.8),
    (SarsaLambda, ACCUMULATING, 0.8),
    (TrueOnlineTD, TRUE_ONLINE, 0.8),
])
@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
def test_spmd_adapters_preserve_td_updates(
    estimator_cls: type[TDZero | TDLambda | SarsaLambda | TrueOnlineTD], mode: int, lam: float, name: str
) -> None:
    env = get_env(name)
    policy = random_policy(env, 11, floor=0.3)
    extra = np.full(env.n_states, -0.1)
    estimator = estimator_cls(eta=0.05, lam=lam, n_steps=500, q_init=0.3)
    actual = estimator.estimate(env, policy, np.random.default_rng(9), 600, extra)
    expected = run_td(
        env, policy, tabular_features(env), 500, 0.05, lam, np.random.default_rng(9), mode,
        record_every=0, extra_cost=extra, q_init=0.3,
    )
    np.testing.assert_array_equal(actual.q, expected.q)
    assert actual.observations == expected.observations == 500
    assert actual.states_visited == expected.states_visited
    with pytest.raises(BudgetExhaustedError):
        estimator.estimate(env, policy, np.random.default_rng(9), 499)


def test_td_zero_adapter_rejects_nonzero_lambda() -> None:
    with pytest.raises(ValueError, match="lam = 0"):
        TDZero(lam=0.8)


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
