"""Exact scoring on the sparse model agrees with dense linear algebra and policy iteration."""

import time

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import ExactScorer, uniform_policy
from ie692_rl.envs import Blackjack
from ie692_rl.types import FloatArray


def dense_values(name: str, policy: FloatArray) -> FloatArray:
    scorer = get_scorer(name)
    P, c = scorer.model.dense_transitions(), scorer.model.costs
    P_pi = np.einsum("sa,sat->st", policy, P)
    result: FloatArray = np.linalg.solve(np.eye(len(c)) - scorer.gamma * P_pi, (policy * c).sum(axis=1))
    return result


@pytest.mark.parametrize("seed", [0, 1])
def test_values_match_dense_solve(small_problem: str, seed: int) -> None:
    env, scorer = get_env(small_problem), get_scorer(small_problem)
    policy = random_policy(env, seed)
    np.testing.assert_allclose(scorer.values(policy), dense_values(small_problem, policy), atol=1e-9)


def test_q_values_are_consistent(small_problem: str) -> None:
    env, scorer = get_env(small_problem), get_scorer(small_problem)
    policy = random_policy(env, 2)
    q, v = scorer.q_values(policy), scorer.values(policy)
    np.testing.assert_allclose((policy * q).sum(axis=1), v, atol=1e-9)


def test_optimal_values_match_policy_iteration(small_problem: str) -> None:
    scorer = get_scorer(small_problem)
    P, c, gamma = scorer.model.dense_transitions(), scorer.model.costs, scorer.gamma
    S = len(c)
    greedy = np.zeros(S, dtype=int)
    v = np.zeros(S)
    for _ in range(200):
        v = np.linalg.solve(np.eye(S) - gamma * P[np.arange(S), greedy], c[np.arange(S), greedy])
        new = (c + gamma * P @ v).argmin(axis=1)
        if np.array_equal(new, greedy):
            break
        greedy = new
    np.testing.assert_allclose(scorer.optimal_values, v, atol=1e-8)


def test_optimal_is_a_lower_bound(small_problem: str) -> None:
    env, scorer = get_env(small_problem), get_scorer(small_problem)
    for seed in range(20):
        assert scorer.gap(random_policy(env, seed)) >= -1e-9
    assert scorer.score(uniform_policy(env.n_states, env.n_actions)) >= scorer.optimal - 1e-9


def test_handout_gridworld_reference_values() -> None:
    scorer = get_scorer("gridworld10")
    assert scorer.optimal == pytest.approx(1.953311185892705, abs=1e-8)  # matches results/divergences
    assert scorer.score(uniform_policy(100, 4)) == pytest.approx(2.2870335199938445, abs=1e-8)


def test_gridworld30_scores_quickly_with_small_residual() -> None:
    env, scorer = get_env("gridworld30"), get_scorer("gridworld30")
    policy = random_policy(env, 3)
    started = time.perf_counter()
    v = scorer.values(policy)
    assert time.perf_counter() - started < 1.0
    q = scorer.q_values(policy)
    residual = np.abs((policy * q).sum(axis=1) - v).max()
    assert residual < 1e-9
    assert 0 <= scorer.optimal <= scorer.score(policy)


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="tol"):
        ExactScorer(Blackjack(), tol=0.0)
    with pytest.raises(ValueError, match="probabilities"):
        get_scorer("blackjack").values(np.full((32, 2), 0.3))


def test_dense_transitions_rows(problem: str) -> None:
    if problem == "gridworld30":
        pytest.skip("dense matrix is only for small problems")
    model = get_scorer(problem).model
    P = model.dense_transitions()
    np.testing.assert_allclose(P.sum(axis=2), model.probs.sum(axis=2))
