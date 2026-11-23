"""Entropy-regularized SPMD: annealed tau, the soft optimum, and convergence to it."""

import math

import numpy as np
import pytest

from conftest import get_env, get_scorer
from ie692_rl.core import TOMC, run_spmd, uniform_policy
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.enhancements.entropy_reg_spmd import AnnealedTau, soft_optimal_policy
from reference import regularized_q


def test_annealed_tau_schedule() -> None:
    schedule = AnnealedTau(0.1, 1.0)
    assert [schedule(k) for k in range(4)] == pytest.approx([0.1, 0.05, 0.1 / 3, 0.025])
    assert AnnealedTau(0.3, 0.0)(50) == pytest.approx(0.3)
    assert AnnealedTau(0.0)(3) == 0.0
    assert AnnealedTau(1.0, 2.0)(10**6) > 0  # never exactly zero, never negative


@pytest.mark.parametrize(("tau0", "power"), [(-0.1, 1.0), (0.1, -1.0), (math.inf, 1.0), (0.1, math.nan)])
def test_annealed_tau_validation(tau0: float, power: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AnnealedTau(tau0, power)


def test_annealed_tau_rejects_negative_iterations() -> None:
    with pytest.raises(ValueError, match="iteration"):
        AnnealedTau(0.1)(-1)


@pytest.mark.parametrize("tau", [0.0, -1.0, math.inf])
def test_soft_optimal_policy_validation(tau: float) -> None:
    with pytest.raises(ValueError, match="tau"):
        soft_optimal_policy(get_scorer("blackjack"), tau)


@pytest.mark.parametrize("tau", [1e-3, 0.05, 1.0, 100.0])
def test_soft_optimal_policy_is_a_fixed_point(small_problem: str, tau: float) -> None:
    """softmax(-Q_tau^pi / tau) = pi for the soft-optimal pi; no NaN even for tiny tau."""
    scorer = get_scorer(small_problem)
    soft = soft_optimal_policy(scorer, tau)
    assert np.all(np.isfinite(soft))
    assert np.all(soft > 0) or tau < 0.01  # tiny tau may underflow rare actions to 0
    np.testing.assert_allclose(soft.sum(axis=1), 1.0)
    P, c = scorer.model.dense_transitions(), scorer.model.costs
    q = regularized_q(P, c, scorer.gamma, soft, tau)
    target = np.exp(-(q - q.min(axis=1, keepdims=True)) / tau)
    target /= target.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(soft, target, atol=1e-6)


def test_small_tau_approaches_the_unregularized_optimum(small_problem: str) -> None:
    scorer = get_scorer(small_problem)
    assert scorer.gap(soft_optimal_policy(scorer, 1e-4)) < 1e-3


def test_large_tau_is_near_uniform_in_the_continuing_gridworld() -> None:
    """Every GridWorld step earns the same entropy bonus, so a huge tau gives a near-uniform policy.
    (Not in Blackjack: drawing prolongs the hand and earns more bonus, so the soft optimum prefers it.)"""
    np.testing.assert_allclose(soft_optimal_policy(get_scorer("gridworld10"), 100.0), 0.25, atol=0.01)


@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
def test_kl_spmd_with_regularized_q_converges_to_soft_optimum(name: str) -> None:
    env, scorer = get_env(name), get_scorer(name)
    tau, eta = 0.05, 5.0
    P, c = scorer.model.dense_transitions(), scorer.model.costs
    policy = uniform_policy(env.n_states, env.n_actions)
    for _ in range(300):
        policy = KL().step(policy, regularized_q(P, c, env.gamma, policy, tau), eta, tau)
    assert np.abs(policy - soft_optimal_policy(scorer, tau)).max() < 1e-6


def test_entropy_keeps_more_probability_on_every_action() -> None:
    """With tau > 0 the smallest action probability stays positive where plain SPMD underflows to 0.

    Not bounded away from 0 by much: once pi(a|s) < pi_min, TOMC fills Q(s, a) with the pessimistic
    1 / (1 - gamma) = 10, and the regularized step then drives that action towards exp(-10 / tau)."""
    env, scorer = get_env("blackjack"), get_scorer("blackjack")
    plain = run_spmd(env, TOMC(), KL(), 100.0, budget=60_000, seed=3, scorer=scorer)
    regularized = run_spmd(env, TOMC(), KL(), 100.0, budget=60_000, seed=3, tau=0.1, scorer=scorer)
    assert regularized.policy.min() > 0
    assert regularized.policy.min() > plain.policy.min()
    assert all(h.tau == 0.1 for h in regularized.history if h.status == "updated")
