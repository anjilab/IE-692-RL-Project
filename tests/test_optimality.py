"""Model-free optimality certificate (advantage gap): bounds with exact Q, the rollout estimator, the
TOMC false-certificate caveat, and the run_spmd hooks."""

import math

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import (
    TOMC,
    RolloutCertifier,
    advantage_gap,
    gap_bounds,
    run_spmd,
    uniform_policy,
)
from ie692_rl.core.exact import backup
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.envs import Blackjack
from ie692_rl.types import FloatArray

CERTIFIED_PROBLEMS = ["blackjack", "gridworld10", "inventory", "gridworld30"]


def threshold_policy(env: Blackjack, draw_below: int) -> FloatArray:
    """Deterministic Blackjack policy: draw while the total is below ``draw_below``, else stop."""
    policy = np.zeros((env.n_states, 2))
    for s in range(env.n_states):
        policy[s, 1 if env.decode(s)[0] < draw_below else 0] = 1.0
    return policy


# ----------------------------------------------------------------------------- advantage gap


def test_advantage_gap_values() -> None:
    q = np.array([[1.0, 3.0], [2.0, 2.0], [5.0, 0.0]])
    policy = np.array([[0.5, 0.5], [0.3, 0.7], [1.0, 0.0]])
    np.testing.assert_allclose(advantage_gap(q, policy), [1.0, 0.0, 5.0])


def test_advantage_gap_is_zero_for_greedy_rows_and_never_negative() -> None:
    q = np.random.default_rng(0).normal(size=(50, 4)) * 1e3
    greedy = np.eye(4)[q.argmin(axis=1)]
    np.testing.assert_array_equal(advantage_gap(q, greedy), 0.0)
    assert np.all(advantage_gap(q, np.random.default_rng(1).dirichlet(np.ones(4), size=50)) >= 0)


def test_advantage_gap_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        advantage_gap(np.zeros((2, 2)), np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError, match="finite"):
        advantage_gap(np.array([[0.0, np.nan]]), np.full((1, 2), 0.5))


# ----------------------------------------------------------------------------- bounds with exact Q


@pytest.mark.parametrize("name", CERTIFIED_PROBLEMS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_bounds_sandwich_the_true_gap(name: str, seed: int) -> None:
    """g(s0) <= V^pi(s0) - V*(s0) <= max_s g(s) / (1 - gamma), for every state on the right."""
    env, scorer = get_env(name), get_scorer(name)
    policy = random_policy(env, seed, floor=0.3 * seed)
    bounds = gap_bounds(scorer.q_values(policy), policy, env.gamma, env.start_distribution())
    assert bounds.lower <= scorer.gap(policy) + 1e-9
    assert scorer.gap(policy) <= bounds.upper + 1e-9
    per_state_gap = scorer.values(policy) - scorer.optimal_values
    assert np.all(bounds.per_state <= per_state_gap + 1e-9)  # the lower bound holds in every state
    assert np.all(per_state_gap <= bounds.upper + 1e-9)


@pytest.mark.parametrize("name", ["blackjack", "gridworld10", "inventory"])
def test_exact_bounds_vanish_at_the_optimum(name: str) -> None:
    env, scorer = get_env(name), get_scorer(name)
    q_star = backup(
        scorer.model.next_states, scorer.model.probs, scorer.model.costs, scorer.optimal_values, env.gamma
    )
    optimal = np.eye(env.n_actions)[q_star.argmin(axis=1)]
    bounds = gap_bounds(scorer.q_values(optimal), optimal, env.gamma, env.start_distribution())
    assert bounds.upper < 1e-8
    assert bounds.lower < 1e-8


def test_exact_upper_bound_shrinks_as_kl_pmd_converges() -> None:
    env, scorer = get_env("gridworld10"), get_scorer("gridworld10")
    policy, uppers = uniform_policy(100, 4), []
    for _ in range(60):
        q = scorer.q_values(policy)
        uppers.append(gap_bounds(q, policy, env.gamma, env.start_distribution()).upper)
        assert scorer.gap(policy) <= uppers[-1] + 1e-9
        policy = KL().step(policy, q, 50.0)
    assert uppers[-1] < 0.01 * uppers[0]  # constant-step PMD slows to O(1/k) near ties


def test_standard_errors_widen_the_bounds() -> None:
    env, scorer = get_env("blackjack"), get_scorer("blackjack")
    policy = random_policy(env, 3, floor=0.4)
    q, start = scorer.q_values(policy), env.start_distribution()
    se = np.full_like(q, 0.01)
    plain = gap_bounds(q, policy, env.gamma, start)
    assert gap_bounds(q, policy, env.gamma, start, se, 0.0)[:2] == pytest.approx(plain[:2])
    lowers = [gap_bounds(q, policy, env.gamma, start, se, z).lower for z in (0.0, 1.0, 3.0)]
    uppers = [gap_bounds(q, policy, env.gamma, start, se, z).upper for z in (0.0, 1.0, 3.0)]
    assert lowers == sorted(lowers, reverse=True)
    assert uppers == sorted(uppers)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"gamma": 1.0}, "gamma"),
        ({"z": -1.0}, "z must"),
        ({"z": math.inf}, "z must"),
        ({"start": np.full(32, 0.5)}, "start"),
        ({"start": np.ones(31) / 31}, "start"),
        ({"q_se": np.full((32, 2), -0.1), "z": 1.0}, "q_se"),
        ({"q_se": np.full((32, 3), 0.1), "z": 1.0}, "q_se"),
        ({"q_se": np.full((32, 2), np.inf), "z": 1.0}, "q_se"),
    ],
)
def test_gap_bounds_validation(kwargs: dict[str, object], error: str) -> None:
    args: dict[str, object] = {
        "q": np.zeros((32, 2)),
        "policy": np.full((32, 2), 0.5),
        "gamma": 0.9,
        "start": np.eye(32)[0],
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=error):
        gap_bounds(**args)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- rollout estimator


@pytest.mark.parametrize("name", ["blackjack", "gridworld10", "inventory"])
def test_rollout_q_matches_exact_q(name: str) -> None:
    env, scorer = get_env(name), get_scorer(name)
    policy = random_policy(env, 4, floor=0.5)
    q, se, steps = RolloutCertifier(n_rollouts=50, n_start_rollouts=500).rollout_q(
        env, policy, np.random.default_rng(5)
    )
    assert steps > 0
    assert np.all(np.abs(q - scorer.q_values(policy)) <= 5 * se + 1e-9)
    assert np.all(se[env.start] < se.max())  # many more rollouts at the start state


@pytest.mark.parametrize("name", CERTIFIED_PROBLEMS)
def test_certificate_brackets_the_true_gap(name: str) -> None:
    env, scorer = get_env(name), get_scorer(name)
    for seed in (6, 7):
        policy = random_policy(env, seed, floor=0.5)
        certificate = RolloutCertifier().certify(env, policy, np.random.default_rng(seed))
        assert 0 <= certificate.lower <= scorer.gap(policy)
        assert scorer.gap(policy) <= certificate.upper
        assert certificate.observations > 0


def test_certificate_is_deterministic_given_the_seed() -> None:
    env = get_env("gridworld10")
    policy = random_policy(env, 8)
    a = RolloutCertifier().certify(env, policy, np.random.default_rng(9))
    b = RolloutCertifier().certify(env, policy, np.random.default_rng(9))
    assert a == b


@pytest.mark.parametrize("kwargs", [{"n_rollouts": 1}, {"n_start_rollouts": 0}, {"z": -0.5}, {"z": math.nan}])
def test_certifier_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        RolloutCertifier(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- the TOMC caveat


def test_tomc_q_falsely_certifies_a_locked_in_policy() -> None:
    """'Draw below 7' is deterministic and suboptimal (the 0.194 gap seen in the Blackjack lock-in).

    TOMC fills the never-taken actions with 1 / (1 - gamma) = 10, so its Q-hat bounds read [0, 0]
    ("optimal"); rollouts that try every action detect the gap.
    """
    env, scorer = Blackjack(), get_scorer("blackjack")
    locked = threshold_policy(env, 7)
    assert scorer.gap(locked) == pytest.approx(0.1937, abs=1e-4)
    q_tomc = TOMC().estimate(env, locked, np.random.default_rng(0), 10**7).q
    false = gap_bounds(q_tomc, locked, env.gamma, env.start_distribution())
    assert (false.lower, false.upper) == (0.0, 0.0)
    certificate = RolloutCertifier().certify(env, locked, np.random.default_rng(0))
    assert 0.05 < certificate.lower <= scorer.gap(locked)


@pytest.mark.parametrize("draw_below", [9, 10])
def test_no_false_alarm_for_optimal_threshold_policies(draw_below: int) -> None:
    env, scorer = Blackjack(), get_scorer("blackjack")
    policy = threshold_policy(env, draw_below)
    assert scorer.gap(policy) < 1e-9
    assert RolloutCertifier().certify(env, policy, np.random.default_rng(1)).lower == 0.0


# ----------------------------------------------------------------------------- run_spmd hooks


def test_run_spmd_certifies_without_changing_training() -> None:
    env, scorer = get_env("blackjack"), get_scorer("blackjack")
    plain = run_spmd(env, TOMC(), KL(), 1.0, budget=30_000, seed=3, scorer=scorer)
    certified = run_spmd(
        env,
        TOMC(),
        KL(),
        1.0,
        budget=30_000,
        seed=3,
        scorer=scorer,
        certifier=RolloutCertifier(),
        certify_every=10,
    )
    np.testing.assert_array_equal(plain.policy, certified.policy)
    assert [h.exact_value for h in plain.history] == [h.exact_value for h in certified.history]
    assert [h.cumulative_observations for h in plain.history] == [
        h.cumulative_observations for h in certified.history
    ]
    for h in certified.history:
        checked = h is certified.history[-1] or (h.status == "updated" and h.iteration % 10 == 0)
        assert math.isnan(h.gap_lower) != checked
        if checked:
            assert 0 <= h.gap_lower <= h.gap + 1e-9 <= h.gap_upper + 2e-9
            assert h.validation_observations > 0
    assert all(math.isnan(h.gap_lower) for h in plain.history)


def test_run_spmd_certifies_only_the_final_policy_by_default() -> None:
    env = get_env("blackjack")
    history = run_spmd(env, TOMC(), KL(), 1.0, budget=10_000, seed=0, certifier=RolloutCertifier()).history
    assert [not math.isnan(h.gap_lower) for h in history] == [False] * (len(history) - 1) + [True]


def test_run_spmd_rejects_negative_certify_every() -> None:
    with pytest.raises(ValueError, match="certify_every"):
        run_spmd(get_env("blackjack"), TOMC(), KL(), 1.0, budget=100, seed=0, certify_every=-1)


# ----------------------------------------------------------------------------- policy-improvement bound


@pytest.mark.parametrize("name", CERTIFIED_PROBLEMS)
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_improvement_bound_sits_between_g_and_the_true_gap(name: str, seed: int) -> None:
    """g(s0) <= V^pi(s0) - V^greedy(s0) <= V^pi(s0) - V*(s0) (performance-difference lemma)."""
    env, scorer = get_env(name), get_scorer(name)
    policy = random_policy(env, seed, floor=0.4)
    q = scorer.q_values(policy)
    greedy = np.eye(env.n_actions)[q.argmin(axis=1)]
    rho = env.start_distribution()
    improvement = float(rho @ (scorer.values(policy) - scorer.values(greedy)))
    assert gap_bounds(q, policy, env.gamma, rho).lower <= improvement + 1e-9
    assert improvement <= scorer.gap(policy) + 1e-9


@pytest.mark.parametrize("name", ["gridworld10", "gridworld30", "inventory"])
def test_improvement_bound_is_much_tighter_on_long_horizons(name: str) -> None:
    env, scorer = get_env(name), get_scorer(name)
    policy = random_policy(env, 10, floor=0.5)
    certificate = RolloutCertifier().certify(env, policy, np.random.default_rng(11))
    assert certificate.improvement_lower <= scorer.gap(policy)
    assert certificate.improvement_lower > 3 * certificate.gap_lower
    assert certificate.improvement_lower > 0.4 * scorer.gap(policy)
    assert certificate.lower == max(certificate.gap_lower, certificate.improvement_lower)


def test_improvement_bound_against_itself_is_zero() -> None:
    env = get_env("gridworld10")
    policy = random_policy(env, 12)
    lower, steps = RolloutCertifier().improvement_lower(env, policy, policy, np.random.default_rng(13))
    assert lower == 0.0
    assert steps > 0
    with pytest.raises(ValueError, match="probabilities"):
        RolloutCertifier().improvement_lower(env, policy, np.full_like(policy, 0.3), np.random.default_rng(0))
