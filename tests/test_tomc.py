"""Baseline TOMC estimator: equivalence with the original algorithm, unbiasedness, budgets, memory."""

import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import get_env, get_scorer, random_policy
from ie692_rl.core import TOMC, BudgetExhaustedError, uniform_policy
from ie692_rl.core.tomc import RETURN_PRECISION, return_window, tail_steps
from reference import reference_tomc


@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("bias", [1e-300, 1e-6, 0.003, 0.01, 0.03, 1.0, 1e6])
def test_tail_steps_is_the_smallest_sufficient_tail(gamma: float, bias: float) -> None:
    m, target = tail_steps(gamma, bias), bias * (1 - gamma)
    assert gamma**m <= target or m == 0
    assert m == 0 or gamma ** (m - 1) > target


@pytest.mark.parametrize(
    ("gamma", "bias"), [(1.0, 0.01), (0.0, 0.01), (0.9, 0.0), (0.9, -1.0), (0.9, 5e-324)]
)
def test_tail_steps_rejects_degenerate_inputs(gamma: float, bias: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        tail_steps(gamma, bias)


def test_return_window_reaches_double_precision() -> None:
    for gamma in (0.5, 0.9, 0.99):
        w = return_window(gamma)
        assert gamma**w <= RETURN_PRECISION
        assert w < 10_000


@pytest.mark.parametrize("bias", [0.0, -0.1, math.nan, math.inf])
def test_invalid_bias(bias: float) -> None:
    with pytest.raises(ValueError, match="bias"):
        TOMC(bias=bias)


@pytest.mark.parametrize("pi_min", [0.0, -0.5, 1.5])
def test_invalid_pi_min(pi_min: float) -> None:
    with pytest.raises(ValueError, match="pi_min"):
        TOMC(pi_min=pi_min)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("name", ["blackjack", "gridworld10", "inventory"])
def test_matches_original_algorithm(name: str, seed: int) -> None:
    """Same random stream as the original stored-trajectory TOMC: identical length, Q equal to rounding."""
    env = get_env(name)
    policy = random_policy(env, seed, floor=0.5)
    extra = None if seed == 0 else np.random.default_rng(seed).uniform(-0.2, 0.0, env.n_states)
    expected_q, expected_n = reference_tomc(env, policy, np.random.default_rng(seed), 0.01, 50_000_000, extra)
    estimate = TOMC(bias=0.01).estimate(env, policy, np.random.default_rng(seed), 50_000_000, extra)
    assert estimate.observations == expected_n
    np.testing.assert_allclose(estimate.q, expected_q, rtol=0, atol=1e-10)


def test_deterministic_given_seed() -> None:
    env = get_env("gridworld10")
    policy = uniform_policy(env.n_states, env.n_actions)
    a = TOMC().estimate(env, policy, np.random.default_rng(7), 10**8)
    b = TOMC().estimate(env, policy, np.random.default_rng(7), 10**8)
    np.testing.assert_array_equal(a.q, b.q)
    assert a.observations == b.observations


@pytest.mark.parametrize("name", ["blackjack", "gridworld10"])
def test_unbiased_for_required_pairs(name: str) -> None:
    """Averaged over independent estimates, Q matches the exact Q within 5 SE plus the truncation bias."""
    env, scorer = get_env(name), get_scorer(name)
    policy = random_policy(env, 3, floor=0.6)
    exact = scorer.q_values(policy)
    runs = 400 if name == "blackjack" else 40
    errors = np.array(
        [
            TOMC(bias=0.01).estimate(env, policy, np.random.default_rng(100 + r), 10**8).q - exact
            for r in range(runs)
        ]
    )
    se = errors.std(axis=0, ddof=1) / np.sqrt(runs)
    assert np.all(np.abs(errors.mean(axis=0)) <= 5 * se + 0.01 * env.cost_bound + 1e-9)


def test_unrequired_pairs_get_pessimistic_fill() -> None:
    env = get_env("gridworld10")
    policy = np.tile(
        [0.35, 0.35, 0.15, 0.15], (env.n_states, 1)
    )  # with pi_min = 0.3 only up/down are required
    estimate = TOMC(pi_min=0.3).estimate(env, policy, np.random.default_rng(0), 10**8)
    np.testing.assert_allclose(estimate.q[:, 2:], env.cost_bound / (1 - env.gamma))
    assert np.all(estimate.q[:, :2] < env.cost_bound / (1 - env.gamma))


def test_no_required_pairs() -> None:
    env = get_env("blackjack")
    with pytest.raises(ValueError, match="pi_min"):
        TOMC(pi_min=0.9).estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 1000)


def test_budget_exhausted() -> None:
    env = get_env("gridworld10")
    policy = uniform_policy(env.n_states, env.n_actions)
    with pytest.raises(BudgetExhaustedError, match="TOMC incomplete"):
        TOMC().estimate(env, policy, np.random.default_rng(0), 1000)
    with pytest.raises(BudgetExhaustedError):
        TOMC().estimate(env, policy, np.random.default_rng(0), 0)


def test_rejects_invalid_policy_and_extra_cost() -> None:
    env = get_env("blackjack")
    with pytest.raises(ValueError, match="probabilities"):
        TOMC().estimate(env, np.full((32, 2), 0.7), np.random.default_rng(0), 1000)
    with pytest.raises(ValueError, match="extra_cost"):
        TOMC().estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 1000, np.full(31, 0.0))
    with pytest.raises(ValueError, match="extra_cost"):
        TOMC().estimate(env, uniform_policy(32, 2), np.random.default_rng(0), 1000, np.full(32, np.nan))


def test_extra_cost_is_added_to_every_step() -> None:
    """Same stream: a constant extra cost c shifts each continuing-task return by about c / (1 - gamma)."""
    env = get_env("gridworld10")
    policy = uniform_policy(env.n_states, env.n_actions)
    plain = TOMC().estimate(env, policy, np.random.default_rng(4), 10**8)
    shifted = TOMC().estimate(env, policy, np.random.default_rng(4), 10**8, np.full(env.n_states, -0.05))
    assert plain.observations == shifted.observations
    # Each return is truncated at least `tail` steps after its first visit, so the shift is at most
    # 0.05 / (1 - gamma) and at least that minus 0.01 * 0.05 / (1 - gamma).
    shift = plain.q - shifted.q
    assert np.all(shift <= 0.5 + 1e-9)
    assert np.all(shift >= 0.5 * (1 - 0.01) - 1e-9)


def test_reports_full_coverage() -> None:
    env = get_env("gridworld10")
    estimate = TOMC().estimate(
        env, uniform_policy(env.n_states, env.n_actions), np.random.default_rng(0), 10**8
    )
    assert estimate.states_visited == env.n_states


PEAK_RSS_SCRIPT = """
import resource, sys
import numpy as np
from ie692_rl.core import TOMC, BudgetExhaustedError, uniform_policy
from ie692_rl.envs import GridWorld
env = GridWorld(size=30)
policy = uniform_policy(env.n_states, env.n_actions)
try:
    TOMC().estimate(env, policy, np.random.default_rng(0), int(sys.argv[1]))
except BudgetExhaustedError:
    pass
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""


def peak_rss_bytes(steps: int) -> int:
    """Peak resident memory of a fresh process that runs one TOMC stream of ``steps`` transitions."""
    out = subprocess.run(
        [sys.executable, "-c", PEAK_RSS_SCRIPT, str(steps)], capture_output=True, text=True, check=True
    ).stdout
    rss = int(out.strip().splitlines()[-1])
    return rss if sys.platform == "darwin" else rss * 1024  # ru_maxrss: bytes on macOS, kB on Linux


def test_memory_does_not_grow_with_stream_length() -> None:
    """The original stored every transition (~100+ B each, so ~5 GB for 50M steps). The ring buffer keeps
    memory O(|S||A|): a 100x longer stream on the 30x30 grid must not raise the peak by more than 20 MB."""
    short, long = peak_rss_bytes(500_000), peak_rss_bytes(50_000_000)
    assert long - short < 20_000_000, f"{(long - short) / 1e6:.1f} MB more for 100x the steps"
