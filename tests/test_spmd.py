"""The SPMD loop: bookkeeping, budgets, dependency injection, and convergence with exact Q."""

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from conftest import get_env, get_scorer
from ie692_rl.core import TOMC, BregmanDivergence, Estimate, QEstimator, as_schedule, run_spmd
from ie692_rl.core.estimator import BudgetExhaustedError
from ie692_rl.core.spmd import IterationRecord
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.envs import Environment
from ie692_rl.types import FloatArray


@dataclass
class ExactQ(QEstimator):
    """Oracle estimator: exact Q of the policy, charged a fixed number of observations."""

    name: str
    cost: int = 100
    calls: list[FloatArray | None] = field(default_factory=list)

    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        self.calls.append(extra_cost)
        if max_observations < self.cost:
            raise BudgetExhaustedError("oracle budget")
        return Estimate(get_scorer(self.name).q_values(policy), self.cost, env.n_states)


@dataclass
class SpyDivergence(KL):
    steps: list[tuple[float, float]] = field(default_factory=list)

    def step(self, policy: FloatArray, q: FloatArray, eta: float, tau: float = 0.0) -> FloatArray:
        self.steps.append((eta, tau))
        return super().step(policy, q, eta, tau)


class NaNEstimator(QEstimator):
    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        return Estimate(np.full((env.n_states, env.n_actions), np.nan), 10, 0)


def test_as_schedule() -> None:
    assert as_schedule(2.5)(0) == 2.5
    assert as_schedule(2.5)(1000) == 2.5
    assert as_schedule(lambda k: 1.0 / (k + 1))(3) == 0.25


def test_history_bookkeeping_with_tomc() -> None:
    env, scorer = get_env("blackjack"), get_scorer("blackjack")
    result = run_spmd(env, TOMC(), KL(), 10.0, budget=20_000, seed=0, scorer=scorer)
    history = result.history
    assert history[0].iteration == 0
    assert history[0].status == "start"
    assert history[0].exact_value == pytest.approx(scorer.score(np.full((32, 2), 0.5)))
    assert math.isnan(history[0].estimated_value)
    cumulative = [h.cumulative_observations for h in history]
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] <= 20_000
    for previous, current in itertools.pairwise(history):
        assert current.cumulative_observations - previous.cumulative_observations == current.observations
        assert current.iteration == previous.iteration + 1
    updated = [h for h in history if h.status == "updated"]
    assert len(updated) > 5
    assert all(h.observations > 0 and np.isfinite(h.estimated_value) for h in updated)
    assert history[-1].gap == pytest.approx(scorer.gap(result.policy))
    assert history[-1].status in ("updated", "budget")
    assert all(0 <= h.min_prob <= 0.5 for h in history)


def test_deterministic_and_seed_dependent() -> None:
    env = get_env("blackjack")
    a = run_spmd(env, TOMC(), KL(), 10.0, budget=5000, seed=1)
    b = run_spmd(env, TOMC(), KL(), 10.0, budget=5000, seed=1)
    c = run_spmd(env, TOMC(), KL(), 10.0, budget=5000, seed=2)
    np.testing.assert_array_equal(a.policy, b.policy)
    assert [h.cumulative_observations for h in a.history] != [h.cumulative_observations for h in c.history]


def test_zero_budget_and_budget_exhaustion() -> None:
    env = get_env("gridworld10")
    assert len(run_spmd(env, TOMC(), KL(), 1.0, budget=0, seed=0).history) == 1
    result = run_spmd(env, TOMC(), KL(), 1.0, budget=1000, seed=0)  # far below one estimate
    assert [h.status for h in result.history] == ["start", "budget"]
    assert result.history[-1].cumulative_observations == 1000
    np.testing.assert_allclose(result.policy, 0.25)
    with pytest.raises(ValueError, match="budget"):
        run_spmd(env, TOMC(), KL(), 1.0, budget=-1, seed=0)


def test_max_iterations() -> None:
    oracle = ExactQ("blackjack")
    result = run_spmd(get_env("blackjack"), oracle, KL(), 1.0, budget=10**9, seed=0, max_iterations=7)
    assert len(result.history) == 8


@pytest.mark.parametrize("step_size", [0.0, -1.0, math.nan, math.inf])
def test_invalid_step_size(step_size: float) -> None:
    with pytest.raises(ValueError, match="Step size"):
        run_spmd(get_env("blackjack"), ExactQ("blackjack"), KL(), step_size, budget=1000, seed=0)


def test_invalid_step_size_schedule_is_caught_when_reached() -> None:
    oracle = ExactQ("blackjack")
    with pytest.raises(ValueError, match="iteration 2"):
        run_spmd(get_env("blackjack"), oracle, KL(), lambda k: 1.0 if k < 2 else 0.0, budget=10**6, seed=0)
    assert len(oracle.calls) == 2


def test_dependency_injection_passes_step_size_tau_and_entropy_cost() -> None:
    env, oracle, spy = get_env("blackjack"), ExactQ("blackjack"), SpyDivergence()
    run_spmd(env, oracle, spy, lambda k: 1.0 + k, budget=500, seed=0, tau=lambda k: 0.1 / (k + 1))
    assert spy.steps == [(1.0, 0.1), (2.0, 0.05), (3.0, 0.1 / 3), (4.0, 0.025), (5.0, 0.02)]
    first = oracle.calls[0]
    assert first is not None
    np.testing.assert_allclose(first, 0.1 * -math.log(2))  # tau * sum pi log pi for the uniform policy


def test_no_entropy_cost_without_tau() -> None:
    oracle = ExactQ("blackjack")
    run_spmd(get_env("blackjack"), oracle, KL(), 1.0, budget=300, seed=0)
    assert oracle.calls == [None, None, None]


def test_non_finite_estimates_stop_the_run() -> None:
    result = run_spmd(get_env("blackjack"), NaNEstimator(), KL(), 1.0, budget=1000, seed=0)
    assert [h.status for h in result.history] == ["start", "diverged"]
    np.testing.assert_allclose(result.policy, 0.5)


@pytest.mark.parametrize("name", ["blackjack", "gridworld10", "inventory"])
def test_kl_spmd_with_exact_q_converges_to_optimal(name: str) -> None:
    """With exact Q, KL-PMD with a large constant step gets near-optimal and never gets worse.

    Constant-step PMD converges at O(1/k) once only near-ties remain (their advantage is ~1e-8, so their
    probability grows by ~1e-6 per step), hence the 1e-4 tolerance after 200 iterations."""
    env, scorer = get_env(name), get_scorer(name)
    eta = 100.0 if name != "inventory" else 10.0
    result = run_spmd(env, ExactQ(name), KL(), eta, budget=10**9, seed=0, scorer=scorer, max_iterations=200)
    assert result.history[-1].gap < 1e-4
    gaps = [h.gap for h in result.history]
    assert all(b <= a + 1e-9 for a, b in itertools.pairwise(gaps))  # exact PMD never gets worse


def test_custom_divergence_is_used() -> None:
    class Frozen(BregmanDivergence):
        name = "frozen"

        def divergence(self, p: FloatArray, center: FloatArray) -> FloatArray:
            return np.zeros(len(p))

        def step(self, policy: FloatArray, q: FloatArray, eta: float, tau: float = 0.0) -> FloatArray:
            return policy

    result = run_spmd(get_env("blackjack"), ExactQ("blackjack"), Frozen(), 1.0, budget=1000, seed=0)
    np.testing.assert_allclose(result.policy, 0.5)
    assert len(result.history) == 11


def test_records_serialise() -> None:
    record = IterationRecord(1, 2, 3, 0.5, 0.4, 0.1, 0.2, 7, 1.0, 0.0, "updated")
    assert record.as_dict()["status"] == "updated"
    assert set(record.as_dict()) >= {"iteration", "observations", "estimated_value", "gap"}
