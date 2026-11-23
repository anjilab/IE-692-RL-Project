"""Performance and 30x30 checks (opt-in: ``pytest -m perf`` and ``pytest -m slow``).

The perf tests assert the throughput the 10-minute budget relies on and run the real pipeline;
``benchmarks/bench.py`` prints the full report.
"""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import tune
from conftest import get_env, get_scorer
from ie692_rl.core import TOMC, BudgetExhaustedError, run_spmd, uniform_policy
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.enhancements.true_online_td import TrueOnlineTD

TEN_MINUTES = 600.0


def steps_per_second(run: Callable[[int], int], seconds: float = 1.0) -> float:
    run(0)  # compile / load the cached kernels
    steps, started, seed = 0, time.perf_counter(), 1
    while time.perf_counter() - started < seconds:
        steps += run(seed)
        seed += 1
    return steps / (time.perf_counter() - started)


@pytest.mark.perf
@pytest.mark.parametrize("name", ["gridworld10", "gridworld30", "inventory"])
def test_tomc_throughput(name: str) -> None:
    env = get_env(name)
    policy = uniform_policy(env.n_states, env.n_actions)

    def run(seed: int) -> int:
        try:
            return TOMC().estimate(env, policy, np.random.default_rng(seed), 2_000_000).observations
        except BudgetExhaustedError:
            return 2_000_000

    assert steps_per_second(run) > 5e6  # pure Python managed about 0.1M steps/s


@pytest.mark.perf
@pytest.mark.parametrize("name", ["gridworld10", "gridworld30", "blackjack", "inventory"])
def test_true_online_td_throughput(name: str) -> None:
    env = get_env(name)
    policy, estimator = uniform_policy(env.n_states, env.n_actions), TrueOnlineTD(n_steps=200_000)
    assert (
        steps_per_second(
            lambda seed: estimator.estimate(env, policy, np.random.default_rng(seed), 10**6).observations
        )
        > 2e6
    )


@pytest.mark.perf
def test_default_pipeline_finishes_within_ten_minutes(tmp_path: Path) -> None:
    started = time.perf_counter()
    summary = tune.main(["--output", str(tmp_path)])
    elapsed = time.perf_counter() - started
    assert elapsed < TEN_MINUTES, f"{elapsed:.0f} s"
    assert {r["problem"] for r in summary} == {"GridWorld", "Blackjack"}


def test_tomc_cannot_cover_gridworld30_under_the_uniform_policy() -> None:
    """Documents the 30x30 limitation: 10^7 steps leave many required pairs unvisited (see README)."""
    env = get_env("gridworld30")
    with pytest.raises(BudgetExhaustedError, match=r"TOMC incomplete: \d+/3600"):
        TOMC().estimate(env, uniform_policy(900, 4), np.random.default_rng(0), 10_000_000)


@pytest.mark.slow
def test_true_online_td_spmd_improves_gridworld30() -> None:
    env, scorer = get_env("gridworld30"), get_scorer("gridworld30")
    result = run_spmd(
        env,
        TrueOnlineTD(eta=0.5, lam=0.5, n_steps=50_000),
        KL(),
        1.0,
        budget=20_000_000,
        seed=0,
        scorer=scorer,
    )
    assert result.history[-1].status in ("updated", "budget")
    assert result.history[-1].gap < 0.5 * result.history[0].gap
