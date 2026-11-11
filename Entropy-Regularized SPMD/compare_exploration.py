"""Does entropy regularization help SPMD? Compare update rules across problems and sampling.

Problems:  the team's GridWorld and blackjack, plus the harder DeceptiveGridWorld and
           FullBlackjack from hard_problems.py.
Sampling:  generative (team's: rollouts from every state/action) and online (episodes
           from the start state(s) only); see estimators.py.
Methods:   the team's Euclidean update, and the KL update from pmd.py with entropy
           weight tau in TAUS (tau = 0 is plain KL).

Protocol, the same for every method:
  1. Tuning: try each step size on TUNING_SEEDS; keep the one with the lowest mean cost.
  2. Testing: rerun with that step size on fresh TEST_SEEDS.
  Every policy is scored exactly from the model (exact_eval.py), so the numbers have no
  scoring noise. Reported cost is the gap to the optimal cost (0 = optimal).
  Each run stops once its training budget is used up (the last iteration may overshoot
  by less than one iteration's samples).

Run from anywhere (about 10 minutes):
    python "Entropy-Regularized SPMD/compare_exploration.py"
    python "Entropy-Regularized SPMD/compare_exploration.py" --quick   # 1-minute smoke test
Larger GridWorlds (about 15 minutes):
    python "Entropy-Regularized SPMD/compare_exploration.py" --problems GridWorld10x10 GridWorld20x20 \
        GridWorld30x30 --taus 0 0.001 0.01 0.03 --output larger_gridworld_results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import kl_spmd  # first: puts the team's code on the import path
import utils
from baseline import GridWorld
from baseline_blackjack import Blackjack
from estimators import generative_q, online_q
from exact_eval import ExactScorer
from hard_problems import DeceptiveGridWorld, FullBlackjack, make_scaled_gridworld
from pmd import run_pmd

TAUS = [0.0, 0.001, 0.01, 0.03, 0.1]
EUCLIDEAN_STEP_SIZES = [0.1, 0.5, 1.0]
KL_STEP_SIZES = [1.0, 5.0, 20.0]
TUNING_SEEDS = [100, 101, 102]
TEST_SEEDS = [200, 201, 202, 203, 204]
CHECKPOINTS = [0.25, 0.5, 1.0]  # fractions of the budget at which policies are scored

# name: (environment, {sampling: (estimator, estimator arguments, training budget in steps)})
# Generative settings follow the team's selected rollouts and horizons where they exist.
PROBLEMS = {
    "GridWorld": (GridWorld(), {
        "generative": (generative_q, (5, 40), 24_000_000),
        "online": (online_q, (100, 80), 2_400_000)}),
    "Blackjack": (Blackjack(), {
        "generative": (generative_q, (5, 4), 120_000),
        "online": (online_q, (200, 8), 120_000)}),
    "DeceptiveGridWorld": (DeceptiveGridWorld(), {
        "generative": (generative_q, (5, 40), 24_000_000),
        "online": (online_q, (100, 80), 1_000_000)}),
    "FullBlackjack": (FullBlackjack(), {
        "generative": (generative_q, (5, 8), 600_000),
        "online": (online_q, (500, 10), 200_000)}),
}
DEFAULT_PROBLEMS = list(PROBLEMS)

# Larger boards with the team's rules and densities, random start squares (hard_problems.py).
# Generative: 100 full sweeps at every size, so each state gets the same amount of data.
# Online: the same 4.8M steps at every size, so bigger boards get less data per state.
for _size in (10, 20, 30):
    PROBLEMS[f"GridWorld{_size}x{_size}"] = (make_scaled_gridworld(_size), {
        "generative": (generative_q, (5, 40), 100 * _size * _size * 4 * 5 * 40),
        "online": (online_q, (200, 80), 4_800_000)})


def methods():
    """Yield (name, tau, step sizes, update rule). Updates take cost-valued Q."""
    yield "Euclidean", 0.0, EUCLIDEAN_STEP_SIZES, utils.mirror_step
    for tau in TAUS:
        yield f"KL tau={tau:g}", tau, KL_STEP_SIZES, lambda p, q, step, tau=tau: run_pmd(p, -q, step, tau)


def train(env, estimator, estimator_args, budget, update, step_size, tau, seed, scorer):
    """Run SPMD until the budget is spent; return the exact cost at each checkpoint."""
    rng = np.random.default_rng(seed)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    used, scores, targets = 0, [], [fraction * budget for fraction in CHECKPOINTS]
    while targets:
        sampled_env = kl_spmd.EntropyCostEnv(env, policy, tau) if tau > 0 else env
        q, count = estimator(sampled_env, policy, *estimator_args, rng)
        used += count
        policy = update(policy, q, step_size)
        while targets and used >= targets[0]:
            scores.append(scorer.score(policy))
            targets.pop(0)
    return scores


def run_setting(problem, sampling, env, scorer, estimator, args, budget):
    rows = []
    for method, tau, step_sizes, update in methods():
        tuning = {step: np.mean([train(env, estimator, args, budget, update, step, tau, seed, scorer)[-1]
                                 for seed in TUNING_SEEDS]) for step in step_sizes}
        best_step = min(tuning, key=tuning.get)
        curves = np.array([train(env, estimator, args, budget, update, best_step, tau, seed, scorer)
                           for seed in TEST_SEEDS]) - scorer.optimal
        final = curves[:, -1]
        rows.append(dict(problem=problem, sampling=sampling, method=method, tau=tau, step_size=best_step,
                         optimal_cost=round(scorer.optimal, 6),
                         **{f"mean_gap_at_{int(100 * f)}pct": round(float(m), 6) for f, m in zip(CHECKPOINTS, curves.mean(0))},
                         final_gap_sd=round(float(final.std(ddof=1)), 6), final_gap_worst=round(float(final.max()), 6)))
        print(f"  {method:14s} step={best_step:<5g} gap at 25/50/100% budget: "
              + " / ".join(f"{m:.4f}" for m in curves.mean(0))
              + f"   (final sd {final.std(ddof=1):.4f}, worst {final.max():.4f})", flush=True)
    return rows


def main():
    global TUNING_SEEDS, TEST_SEEDS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "exploration_results.csv")
    parser.add_argument("--quick", action="store_true", help="1 tuning seed, 2 test seeds, 10%% budgets")
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=DEFAULT_PROBLEMS)
    parser.add_argument("--taus", nargs="+", type=float, default=TAUS, help="entropy weights for KL")
    args = parser.parse_args()
    TAUS[:] = args.taus
    if args.output.exists():
        parser.error(f"{args.output} already exists; choose a new --output")
    scale = 0.1 if args.quick else 1.0
    if args.quick:
        TUNING_SEEDS, TEST_SEEDS = TUNING_SEEDS[:1], TEST_SEEDS[:2]

    rows = []
    for problem in args.problems:
        env, samplings = PROBLEMS[problem]
        scorer = ExactScorer(env)
        for sampling, (estimator, estimator_args, budget) in samplings.items():
            print(f"\n{problem}, {sampling} sampling (optimal cost {scorer.optimal:.4f}, budget {int(budget * scale):,} steps)")
            rows += run_setting(problem, sampling, env, scorer, estimator, estimator_args, int(budget * scale))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
