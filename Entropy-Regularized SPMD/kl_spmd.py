"""Shared pieces for running the team's SPMD experiments with the KL update from pmd.py.

Only two things change from the team baseline:
  1. The policy update is the KL step from pmd.py instead of the Euclidean step.
  2. With tau > 0, training rollouts add tau * (negative entropy of the policy) to the
     cost of every step. Random policies then cost less, which encourages exploration.
Scoring always uses the environment's real cost, so results compare directly with the
team's baseline results. The scripts gridworld_kl_spmd.py and blackjack_kl_spmd.py use this.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # team code: utils.py, baseline.py, baseline_blackjack.py

import utils
from pmd import run_pmd

team_monte_carlo_q = utils.monte_carlo_q


class EntropyCostEnv:
    """Wraps an environment so each step also costs tau * sum_a pi(a|s) log pi(a|s).

    That sum is the negative entropy of the policy at the current state, so it is at
    most 0: the more random the policy is there, the lower the cost.
    """

    def __init__(self, env, policy, tau):
        self.env = env
        self.gamma = env.gamma
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.entropy_cost = tau * np.sum(policy * np.log(policy + 1e-12), axis=1)

    def step(self, states, actions, rng):
        next_states, costs, done = self.env.step(states, actions, rng)
        return next_states, costs + self.entropy_cost[states], done

    def initial_states(self, count, rng):
        return self.env.initial_states(count, rng)


def use_kl_update(tau):
    """Make the team's train() use the (entropy-regularized) KL step and Q estimates."""

    def kl_mirror_step(policy, q, step_size):
        # The team's Q holds costs, while run_pmd maximizes, so negate Q.
        return run_pmd(policy, -q, step_size, tau)

    def regularized_monte_carlo_q(env, policy, rollouts, horizon, rng):
        return team_monte_carlo_q(EntropyCostEnv(env, policy, tau), policy, rollouts, horizon, rng)

    # train() looks these names up on every iteration, so replacing them swaps the method.
    utils.mirror_step = kl_mirror_step
    utils.monte_carlo_q = regularized_monte_carlo_q if tau > 0 else team_monte_carlo_q


def prepare_arguments(argv, default_tau, step_sizes, name):
    """Read --tau (unknown to the team's parser) and fill in KL defaults; return tau.

    Removes --tau from argv, then adds the KL step sizes and an output folder in this
    directory unless the command line already gives them.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tau", type=float, default=default_tau)
    args, rest = parser.parse_known_args(argv[1:])
    if not args.tau >= 0:
        sys.exit("--tau must be non-negative")
    argv[1:] = rest
    if "--step-sizes" not in argv:
        argv += ["--step-sizes", *map(str, step_sizes)]
    if "--output" not in argv:
        argv += ["--output", str(HERE / f"{name}_kl_tau_{args.tau:g}_results")]
    return args.tau
