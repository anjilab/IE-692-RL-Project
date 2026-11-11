"""Exact (noise-free) policy scoring from each environment's known rules.

exact_model(env) returns P (S, A, S) and expected one-step cost c (S, A). Rows of P
sum to less than 1 where an episode can end. The team's GridWorld and blackjack get
models built here from the rules in baseline.py and baseline_blackjack.py; the hard
problems provide their own exact_model() method.
"""
import numpy as np

import kl_spmd  # noqa: F401  puts the team's code on the import path
from baseline import GridWorld
from baseline_blackjack import Blackjack


def gridworld_model(env):
    S, C = env.n_states, env.cols
    P, c = np.zeros((S, 4, S)), np.zeros((S, 4))
    for s in range(S):
        special = s in env.goals or s in env.hazards
        c[s] = 0.0 if s in env.goals else 1.0 if s in env.hazards else 0.2
        for a in range(4):
            for actual, (dr, dc) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                prob = env.intended_probability if actual == a else (1 - env.intended_probability) / 3
                row = min(max(s // C + dr, 0), env.rows - 1)
                col = min(max(s % C + dc, 0), env.cols - 1)
                P[s, a, row * C + col] += prob * ((1 - env.reset_probability) if special else 1.0)
            if special:
                P[s, a, env.start] += env.reset_probability
    return P, c


def blackjack_model(env):
    S = env.n_states
    P, c = np.zeros((S, 2, S)), np.zeros((S, 2))

    def dealer_final(visible):
        dist, final = {visible: 1.0}, {}
        while dist:
            new = {}
            for total, prob in dist.items():
                for card in range(2, 6):
                    t = total + card
                    target = final if t >= 9 else new
                    target[t] = target.get(t, 0.0) + prob / 4
            dist = new
        return final

    for s in range(S):
        total, visible = map(int, env.decode(s))
        c[s, 0] = sum(p * (-1.0 if d > 11 else np.sign(d - total)) for d, p in dealer_final(visible).items())
        for card in range(2, 6):
            if total + card > 11:
                c[s, 1] += 0.25
            else:
                P[s, 1, int(env.encode(total + card, visible))] += 0.25
    return P, c


def exact_model(env):
    if isinstance(env, GridWorld):
        return gridworld_model(env)
    if isinstance(env, Blackjack):
        return blackjack_model(env)
    return env.exact_model()


def start_distribution(env):
    if hasattr(env, "start_distribution"):
        return env.start_distribution()
    weights = np.zeros(env.n_states)
    weights[env.initial_states(1, np.random.default_rng(0))[0]] = 1.0  # fixed start state
    return weights


class ExactScorer:
    """Caches an environment's model so many policies can be scored quickly."""

    def __init__(self, env):
        self.gamma = env.gamma
        self.P, self.c = exact_model(env)
        self.start = start_distribution(env)
        self.optimal = self.start @ self.optimal_values()

    def values(self, policy):
        P_pi = np.einsum("sa,sat->st", policy, self.P)
        c_pi = (policy * self.c).sum(axis=1)
        return np.linalg.solve(np.eye(len(c_pi)) - self.gamma * P_pi, c_pi)

    def q_values(self, policy):
        return self.c + self.gamma * self.P @ self.values(policy)

    def optimal_values(self, iterations=3000):
        v = np.zeros(len(self.c))
        for _ in range(iterations):
            v = (self.c + self.gamma * self.P @ v).min(axis=1)
        return v

    def score(self, policy):
        """Expected discounted cost from the start distribution (lower is better)."""
        return float(self.start @ self.values(policy))
