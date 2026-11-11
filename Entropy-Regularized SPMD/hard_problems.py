"""Harder GridWorld and blackjack problems for testing entropy-regularized SPMD.

Both classes follow the team's environment interface (n_states, n_actions, gamma,
step, initial_states, state_label, action_names), so the team's Monte Carlo code
and the online estimator in estimators.py can sample from them. Each also provides
exact_model(), which exact_eval.py uses to score policies without sampling noise.

DeceptiveGridWorld
    12 x 12 board, start at the top-left corner. A "decoy" exit two squares to the
    right ends the episode with a lump cost of 1; the real goal at row 3, column 3
    ends it for free but is six moves away, with two hazards blocking the approach
    from above. Every step's cost also gets Gaussian noise (mean 0, sd 1), which makes
    Q estimates noisy. Exactly: the best policy reaches the goal 99% of the time at
    cost 1.198, while settling for the decoy costs 1.334 and the uniform policy 1.840.
    A random walk from the start finds the decoy far more often than the goal.

FullBlackjack
    Blackjack closer to the real game: cards 2-10 with 10 four times as likely (no
    aces), bust above 21, dealer draws until 17. 162 decision states instead of 32,
    and each hand starts from a random deal rather than one fixed hand.

Costs, as in the team code: lower is better.
"""
from dataclasses import dataclass

import numpy as np

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right


@dataclass(frozen=True)
class DeceptiveGridWorld:
    rows: int = 12
    cols: int = 12
    start: int = 0
    goal: int = 3 * 12 + 3
    decoy: int = 2
    decoy_cost: float = 1.0
    hazards: tuple = (2 * 12 + 2, 2 * 12 + 3)  # directly above the goal
    cost_noise: float = 1.0
    gamma: float = 0.9
    intended_probability: float = 0.8
    reset_probability: float = 0.8

    action_names = ("up", "down", "left", "right")
    n_actions = 4

    @property
    def n_states(self):
        return self.rows * self.cols

    def terminal(self, states):
        return (states == self.goal) | (states == self.decoy)

    def base_cost(self, states):
        return np.where(self.terminal(states), 0.0, np.where(np.isin(states, self.hazards), 1.0, 0.2))

    def move(self, states, directions):
        row = np.clip(states // self.cols + MOVES[directions, 0], 0, self.rows - 1)
        col = np.clip(states % self.cols + MOVES[directions, 1], 0, self.cols - 1)
        return row * self.cols + col

    def step(self, states, actions, rng):
        """One move per pair. Terminal squares end the episode at no cost.

        Elsewhere: pay the square's cost plus noise, move as intended with probability
        .8 (else one of the other three directions), hazards send you back to the start
        with probability .8, and stepping onto the decoy adds decoy_cost and ends it.
        """
        states, actions = np.broadcast_arrays(np.asarray(states, dtype=int), np.asarray(actions, dtype=int))
        done_already = self.terminal(states)
        other = (actions + rng.integers(1, 4, size=states.shape)) % 4
        actual = np.where(rng.random(states.shape) < self.intended_probability, actions, other)
        next_states = self.move(states, actual)
        reset = np.isin(states, self.hazards) & (rng.random(states.shape) < self.reset_probability)
        next_states = np.where(reset, self.start, next_states)
        costs = self.base_cost(states) + self.cost_noise * rng.normal(size=states.shape)
        costs += np.where(next_states == self.decoy, self.decoy_cost, 0.0)
        costs = np.where(done_already, 0.0, costs)
        next_states = np.where(done_already, states, next_states)
        return next_states, costs, done_already | self.terminal(next_states)

    def initial_states(self, count, rng):
        return np.full(count, self.start, dtype=int)

    def start_distribution(self):
        weights = np.zeros(self.n_states)
        weights[self.start] = 1.0
        return weights

    def state_label(self, state):
        return {"row": state // self.cols, "column": state % self.cols}

    def exact_model(self):
        """Return P (S, A, S) and expected cost c (S, A). Terminal rows of P are all zero."""
        S, A = self.n_states, self.n_actions
        P, c = np.zeros((S, A, S)), np.zeros((S, A))
        states = np.arange(S)
        for a in range(A):
            for actual in range(4):
                prob = self.intended_probability if actual == a else (1 - self.intended_probability) / 3
                nxt = self.move(states, np.full(S, actual))
                hazard = np.isin(states, self.hazards)
                for s in states[~self.terminal(states)]:
                    stay = (1 - self.reset_probability) if hazard[s] else 1.0
                    P[s, a, nxt[s]] += prob * stay
                    c[s, a] += prob * stay * (self.decoy_cost if nxt[s] == self.decoy else 0.0)
                    if hazard[s]:
                        P[s, a, self.start] += prob * self.reset_probability
        c += self.base_cost(states)[:, None]
        # Terminal squares cost nothing and lead nowhere, so their value is 0.
        terminal_rows = self.terminal(states)
        P[terminal_rows] = 0.0
        c[terminal_rows] = 0.0
        return P, c


@dataclass(frozen=True)
class FullBlackjack:
    gamma: float = 0.9
    action_names = ("stop", "draw")
    n_actions = 2
    n_states = 18 * 9  # player total 4..21, dealer visible card 2..10
    card_values = np.arange(2, 11)
    card_probabilities = np.array([1, 1, 1, 1, 1, 1, 1, 1, 4]) / 12

    @staticmethod
    def encode(total, dealer):
        return (np.asarray(total) - 4) * 9 + np.asarray(dealer) - 2

    @staticmethod
    def decode(states):
        states = np.asarray(states)
        return states // 9 + 4, states % 9 + 2

    def draw_cards(self, rng, size):
        return rng.choice(self.card_values, size=size, p=self.card_probabilities)

    def initial_states(self, count, rng):
        total = self.draw_cards(rng, count) + self.draw_cards(rng, count)
        return self.encode(total, self.draw_cards(rng, count))

    def start_distribution(self):
        weights = np.zeros(self.n_states)
        p = self.card_probabilities
        for i, a in enumerate(self.card_values):
            for j, b in enumerate(self.card_values):
                for k, d in enumerate(self.card_values):
                    weights[self.encode(a + b, d)] += p[i] * p[j] * p[k]
        return weights

    def state_label(self, state):
        total, dealer = self.decode(state)
        return dict(player_total=int(total), dealer_card=int(dealer))

    def step(self, states, actions, rng):
        states, actions = np.broadcast_arrays(np.asarray(states, dtype=int), np.asarray(actions, dtype=int))
        shape = states.shape
        states, actions = states.ravel(), actions.ravel()
        total, visible = self.decode(states)
        next_states, costs = states.copy(), np.zeros(len(states))
        done = np.zeros(len(states), dtype=bool)
        draw = np.flatnonzero(actions == 1)
        new_total = total[draw] + self.draw_cards(rng, len(draw))
        bust = new_total > 21
        done[draw], costs[draw] = bust, bust.astype(float)
        next_states[draw[~bust]] = self.encode(new_total[~bust], visible[draw[~bust]])
        stop = np.flatnonzero(actions == 0)
        dealer = visible[stop] + self.draw_cards(rng, len(stop))
        while np.any(dealer < 17):
            needs = dealer < 17
            dealer[needs] += self.draw_cards(rng, int(needs.sum()))
        costs[stop] = np.where(dealer > 21, -1, np.sign(dealer - total[stop]))
        done[stop] = True
        return next_states.reshape(shape), costs.reshape(shape), done.reshape(shape)

    def dealer_final_distribution(self, visible):
        """Probabilities of the dealer's final total (17..21, or 22 for bust)."""
        dist = {visible: 1.0}
        final = np.zeros(6)
        while dist:
            new = {}
            for total, prob in dist.items():
                for value, p in zip(self.card_values, self.card_probabilities):
                    t = total + value
                    if t >= 17:
                        final[min(t, 22) - 17] += prob * p
                    else:
                        new[t] = new.get(t, 0.0) + prob * p
            dist = new
        return final

    def exact_model(self):
        S = self.n_states
        P, c = np.zeros((S, 2, S)), np.zeros((S, 2))
        for s in range(S):
            total, visible = map(int, self.decode(s))
            final = self.dealer_final_distribution(visible)
            outcomes = np.array([np.sign(d - total) for d in range(17, 22)] + [-1.0])
            c[s, 0] = final @ outcomes
            for value, p in zip(self.card_values, self.card_probabilities):
                if total + value > 21:
                    c[s, 1] += p
                else:
                    P[s, 1, self.encode(total + value, visible)] += p
        return P, c


# ---------------------------------------------------------------------------
# Larger GridWorlds: the team's rules on bigger boards
# ---------------------------------------------------------------------------
import kl_spmd  # noqa: E402,F401  puts the team's code on the import path
from baseline import GridWorld  # noqa: E402


@dataclass(frozen=True)
class ScaledGridWorld(GridWorld):
    """The team's GridWorld rules on a larger board, starting from a random square.

    Movement, costs and resets are exactly the team's (baseline.py). Two differences:
    episodes start on a uniformly random square, and policies are scored by the
    average cost over all squares. With a fixed corner start and gamma = 0.9 only the
    ~10 moves around the corner would matter, so extra states would not change the
    score. Hazards and goals still send you back to square 0 with probability .8.
    """

    def initial_states(self, count, rng):
        return rng.integers(0, self.n_states, size=count)

    def start_distribution(self):
        return np.full(self.n_states, 1 / self.n_states)


def make_scaled_gridworld(size, seed=0, goal_fraction=0.03, hazard_fraction=0.10):
    """size x size board with the team's goal and hazard densities (3 and 10 per 100 squares),
    placed at random (fixed seed) anywhere except square 0."""
    rng = np.random.default_rng(seed)
    n_goals = round(goal_fraction * size * size)
    n_hazards = round(hazard_fraction * size * size)
    cells = rng.choice(np.arange(1, size * size), size=n_goals + n_hazards, replace=False)
    return ScaledGridWorld(rows=size, cols=size, goals=tuple(int(c) for c in sorted(cells[:n_goals])),
                           hazards=tuple(int(c) for c in sorted(cells[n_goals:])))
