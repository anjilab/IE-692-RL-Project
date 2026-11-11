"""Simplified blackjack.

Cards 2,3,4,5 have equal probability, with replacement. Two cards are initially
held by each participant; only one dealer card is visible. No aces or natural
bonuses. Player chooses stop (0) or draw (1); dealer draws until total >= 9.
Above 11 is a bust. Terminal costs: win -1, draw 0, loss +1; intermediate cost 0.
These dealing, tie, and cost conventions fill details not specified by the PDF.

State = (player total 4..11, dealer visible card 2..5), so 32 decision states.
The hidden card is sampled when stopping: its conditional distribution remains
uniform because draws use replacement and player decisions do not reveal it.
Dealer play is part of the stop action, not additional player time steps.
Evaluation uses the fixed state (8,3), allowed by the project handout.
"""
from dataclasses import dataclass
import numpy as np
if __package__:
    from .utils import run_experiment
else:
    from utils import run_experiment


@dataclass(frozen=True)
class Blackjack:
    gamma: float = 0.9
    evaluation_total: int = 8
    evaluation_dealer: int = 3
    action_names = ('stop', 'draw')
    n_states = 32
    n_actions = 2

    @staticmethod
    def encode(total, dealer):
        return (np.asarray(total) - 4) * 4 + np.asarray(dealer) - 2

    @staticmethod
    def decode(states):
        states = np.asarray(states)
        return states // 4 + 4, states % 4 + 2

    def initial_states(self, count, rng):
        return np.full(count, self.encode(self.evaluation_total, self.evaluation_dealer), dtype=int)

    def deal(self, count, rng):
        """Fresh initial hands for optional evaluation over the deal distribution."""
        total = rng.integers(2, 6, size=(count, 2)).sum(axis=1)
        dealer = rng.integers(2, 6, size=count)
        return self.encode(total, dealer)

    def state_label(self, state):
        total, dealer = self.decode(state)
        return dict(player_total=int(total), dealer_card=int(dealer))

    def step_one(self, s_t, a_t, rng):
        """Same blackjack rules as step(), for one online player decision."""
        player = s_t // 4 + 4
        visible = s_t % 4 + 2
        if a_t == 1:
            player += int(rng.integers(2, 6))
            if player > 11:
                return s_t, 1.0, True
            return (player-4)*4 + visible-2, 0.0, False
        dealer = visible + int(rng.integers(2, 6))
        while dealer < 9:
            dealer += int(rng.integers(2, 6))
        cost = -1.0 if dealer > 11 or player > dealer else 1.0 if player < dealer else 0.0
        return s_t, cost, True

    def step(self, states, actions, rng):
        states, actions = np.broadcast_arrays(np.asarray(states, dtype=int), np.asarray(actions, dtype=int))
        if np.any((states < 0) | (states >= self.n_states)) or np.any((actions < 0) | (actions >= 2)):
            raise ValueError('Invalid blackjack state or action')
        shape = states.shape
        states, actions = states.ravel(), actions.ravel()
        total, visible = self.decode(states)
        next_states = states.copy()
        costs = np.zeros(len(states))
        done = np.zeros(len(states), dtype=bool)
        draw = np.flatnonzero(actions == 1)
        new_total = total[draw] + rng.integers(2, 6, size=len(draw))
        bust = new_total > 11
        done[draw] = bust
        costs[draw] = bust.astype(float)
        surviving = draw[~bust]
        next_states[surviving] = self.encode(new_total[~bust], visible[surviving])
        stop = np.flatnonzero(actions == 0)
        dealer = visible[stop] + rng.integers(2, 6, size=len(stop))
        while np.any(dealer < 9):
            needs_card = dealer < 9
            dealer[needs_card] += rng.integers(2, 6, size=int(needs_card.sum()))
        costs[stop] = np.where(dealer > 11, -1, np.sign(dealer - total[stop]))
        done[stop] = True
        # next_state is a placeholder when done=True; shared code never uses it.
        return next_states.reshape(shape), costs.reshape(shape), done.reshape(shape)


if __name__ == '__main__':
    run_experiment(Blackjack(), 'blackjack', [1, 2, 4], [2, 3, 4], 120_000, 8)
