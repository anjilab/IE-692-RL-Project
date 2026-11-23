"""Simplified online Blackjack for IE 692.

Adapted from Gymnasium's Blackjack (https://gymnasium.farama.org/environments/toy_text/blackjack/);
see the AI-usage section of the report for the adaptation prompts.

Cards: 2, 3, 4, 5 (uniform, infinite deck with replacement). Actions: 0 = stop, 1 = draw.
The dealer draws until the total is at least 9; totals above 11 bust. Costs: win -1, tie 0, loss +1.

State: (player total, dealer visible card). The player starts with two cards (total 4-10) and can
draw while not bust, so non-bust totals are 4..11 (8 values); the dealer shows 2..5 (4 values), so
there are 8 * 4 = 32 states. Bust and stop end the hand (``done=True``); the next hand is a fresh deal.
Values are reported at the fixed state (total 8, dealer shows 3).
"""

from typing import Any

import numpy as np

from ie692_rl._jit import njit
from ie692_rl.envs.base import Environment, Kernel, TabularModel
from ie692_rl.types import IntArray

LOW_CARD, HIGH_CARD = 2, 5
DEALER_STANDS, BUST_ABOVE = 9, 11
MIN_TOTAL = 2 * LOW_CARD
STOP, DRAW = 0, 1


@njit
def encode(total: int, dealer: int) -> int:
    """Map (player total, visible dealer card) to a 0..31 state index."""
    return (total - MIN_TOTAL) * 4 + dealer - LOW_CARD


@njit
def _draw(rng: np.random.Generator) -> int:
    return int(rng.integers(LOW_CARD, HIGH_CARD + 1))


@njit
def _step(
    params: tuple[Any, ...], state: int, action: int, rng: np.random.Generator
) -> tuple[int, float, bool]:
    player, visible = state // 4 + MIN_TOTAL, state % 4 + LOW_CARD
    if action == DRAW:
        player += _draw(rng)
        if player > BUST_ABOVE:
            return state, 1.0, True  # bust: loss, hand over (next_state is a placeholder)
        return encode(player, visible), 0.0, False
    # Stop: the hidden dealer card is only sampled now; it is independent of the player's actions.
    dealer = visible + _draw(rng)
    while dealer < DEALER_STANDS:
        dealer += _draw(rng)
    if dealer > BUST_ABOVE or player > dealer:
        return state, -1.0, True
    if player < dealer:
        return state, 1.0, True
    return state, 0.0, True


@njit
def _reset(params: tuple[Any, ...], rng: np.random.Generator) -> int:
    """A fresh deal: two player cards and one visible dealer card."""
    total = _draw(rng) + _draw(rng)
    return encode(total, _draw(rng))


def dealer_final_distribution(visible: int) -> dict[int, float]:
    """Exact distribution of the dealer's final total given the visible card."""
    pending: dict[int, float] = {visible: 1.0}
    final: dict[int, float] = {}
    n_cards = HIGH_CARD - LOW_CARD + 1
    while pending:
        nxt: dict[int, float] = {}
        for total, prob in pending.items():
            for card in range(LOW_CARD, HIGH_CARD + 1):
                target = final if total + card >= DEALER_STANDS else nxt
                target[total + card] = target.get(total + card, 0.0) + prob / n_cards
        pending = nxt
    return final


class Blackjack(Environment):
    action_names = ("stop", "draw")

    def __init__(self, gamma: float = 0.9) -> None:
        if not 0 < gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        self.gamma = float(gamma)
        self.n_states, self.n_actions = 32, 2
        self.evaluation_total, self.evaluation_dealer = 8, 3
        self.start = self.encode(self.evaluation_total, self.evaluation_dealer)
        self.cost_bound = 1.0
        self._kernel = Kernel(_step, _reset, ())

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def encode(self, total: int, dealer: int) -> int:
        return int(encode(int(total), int(dealer)))

    def decode(self, state: int) -> tuple[int, int]:
        """(player total, visible dealer card)."""
        return int(state) // 4 + MIN_TOTAL, int(state) % 4 + LOW_CARD

    def deal(self, count: int, rng: np.random.Generator) -> IntArray:
        """``count`` fresh deals, drawn with the same kernel as ``reset``."""
        return np.array([self.reset(rng) for _ in range(count)], dtype=np.int64)

    def model(self) -> TabularModel:
        """Stop always ends the hand; draw moves to one of 4 totals or busts."""
        S, n_cards = self.n_states, HIGH_CARD - LOW_CARD + 1
        next_states = np.zeros((S, 2, n_cards), dtype=np.int64)
        probs = np.zeros((S, 2, n_cards))
        costs = np.zeros((S, 2))
        for s in range(S):
            total, visible = self.decode(s)
            costs[s, STOP] = sum(
                p * (-1.0 if d > BUST_ABOVE else float(np.sign(d - total)))
                for d, p in dealer_final_distribution(visible).items()
            )
            for k, card in enumerate(range(LOW_CARD, HIGH_CARD + 1)):
                if total + card > BUST_ABOVE:
                    costs[s, DRAW] += 1.0 / n_cards
                else:
                    next_states[s, DRAW, k] = self.encode(total + card, visible)
                    probs[s, DRAW, k] = 1.0 / n_cards
        return TabularModel(next_states, probs, costs)

    def state_label(self, state: int) -> dict[str, int]:
        total, visible = self.decode(state)
        return {"player_total": total, "dealer_card": visible}
