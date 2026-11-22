"""Simplified online Blackjack for IE 692.

We first extracted the code from original blackjack "https://gymnasium.farama.org/environments/toy_text/blackjack/"  and adapted to our requirements. 
See the adapation process and the prompts in the AI usage. 

Cards: 2, 3, 4, 5 (uniform, with replacement).
Actions: 0 = stop, 1 = draw.
Dealer draws until total >= 9; totals > 11 bust.
Costs: win = -1, tie = 0, loss = +1.
State: (player total, dealer visible card).  
    - player get 2 cards: min total = 2+2 =4 and max initial total=10, player can draw another card without busting so 4 + 2 + 5 = 11. so non-bust player totals = {4,5,6,7,8,9,10,11} = 8 values.
    - dealer gets 1 visible card: min = 2, max = 5. so 4 possibilities
    - total states = 8 * 4 = 32. Bust totals are terminal outcomes.
Each card is drawn independently with replacement.
"""


import numpy as np


class Blackjack:
    def __init__(self, gamma=0.9):
        self.gamma = gamma 
        self.n_states = 32 
        self.n_actions = 2  #  all other possible actions like  No double-down, split etc) on black jack are removed and only 2 actions are kept.
        self.action_names = ("stop", "draw")
        self.evaluation_total = 8
        self.evaluation_dealer = 3

    def encode(self, total, dealer):
        """Map (player total, visible dealer card) to a 0..31 state ID."""
        return (total - 4) * 4 + dealer - 2

    def decode(self, state):
        """Recover the player total and visible dealer card from a state ID."""
        return state // 4 + 4, state % 4 + 2

    def initial_states(self, count, rng):
        """Use the fixed (8, 3) state for value-comparison experiments.
        Prepares multiple identical starting situations so we can evaluate the same Blackjack state repeatedly 
        and compare policy performance across training iterations.
        """
        start = self.encode(self.evaluation_total, self.evaluation_dealer)
        return np.full(count, start, dtype=int)

    def deal(self, count, rng):
        """Generate random hands: two player cards and one visible dealer card.

        All draws are uniform from 2, 3, 4, 5 with replacement. The hidden
        dealer card is sampled later on stopping; it stays uniform because
        player actions provide no information about that card.
        """
        totals = rng.integers(2, 6, size=(count, 2)).sum(axis=1)
        visible = rng.integers(2, 6, size=count)
        return self.encode(totals, visible)

    def step_one(self, state, action, rng):
        """Simulate one online decision; return (next_state, cost, done).

        done=True means the player busted or stopped and the round ended.
        On terminal transitions, next_state is a placeholder and must not
        be used to bootstrap future values.
        """
        if not 0 <= state < self.n_states or action not in (0, 1):
            raise ValueError("Invalid state or action")

        player, visible = self.decode(state)

        if action == 1:  # Draw: sample a card from 2..5, with replacement.
            player += int(rng.integers(2, 6))
            if player > 11:  # 11 is allowed; strictly greater than 11 busts.
                return state, 1.0, True  # Loss cost; round ends immediately.
            return self.encode(player, visible), 0.0, False  # No intermediate cost.

        # Stop: sample the dealer's hidden card only when needed.
        # This models two initial dealer cards without storing the hidden one.
        dealer = visible + int(rng.integers(2, 6))
        while dealer < 9:  # Dealer stands on 9, 10, or 11 (or busts above 11).
            dealer += int(rng.integers(2, 6))

        # Convert game rewards to costs for our cost-minimizing algorithms.
        if dealer > 11 or player > dealer:
            cost = -1.0
        elif player < dealer:
            cost = 1.0
        else:
            cost = 0.0
        return state, cost, True

    def step(self, states, actions, rng):
        """Apply step_one repeatedly for convenient batched sampling.

        This simpler loop is easier to read than a vectorized implementation.
        """
        s, a = np.broadcast_arrays(np.asarray(states), np.asarray(actions))
        next_s = np.empty(s.shape, dtype=int)
        costs = np.empty(s.shape, dtype=float)
        done = np.empty(s.shape, dtype=bool)
        for index in np.ndindex(s.shape):
            next_s[index], costs[index], done[index] = self.step_one(
                int(s[index]), int(a[index]), rng
            )
        return next_s, costs, done

    def state_label(self, state):
        """Human-readable description of an encoded state."""
        total, visible = self.decode(state)
        return {"player_total": total, "dealer_card": visible}


if __name__ == "__main__":
    env = Blackjack()
    rng = np.random.default_rng(42)
    assert env.n_states == 32
    assert env.decode(env.encode(8, 3)) == (8, 3)
    assert np.all((env.deal(100, rng) >= 0) & (env.deal(100, rng) < 32))
    result = env.step_one(env.encode(8, 3), 0, rng)
    assert result[2]
    print("Blackjack check passed:", result)
