"""10x10 GridWorld problem.

Actions: 0=up, 1=down, 2=left, 3=right.
Costs are charged at the current state. Goal/hazard states reset to start
with probability 0.8; otherwise movement occurs normally.
"""

import numpy as np

class GridWorld:
    def __init__(self, p=0.8, q=0.2, gamma=0.9):
        self.rows = 10
        self.cols = 10
        self.n_states = self.rows * self.cols
        self.n_actions = 4
        self.goals = (9, 90, 99)
        self.hazards = (12, 17, 24, 36, 43, 58, 62, 75, 81, 87)
        self.start = 0
        self.gamma = gamma
        self.p = p  # Probability of intended direction
        self.q = q  # Probability of moving instead of resetting

    action_names = ("up", "down", "left", "right")

    def sample(self, states, actions, rng):
        """Generate next states and costs from online samples (no transition matrix)."""
        s, a = np.broadcast_arrays(np.asarray(states, dtype=int),
                                   np.asarray(actions, dtype=int))
        if np.any((s < 0) | (s >= self.n_states)) or np.any((a < 0) | (a >= 4)):
            raise ValueError("Invalid state or action")

        goal, hazard = np.isin(s, self.goals), np.isin(s, self.hazards)
        cost = np.where(goal, 0.0, np.where(hazard, 1.0, 0.2))
        other = (a + rng.integers(1, 4, size=s.shape)) % 4
        direction = np.where(rng.random(s.shape) < self.p, a, other)
        dr, dc = np.array([-1, 1, 0, 0]), np.array([0, 0, -1, 1])
        row = np.clip(s // self.cols + dr[direction], 0, self.rows - 1)
        col = np.clip(s % self.cols + dc[direction], 0, self.cols - 1)
        reset = (goal | hazard) & (rng.random(s.shape) < 1 - self.q)
        next_s = np.where(reset, self.start, row * self.cols + col)
        return next_s, cost

    def step(self, states, actions, rng):
        """Compatibility interface for RL training: (next_state, cost, done)."""
        next_s, cost = self.sample(states, actions, rng)
        return next_s, cost, np.zeros(np.shape(next_s), dtype=bool)

    def step_one(self, state, action, rng):
        next_s, cost = self.sample(state, action, rng)
        return int(next_s), float(cost), False

    def initial_states(self, count, rng):
        return np.full(count, self.start, dtype=int)

    def state_label(self, state):
        return {"row": int(state) // self.cols, "column": int(state) % self.cols}


if __name__ == "__main__":
    env = GridWorld()
    rng = np.random.default_rng(42)
    next_s, cost, done = env.step_one(0, 3, rng) # simulation happens here.
    assert env.n_states == 100 and len(env.goals) == 3 and len(env.hazards) == 10
    assert 0 <= next_s < 100 and not done
    print("GridWorld check passed:", next_s, cost)
