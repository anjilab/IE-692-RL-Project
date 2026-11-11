"""10 x 10 GridWorld baseline with dynamic on-policy Monte Carlo. Run python3 baseline.py --help.
Movement and costs follow assignments/RL_Assignment1_GridWorldProblem.pdf.
Goals/hazards reset on the next move with probability .8, not termination.

"""
from dataclasses import dataclass
import numpy as np
if __package__:
    from .utils import run_experiment
else:
    from utils import run_experiment


@dataclass(frozen=True)
class GridWorld:
    rows: int = 10
    cols: int = 10
    goals: tuple = (9, 90, 99)
    hazards: tuple = (12, 17, 24, 36, 43, 58, 62, 75, 81, 87)
    start: int = 0
    gamma: float = 0.9
    intended_probability: float = 0.8
    reset_probability: float = 0.8

    action_names = ("up", "down", "left", "right")

    @property
    def n_states(self):
        return self.rows * self.cols

    @property
    def n_actions(self):
        return 4

    def sample(self, states, actions, rng):
        """Sample one move per input pair; cost is charged before moving.

        At goals/hazards, reset with probability .8; otherwise use the normal
        movement rule. Walls clip movement. Goals/hazards do not end a run.
        """
        states, actions = np.broadcast_arrays(
            np.asarray(states, dtype=int), np.asarray(actions, dtype=int))
        if np.any((states < 0) | (states >= self.n_states)):
            raise ValueError('State outside the board')
        if np.any((actions < 0) | (actions >= self.n_actions)):
            raise ValueError('Action must be 0, 1, 2, or 3')
        goal = np.isin(states, self.goals)
        hazard = np.isin(states, self.hazards)
        costs = np.where(goal, 0.0, np.where(hazard, 1.0, 0.2))
        # Offset 1, 2, or 3 selects each other action with equal probability.
        other = (actions + rng.integers(1, 4, size=states.shape)) % 4
        actual = np.where(rng.random(states.shape) < self.intended_probability,
                          actions, other)
        dr = np.array([-1, 1, 0, 0])[actual]
        dc = np.array([0, 0, -1, 1])[actual]
        row = np.clip(states // self.cols + dr, 0, self.rows - 1)
        col = np.clip(states % self.cols + dc, 0, self.cols - 1)
        reset = (goal | hazard) & (rng.random(states.shape) < self.reset_probability)
        return np.where(reset, self.start, row * self.cols + col), costs


    def step_one(self, s_t, a_t, rng):
        """Same game rules as step(), optimized for a single online move."""
        special = s_t in self.goals or s_t in self.hazards
        cost = 0.0 if s_t in self.goals else 1.0 if s_t in self.hazards else 0.2
        if special and rng.random() < self.reset_probability:
            return self.start, cost, False
        actual = a_t
        if rng.random() >= self.intended_probability:
            actual = (a_t + int(rng.integers(1, 4))) % 4
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[actual]
        row = min(self.rows-1, max(0, s_t // self.cols + dr))
        col = min(self.cols-1, max(0, s_t % self.cols + dc))
        return row*self.cols + col, cost, False

    def step(self, states, actions, rng):
        next_states, costs = self.sample(states, actions, rng)
        return next_states, costs, np.zeros(np.shape(next_states), dtype=bool)

    def initial_states(self, count, rng):
        return np.full(count, self.start, dtype=int)

    def state_label(self, state):
        return {"row": state // self.cols, "column": state % self.cols}


def plot_board(env, output):
    # Import only on request: training does not depend on Matplotlib.
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    board = np.zeros(env.n_states, dtype=int)
    board[list(env.hazards)] = 1
    board[list(env.goals)] = 2
    board[env.start] = 3
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(board.reshape(env.rows, env.cols), vmin=0, vmax=3,
              cmap=ListedColormap(['#eeeeee', '#ef9a9a', '#a5d6a7', '#90caf9']))
    for s in range(env.n_states):
        label = 'S' if s == env.start else 'G' if s in env.goals else 'H' if s in env.hazards else ''
        ax.text(s % env.cols, s // env.cols, f'{s}\n{label}', ha='center', va='center', fontsize=8)
    ax.set_title('10 × 10 GridWorld: S=start, G=goal, H=hazard')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(output / 'board.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    run_experiment(GridWorld(), "gridworld", [1, 2, 4], [40, 70, 100],
                   1_000_000, 100, plot_board)
