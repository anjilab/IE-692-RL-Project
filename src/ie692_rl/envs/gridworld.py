"""Square GridWorld from Lecture 1, enlarged as the project handout asks.

Actions: 0=up, 1=down, 2=left, 3=right. The intended move happens with probability ``p``;
otherwise one of the other three directions is taken uniformly. Walls clip movement.
The cost is charged at the current state: 0 at a goal, 1 at a hazard, 0.2 elsewhere. From a goal or
hazard the agent is sent back to the start with probability ``1 - q``; otherwise it moves normally.
The task never terminates.

``size=10`` is the handout's layout (10 hazards, 3 goals, start in the top-left corner). Other sizes
keep the start in the top-left corner and goals in the other three corners, and place
``round(hazard_density * size**2)`` hazards (10% by default, so 90 on a 30x30 grid) uniformly at
random with a fixed ``layout_seed``, never on the start or a goal.
"""

from typing import Any

import numpy as np
from numba import njit

from ie692_rl.envs.base import Environment, Kernel, TabularModel
from ie692_rl.types import BoolArray

HANDOUT_GOALS = (9, 90, 99)
HANDOUT_HAZARDS = (12, 17, 24, 36, 43, 58, 62, 75, 81, 87)
GOAL_COST, HAZARD_COST, STEP_COST = 0.0, 1.0, 0.2
ROW_MOVE = np.array([-1, 1, 0, 0], dtype=np.int64)
COL_MOVE = np.array([0, 0, -1, 1], dtype=np.int64)


def layout(size: int, hazard_density: float, layout_seed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(goals, hazards) as flat cell indices; the start is always cell 0."""
    if size == 10:
        return HANDOUT_GOALS, HANDOUT_HAZARDS
    last = size * size - 1
    goals = (size - 1, last - (size - 1), last)
    free = np.setdiff1d(np.arange(1, size * size), goals)
    n_hazards = round(hazard_density * size * size)
    if not 0 <= n_hazards <= len(free):
        raise ValueError("hazard_density leaves no room for the hazards")
    hazards = np.random.default_rng(layout_seed).choice(free, size=n_hazards, replace=False)
    return goals, tuple(sorted(int(h) for h in hazards))


class GridWorld(Environment):
    action_names = ("up", "down", "left", "right")

    def __init__(
        self,
        size: int = 10,
        p: float = 0.8,
        q: float = 0.2,
        gamma: float = 0.9,
        hazard_density: float = 0.1,
        layout_seed: int = 0,
    ) -> None:
        if size < 2:
            raise ValueError("size must be at least 2")
        if not (0 <= p <= 1 and 0 <= q <= 1 and 0 < gamma < 1):
            raise ValueError("Need 0 <= p, q <= 1 and 0 < gamma < 1")
        self.rows = self.cols = int(size)
        self.n_states = size * size
        self.n_actions = 4
        self.start = 0
        self.gamma, self.p, self.q = float(gamma), float(p), float(q)
        self.goals, self.hazards = layout(size, hazard_density, layout_seed)
        self.cost_bound = HAZARD_COST
        self.state_cost = np.full(self.n_states, STEP_COST)
        self.state_cost[list(self.goals)] = GOAL_COST
        self.state_cost[list(self.hazards)] = HAZARD_COST
        self.special: BoolArray = self.state_cost != STEP_COST
        self._kernel = Kernel(
            _step,
            _reset,
            (self.rows, self.cols, self.p, self.q, self.start, self.state_cost, self.special),
        )

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def model(self) -> TabularModel:
        """Sparse model: 4 movement outcomes plus a reset slot per (state, action)."""
        S, C = self.n_states, self.cols
        rows, cols = np.divmod(np.arange(S), C)
        next_states = np.zeros((S, 4, 5), dtype=np.int64)
        probs = np.zeros((S, 4, 5))
        move_weight = np.where(self.special, self.q, 1.0)
        for a in range(4):
            for actual in range(4):
                r = np.clip(rows + ROW_MOVE[actual], 0, self.rows - 1)
                c = np.clip(cols + COL_MOVE[actual], 0, self.cols - 1)
                next_states[:, a, actual] = r * C + c
                probs[:, a, actual] = (self.p if actual == a else (1 - self.p) / 3) * move_weight
            next_states[:, a, 4] = self.start
            probs[:, a, 4] = np.where(self.special, 1 - self.q, 0.0)
        costs = np.repeat(self.state_cost[:, None], 4, axis=1)
        return TabularModel(next_states, probs, costs)

    def state_label(self, state: int) -> dict[str, int]:
        return {"row": int(state) // self.cols, "column": int(state) % self.cols}

    def board(self) -> str:
        """Text picture of the grid: S start, G goal, H hazard."""
        cells = np.full(self.n_states, ".")
        cells[list(self.goals)] = "G"
        cells[list(self.hazards)] = "H"
        cells[self.start] = "S"
        return "\n".join(" ".join(row) for row in cells.reshape(self.rows, self.cols))


@njit(cache=True)
def _step(
    params: tuple[Any, ...], state: int, action: int, rng: np.random.Generator
) -> tuple[int, float, bool]:
    rows, cols, p, q, start, state_cost, special = params
    cost = state_cost[state]
    if special[state] and rng.random() >= q:  # reset with probability 1 - q
        return start, cost, False
    direction = action if rng.random() < p else (action + rng.integers(1, 4)) % 4
    row = min(max(state // cols + ROW_MOVE[direction], 0), rows - 1)
    col = min(max(state % cols + COL_MOVE[direction], 0), cols - 1)
    return row * cols + col, cost, False


@njit(cache=True)
def _reset(params: tuple[Any, ...], rng: np.random.Generator) -> int:
    start: int = params[4]
    return start
