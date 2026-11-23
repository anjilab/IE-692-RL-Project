"""GridWorld: handout layout (10x10), scaled layouts (30x30), rules, model and simulator agreement."""

import numpy as np
import pytest

from ie692_rl.envs.gridworld import HANDOUT_GOALS, HANDOUT_HAZARDS, GridWorld, layout

UP, DOWN, LEFT, RIGHT = range(4)


def test_handout_layout() -> None:
    env = GridWorld()
    assert (env.rows, env.cols, env.n_states, env.n_actions) == (10, 10, 100, 4)
    assert len(env.hazards) == 10
    assert len(env.goals) == 3
    assert env.start == 0
    assert env.gamma == 0.9
    assert env.goals == HANDOUT_GOALS
    assert env.hazards == HANDOUT_HAZARDS
    assert not set(env.goals) & set(env.hazards)
    assert env.start not in env.goals + env.hazards


@pytest.mark.parametrize(("size", "n_hazards"), [(2, 0), (5, 2), (30, 90), (31, 96)])
def test_scaled_layout(size: int, n_hazards: int) -> None:
    env = GridWorld(size=size)
    last = size * size - 1
    assert env.n_states == size * size
    assert env.goals == (size - 1, last - size + 1, last)  # the three corners other than the start
    assert len(env.hazards) == n_hazards
    assert len(set(env.hazards)) == n_hazards
    assert not set(env.hazards) & {env.start, *env.goals}
    assert all(0 <= h < env.n_states for h in env.hazards)


def test_layout_is_reproducible_and_seeded() -> None:
    assert layout(30, 0.1, 0) == layout(30, 0.1, 0)
    assert layout(30, 0.1, 0)[1] != layout(30, 0.1, 1)[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 1},
        {"p": 1.5},
        {"p": -0.1},
        {"q": 2.0},
        {"gamma": 1.0},
        {"gamma": 0.0},
        {"size": 3, "hazard_density": 1.0},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        GridWorld(**kwargs)  # type: ignore[arg-type]


def test_costs_by_cell_type() -> None:
    env = GridWorld()
    assert env.state_cost[env.start] == pytest.approx(0.2)
    assert all(env.state_cost[g] == 0.0 for g in env.goals)
    assert all(env.state_cost[h] == 1.0 for h in env.hazards)
    assert env.cost_bound == 1.0


def test_deterministic_moves_and_walls() -> None:
    env, rng = GridWorld(p=1.0), np.random.default_rng(0)
    centre = 5 * 10 + 5
    next_state, cost, done = env.step_one(centre, UP, rng)
    assert (next_state, done) == (centre - 10, False)
    assert cost == pytest.approx(0.2)
    assert env.step_one(centre, DOWN, rng)[0] == centre + 10
    assert env.step_one(centre, LEFT, rng)[0] == centre - 1
    assert env.step_one(centre, RIGHT, rng)[0] == centre + 1
    assert env.step_one(0, UP, rng)[0] == 0  # walls clip
    assert env.step_one(0, LEFT, rng)[0] == 0
    assert env.step_one(9 * 10 + 1, DOWN, rng)[0] == 9 * 10 + 1


def test_special_cells_reset_or_move() -> None:
    rng = np.random.default_rng(1)
    always_reset = GridWorld(p=1.0, q=0.0)
    hazard = always_reset.hazards[0]
    assert all(always_reset.step_one(hazard, RIGHT, rng) == (0, 1.0, False) for _ in range(100))
    never_reset = GridWorld(p=1.0, q=1.0)
    goal = never_reset.goals[1]  # bottom-left corner (90): moving right goes to 91
    assert all(never_reset.step_one(goal, RIGHT, rng) == (91, 0.0, False) for _ in range(100))


@pytest.mark.parametrize("size", [2, 10, 30])
def test_model_is_a_valid_distribution(size: int) -> None:
    env = GridWorld(size=size)
    model = env.model()
    assert model.probs.shape == model.next_states.shape == (env.n_states, 4, 5)
    assert np.all(model.probs >= 0)
    np.testing.assert_allclose(model.probs.sum(axis=2), 1.0)
    assert np.all((model.next_states >= 0) & (model.next_states < env.n_states))
    np.testing.assert_array_equal(model.costs, np.repeat(env.state_cost[:, None], 4, axis=1))


@pytest.mark.parametrize("size", [10, 30])
def test_simulator_matches_model(size: int) -> None:
    """Empirical next-state frequencies are within 5 standard errors of the exact probabilities."""
    env, rng, n = GridWorld(size=size), np.random.default_rng(2), 100_000
    P = env.model().dense_transitions()
    corner = size - 1
    for s, a in [(0, UP), (size + 1, RIGHT), (env.hazards[0], LEFT), (env.goals[2], DOWN), (corner, RIGHT)]:
        next_states, costs, done = env.step(np.full(n, s), np.full(n, a), rng)
        freq = np.bincount(next_states, minlength=env.n_states) / n
        se = np.sqrt(P[s, a] * (1 - P[s, a]) / n) + 1e-12
        assert np.all(np.abs(freq - P[s, a]) <= 5 * se + 1e-9), (s, a)
        assert np.all(costs == env.state_cost[s])
        assert not done.any()


def test_batched_step_equals_repeated_step_one() -> None:
    env = GridWorld()
    states = np.random.default_rng(3).integers(0, env.n_states, 200)
    actions = np.random.default_rng(4).integers(0, 4, 200)
    batched = env.step(states, actions, np.random.default_rng(5))
    rng = np.random.default_rng(5)
    one_by_one = [env.step_one(int(s), int(a), rng) for s, a in zip(states, actions, strict=True)]
    np.testing.assert_array_equal(batched[0], [r[0] for r in one_by_one])
    np.testing.assert_array_equal(batched[1], [r[1] for r in one_by_one])


def test_step_keeps_input_shape() -> None:
    env = GridWorld()
    next_states, costs, done = env.step(
        np.zeros((3, 2), dtype=np.int64), np.ones((3, 2), dtype=np.int64), np.random.default_rng(0)
    )
    assert next_states.shape == costs.shape == done.shape == (3, 2)


@pytest.mark.parametrize(("state", "action"), [(-1, 0), (100, 0), (0, 4), (0, -1)])
def test_invalid_state_or_action(state: int, action: int) -> None:
    env = GridWorld()
    with pytest.raises(ValueError, match="Invalid state or action"):
        env.step_one(state, action, np.random.default_rng(0))
    with pytest.raises(ValueError, match="Invalid state or action"):
        env.step(np.array([state]), np.array([action]), np.random.default_rng(0))


def test_reset_labels_and_board() -> None:
    env = GridWorld()
    assert env.reset(np.random.default_rng(0)) == 0
    np.testing.assert_array_equal(env.initial_states(3, np.random.default_rng(0)), [0, 0, 0])
    np.testing.assert_array_equal(env.start_distribution(), np.eye(100)[0])
    assert env.state_label(57) == {"row": 5, "column": 7}
    rows = env.board().splitlines()
    assert len(rows) == 10
    assert rows[0].split()[0] == "S"
    assert rows[0].split()[9] == "G"
    assert sum(r.count("H") for r in rows) == 10
