"""Two ways to estimate Q from samples, with the same call signature.

generative_q (the team's method)
    Starts rollouts from every state and every action, then follows the policy.
    Every action's Q is estimated each iteration, whatever the policy prefers,
    so the sampler itself does the exploring.

online_q
    Plays whole episodes from the environment's start state(s), following the
    policy, as an agent would online. Only (state, action) pairs the policy actually
    visits get estimates, so actions and states it avoids stay unknown. This is where
    entropy regularization can matter: it keeps the policy visiting alternatives.

Both return Q as costs (lower is better) and the number of environment steps used.
"""
import numpy as np

import kl_spmd  # noqa: F401  puts the team's code on the import path
import utils


def generative_q(env, policy, rollouts, horizon, rng):
    return utils.monte_carlo_q(env, policy, rollouts, horizon, rng)


def online_q(env, policy, episodes, horizon, rng, record_steps=None):
    """Every-visit Monte Carlo Q from on-policy episodes started at the start state(s).

    Each episode runs for at most `horizon` steps. Only visits in the first
    `record_steps` steps (default horizon // 2) are used, so each recorded return
    still sees at least horizon - record_steps further steps before being cut off.

    Pairs with no visits are filled in so the KL update leaves them alone:
    an untried action at a visited state gets that state's average return, and
    every action at an unvisited state gets 0 (a constant row does not change the
    plain KL or Euclidean update).
    """
    record_steps = horizon // 2 if record_steps is None else record_steps
    states = np.array(env.initial_states(episodes, rng), dtype=int)
    alive = np.ones(episodes, dtype=bool)
    visited_states = np.zeros((horizon, episodes), dtype=int)
    visited_actions = np.zeros((horizon, episodes), dtype=int)
    costs = np.zeros((horizon, episodes))
    was_alive = np.zeros((horizon, episodes), dtype=bool)
    observations = 0
    for t in range(horizon):
        active = np.flatnonzero(alive)
        if not len(active):
            break
        actions = utils.draw_actions(policy, states[active], rng)
        next_states, step_costs, done = env.step(states[active], actions, rng)
        observations += len(active)
        visited_states[t, active], visited_actions[t, active] = states[active], actions
        costs[t, active], was_alive[t, active] = step_costs, True
        states[active] = next_states
        alive[active[done]] = False

    returns = np.zeros_like(costs)
    future = np.zeros(episodes)
    for t in reversed(range(horizon)):
        future = costs[t] + env.gamma * future  # costs are 0 once an episode has ended
        returns[t] = future

    used = was_alive[:record_steps]
    s, a, g = visited_states[:record_steps][used], visited_actions[:record_steps][used], returns[:record_steps][used]
    q_sum, q_count = np.zeros((env.n_states, env.n_actions)), np.zeros((env.n_states, env.n_actions))
    np.add.at(q_sum, (s, a), g)
    np.add.at(q_count, (s, a), 1)
    state_count = q_count.sum(axis=1, keepdims=True)
    state_average = np.divide(q_sum.sum(axis=1, keepdims=True), state_count,
                              out=np.zeros_like(state_count), where=state_count > 0)
    q = np.where(q_count > 0, q_sum / np.maximum(q_count, 1), state_average)
    return q, observations
