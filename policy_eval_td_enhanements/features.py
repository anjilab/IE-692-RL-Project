"""Two simple feature matrices for action-value TD: Q = (Phi @ theta)."""
import numpy as np


def tabular_features(env):
    """One independent parameter for every (state, action) pair."""
    return np.eye(env.n_states * env.n_actions)


def linear_features(env):
    """Share one parameter across nearby states, separately for each action.

    GridWorld: group every 2x2 block of cells.
    Blackjack: group player totals 4-7 and 8-11 for each dealer card.
    """
    S, A = env.n_states, env.n_actions

    if hasattr(env, "rows"):  # GridWorld
        n_columns = (env.cols + 1) // 2
        group = [(s // env.cols // 2) * n_columns + (s % env.cols) // 2
                 for s in range(S)]
        n_groups = ((env.rows + 1) // 2) * n_columns
    else:  # Blackjack, states encoded as (total - 4) * 4 + dealer - 2
        group = [(s // 4 // 4) * 4 + (s % 4) for s in range(S)]
        n_groups = 8

    Phi = np.zeros((S * A, n_groups * A))
    for s in range(S):
        for a in range(A):
            Phi[s * A + a, group[s] * A + a] = 1.0
    return Phi
