"""One-hot feature maps for action-value TD: Q_theta(s, a) = theta[feature_of[s, a]].

Both feature sets used by the team are one-hot (each (s, a) activates exactly one feature with
value 1), so a feature map is stored as an integer table ``feature_of[s, a]`` instead of a dense
(|S||A| x d) matrix. For a 30x30 GridWorld the dense tabular matrix would take 104 MB; the table
takes 29 kB, and every TD update touches one feature instead of a d-dimensional dot product.
"""

from typing import NamedTuple

import numpy as np

from ie692_rl.envs.base import Environment
from ie692_rl.envs.blackjack import Blackjack
from ie692_rl.envs.gridworld import GridWorld
from ie692_rl.envs.inventory import RetailInventory
from ie692_rl.types import FloatArray, IntArray


class FeatureMap(NamedTuple):
    feature_of: IntArray  # (n_states, n_actions) -> feature id
    n_features: int

    def dense(self) -> FloatArray:
        """The equivalent dense matrix Phi[(s * A + a), feature] (for small problems and tests)."""
        S, A = self.feature_of.shape
        Phi = np.zeros((S * A, self.n_features))
        Phi[np.arange(S * A), self.feature_of.ravel()] = 1.0
        return Phi


def tabular_features(env: Environment) -> FeatureMap:
    """One independent parameter for every (state, action) pair."""
    S, A = env.n_states, env.n_actions
    return FeatureMap(np.arange(S * A, dtype=np.int64).reshape(S, A), S * A)


def linear_features(env: Environment) -> FeatureMap:
    """State aggregation: share one parameter across nearby states, separately for each action.

    GridWorld: every 2x2 block of cells. Blackjack: player totals 4-7 and 8-11 for each dealer card.
    Inventory: pairs of neighbouring stock levels (0-1, 2-3, ...) on the same weekday.
    """
    S, A = env.n_states, env.n_actions
    states = np.arange(S, dtype=np.int64)
    if isinstance(env, GridWorld):
        n_block_cols = (env.cols + 1) // 2
        group = (states // env.cols // 2) * n_block_cols + (states % env.cols) // 2
        n_groups = ((env.rows + 1) // 2) * n_block_cols
    elif isinstance(env, Blackjack):  # states are (total - 4) * 4 + dealer - 2
        group = (states // 16) * 4 + states % 4
        n_groups = 8
    elif isinstance(env, RetailInventory):
        stock, day = env.decode(states)
        n_pairs = (env.n_stock + 1) // 2
        group, n_groups = day * n_pairs + stock // 2, env.n_days * n_pairs
    else:
        raise ValueError(f"No aggregated features defined for {type(env).__name__}")
    return FeatureMap(group[:, None] * A + np.arange(A, dtype=np.int64)[None, :], n_groups * A)


FEATURES = {"tabular": tabular_features, "linear": linear_features}
