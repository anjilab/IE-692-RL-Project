"""Squared Euclidean divergence, w(p) = ||p||^2 / 2, so D(p, c) = ||p - c||^2 / 2.

The SPMD step is the Euclidean projection of ``policy - eta * Q`` onto the simplex, computed exactly
with the sort-based algorithm of Duchi et al. (ICML 2008) for all states at once: O(|A| log |A|) per
state, no iteration. Unlike KL, actions can reach probability exactly 0 and can come back later.
"""

import numpy as np

from ie692_rl.core.divergence import BregmanDivergence
from ie692_rl.types import FloatArray


class Euclidean(BregmanDivergence):
    name = "euclidean"

    def divergence(self, p: FloatArray, center: FloatArray) -> FloatArray:
        result: FloatArray = 0.5 * ((p - center) ** 2).sum(axis=-1)
        return result

    def step(self, policy: FloatArray, q: FloatArray, eta: float, tau: float = 0.0) -> FloatArray:
        self.check_step_inputs(policy, q, eta, tau)
        if tau > 0:
            raise NotImplementedError("Entropy regularization (tau > 0) has a closed form only for KL")
        target = policy - self.scaled_costs(q, eta)
        # Entries more than 1 below the row maximum project to 0, so clipping them to max - 2 changes
        # nothing and keeps the sort and cumulative sums finite when eta * Q overflowed to inf.
        target = np.maximum(target, target.max(axis=1, keepdims=True) - 2.0)
        return project_to_simplex(target)


def project_to_simplex(v: FloatArray) -> FloatArray:
    """Row-wise argmin_{p in simplex} ||p - v||^2 (Duchi, Shalev-Shwartz, Singer & Chandra, 2008)."""
    n_rows, n_cols = v.shape
    descending = -np.sort(-v, axis=1)
    excess = np.cumsum(descending, axis=1) - 1.0
    positive = descending - excess / np.arange(1, n_cols + 1) > 0  # always True in column 0
    last = n_cols - 1 - np.argmax(positive[:, ::-1], axis=1)  # number of positive entries - 1
    threshold = excess[np.arange(n_rows), last] / (last + 1)
    projected = np.maximum(v - threshold[:, None], 0.0)
    normalized: FloatArray = projected / projected.sum(axis=1, keepdims=True)  # remove rounding drift
    return normalized
