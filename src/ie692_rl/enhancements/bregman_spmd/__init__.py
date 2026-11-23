"""Enhancement 1: SPMD with three Bregman divergences.

KL          w(p) = sum p log p        closed form; the baseline step, and with tau > 0 the
                                      entropy-regularized step (enhancement 3)
Euclidean   w(p) = ||p||^2 / 2        exact sort-based projection onto the simplex
Tsallis     w(p) = (sum p^q - 1) / (q (q - 1))   bracketed Newton root-finding per state
"""

from ie692_rl.enhancements.bregman_spmd.euclidean import Euclidean, project_to_simplex
from ie692_rl.enhancements.bregman_spmd.kl import KL
from ie692_rl.enhancements.bregman_spmd.tsallis import Tsallis, index_from_gamma

__all__ = ["KL", "Euclidean", "Tsallis", "index_from_gamma", "project_to_simplex"]
