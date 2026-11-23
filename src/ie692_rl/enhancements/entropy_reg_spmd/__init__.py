"""Enhancement 3: entropy-regularized SPMD (see ``entropy.py`` for where each piece lives)."""

from ie692_rl.enhancements.entropy_reg_spmd.entropy import AnnealedTau, soft_optimal_policy

__all__ = ["AnnealedTau", "soft_optimal_policy"]
