"""KL divergence, w(p) = sum p log p: the baseline SPMD step (tau = 0) and, with tau > 0, the
entropy-regularized step of enhancement 3."""

import numpy as np

from ie692_rl.core.divergence import BregmanDivergence
from ie692_rl.types import FloatArray


class KL(BregmanDivergence):
    """KL divergence, w(p) = sum p log p. The baseline SPMD step (tau = 0) and the entropy enhancement.

    Closed form: pi_{k+1}(a|s) proportional to pi_k(a|s)^(1/(1+eta tau)) * exp(-eta Q(s,a) / (1+eta tau)).
    Computed in log space with each row shifted so that the largest term is exp(0), which keeps
    every step size finite and NaN-free. Actions with probability 0 keep probability 0.
    """

    name = "kl"

    def divergence(self, p: FloatArray, center: FloatArray) -> FloatArray:
        """sum p log(p / center); +inf where p puts mass on an action that ``center`` excludes."""
        with np.errstate(divide="ignore"):
            log_ratio = np.log(np.where(p > 0, p, 1.0)) - np.log(np.where(p > 0, center, 1.0))
        terms = np.where(p > 0, p * log_ratio, 0.0)
        result: FloatArray = terms.sum(axis=1)
        return result

    def step(self, policy: FloatArray, q: FloatArray, eta: float, tau: float = 0.0) -> FloatArray:
        self.check_step_inputs(policy, q, eta, tau)
        support = policy > 0
        # Shifting a row of Q by a constant does not change the update; it keeps eta * Q bounded
        # for the best action. Weights are written so that eta = inf or eta * tau = inf stay defined.
        eta_tau = 0.0 if tau == 0 else eta * tau
        weight_old = 1.0 / (1.0 + eta_tau)  # exponent on pi_k
        weight_q = eta if tau == 0 else 1.0 / (1.0 / eta + tau)  # eta / (1 + eta tau)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            shifted = q - q.min(axis=1, keepdims=True)
            logits = np.where(
                support, weight_old * np.log(np.where(support, policy, 1.0)) - weight_q * shifted, -np.inf
            )
        row_max = logits.max(axis=1, keepdims=True)
        finite = np.isfinite(row_max[:, 0])
        new = np.zeros_like(policy)
        with np.errstate(under="ignore"):  # vanishing actions underflow to probability 0, as intended
            new[finite] = np.exp(logits[finite] - row_max[finite])
        if not finite.all():  # eta so large that the logits overflowed: the greedy limit on the support
            masked_q = np.where(support[~finite], q[~finite], np.inf)
            new[~finite] = masked_q == masked_q.min(axis=1, keepdims=True)
        normalized: FloatArray = new / new.sum(axis=1, keepdims=True)
        return normalized
