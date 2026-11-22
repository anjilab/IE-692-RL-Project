import numpy as np

# Here
# pi_k : policy matrix at kth step |S| x |A|
# q_k: action-value matrix at kth step |S| x |A|
# eta: step size for PMD update


def entropy_reg_spmd(pi_k, q_k, eta, tau=0.0):
    """
    Run one Policy Mirror Descent (PMD) step with KL divergence and optional entropy regularization.

    Parameters:
    pi_k : numpy.ndarray
        The policy matrix at the kth step, shape (|S|, |A|).
    q_k : numpy.ndarray
        The action-value matrix at the kth step, shape (|S|, |A|). Treated as rewards (higher is better).
        With tau > 0 it should be the entropy-regularized Q of pi_k.
    eta : float
        The step size for the PMD update.
    tau : float
        Entropy weight. 0 gives the plain KL update; larger values keep the policy more random.

    Returns:
    numpy.ndarray
        The updated policy, shape (|S|, |A|), each row summing to 1.
    """

    assert (
        pi_k.shape == q_k.shape
    ), "Policy and action-value matrices must have the same shape."
    assert tau >= 0, "Entropy weight tau must be non-negative."

    pi_k_next = pi_k ** (1 / (1 + eta * tau)) * np.exp(-eta * q_k / (1 + eta * tau))

    # normalization; row summing to 1
    pi_k_next /= np.sum(pi_k_next, axis=1, keepdims=True)

    return pi_k_next