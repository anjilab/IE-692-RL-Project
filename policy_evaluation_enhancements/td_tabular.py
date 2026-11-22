"""Ordinary TD(lambda), with one value per state/action pair. """


def update(theta, trace, q_old, z, z_next, cost, gamma, alpha, lam, terminal=False):
    q = float(theta[z])
    q_next = 0.0 if terminal else float(theta[z_next])
    delta = cost + gamma * q_next - q
    trace *= gamma * lam
    trace[z] += 1.0
    theta += alpha * delta * trace
    return q_next
