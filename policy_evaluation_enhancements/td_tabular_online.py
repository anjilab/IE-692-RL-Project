"""True Online TD(lambda), with tabular values"""


def update(theta, trace, q_old, z, z_next, cost, gamma, alpha, lam, terminal=False):
    q = float(theta[z])
    q_next = 0.0 if terminal else float(theta[z_next])
    delta = cost + gamma * q_next - q
    old_trace = float(trace[z])
    trace *= gamma * lam
    trace[z] += 1.0 - alpha * gamma * lam * old_trace
    correction = q - q_old
    theta += alpha * (delta + correction) * trace
    theta[z] -= alpha * correction
    return q_next
