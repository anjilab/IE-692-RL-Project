"""True Online TD(lambda) with linear features and Dutch traces."""


def update(theta, trace, q_old, phi, phi_next, cost, gamma, alpha, lam):
    q = float(phi @ theta)
    q_next = float(phi_next @ theta)
    delta = cost + gamma * q_next - q
    old_trace_dot = float(trace @ phi)
    trace *= gamma * lam
    trace += (1.0 - alpha * gamma * lam * old_trace_dot) * phi
    correction = q - q_old
    theta += alpha * (delta + correction) * trace
    theta -= alpha * correction * phi
    return q_next
