"""Online, fixed-policy, cost-based action-value TD.

All three functions use the lecture notation Q_theta(s,a) = phi(s,a)^T theta.
The SAME functions work with tabular and linear Phi from features.py.
Each result is (Q, observation_count, history), where history stores
(observations, Q_snapshot) at the requested frequency.
"""
import numpy as np


def _start(env, pi, rng):
    # A new Blackjack hand; GridWorld always starts at its origin.
    s = int(env.deal(1, rng)[0]) if hasattr(env, "deal") else env.start
    a = int(rng.choice(env.n_actions, p=pi[s]))
    return s, a


def td_zero(env, pi, Phi, n_steps, eta, rng, record_every=100):
    """Stochastic TD(0): theta <- theta + eta * delta * phi."""
    S, A = env.n_states, env.n_actions
    theta = np.zeros(Phi.shape[1])
    history = []
    s, a = _start(env, pi, rng)

    for t in range(n_steps):
        next_s, cost, done = env.step_one(s, a, rng)
        phi = Phi[s * A + a]
        if done:
            phi_next = np.zeros_like(phi)
            next_a = None
        else:
            next_a = int(rng.choice(A, p=pi[next_s]))
            phi_next = Phi[next_s * A + next_a]

        # tilde_F_t = -delta_t * phi_t, as in Lecture 19 / Q2.
        delta = cost + env.gamma * (phi_next @ theta) - (phi @ theta)
        theta += eta * delta * phi

        if record_every and (t + 1) % record_every == 0:
            history.append((t + 1, (Phi @ theta).reshape(S, A).copy()))
        s, a = _start(env, pi, rng) if done else (next_s, next_a)

    Q = (Phi @ theta).reshape(S, A)
    return Q, n_steps, history


def td_lambda(env, pi, Phi, n_steps, eta, lam, rng, record_every=100):
    """Conventional TD(lambda): accumulating eligibility trace."""
    S, A = env.n_states, env.n_actions
    theta = np.zeros(Phi.shape[1])
    e = np.zeros_like(theta)
    history = []
    s, a = _start(env, pi, rng)

    for t in range(n_steps):
        next_s, cost, done = env.step_one(s, a, rng)
        phi = Phi[s * A + a]
        if done:
            phi_next = np.zeros_like(phi)
            next_a = None
        else:
            next_a = int(rng.choice(A, p=pi[next_s]))
            phi_next = Phi[next_s * A + next_a]

        delta = cost + env.gamma * (phi_next @ theta) - (phi @ theta)
        e = env.gamma * lam * e + phi
        theta += eta * delta * e

        if record_every and (t + 1) % record_every == 0:
            history.append((t + 1, (Phi @ theta).reshape(S, A).copy()))
        if done:
            e.fill(0)  # New Blackjack hand = new episode.
        s, a = _start(env, pi, rng) if done else (next_s, next_a)

    Q = (Phi @ theta).reshape(S, A)
    return Q, n_steps, history


def true_online_td_lambda(env, pi, Phi, n_steps, eta, lam, rng, record_every=100):
    """True Online TD(lambda) for Q: Dutch eligibility trace + correction."""
    S, A = env.n_states, env.n_actions
    theta = np.zeros(Phi.shape[1])
    e = np.zeros_like(theta)
    q_old = 0.0
    history = []
    s, a = _start(env, pi, rng)

    for t in range(n_steps):
        next_s, cost, done = env.step_one(s, a, rng)
        phi = Phi[s * A + a]
        if done:
            phi_next = np.zeros_like(phi)
            next_a = None
        else:
            next_a = int(rng.choice(A, p=pi[next_s]))
            phi_next = Phi[next_s * A + next_a]

        # Calculate values BEFORE the parameter update.
        q = phi @ theta
        q_next = phi_next @ theta
        delta = cost + env.gamma * q_next - q

        # Dutch trace, followed by true-online correction.
        e = env.gamma * lam * e + (1 - eta * env.gamma * lam * (e @ phi)) * phi
        theta += eta * (delta + q - q_old) * e - eta * (q - q_old) * phi
        q_old = q_next

        if record_every and (t + 1) % record_every == 0:
            history.append((t + 1, (Phi @ theta).reshape(S, A).copy()))
        if done:
            e.fill(0)
            q_old = 0.0
        s, a = _start(env, pi, rng) if done else (next_s, next_a)

    Q = (Phi @ theta).reshape(S, A)
    return Q, n_steps, history
