"""Ordinary TD(lambda) with compact linear features: Q = Phi @ theta."""

import numpy as np


def features(env):
    s = np.arange(env.n_states)
    if hasattr(env, 'rows'):
        u = (s // env.cols) / (env.rows - 1)
        v = (s % env.cols) / (env.cols - 1)
        base = np.column_stack([np.ones(len(s)), u, v, u*u, v*v, u*v,
                                np.isin(s, env.goals), np.isin(s, env.hazards)])
    else:
        player, dealer = env.decode(s)
        u, v = (player - 4) / 7, (dealer - 2) / 3
        base = np.column_stack([np.ones(len(s)), u, v, u*u, v*v, u*v])
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    d = base.shape[1]
    Phi = np.zeros((env.n_states * env.n_actions, d * env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            Phi[s * env.n_actions + a, a*d:(a+1)*d] = base[s]
    return Phi


def update(theta, trace, q_old, phi, phi_next, cost, gamma, alpha, lam):
    q = float(phi @ theta)
    q_next = float(phi_next @ theta)
    delta = cost + gamma * q_next - q
    trace *= gamma * lam
    trace += phi
    theta += alpha * delta * trace
    return q_next
