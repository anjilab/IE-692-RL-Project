"""First-visit truncated on-policy Monte Carlo with dynamic exploration time."""

import numpy as np


def start(policy, gamma):
    threshold = (1.0 - gamma) / policy.shape[1]
    required = policy >= threshold
    return dict(first_visit=np.full(policy.shape, -1, dtype=int),
                required=required, remaining=int(required.sum()),
                costs=[], terminals=[], finish_at=None)


def observe(data, s, a, cost, terminal, horizon):
    t = len(data['costs'])
    data['costs'].append(cost)
    data['terminals'].append(terminal)
    if data['first_visit'][s, a] < 0:
        data['first_visit'][s, a] = t
        if data['required'][s, a]:
            data['remaining'] -= 1
            if data['remaining'] == 0:
                data['finish_at'] = t + horizon
    return data['finish_at'] is not None and t + 1 >= data['finish_at']


def estimate(data, gamma, horizon):
    if data['finish_at'] is None or len(data['costs']) < data['finish_at']:
        raise ValueError('TOMC is still waiting for coverage or return tails.')
    # The assignment's pessimistic bound is used for excluded actions.
    Q = np.full(data['first_visit'].shape, 1.0 / (1.0 - gamma))
    for s, a in np.argwhere(data['required']):
        t0 = data['first_visit'][s, a]
        total = 0.0
        for k in range(horizon):
            total += gamma ** k * data['costs'][t0 + k]
            if data['terminals'][t0 + k]:
                break
        Q[s, a] = total
    return Q
