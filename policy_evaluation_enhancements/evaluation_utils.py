"""Shared sampling, scoring, saving, and plotting used by policy_eval.ipynb."""

import csv
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from . import td_tabular, td_tabular_online, td_linear, td_linear_online, tomc
from .utils import mirror_step

ROOT = Path(__file__).resolve().parent / 'results'
METHODS = ['baseline', 'td_tabular_0', 'td_tabular', 'td_tabular_online', 'td_linear', 'td_linear_online']
LABELS = ['TOMC (dynamic exploration)', 'TD(0), tabular', 'TD(λ), tabular', 'True Online TD(λ), tabular',
          'TD(λ), linear', 'True Online TD(λ), linear']
UPDATES = {'td_tabular_0': td_tabular.update, 'td_tabular': td_tabular.update,
           'td_tabular_online': td_tabular_online.update,
           'td_linear': td_linear.update, 'td_linear_online': td_linear_online.update}


def result_folder(method, environment):
    if method not in METHODS or environment not in ['gridworld', 'blackjack']:
        raise ValueError('Choose a listed method and gridworld or blackjack.')
    return ROOT / method / environment


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def initial_state(env, rng):
    if hasattr(env, 'rows'):
        return env.start
    player = int(rng.integers(2, 6)) + int(rng.integers(2, 6))
    dealer = int(rng.integers(2, 6))
    return int(env.encode(player, dealer))


def action(cumulative_policy, s, rng):
    return min(int(np.searchsorted(cumulative_policy[s], rng.random(), side='right')),
               cumulative_policy.shape[1] - 1)


@lru_cache(maxsize=None)
def reference_model(env):
    # Diagnostic model only. Learning below calls env.step_one, never this model.
    P = np.zeros((env.n_states, env.n_states, env.n_actions))
    costs = np.zeros((env.n_states, env.n_actions))
    if hasattr(env, 'rows'):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for s in range(env.n_states):
            special = s in env.goals or s in env.hazards
            reset = env.reset_probability if special else 0.0
            costs[s] = 0.0 if s in env.goals else 1.0 if s in env.hazards else 0.2
            for a in range(env.n_actions):
                P[env.start, s, a] += reset
                for move, (dr, dc) in enumerate(directions):
                    prob = env.intended_probability if a == move else (1-env.intended_probability)/3
                    row = min(env.rows-1, max(0, s//env.cols + dr))
                    col = min(env.cols-1, max(0, s%env.cols + dc))
                    P[row*env.cols + col, s, a] += (1-reset)*prob
    else:
        @lru_cache(maxsize=None)
        def dealer_cost(player, total):
            if total >= 9:
                return -1.0 if total > 11 else float(np.sign(total-player))
            return sum(dealer_cost(player, total+card) for card in range(2, 6))/4
        for s in range(env.n_states):
            player, dealer = map(int, env.decode(s))
            costs[s, 0] = sum(dealer_cost(player, dealer+card) for card in range(2, 6))/4
            for card in range(2, 6):
                if player+card > 11:
                    costs[s, 1] += 0.25
                else:
                    P[int(env.encode(player+card, dealer)), s, 1] += 0.25
    return P, costs


def exact_value(P, costs, policy, gamma):
    P_pi = np.einsum('nsa,sa->sn', P, policy)
    V = np.linalg.solve(np.eye(len(policy))-gamma*P_pi, (costs*policy).sum(axis=1))
    Q = costs + gamma*np.einsum('nsa,n->sa', P, V)
    return V, Q


def references(env):
    P, costs = reference_model(env)
    policy = np.full(costs.shape, 1/env.n_actions)
    _, Q_pi = exact_value(P, costs, policy, env.gamma)
    P_pi = np.einsum('nsa,sa->sn', P, policy)
    restart = np.zeros(env.n_states)
    if hasattr(env, 'rows'):
        restart[env.start] = 1.0
    else:
        for c1, c2, dealer in product(range(2, 6), repeat=3):
            restart[int(env.encode(c1+c2, dealer))] += 1/64
    stream = P_pi + (1-P_pi.sum(axis=1))[:, None]*restart
    A = stream.T - np.eye(env.n_states)
    A[-1] = 1.0
    b = np.zeros(env.n_states)
    b[-1] = 1.0
    nu = np.linalg.solve(A, b)
    if np.min(nu) < -1e-10 or not np.allclose(nu @ stream, nu, atol=1e-10):
        raise ValueError('Invalid reference weights.')
    nu = np.maximum(nu, 0)
    nu /= nu.sum()
    V = np.zeros(env.n_states)
    for _ in range(10000):
        V_next = np.min(costs + env.gamma*np.einsum('nsa,n->sa', P, V), axis=1)
        if np.max(np.abs(V_next-V)) < 1e-12:
            return P, costs, Q_pi, nu[:, None]*policy, V_next
        V = V_next
    raise RuntimeError('Reference value iteration did not converge.')


def run(env, method, task, n_steps, seed, alpha, lam, horizon, step_size=0.5,
        update_every=10000, verbose=True):
    """One method, one seed, one task. No tuning or other game runs implicitly."""
    if method not in METHODS or task not in ['evaluation', 'policy_improvement']:
        raise ValueError('Unknown method or task.')
    if n_steps < 1 or horizon < 1 or update_every < 1:
        raise ValueError('Step counts and horizon must be positive.')
    if method == 'td_tabular_0':
        lam = 0.0
    P, costs, Q_pi, weights, V_star = references(env)
    policy = np.full(costs.shape, 1/env.n_actions)
    cumulative = policy.cumsum(axis=1)
    rng = np.random.default_rng(seed)
    s = initial_state(env, rng)
    a = action(cumulative, s, rng)
    s0 = env.start if hasattr(env, 'rows') else int(env.encode(env.evaluation_total, env.evaluation_dealer))
    Q = np.zeros_like(policy)
    completed = updates = 0
    q_old = 0.0
    if method == 'baseline':
        data = tomc.start(policy, env.gamma)
    else:
        linear = method in ['td_linear', 'td_linear_online']
        Phi = td_linear.features(env) if linear else None
        theta = np.zeros(Phi.shape[1] if linear else policy.size)
        trace = np.zeros_like(theta)
        zero = np.zeros_like(theta)
    checkpoints = set(np.linspace(0, n_steps, 51, dtype=int))
    records = []

    def record(t):
        nonlocal Q
        if method != 'baseline':
            Q = (Phi @ theta if linear else theta).reshape(policy.shape)
            if not np.isfinite(Q).all():
                raise FloatingPointError('TD values became non-finite. Reduce the learning rate.')
        ready = method != 'baseline' or completed > 0
        status = 'ready'
        if method == 'baseline' and (task == 'policy_improvement' or not ready):
            status = 'waiting_for_coverage' if data['remaining'] else 'waiting_for_return_tail'
        error = Q-Q_pi
        row = dict(seed=seed, observations=t, status=status, estimate_ready=ready,
                   completed_rounds=completed, pending_pairs=data['remaining'] if method=='baseline' else 0,
                   weighted_error=float(np.sqrt(np.sum(weights*error*error))) if ready and task=='evaluation' else '',
                   uniform_rmse=float(np.sqrt(np.mean(error*error))) if ready and task=='evaluation' else '',
                   policy_updates=updates, exact_value='', estimated_value='', gap='',
                   plot_checkpoint=t in checkpoints)
        if task == 'policy_improvement':
            V, _ = exact_value(P, costs, policy, env.gamma)
            gap = float(V[s0]-V_star[s0])
            if gap < -1e-9:
                raise ValueError('Reference produced a negative optimality gap.')
            row.update(exact_value=float(V[s0]), gap=max(0.0, gap),
                       estimated_value=float(policy[s0] @ Q[s0]) if ready else '')
        records.append(row)

    record(0)
    for t in range(1, n_steps+1):
        s_next, cost, terminal = env.step_one(s, a, rng)
        next_state = initial_state(env, rng) if terminal else s_next
        a_next = action(cumulative, next_state, rng)
        improve = False
        if method == 'baseline':
            if tomc.observe(data, s, a, cost, terminal, horizon):
                Q_round = tomc.estimate(data, env.gamma, horizon)
                completed += 1
                if task == 'evaluation':
                    Q += (Q_round-Q)/completed
                else:
                    Q = Q_round
                    improve = True
                data = tomc.start(policy, env.gamma)
        else:
            z, z_next = s*env.n_actions+a, s_next*env.n_actions+a_next
            if linear:
                q_old = UPDATES[method](theta, trace, q_old, Phi[z], zero if terminal else Phi[z_next],
                                        cost, env.gamma, alpha, lam)
            else:
                q_old = UPDATES[method](theta, trace, q_old, z, z_next, cost, env.gamma, alpha, lam, terminal)
            if terminal:
                trace.fill(0)
                q_old = 0.0
            if task == 'policy_improvement' and t % update_every == 0:
                Q = (Phi @ theta if linear else theta).reshape(policy.shape)
                improve = True
        if improve:
            policy = mirror_step(policy, Q, step_size)
            cumulative = policy.cumsum(axis=1)
            updates += 1
            if method == 'baseline':
                data = tomc.start(policy, env.gamma)
            else:
                trace.fill(0)
                q_old = 0.0
            a_next = action(cumulative, next_state, rng)
            record(t)
        s, a = next_state, a_next
        if t in checkpoints and records[-1]['observations'] != t:
            record(t)
        if verbose and t % max(1, n_steps//10) == 0:
            pending = data['remaining'] if method == 'baseline' else 0
            print(f'{method}, seed {seed}: {t}/{n_steps} steps; missing pairs={pending}', flush=True)
    return records


def tune(env, method, n_steps, seeds, starting, update_every):
    """Optional tuning: mean error over all three tuning seeds, never test seeds."""
    if method == 'baseline':
        horizons = [40, 70, 100] if hasattr(env, 'rows') else [2, 3, 4]
        candidates = [dict(starting, horizon=T) for T in horizons]
    else:
        lambdas = [0.0] if method == 'td_tabular_0' else [0.3, 0.6, 0.9]
        candidates = [dict(starting, alpha=alpha, lam=lam)
                      for alpha, lam in product([0.01, 0.05, 0.1], lambdas)]
    rows, summaries = [], []
    for params in candidates:
        scores = []
        for seed in seeds:
            history = run(env, method, 'evaluation', n_steps, seed, **params,
                          update_every=update_every, verbose=False)
            last = history[-1]
            score = last['weighted_error']
            scores.append(score)
            rows.append(dict(seed=seed, alpha=params['alpha'], lam=params['lam'],
                             horizon=params['horizon'], step_size=params['step_size'],
                             score=score, status=last['status']))
        eligible = all(score != '' and np.isfinite(score) for score in scores)
        mean = float(np.mean(scores)) if eligible else ''
        summaries.append(dict(**params, mean_error=mean, eligible=eligible))
        print(f'Tuning {method}: alpha={params["alpha"]}, lambda={params["lam"]}, '
              f'horizon={params["horizon"]}, mean error={mean if eligible else "coverage incomplete"}', flush=True)
    valid = [r for r in summaries if r['eligible']]
    if not valid:
        return dict(starting), rows, summaries, 'starting values: tuning coverage incomplete'
    best = min(valid, key=lambda r:r['mean_error'])
    params = {key: best[key] for key in starting}
    return params, rows, summaries, 'selected by mean over tuning seeds'


def plot_results(environment, task, seeds, methods=None):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    filename = 'evaluation.csv' if task == 'evaluation' else 'policy_improvement.csv'
    metric = 'weighted_error' if task == 'evaluation' else 'gap'
    fig, ax = plt.subplots(figsize=(10, 4.5))
    plotted = 0
    selected = METHODS if methods is None else list(methods)
    if not selected or any(method not in METHODS for method in selected):
        raise ValueError('Choose at least one listed method to plot.')
    colors = ['#eb6834', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b']
    styles = ['-', '--', '-', '-.', '--', '-']
    for method, label, color, style in zip(METHODS, LABELS, colors, styles):
        if method not in selected:
            continue
        path = result_folder(method, environment) / filename
        if not path.exists():
            continue
        with path.open() as f:
            rows = [r for r in csv.DictReader(f) if int(r['seed']) in seeds and r['plot_checkpoint']=='True']
        xs = sorted({int(r['observations']) for r in rows})
        means, errors = [], []
        for x in xs:
            point = [r for r in rows if int(r['observations']) == x]
            ready = {int(r['seed']) for r in point} == set(seeds) and all(r[metric] != '' for r in point)
            values = np.array([float(r[metric]) for r in point]) if ready else np.array([])
            ready = ready and np.isfinite(values).all()
            means.append(float(values.mean()) if ready else np.nan)
            errors.append(float(values.std(ddof=1)/np.sqrt(len(values))) if ready and len(values)>1 else 0.0)
        if not np.isfinite(means).any():
            print(f'{method}: no complete {task} estimate for all selected seeds yet.')
            continue
        x = np.array(xs) / (1e6 if environment=='gridworld' else 1e3)
        mean, se = np.array(means), np.array(errors)
        ax.plot(x, mean, color=color, linestyle=style, label=label, linewidth=2.6 if method=='baseline' else 2)
        ax.fill_between(x, np.maximum(0, mean-se), mean+se, color=color, alpha=0.15)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print('No completed results to plot. Run a method first; TOMC may still be waiting for coverage.')
        return
    ax.set_title(f'{environment.title()} — {"policy evaluation" if task=="evaluation" else "same SPMD, different evaluators"}', loc='left')
    ax.set_xlabel(f'Environment steps ({"millions" if environment=="gridworld" else "thousands"})')
    ax.set_ylabel(r"$\|\tilde{Q}^\pi - Q^\pi\|_W$" if task=='evaluation' else 'Gap to optimal cost')
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#e1e0d9', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8)
    title = (f'Mean of {len(seeds)} test seeds; band = ±1 standard error'
             if len(seeds)>1 else f'Test seed {seeds[0]}')
    fig.suptitle(title)
    fig.tight_layout()
    folder = ROOT / 'figures' / environment
    folder.mkdir(parents=True, exist_ok=True)
    name = 'policy_evaluation_error' if task=='evaluation' else 'policy_evaluation_policy_gap'
    for suffix in ['png', 'pdf']:
        fig.savefig(folder / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.show()
