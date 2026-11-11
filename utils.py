"""Shared online Monte Carlo evaluation, Euclidean SPMD, tuning and outputs.

The training baseline follows Assignment 2's dynamic coverage rule along one
on-policy stream. The generative Monte Carlo helper is retained only for
independent reference estimates and earlier validation tests.
"""
import argparse
import csv
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
import numpy as np


def draw_actions(policy, states, rng):
    cumulative = policy[states].cumsum(axis=1)
    cumulative[:, -1] = 1.0
    return (rng.random(len(states))[:, None] > cumulative).sum(axis=1)


def sample_returns(env, policy, states, horizon, rng, first_actions=None):
    states = np.array(states, dtype=int, copy=True)
    returns = np.zeros(len(states))
    active = np.arange(len(states))
    observations = 0
    for t in range(horizon):
        if not len(active):
            break
        actions = first_actions[active] if t == 0 and first_actions is not None else draw_actions(policy, states[active], rng)
        next_states, costs, done = env.step(states[active], actions, rng)
        observations += len(active)
        returns[active] += env.gamma ** t * costs
        states[active] = next_states
        active = active[~done]
    return returns, observations


def monte_carlo_q(env, policy, rollouts, horizon, rng):
    """Generative reference estimator; NOT the dynamic online training baseline."""
    states = np.repeat(np.arange(env.n_states), env.n_actions * rollouts)
    actions = np.tile(np.repeat(np.arange(env.n_actions), rollouts), env.n_states)
    returns, count = sample_returns(env, policy, states, horizon, rng, actions)
    return returns.reshape(env.n_states, env.n_actions, rollouts).mean(axis=2), count


def _training_start(env, rng):
    """GridWorld starts at its origin; blackjack starts with a fresh deal."""
    if hasattr(env, 'deal'):
        return int(env.deal(1, rng)[0])
    return int(env.initial_states(1, rng)[0])


def _collect_online_data(env, pi, s_last, T, rng, max_observations, required=None):
    """Follow pi continuously. With required=None, collect a fixed-size path.

    With a coverage mask, stop T-1 moves after the last required first visit.
    This gives T costs starting at that visit. Blackjack resets only after a
    real terminal hand; terminal flags prevent returns crossing between hands.
    """
    if T < 1 or max_observations < 0:
        raise ValueError('T must be positive and max_observations nonnegative')
    if pi.shape != (env.n_states, env.n_actions):
        raise ValueError('Policy must have shape (states, actions)')
    if np.any(pi < 0) or not np.allclose(pi.sum(axis=1), 1):
        raise ValueError('Policy rows must be probability distributions')
    s_t = _training_start(env, rng) if s_last is None else int(s_last)
    first_time_sa = np.full(pi.shape, -1, dtype=int)
    unvisited_sa = None if required is None else required.copy()
    remaining = 0 if required is None else int(required.sum())
    if required is not None and remaining == 0:
        raise ValueError('Coverage mask must contain at least one pair')
    states, actions, costs, next_states, terminals = [], [], [], [], []
    coverage_time = None
    target_length = max_observations
    # 1. Observe one move at a time under the unchanged policy.
    for t in range(max_observations):
        a_t = int(rng.choice(env.n_actions, p=pi[s_t]))
        if hasattr(env, 'step_one'):
            s_next, cost, done = env.step_one(s_t, a_t, rng)
        else:
            s_next, cost, done = env.step(np.array([s_t]), np.array([a_t]), rng)
            s_next, cost, done = int(s_next[0]), float(cost[0]), bool(done[0])
        states.append(s_t)
        actions.append(a_t)
        costs.append(cost)
        next_states.append(s_next)
        terminals.append(done)
        if first_time_sa[s_t, a_t] == -1:
            first_time_sa[s_t, a_t] = t
        # 2. Record first coverage of each sufficiently probable pair.
        if required is not None and unvisited_sa[s_t, a_t]:
            unvisited_sa[s_t, a_t] = False
            remaining -= 1
            if remaining == 0:
                coverage_time = t + 1
                target_length = coverage_time + T - 1
        # A new blackjack hand is not a continuation of the previous return.
        s_t = _training_start(env, rng) if done else s_next
        # 3. Once coverage and the truncated-return tail are available, stop.
        if required is not None and coverage_time is not None and t + 1 >= target_length:
            break
    observations = len(costs)
    complete = required is None or (coverage_time is not None and observations >= target_length)
    return dict(states=states, actions=actions, costs=costs, next_states=next_states,
                terminals=terminals, first_time_sa=first_time_sa, s_last=s_t,
                observations=observations, coverage_time=coverage_time,
                exploration_observations=observations if coverage_time is None else coverage_time,
                tail_observations=0 if coverage_time is None else observations-coverage_time,
                required_pairs=0 if required is None else int(required.sum()),
                visited_required_pairs=0 if required is None else int(required.sum())-remaining,
                complete=complete)


def get_data_with_dynamic_exploration_time(env, pi, s_last, unpi, T, rng, max_observations):
    """Assignment 2 coverage rule; our policy layout is [state, action]."""
    if not 0 < unpi <= 1 / env.n_actions:
        raise ValueError('Threshold must be in (0, 1 / number of actions]')
    return _collect_online_data(env, pi, s_last, T, rng, max_observations, pi >= unpi)


def truncated_first_visit_q(data, pi, gamma, T, unpi, max_cost=1.0):
    """Construct Q only when all required first visits have complete tails.

    Low-probability actions get the assignment's pessimistic upper cost bound.
    They are not incorrectly presented as observed estimates.
    """
    if not data['complete']:
        raise ValueError('Incomplete coverage or tail; no policy update is allowed')
    nS, nA = pi.shape
    Q = np.full((nS, nA), max_cost / (1 - gamma))
    for s in range(nS):
        for a in range(nA):
            if pi[s, a] < unpi:
                continue
            t_0 = data['first_time_sa'][s, a]
            if t_0 < 0:
                raise ValueError('Missing required first visit')
            Q[s, a] = 0.0
            for k in range(T):
                # Discount relative to the visit, not absolute exploration time.
                t = t_0 + k
                Q[s, a] += gamma ** k * data['costs'][t]
                if data['terminals'][t]:
                    break
    return Q


def online_data_to_paths(data, pi, rng):
    """Convert the observed stream to the TD notebook's transition format."""
    paths = []
    for t in range(data['observations']):
        terminal = data['terminals'][t]
        ns = data['next_states'][t]
        if terminal:
            next_action = 0  # Unused: terminal continuation is zero.
        elif t + 1 < data['observations']:
            next_action = data['actions'][t+1]
        else:
            next_action = int(rng.choice(pi.shape[1], p=pi[ns]))
        reset = t == 0 or data['terminals'][t-1]
        paths.append((data['states'][t], data['actions'][t], data['costs'][t],
                      ns, next_action, terminal, reset))
    return paths


def collect_on_policy_steps(env, pi, budget, rng, s_last=None):
    """Fixed observation budget for TD, without forced state/action starts."""
    data = _collect_online_data(env, pi, s_last, 1, rng, budget)
    return online_data_to_paths(data, pi, rng), data['s_last']


def estimate_Q_online_mc_dynamic(env, pi, T, rng, max_observations,
                                 s_last=None, coverage_rounds=1, return_paths=False):
    """Truncated on-policy MC with a coverage-dependent exploration time.

    Average first-visit estimates across complete coverage rounds. If any round
    hits the sample cap, return Q=None and charge every move already collected.
    """
    if coverage_rounds < 1 or max_observations < 0:
        raise ValueError('Invalid coverage rounds or observation budget')
    unpi = (1 - env.gamma) / env.n_actions
    Q_sum = np.zeros_like(pi)
    info = dict(observations=0, exploration_observations=0, tail_observations=0,
                coverage_rounds=coverage_rounds, completed_rounds=0,
                required_pairs=int((pi >= unpi).sum()), visited_required_pairs=0,
                threshold=unpi, complete=False, status='budget_exhausted')
    if return_paths:
        info['paths'] = []
    for _ in range(coverage_rounds):
        data = get_data_with_dynamic_exploration_time(
            env, pi, s_last, unpi, T, rng, max_observations-info['observations'])
        s_last = data['s_last']
        for key in ['observations', 'exploration_observations', 'tail_observations']:
            info[key] += data[key]
        info['visited_required_pairs'] = data['visited_required_pairs']
        if return_paths:
            info['paths'].extend(online_data_to_paths(data, pi, rng))
        if not data['complete']:
            info['status'] = 'incomplete_coverage' if data['coverage_time'] is None else 'incomplete_return_tail'
            return None, s_last, info
        Q_sum += truncated_first_visit_q(data, pi, env.gamma, T, unpi)
        info['completed_rounds'] += 1
    info['complete'] = True
    info['status'] = 'complete'
    return Q_sum / coverage_rounds, s_last, info


def mirror_step(policy, q, step_size):
    """Euclidean SPMD: project policy - step_size * Q onto probability rows.

    Minimizes step_size * <Q, p> + 0.5 * ||p - old_policy||^2.
    This is the plain baseline; no entropy term or constraint penalty is added.
    """
    proposal = policy - step_size * q
    sorted_rows = np.sort(proposal, axis=1)[:, ::-1]
    shifted_sums = np.cumsum(sorted_rows, axis=1) - 1.0
    counts = np.arange(1, policy.shape[1] + 1)
    positive = sorted_rows - shifted_sums / counts > 0
    rho = positive.sum(axis=1) - 1
    threshold = shifted_sums[np.arange(len(policy)), rho] / (rho + 1)
    return np.maximum(proposal - threshold[:, None], 0.0)

def evaluate_start(env, policy, rollouts, horizon, rng):
    states = env.initial_states(rollouts, rng)
    returns, count = sample_returns(env, policy, states, horizon, rng)
    return float(returns.mean()), float(returns.std(ddof=1) / np.sqrt(rollouts)), count


def train(env, budget, rollouts, horizon, eval_rollouts, eval_horizon, step_size, seed,
          record_each=True):
    """rollouts now means complete coverage rounds per policy evaluation."""
    training_seed, evaluation_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(training_seed)
    eval_rng = np.random.default_rng(evaluation_seed)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    history = []
    train_total = eval_total = iteration = 0
    s_last = None

    def record(train_count, info):
        nonlocal eval_total
        value, stderr, eval_count = evaluate_start(env, policy, eval_rollouts, eval_horizon, eval_rng)
        eval_total += eval_count
        history.append(dict(iteration=iteration, start_cost=value, standard_error=stderr,
                            training_observations=train_count, evaluation_observations=eval_count,
                            cumulative_training_observations=train_total,
                            cumulative_observations=train_total + eval_total,
                            exploration_observations=info.get('exploration_observations', 0),
                            tail_observations=info.get('tail_observations', 0),
                            required_pairs=info.get('required_pairs', 0),
                            visited_required_pairs=info.get('visited_required_pairs', 0),
                            completed_rounds=info.get('completed_rounds', 0),
                            status=info['status']))

    if record_each:
        record(0, {'status': 'initial'})
    while train_total < budget:
        Q_t, s_last, info = estimate_Q_online_mc_dynamic(
            env, policy, horizon, rng, budget-train_total, s_last, rollouts)
        count = info['observations']
        train_total += count
        if info['complete']:
            policy = mirror_step(policy, Q_t, step_size)
            iteration += 1
        if record_each:
            record(count, info)
        if not info['complete']:
            break
    if not record_each:
        record(train_total, info)
    return policy, history


def write_csv(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_learning(history, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    iterations = [r['iteration'] for r in history]
    axes[0].errorbar(iterations, [r['start_cost'] for r in history],
                     yerr=[1.96*r['standard_error'] for r in history], capsize=2)
    axes[0].set(xlabel='Policy update', ylabel='Discounted cost (lower is better)',
                title='Performance (±1.96 standard errors)')
    for key, label in [('training_observations', 'Training'), ('evaluation_observations', 'Evaluation')]:
        axes[1].plot(iterations, [r[key] for r in history], label=label)
    axes[1].set(xlabel='Policy update', ylabel='Sampled moves per iteration', title='Observations')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'learning.png', dpi=160)
    plt.close(fig)


def run_experiment(env, name, rollout_values, horizon_values, budget, eval_horizon, board_plot=None):
    parser = argparse.ArgumentParser(description=f'Tuned Monte Carlo + SPMD baseline: {name}')
    parser.add_argument('--coverage-rounds', '--rollouts', dest='rollouts', type=int, nargs='+', default=rollout_values, help='Complete coverage rounds averaged per update')
    parser.add_argument('--horizons', type=int, nargs='+', default=horizon_values)
    parser.add_argument('--step-sizes', type=float, nargs='+', default=[0.1, 0.5, 1.0])
    parser.add_argument('--budget', type=int, default=budget, help='Full-run training-move budget')
    parser.add_argument('--tune-fraction', type=float, default=0.1, help='Per-candidate fraction of full budget')
    parser.add_argument('--eval-rollouts', type=int, default=1000)
    parser.add_argument('--eval-horizon', type=int, default=eval_horizon,
                        help='Fixed scoring length shared by all candidates')
    parser.add_argument('--seed', type=int, default=692)
    parser.add_argument('--output', type=Path, default=Path(f'{name}_dynamic_baseline_results'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if min(args.rollouts + args.horizons + [args.budget, args.eval_horizon]) < 1 or args.eval_rollouts < 2:
        parser.error('Counts must be positive and eval-rollouts at least 2')
    if args.seed < 0 or not 0 < args.tune_fraction <= 1 or any(not np.isfinite(x) or x <= 0 for x in args.step_sizes):
        parser.error('Invalid seed, tuning fraction, or step size')
    if args.eval_horizon < max(args.horizons):
        parser.error('Evaluation horizon must be at least the largest training horizon')
    tune_budget = int(args.budget * args.tune_fraction)
    if tune_budget < 1:
        parser.error('Tuning budget must be positive')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new --output directory')
    args.output.mkdir(parents=True)
    tuning = []
    for rollouts, horizon, step_size in product(args.rollouts, args.horizons, args.step_sizes):
        _, history = train(env, tune_budget, rollouts, horizon, args.eval_rollouts,
                           args.eval_horizon, step_size, args.seed, record_each=False)
        last = history[-1]
        tuning.append(dict(rollouts=rollouts, horizon=horizon, step_size=step_size,
                           start_cost=last['start_cost'], standard_error=last['standard_error'],
                           iterations=last['iteration'], status=last['status'],
                           required_pairs=last['required_pairs'], visited_required_pairs=last['visited_required_pairs'],
                           training_budget=tune_budget,
                           training_observations=last['cumulative_training_observations'],
                           evaluation_observations=last['evaluation_observations']))
        print(f'Tuning rounds={rollouts}, T={horizon}, step={step_size}: cost={last["start_cost"]:.5f}, updates={last["iteration"]}, {last["status"]}', flush=True)
    write_csv(args.output / 'tuning.csv', tuning)
    valid = [row for row in tuning if row['iterations'] > 0]
    if not valid:
        config = vars(args).copy()
        config.update(output=str(args.output), environment=asdict(env), name=name,
                      evaluator='truncated_on_policy_mc_dynamic_exploration',
                      status='no_candidate_completed_an_update',
                      tuning_budget_per_candidate=tune_budget,
                      total_tuning_observations=sum(r['training_observations']+r['evaluation_observations'] for r in tuning))
        (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
        print('No candidate completed coverage and its return tail. No tuned policy claimed; see tuning.csv.')
        return
    best = min(valid, key=lambda r: r['start_cost'])
    policy, history = train(env, args.budget, best['rollouts'], best['horizon'], args.eval_rollouts,
                            args.eval_horizon, best['step_size'], args.seed + 1)
    write_csv(args.output / 'tuning.csv', tuning)
    write_csv(args.output / 'history.csv', history)
    write_csv(args.output / 'policy.csv', [dict(state=s, **env.state_label(s),
              **{a: float(p) for a, p in zip(env.action_names, row)}) for s, row in enumerate(policy)])
    config = vars(args).copy()
    config['output'] = str(args.output)
    config.update(environment=asdict(env), name=name, selected_settings=best,
                  evaluator='truncated_on_policy_mc_dynamic_exploration',
                  coverage_threshold=(1-env.gamma)/env.n_actions,
                  policy_update='unchanged Euclidean SPMD',
                  final_status=history[-1]['status'],
                  evaluation_start='fixed square 0' if name == 'gridworld' else 'player total 8, dealer showing 3',
                  training_seed=args.seed+1, tuning_budget_per_candidate=tune_budget,
                  total_tuning_observations=sum(r['training_observations']+r['evaluation_observations'] for r in tuning),
                  full_run_observations=history[-1]['cumulative_observations'],
                  training_tail_bound=env.gamma**best['horizon']/(1-env.gamma) if name=='gridworld' else None,
                  notes=['Assignment 2 dynamic exploration: first visits to all pairs with pi >= (1-gamma)/n_actions, followed by T-1 tail moves.',
                         'rollouts denotes complete coverage rounds per update, not forced starts at state/action pairs.',
                         'Training follows the policy continuously; blackjack restarts only with a fresh deal after a terminal hand.',
                         'Returns stop at terminal hands and use gamma raised to time since the first visit.',
                         'Actions below the threshold receive the pessimistic bound 1/(1-gamma).',
                         'Incomplete coverage or tails consume budget but never trigger a policy update.',
                         'The last history row can be an incomplete attempt on the unchanged policy.',
                         'Evaluation samples are additional and counted separately; previous generative results are not comparable.'])
    (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Selected: coverage_rounds={best["rollouts"]}, horizon={best["horizon"]}, step={best["step_size"]}')
    print(f'Cost: {history[0]["start_cost"]:.5f} -> {history[-1]["start_cost"]:.5f}; saved to {args.output}')
    if args.plot:
        try:
            plot_learning(history, args.output)
            if board_plot:
                board_plot(env, args.output)
        except ImportError as error:
            print(f'Plotting unavailable: {error}. Numeric results were saved.')
