"""Shared sampled evaluation, Euclidean SPMD, tuning, and result files.

Sampling assumption: each state/action can initialize a simulation (the supplied
notebook's generative-sampling interface). After that first action, every action
follows the fixed policy. This is not restricted to one live trajectory.
Lower costs are better. Training and scoring use separate random generators.
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
    states = np.repeat(np.arange(env.n_states), env.n_actions * rollouts)
    actions = np.tile(np.repeat(np.arange(env.n_actions), rollouts), env.n_states)
    returns, count = sample_returns(env, policy, states, horizon, rng, actions)
    return returns.reshape(env.n_states, env.n_actions, rollouts).mean(axis=2), count


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
    training_seed, evaluation_seed = np.random.SeedSequence(seed).spawn(2)
    rng, eval_rng = np.random.default_rng(training_seed), np.random.default_rng(evaluation_seed)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    history = []
    train_total = eval_total = iteration = 0
    # Reserve the worst-case length of a full evaluation sweep. This guarantees
    # the budget is never exceeded, even when blackjack episodes end early.
    sweep_bound = env.n_states * env.n_actions * rollouts * horizon
    if sweep_bound > budget:
        raise ValueError('Budget cannot fit one complete state/action evaluation sweep')

    def record(train_count):
        nonlocal eval_total
        value, stderr, eval_count = evaluate_start(env, policy, eval_rollouts, eval_horizon, eval_rng)
        eval_total += eval_count
        history.append(dict(iteration=iteration, start_cost=value, standard_error=stderr,
                            training_observations=train_count, evaluation_observations=eval_count,
                            cumulative_training_observations=train_total,
                            cumulative_observations=train_total + eval_total))

    if record_each:
        record(0)
    while train_total + sweep_bound <= budget:
        q, count = monte_carlo_q(env, policy, rollouts, horizon, rng)
        train_total += count
        policy = mirror_step(policy, q, step_size)
        iteration += 1
        if record_each:
            record(count)
    if not record_each:
        record(train_total)
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
    parser.add_argument('--rollouts', type=int, nargs='+', default=rollout_values)
    parser.add_argument('--horizons', type=int, nargs='+', default=horizon_values)
    parser.add_argument('--step-sizes', type=float, nargs='+', default=[0.1, 0.5, 1.0])
    parser.add_argument('--budget', type=int, default=budget, help='Full-run training-move budget')
    parser.add_argument('--tune-fraction', type=float, default=0.1, help='Per-candidate fraction of full budget')
    parser.add_argument('--eval-rollouts', type=int, default=1000)
    parser.add_argument('--eval-horizon', type=int, default=eval_horizon,
                        help='Fixed scoring length shared by all candidates')
    parser.add_argument('--seed', type=int, default=692)
    parser.add_argument('--output', type=Path, default=Path(f'{name}_baseline_results'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if min(args.rollouts + args.horizons + [args.budget, args.eval_horizon]) < 1 or args.eval_rollouts < 2:
        parser.error('Counts must be positive and eval-rollouts at least 2')
    if args.seed < 0 or not 0 < args.tune_fraction <= 1 or any(not np.isfinite(x) or x <= 0 for x in args.step_sizes):
        parser.error('Invalid seed, tuning fraction, or step size')
    if args.eval_horizon < max(args.horizons):
        parser.error('Evaluation horizon must be at least the largest training horizon')
    tune_budget = int(args.budget * args.tune_fraction)
    largest_sweep = env.n_states * env.n_actions * max(args.rollouts) * max(args.horizons)
    if tune_budget < largest_sweep:
        parser.error(f'Tuning budget must be at least {largest_sweep}; increase --budget')
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
                           iterations=last['iteration'], training_budget=tune_budget,
                           training_observations=last['cumulative_training_observations'],
                           evaluation_observations=last['evaluation_observations']))
        print(f'Tuning runs={rollouts}, length={horizon}, step={step_size}: {last["start_cost"]:.5f}', flush=True)
    best = min(tuning, key=lambda r: r['start_cost'])
    policy, history = train(env, args.budget, best['rollouts'], best['horizon'], args.eval_rollouts,
                            args.eval_horizon, best['step_size'], args.seed + 1)
    write_csv(args.output / 'tuning.csv', tuning)
    write_csv(args.output / 'history.csv', history)
    write_csv(args.output / 'policy.csv', [dict(state=s, **env.state_label(s),
              **{a: float(p) for a, p in zip(env.action_names, row)}) for s, row in enumerate(policy)])
    config = vars(args).copy()
    config['output'] = str(args.output)
    config.update(environment=asdict(env), name=name, selected_settings=best,
                  evaluation_start='fixed square 0' if name == 'gridworld' else 'player total 8, dealer showing 3',
                  training_seed=args.seed+1, tuning_budget_per_candidate=tune_budget,
                  total_tuning_observations=sum(r['training_observations']+r['evaluation_observations'] for r in tuning),
                  full_run_observations=history[-1]['cumulative_observations'],
                  training_tail_bound=env.gamma**best['horizon']/(1-env.gamma) if name=='gridworld' else None,
                  notes=['Each candidate has the same training-budget cap. Unused budget is less than one worst-case sweep.',
                         'Scoring samples are additional and counted separately. All candidates use the same scoring horizon.',
                         'Initial state/action sampling follows the notebook generative interface; subsequent actions are on-policy.',
                         'Blackjack training may truncate unfinished hands; scoring horizon defaults to 8 and completes every hand.',
                         'One seed is preliminary evidence; repeat runs before drawing final conclusions.'])
    (args.output / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'Selected: rollouts={best["rollouts"]}, horizon={best["horizon"]}, step={best["step_size"]}')
    print(f'Cost: {history[0]["start_cost"]:.5f} -> {history[-1]["start_cost"]:.5f}; saved to {args.output}')
    if args.plot:
        try:
            plot_learning(history, args.output)
            if board_plot:
                board_plot(env, args.output)
        except ImportError as error:
            print(f'Plotting unavailable: {error}. Numeric results were saved.')
