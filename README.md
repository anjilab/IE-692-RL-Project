# IE 692 Course Project: Optimization Methods for Reinforcement Learning

Team: Jack Morris, Anjila Budathoki, Manish Dhakal.

A reinforcement-learning library written from scratch. It contains the project baseline (Truncated
On-Policy Monte Carlo with dynamic exploration time, plus Stochastic Policy Mirror Descent) and the
three enhancements from our proposal.

## Installation

Requires Python 3.12 or newer.

```bash
uv sync --all-extras                     # with uv
pip install -e ".[dev,notebooks]"        # or with pip, inside a virtual environment
```

The dependencies are `numpy`, `numba` (which compiles the sampling loops: TOMC, TD and the
certifier's rollouts) and `matplotlib`.

## Running

```bash
python tune.py                           # every experiment: grid search, full runs, CSVs, figures -> results/tuned/
python tune.py --problems Blackjack --methods baseline bregman_tsallis
python tune.py --plot-only               # redraw the figures from saved CSVs
pytest                                   # tests (ruff check . for lint)
python benchmarks/bench.py --run         # runtime and memory report -> results/benchmark.md
```

`config.cfg` is the only config, and `python tune.py` runs every experiment in it together: problems,
budgets, seeds, each method's parameter grid, and the display names used in figures. Every method is
SPMD, a Q estimator plus a Bregman-divergence step:

| Grid (`[figures]` group) | Methods | Estimator | Step |
|---|---|---|---|
| `bregman`: enhancement 1 | `baseline`, `bregman_euclidean`, `bregman_tsallis` | TOMC | KL, Euclidean, Tsallis |
| `evaluation`: enhancement 2 | `true_online_td`, plus `td_zero_tabular`, `td_lambda_tabular` and the `_linear` versions | TD | KL |
| `entropy`: enhancement 3 | `entropy_reg`, `entropy_reg_annealed` | TOMC | KL with τ > 0 |
| `combined` | `true_online_td` (KL), `td_euclidean`, `td_tsallis`, `combined_entropy`, `combined_annealed` | True Online TD(λ) | KL, Euclidean, Tsallis; KL with τ > 0 |

Each of the four grids is tuned separately on each of the three problems (GridWorld, Blackjack and
the M5 Inventory problem): 4 grids x 3 problems. The combined grid is tuned second: its methods
(`estimator_from = true_online_td`) use the TD step size and λ that the evaluation grid chose for
True Online TD(λ) on the same problem, and tune their own step size and Tsallis index or entropy
weight. Entropy regularization has a closed-form step only for KL, so the combined grid pairs it
with KL.

Every TD estimator restarts from Q = 0 at each SPMD iteration (`estimator.q_init`), so an action it
never samples keeps Q = 0, which looks cheaper than any sampled action. KL and Tsallis with index
below 1 raise such an action's probability gradually; Euclidean can move a whole state's probability
onto it in one step, which is why `td_euclidean` does poorly on GridWorld.

Tuning follows the handout: at most three parameters with three values each, run on 10% of the
budget. The best combination is the one with the smallest certified lower bound on the optimality gap
(model-free, `core/optimality.py`); it then runs on the full budget with 20 test seeds. `[figures]` in
`config.cfg` splits the methods into figure groups (at most eight methods each, baseline first). Per
problem and group, `tuning_grid_<problem>_<group>.png` shows every combination's lower bound as a
heatmap (chosen cell outlined) and `learning_curves_<problem>_<group>.png` the full runs.

## Problem setup

All problems minimize discounted **cost** with γ = 0.9, and learners only see sampled
`(next_state, cost, done)` transitions (the online model). The exact model is used only to report
the gap to the optimum.

| Problem | States × actions | Rules |
|---|---|---|
| **GridWorld** (`envs/gridworld.py`) | 100 × 4 | The handout's 10×10 grid: 10 hazards, 3 goals, start in the top-left corner. The intended move happens with probability 0.8, otherwise a random other direction; walls clip movement. Cost 0 at a goal, 1 at a hazard, 0.2 elsewhere. From a goal or hazard the agent returns to the start with probability 0.8. Never terminates; values are reported at the start. |
| **Blackjack** (`envs/blackjack.py`) | 32 × 2 | Cards 2–5 drawn with replacement; actions are stop or draw. The dealer draws to at least 9; above 11 busts. State: (player total 4–11, dealer card 2–5). Cost −1 win, 0 tie, +1 loss; each hand is a fresh deal. Values are reported at (8, 3). |
| **Inventory** (`envs/inventory.py`, bonus) | 90 × 13 | Daily reordering of one Walmart product with demand from the M5 dataset (data included). Cost = ordering + holding − revenue. |

Each iteration records the estimated V^{π_t}(s₀) and the observations used, as the handout asks, plus
the exact gap to the optimum.

## Code layout and enhancements

```
tune.py, config.cfg       tuning and full runs
src/ie692_rl/
  envs/                   GridWorld, Blackjack, RetailInventory
  core/                   baseline and shared code: TOMC (tomc.py), the SPMD loop (spmd.py),
                          divergence and estimator interfaces, exact scoring, optimality certificate
  enhancements/
    bregman_spmd/         1. SPMD with three Bregman divergences
    true_online_td/       2. True Online TD(λ) policy evaluation
    entropy_reg_spmd/     3. Entropy-regularized SPMD
notebooks/                baseline runner, entropy-regularized SPMD study, TD comparison
tests/                    pytest suite
```

`run_spmd(env, estimator, divergence, step_size, budget, seed, tau)` takes its policy evaluator and
its Bregman divergence as objects. The baseline is `TOMC` + `KL`, and each enhancement swaps one part:

1. **Bregman divergences** (`enhancements/bregman_spmd/`): `KL` (the baseline step), `Euclidean`
   (exact projection onto the simplex) and `Tsallis(index)` (per-state Newton solve for the
   normalizing multiplier). Config methods: `baseline`, `bregman_euclidean`, `bregman_tsallis`.
2. **True Online TD(λ)** (`enhancements/true_online_td/`): `TrueOnlineTD(eta, lam, n_steps)`
   replaces TOMC as the Q estimator (van Seijen & Sutton, 2014), with tabular or aggregated features.
   Config method: `true_online_td`.
3. **Entropy-regularized SPMD** (`enhancements/entropy_reg_spmd/`): `tau > 0` adds τ·Σ π log π to the
   sampled costs and uses the closed-form KL step π ∝ π^{1/(1+ητ)} exp(−ηQ/(1+ητ)); `AnnealedTau`
   gives τ_k = τ₀/(k+1). Config methods: `entropy_reg`, `entropy_reg_annealed`.
