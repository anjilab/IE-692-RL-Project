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

The dependencies are `numpy`, `numba` (which compiles the sampling loops) and `matplotlib`.

## Running

```bash
python tune.py                           # grid search, full runs, CSVs and figures -> results/tuned/
python tune.py --problems Blackjack --methods baseline bregman_tsallis
python tune.py --plot-only               # redraw the figures from saved CSVs
pytest                                   # tests (ruff check . and mypy for lint and types)
python benchmarks/bench.py --run         # runtime and memory report -> results/benchmark.md
```

All settings are in `config.cfg`: problems, budgets, seeds, each method's parameter grid, and the
display names used in figures. Tuning follows the handout: at most three parameters with three values
each, run on 10% of the budget. The best combination is the one with the smallest certified lower
bound on the optimality gap (model-free, `core/optimality.py`); it then runs on the full budget with
20 test seeds.

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
   gives τ_k = τ₀/(k+1)^p (set `spmd.tau_power = p`). Config method: `entropy_reg`.
