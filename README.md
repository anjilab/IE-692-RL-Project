# IE 692 Course Project: Optimization Methods for Reinforcement Learning

Team: Jack Morris, Anjila Budathoki, Manish Dhakal.

A small reinforcement-learning library written from scratch. It contains the project baseline
(Truncated On-Policy Monte Carlo with dynamic exploration time for policy evaluation, plus Stochastic
Policy Mirror Descent for policy optimization) and the three enhancements from our proposal. Everything
runs on the handout's two MDPs, plus a real-data inventory MDP for the bonus.

## Installation

You need Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras          # creates .venv with the package (editable), dev tools and notebook tools
```

With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"   # or `pip install .` for the library only
```

Runtime dependencies are `numpy`, `numba` (which compiles the sampling loops) and `matplotlib`.
The first call to each compiled function takes about a second. Simple kernels are cached on disk;
kernels that take an environment as an argument are recompiled once in each process.

## Quick start

```bash
python tune.py                          # tune every method, run it in full, write results/tuned/ (about 50 s)
python tune.py --problems Blackjack --tune-only
python tune.py --plot-only              # redraw the figures from saved CSVs
python benchmarks/bench.py --run        # performance and memory report -> results/benchmark.md
pytest                                  # test suite (about 40 s)
```

`tune.py` writes the following to `results/tuned/`:

- `tuning.csv` and `tuning.png`: every grid-search combination
- `history.csv`: every iteration of every full run, with estimated V(s₀), observations, exact gap,
  smallest action probability and states visited
- `summary.csv`: final gap per method with a z-test against the baseline
- `learning_curves.png`: exact gap and estimated V(s₀) against observations

All settings are in `config.cfg`.

## Problem setup

Every problem minimizes discounted **cost** with γ = 0.9 and runs in the **online** model: learners
only see sampled `(next_state, cost, done)` transitions. The exact model is used only to report the
gap to the optimum and in tests.

| Problem | States × actions | Rules |
|---|---|---|
| **GridWorld** (`envs/gridworld.py`) | 100 × 4 (10×10) | The handout's 10×10 grid has 10 hazards, 3 goals and a start in the top-left corner (see `figures/gridworld_problem.png`). The intended move happens with probability 0.8; otherwise one of the other three directions is taken uniformly, and walls clip movement. The cost is charged at the current cell: 0 at a goal, 1 at a hazard, 0.2 elsewhere. From a goal or hazard the agent returns to the start with probability 0.8. The task never ends. Values are reported at the start cell. |
| GridWorld, other sizes | `GridWorld(size=N)` | Goals sit in the three non-start corners. Hazards keep the 10% density, so a 30×30 grid has 90, placed with a fixed `layout_seed` and never on the start or a goal. |
| **Blackjack** (`envs/blackjack.py`) | 32 × 2 | Cards 2–5 are drawn uniformly with replacement, and the actions are stop or draw. The dealer draws until reaching at least 9, and a total above 11 busts. The state is (player total 4–11, dealer card 2–5). Costs are −1 for a win, 0 for a tie and +1 for a loss. A new hand is dealt after each finished one. Values are reported at (8, 3). |
| **Inventory** (`envs/inventory.py`, bonus) | 90 × 13 | Daily reordering of one Walmart product (M5 dataset, FOODS_3_377 at WI_1). Demand comes from the weekday's empirical sales. The cost is ordering + holding − revenue. The data ships with the package. |

Metrics, as the handout asks: the estimated value V^{π_t}(s₀) and the number of observations at
every iteration. We also record the exact gap to the optimal cost, the smallest π(a|s), and the
number of distinct states visited.

## Methods and where they live

```
tune.py                      main tuning / training script (grid search -> full runs -> CSV + figures)
config.cfg                   problems, budgets, seeds and each method's parameter grid
src/ie692_rl/
  envs/                      GridWorld, Blackjack, RetailInventory (+ data/), Environment base class
  core/                      shared by every method
    tomc.py                  baseline TOMC estimator
    spmd.py                  SPMD loop: run_spmd(env, estimator, divergence, step_size, budget, seed, tau)
    divergence.py            BregmanDivergence abstract base class
    estimator.py             QEstimator abstract base class
    exact.py                 exact scoring on the sparse model (reporting and tests only)
    optimality.py            model-free optimality certificate (advantage gap)
  enhancements/
    bregman_spmd/            enhancement 1: KL (the baseline step), Euclidean and Tsallis divergences
    true_online_td/          enhancement 2: True Online TD(λ) (+ TD(0), TD(λ)) as a Q estimator
    entropy_reg_spmd/        enhancement 3: entropy-regularized SPMD (annealed τ, soft optimum)
  experiment.py, plotting.py config parsing, tuning protocol, statistics, figures used by tune.py
notebooks/                   baseline runner, entropy-regularized SPMD study, TD evaluation comparison
tests/                       pytest suite;  benchmarks/bench.py: performance report
```

**Baseline.** `run_spmd(env, TOMC(bias), KL(), step_size, ...)`. Each iteration runs one TOMC
estimate. The on-policy stream continues until every pair with π(a|s) ≥ (1−γ)/|A| has been visited
*and* the discounted tail after the last first visit is below `bias·(1−γ)`. Q(s, a) is the return
after the first visit; pairs that are not required get the pessimistic `cost_bound/(1−γ)`. Then the
KL step is applied: π ← π·exp(−ηQ), normalized.

**Dependency injection.** `run_spmd` receives its policy evaluator and its divergence as objects:

- `QEstimator.estimate(env, policy, rng, max_observations, extra_cost)` returns `(Q, observations, states_visited)`.
- `BregmanDivergence.step(policy, q, eta, tau)` returns the new policy, and `divergence(p, center)` returns D(p, center) per state.

The SPMD subproblem is `argmin_p η(⟨Q, p⟩ + τ h(p)) + D(p, π_k)`, where h(p) = Σ p log p.
`config.cfg` picks the classes by name (`estimator = tomc | true_online_td`,
`divergence = kl | euclidean | tsallis`) and passes `estimator.*` / `divergence.*` keys to their
constructors.

### Enhancement 1: SPMD with three Bregman divergences (`enhancements/bregman_spmd/`)

The three divergences are in `enhancements/bregman_spmd/`; KL is also the baseline's step. Each
subclasses `BregmanDivergence` and is selected with `divergence = kl | euclidean | tsallis`.

| Divergence | w(p) | Step | Notes |
|---|---|---|---|
| `KL()` | Σ p log p | closed form π ∝ π^{1/(1+ητ)} exp(−ηQ/(1+ητ)) | the only one with τ > 0 (enhancement 3) |
| `Euclidean()` | ½‖p‖² | exact projection of π − ηQ onto the simplex (sort-based, Duchi et al. 2008) | actions can reach exactly 0 and come back |
| `Tsallis(index)` | (Σ p^q − 1)/(q(q−1)) | per-state root-finding for the KKT multiplier (below) | q → 1 is KL, q = 2 is Euclidean; q < 1 is a barrier |

**Tsallis solver.** The step is x_a(λ) = ((q−1)(c_a − λ))^{1/(q−1)} with c_a = π_a^{q−1}/(q−1) − ηQ_a,
where λ solves Σ x_a(λ) = 1. Our starting code (pg-termination's `tsallis_policy_update`) searches
for a bracket exponentially, then bisects to 1e-6 in Python. The new solver:

- **Exact bracket, no search:** f(lo) ≥ 0 ≥ f(hi), with lo = max c − 1/(q−1) and hi = max c − n^{1−q}/(q−1).
- **Safeguarded Newton:** f′(λ) = −Σ x^{2−q} is known in closed form, so steps converge
  quadratically; a step that leaves the bracket falls back to bisection.
- **Warm start** from the first-order solution λ₀ = −Σ π^{2−q}ηQ / Σ π^{2−q}.
- **Compiled with numba,** with no clamping.

It reaches machine precision in 3–10 iterations per state and runs 30–130× faster than the reference
(0.6 ms instead of 65 ms for a 30×30 grid). `index_from_gamma(γ)` gives pg-termination's index,
1/(1 − log(1−γ)/log 2) ≈ 0.231 for γ = 0.9.

**Two bugs in the starting code** are documented and fixed in the port used as a test oracle
(`tests/reference.py`):

- When the initial sum is within tolerance, `tsallis_update_lambda` returns the policy row instead
  of the multiplier.
- For a negative multiplier, the bisection swaps `lo` and `hi`, so it converges to the far end of the
  bracket. That point is outside the domain, where the 1e-6 clamp creates huge entries; for example
  [0.39, 0.59, 0.01, 0.001] became [0.5, 0.5, 0, 0], worse on the objective than not moving.

With both fixed, it agrees with the new solver to about 1e-4 (its tolerance), and the new solver is
never worse on the objective.

**Tests.** The contract tests in `tests/test_divergence.py` run on every divergence: D(p, p) = 0,
D ≥ 0, steps are probability vectors, `step` minimizes the subproblem, small steps barely move,
expected cost never rises, huge or infinite steps are greedy, and extreme inputs stay finite.
Specific checks cover Tsallis(2) = Euclidean, Tsallis(1 ± 1e-6) ≈ KL, and agreement with the
reference.

### Enhancement 2: True Online TD(λ) (`enhancements/true_online_td/`)

`TrueOnlineTD(eta, lam, n_steps, features)` is a `QEstimator`. Each SPMD iteration uses `n_steps`
fresh on-policy transitions with a Dutch trace and the true-online correction (van Seijen & Sutton,
2014). Features are one-hot (`tabular`, or `linear` = 2×2-block or total-band state aggregation).
`td_zero`, `td_lambda` and `true_online_td_lambda` run a fixed-policy comparison
(`notebooks/td_evaluation.ipynb`). As in the original code, the target samples the next action
(SARSA style); the proposal's formula uses the expectation over π instead.

### Enhancement 3: Entropy-regularized SPMD (`enhancements/entropy_reg_spmd/`)

Use `run_spmd(..., tau=τ)` with a fixed τ, or `tau=AnnealedTau(τ₀, p)` for τ_k = τ₀/(k+1)^p. Each
sampled cost at state s then includes τ Σ_a π(a|s) log π(a|s), so the estimator returns the
regularized Q. `KL().step(..., tau)` applies the closed form
π ∝ π^{1/(1+ητ)} exp(−ηQ/(1+ητ)). `soft_optimal_policy` gives the regularized optimum the method
converges to. Its estimated V(s₀) includes the (negative) entropy term, so it sits below the other
methods' estimates in `learning_curves.png`; the exact gap is computed without regularization.

## Model-free optimality certificate (`core/optimality.py`)

The exact gap needs the model P. The **advantage gap** g(s) = V^π(s) − min_a Q^π(s, a) needs only
Q^π. Since Q^π ≥ Q* and by the performance-difference lemma,

  g(s₀) ≤ V^π(s₀) − V*(s₀) ≤ max_s g(s) / (1 − γ),

and g = 0 everywhere exactly when π is optimal. The upper bound is loose (the 1/(1−γ) factor and
the max over states).

**A tighter lower bound** follows from the same gap. Let π′ be greedy for Q^π. By the
performance-difference lemma, V^π(s₀) − V^{π′}(s₀) = E_{d^{π′}}[g]/(1−γ), which is at least g(s₀).
And since V* ≤ V^{π′} for any π′, it is at most the true gap, even when π′ comes from a noisy Q̂.

| Random policy | true gap | g(s₀) | V^π − V^greedy |
|---|---|---|---|
| GridWorld 10×10 | 0.36 | 0.02 | 0.31 |
| Inventory | 6.1 | 0.49 | 5.7 |
| Blackjack | 0.25 | 0.06 | 0.06 |

The certifier's lower bound is the larger of the two estimated bounds.

- `advantage_gap(q, policy)` computes g, and `gap_bounds(q, policy, gamma, start, q_se, z)` returns
  both bounds from exact or estimated Q, with z-standard-error margins on the action differences.
- `RolloutCertifier(n_rollouts, n_start_rollouts, z).certify(env, policy, rng)` estimates Q^π for
  every (s, a): take a, then follow π, until γ^t < 1e-16. It uses the simulator only (no P), started
  in arbitrary states, with many more rollouts at the start state, where the lower bound is read.
  It then rolls out π and the greedy π′ from the start state for the improvement bound. Each z-level
  margin is applied separately, and `lower = max(gap_lower, improvement_lower)`. A certificate takes
  about 0.2 s on GridWorld and 0.8 s on 30×30.
- `run_spmd(..., certifier=RolloutCertifier(), certify_every=k)` adds `gap_lower`, `gap_upper` and
  `validation_observations` to the history. These run on their own random stream and are not
  charged to the training budget, so training is unchanged.

**Do not use TOMC's training Q̂ for this.** Actions with π < π_min are filled with 1/(1−γ), so a
deterministic, locked-in policy looks optimal. In Blackjack, "draw below 7" has a true gap of 0.194,
TOMC's Q̂ gives bounds [0, 0], and the rollout certifier reports a lower bound of 0.10. For the
optimal thresholds it reports 0.

## Experiments and tuning protocol

The protocol follows the handout. Each method grid-searches at most three parameters, three values
each; `config.cfg` is validated so that more than three tuned parameters is an error. Every
combination runs on 10% of the full observation budget with tuning seeds 100–102.

**Selection is model-free by default** (`[tuning] selection = gap_lower`). Every run's final policy
is certified (`certifier_*` settings), and the combination with the smallest mean certified lower
bound on the optimality gap is chosen, with ties broken by the mean upper bound. Only combinations
where every seed made at least one update are eligible. `selection = exact_gap` (or
`tune.py --selection exact_gap`) uses the exact gap instead, which needs the model.

The chosen combination then runs on the full budget with 20 test seeds (200–219) and is compared with
the baseline by a two-sample z-test on the exact final gap (|z| ≥ 2). `summary.csv` also reports the
mean certified lower bound. For nearly converged policies the lower bound is loose: on GridWorld the
chosen combinations certify 0.001–0.007 against exact gaps of about 0.075. It still ranked the
divergences in the same order as the exact gap.

| Method (`config.cfg`) | Estimator | Divergence | Tuned |
|---|---|---|---|
| `baseline` | TOMC | KL | step size × bias |
| `entropy_reg` | TOMC | KL, τ > 0 | τ × step size × bias |
| `entropy_reg_annealed` | TOMC | KL, τ_k = τ₀/(k+1) | τ₀ × step size × bias |
| `true_online_td` | True Online TD | KL | step size × TD step × λ |
| `bregman_euclidean` | TOMC | Euclidean | step size × bias |
| `bregman_tsallis` | TOMC | Tsallis | step size × Tsallis index (0.2314, 0.5, 3.5) |

Budgets are 20M steps for GridWorld (one uniform-policy TOMC estimate takes 0.2–1.4M steps), 120k for
Blackjack and 3M for Inventory. `[method NAME @ PROBLEM]` sections override settings for one
problem, such as TD's steps per iteration.

## Testing, linting, type checking

```bash
pytest                    # unit, statistical and end-to-end tests (perf and slow tests deselected)
pytest -m "perf or slow"  # throughput, the 10-minute pipeline check, 30x30 SPMD
ruff check . && ruff format --check .
mypy                      # strict, over src/, tests/, benchmarks/ and tune.py
```

The suite covers every module:

- **Environments:** the handout layout, 30×30 layouts, hand-computed transitions, and simulator
  frequencies against the exact model.
- **TOMC and TD:** both are checked against straightforward re-implementations of the team's
  original algorithms (`tests/reference.py`), to 1e-10. TOMC is also tested for unbiasedness
  against the exact Q, and TD for convergence to it.
- **Divergences and exact scoring:** the divergence contract; dense linear algebra and policy
  iteration for exact scoring.
- **SPMD and the tuning pipeline:** convergence with exact Q, and the full tuning pipeline end to end.
- **Numerical edge cases:**
  - log 0, and KL with zero-probability actions
  - η → ∞ and η·Q overflow
  - Q ranges of ±1e308
  - underflowing tail bounds
  - standard errors from one seed
  - z-tests with zero variance
  - diverging TD step sizes, which stop the run with status `diverged`

Warnings are errors in the suite, so a silent `divide by zero` or `invalid value` fails a test.

## Performance and memory

Measured on an Apple M3 (8 cores) with 7 workers; see `results/benchmark.md`, regenerated by
`python benchmarks/bench.py --run`.

- **Full pipeline:** the default `python tune.py` (GridWorld 10×10 and Blackjack, six methods,
  648 tuning runs, 240 full runs, each final policy certified) takes **about 47 s** end to end. The limit is 10 minutes, and the
  previous pure-Python version needed about 2 hours for a smaller experiment.
- **Throughput:** TOMC samples about 30M steps/s on GridWorld and True Online TD about 10M steps/s,
  roughly 100× the pure-Python loops.
- **Memory:** peak memory is about 210 MB per process. The original TOMC stored the whole stream
  (≈105 B per step, so 10 GB for 10⁸ steps). Returns now come from a ring buffer of the last ~350
  costs (γ^350 < 1e-16), so memory is O(|S||A|) for any stream length. Exact models are sparse
  (0.3 MB instead of 26 MB for 30×30), TD features are index tables (29 kB instead of a 104 MB dense
  Φ), and TD history keeps V(s₀) rather than full Q snapshots.

**GridWorld 30×30.** True Online TD + SPMD runs the full protocol on 30×30 in about 3 minutes
(`python tune.py --problems GridWorld30 --methods true_online_td`). **TOMC-based methods cannot run on
30×30 in any reasonable budget.** Under the uniform policy, 10⁹ steps (31 s) still leave 461 of the
3600 required pairs unvisited, and coverage grows by only a few hundred pairs per tenfold increase in
steps. Extrapolating that trend (measured up to 10⁹ steps), one estimate would not finish within the
~1.8×10¹⁰ steps a core samples in 10 minutes. With 90 hazards that send the agent back to the start
80% of the time, a random walk rarely reaches the far side, and no sampling speed changes that. The handout lets us adjust the environment if this happens, for
example with exploring starts or a lower reset probability; the handout's 10×10 grid has no such
problem.

## Findings worth knowing

- **Blackjack's tuning budget is too small to separate methods.** 10% of 120k steps allows about 2
  updates, so the grid search picks the largest step size, which jumps to the greedy policy at
  once. Every divergence has the same greedy limit, so all TOMC-based methods end with identical
  Blackjack results. A larger Blackjack budget (or tuning fraction) is needed for a meaningful
  comparison there.

- **Entropy regularization vs. TOMC's fill value.** Once π(a|s) drops below π_min, TOMC fills
  Q(s, a) with the pessimistic 1/(1−γ) = 10. The regularized step then drives that action towards
  exp(−10/τ). τ keeps probabilities positive where plain SPMD underflows to exactly 0, but not far
  from 0.
- **Constant-step KL-PMD with exact Q** reaches a gap of about 1e-5 quickly, then improves at O(1/k):
  in near-tie states the advantage is about 1e-8, so probability moves very slowly there.
- **Results differ numerically from earlier runs.** Actions are now drawn with one uniform number
  per step inside compiled code, so random streams differ from the old `rng.choice` loops. The
  algorithms themselves are unchanged, and the tests check that against the original code.
