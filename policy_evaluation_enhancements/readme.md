# Run the policy-evaluation comparison

Run these commands from the `Project` folder. The existing `.venv` contains NumPy and Matplotlib; no new installation is needed.

Quick execution check (small budgets, not report-quality results):

```sh
MPLCONFIGDIR=/tmp/ie692-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m policy_evaluation_enhancements.policy_evaluation --quick
```

Full experiment, including tuning and five separate test seeds:

```sh
MPLCONFIGDIR=/tmp/ie692-matplotlib OPENBLAS_NUM_THREADS=1 .venv/bin/python -m policy_evaluation_enhancements.policy_evaluation
```

Each run creates a new timestamped directory under `policy_evaluation_enhancements/policy_eval_results`. An explicit `--output PATH` is also supported; PATH must not already exist. The completed report run is `policy_eval_results/lecture_aligned_five_seed`.

Regenerate its figures without rerunning learning:

```sh
MPLCONFIGDIR=/tmp/ie692-matplotlib .venv/bin/python -m policy_evaluation_enhancements.policy_evaluation --plot-only policy_evaluation_enhancements/policy_eval_results/lecture_aligned_five_seed
```

Run the checks:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest policy_evaluation_enhancements.test_baselines policy_evaluation_enhancements.test_policy_evaluation
```

## Notebook

Open `policy_eval.ipynb` in a Python notebook environment and run all cells. Its default is to load the completed five-seed run. Set `RUN_EXPERIMENTS = True` to run a fresh experiment; `QUICK_TEST = True` reduces the budgets. The notebook contains the same executable functions as `policy_evaluation.py`, organized into lecture-style cells with explanatory text. Terminal runs do not require Jupyter or IPython.

## What is compared

1. TOMC with dynamic exploration: collect an online trajectory until every required state/action pair has a first visit, then finish its truncated return. A visit is required when its policy probability is at least `(1-gamma)/number_of_actions`. This is coverage-based exploration time, not a fixed rollout count.
2. Ordinary TD(λ) with one parameter per state/action pair.
3. True Online TD(λ) with one parameter per state/action pair.
4. Ordinary TD(λ) with compact linear features.
5. True Online TD(λ) with the same compact linear features.

All control runs use the existing Euclidean SPMD update. Only the evaluator changes. The baseline environments retain their existing rules; GridWorld goals/hazards can reset the player and do not terminate the process. Blackjack hands terminate and start again.

The sampled next-action TD target follows Lectures 19/22. The proposal's sum over next actions is an expected target and must be revised if describing this implementation. The actual TOMC baseline uses first-visit truncated returns; the fixed-policy experiment averages completed replications. It is not the proposal's constant-step-size update from complete episodes.

## Budgets and tuning

| Environment | Fixed-policy training steps | Control training steps | TD steps per policy update |
|---|---:|---:|---:|
| GridWorld | 1,200,000 | 2,400,000 | 10,000 |
| Blackjack | 40,000 | 120,000 | 500 |

The fixed-policy GridWorld budget was increased from 600,000 after a coverage check found one seed needed 762,146 observations for its first TOMC estimate. All methods were rerun with the same 1,200,000-observation budget.

Each candidate gets a tuning budget of 10% of the full **control** budget. Tuning seed 17 is separate from test seeds 101, 202, 303, 404, and 505. Fixed-policy candidates reuse the same sampled stream; replay is not additional environment interaction.

TD tunes learning rate `{0.01, 0.05, 0.1}` and nonzero trace parameter `{0.3, 0.6, 0.9}`. TD(0) is recorded as a separate ablation, not silently discarded. TOMC considers three horizons; the shared SPMD step considers `{0.1, 0.5, 1}` using the baseline. All evaluators then use that same SPMD step. TD parameters carry across policy updates; traces reset at policy changes and terminal hands.

**GridWorld limitation:** initial coverage did not finish in the tuning budget. Its TOMC horizon and shared SPMD step therefore use explicitly recorded defaults, not successfully tuned values. Full-budget TOMC can improve the policy, but later coverage may again remain incomplete. No estimate or policy update is invented when coverage is incomplete.

## Reading the output

- `policy_evaluation_error.png` / `.pdf`: fixed-uniform-policy error, weighted by stationary state/action visitation, like the lecture diagnostic.
- `policy_evaluation_policy_gap.png` / `.pdf`: gap to optimal cost versus training observations, like the teammate's figure.
- `policy_evaluation_uniform_error.png` / `.pdf`: supplementary unweighted error across all state/action pairs.
- `summary.csv`: final means and standard errors; `selected_settings.csv` and `tuning.csv`: settings and tuning evidence.
- `fixed_policy.csv` and `policy_improvement.csv`: observations, errors, completed updates, coverage status, and separate validation counts.
- `config.json`, `references.csv`, and diagnostic `.npz` files: experiment configuration and reference calculations.

Both primary figures have GridWorld and Blackjack panels and five curves. Shading is ±1 standard error across five test seeds. Exact value calculations and stationary weights are for scoring only; learning uses online samples. In addition to exact diagnostic gaps, each control update records an independently sampled value estimate and its observation cost. The x-axis counts training observations; validation counts are separately available in the CSV.

The fixed-policy plot shows TOMC only after all seeds have completed their first estimate. Under control, a waiting TOMC run keeps its current policy; other methods continue independently to their own full budget. Compact features have fewer parameters and can introduce approximation error. See `POLICY_EVALUATION_FINDINGS.md` for measured outcomes and outstanding report work.
