# Policy Evaluation: GridWorld

This folder compares **truncated on-policy Monte Carlo with dynamic exploration time (TOMC)**, **TD(λ)**, and **True Online TD(λ)** on the team's 10×10 GridWorld. Both TD methods are evaluated with tabular and compact linear representations.

**Start with `policy_eval.ipynb`.** It imports the method files and produces the comparisons. There is no need to run a separate Python script first.

For a single Blackjack plot comparing TOMC, tabular TD(0), and tabular TD(λ), follow [Blackjack_readme.md](Blackjack_readme.md).

## Requirements

- Python 3 with NumPy and Matplotlib.
- Jupyter or a notebook-capable editor, using the project's Python environment.
- Keep the notebook, method files, `evaluation_utils.py`, `baseline.py`, and `utils.py` together in this folder. `baseline_blackjack.py` is only needed for the optional Blackjack runs.

## Reproduce the figure

Open `policy_eval.ipynb` in your editor, or from the `Project` folder:

```bash
jupyter notebook policy_evaluation_enhancements/policy_eval.ipynb
```

In **Choose what to run**, set:

```python
ENVIRONMENT = 'gridworld'
METHOD = 'baseline'
TASK = 'evaluation'
TUNE = False
```

Select **Restart Kernel and Run All**. This runs only the selected method. Repeat with each method name below, then run the final plotting cell to compare the saved results.

| `METHOD` | Evaluator |
|---|---|
| `baseline` | TOMC with dynamic exploration time |
| `td_tabular_0` | Basic tabular TD(0), λ fixed at zero |
| `td_tabular` | Ordinary TD(λ), tabular |
| `td_tabular_online` | True Online TD(λ), tabular |
| `td_linear` | Ordinary TD(λ), linear features |
| `td_linear_online` | True Online TD(λ), linear features |

`METHOD = 'all'` explicitly runs all six methods and takes longer. For Blackjack's three-method figure, use `ENVIRONMENT = 'blackjack'` and `METHOD = 'blackjack_comparison'`. The `_online` suffix means **True Online TD**; all methods learn from online samples.

Use `TASK = 'policy_improvement'` for the policy-gap figure, with the same Euclidean SPMD update for every evaluator. Use `TASK = 'evaluation'` for the fixed-policy error figure. Running one task does not automatically run the other.

To redraw a figure without learning again, run only the imports, configuration, and final plotting cell.

## What is run

- **Game:** 10×10 GridWorld with the existing goals, hazards, movement costs, and reset rules. Goals do not terminate the continuing process.
- **Fixed policy:** uniform action probabilities; error is measured against its exact action values.
- **TD target:** sampled next-action cost target, as in the lectures. True Online TD adds Dutch traces and a correction term.
- **Representations:** 400 tabular parameters or 32 compact linear parameters.
- **Seeds:** tuning uses 100–102; testing uses 200–204, matching the entropy experiment's seed procedure. Figures show the test-seed mean and ±1 standard error.
- **Policy improvement:** every method uses the same SPMD step of 0.5. It is a provided starting value, not a claim of successful GridWorld baseline tuning.
- **Scoring:** exact models are used only to measure error and policy gaps. The learner uses sampled moves. `estimated_value` in the policy-improvement CSV is the current policy's average learned action value at the start state; it is not an additional rollout estimate.

`N_STEPS` sets how many moves each run takes. `SEEDS = [200]` runs one repeat instead of five. A short TOMC run may finish before its first coverage round is complete.

## Optional parameter tuning

Set `TUNE = True` to select evaluator parameters using the mean error over tuning seeds 100, 101, and 102:

- TD learning rate: 0.01, 0.05, 0.1.
- TD trace parameter λ: 0.3, 0.6, 0.9.
- TD(0) keeps λ = 0 and tunes only the learning rate.
- TOMC horizon: 40, 70, 100.

All three tuning seeds must produce an estimate. Incomplete coverage cannot win by dropping unsuccessful seeds. If no TOMC candidate completes, the starting values are retained and identified as such. The common SPMD step stays fixed.

Tuning is off by default. The notebook reuses an existing `parameters.json` when present. The provided TD starting values come from the earlier three-seed comparison; GridWorld TOMC's starting values were fallback defaults.

## Output files

Results have fixed locations; repeating a task replaces its CSV in the same method folder.

```text
results/
  baseline/gridworld/
  td_tabular_0/gridworld/
  td_tabular/gridworld/
  td_tabular_online/gridworld/
  td_linear/gridworld/
  td_linear_online/gridworld/
  figures/gridworld/
```

Each method folder contains `evaluation.csv` or `policy_improvement.csv`, the task's settings, and `parameters.json`. Optional tuning also writes `tuning.csv` and `tuning_means.csv`.

Figures are `results/figures/gridworld/policy_evaluation_error.png` and `policy_evaluation_policy_gap.png`, with PDF copies. The baseline is orange; TD curves are blue. Only methods already run for the selected task and seeds are shown.

Blackjack is optional: setting `ENVIRONMENT = 'blackjack'` writes under `results/<method>/blackjack/`. It does not run as part of the GridWorld submission.

## Reading TOMC results

TOMC waits until the required state/action pairs have been visited and their truncated returns are available. Before its first estimate, the error column is blank, `status` explains the wait, and `pending_pairs` counts missing pairs. A blank does not mean zero error. The curve starts only when all selected seeds have an estimate.

Older CSVs used `NaN` for this same waiting period. GridWorld can take longer than Blackjack because rare pairs can take many moves to reach.

## Files used

| File | Role |
|---|---|
| `policy_eval.ipynb` | **Entry point:** choose, run, and plot |
| `tomc.py` | Dynamic exploration and truncated first-visit returns |
| `td_tabular.py` | Ordinary tabular TD(λ) update |
| `td_tabular_online.py` | True Online tabular TD(λ) update |
| `td_linear.py` | Compact features and ordinary linear TD(λ) update |
| `td_linear_online.py` | True Online linear TD(λ) update |
| `evaluation_utils.py` | Shared sampling, scoring, tuning, and plotting |
| `baseline.py`, `baseline_blackjack.py` | Existing game setups |
| `utils.py` | Existing shared SPMD update and baseline helpers |

`policy_evaluation.py` and `test_policy_evaluation.py` belong to the older implementation and are not used by this notebook. See `CLEANUP.md` for optional cleanup.
