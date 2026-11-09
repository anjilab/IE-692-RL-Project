# Baseline experiments

- `baseline.py`: 10 × 10 GridWorld and optional board figure.
- `baseline_blackjack.py`: simplified blackjack.
- `utils.py`: Monte Carlo estimates, Euclidean SPMD updates, tuning, evaluation, CSV output, and optional learning plots.
- `test_baselines.py`: rule checks, independent exact blackjack reference, budget and repeatability checks.

## Run

Use a Python environment with NumPy installed. No packages were installed or changed by this implementation. Optional figures require a compatible Matplotlib installation (the previously used environment has a NumPy/Matplotlib compatibility problem).

```bash
python3 baseline.py --output gridworld_run_2
python3 baseline_blackjack.py --output blackjack_run_2
python3 -m unittest test_baselines.py
```

Choose a new output directory for each experiment; existing directories are not overwritten. Add `--plot` to save figures when Matplotlib works. Use `--help` to see the settings. Earlier `--iterations` and `--horizon` arguments have been replaced by `--budget` and `--horizons`, because tuning now compares different rollout counts and lengths under a common training budget.

The default grid search compares 27 combinations:

| Setting | GridWorld | Blackjack |
|---|---|---|
| Sampled runs per state/action | 5, 10, 20 | 5, 10, 20 |
| Maximum moves per run | 40, 70, 100 | 2, 3, 4 |
| SPMD update size | 0.1, 0.5, 1.0 | 0.1, 0.5, 1.0 |

Each candidate receives a training-budget cap of 10% of the full-run cap. Evaluation samples are extra and recorded separately. Complete evaluation sweeps are kept intact; less than one worst-case sweep of budget may remain unused. Thus actual training counts are recorded rather than assumed equal. Tuning uses only final-policy scoring; the full run records scores at every update.

All candidates are scored using the same fixed maximum run length: 100 for GridWorld, 8 for blackjack. This prevents shorter training horizons from appearing better simply because their evaluation omits future costs. Blackjack always ends within this default evaluation length. The selected settings are rerun from scratch using a separate seed.

## Rules and assumptions

GridWorld follows the executable assignment code: cost 0 at goals, 1 at hazards, .2 elsewhere, charged before moving. Goals/hazards return to the start with probability .8 on the next move; otherwise normal movement applies. Intended movement occurs with probability .8; remaining probability is split among the other three directions. Edges keep movement on the board. Locations match `project_steup.ipynb`.

Blackjack uses equally likely cards 2–5 with replacement, dealer drawing until at least 9, and bust above 11. Two initial cards, one visible dealer card, ties, and costs are explicitly chosen conventions: win = -1, tie = 0, loss = +1. No aces or natural bonus. The dealer's hidden card can be sampled when the player stops because no observation has revealed it and draws are independent. Dealer play is resolved within the stop action. Terminal hands receive no further samples or costs.

Both use discount .9. Blackjack evaluates the fixed situation player total 8 / dealer showing 3; its reported score is not the overall win rate over random initial hands. `Blackjack.deal` is available for later evaluation over initial deals.

The learning code follows the supplied notebook's generative-sampling interface: simulations can start from every state/action, and subsequent choices follow the fixed policy. This access assumption should be checked against the instructor's interpretation of the online model. Training uses fresh sampled experience, not exact transition probabilities. GridWorld and some blackjack training runs are truncated; blackjack's short horizons deliberately test that approximation. Zero costs beyond the truncation are assumed.

## Outputs

- `tuning.csv`: all candidates, scores, uncertainty, updates, and actual sample counts.
- `history.csv`: full-run performance and sample counts per update, including initial-policy evaluation.
- `policy.csv`: probabilities of each action, with board coordinates or blackjack hand labels.
- `config.json`: settings, seeds, selected combination, budgets, and assumptions.
- Optional `learning.png`, plus `board.png` for GridWorld.

Lower scores are better. Standard errors describe scoring uncertainty, not variability between independent training runs. Repeat with different `--seed` values for final conclusions. No enhancement algorithms are included.

Sources: `course_project_v2.pdf`, page 3, and `assignments/RL_Assignment1_GridWorldProblem.pdf`. The final course submission must include the required AI-use disclosure with prompts and generated output; retain the conversation as supporting material.
