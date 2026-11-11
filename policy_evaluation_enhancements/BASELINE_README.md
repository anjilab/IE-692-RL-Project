# Baselines: dynamic on-policy Monte Carlo + SPMD

`baseline.py` supplies GridWorld; `baseline_blackjack.py` supplies blackjack.
`utils.py` contains their shared evaluator, unchanged Euclidean SPMD update,
tuning, observation accounting, and optional figures. `policy_eval.ipynb`
compares standard and True Online TD against this dynamic MC baseline.

The implementation follows the **Online setting** section of
`assignments/Assign2_soln.ipynb`. It uses the assignment's simple sampling loops,
`first_time_sa`, coverage mask, and truncated first-visit returns. Policies here
use `[state, action]`, rather than the assignment's `[action, state]` layout.

## Dynamic exploration time

For each fixed policy:

1. Start GridWorld at its origin, or blackjack with a fresh initial deal. Carry
   the last environment state between evaluation calls. Every action follows
   the policy; there are no forced state/action starts during training.
2. Track first visits to every pair with action probability at least
   `(1 - gamma) / n_actions`.
3. Once those pairs have all been visited, collect `T - 1` more moves. This
   provides a full `T`-cost return beginning at the last first visit. The
   assignment collects one additional unused move; this version omits it.
4. Estimate each required pair from its first visit. Use `gamma ** k` for the
   k-th cost after that visit, correcting the assignment's absolute-time
   discounting. Stop each return at a real terminal hand, even if new blackjack
   hands were observed later in the same coverage period.
5. Assign excluded low-probability actions the assignment's pessimistic cost
   bound `1 / (1 - gamma)`. Both environments have maximum per-move cost 1.
6. Optionally average several complete coverage rounds before the unchanged
   SPMD update.

**T is fixed within an experiment. Exploration time varies with coverage.**
The remaining sample budget is a hard cap. Missing coverage or an incomplete
return tail produces no Q estimate and no policy update. All consumed moves
are still recorded. Reaching the cap is not reported as convergence.

Coverage can fail when some states are unreachable or rarely visited under the
current policy. This is especially relevant once Euclidean SPMD assigns zero
probability to actions. The code does not silently add exploration, change the
board, or substitute generative estimates. If no tuning candidate completes an
update, it saves tuning diagnostics and explicitly reports that no tuned policy
was selected.

## Run

Requires Python with NumPy. No dependencies were installed or changed.
Matplotlib is optional and has a known compatibility issue in the previously
used environment.

```bash
python3 baseline.py --output gridworld_dynamic_run_2
python3 baseline_blackjack.py --output blackjack_dynamic_run_2
python3 -m unittest test_baselines.py
```

Add `--plot` for figures in a compatible environment. Choose a new output
directory each time. Default directories have `_dynamic_baseline_results` in
their names and do not overwrite older generative-baseline results.

The default grid search still has 27 combinations:

| Setting | GridWorld | Blackjack |
|---|---|---|
| Complete coverage rounds per evaluation | 1, 2, 4 | 1, 2, 4 |
| Return length T | 40, 70, 100 | 2, 3, 4 |
| SPMD update size | 0.1, 0.5, 1.0 | 0.1, 0.5, 1.0 |

Use `--coverage-rounds`, `--horizons`, and `--step-sizes` to change candidates.
`--rollouts` remains an alias for coverage rounds; its meaning is no longer
forced-start trajectories per state/action. `--budget` caps full-run training
moves; `--tune-fraction` defaults to 0.1 per candidate. Online coverage can be
substantially slower than the previous vectorized generative baseline.

Scoring uses separate random generators and a fixed horizon (100 for GridWorld,
8 for blackjack). Scoring observations are additional and recorded separately.
The selected settings are run afresh with a new seed. Only candidates with at
least one completed update are eligible for selection; later coverage failures
remain visible in their status fields.

## Outputs and interpretation

- `tuning.csv`: candidate settings, costs, completed update counts, coverage
  status, required/visited pairs, and observation counts.
- `history.csv`: completed updates and incomplete attempts, including exploration
  and return-tail counts. A failure row retains the same update number.
- `policy.csv`: final action probabilities, when a candidate was selected.
- `config.json`: evaluator identity, settings, seeds, budget, and final status.
- Optional `learning.png` and GridWorld `board.png`.

An incomplete final attempt does not alter the policy. Its cost is a fresh
independent evaluation and can differ through sampling noise. Earlier saved
results used a different estimator and must not be presented as dynamic-MC runs.

The generative `monte_carlo_q` helper is retained for independent reference
estimates and validation, not baseline training. In `policy_eval.ipynb`, TD
training uses online streams as well. The control comparison matches each TD
method's observation count to the dynamic MC attempt and stops all methods
without updating if MC coverage fails. Separate scoring references still use
generative access and report that cost explicitly. Missing MC errors are
reported as missing, not as zero.

## Game rules and sources

GridWorld uses a 10 × 10 board, 10 hazards, 3 goals, and one origin. Normal moves
follow the chosen direction with probability .8; remaining probability is split
among the other three directions. Walls clip movement. Costs before moving are
0 at goals, 1 at hazards, and .2 elsewhere. From goals/hazards the next move
returns to the origin with probability .8; otherwise normal movement applies.

Blackjack draws cards 2–5 uniformly with replacement. The dealer draws until
at least 9; totals above 11 bust. Chosen dealing conventions: two cards each,
one dealer card visible, no natural bonuses. Costs: win -1, tie 0, loss +1;
intermediate moves cost 0. New hands begin only after termination. Evaluation
uses player total 8 and dealer showing 3; it is not an overall win rate.

Both games use gamma .9. Sources: `course_project_v2.pdf`,
`assignments/RL_Assignment1_GridWorldProblem.pdf`, and
`assignments/Assign2_soln.ipynb`. Keep the prompts and generated output for the
required AI-use disclosure. Repeat experiments with multiple seeds before final
conclusions.
