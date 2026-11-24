# AI usage

**Student:** Manish Dhakal, Group 5 (Budathoki, Dhakal, Morris), IE 692

**Covers:** my own use of AI on the entropy-regularized SPMD enhancement and the M5 (Walmart) inventory bonus. My teammates' AI use is not covered here.

**Tool:** Anthropic, *Claude Code* (VS Code extension), models Claude Opus 5 and Claude Opus 5.5, used September 15 to November 24, 2026. It could read and edit files, run code and use git in my working copy.

**Prompts and generated output:** [prompts_and_responses.txt](prompts_and_responses.txt) has 30 major exchanges, with my prompts and the AI's replies verbatim and the main code it wrote. They were selected from about 200 prompts in four sessions. This summary and that file were also prepared with the AI. Numbers in brackets below refer to that file.

## How I worked with the AI

I gave short requests one step at a time. The AI wrote and ran code and reported results. I then asked questions, changed direction or rejected parts of the work.

| Stage | What I asked for | What the AI produced |
|---|---|---|
| Proposal [1, 2, 4] | Enhancement options, a validity check of entropy regularization, a hypothesis | Options, critique, suggested hypotheses; edits to my part of the proposal |
| Theory [3, 5, 19, 26] | Derivations and explanations (PMD, Q and V, costs, optimal values, transition probabilities) | Explanations and the closed-form update derivation |
| First implementation [6-10] | Tests for my `run_pmd`, the entropy term, GridWorld and Blackjack experiments | Tests, the entropy-regularized update, experiment scripts, a validation of the update |
| Estimator and tuning [11-13] | Switch to the team's TOMC, tune on 10% of the budget, make it faster | Notebook changes, a tuning script, a faster estimator (both later replaced by the team's versions) |
| Sweeps and full runs [14-17] | τ × step size × rollouts sweep, full runs, value plots | `param_sweep.py`, `export_tuning.py`, `estimated_value.py`, figures |
| Equations [18] | LaTeX for the SPMD and entropy-regularized updates | Equations for slides and report |
| τ schedules [20, 21] | Polynomial, linear and cosine schedules | Code and comparison; I later had the new schedule code removed |
| Real-data bonus [22-25, 27] | Find a dataset, build the inventory problem, test on unseen sales | First version of the inventory environment, held-out backtest, figures |
| Notebooks and git [28-30] | Notebooks that reproduce the figures, merge conflicts | `entropy_reg_spmd.ipynb` (experiment part), `m5_walmart.ipynb`, conflict resolution, some commits |

I also used it for presentation planning, figure labels and README text.

## Decisions I made

- Chose entropy-regularized SPMD as my enhancement, and the M5 inventory problem for the bonus.
- Limited validation to online sampling, as the project requires, and dropped the generative setting [10].
- Used the team's TOMC estimator, so the baseline and my method differ only in the policy update [11].
- Set the TOMC bias to 0.01 after asking for a comparison that covered both problems.
- Asked whether the full-budget sweep followed the tuning guideline [15]. The full runs use settings chosen on 10% of the budget.
- Questioned results, for example the number of inventory states [27] and whether all enhancements had been run [25].
- Chose which methods and labels appear in the figures, and dropped the annealed τ method from the comparison.

## AI mistakes found and corrected

- An early GridWorld result favored τ = 0.001 (z = −2.3, 5 seeds). It did not hold with the team's TOMC estimator [10, 11].
- The AI called the joint search "clearly better" than the sequential one; a check on fresh seeds did not support that, and it withdrew the claim [15].
- Early inventory profits came from settings picked by hand; they were replaced by runs tuned on the training period only [24].
- It committed a script whose documented command failed; this was found later and fixed.
- Removing the τ schedules broke the old inventory backtest; `m5_walmart.ipynb` replaces it for the two figures shown [21, 29].

## What the results say

With settings tuned on 10% of the budget and 20 test seeds, entropy regularization had a lower mean gap than the baseline on GridWorld (0.072 vs 0.134) and Blackjack (0.058 vs 0.107). Neither difference is statistically significant (z = −1.4 and −1.6). These settings were chosen by the exact gap, which uses the model.

## AI-generated code in the repository (from my sessions)

- `benchmarks/param_sweep.py`, `benchmarks/export_tuning.py`, `benchmarks/estimated_value.py` (teammates edited some of these later)
- `notebooks/m5_walmart.ipynb` and the experiment part of `notebooks/entropy_reg_spmd.ipynb`
- The first version of the inventory environment; the version in `src/ie692_rl/envs/` was restructured by the team
- Small edits to `config.cfg`, `tune.py`, `README.md`, `tests/test_experiment.py` and `src/ie692_rl/plotting.py`

## Data source

M5 Forecasting competition (Walmart unit sales, 2011-2016), <https://www.kaggle.com/c/m5-forecasting-accuracy>, downloaded from Nixtla's public mirror.
