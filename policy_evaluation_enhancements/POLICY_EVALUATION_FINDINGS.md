# Policy evaluation findings and completion check

The comparison implements the requested evaluators, follows the lecture notation and assignment-style NumPy loops, and uses the existing two baseline environments. It does **not** establish that True Online TD(λ) always wins.

## Measured results

All numbers below are means ±1 standard error over five test seeds. Lower is better. Fixed-policy errors use a uniform policy and stationary visitation weights. Policy gaps use the same Euclidean SPMD update and equal training-observation budgets.

### GridWorld

| Evaluator | Fixed-policy error | Final policy gap |
|---|---:|---:|
| TOMC, dynamic exploration | 0.17637 ± 0.02532 | 0.08761 ± 0.00799 |
| TD(λ), tabular | 0.05521 ± 0.00509 | 0.00504 ± 0.00107 |
| True Online TD(λ), tabular | 0.05472 ± 0.00439 | 0.00510 ± 0.00110 |
| TD(λ), linear | 0.11909 ± 0.00844 | 0.01377 ± 0.00223 |
| True Online TD(λ), linear | 0.13191 ± 0.00947 | 0.03880 ± 0.00234 |

### Blackjack

| Evaluator | Fixed-policy error | Final policy gap |
|---|---:|---:|
| TOMC, dynamic exploration | 0.08208 ± 0.00372 | 0.07750 ± 0.04746 |
| TD(λ), tabular | 0.10527 ± 0.00696 | 0.00044 ± 0.00044 |
| True Online TD(λ), tabular | 0.10527 ± 0.00696 | 0.00044 ± 0.00044 |
| TD(λ), linear | 0.18273 ± 0.00433 | 0.00000 ± 0.00000 |
| True Online TD(λ), linear | 0.18274 ± 0.00433 | 0.00000 ± 0.00000 |

## Observations

- **GridWorld evaluation:** tabular TD(λ) and True Online TD(λ) finish with almost identical weighted errors (0.05521 and 0.05472), below TOMC (0.17637). The observed difference between the two tabular TD methods is small relative to their standard-error bands.
- **GridWorld policy improvement:** tabular TD(λ) and True Online TD(λ) finish near a 0.0051 gap, compared with 0.08761 for TOMC. Ordinary linear TD finishes at 0.01377; true-online linear TD is worse here at 0.03880. The correction is not uniformly beneficial under these settings.
- **Blackjack evaluation:** TOMC finishes with the lowest weighted error (0.08208), compared with 0.10527 for either tabular TD method and about 0.18273 for the compact linear methods. Thus the enhancement does not beat TOMC on every evaluation problem.
- **Blackjack policy improvement:** both linear methods finish at zero gap at the chosen evaluation state `(8,3)`; both tabular TD methods average a 0.00044 gap. TOMC averages 0.07750, with two seeds ending at 0.19375. Accurate evaluation under a uniform policy and good control from one start state are different outcomes; zero start-state gap is not evidence of optimality at every state.
- **Overlapping Blackjack curves:** the tabular ordinary and true-online curves coincide. Player totals increase within a hand, so a tabular state/action pair is not revisited before the terminal trace reset; the true-online correction consequently vanishes for these trajectories. This agreement is consistent with the game structure.
- **Metric choice matters:** GridWorld's final uniform RMSE is 0.17078 for ordinary linear TD versus 0.17676 for ordinary tabular TD, reversing their ordering under visitation-weighted error. Keep the supplemental plot when discussing rarely visited states.

## Description for the report

Under each fixed policy, TOMC collects a single online stream until all required state/action pairs have been visited, then waits for their truncated return tails. It uses the first visit for each pair. During the fixed-policy experiment, completed TOMC replications are averaged. During policy improvement, each completed round supplies a fresh estimate to SPMD.

Ordinary TD(λ) and True Online TD(λ) update after each transition. Both use the sampled next-action prediction, `delta = cost + gamma * Q(s_next, a_next) - Q(s, a)`. Ordinary TD(λ) uses accumulating traces. True Online TD(λ) uses Dutch traces plus a correction for predictions changing during learning. Terminal hands have zero continuation value and reset the traces. Policy changes also reset traces while retaining learned parameters.

Tabular features allocate a parameter to every state/action pair. Compact linear features use normalized coordinate/polynomial terms and action-specific blocks; GridWorld also includes goal/hazard indicators. The compact representation has 32 parameters versus 400 tabular parameters in GridWorld and 12 versus 64 in Blackjack. These particular features are a design choice, not a prescribed course feature set.

**Hypothesis:** incremental TD updates may use a limited observation budget more effectively than waiting for complete TOMC coverage. Traces may propagate costs sooner. The true-online correction may improve finite-step-size behavior, but approximation error and tuning can change the outcome; superiority is not guaranteed.

## Lecture and assignment alignment

| Source | Implementation |
|---|---|
| Assignment 2 | First-visit truncated returns; dynamic required-pair coverage; low-probability actions use the assignment's pessimistic bound. |
| Lectures 18–19 | Flat state/action index, `Phi`, `theta_t`, sampled TD direction, and parameter updates. |
| Lectures 19 and 22 | Online transitions; the learner does not sample directly from a known stationary distribution. |
| Lecture 20 | Trace comparison and averages of completed Monte Carlo replications. |
| Lecture 21 notebook | NumPy functions, explicit loops, weighted action-value error, and ggplot-style figures. Incomplete placeholders/typos are not copied. |
| True Online TD(λ), van Seijen and Sutton (2014) | Dutch traces and true-online correction; an extension beyond the lecture implementation. |

## Completion and limitations

- **Implemented and tested:** dynamic TOMC baseline, both TD variants, tabular/compact-linear comparison, both games, fixed-policy error and SPMD policy-gap figures, five test seeds, uncertainty bands, tuning records, per-update sampled validation values, and separate training/validation observation counts.
- **Checked:** 13 automated tests, including the true-online update against its online forward view, terminal handling, model references, and independent method budgets. The notebook executes using the saved results.
- **Baseline tuning is incomplete for GridWorld:** no required-pair coverage completed in its 240,000-observation tuning allocation. The horizon of 100 and common SPMD step of 0.5 are transparent fallback defaults. Do not call this a fully tuned GridWorld baseline. Blackjack tuning status is in `selected_settings.csv`.
- **Budget adjustment:** the initial fixed-policy GridWorld budget of 600,000 left one seed without a TOMC estimate. A coverage diagnostic found all first estimates complete by 762,146 observations. All five evaluators were therefore rerun from the same seeds for 1,200,000 observations. This adjustment followed coverage inspection; it was not preregistered. Control runs retain 2,400,000 observations. Blackjack uses 40,000 fixed-policy and 120,000 control observations.
- **Tuning accounting:** each candidate uses 10% of the control budget, which is larger than 10% of the fixed-policy budget. The cached fixed-policy stream is reused across candidates. Tuning and validation are separate from the training axis.
- **Interpretation:** different evaluators complete different numbers of policy updates. This is a comparison per training observation, not per update. Validation overhead also differs; the CSV records it. Compact linear features can introduce approximation error, and differing selected parameters mean the curves do not isolate only the true-online correction.
- **Remaining report work:** replace the proposal's expected-next-action TD equation with the sampled target actually implemented; describe truncated first-visit TOMC rather than complete-episode constant-step-size Monte Carlo; integrate these figures and findings into the final report. A claim of fully tuned GridWorld TOMC needs a justified solution to the tuning coverage limitation.

## Figures

- `policy_eval_results/lecture_aligned_five_seed/policy_evaluation_error.png` (and PDF): fixed-policy evaluation error.
- `policy_eval_results/lecture_aligned_five_seed/policy_evaluation_policy_gap.png` (and PDF): policy gap, matching the teammate's two-panel comparison format.
- `policy_eval_results/lecture_aligned_five_seed/policy_evaluation_uniform_error.png` (and PDF): supplementary error over all pairs.

Suggested caption: “Comparison of dynamic-exploration TOMC, ordinary TD(λ), and True Online TD(λ), using tabular and compact linear representations. Lines show means across five test seeds; shading is ±1 standard error. The horizontal axis counts training transitions. All policy-improvement runs use the same Euclidean SPMD update. TOMC evaluation estimates are plotted only after every seed has completed initial coverage.”
