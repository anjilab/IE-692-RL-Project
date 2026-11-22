# IE 692 project: TODO

Deadlines (EST): code and slides **Nov 24, 8:00am**; presentation Nov 24, 12:55pm; report **Nov 24, 11:59pm**.
Requirements are from `course_project_v3.pdf`.

## Entropy-Regularized SPMD (`entropy_reg_spmd/`)

- [] **Use the team's TOMC estimator.** Switch `optimization.ipynb` from `online_q` to
      `utils.estimate_Q_online_mc_dynamic` (TOMC with dynamic exploration time), so baseline and
      enhancement differ only in the update rule.
- [ ] **Fix `entropy_reg_spmd`:** compute in log space with the row maximum subtracted (it returns NaN at
      large step sizes), and correct the docstring ("rewards" -> costs).
- [ ] **Tune as the guideline asks:** grid search over up to 3 parameters x 3 values (τ, step size, rollouts),
      each combination run on 5-10% of the full budget, then run the best combination on the full budget.
- [ ] **Record the required metrics:** estimated V^πt(s₀) per iteration and observations per iteration
      (as `utils.train` does). Keep the exact gap to optimal as an extra.
- [ ] **Annealed τ:** add τ_k = τ₀ / (k+1)^p to the notebook, as promised in the proposal.
- [ ] **Proposal diagnostics:** track the smallest π(a|s) and GridWorld state coverage per iteration.
- [ ] **README:** installation (uv) and steps to reproduce every table and plot.
- [ ] Rerun the notebook and keep the final plots and tables for the report.

## Other enhancements

- [ ] **SPMD with three Bregman divergences** (Euclidean, KL, Tsallis): no Tsallis code is in the repo yet;
      confirm who owns it and push it.
- [ ] **True Online TD(λ)** (`policy_evaluation_enhancements/`): check it uses the same metrics and the same
      tuning protocol as above.

## Combine all enhancements

- [ ] **Final combined method:** True Online TD(λ) for policy evaluation + the best Bregman divergence + entropy
      regularization (fixed or annealed τ) for policy optimization.
- [ ] Compare the combination against the baseline (TOMC + Euclidean SPMD) and against each enhancement
      alone, on GridWorld and blackjack, with the same budget, tuning protocol and metrics.
- [ ] One summary plot and table across all methods.

## Report and presentation

- [ ] **Report** (PDF): title, authors, executive summary, introduction with hypotheses, methodology,
      results and analysis (value and observation plots, results vs hypotheses, reasons for (un)expected
      behaviour), conclusion, team contributions, references and AI-use citation.
- [ ] Present the M5 inventory problem as the real-world-data bonus.
- [ ] **Slides** (PDF, at most 10 minutes): methods, results, discussion, contributions.
- [ ] Code check: readable, runs from a fresh clone with the README instructions.
