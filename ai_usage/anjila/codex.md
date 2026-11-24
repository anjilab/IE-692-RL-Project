# Review policy evaluation enhancement

> /Users/anjilabudathoki/Downloads/UTK Documents/Fall_2026_Course/IE692/Project/course_project_v3.pdf
>
> Based on details, focus on policy evaluation enhancement part.
> Mainly what is expected from the requirement and what has been already done.
> /Users/anjilabudathoki/Downloads/UTK Documents/Fall_2026_Course/IE692/Project/notebooks/td_evaluation.ipynb
> see this for results reporting.&#x20;

<details><summary>6 previous messages</summary>

> I’ll compare the policy evaluation enhancement requirements in the PDF with the notebook’s methods and reported results, then identify what is complete and what is still missing. I’ll keep this review read-only.
>
> <details><summary>Explored 2 files, 2 lists, ran 5 commands</summary>
>
> - Ran `pwd; rg --files -g 'AGENTS.md' -g '*course_project_v3*' -g '*td_evaluation*' -g '*README*' -g '*requirements*' -g 'pyproject.toml' -g '*td*' -g '*eval*'`
>
> <details><summary>Explored 2 files, 2 lists</summary>
>
> - Listed files
> - Listed files in `src`
> - Read `./pyproject.toml`
> - Read `./README.md`
>
> </details>
>
> - Ran `python - <<'PY'
> import importlib.util,shutil
> for m in ['pypdf','PyPDF2','fitz']:
>  print(m, bool(importlib.util.find_spec(m)))
> print('pdftotext',shutil.which('pdftotext'))
> import json
> p='notebooks/td_evaluation.ipynb'
> n=json.load(open(p))
> print('NOTEBOOK',p,'cells',len(n['cells']))
> for i,c in enumerate(n['cells']):
>  print('\n'+'='*24+' CELL '+str(i)+' '+c['cell_type']+' '+'='*24)
>  print(''.join(c.get('source',[])))
>  for o in c.get('outputs',[]):
>   if 'text' in o: print('OUTPUT:', ''.join(o['text']))
>   if 'text/plain' in o.get('data',{}): print('OUTPUT:', ''.join(o['data']['text/plain']))
>   if o.get('output_type')=='error':print('ERROR:',o)
> PY`
>
> - Ran `pdftotext -layout course_project_v3.pdf -`
>
> - Ran `python - <<'PY'
> import json
> n=json.load(open('notebooks/td_evaluation.ipynb'))
> for i in [3,4,5,6]:
>  c=n['cells'][i]; print('\nCELL',i);print(''.join(c['source']))
>  if i==5:
>   for o in c.get('outputs',[]):
>    if 'text' in o:print(''.join(o['text']))
>    if 'text/plain' in o.get('data',{}):print(''.join(o['data']['text/plain']))
> PY`
>
> - Ran `cat configs/td_spmd.cfg configs/td_spmd_linear.cfg; rg --files results/td_spmd results/td_spmd_linear; cat policy_evaluation_results/td_selected_parameters.csv; cat policy_evaluation_results/evaluation_summary.csv`
>
> </details>
>
> The notebook already covers both fixed-policy evaluation and TD estimators inside SPMD, including the two required reporting metrics. I’m checking two points before assessing completeness: how the implementation handles the handout’s “mixing length,” and whether the saved results support the comparisons.
>
> <details><summary>Explored 3 files, ran 2 commands</summary>
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./src/ie692_rl/enhancements/true_online_td/td.py`
> - Read `./src/ie692_rl/enhancements/true_online_td/features.py`
> - Read `./src/ie692_rl/core/spmd.py`
>
> </details>
>
> - Ran `cat results/td_spmd/summary.csv results/td_spmd_linear/summary.csv results/td_spmd/policy_value_summary.csv; rg -n 'mix|burn|n_steps|TrueOnline|TDLambda|TDZero|exact_gap|selection|estimated_value|observations|start' src/ie692_rl/experiment.py src/ie692_rl/enhancements/true_online_td/td.py tests/test_td.py tune.py; nl -ba tests/test_td.py | sed -n '1,310p'; head -n 5 results/td_spmd/history.csv; head -n 5 results/td_spmd_linear/history.csv`
>
> - Ran `python - <<'PY'
> import csv, collections, statistics, math
> for p in ['results/td_spmd/summary.csv','results/td_spmd_linear/summary.csv','results/td_spmd/policy_value_summary.csv']:
>  print('\nFILE',p)
>  rows=list(csv.DictReader(open(p)))
>  print('columns:',list(rows[0]) if rows else [])
>  for row in rows:print(row)
> for p in ['results/td_spmd/history.csv','results/td_spmd_linear/history.csv']:
>  runs=collections.defaultdict(list)
>  for r in csv.DictReader(open(p)): runs[(r['problem'],r['method'],r['seed'])].append(r)
>  groups=collections.defaultdict(list)
>  for (env,method,seed),rs in runs.items():
>   final=max(rs,key=lambda r:int(r['iteration']))
>   groups[(env,method)].append((float(final['exact_value']),float(final['gap']),int(final['cumulative_observations']),final['status'],sum(r['status']=='updated' for r in rs)))
>  print('\nAGGREGATED FROM',p)
>  for k,vs in sorted(groups.items()):
>   vals=[v[0] for v in vs]
>   print(k,'n',len(vals),'value',round(statistics.mean(vals),6),'SE',round(statistics.stdev(vals)/math.sqrt(len(vals)),6),'gap',round(statistics.mean(v[1] for v in vs),6),'obs range',min(v[2] for v in vs),max(v[2] for v in vs),'stops',dict(collections.Counter(v[3] for v in vs)),'mean updates',round(statistics.mean(v[4] for v in vs),1))
> PY`
>
> </details>
>
> Viewed 3 images
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python - <<'PY'
> import os,datetime,csv,collections
> for p in ['results/td_spmd/history.csv','results/td_spmd/summary.csv','results/td_spmd/policy_value_comparison.png','results/td_spmd_linear/history.csv','notebooks/td_evaluation.ipynb']:
>  s=os.stat(p);print(p,datetime.datetime.fromtimestamp(s.st_mtime).isoformat(),s.st_size)
> rs=collections.defaultdict(list)
> for r in csv.DictReader(open('results/td_spmd/history.csv')):
>  if r['problem']=='Blackjack' and r['method']=='td_lambda': rs[int(r['iteration'])].append(float(r['exact_value']))
> for t in [58,59,60]:print(t,sum(rs[t])/len(rs[t]))
> PY`
> - Ran `nl -ba src/ie692_rl/experiment.py | sed -n '330,520p'; nl -ba src/ie692_rl/enhancements/true_online_td/td.py | sed -n '340,460p'; nl -ba src/ie692_rl/core/spmd.py | sed -n '110,245p'; rg -n 'mixing|burn.in|thinn|decorrel|mixing length' src notebooks/td_evaluation.ipynb configs/td_spmd*; git status --short`
>
> </details>
</details>

The policy evaluation enhancement is **largely implemented and supported by saved experiments**. The main remaining work is explaining the findings clearly, addressing the handout’s reference to mixing length, and limiting claims where the experiments provide weaker evidence.

I reviewed the [project requirements](<./course_project_v3.pdf>), [td_evaluation.ipynb](<./notebooks/td_evaluation.ipynb>), the relevant implementation, and saved results. I did not modify files or rerun experiments.

The handout expects you to **investigate whether an alternative policy evaluator offers useful improvements over TOMC**, using a controlled numerical comparison. It does not require the enhancement to outperform TOMC everywhere; explainable negative results also satisfy the scientific investigation.

| Expected from the handout | What has already been done | Assessment |
|---|---|---|
| Retain TOMC with dynamic exploration time as the baseline | TOMC uses state–action coverage and a discounted-tail stopping criterion | Implemented |
| Investigate a policy evaluation enhancement | TD(0), TD(λ), and True Online TD(λ) are compared with TOMC | Implemented |
| Work with online sampled data on Blackjack and GridWorld, with γ = 0.9 | Evaluators learn from sampled transitions; exact models provide reference scores | Implemented |
| Investigate function approximation | Tabular features and shared linear features are compared | Implemented |
| Tune up to three important parameters, with three candidate values each and a small observation budget | Fixed-policy experiments tune learning rate and λ using 10% budgets; SPMD experiments also tune the policy step size | Implemented; mixing length needs discussion |
| Report estimated state value and observations at each iteration | SPMD histories retain both; the notebook produces the required iteration plots | Implemented |
| Compare against the standard implementation and explain results | Fixed-policy accuracy and resulting SPMD policies are both compared | Experiments complete; written interpretation needs development |
| Present hypotheses, methodology, findings, limitations, and future directions | Notebook provides methodological explanations, figures, and reproduction instructions | Material is available, but it still needs a report narrative |

The three TD variants can be presented as comparisons **within your policy evaluation enhancement**. The project’s requirement for three distinct enhancements applies to the whole team project.

The notebook answers two useful questions.

First, **how accurately can each method evaluate the same fixed policy?** It uses a uniform policy and reports

\[
\widehat V^\pi(s_0)=\sum_a\pi(a\mid s_0)\widehat Q^\pi(s_0,a),
\]

along with its absolute error against the exact value. Blackjack is evaluated at player total 8/dealer card 3; GridWorld uses its origin. The TD grid search contains 84 candidates, followed by 12 full runs covering both environments, both feature representations, and all three TD methods.

The saved [evaluation summary](<./policy_evaluation_results/evaluation_summary.csv>) shows:

- **Blackjack, tabular:** all three TD methods have the same final value error, **0.011418**. Tuning selected λ = 0 for both λ methods, so their identical curves are expected.
- **GridWorld, tabular:** final errors are **0.016083** for TD(0), **0.017288** for TD(λ), and **0.018526** for True Online TD(λ). True Online TD is therefore not the best final estimator at the selected state in this run.
- **TOMC:** completed after **736 observations** in Blackjack and **265,574** in GridWorld, with errors **0.540625** and **0.440395**, respectively.

Those TOMC results reflect a single completed sampling attempt using first-visit returns. Its stopping rule controls coverage and truncation, so completion does not guarantee low sampling error. Also, these TOMC estimates use far fewer observations than the final TD estimates.

Second, **does replacing TOMC with TD improve policy optimization?** The notebook inserts each evaluator into the same KL-SPMD framework. These experiments are stronger evidence for practical usefulness: they use three tuning seeds and **20 separate test seeds**, with equal training budgets of 120,000 observations for Blackjack and 20 million for GridWorld.

The saved tabular results report the following final optimality gaps; lower is better:

| Evaluator inside KL-SPMD | Blackjack mean gap ± SE | GridWorld mean gap ± SE |
|---|---:|---:|
| TOMC | 0.08719 ± 0.02211 | 0.14158 ± 0.04295 |
| TD(0) | 0.08317 ± 0.01144 | 0.03316 ± 0.00039 |
| TD(λ) | **0.01943 ± 0.01333** | 0.04697 ± 0.00223 |
| True Online TD(λ) | 0.08658 ± 0.02196 | **0.01494 ± 0.00046** |

Source: [tabular SPMD summary](<./results/td_spmd/summary.csv>).

These results support an environment-dependent conclusion: **tabular TD(λ) performed best in Blackjack, while tabular True Online TD(λ) performed best in GridWorld**. The shared linear features generally produced poorer final policies than their tabular counterparts. A plausible explanation is that aggregation forces states with different costs or action preferences to share parameters; this should be presented as an interpretation rather than a proven cause.

Before calling this enhancement fully documented, I would address these points:

1. **Explain mixing length.** The handout names it in the TD example. The current experiments use consecutive transitions, and the SPMD evaluation horizons are fixed at 2,000 observations for Blackjack and 50,000 for GridWorld. I did not find a separate mixing-length parameter or tuning study. Explain your sampling choice and interpretation of this term. λ controls trace decay and should not automatically be described as mixing length.

2. **State the precise TD(λ) update.** Your `td_lambda` implements the accumulated-operator formulation described in its lecture-based docstring. A separate `td_lambda_sarsa` implements accumulating eligibility traces. Include the actual recurrence so readers know which method the reported label represents.

3. **Keep fixed-policy claims descriptive.** Those runs use one test seed. Their final errors cannot establish a consistent ranking, especially when TOMC stops much earlier. The 20-seed SPMD experiments provide broader evidence.

4. **Explain the tuning metric.** Fixed-policy tuning minimizes Q RMSE across all state–action pairs, while the main accuracy figure measures value error at one state. These criteria can favor different methods.

5. **Use all required reporting figures.** Include the [iteration metrics](<./results/td_spmd/iteration_metrics.png>) for the handout’s explicit requirements, the [fixed-policy error figure](<./policy_evaluation_results/tuned_evaluation_value_errors.png>) for evaluation accuracy, and the [policy-value comparison](<./results/td_spmd/policy_value_comparison.png>) for control performance. The report should connect these findings to your original hypotheses and explain the unexpected behavior.

> 1. **Explain mixing length.** The handout names it in the TD example. The current experiments use consecutive transitions, and the SPMD evaluation horizons are fixed at 2,000 observations for Blackjack and 50,000 for GridWorld. I did not find a separate mixing-length parameter or tuning study. Explain your sampling choice and interpretation of this term. λ controls trace decay and should not automatically be described as mixing length.
>
> how to do this and what does this mean ?

<details><summary>2 previous messages</summary>

> I’ll check what “mixing length” means in TD learning and relate it to your implementation, then explain how you could address it in the experiment and report.
>
> <details><summary>Explored a search, searched the web</summary>
>
> Searched the web for `temporal difference learning mixing time Markov chain sample spacing mixing length policy evaluation`
>
> <details><summary>Explored a search</summary>
>
> - Searched for `mixing|mixing length|burn.in|thinning` in `README.md`
>
> </details>
>
> Searched the web for `https://arxiv.org/abs/1907.12530`
>
> </details>
</details>

**Mixing length concerns dependence between samples collected along a trajectory.** Your current TD methods use every consecutive transition. That is a valid sampling choice; my earlier assessment should have described mixing length as something to clarify, rather than a definite missing requirement.

The handout mentions the term but does not define its exact implementation, so the interpretation below should be checked against your lecture notes.

Imagine GridWorld: if the agent is near the origin now, it will probably still be nearby on the next step. Consecutive observations therefore carry related information. After more movement, its location may depend less on where it started.

**Mixing time** describes how long the state distribution takes to approach its long-run distribution under a fixed policy. It is a property of the environment **and policy**. Samples sufficiently far apart can have weaker dependence, which is relevant to TD analysis. [Doan, Maguluri, and Romberg](https://arxiv.org/abs/1907.12530)

A practical **mixing-length parameter** can specify how many environment transitions separate the samples used for learning. This is one possible experimental interpretation—not a definition supplied by your handout.

| Quantity | Meaning in your project |
|---|---|
| Learning rate η | Size of each parameter update |
| Trace parameter λ | How past information contributes to updates |
| Evaluation horizon `n_steps` | Total transitions allocated to one policy evaluation |
| Sampling interval \(m\) | Spacing between transitions selected for learning |

Your existing sampling interval is effectively **\(m=1\)**: learn from every transition. The horizons of 2,000 and 50,000 are evaluation budgets; they do not measure mixing time.

To investigate this experimentally, I would start with a **small TD(0) comparison under the fixed uniform policy**:

1. Try sampling intervals **\(m \in \{1,5,10\}\)**. These are exploratory candidates, not measured mixing times.
2. Continue following the policy at every environment step.
3. Update TD(0) using only every \(m\)-th **one-step transition**. The other transitions advance the environment without updating the parameters.
4. Count **every environment transition** against the observation budget.
5. Compare value error at equal observation counts across several seeds. Use the existing 10% tuning budget, then evaluate the selected configuration with separate seeds.

For example:

```text
m = 1:  learn, learn, learn, learn, learn, ...
m = 5:  learn, move,  move,  move,  move,  learn, ...
```

Each retained sample still contains its actual one-step cost and successor state. Do not connect two retained states into a one-step TD target—that would change the algorithm.

With 100,000 observations, \(m=1\) gives approximately 100,000 updates, while \(m=5\) gives approximately 20,000. The experiment tests the tradeoff between **weaker sample dependence and fewer learning updates**. Larger spacing may perform worse because it discards useful data.

I suggest starting with TD(0) because skipping transitions in TD(λ) and True Online TD(λ) also requires a precise decision about how traces evolve during skipped steps. That would need a separate method specification.

If you keep the current implementation, this would be an accurate report statement:

> We use consecutive on-policy transitions, corresponding to a sampling interval of one, and perform a TD update after every observation. This uses the full observation budget for learning while retaining temporal dependence between samples. We tune the learning rate and trace parameter but do not separately estimate mixing time or tune sample spacing. The evaluation horizon specifies the number of transitions collected per policy evaluation.

That explains what you actually did. **It does not claim that mixing length was tuned.** Whether an additional spacing experiment is required depends on how the instructor defined “mixing length”; the PDF alone does not settle that.

> if i want similar to this figure, in our code where is this ?
>
> User context
>
> Images:
> - `./IMG_8705.jpg`

<details><summary>2 previous messages</summary>

> I’ll locate the plot that corresponds to this slide and check which of its algorithms are already in your code.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `python - <<'PY'
> import json
> n=json.load(open('notebooks/td_evaluation.ipynb'))
> for i,c in enumerate(n['cells']):
>  s=''.join(c['source'])
>  if 'Policy optimization' in s or 'policy_value_comparison.png' in s or 'TD_SPMD_EXPERIMENTS' in s:
>   print('\nCELL index',i,'(cell number',i+1,')\n'+s)
> PY`
> - Ran `rg -n -i 'watkins|q.learning|sarsa|def gap_curve|class TrueOnlineTD|class TDLambda|def td_lambda_sarsa|exact_value|policy_value_comparison' src configs/td_spmd*; ls -l results/td_spmd/policy_value_comparison.png`
>
> </details>
</details>

**The corresponding figure is already in your notebook’s policy optimization section.** Your photo plots the **true value of the changing policy versus total observations**, which matches our SPMD comparison.

Open [td_evaluation.ipynb](<./notebooks/td_evaluation.ipynb>) and find:

> **Policy optimization: tabular and linear Q estimators inside KL-SPMD**

The plotting code is in the **13th notebook cell (index 12)**, beginning with:

```python
# Compare the policy in use at each observation count, using a linear y-axis.
```

The saved figure is [policy_value_comparison.png](<./results/td_spmd/policy_value_comparison.png>).

| Element in your photo | Corresponding code |
|---|---|
| Total observations on x-axis | `x = td_spmd_grid * budget` |
| True policy value on y-axis | `gap_curve(..., key="exact_value")` |
| Method curves | `ax.step(x, mean, ...)` |
| Shaded uncertainty | `ax.fill_between(...)` — ours shows ±1 standard error across 20 seeds |
| Optimal-value reference | `ax.axhline(scorer.optimal, ...)` |

Our figure has **four panels**: Blackjack and GridWorld, each with tabular and linear features. For the photo’s two-panel layout, the closest equivalent would be the **two tabular panels arranged side by side**.

The algorithms differ:

| Photo | Your current comparison |
|---|---|
| KL + TOMC | Included as `baseline` |
| KL + SARSA(λ) | Implementation exists, but is **not included in these saved experiments** |
| Watkins’s Q(λ) | I did not find an implementation |
| — | Your comparison includes TD(0), TD(λ), and True Online TD(λ) |

SARSA(λ) is available as [`SarsaLambda`](<./src/ie692_rl/enhancements/true_online_td/td.py:328>). The notebook’s **TD(λ) curve uses a different update**, so it should retain its current label.

**You already have the data for a similar presentation figure using your existing methods.** Matching the photo’s exact algorithm comparison would require additional experiments.

> Policy optimization: tabular vs. linear (mean ± 1 SE, 20 test seeds)
>
> why 20 seeds from where ?

<details><summary>2 previous messages</summary>

> I’ll trace the “20 test seeds” label to the configuration and check the saved runs.
>
> <details><summary>Ran a command</summary>
>
> - Ran `rg -n 'tuning_seeds|test_seeds|seed|default_rng' configs/td_spmd.cfg configs/td_spmd_linear.cfg tune.py src/ie692_rl/experiment.py src/ie692_rl/core/spmd.py; python - <<'PY'
> import csv, collections
> for p in ['results/td_spmd/history.csv','results/td_spmd_linear/history.csv']:
>  groups=collections.defaultdict(set)
>  for r in csv.DictReader(open(p)): groups[(r['problem'],r['method'])].add(int(r['seed']))
>  print('\n'+p)
>  for k,seeds in sorted(groups.items()): print(k, 'count=',len(seeds),'seeds=',','.join(map(str,sorted(seeds))))
> PY`
>
> </details>
</details>

**“20 seeds” means each method was trained from scratch 20 times with different random-number sequences.** The number comes from your experiment configuration.

In [td_spmd.cfg, line 8](<./configs/td_spmd.cfg:8>):

```ini
test_seeds = 200, 201, 202, ..., 219
```

The [linear configuration](<./configs/td_spmd_linear.cfg:10>) uses the same seeds. I verified that the saved histories contain all 20 for every method and environment.

A seed controls randomness such as card draws, GridWorld movements, and sampled actions. Repeating the experiment shows how sensitive performance is to that randomness.

Your workflow is:

| Stage | Seeds | Purpose |
|---|---|---|
| Parameter tuning | 100, 101, 102 | Select hyperparameters using reduced budgets |
| Full experiments | 200–219 | Evaluate the selected parameters across 20 fresh runs |

At each observation checkpoint, the figure displays:

- **Line:** average policy value across the 20 runs.
- **Shading:** mean ± one standard error, where \(\mathrm{SE}=\mathrm{SD}/\sqrt{20}\).

**Twenty is an experiment-design choice in your code, not a number required by the handout.** It provides evidence beyond a single lucky or unlucky run. The earlier fixed-policy evaluation figures use only one full-run seed, so this label applies specifically to the policy optimization figure.

> in ipynb from where cfg is used ?

In [td_evaluation.ipynb](<./notebooks/td_evaluation.ipynb>), the CFG files are used in the **“Policy optimization: tabular and linear Q estimators inside KL-SPMD”** section.

The **11th cell (index 10)** defines their paths:

```python
TD_SPMD_EXPERIMENTS = {
    "Tabular": (
        root / "configs" / "td_spmd.cfg",
        TD_SPMD_OUTPUT,
    ),
    "Linear": (
        root / "configs" / "td_spmd_linear.cfg",
        root / "results" / "td_spmd_linear",
    ),
}
```

It then reads the tabular configuration:

```python
td_spmd_config = load_config(TD_SPMD_EXPERIMENTS["Tabular"][0])
```

The **12th cell (index 11)** reads each configuration:

```python
for representation, (config_path, result_dir) in TD_SPMD_EXPERIMENTS.items():
    config = load_config(config_path)
```

The configuration is used in two ways:

- **To run experiments:** if results are missing or `RERUN_TD_SPMD = True`, the notebook passes the file to `tune.py` through `--config`.
- **To load and validate saved results:** it reads the expected methods, budgets, feature representations, and seeds. For example:

```python
expected_seeds = {
    int(seed) for seed in config["tuning"]["test_seeds"].split(",")
}
```

Currently, **`RERUN_TD_SPMD = False`**, so existing CSV results are loaded. Editing a CFG alone will not regenerate those results.

The earlier **fixed-policy evaluation** section uses settings defined directly in the notebook, such as `eta_grid`, `lam_grid`, and `N_STEPS_BY_ENV`; it does not use these CFG files.

> Also why are we defining our tomc again here /Users/anjilabudathoki/Downloads/UTK Documents/Fall_2026_Course/IE692/Project/src/ie692_rl/core/tomc.py we were supposed to import from here right ?

<details><summary>2 previous messages</summary>

> I’ll compare the notebook’s TOMC definition with the shared implementation to see whether the duplication is needed.
>
> <details><summary>Explored 2 files, ran a command</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./src/ie692_rl/core/tomc.py`
> - Read `./src/ie692_rl/core/estimator.py`
>
> </details>
>
> - Ran `python - <<'PY'
> import json
> n=json.load(open('notebooks/td_evaluation.ipynb'))
> for i,c in enumerate(n['cells']):
>  s=''.join(c['source'])
>  if 'def tomc(' in s or ('import' in s and 'TOMC' in s): print('CELL',i,'\n'+s)
> PY
> rg -n 'TOMC\(|"tomc"|make_estimator|ESTIMATORS|def make_' src/ie692_rl/experiment.py tests/test_tomc.py`
>
> </details>
</details>

**Yes—the notebook should import the shared `TOMC` implementation from [core/tomc.py](<./src/ie692_rl/core/tomc.py:53>).** Defining the algorithm again creates unnecessary duplication.

The notebook’s local version adds a `log_transition` callback. Its wrapper, `run_tomc_with_history`, uses that callback to record provisional values and coverage during sampling. The shared implementation currently returns only the completed estimate and observation count.

However, **the notebook now plots TOMC only as a single point at completion**. Those intermediate histories therefore do not justify maintaining another implementation of the algorithm.

The intended usage is:

```python
from ie692_rl.core.tomc import TOMC

result = TOMC(bias=TOMC_BIAS).estimate(
    env=env,
    policy=pi,
    rng=np.random.default_rng(seed),
    max_observations=budget,
)

Q = result.q
observations = result.observations
value = estimated_state_value(Q, pi, state)
```

Your two experiment sections currently behave differently:

| Notebook section | TOMC implementation used |
|---|---|
| Fixed-policy evaluation | Local `def tomc(...)` in the notebook |
| Policy optimization through `tune.py` | Shared `core.tomc.TOMC` |

I recommend replacing the local algorithm with the import and retaining a small notebook wrapper for computing the reference error and formatting results. If sampling cannot finish, the wrapper should catch `BudgetExhaustedError` and report an incomplete attempt; the shared interface does not return a provisional Q estimate.

The shared version also uses a compiled sampling loop and bounded-memory return calculation, while the notebook version stores the entire trajectory.

**The fixed-policy TOMC results would need to be regenerated after this change**, because the implementations use different sampling routines and identical seeds need not produce identical trajectories. The saved SPMD experiments already use the shared implementation.

No files have been changed.

> Lets update using the tomc part.

<details><summary>4 previous messages</summary>

> I’ll replace the notebook’s duplicate TOMC implementation with the shared import, update its reporting wrapper, and refresh the TOMC results and affected figures. The TD experiments can stay as they are.
>
> <details><summary>Ran 5 commands</summary>
>
> - Ran `rg --files -g AGENTS.md -g '*td_evaluation*'; git status --short; python - <<'PY'
> import json
> from pathlib import Path
> p=Path('notebooks/td_evaluation.ipynb')
> n=json.loads(p.read_text())
> print('METADATA',n['metadata'])
> for i,c in enumerate(n['cells']):
>  print('\nCELL',i,c['cell_type'],'execution_count',c.get('execution_count'),'outputs',len(c.get('outputs',[])))
>  print(''.join(c['source']))
> PY`
> - Ran `ls .venv/bin/python* .venv/bin/pytest .venv/bin/ruff; sed -n '1,90p' src/ie692_rl/_jit.py; sed -n '1,90p' tests/test_tomc.py; python - <<'PY'
> from pathlib import Path
> import json
> p=Path('notebooks/td_evaluation.ipynb')
> s=p.read_text()
> print('JSON serialization matches:',s==json.dumps(json.loads(s),ensure_ascii=False,indent=1)+'\n')
> PY`
> - Ran ``python - <<'PY'
> import json
> from pathlib import Path
> p = Path('notebooks/td_evaluation.ipynb')
> n = json.loads(p.read_text())
> def source(i):
>     return ''.join(n['cells'][i]['source'])
> def replace(i, old, new):
>     s = source(i)
>     assert old in s, (i, old)
>     n['cells'][i]['source'] = s.replace(old, new).splitlines(keepends=True)
> replace(1, 'from ie692_rl.core.exact import ExactScorer\nfrom ie692_rl.core.policy import check_policy',
>         'from ie692_rl.core.estimator import BudgetExhaustedError\nfrom ie692_rl.core.exact import ExactScorer\nfrom ie692_rl.core.tomc import TOMC')
> replace(2, 'pairs and all 400 GridWorld pairs are required. An incomplete curve is provisional, ends at\nits actual budget, and is labeled accordingly. Completion does not guarantee low statistical\nerror: TOMC uses one first-visit return per pair.',
>         'pairs and all 400 GridWorld pairs are required. Both experiments use the shared\n`ie692_rl.core.tomc.TOMC` estimator. It returns an estimate only on completion; if its\nbudget is exhausted first, report the stopping reason and leave the estimate/error missing.\nCompletion does not guarantee low statistical error: TOMC uses one first-visit return per pair.')
> prefix = source(3).split('\ndef tomc(')[0]
> wrapper = '''
> def run_tomc_evaluation(env, pi, Q_reference, state, budget, seed):
>     """Format one shared TOMC evaluation; incomplete attempts have no estimate."""
>     required_pairs = int((pi >= (1 - env.gamma) / env.n_actions).sum())
>     try:
>         result = TOMC(bias=TOMC_BIAS).estimate(
>             env=env, policy=pi, rng=np.random.default_rng(seed),
>             max_observations=budget,
>         )
>     except BudgetExhaustedError as error:
>         count, final_value = budget, np.nan
>         completed, status = False, "incomplete evaluation"
>         covered_pairs, coverage = np.nan, "Not returned"
>         stopping_reason = str(error)
>     else:
>         count = result.observations
>         final_value = estimated_state_value(result.q, pi, state)
>         completed, status = True, "completed"
>         covered_pairs, coverage = required_pairs, f"{required_pairs}/{required_pairs}"
>         stopping_reason = "coverage and discount-tail criteria satisfied"
>
>     history = state_value_performance([count], [final_value], Q_reference, pi, state)
>     history["method"] = "TOMC"
>     history["completed"] = completed
>     history["status"] = status
>     history["stopping_reason"] = stopping_reason
>     summary = {
>         "method": "TOMC", "features": "Feature independent", "seed": seed,
>         "observations": count, "estimated_value": final_value,
>         "reference_value": history.reference_value.iloc[-1],
>         "absolute_value_error": history.absolute_value_error.iloc[-1],
>         "completed": completed, "status": status,
>         "coverage": coverage, "covered_pairs": covered_pairs,
>         "required_pairs": required_pairs, "stopping_reason": stopping_reason,
>     }
>     return {"value_history": history, "summary": summary}
> '''
> n['cells'][3]['source'] = (prefix + wrapper).splitlines(keepends=True)
> replace(4, 'TOMC may finish earlier under its original stopping rule; its curve ends there.',
>         'TOMC may finish earlier under its original stopping rule; plot its returned estimate there.')
> replace(6, 'run_tomc_with_history(', 'run_tomc_evaluation(')
> replace(7, '**TOMC returns one estimate when its sampling attempt stops**, so show its error/value\nas a single point at the actual observation count. A filled point means completion;\nan open point means an unfinished estimate. Status and stopping count appear in each\npanel. Provisional return prefixes and unvisited-pair upper-bound guesses are saved\nbut omitted from the charts because they are not completed TOMC estimates.',
>         '**The shared TOMC estimator returns one estimate on completion**, so show its\nerror/value as a single filled point at the actual observation count. If the budget\nis exhausted first, no point is plotted and its estimate/error are missing in the CSVs.\nStatus and stopping count appear in each panel; the exception message is retained as\n`stopping_reason` in the summary and single-row TOMC history. No provisional estimates\nor intermediate TOMC histories are reconstructed.')
> replace(8, '            ax.plot(stopping_count, baseline_value, marker="o", linestyle="none",\n                    color=colors["TOMC"], markersize=6, label="TOMC at stopping point",\n                    fillstyle="full" if info["completed"] else "none")',
>         '            if info["completed"]:\n                ax.plot(stopping_count, baseline_value, marker="o", linestyle="none",\n                        color=colors["TOMC"], markersize=6, label="TOMC at stopping point")')
> replace(14, 'Fixed-policy CSVs contain the shared observation checkpoints.',
>         'Fixed-policy TD CSVs contain the shared observation checkpoints. Each TOMC history\ncontains only its stopping record, with a value/error only if evaluation completed.')
> p.write_text(json.dumps(n, ensure_ascii=False, indent=1) + '\n')
> print('Updated shared TOMC import, reporting wrapper, plotting guard, and explanatory text.')
> PY``
> - Ran `.venv/bin/python - <<'PY'
> import importlib.util
> for name in ['nbformat','nbclient','ipykernel','pytest','pandas']:
>  print(name, bool(importlib.util.find_spec(name)))
> PY
> sed -n '90,180p' tests/test_tomc.py
> .venv/bin/ruff check notebooks/td_evaluation.ipynb`
> - Ran `git diff --numstat -- notebooks/td_evaluation.ipynb; git diff -- notebooks/td_evaluation.ipynb`
>
> </details>
>
> The notebook now calls the shared TOMC estimator. It also reports unfinished evaluations with missing value/error fields and a stopping reason, since the shared API returns no partial estimate. I’m checking both completion and budget-exhaustion handling, then refreshing the TOMC points and figures.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `cat > /tmp/refresh_ie692_tomc.py <<'PY'
> import ast
> import hashlib
> import json
> from pathlib import Path
>
> import matplotlib
> matplotlib.use('module://matplotlib_inline.backend_inline')
> import nbformat
> from IPython.core.interactiveshell import InteractiveShell
> from IPython.utils.capture import capture_output
>
> root = Path.cwd()
> path = root / 'notebooks/td_evaluation.ipynb'
> initial_text = path.read_text()
> notebook = json.loads(initial_text)
> source = lambda i: ''.join(notebook['cells'][i]['source'])
>
> # TD and policy-optimization artifacts must be reused without rewriting them.
> protected = list((root / 'results/td_spmd').glob('*'))
> protected += list((root / 'results/td_spmd_linear').glob('*'))
> protected += list((root / 'policy_evaluation_results').glob('td_*.csv'))
> before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
>
> shell = InteractiveShell.instance()
> for i in [1, 3]:
>     with capture_output():
>         execution = shell.run_cell(source(i))
>     execution.raise_error()
> ns = shell.user_ns
> np = ns['np']
> pd = ns['pd']
>
> # Check that the notebook delegates the algorithm, including the failure path.
> assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'tomc'
>                for i, c in enumerate(notebook['cells']) if c['cell_type'] == 'code'
>                for n in ast.walk(ast.parse(source(i))))
> for name in ns['ENVIRONMENTS']:
>     env, pi, state = ns['make_evaluation'](name)
>     reference = ns['ExactScorer'](env).q_values(pi)
>     failed = ns['run_tomc_evaluation'](env, pi, reference, state, 1, ns['SEED'] + 1)
>     summary = failed['summary']
>     assert not summary['completed'] and summary['observations'] == 1
>     assert np.isnan(summary['estimated_value']) and np.isnan(summary['absolute_value_error'])
>     assert np.isnan(summary['covered_pairs'])
>     assert 'TOMC incomplete:' in summary['stopping_reason']
>     assert len(failed['value_history']) == 1
>     print(f'{name}: budget-exhaustion reporting passed')
>
> # Refresh only the TOMC suffix of the full-run cell, using existing TD results.
> ns['tuned_value_results'] = pd.read_csv(ns['output'] / 'td_tuned_value_history.csv')
> ns['full_run_results'] = pd.read_csv(ns['output'] / 'td_tuned_full_runs.csv')
> ns['full_rows'] = ns['full_run_results'].to_dict('records')
> tomc_code = 'tomc_results = {}' + source(6).split('tomc_results = {}', 1)[1]
>
> def execute_and_capture(code):
>     with capture_output() as captured:
>         execution = shell.run_cell(code)
>     execution.raise_error()
>     outputs = []
>     if captured.stdout:
>         outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
>     if captured.stderr:
>         outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
>     for rich in captured.outputs:
>         outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
>     return outputs
>
> notebook['cells'][6]['outputs'] = execute_and_capture(tomc_code)
> for name, result in ns['tomc_results'].items():
>     env, pi, state = ns['make_evaluation'](name)
>     direct = ns['TOMC'](bias=ns['TOMC_BIAS']).estimate(
>         env, pi, np.random.default_rng(ns['SEED'] + 1), ns['N_STEPS_BY_ENV'][name])
>     summary = result['summary']
>     assert summary['completed']
>     assert summary['observations'] == direct.observations
>     assert summary['estimated_value'] == ns['estimated_state_value'](direct.q, pi, state)
>     assert len(result['value_history']) == 1
>     print(f"{name}: shared-estimator result verified; {summary['observations']:,} observations, "
>           f"value={summary['estimated_value']:.6f}, error={summary['absolute_value_error']:.6f}")
>
> notebook['cells'][8]['outputs'] = execute_and_capture(source(8))
> figures = [o for o in notebook['cells'][8]['outputs'] if 'image/png' in o.get('data', {})]
> assert len(figures) == 2, f'Expected two embedded figures; got {len(figures)}'
> assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in before.items())
> assert path.read_text() == initial_text, 'Notebook changed externally during refresh'
> nbformat.validate(nbformat.from_dict(notebook))
> path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
> print('Refreshed summary, single-row TOMC histories, both figures, and embedded notebook outputs.')
> print('Verified existing TD and policy-optimization artifacts are unchanged.')
> PY
> .venv/bin/python /tmp/refresh_ie692_tomc.py`
> - Ran `.venv/bin/python -m pytest tests/test_tomc.py -k 'not memory_does_not_grow_with_stream_length and not unbiased_for_required_pairs' -q`
> - Ran `.venv/bin/ruff check notebooks/td_evaluation.ipynb
> .venv/bin/python - <<'PY'
> import ast,json
> from pathlib import Path
> import nbformat
> p=Path('notebooks/td_evaluation.ipynb')
> n=json.loads(p.read_text())
> nbformat.validate(nbformat.from_dict(n))
> for i,c in enumerate(n['cells']):
>  if c['cell_type']=='code':
>   ast.parse(''.join(c['source']))
>   assert all(o['output_type'] != 'error' for o in c.get('outputs', [])), i
> s='\n'.join(''.join(c['source']) for c in n['cells'])
> for stale in ['def tomc(', 'run_tomc_with_history(', 'log_transition', 'Q_prefix', 'an open point']:
>  assert stale not in s, stale
> for name in ['blackjack','gridworld']:
>  p=Path('policy_evaluation_results') / f'{name}_tuned_tomc_value_history.csv'
>  print(p, 'records:',len(p.read_text().splitlines())-1)
> print('Notebook schema, code cells, refreshed outputs, and removal of duplicate TOMC verified.')
> PY
> git status --short`
>
> </details>
</details>

Updated [td_evaluation.ipynb](<./notebooks/td_evaluation.ipynb>) to use:

```python
from ie692_rl.core.tomc import TOMC
```

Replaced the duplicate algorithm with a reporting wrapper. Incomplete evaluations now record a stopping reason and missing value/error.

Refreshed the TOMC CSVs and notebook figures; numerical results stayed the same. Existing TD and SPMD results were preserved.

Validation: **50 targeted tests passed**, notebook validation and Ruff passed.

> I am trying to understand here for the performance measurement. The main role of policy evaluation is given a policy, estimate its value V(\pi) from sampled data. Our goal is to check how well each method does that job. But  better evaluation only counts if it leads to better decisions so we compare current policy vs best possible policy i.e do better estimate give better policies.  is Is my understanding correct ?

**Yes, your understanding is correct, with one qualification: evaluation accuracy and decision quality are two separate measures of success.** Better evaluation is valuable even when it does not improve the resulting policy.

Your notebook investigates both:

1. **How accurately does the method evaluate a given policy?**

   Keep the policy \(\pi\) fixed, collect sampled transitions, and compare the estimated value with its true value:

   \[
   \left|\widehat V^\pi(s_0)-V^\pi(s_0)\right|.
   \]

   **Smaller error means better evaluation at that state.** This is what your fixed-policy experiments measure.

2. **How useful is the evaluator for learning better policies?**

   Put each evaluator inside the same SPMD framework. It estimates \(Q^{\pi_t}\), which SPMD uses to update the policy. Then measure the resulting policy’s true optimality gap:

   \[
   V^{\pi_t}(s_0)-V^*(s_0).
   \]

   Your project minimizes costs, so **a smaller gap means a better policy**. This is what your policy optimization experiments measure.

There is **no guarantee that a more accurate evaluator produces a better policy**. Policy updates depend on the estimated differences between actions. An estimator can have a small value error at \(s_0\) while ranking actions incorrectly elsewhere. Sampling cost also matters: an accurate but expensive evaluation may allow fewer policy updates within the same observation budget.

Also, the optimization comparison must use the **true value of the learned policy**, not just its estimated value. An inaccurate evaluator could otherwise make a poor policy appear good.

You can describe your study as:

> We first compare how accurately each method evaluates a fixed policy from sampled data. We then compare the methods within the same policy optimization framework to determine whether their estimates lead to better policies under equal observation budgets.
