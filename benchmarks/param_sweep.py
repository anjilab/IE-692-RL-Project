"""Grid search over tau x step size x rollouts for entropy-regularized SPMD (TOMC bias fixed at 0.01).

A *rollout* is one independent TOMC estimate; with M rollouts, Q is the average of M estimates of the
same policy before each update (all M count against the observation budget). M = 1 is the usual method.

    tau        0.001, 0.01, 0.1      (tau = 0, the baseline, runs on the same step sizes and rollouts)
    step size  1, 10, 100
    rollouts   1, 4, 16

Every combination runs on --budget-scale times the config's full budget with the same seeds, which are
separate from the tuning (100-102) and test (200-219) seeds. Score: exact gap of the final policy.

    .venv/bin/python benchmarks/param_sweep.py                       # 10% of the full budget
    .venv/bin/python benchmarks/param_sweep.py --budget-scale 1
    .venv/bin/python benchmarks/param_sweep.py --plot-only           # redraw from runs.csv
    .venv/bin/python benchmarks/param_sweep.py --confirm             # search strategies, fresh seeds

Writes to results/param_sweep/scale_<s>/: runs.csv (one row per run), grid.csv (one row per
combination), tau_step_heatmap.png (M = 1), joint_heatmap.png (every M) and confirm.csv (--confirm).
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

from ie692_rl.core.estimator import Estimate, QEstimator
from ie692_rl.core.exact import ExactScorer
from ie692_rl.core.spmd import run_spmd
from ie692_rl.core.tomc import TOMC
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.envs.base import Environment
from ie692_rl.experiment import (
    Row,
    default_workers,
    load_config,
    make_env,
    mean_and_se,
    problem_budget,
    read_csv,
    write_csv,
)
from ie692_rl.types import FloatArray

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.cfg"
PROBLEMS = ("GridWorld", "Blackjack")
TAUS = (0.001, 0.01, 0.1)
STEP_SIZES = (1.0, 10.0, 100.0)
ROLLOUTS = (1, 4, 16)
BIAS = 0.01
FIRST_SEED, CONFIRM_SEED = 1000, 5000
SURFACE, INK, INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
GREENS = LinearSegmentedColormap.from_list("greens", ["#e6f4e6", "#8fcf8f", "#2f9a2f", "#005300"])


@dataclass(frozen=True)
class AveragedEstimator(QEstimator):
    """Mean of ``rollouts`` independent estimates of the same policy."""

    inner: QEstimator
    rollouts: int = 1

    def estimate(
        self,
        env: Environment,
        policy: FloatArray,
        rng: np.random.Generator,
        max_observations: int,
        extra_cost: FloatArray | None = None,
    ) -> Estimate:
        total, used, visited = np.zeros_like(policy), 0, 0
        for _ in range(self.rollouts):  # a budget overrun raises: an incomplete batch makes no update
            q, n, seen = self.inner.estimate(env, policy, rng, max_observations - used, extra_cost)
            total += q
            used += n
            visited = max(visited, seen)
        return Estimate(total / self.rollouts, used, visited)


Job = tuple[str, float, float, int, int, int]  # problem, tau, step size, rollouts, seed, budget


@cache
def problem(name: str) -> tuple[Environment, ExactScorer]:
    env = make_env(load_config(CONFIG), name)
    return env, ExactScorer(env)


def run_job(job: Job) -> Row:
    name, tau, eta, rollouts, seed, budget = job
    env, scorer = problem(name)
    estimator = AveragedEstimator(TOMC(BIAS), rollouts)
    history = run_spmd(env, estimator, KL(), eta, budget, seed, tau=tau, scorer=scorer).history
    return {
        "problem": name,
        "tau": tau,
        "step_size": eta,
        "rollouts": rollouts,
        "seed": seed,
        "budget": budget,
        "updates": sum(r.status == "updated" for r in history),
        "gap": history[-1].gap,
    }


def run_jobs(jobs: list[Job], workers: int) -> list[Row]:
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(run_job, sorted(jobs, key=lambda j: -j[-1]), chunksize=4))


def aggregate(runs: list[Row]) -> list[Row]:
    """One row per combination."""
    groups: dict[tuple[Any, ...], dict[int, Row]] = defaultdict(dict)
    for r in runs:
        groups[r["problem"], r["tau"], r["step_size"], r["rollouts"]][r["seed"]] = r
    rows = []
    for (name, tau, eta, m), by_seed in sorted(groups.items()):
        mean, se = mean_and_se([r["gap"] for r in by_seed.values()])
        updates = [r["updates"] for r in by_seed.values()]
        rows.append(
            {
                "problem": name,
                "tau": tau,
                "step_size": eta,
                "rollouts": m,
                "seeds": len(by_seed),
                "budget": next(iter(by_seed.values()))["budget"],
                "mean_gap": mean,
                "standard_error": se,
                "optimal_seeds": sum(r["gap"] < 1e-9 for r in by_seed.values()),
                "median_updates": float(np.median(updates)),
                "no_update_seeds": sum(u == 0 for u in updates),
            }
        )
    return rows


def key_of(r: Row) -> tuple[Any, ...]:
    return r["problem"], r["tau"], r["step_size"], r["rollouts"]


def paired(runs: list[Row], a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[float, float]:
    """Mean and standard error of gap(a) - gap(b) over their shared seeds."""
    gaps: dict[tuple[Any, ...], dict[int, float]] = defaultdict(dict)
    for r in runs:
        gaps[key_of(r)][r["seed"]] = r["gap"]
    return mean_and_se([gaps[a][s] - gaps[b][s] for s in gaps[a]])


def best(rows: list[Row], name: str, *, baseline: bool = False, rollouts: int | None = None) -> Row:
    candidates = [
        r
        for r in rows
        if r["problem"] == name
        and (r["tau"] == 0) == baseline
        and (rollouts is None or r["rollouts"] == rollouts)
    ]
    return min(candidates, key=lambda r: r["mean_gap"])


def heatmaps(
    rows: list[Row], runs: list[Row], rollouts: tuple[int, ...], path: Path, *, brief: bool = False
) -> None:
    """Rows: problems; columns: rollouts. Darker green = smaller gap; the best cell is outlined and cells
    within 2 SE (paired by seed) of it are marked with ~. `brief` drops those marks and shortens the title."""
    fig, axes = plt.subplots(
        len(PROBLEMS) if len(rollouts) > 1 else 1,
        len(rollouts) if len(rollouts) > 1 else len(PROBLEMS),
        figsize=(
            4.3 * max(len(rollouts), len(PROBLEMS)),
            3.9 * (len(PROBLEMS) if len(rollouts) > 1 else 1) + 0.3,
        ),
        squeeze=False,
        layout="constrained",
        facecolor=SURFACE,
    )
    for i, name in enumerate(PROBLEMS):
        shown = [r for r in rows if r["problem"] == name and r["tau"] > 0 and r["rollouts"] in rollouts]
        gaps = [r["mean_gap"] for r in shown]
        norm = Normalize(min(gaps), max(gaps) if max(gaps) > min(gaps) else min(gaps) + 1e-9)
        winner = min(shown, key=lambda r: r["mean_gap"])
        row_axes = axes[i] if len(rollouts) > 1 else axes[0][i : i + 1]
        for ax, m in zip(row_axes, rollouts, strict=True):
            grid = np.full((len(TAUS), len(STEP_SIZES)), np.nan)
            for r in shown:
                if r["rollouts"] != m:
                    continue
                y, x = TAUS.index(r["tau"]), STEP_SIZES.index(r["step_size"])
                grid[y, x] = r["mean_gap"]
                diff, se = paired(runs, key_of(r), key_of(winner))
                mark = "" if brief or r is winner or diff > 2 * se else " ~"
                red, green, blue, _ = GREENS(1 - norm(r["mean_gap"]))
                color = "#ffffff" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45 else INK
                style: dict[str, Any] = {"ha": "center", "va": "center", "color": color}
                ax.text(x, y + 0.1, f"{r['mean_gap']:.3f}{mark}", fontsize=12, **style)
                ax.text(x, y - 0.2, f"± {r['standard_error']:.3f}", fontsize=8, **style)
                if r is winner:
                    ax.add_patch(
                        Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, lw=2.5)
                    )
            image = ax.imshow(grid, cmap=GREENS.reversed(), norm=norm, aspect="auto", origin="lower")
            ax.set_xticks(range(len(STEP_SIZES)), [f"{e:g}" for e in STEP_SIZES])
            ax.set_yticks(range(len(TAUS)), [f"{t:g}" for t in TAUS])
            ax.set_xticks(np.arange(len(STEP_SIZES) - 1) + 0.5, minor=True)
            ax.set_yticks(np.arange(len(TAUS) - 1) + 0.5, minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="both", length=0, labelsize=9, labelcolor=INK_SECONDARY)
            ax.spines[:].set_visible(False)
            suffix = f" · {m} rollout{'s' if m > 1 else ''} per update" if len(rollouts) > 1 else ""
            ax.set_title(f"{name}{suffix}", loc="left", fontsize=11, color=INK)
            ax.set_xlabel("step size η", fontsize=10, color=INK_SECONDARY)
            ax.set_ylabel("entropy weight τ", fontsize=10, color=INK_SECONDARY)
        bar = fig.colorbar(image, ax=list(row_axes), shrink=0.85, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.invert_yaxis()
        bar.ax.tick_params(labelsize=8, labelcolor=INK_SECONDARY, length=0)
        bar.set_label("gap to optimal (darker = better)", fontsize=8, color=INK_SECONDARY)
    seeds = rows[0]["seeds"]
    budgets = {r["problem"]: r["budget"] for r in rows}
    budget_text = ", ".join(f"{n} {budgets[n]:,} steps" for n in PROBLEMS)
    title = (
        f"Entropy-Reg SPMD\n{budget_text}. Box = best"
        if brief
        else f"Entropy-Reg SPMD: mean final gap ± SE, {seeds} seeds, TOMC bias {BIAS:g}\n"
        f"{budget_text}. Box = best; ~ = within 2 SE of best"
    )
    fig.suptitle(
        title,
        x=0.01,
        ha="left",
        fontsize=10.5,
        color=INK,
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def report(rows: list[Row], runs: list[Row]) -> None:
    for name in PROBLEMS:
        print(f"\n{name} ({best(rows, name)['budget']:,} steps, {rows[0]['seeds']} seeds)")
        print("  tau    eta  M   gap              optimal  updates  no-update seeds")
        for r in sorted((r for r in rows if r["problem"] == name), key=lambda r: r["mean_gap"]):
            print(
                f"  {r['tau']:<6g} {r['step_size']:<4g} {r['rollouts']:<3d} {r['mean_gap']:.4f} ± "
                f"{r['standard_error']:.4f}  {r['optimal_seeds']:3d}      {r['median_updates']:5.0f}    "
                f"{r['no_update_seeds']}"
            )
        joint = best(rows, name)
        for label, other in [
            ("best M=1 cell", best(rows, name, rollouts=1)),
            ("best baseline M=1", best(rows, name, baseline=True, rollouts=1)),
            ("best baseline any M", best(rows, name, baseline=True)),
        ]:
            diff, se = paired(runs, key_of(joint), key_of(other))
            z = diff / se if se > 0 else 0.0
            print(
                f"  joint best {key_of(joint)[1:]} vs {label} {key_of(other)[1:]}: {diff:+.4f} (z {z:+.1f})"
            )


def strategies(rows: list[Row], name: str) -> dict[str, Row]:
    """The setting each search strategy would choose from one sweep."""
    first = best(rows, name, rollouts=1)
    same = [r for r in rows if key_of(r)[:3] == key_of(first)[:3]]
    return {
        "baseline, M=1": best(rows, name, baseline=True, rollouts=1),
        "baseline, joint eta x M": best(rows, name, baseline=True),
        "entropy, tau x eta only (M=1)": first,
        "entropy, sequential (then M)": min(same, key=lambda r: r["mean_gap"]),
        "entropy, joint 3x3x3": best(rows, name),
    }


def confirm(scale: float, seeds: int, workers: int, output: Path) -> None:
    """Run every strategy's choice on 10x the sweep budget (so the sweep was 10%) with fresh seeds."""
    rows = list(read_csv(output / "grid.csv"))
    config = load_config(CONFIG)
    picks = {name: strategies(rows, name) for name in PROBLEMS}
    settings = {key_of(r) for chosen in picks.values() for r in chosen.values()}
    runs = run_jobs(
        [
            (*key, seed, round(10 * scale * problem_budget(config, key[0])))
            for key in sorted(settings)
            for seed in range(CONFIRM_SEED, CONFIRM_SEED + seeds)
        ],
        workers,
    )
    write_csv(output / "confirm_runs.csv", runs)
    final = {key_of(r): r for r in aggregate(runs)}
    table = []
    for name, chosen in picks.items():
        print(
            f"\n{name}: full budget {round(10 * scale * problem_budget(config, name)):,}, {seeds} fresh seeds"
        )
        reference = key_of(chosen["baseline, M=1"])
        rival = key_of(chosen["baseline, joint eta x M"])
        for label, r in chosen.items():
            key, f = key_of(r), final[key_of(r)]
            z = []
            for other in (reference, rival):
                diff, se = paired(runs, key, other)
                z.append(diff / se if se > 0 else 0.0)
            print(
                f"  {label:30s} tau {key[1]:<6g} eta {key[2]:<4g} M {key[3]:<3d} gap {f['mean_gap']:.4f} ± "
                f"{f['standard_error']:.4f}  optimal {f['optimal_seeds']:3d}"
                f"  updates {f['median_updates']:.0f}"
                f"  z vs baseline M=1 {z[0]:+.1f}, vs baseline joint {z[1]:+.1f}"
            )
            table.append({"strategy": label, **f, "z_vs_baseline_m1": z[0], "z_vs_baseline_joint": z[1]})
    write_csv(output / "confirm.csv", table)


def baseline_heatmap(rows: list[Row], runs: list[Row], path: Path) -> None:
    """Baseline (tau = 0): step size x rollouts, one panel per problem. x = a seed made no update."""
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(9.2, 4.4), layout="constrained", facecolor=SURFACE)
    for ax, name in zip(axes, PROBLEMS, strict=True):
        shown = [r for r in rows if r["problem"] == name and r["tau"] == 0]
        winner = min((r for r in shown if r["no_update_seeds"] == 0), key=lambda r: r["mean_gap"])
        gaps = [r["mean_gap"] for r in shown]
        norm = Normalize(min(gaps), max(gaps))
        grid = np.full((len(ROLLOUTS), len(STEP_SIZES)), np.nan)
        for r in shown:
            y, x = ROLLOUTS.index(r["rollouts"]), STEP_SIZES.index(r["step_size"])
            grid[y, x] = r["mean_gap"]
            diff, se = paired(runs, key_of(r), key_of(winner))
            mark = " x" if r["no_update_seeds"] else " ~" if r is not winner and diff <= 2 * se else ""
            red, green, blue, _ = GREENS(1 - norm(r["mean_gap"]))
            color = "#ffffff" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.45 else INK
            style: dict[str, Any] = {"ha": "center", "va": "center", "color": color}
            ax.text(x, y + 0.1, f"{r['mean_gap']:.3f}{mark}", fontsize=12, **style)
            ax.text(x, y - 0.2, f"± {r['standard_error']:.3f}", fontsize=8, **style)
            if r is winner:
                ax.add_patch(Rectangle((x - 0.46, y - 0.46), 0.92, 0.92, fill=False, edgecolor=INK, lw=2.5))
        image = ax.imshow(grid, cmap=GREENS.reversed(), norm=norm, aspect="auto", origin="lower")
        ax.set_xticks(range(len(STEP_SIZES)), [f"{e:g}" for e in STEP_SIZES])
        ax.set_yticks(range(len(ROLLOUTS)), [str(m) for m in ROLLOUTS])
        ax.set_xticks(np.arange(len(STEP_SIZES) - 1) + 0.5, minor=True)
        ax.set_yticks(np.arange(len(ROLLOUTS) - 1) + 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0, labelsize=9, labelcolor=INK_SECONDARY)
        ax.spines[:].set_visible(False)
        ax.set_title(f"{name} · {winner['budget']:,} steps", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("step size η", fontsize=10, color=INK_SECONDARY)
        ax.set_ylabel("rollouts per update M", fontsize=10, color=INK_SECONDARY)
        bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.invert_yaxis()
        bar.ax.tick_params(labelsize=8, labelcolor=INK_SECONDARY, length=0)
        bar.set_label("gap to optimal (darker = better)", fontsize=8, color=INK_SECONDARY)
    fig.suptitle(
        f"Baseline SPMD (τ = 0): mean final gap ± SE, {rows[0]['seeds']} seeds, TOMC bias {BIAS:g}\n"
        "Box = chosen; ~ = within 2 SE of it; x = not eligible (a seed made no update)",
        x=0.01,
        ha="left",
        fontsize=10.5,
        color=INK,
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def baseline(scale: float, seeds: int, workers: int, output: Path, *, evaluate: bool) -> None:
    """Choose the baseline from the sweep; with ``evaluate``, run all nine on 10x budget, fresh seeds."""
    runs = list(read_csv(output / "runs.csv"))
    rows = aggregate(runs)
    baseline_heatmap(rows, runs, output / "baseline_heatmap.png")
    config = load_config(CONFIG)
    final: dict[tuple[Any, ...], Row] = {}
    fresh: list[Row] = []
    if evaluate:
        fresh = run_jobs(
            [
                (name, 0.0, eta, m, seed, round(10 * scale * problem_budget(config, name)))
                for name in PROBLEMS
                for eta in STEP_SIZES
                for m in ROLLOUTS
                for seed in range(CONFIRM_SEED, CONFIRM_SEED + seeds)
            ],
            workers,
        )
        write_csv(output / "baseline_full_runs.csv", fresh)
        final = {key_of(r): r for r in aggregate(fresh)}
    for name in PROBLEMS:
        shown = [r for r in rows if r["problem"] == name and r["tau"] == 0]
        eligible = [r for r in shown if r["no_update_seeds"] == 0]
        chosen = min(eligible, key=lambda r: r["mean_gap"])
        single = min((r for r in eligible if r["rollouts"] == 1), key=lambda r: r["mean_gap"])
        print(
            f"\n{name}: chosen eta {chosen['step_size']:g} M {chosen['rollouts']}, "
            f"M = 1 only: eta {single['step_size']:g}"
        )
        for r in sorted(shown, key=lambda r: r["mean_gap"]):
            diff, se = paired(runs, key_of(r), key_of(chosen))
            text = (
                f"  eta {r['step_size']:<4g} M {r['rollouts']:<3d} sweep {r['mean_gap']:.4f} ± "
                f"{r['standard_error']:.4f} z {diff / se if se > 0 else 0.0:+5.1f}"
                f"  no-update {r['no_update_seeds']:3d}"
            )
            if key_of(r) in final:
                f = final[key_of(r)]
                diff, se = paired(fresh, key_of(r), key_of(chosen))
                text += (
                    f" | full run {f['mean_gap']:.4f} ± {f['standard_error']:.4f}"
                    f" z {diff / se if se > 0 else 0.0:+5.1f} optimal {f['optimal_seeds']:3d}"
                    f" updates {f['median_updates']:.0f}"
                )
            print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--budget-scale", type=float, default=0.1, help="fraction of the config's full budget"
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--confirm", action="store_true", help="compare search strategies on fresh seeds")
    parser.add_argument("--baseline", action="store_true", help="choose the baseline from runs.csv")
    parser.add_argument("--evaluate", action="store_true", help="with --baseline: 10x budget, fresh seeds")
    args = parser.parse_args()
    output = ROOT / "results" / "param_sweep" / f"scale_{args.budget_scale:g}"
    output.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        baseline(args.budget_scale, args.seeds, args.workers, output, evaluate=args.evaluate)
        return
    if args.confirm:
        confirm(args.budget_scale, args.seeds, args.workers, output)
        return
    if args.plot_only:
        runs = list(read_csv(output / "runs.csv"))
    else:
        config = load_config(CONFIG)
        runs = run_jobs(
            [
                (name, tau, eta, m, seed, round(args.budget_scale * problem_budget(config, name)))
                for name in PROBLEMS
                for tau in (0.0, *TAUS)
                for eta in STEP_SIZES
                for m in ROLLOUTS
                for seed in range(FIRST_SEED, FIRST_SEED + args.seeds)
            ],
            args.workers,
        )
        write_csv(output / "runs.csv", runs)
    rows = aggregate(runs)
    write_csv(output / "grid.csv", rows)
    heatmaps(rows, runs, (1,), output / "tau_step_heatmap.png")
    heatmaps(rows, runs, ROLLOUTS, output / "joint_heatmap.png", brief=True)
    report(rows, runs)
    print(f"\nsaved {output}")


if __name__ == "__main__":
    main()
