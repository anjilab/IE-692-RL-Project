"""Write the M = 1 part of results/param_sweep/scale_0.1 (the 10% sweep, 100 seeds) as a tuning.csv that
``tune.py --reuse-tuning`` can read, for the methods baseline and entropy_reg.

    .venv/bin/python benchmarks/export_tuning.py
    .venv/bin/python tune.py --reuse-tuning --methods baseline entropy_reg --output results/full_runs
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from ie692_rl.experiment import Row, mean_and_se, read_csv, write_csv

ROOT = Path(__file__).resolve().parent.parent
BIAS = 0.01


def main() -> None:
    groups: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for r in read_csv(ROOT / "results" / "param_sweep" / "scale_0.1" / "runs.csv"):
        if r["rollouts"] == 1:
            groups[r["problem"], float(r["tau"]), float(r["step_size"])].append(r)
    rows: list[Row] = []
    for (problem, tau, eta), runs in sorted(
        groups.items(), key=lambda item: (item[0][0] != "GridWorld", item[0])
    ):
        mean, se = mean_and_se([r["gap"] for r in runs])
        rows.append(
            {
                "problem": problem,
                "method": "entropy_reg" if tau > 0 else "baseline",
                "estimator.bias": BIAS,
                "spmd.step_size": eta,
                **({"spmd.tau": tau} if tau > 0 else {}),
                "tuning_budget": runs[0]["budget"],
                "seeds": len(runs),
                "min_updates": min(r["updates"] for r in runs),
                "mean_final_gap": mean,
                "standard_error": se,
            }
        )
    output = ROOT / "results" / "full_runs"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "tuning.csv", rows)
    print(f"wrote {len(rows)} rows to {output / 'tuning.csv'}")


if __name__ == "__main__":
    main()
