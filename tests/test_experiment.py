"""Config parsing, the tuning protocol, statistics, CSV round trips, and an end-to-end tune.py run."""

import math
from pathlib import Path

import numpy as np
import pytest

import tune
from ie692_rl.core import TOMC
from ie692_rl.enhancements.bregman_spmd import KL
from ie692_rl.enhancements.entropy_reg_spmd import AnnealedTau
from ie692_rl.enhancements.true_online_td import TrueOnlineTD
from ie692_rl.envs import GridWorld, RetailInventory
from ie692_rl.experiment import (
    MethodSpec,
    best_per_method,
    build_method,
    figure_groups,
    gap_curve,
    int_list,
    load_config,
    make_certifier,
    make_env,
    mean_and_se,
    method_label,
    method_spec,
    names,
    parse_list,
    parse_value,
    problem_budget,
    problem_label,
    read_csv,
    selection_of,
    summarize,
    write_csv,
    z_verdict,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(ROOT / "config.cfg")

TINY_CONFIG = """
[tuning]
problems = Blackjack
methods = baseline, td, euclid
baseline = baseline
tune_fraction = 0.5
tuning_seeds = 1, 2
test_seeds = 3, 4, 5
z_threshold = 2.0
min_worse_diff = 0.001

[problem Blackjack]
environment = blackjack
budget = 6000

[method baseline]
estimator = tomc
divergence = kl
spmd.step_size = 1.0, 10.0
estimator.bias = 0.01

[method td]
estimator = true_online_td
divergence = kl
spmd.step_size = 10.0
estimator.eta = 0.1
estimator.lam = 0.5, 0.9
estimator.n_steps = 500

[method euclid]
estimator = tomc
divergence = euclidean
spmd.step_size = 1.0
"""


def write_config(tmp_path: Path, text: str = TINY_CONFIG) -> Path:
    path = tmp_path / "config.cfg"
    path.write_text(text)
    return path


# ----------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("3", 3),
        (" 2.5 ", 2.5),
        ("1e-3", 1e-3),
        ("true", True),
        ("False", False),
        ("tabular", "tabular"),
        ("nan", None),
    ],
)
def test_parse_value(text: str, value: object) -> None:
    parsed = parse_value(text)
    if value is None:
        assert isinstance(parsed, float)
        assert math.isnan(parsed)
    else:
        assert parsed == value
        assert type(parsed) is type(value)


def test_lists() -> None:
    assert parse_list("1, 2.5, x,") == [1, 2.5, "x"]
    assert names(" GridWorld , Blackjack ") == ["GridWorld", "Blackjack"]
    assert int_list("1, 2") == [1, 2]
    with pytest.raises(ValueError, match="integers"):
        int_list("1, 2.5")


# ----------------------------------------------------------------------------- the repository config


def test_repository_config_follows_the_handout() -> None:
    grid = make_env(CONFIG, "GridWorld")
    assert isinstance(grid, GridWorld)
    assert (grid.rows, len(grid.hazards), len(grid.goals), grid.gamma) == (10, 10, 3, 0.9)
    assert make_env(CONFIG, "Blackjack").gamma == 0.9
    assert 0.05 <= float(CONFIG["tuning"]["tune_fraction"]) <= 0.1  # guideline: 5-10% of the full budget
    assert CONFIG["tuning"]["baseline"] in names(CONFIG["tuning"]["methods"])


def test_every_method_parses_on_every_problem_with_at_most_three_tuned_parameters() -> None:
    problems = [s.removeprefix("problem ") for s in CONFIG.sections() if s.startswith("problem ")]
    for problem in problems:
        for method in names(CONFIG["tuning"]["methods"]):
            spec = method_spec(CONFIG, method, problem)
            assert len(spec.tuned) <= 3
            assert all(len(values) <= 3 for values in spec.grid.values())
            build_method(spec, spec.combinations()[0])  # every combination's classes can be built


def test_inventory_settings_and_step_size_override() -> None:
    env = make_env(CONFIG, "Inventory")
    assert isinstance(env, RetailInventory)
    assert env.capacity == 12
    assert method_spec(CONFIG, "baseline", "Inventory").grid["spmd.step_size"] == [0.1, 1.0, 10.0]
    assert method_spec(CONFIG, "true_online_td", "Blackjack").grid["estimator.n_steps"] == [2000]


# ----------------------------------------------------------------------------- method specs


def test_combinations_and_building() -> None:
    spec = method_spec(CONFIG, "entropy_reg_annealed", "GridWorld")
    combos = spec.combinations()
    expected = math.prod(len(values) for values in spec.grid.values())
    assert expected > 1
    assert len(combos) == expected
    assert len(set(combos)) == expected
    method = build_method(spec, combos[0])
    assert isinstance(method.estimator, TOMC)
    assert isinstance(method.divergence, KL)
    assert isinstance(method.tau, AnnealedTau)
    td = build_method(
        method_spec(CONFIG, "true_online_td", "GridWorld"),
        method_spec(CONFIG, "true_online_td", "GridWorld").combinations()[0],
    )
    assert isinstance(td.estimator, TrueOnlineTD)
    assert td.tau == 0.0


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("estimator = tomc\ndivergence = kl\n", "step_size is required"),
        ("estimator = mc\ndivergence = kl\nspmd.step_size = 1", "unknown estimator"),
        ("estimator = tomc\ndivergence = l2\nspmd.step_size = 1", "unknown estimator"),
        ("estimator = tomc\ndivergence = kl\nspmd.step_size = 1\nbias = 0.1", "must start with"),
        ("estimator = tomc\ndivergence = kl\nspmd.step_size = 1\nspmd.momentum = 0.9", "unknown SPMD"),
        (
            "estimator = tomc\ndivergence = kl\nspmd.step_size = 1, 2\nspmd.tau = 1, 2\n"
            "estimator.bias = 1, 2\n"
            "estimator.pi_min = 0.1, 0.2",
            "at most 3 tuned",
        ),
    ],
)
def test_invalid_method_sections(tmp_path: Path, body: str, error: str) -> None:
    path = write_config(tmp_path, f"[problem P]\nenvironment = blackjack\nbudget = 10\n[method m]\n{body}\n")
    with pytest.raises(ValueError, match=error):
        method_spec(load_config(path), "m", "P")


def test_missing_sections(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.cfg")
    with pytest.raises(KeyError, match="No \\[method nope\\]"):
        method_spec(CONFIG, "nope", "GridWorld")


# ----------------------------------------------------------------------------- statistics


def test_mean_and_se_edge_cases() -> None:
    mean, se = mean_and_se([])
    assert math.isnan(mean)
    assert math.isnan(se)
    mean, se = mean_and_se([2.0])  # one value: no ddof=1 warning (warnings are errors in this suite)
    assert mean == 2.0
    assert math.isnan(se)
    mean, se = mean_and_se([1.0, 3.0])
    assert mean == 2.0
    assert se == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("diff", "se", "expected"),
    [
        (0.0, 0.0, "tie"),
        (0.1, 0.0, "worse"),
        (-0.1, 0.0, "better"),
        (0.0005, 0.0, "tie"),
        (-0.3, 0.1, "better"),
        (0.3, 0.1, "worse"),
        (0.1, 0.1, "tie"),
        (0.1, math.nan, "n/a"),
    ],
)
def test_z_verdict(diff: float, se: float, expected: str) -> None:
    assert z_verdict(diff, se, 2.0, 0.001)[1] == expected


def history(problem: str, method: str, seed: int, gaps: list[float]) -> list[dict[str, object]]:
    return [
        {
            "problem": problem,
            "method": method,
            "seed": seed,
            "gap": g,
            "cumulative_observations": 10 * i,
            "status": "updated",
        }
        for i, g in enumerate(gaps)
    ]


def test_summarize_compares_with_baseline() -> None:
    rows = [r for seed in range(5) for r in history("P", "baseline", seed, [1.0, 0.5 + 0.01 * seed])]
    rows += [r for seed in range(5) for r in history("P", "better", seed, [1.0, 0.1 + 0.01 * seed])]
    summary = {r["method"]: r for r in summarize(rows, "baseline")}
    assert summary["baseline"]["verdict"] == ""
    assert summary["better"]["verdict"] == "better"
    assert summary["better"]["diff_vs_baseline"] == pytest.approx(-0.4)
    assert summary["better"]["seeds"] == 5


def test_summarize_single_seed_is_not_a_crash() -> None:
    rows = history("P", "baseline", 0, [1.0, 0.5]) + history("P", "other", 0, [1.0, 0.4])
    summary = {r["method"]: r for r in summarize(rows, "baseline")}
    assert summary["other"]["verdict"] == "n/a"


def test_best_per_method_skips_ineligible_rows() -> None:
    rows = [
        {"problem": "P", "method": "m", "min_updates": 0, "mean_final_gap": 0.0},
        {"problem": "P", "method": "m", "min_updates": 2, "mean_final_gap": math.nan},
        {"problem": "P", "method": "m", "min_updates": 1, "mean_final_gap": 0.3},
        {"problem": "P", "method": "m", "min_updates": 3, "mean_final_gap": 0.2},
        {"problem": "P", "method": "z", "min_updates": 0, "mean_final_gap": 0.1},
    ]
    best = best_per_method(rows, "exact_gap")
    assert best["P", "m"]["mean_final_gap"] == 0.2
    assert ("P", "z") not in best


def test_gap_lower_selection_is_model_free_with_upper_bound_tie_break() -> None:
    rows = [
        {
            "problem": "P",
            "method": "m",
            "min_updates": 2,
            "mean_final_gap": 0.0,
            "mean_gap_lower": 0.3,
            "mean_gap_upper": 5.0,
        },
        {
            "problem": "P",
            "method": "m",
            "min_updates": 2,
            "mean_final_gap": 0.9,
            "mean_gap_lower": 0.1,
            "mean_gap_upper": 9.0,
        },
        {
            "problem": "P",
            "method": "m",
            "min_updates": 2,
            "mean_final_gap": 0.5,
            "mean_gap_lower": 0.1,
            "mean_gap_upper": 4.0,
        },
        {
            "problem": "P",
            "method": "m",
            "min_updates": 0,
            "mean_final_gap": 0.0,
            "mean_gap_lower": 0.0,
            "mean_gap_upper": 0.0,
        },
        {
            "problem": "P",
            "method": "m",
            "min_updates": 2,
            "mean_final_gap": 0.0,
            "mean_gap_lower": math.nan,
            "mean_gap_upper": 0.0,
        },
    ]
    chosen = best_per_method(rows, "gap_lower")["P", "m"]
    assert (chosen["mean_gap_lower"], chosen["mean_gap_upper"]) == (0.1, 4.0)  # exact gap is ignored
    assert best_per_method(rows, "exact_gap")["P", "m"]["mean_final_gap"] == 0.0
    with pytest.raises(ValueError, match="selection"):
        best_per_method(rows, "fastest")


def test_selection_and_certifier_settings(tmp_path: Path) -> None:
    assert selection_of(CONFIG) == "gap_lower"
    certifier = make_certifier(CONFIG)
    assert (certifier.n_rollouts, certifier.n_start_rollouts, certifier.z) == (20, 2000, 3.0)
    bad = load_config(write_config(tmp_path, TINY_CONFIG.replace("[tuning]", "[tuning]\nselection = luck")))
    with pytest.raises(ValueError, match="selection"):
        selection_of(bad)
    defaults = load_config(write_config(tmp_path, TINY_CONFIG))
    assert selection_of(defaults) == "gap_lower"
    assert make_certifier(defaults) == certifier


def test_gap_curve_is_a_step_function() -> None:
    rows = history("P", "m", 0, [1.0, 0.6, 0.2])  # updates at 10 and 20 observations
    curve = gap_curve(rows, 40, np.array([0.0, 0.2, 0.25, 0.5, 1.0]))
    np.testing.assert_allclose(curve, [1.0, 1.0, 0.6, 0.2, 0.2])
    with pytest.raises(ValueError, match="budget"):
        gap_curve(rows, 0, np.array([0.0]))


def test_csv_round_trip(tmp_path: Path) -> None:
    rows = [{"problem": "P", "method": "m", "a": 1, "b": 2.5, "c": math.nan, "d": "updated", "e": True}]
    write_csv(tmp_path / "x.csv", rows)
    back = list(read_csv(tmp_path / "x.csv"))
    assert back[0]["a"] == 1
    assert back[0]["b"] == 2.5
    assert math.isnan(back[0]["c"])
    assert back[0]["d"] == "updated"
    assert back[0]["e"] is True
    write_csv(tmp_path / "empty.csv", [])
    assert not (tmp_path / "empty.csv").exists()


# ----------------------------------------------------------------------------- end to end


def test_tune_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config, out = write_config(tmp_path), tmp_path / "out"
    summary = tune.main(["--config", str(config), "--output", str(out), "--workers", "1"])
    for name in (
        "tuning.csv",
        "history.csv",
        "summary.csv",
        "tuning_blackjack.png",
        "tuning_grid_blackjack.png",
        "learning_curves_blackjack.png",
    ):
        assert (out / name).exists(), name
    assert not (out / "tuning.png").exists()  # one figure per problem, no combined file
    tuning_rows = list(read_csv(out / "tuning.csv"))
    assert len(tuning_rows) == 5  # baseline: 2 step sizes; td: 2 lambdas; euclid: 1
    assert all(r["tuning_budget"] == 3000 for r in tuning_rows)
    assert {r["method"] for r in summary} == {"baseline", "td", "euclid"}
    assert all(r["seeds"] == 3 for r in summary)
    assert all(0 <= r["mean_gap_lower"] <= r["mean_gap_upper"] for r in tuning_rows)
    assert all(r["mean_final_gap_lower"] <= r["mean_final_gap"] + 1e-9 for r in summary)
    history_rows = list(read_csv(out / "history.csv"))
    finals = {(r["method"], r["seed"]): r for r in history_rows}  # last row per run
    assert all(
        r["gap_lower"] <= r["gap"] + 1e-9 and r["validation_observations"] > 0 for r in finals.values()
    )
    assert max(r["cumulative_observations"] for r in history_rows) <= problem_budget(
        load_config(config), "Blackjack"
    )

    # Reusing the saved tuning and redrawing figures must not rerun the search.
    (out / "tuning_blackjack.png").unlink()
    tune.main(
        ["--config", str(config), "--output", str(out), "--workers", "1", "--reuse-tuning", "--tune-only"]
    )
    tune.main(["--config", str(config), "--output", str(out), "--plot-only"])
    assert (out / "tuning_blackjack.png").exists()


def test_labels_default_to_names_and_stay_out_of_parameters(tmp_path: Path) -> None:
    assert method_label(CONFIG, "true_online_td") == r"True Online TD($\lambda$)"
    assert problem_label(CONFIG, "GridWorld") == r"GridWorld ($10 \times 10$)"
    assert "label" not in method_spec(CONFIG, "baseline", "GridWorld").grid  # not a tuned parameter
    assert isinstance(make_env(CONFIG, "GridWorld"), GridWorld)  # not passed to the constructor
    tiny = load_config(write_config(tmp_path))
    assert method_label(tiny, "td") == "td"
    assert problem_label(tiny, "Blackjack") == "Blackjack"


def test_figures_render_mathtext_labels_per_problem(tmp_path: Path) -> None:
    """Every configured label (with LaTeX math) renders, and each problem gets its own file."""
    from ie692_rl.plotting import plot_learning_curves, plot_tuning

    methods = names(CONFIG["tuning"]["methods"])
    labels = {m: method_label(CONFIG, m) for m in methods}
    problems = {p: problem_label(CONFIG, p) for p in ("GridWorld", "Blackjack")}
    history = [
        r | {"problem": p, "method": m}
        for p in problems
        for m in methods
        for seed in range(2)
        for r in history_rows_for(seed)
    ]
    tuning = [
        {
            "problem": p,
            "method": m,
            "min_updates": 1,
            "mean_gap_lower": 0.1 * i,
            "mean_gap_upper": 1.0,
            "gap_lower_standard_error": 0.01,
        }
        for p in problems
        for i, m in enumerate(methods)
    ]
    curves, grids = [], []
    for group, members in figure_groups(CONFIG, methods).items():
        curves += plot_learning_curves(
            history, {"GridWorld": 100, "Blackjack": 100}, members, labels, problems, tmp_path, group
        )
        grids += plot_tuning(tuning, members, labels, problems, tmp_path, group=group)
        assert [p.name for p in curves[-2:]] == [
            f"learning_curves_gridworld_{group}.png",
            f"learning_curves_blackjack_{group}.png",
        ]
        assert [p.name for p in grids[-2:]] == [
            f"tuning_gridworld_{group}.png",
            f"tuning_blackjack_{group}.png",
        ]
    assert all(p.stat().st_size > 10_000 for p in curves + grids)


def test_figure_groups_cover_every_method_and_reject_unknown_ones(tmp_path: Path) -> None:
    methods = names(CONFIG["tuning"]["methods"])
    groups = figure_groups(CONFIG, methods)
    assert {m for members in groups.values() for m in members} == set(methods)
    baseline = CONFIG["tuning"]["baseline"]
    assert all(members[0] == baseline and len(members) <= 8 for members in groups.values())
    tiny = load_config(write_config(tmp_path))
    assert figure_groups(tiny, ["td"]) == {"": ["td"]}  # no [figures]: one figure, no file suffix
    bad = load_config(write_config(tmp_path, TINY_CONFIG + "\n[figures]\nall = td, nope\n"))
    with pytest.raises(ValueError, match="nope"):
        figure_groups(bad, ["td"])


def test_tuning_grid_lays_out_every_combination(tmp_path: Path) -> None:
    """Rows = step size, columns = outer x inner of the other tuned parameters; fixed parameters are
    not axes; a combination without an update is blank; the chosen combination is found."""
    from ie692_rl.plotting import _Grid, plot_tuning_grid

    tuning = [
        {
            "problem": "GridWorld",
            "method": "true_online_td",
            "spmd.step_size": eta,
            "estimator.eta": alpha,
            "estimator.lam": lam,
            "estimator.n_steps": 50_000,
            "min_updates": 0 if (eta, alpha, lam) == (100.0, 0.5, 0.95) else 2,
            "mean_gap_lower": eta / 100 + alpha + lam / 10,
            "mean_gap_upper": 1.0,
        }
        for eta in (1.0, 10.0, 100.0)
        for alpha in (0.02, 0.1, 0.5)
        for lam in (0.5, 0.8, 0.95)
    ] + [
        {
            "problem": "GridWorld",
            "method": "baseline",
            "spmd.step_size": eta,
            "estimator.bias": 0.01,
            "min_updates": 1,
            "mean_gap_lower": 1 / eta,
            "mean_gap_upper": 1.0,
        }
        for eta in (1.0, 10.0, 100.0)
    ]
    grid = _Grid(tuning[:27], "mean_gap_lower")
    assert grid.row_key == "spmd.step_size"
    assert grid.col_keys == ["estimator.eta", "estimator.lam"]
    assert grid.values.shape == (3, 9)
    assert np.isnan(grid.values[2, 8])
    assert np.isfinite(grid.values).sum() == 26
    assert grid.values[0, 0] == pytest.approx(0.01 + 0.02 + 0.05)
    baseline = _Grid(tuning[27:], "mean_gap_lower")
    assert baseline.col_keys == []
    assert baseline.values.shape == (3, 1)
    labels = {"true_online_td": r"True Online TD($\lambda$)", "baseline": "KL (Baseline)"}
    [path] = plot_tuning_grid(
        tuning, ["baseline", "true_online_td"], labels, {"GridWorld": "GridWorld"}, tmp_path
    )
    assert path.name == "tuning_grid_gridworld.png"
    assert path.stat().st_size > 10_000


def history_rows_for(seed: int) -> list[dict[str, object]]:
    return [
        {"seed": seed, "gap": g, "cumulative_observations": 10 * i, "status": "updated"}
        for i, g in enumerate([0.3, 0.2 - 0.01 * seed, 0.1])
    ]


def test_selection_can_be_overridden(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(tmp_path)
    tune.main(
        [
            "--config",
            str(config),
            "--output",
            str(tmp_path / "o"),
            "--workers",
            "1",
            "--methods",
            "baseline",
            "--tune-only",
            "--selection",
            "exact_gap",
        ]
    )
    assert "selection: exact_gap" in capsys.readouterr().out


def test_tune_in_parallel_matches_serial(tmp_path: Path) -> None:
    config = write_config(tmp_path)
    serial = tune.main(
        ["--config", str(config), "--output", str(tmp_path / "a"), "--workers", "1", "--methods", "baseline"]
    )
    parallel = tune.main(
        ["--config", str(config), "--output", str(tmp_path / "b"), "--workers", "2", "--methods", "baseline"]
    )
    assert [r["mean_final_gap"] for r in serial] == [r["mean_final_gap"] for r in parallel]


def test_method_spec_dataclass() -> None:
    spec = MethodSpec("m", "tomc", "kl", {"spmd.step_size": [1.0, 2.0], "estimator.bias": [0.1]})
    assert spec.tuned == ["spmd.step_size"]
    assert spec.combinations() == [
        (("estimator.bias", 0.1), ("spmd.step_size", 1.0)),
        (("estimator.bias", 0.1), ("spmd.step_size", 2.0)),
    ]
