"""GridWorld SPMD with the KL update from pmd.py, optionally entropy-regularized.

Uses the team's GridWorld, Monte Carlo estimates, budget, grid search and scoring;
see kl_spmd.py for what changes. Results compare directly with gridworld_baseline_results/.

Run from anywhere:
    python "Entropy-Regularized SPMD/gridworld_kl_spmd.py" --tau 0.001
    python "Entropy-Regularized SPMD/gridworld_kl_spmd.py" --tau 0 --step-sizes 5 --output plain_kl_run
Add --help for the shared settings, which are the same as baseline.py.
"""
import sys

import kl_spmd  # first: puts the team's code on the import path
import utils
from baseline import GridWorld, plot_board

# Cost gaps between actions are small (median about 0.27), so the KL update needs
# larger step sizes than the Euclidean update's 0.1, 0.5 and 1.0 to move the policy.
KL_STEP_SIZES = [1, 5, 20]
# Over 5 seeds at full budget, tau = 0.001 roughly doubles policy entropy for almost no
# extra cost; tau >= 0.003 noticeably raises the true GridWorld cost.
DEFAULT_TAU = 0.001

if __name__ == "__main__":
    tau = kl_spmd.prepare_arguments(sys.argv, DEFAULT_TAU, KL_STEP_SIZES, "gridworld")
    kl_spmd.use_kl_update(tau)
    print(f"KL SPMD with entropy weight tau = {tau:g}")
    utils.run_experiment(GridWorld(), "gridworld", [5, 10, 20], [40, 70, 100],
                         24_000_000, 100, plot_board)
