"""Blackjack SPMD with the KL update from pmd.py, optionally entropy-regularized.

Uses the team's blackjack, Monte Carlo estimates, budget, grid search and scoring;
see kl_spmd.py for what changes. Results compare directly with blackjack_baseline_results/.

Run from anywhere:
    python "Entropy-Regularized SPMD/blackjack_kl_spmd.py" --tau 0.01
    python "Entropy-Regularized SPMD/blackjack_kl_spmd.py" --tau 0 --output plain_kl_run
Add --help for the shared settings, which are the same as baseline_blackjack.py.
"""
import sys

import kl_spmd  # first: puts the team's code on the import path
import utils
from baseline_blackjack import Blackjack

# Over 10 seeds (exact scoring), step sizes 1 and 5 reached the optimal policy every time
# while 20 often settled on a worse one, so 20 is left out of the grid.
KL_STEP_SIZES = [0.5, 1, 5]
# tau = 0.01 matched plain KL at step size 1; tau >= 0.03 raised the true cost.
DEFAULT_TAU = 0.01

if __name__ == "__main__":
    tau = kl_spmd.prepare_arguments(sys.argv, DEFAULT_TAU, KL_STEP_SIZES, "blackjack")
    kl_spmd.use_kl_update(tau)
    print(f"KL SPMD with entropy weight tau = {tau:g}")
    utils.run_experiment(Blackjack(), "blackjack", [5, 10, 20], [2, 3, 4], 120_000, 8)
